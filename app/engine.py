"""Thread-safe wrapper around ``quan_transcribe``.

Two properties drive the design:

* **Loading is slow and expensive.** The weights plus Google's
  ``google/embeddinggemma-2`` audio tower must be downloaded and read once, then
  kept resident. Loading happens behind a lock, so a burst of concurrent
  requests triggers exactly one load.
* **The model is not re-entrant here.** Decoding is a plain forward pass, but we
  serialise calls anyway: it keeps VRAM use predictable and makes the numbers
  reported by ``GET /metrics`` meaningful.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .audio import SAMPLE_RATE, Segment
from .batch import PieceBatcher
from .config import Settings

logger = logging.getLogger(__name__)


class EngineNotReady(RuntimeError):
    """Raised when the model cannot be loaded."""


@dataclass
class StreamEvent:
    """One incremental update from a streaming decode."""

    kind: str  # "partial" | "final" | "done"
    text: str
    start: float
    end: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
        }


class TranscriptionEngine:
    """Loads QUAN-Transcribe once and serves transcription calls."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._asr: Any | None = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.RLock()
        self._load_error: str | None = None
        self._load_seconds: float | None = None
        self._loaded_at: float | None = None
        self.requests = 0
        self.audio_seconds = 0.0
        self.decode_seconds = 0.0
        # Cross-request batching. Created on load, once the sample-rate-derived
        # budget can be trusted, and None when the feature is off.
        self._batcher: PieceBatcher | None = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._asr is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def loaded_at(self) -> float | None:
        """Unix timestamp of the successful load, or ``None``."""
        return self._loaded_at

    def load(self) -> Any:
        """Load the model if needed and return the ``QuanTranscribe`` instance."""
        if self._asr is not None:
            return self._asr
        with self._load_lock:
            if self._asr is not None:
                return self._asr
            if self._load_error is not None:
                raise EngineNotReady(self._load_error)
            self._asr = self._load()
            return self._asr

    def _load(self) -> Any:
        settings = self.settings
        import os

        if settings.hf_token:
            os.environ.setdefault("HF_TOKEN", settings.hf_token)

        try:
            from quan_transcribe import QuanTranscribe
        except ImportError as exc:  # pragma: no cover - image always has it
            raise EngineNotReady(
                "the 'quan-transcribe' package is not installed"
            ) from exc

        lm: bool | str = False
        if settings.lm_path:
            lm = settings.lm_path
        elif settings.use_lm:
            lm = True

        started = time.time()
        logger.info(
            "loading model %s (device=%s, lm=%s)",
            settings.model_path,
            settings.device or "auto",
            lm,
        )
        try:
            asr = QuanTranscribe.from_pretrained(
                settings.model_path, device=settings.device, lm=lm
            )
        except Exception as exc:
            self._load_error = f"failed to load {settings.model_path}: {exc}"
            logger.exception("model load failed")
            raise EngineNotReady(self._load_error) from exc

        # ``lm_config.json`` ships tuned values; an explicit setting wins.
        decoder = getattr(asr, "lm_decoder", None)
        if decoder is not None:
            if settings.lm_alpha is not None:
                decoder.alpha = settings.lm_alpha
            if settings.lm_beta is not None:
                decoder.beta = settings.lm_beta
            logger.info(
                "kenlm beam search ready (beam_width=%d, alpha=%s, beta=%s)",
                settings.beam_width,
                decoder.alpha,
                decoder.beta,
            )

        self._load_seconds = time.time() - started
        self._loaded_at = time.time()
        self._configure_torch(asr)
        self._start_batcher()
        logger.info(
            "model ready in %.1fs on %s (profile=%s)",
            self._load_seconds,
            asr.device,
            self.settings.profile,
        )
        return asr

    def _configure_torch(self, asr: Any) -> None:
        """Apply the throughput knobs that only take effect before the first pass.

        TF32 lets Ampere and later run FP32 matmuls on the tensor cores. For a
        CTC model the precision loss is irrelevant next to the speedup, so it is
        on by default; ``TF32=false`` restores exact FP32.
        """
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is in the image
            return

        if self.settings.tf32 and asr.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 matmuls enabled")

        # On CPU the model is a plain FP32 GEMM pipeline and scales with cores,
        # so an explicit thread count is the main lever there. Leaving it unset
        # keeps torch's own heuristic.
        threads = self.settings.cpu_threads
        if threads is not None and threads > 0:
            torch.set_num_threads(threads)
            logger.info("torch limited to %d CPU thread(s)", threads)

    def _start_batcher(self) -> None:
        """Bring up cross-request batching unless it is disabled."""
        if self.settings.microbatch_seconds <= 0:
            logger.info("micro-batching disabled (MICROBATCH_SECONDS=0)")
            return
        self._batcher = PieceBatcher(
            self._run_batch,
            sample_rate=SAMPLE_RATE,
            window_ms=self.settings.microbatch_max_wait_ms,
            max_requests=self.settings.microbatch_max_requests,
            batch_seconds=self.settings.batch_seconds,
        )
        self._batcher.start()
        logger.info(
            "micro-batching on (window=%.0fms, max_requests=%d)",
            self.settings.microbatch_max_wait_ms,
            self.settings.microbatch_max_requests,
        )

    def warmup(self) -> None:
        """Run one silent forward pass so the first real request is not slow."""
        try:
            asr = self.load()
        except EngineNotReady:
            return
        silent = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
        try:
            with self._run_lock:
                asr._forward([silent], False, None)
        except Exception:
            logger.debug("warmup failed", exc_info=True)

    # -- introspection -------------------------------------------------------

    def info(self) -> dict[str, Any]:
        asr = self._asr
        device = str(getattr(asr, "device", self.settings.device or "unknown"))
        chunk_frames = None
        if asr is not None:
            chunk_frames = int(getattr(asr.model.cfg, "chunk_frames", 12))
        decoder = getattr(asr, "lm_decoder", None)
        return {
            "model": self.settings.served_model_name,
            "model_path": self.settings.model_path,
            "loaded": self.is_loaded,
            "device": device,
            "profile": self.settings.profile,
            "language": "vi",
            "lm": decoder is not None,
            "lm_path": self.settings.lm_path,
            "lm_alpha": getattr(decoder, "alpha", self.settings.lm_alpha),
            "lm_beta": getattr(decoder, "beta", self.settings.lm_beta),
            "beam_width": self.settings.beam_width,
            "chunk_seconds": (chunk_frames or 12) * 0.04,
            "stream_step_seconds": self.settings.stream_step_seconds,
            "left_chunks": self.settings.left_chunks,
            "microbatch_seconds": self.settings.microbatch_seconds,
            "microbatch_max_requests": self.settings.microbatch_max_requests,
            "tf32": self.settings.tf32,
            "load_error": self._load_error,
            "load_seconds": self._load_seconds,
            "loaded_at": self._loaded_at,
        }

    def metrics(self) -> dict[str, Any]:
        rtf = (self.decode_seconds / self.audio_seconds) if self.audio_seconds else None
        batcher = self._batcher.stats() if self._batcher is not None else None
        return {
            "requests_total": self.requests,
            "audio_seconds_total": round(self.audio_seconds, 3),
            "decode_seconds_total": round(self.decode_seconds, 3),
            "realtime_factor": round(rtf, 4) if rtf else None,
            "loaded": self.is_loaded,
            "batcher": batcher,
        }

    def describe_device(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return f"cuda ({torch.cuda.get_device_name(0)})"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            # torch is absent only in a stripped-down test environment.
            pass
        return self.settings.device or "cpu"

    # -- decoding ------------------------------------------------------------

    def _decoder(self, use_lm: bool | None = None):
        """The KenLM decoder, or ``None`` to force greedy decoding.

        ``use_lm=True`` on a deployment without a loaded LM is an error rather
        than a silent fall back to greedy decoding: the caller asked for the
        accuracy the LM provides, and silently returning worse output would hide
        a misconfigured deployment.
        """
        asr = self._asr
        if use_lm is False or not self.settings.supports_lm:
            if use_lm:
                raise EngineNotReady(
                    "no KenLM model is loaded; start the server with USE_LM=true "
                    "(the lm/ folder is fetched from the model repo) or set "
                    "LM_PATH to a local KenLM model"
                )
            return None
        decoder = getattr(asr, "lm_decoder", None) if asr is not None else None
        if decoder is None and use_lm:
            raise EngineNotReady("the model loaded without a KenLM decoder")
        return decoder

    def _run_batch(self, wavs: list[np.ndarray], context: Any) -> list[str]:
        """Run one already-grouped batch. Called on the batcher's thread.

        The batcher is single-threaded, but streaming and live sessions call the
        model from their own threads, so the batch still takes ``_run_lock``: it
        is the one place that guarantees only one forward pass is in flight.
        """
        asr = self._asr
        if asr is None:  # pragma: no cover - guarded by load() before enqueue
            raise EngineNotReady("the model is not loaded")
        streaming, left_chunks, decoder, beam_width = context
        with self._run_lock:
            return asr._run(wavs, streaming, left_chunks, decoder, beam_width)

    def _decode_locked(
        self,
        pieces: list[np.ndarray],
        streaming: bool,
        left_chunks: int | None,
        *,
        use_lm: bool | None = None,
        beam_width: int | None = None,
    ) -> list[str]:
        """Decode pieces, taking the model lock only when there is no batcher.

        When the batcher is active it serialises batches itself, and its runner
        takes ``_run_lock``. Holding the lock here as well would deadlock, since
        this thread has to wait for the collector to run and the collector has
        to wait for the lock this thread is holding.
        """
        if self._batcher is not None:
            return self._decode_pieces(
                pieces,
                streaming,
                left_chunks,
                use_lm=use_lm,
                beam_width=beam_width,
            )
        with self._run_lock:
            return self._decode_pieces(
                pieces,
                streaming,
                left_chunks,
                use_lm=use_lm,
                beam_width=beam_width,
            )

    def _decode_pieces(
        self,
        pieces: list[np.ndarray],
        streaming: bool,
        left_chunks: int | None,
        *,
        use_lm: bool | None = None,
        beam_width: int | None = None,
    ) -> list[str]:
        """Decode pieces, letting concurrent requests share a forward pass.

        With micro-batching on, the pieces go through the collector so that
        requests arriving together are padded and launched once. With it off
        this is the plain length-sorted batching the upstream library uses.
        """
        asr = self._asr
        if asr is None:
            raise EngineNotReady("the model is not loaded")
        decoder = self._decoder(use_lm)
        # Beam search is CPU-bound and dominates on a GPU, so an unsupported
        # request is worth warning about rather than ignoring.
        if decoder is not None and streaming:
            logger.debug(
                "beam search is applied per streaming utterance; on GPU this "
                "is usually the bottleneck"
            )
        beam_width = beam_width or self.settings.beam_width

        batcher = self._batcher
        if batcher is not None:
            # The decoder has to be part of both the grouping key and the
            # context, so that greedy and beam pieces never share a launch.
            signature = (bool(streaming), left_chunks, id(decoder), beam_width)
            context = (bool(streaming), left_chunks, decoder, beam_width)
            return batcher.decode(pieces, signature, context)

        return self._decode_serial(
            pieces, streaming, left_chunks, decoder, beam_width
        )

    def _decode_serial(
        self,
        pieces: list[np.ndarray],
        streaming: bool,
        left_chunks: int | None,
        decoder: Any,
        beam_width: int,
    ) -> list[str]:
        """The upstream batching strategy: one call, length-sorted pieces."""
        asr = self._asr
        batch_seconds = self.settings.batch_seconds
        texts: list[str | None] = [None] * len(pieces)
        order = sorted(range(len(pieces)), key=lambda k: len(pieces[k]))
        batch: list[int] = []

        for k in [*order, None]:
            if k is not None:
                longest = len(pieces[k]) / SAMPLE_RATE
                if not batch or longest * (len(batch) + 1) <= batch_seconds:
                    batch.append(k)
                    continue
            if batch:
                wavs = [pieces[b] for b in batch]
                results = asr._run(wavs, streaming, left_chunks, decoder, beam_width)
                for index, text in zip(batch, results, strict=False):
                    texts[index] = text
            batch = [k] if k is not None else []

        return [t or "" for t in texts]

    def transcribe(
        self,
        wav: np.ndarray,
        *,
        streaming: bool = False,
        left_chunks: int | None = None,
        chunk_seconds: float | None = None,
        use_lm: bool | None = None,
        beam_width: int | None = None,
    ) -> list[Segment]:
        """Transcribe a full waveform and return timestamped segments.

        The waveform is cut at quiet points exactly like the upstream
        ``transcribe()`` does, but we keep the offsets so the response can carry
        real timestamps.
        """
        from .audio import build_segments, split_with_offsets

        self.load()
        pieces_and_offsets = split_with_offsets(
            wav, chunk_seconds or self.settings.chunk_seconds
        )
        if not pieces_and_offsets:
            return []

        pieces = [p[0] for p in pieces_and_offsets]
        offsets = [(p[1], p[2]) for p in pieces_and_offsets]

        started = time.time()
        texts = self._decode_locked(
            pieces,
            streaming,
            left_chunks,
            use_lm=use_lm,
            beam_width=beam_width,
        )
        self._record(len(wav) / SAMPLE_RATE, time.time() - started)

        return build_segments(texts, offsets)

    def iter_stream(
        self,
        wav: np.ndarray,
        *,
        left_chunks: int | None = None,
        step_seconds: float | None = None,
        chunk_seconds: float | None = None,
        use_lm: bool | None = None,
        beam_width: int | None = None,
    ) -> Iterator[StreamEvent]:
        """Replay a whole file through the streaming decoder, step by step.

        This is what a streaming deployment would emit: an utterance in progress
        is re-decoded every step, finished utterances are decoded once more with
        full context, and only the newest words can still change.
        """
        self.load()
        step_seconds = step_seconds or self.settings.stream_step_seconds
        step = max(1, round(step_seconds * SAMPLE_RATE))
        step_seconds = step / SAMPLE_RATE

        session = self._new_live_session(
            left_chunks, chunk_seconds, use_lm, beam_width
        )
        total = len(wav)
        consumed = 0
        last_end = 0.0
        started = time.time()

        # The lock is taken per step rather than around the whole loop. A long
        # SSE stream is a sequence of independent 0.48 s decodes, so releasing
        # between them lets other requests through instead of making a slow
        # stream block the server for its full duration.
        while consumed < total:
            block = wav[consumed : consumed + step]
            consumed += len(block)
            now = consumed / SAMPLE_RATE
            finals, partial = self._session_step(session, block)
            for final in finals:
                yield StreamEvent("final", final, last_end, now)
                last_end = now
            if partial:
                yield StreamEvent("partial", partial, last_end, now)

        with self._run_lock:
            tail = session.flush()
        if tail:
            yield StreamEvent("final", tail, last_end, total / SAMPLE_RATE)

        self._record(total / SAMPLE_RATE, time.time() - started)
        yield StreamEvent("done", "", last_end, total / SAMPLE_RATE)

    def _session_step(self, session: Any, block: np.ndarray) -> tuple[list[str], str]:
        """Advance a live session by one block, under the model lock.

        ``LiveSession.accept`` both decodes the partial hypothesis and may
        finalise a finished utterance, so the whole call is one critical
        section. It deliberately does not go through the batcher: streaming is
        already incremental, and delaying a 0.48 s step to gather companions
        would only add latency to the one path where it is visible.
        """
        with self._run_lock:
            finals = session.accept(block)
            partial = session.partial
        return finals, partial

    def stream_segments(
        self,
        wav: np.ndarray,
        *,
        left_chunks: int | None = None,
        step_seconds: float | None = None,
        use_lm: bool | None = None,
        beam_width: int | None = None,
    ) -> list[Segment]:
        """Convenience wrapper: consume :meth:`iter_stream` into final segments."""
        segments: list[Segment] = []
        for event in self.iter_stream(
            wav,
            left_chunks=left_chunks,
            step_seconds=step_seconds,
            use_lm=use_lm,
            beam_width=beam_width,
        ):
            if event.kind == "final" and event.text:
                segments.append(
                    Segment(text=event.text, start=event.start, end=event.end)
                )
        return segments

    # -- live sessions -------------------------------------------------------

    def _new_live_session(
        self,
        left_chunks: int | None,
        chunk_seconds: float | None = None,
        use_lm: bool | None = None,
        beam_width: int | None = None,
    ):
        from quan_transcribe.live import LiveSession

        asr = self.load()
        decoder = self._decoder(use_lm)
        beam_width = beam_width or self.settings.beam_width
        session = LiveSession(
            asr,
            left_chunks if left_chunks is not None else self.settings.left_chunks,
            self.settings.endpoint_s,
            chunk_seconds or self.settings.max_utterance_s,
            beam_width,
        )
        if decoder is not getattr(asr, "lm_decoder", None):
            # LiveSession takes the decoder off the model rather than as an
            # argument, so a per-request override has to be applied here.
            # Rebinding the instance method keeps the change local to this
            # session instead of mutating state another request could see.
            session._final = lambda wav: asr._run(
                [wav], False, None, decoder, beam_width
            )[0]
        return session

    def new_live_session(
        self,
        left_chunks: int | None = None,
        chunk_seconds: float | None = None,
        use_lm: bool | None = None,
        beam_width: int | None = None,
    ) -> LiveStream:
        return LiveStream(self, left_chunks, chunk_seconds, use_lm, beam_width)

    # -- bookkeeping ---------------------------------------------------------

    def _record(self, audio_seconds: float, decode_seconds: float) -> None:
        self.requests += 1
        self.audio_seconds += audio_seconds
        self.decode_seconds += decode_seconds

    def close(self) -> None:
        if self._batcher is not None:
            self._batcher.stop()
            self._batcher = None
        with self._run_lock:
            self._asr = None
        try:  # pragma: no cover - only meaningful on CUDA
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            # torch is always present in the image; a missing import here just
            # means there is no CUDA cache to release.
            pass


class LiveStream:
    """A push-based live stream, used by the WebSocket endpoint.

    Feed 16 kHz mono float32 audio with :meth:`accept` and read back the events
    the server should forward. All model access goes through the engine's lock,
    which is held only for the duration of a single :meth:`accept` call.
    """

    def __init__(
        self,
        engine: TranscriptionEngine,
        left_chunks: int | None = None,
        chunk_seconds: float | None = None,
        use_lm: bool | None = None,
        beam_width: int | None = None,
    ):
        self.engine = engine
        self.left_chunks = (
            left_chunks if left_chunks is not None else engine.settings.left_chunks
        )
        self.chunk_seconds = chunk_seconds
        self.use_lm = use_lm
        self.beam_width = beam_width
        self._session = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._consumed = 0
        self._last_end = 0.0
        self._closed = False

    @property
    def partial(self) -> str:
        return self._session.partial if self._session is not None else ""

    @property
    def consumed_seconds(self) -> float:
        return self._consumed / SAMPLE_RATE

    def accept(self, samples: np.ndarray) -> list[StreamEvent]:
        """Add audio and return the events produced by this step."""
        if self._closed:
            raise RuntimeError("the stream is closed")

        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return []

        self._pending = np.concatenate([self._pending, block])
        step = max(
            1,
            round(self.engine.settings.stream_step_seconds * SAMPLE_RATE),
        )
        if self._pending.size < step:
            return []

        if self._session is None:
            self._session = self.engine._new_live_session(
                self.left_chunks,
                self.chunk_seconds,
                self.use_lm,
                self.beam_width,
            )

        audio = self._pending
        self._pending = np.zeros(0, dtype=np.float32)
        self._consumed += audio.size
        now = self._consumed / SAMPLE_RATE

        events: list[StreamEvent] = []
        finals, partial = self.engine._session_step(self._session, audio)
        for final in finals:
            events.append(StreamEvent("final", final, self._last_end, now))
            self._last_end = now
        if partial:
            events.append(
                StreamEvent("partial", partial, self._last_end, now)
            )
        return events

    def flush(self) -> str:
        """Finish the utterance in progress and return it (may be empty)."""
        if self._closed:
            return ""
        if self._session is None:
            # Too short to have reached a decoding step; decode it directly.
            if self._pending.size == 0:
                self._closed = True
                return ""
            text = self.engine.transcribe(
                self._pending,
                streaming=False,
                chunk_seconds=self.chunk_seconds,
                use_lm=self.use_lm,
                beam_width=self.beam_width,
            )[0].text
            self._closed = True
            return text

        with self.engine._run_lock:
            text = self._session.flush()
        self._pending = np.zeros(0, dtype=np.float32)
        self._closed = True
        return text
    def close(self) -> None:
        self._closed = True
        self._session = None
        self._pending = np.zeros(0, dtype=np.float32)

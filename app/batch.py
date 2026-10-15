"""Cross-request micro-batching for the decode step.

Every piece of audio decoded by the engine is padded to the longest piece in
its batch, because that is how a single forward pass is assembled. Upstream the
batching is *within* one ``transcribe()`` call: a 10-minute file is cut into
~20 pieces and they go through the model together. Two independent HTTP requests
arriving at the same time, however, would each launch their own forward pass,
and each would leave the GPU idle for the duration of its own padding overhead
plus launch latency.

This module widens the batch window across requests. A single collector thread
owns the model for batching purposes:

* requests hand it individual pieces, each tagged with the decode signature it
  needs and a request id;
* the collector wakes on the first piece, waits a short window so near-
  simultaneous requests can join, then groups everything it has by signature;
* within a signature, pieces are sorted by length and greedily packed into
  sub-batches under the same padded-cost budget the library uses.

The effect is that N concurrent requests share one launch instead of paying for
N. The cost is up to ``wait_ms`` of added latency per piece, which is why the
defaults are small and why ``microbatch_seconds=0`` bypasses all of this and
falls back to the plain per-request path.

Pieces whose decode signature differs (streaming vs not, a different
``left_chunks``, beam search vs greedy) are never mixed: they would need
different model behaviour and batching them would force one to be wrong.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Hashable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: A decode signature: everything that changes how a piece is decoded.
Signature = Hashable

#: ``run(wavs, context) -> list[str]``, called on the collector thread. The
#: context is whatever the caller attached to the pieces, opaque to the batcher.
RunFn = Callable[[list[np.ndarray], Any], list[str]]


class _Piece:
    """One queued piece waiting for its turn through the model."""

    __slots__ = (
        "context",
        "done",
        "error",
        "request_id",
        "signature",
        "text",
        "wav",
    )

    def __init__(
        self,
        request_id: int,
        wav: np.ndarray,
        signature: Signature,
        context: Any,
    ):
        self.request_id = request_id
        self.wav = wav
        self.signature = signature
        self.context = context
        self.text: str | None = None
        self.error: BaseException | None = None
        self.done = threading.Event()


class PieceBatcher:
    """Collects decode pieces from concurrent requests and runs them together."""

    def __init__(
        self,
        run: RunFn,
        *,
        sample_rate: int,
        window_ms: float,
        max_requests: int,
        batch_seconds: float,
    ) -> None:
        self._run = run
        self._sample_rate = sample_rate
        #: How long the collector lingers after the first piece, waiting for
        #: companions. This is the latency the batching can add.
        self._window_s = max(0.0, window_ms) / 1000.0
        self._max_requests = max(1, max_requests)
        self._max_samples = max(1, int(batch_seconds * sample_rate))

        self._cv = threading.Condition()
        self._queue: deque[_Piece] = deque()
        self._next_request_id = 1
        self._stopping = False
        self._thread: threading.Thread | None = None

        self.batches_run = 0
        self.pieces_run = 0

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="quan-asr-batcher", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stopping = True
            self._cv.notify_all()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    # -- public API ----------------------------------------------------------

    def decode(
        self, wavs: list[np.ndarray], signature: Signature, context: Any = None
    ) -> list[str]:
        """Decode ``wavs`` with the given signature, blocking until done."""
        if not wavs:
            return []

        with self._cv:
            request_id = self._next_request_id
            self._next_request_id += 1
            pending = [
                _Piece(request_id, wav, signature, context) for wav in wavs
            ]
            self._queue.extend(pending)
            self._cv.notify_all()

        # Every piece carries its own event, so a caller only unblocks once its
        # own pieces are filled in, never when an unrelated batch finishes.
        for piece in pending:
            piece.done.wait()

        for piece in pending:
            if piece.error is not None:
                raise piece.error
        return [piece.text or "" for piece in pending]

    # -- collector -----------------------------------------------------------

    def _loop(self) -> None:
        while True:
            batch = self._collect()
            if batch is None:  # told to stop
                return
            try:
                self._execute(batch)
            except BaseException as exc:
                logger.exception("micro-batch decode failed")
                for piece in batch:
                    piece.error = exc
            finally:
                for piece in batch:
                    piece.done.set()

    def _collect(self) -> list[_Piece] | None:
        """Wait for work, then give companions a moment to arrive."""
        with self._cv:
            while not self._queue and not self._stopping:
                self._cv.wait()
            if self._stopping and not self._queue:
                return None

            # A window is only worth spending when a companion could still show
            # up before the gather is full.
            self._await_companions()

            batch = list(self._queue)
            self._queue.clear()
            return batch

    def _await_companions(self) -> None:
        """Sleep out the gather window while still saturating ``max_requests``.

        Caller holds ``self._cv``. Returns as soon as the window closes or the
        queue holds pieces from ``max_requests`` distinct requests, since there
        is nothing to gain from waiting once the batch is as wide as allowed.
        """
        deadline = time.monotonic() + self._window_s
        while True:
            requests = {piece.request_id for piece in self._queue}
            if len(requests) >= self._max_requests:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._cv.wait(timeout=remaining)

    def _execute(self, batch: list[_Piece]) -> None:
        """Group by signature and run each group as one or more sub-batches."""
        groups: dict[Signature, list[_Piece]] = {}
        for piece in batch:
            groups.setdefault(piece.signature, []).append(piece)

        for _signature, pieces in groups.items():
            for sub in self._pack(pieces):
                wavs = [piece.wav for piece in sub]
                # The context is identical for every piece in a group by
                # construction, since it is derived from the signature.
                texts = self._run(wavs, sub[0].context)
                if len(texts) != len(sub):  # pragma: no cover - defensive
                    raise RuntimeError(
                        f"expected {len(sub)} transcripts, got {len(texts)}"
                    )
                for piece, text in zip(sub, texts, strict=True):
                    piece.text = text or ""
                self.batches_run += 1
                self.pieces_run += len(sub)

    def _pack(self, pieces: list[_Piece]) -> list[list[_Piece]]:
        """Greedily pack pieces by padded cost, like the library's own batching.

        A forward pass pads every piece to the longest in the batch, so a batch
        of *k* pieces of at most *n* samples costs ``k * n``. Sorting ascending
        and admitting a piece while that product still fits the budget keeps
        short pieces from being dragged up to a long piece's length.
        """
        ordered = sorted(pieces, key=lambda piece: len(piece.wav))
        batches: list[list[_Piece]] = []
        current: list[_Piece] = []

        for piece in ordered:
            n = len(piece.wav)
            padded = n * (len(current) + 1)
            if current and padded > self._max_samples:
                batches.append(current)
                current = [piece]
            else:
                current.append(piece)

        if current:
            batches.append(current)
        return batches

    def stats(self) -> dict[str, Any]:
        return {
            "batches_run": self.batches_run,
            "pieces_run": self.pieces_run,
            "queued": len(self._queue),
        }

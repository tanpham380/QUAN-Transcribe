"""Tests for the pure-Python pieces, plus the HTTP API with a stubbed engine.

The suite deliberately never imports ``torch`` or ``quan_transcribe``: the model
is replaced by a fake whose ``transcribe``/``iter_stream`` behaviour is scripted.
That keeps it fast and runnable on a laptop, and it still covers the parts most
likely to break: parameter validation, response rendering, error envelopes, the
SSE framing and the WebSocket protocol.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio import SAMPLE_RATE, Segment, split_with_offsets, to_16k_mono
from app.config import Settings
from app.engine import StreamEvent
from app.formats import merge_segments_by_sentence, render, to_srt, to_vtt
from app.schemas import APIError, parse_transcription_form

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_settings_defaults():
    settings = Settings.from_env()
    assert settings.model_path
    assert settings.served_model_name == "QUAN-Transcribe-1.0"
    assert settings.port == 8000
    assert settings.preload is True
    assert settings.left_chunks is None
    assert settings.chunk_seconds == 30.0
    assert settings.stream_step_seconds == 0.48
    assert settings.cors_origins == ("*",)
    # Greedy by default: the LM adds a CPU cost and a separate download.
    assert settings.use_lm is False
    assert settings.lm_path is None
    assert settings.supports_lm is False
    assert settings.beam_width == 32
    # Unset alpha/beta deliberately defer to the LM's own lm_config.json.
    assert settings.lm_alpha is None
    assert settings.lm_beta is None
    assert settings.lm_source == "none (greedy decoding)"


def test_settings_lm_from_env(monkeypatch):
    monkeypatch.setenv("USE_LM", "true")
    monkeypatch.setenv("BEAM_WIDTH", "8")
    monkeypatch.setenv("LM_ALPHA", "0.5")
    monkeypatch.setenv("LM_BETA", "2.0")
    settings = Settings.from_env()
    assert settings.use_lm is True
    assert settings.supports_lm is True
    assert settings.beam_width == 8
    assert settings.lm_alpha == 0.5
    assert settings.lm_beta == 2.0
    assert settings.lm_source == "bundled lm/ from the model repo"


def test_lm_path_implies_lm(monkeypatch):
    monkeypatch.setenv("LM_PATH", "/models/vi_4gram.bin")
    settings = Settings.from_env()
    # LM_PATH alone is enough: the user asked for a specific model.
    assert settings.supports_lm is True
    assert settings.lm_source == "/models/vi_4gram.bin"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/models/quan")
    monkeypatch.setenv("PORT", "9999")
    monkeypatch.setenv("USE_LM", "true")
    monkeypatch.setenv("LEFT_CHUNKS", "3")
    monkeypatch.setenv("PRELOAD", "false")
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example")
    settings = Settings.from_env()
    assert settings.model_path == "/models/quan"
    assert settings.port == 9999
    assert settings.use_lm is True
    assert settings.left_chunks == 3
    assert settings.preload is False
    assert settings.cors_origins == ("https://a.example", "https://b.example")


def test_settings_merge_rejects_unknown():
    settings = Settings.from_env()
    with pytest.raises(TypeError):
        settings.merge(nonsense=1)
    assert settings.merge(port=1234).port == 1234


def test_blank_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PORT", "   ")
    monkeypatch.setenv("MODEL_PATH", "")
    settings = Settings.from_env()
    assert settings.port == 8000
    assert settings.model_path


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #


def test_to_16k_mono_downmixes_and_resamples():
    stereo = np.stack([np.ones(32_000, dtype=np.float32), np.zeros(32_000, dtype=np.float32)], axis=1)
    mono = to_16k_mono(stereo, 32_000)
    assert mono.ndim == 1
    # 1 s at 32 kHz becomes 1 s at 16 kHz.
    assert abs(len(mono) - SAMPLE_RATE) <= 2
    assert np.allclose(mono, 0.5, atol=1e-4)


def test_to_16k_mono_accepts_channels_first():
    stereo = np.stack([np.ones(100, dtype=np.float32)] * 2, axis=0)
    assert to_16k_mono(stereo, SAMPLE_RATE).shape == (100,)


def test_split_with_offsets_is_contiguous_and_lossless():
    wav = np.random.default_rng(0).normal(0, 0.1, 5 * SAMPLE_RATE).astype(np.float32)
    pieces = split_with_offsets(wav, chunk_s=2.0, search_s=0.5)
    assert len(pieces) >= 3
    # Offsets must be strictly increasing and cover the whole waveform.
    assert pieces[0][1] == 0
    assert pieces[-1][2] == len(wav)
    for (_, _, end), (_, start, _) in itertools.pairwise(pieces):
        assert end == start
    # No sample is invented or dropped.
    total = sum(len(p[0]) for p in pieces)
    assert total == len(wav)


def test_split_with_offsets_respects_the_chunk_length():
    wav = np.zeros(10 * SAMPLE_RATE, dtype=np.float32)
    for piece, start, end in split_with_offsets(wav, chunk_s=3.0):
        assert end - start == len(piece)
        assert len(piece) <= 3.0 * SAMPLE_RATE


def test_split_with_offsets_drops_very_short_tail():
    # 100 ms of audible audio followed by a long silence: the tail piece is
    # under MIN_PIECE_SECONDS and must be filtered out.
    wav = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
    pieces = split_with_offsets(wav, chunk_s=10.0)
    assert pieces == []


# --------------------------------------------------------------------------- #
# formats
# --------------------------------------------------------------------------- #


@pytest.fixture()
def segments():
    return [
        Segment(text="xin chào", start=0.0, end=1.25),
        Segment(text="tôi tên là nam", start=1.25, end=3.5),
    ]


def test_srt(segments):
    out = to_srt(segments)
    assert out.startswith("1\n00:00:00,000 --> 00:00:01,250\nxin chào")
    assert "00:00:01,250 --> 00:00:03,500" in out
    assert out.endswith("\n")


def test_vtt(segments):
    out = to_vtt(segments)
    assert out.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:01.250" in out


def test_json_and_text(segments):
    payload = json.loads(
        render("json", segments, language="vi", duration=3.5, model="m")
    )
    assert payload == {"text": "xin chào tôi tên là nam"}
    assert (
        render("text", segments, language="vi", duration=3.5, model="m")
        == "xin chào tôi tên là nam"
    )


def test_verbose_json_shape(segments):
    payload = json.loads(
        render("verbose_json", segments, language="vi", duration=3.5, model="m")
    )
    assert payload["language"] == "vi"
    assert payload["task"] == "transcribe"
    assert payload["duration"] == 3.5
    assert payload["model"] == "m"
    assert len(payload["segments"]) == 2
    assert payload["segments"][0]["start"] == 0.0
    assert payload["segments"][1]["end"] == 3.5


def test_render_rejects_unknown_format(segments):
    with pytest.raises(ValueError):
        render("yaml", segments, language="vi", duration=1.0, model="m")


def test_timestamp_rollover():
    out = to_srt([Segment(text="x", start=3661.5, end=3662.0)])
    assert "01:01:01,500 --> 01:01:02,000" in out


def test_merge_segments_by_sentence():
    # No punctuation at the boundary and no gap => one merged segment.
    merged = merge_segments_by_sentence(
        [Segment("toi di", 0.0, 1.0), Segment("hoc moi ngay", 1.0, 2.0)]
    )
    assert len(merged) == 1
    assert merged[0].text == "toi di hoc moi ngay"
    assert merged[0].start == 0.0 and merged[0].end == 2.0

    # A sentence ending in a full stop is a real boundary.
    kept = merge_segments_by_sentence(
        [Segment("toi di hoc.", 0.0, 1.0), Segment("moi ngay", 1.0, 2.0)]
    )
    assert [s.text for s in kept] == ["toi di hoc.", "moi ngay"]

    # A long silence is a boundary even without punctuation.
    kept = merge_segments_by_sentence(
        [Segment("toi di", 0.0, 1.0), Segment("hoc", 3.0, 4.0)]
    )
    assert len(kept) == 2


# --------------------------------------------------------------------------- #
# request parsing
# --------------------------------------------------------------------------- #


class _Form(dict):
    """Minimal stand-in for Starlette's FormData."""

    def __init__(self, data, lists=None):
        super().__init__(data)
        self._lists = lists or {}

    def getlist(self, key):
        return self._lists.get(key, [])


def test_parse_minimal_form():
    params = parse_transcription_form(_Form({"file": object()}))
    assert params.response_format == "json"
    assert params.streaming is False
    assert params.language is None


def test_parse_rejects_unsupported_language():
    with pytest.raises(APIError) as excinfo:
        parse_transcription_form(_Form({"file": object(), "language": "en"}))
    assert excinfo.value.param == "language"
    assert excinfo.value.code == "unsupported_language"


@pytest.mark.parametrize("value", ["vi", "VI", "vi-VN", "vietnamese"])
def test_parse_accepts_vietnamese_language(value):
    params = parse_transcription_form(_Form({"file": object(), "language": value}))
    assert params.language == "vi"


def test_parse_auto_language_is_none():
    params = parse_transcription_form(_Form({"file": object(), "language": "auto"}))
    assert params.language is None


@pytest.mark.parametrize(
    "field,value,param",
    [
        ("response_format", "yaml", "response_format"),
        ("temperature", "5", "temperature"),
        ("temperature", "warm", "temperature"),
        ("stream", "maybe", "stream"),
        ("left_chunks", "-1", "left_chunks"),
        ("beam_width", "0", "beam_width"),
        ("beam_width", "9999", "beam_width"),
        ("chunk_seconds", "0.01", "chunk_seconds"),
        ("chunk_seconds", "9999", "chunk_seconds"),
    ],
)
def test_parse_rejects_out_of_range(field, value, param):
    with pytest.raises(APIError) as excinfo:
        parse_transcription_form(_Form({"file": object(), field: value}))
    assert excinfo.value.param == param


def test_parse_rejects_timestamps_without_timed_format():
    form = _Form(
        {"file": object(), "response_format": "json"},
        {"timestamp_granularities[]": ["segment"]},
    )
    with pytest.raises(APIError) as excinfo:
        parse_transcription_form(form)
    assert excinfo.value.param == "timestamp_granularities[]"


def test_parse_accepts_timestamps_with_verbose_json():
    form = _Form(
        {"file": object(), "response_format": "verbose_json"},
        {"timestamp_granularities[]": ["segment", "segment"]},
    )
    params = parse_transcription_form(form)
    assert params.timestamp_granularities == ["segment"]


def test_parse_reads_the_extensions():
    params = parse_transcription_form(
        _Form(
            {
                "file": object(),
                "stream": "true",
                "stream_partials": "false",
                "left_chunks": "0",
                "use_lm": "true",
                "beam_width": "8",
                "chunk_seconds": "12.5",
            }
        )
    )
    assert params.streaming is True
    assert params.streaming_partials is False
    assert params.left_chunks == 0
    assert params.use_lm is True
    assert params.beam_width == 8
    assert params.chunk_seconds == 12.5


def test_api_error_payload_shape():
    payload = APIError("boom", param="p", code="c", status_code=418).to_payload()
    assert payload["error"]["message"] == "boom"
    assert payload["error"]["param"] == "p"
    assert payload["error"]["code"] == "c"
    assert payload["error"]["type"] == "invalid_request_error"


# --------------------------------------------------------------------------- #
# LM selection
# --------------------------------------------------------------------------- #


class _StubDecoder:
    def __init__(self):
        self.alpha = 0.3
        self.beta = 1.0


class _StubAsr:
    """The subset of ``QuanTranscribe`` the engine touches."""

    def __init__(self, with_lm=True):
        self.device = "cpu"
        self.lm_decoder = _StubDecoder() if with_lm else None
        self.model = type("M", (), {"cfg": type("C", (), {"chunk_frames": 12})})()
        self.run_calls = []

    def _run(self, wavs, streaming, left_chunks, decoder, beam_width):
        self.run_calls.append(
            {
                "decoder_is_none": decoder is None,
                "beam_width": beam_width,
                "streaming": streaming,
                "left_chunks": left_chunks,
            }
        )
        return ["" for _ in wavs]

    def _forward(self, wavs, streaming, left_chunks):
        return None, None


def _engine(**overrides):
    from app.engine import TranscriptionEngine

    settings = replace(Settings.from_env(), **overrides)
    engine = TranscriptionEngine(settings)
    return engine


def test_greedy_when_no_lm_configured():
    from app.audio import Segment

    engine = _engine()
    stub = _StubAsr()
    engine._asr = stub
    engine.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert stub.run_calls[-1]["decoder_is_none"] is True
    assert isinstance(Segment("", 0.0, 0.0), Segment)


def test_lm_used_when_configured():
    engine = _engine(use_lm=True)
    stub = _StubAsr(with_lm=True)
    engine._asr = stub
    engine.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert stub.run_calls[-1]["decoder_is_none"] is False


def test_use_lm_false_forces_greedy_even_with_lm_loaded():
    engine = _engine(use_lm=True)
    stub = _StubAsr(with_lm=True)
    engine._asr = stub
    engine.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), use_lm=False)
    assert stub.run_calls[-1]["decoder_is_none"] is True


def test_requesting_lm_without_one_is_an_error():
    from app.engine import EngineNotReady

    engine = _engine()
    engine._asr = _StubAsr(with_lm=False)
    with pytest.raises(EngineNotReady, match="no KenLM model is loaded"):
        engine.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), use_lm=True)


def test_per_request_beam_width_overrides_the_default():
    engine = _engine(use_lm=True, beam_width=32)
    stub = _StubAsr()
    engine._asr = stub
    engine.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), beam_width=8)
    assert stub.run_calls[-1]["beam_width"] == 8


def test_engine_reports_lm_details():
    engine = _engine(use_lm=True, beam_width=16)
    engine._asr = _StubAsr()
    info = engine.info()
    assert info["lm"] is True
    assert info["beam_width"] == 16
    assert info["lm_alpha"] == 0.3
    assert info["lm_beta"] == 1.0

# --------------------------------------------------------------------------- #
# hardware profile and micro-batching
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("device", "profile"),
    [
        (None, "auto"),
        ("auto", "auto"),
        ("cpu", "cpu"),
        ("mps", "mps"),
        ("cuda", "cuda"),
        ("cuda:1", "cuda"),
        ("rocm", "cuda"),
    ],
)
def test_profile_derives_from_device(device, profile):
    assert _engine(device=device).settings.profile == profile

def test_microbatch_defaults_are_on():
    settings = Settings.from_env()
    assert settings.microbatch_seconds == 0.15
    assert settings.microbatch_max_wait_ms == 20.0
    assert settings.microbatch_max_requests == 8
    assert settings.tf32 is True
    assert settings.cpu_threads is None

def test_microbatch_can_be_disabled_from_env(monkeypatch):
    monkeypatch.setenv("MICROBATCH_SECONDS", "0")
    assert Settings.from_env().microbatch_seconds == 0.0

def test_zero_wait_is_preserved(monkeypatch):
    """0 is a meaningful wait: batch only what is already queued."""
    monkeypatch.setenv("MICROBATCH_MAX_WAIT_MS", "0")
    assert Settings.from_env().microbatch_max_wait_ms == 0.0

def test_batcher_groups_concurrent_requests(tmp_path):
    """Two overlapping requests should share one forward pass, not two."""
    import threading

    engine = _engine(microbatch_seconds=0.15, microbatch_max_wait_ms=50.0)
    stub = _StubAsr()
    stub.run_calls = []
    engine._asr = stub
    engine._start_batcher()
    try:
        barrier = threading.Barrier(2)

        def call():
            barrier.wait()
            engine.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))

        threads = [threading.Thread(target=call) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        stats = engine._batcher.stats()
        assert stats["pieces_run"] == 2
        # The whole point: two requests, one launch.
        assert engine._batcher.batches_run == 1
    finally:
        engine.close()

def test_batcher_packs_by_padded_cost():
    """A long piece must not drag an unrelated short one into its batch."""
    from app.batch import PieceBatcher

    seen: list[list[int]] = []

    def run(wavs, _context):
        seen.append([len(w) for w in wavs])
        return ["" for _ in wavs]

    batcher = PieceBatcher(
        run,
        sample_rate=100,
        window_ms=0.0,
        max_requests=8,
        batch_seconds=1.0,  # 100 samples of padded cost
    )
    batcher.start()
    try:
        # 100 + 100 does not fit a budget of 100, so each gets its own batch;
        # ascending order means the short one is decoded first.
        batcher.decode(
            [np.zeros(10, dtype=np.float32), np.zeros(100, dtype=np.float32)],
            ("sig",),
        )
    finally:
        batcher.stop()
    assert [len(group) for group in seen] == [1, 1]

def test_batcher_keeps_different_signatures_apart():
    from app.batch import PieceBatcher

    calls: list[tuple] = []

    def run(wavs, context):
        calls.append((context, len(wavs)))
        return ["" for _ in wavs]

    batcher = PieceBatcher(
        run, sample_rate=100, window_ms=0.0, max_requests=8, batch_seconds=10.0
    )
    batcher.start()
    try:
        results = batcher.decode(
            [np.zeros(5, dtype=np.float32), np.zeros(5, dtype=np.float32)],
            ("greedy",),
            context="greedy",
        )
        assert len(results) == 2
    finally:
        batcher.stop()
    assert calls == [("greedy", 2)]

def test_batcher_propagates_errors():
    from app.batch import PieceBatcher

    def run(wavs, _context):
        raise RuntimeError("cuda out of memory")

    batcher = PieceBatcher(
        run, sample_rate=100, window_ms=0.0, max_requests=8, batch_seconds=10.0
    )
    batcher.start()
    try:
        with pytest.raises(RuntimeError, match="cuda out of memory"):
            batcher.decode([np.zeros(5, dtype=np.float32)], ("sig",))
    finally:
        batcher.stop()

def test_decode_serial_is_used_without_a_batcher():
    """With batching off the engine must not even construct a collector."""
    engine = _engine(microbatch_seconds=0)
    stub = _StubAsr()
    engine._asr = stub
    engine._start_batcher()
    assert engine._batcher is None
    engine.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert len(stub.run_calls) == 1


# --------------------------------------------------------------------------- #
# HTTP API, with the model replaced by a scripted fake
# --------------------------------------------------------------------------- #


class FakeEngine:
    """Stands in for ``TranscriptionEngine``; never touches torch."""

    def __init__(self, segments=None, events=None):
        self.settings = replace(Settings.from_env(), preload=False)
        self._segments = segments or [Segment("xin chào", 0.0, 1.0)]
        self._events = events
        self.requests = 0
        self.loaded_at = 1_700_000_000.0
        self.load_error = None
        self.closed = False
        self.calls: list[dict] = []

    @property
    def is_loaded(self):
        return True

    def load(self):
        return self

    def warmup(self):
        return None

    def close(self):
        self.closed = True

    def describe_device(self):
        return "cpu (fake)"

    def info(self):
        return {
            "model": self.settings.served_model_name,
            "model_path": self.settings.model_path,
            "loaded": True,
            "device": "cpu",
            "language": "vi",
            "lm": False,
            "beam_width": self.settings.beam_width,
            "chunk_seconds": 0.48,
            "stream_step_seconds": 0.48,
            "left_chunks": None,
            "load_error": None,
            "load_seconds": 0.1,
            "loaded_at": self.loaded_at,
        }

    def metrics(self):
        return {"requests_total": self.requests, "loaded": True}

    def transcribe(self, wav, **kwargs):
        assert isinstance(wav, np.ndarray)
        self.calls.append(kwargs)
        self.requests += 1
        return list(self._segments)

    def iter_stream(self, wav, **kwargs):
        self.calls.append(kwargs)
        self.requests += 1
        if self._events is not None:
            yield from self._events
            return
        yield StreamEvent("partial", "xin", 0.0, 0.48)
        yield StreamEvent("final", "xin chào", 0.0, 1.2)
        yield StreamEvent("done", "", 0.0, len(wav) / SAMPLE_RATE)

    def new_live_session(
        self, left_chunks=None, chunk_seconds=None, use_lm=None, beam_width=None
    ):
        self.calls.append(
            {
                "left_chunks": left_chunks,
                "chunk_seconds": chunk_seconds,
                "use_lm": use_lm,
                "beam_width": beam_width,
            }
        )
        return FakeLiveSession(self)


class FakeLiveSession:
    def __init__(self, engine):
        self.engine = engine
        self._closed = False
        self.consumed_seconds = 0.0
        self._seen = 0

    @property
    def partial(self):
        return ""

    def accept(self, samples):
        self._seen += len(samples)
        self.consumed_seconds = self._seen / SAMPLE_RATE
        if self._seen >= int(0.48 * SAMPLE_RATE):
            return [StreamEvent("final", "xin chào", 0.0, self.consumed_seconds)]
        return [StreamEvent("partial", "xin", 0.0, self.consumed_seconds)]

    def flush(self):
        self._closed = True
        return "kết thúc"

    def close(self):
        self._closed = True


def wav_bytes(seconds: float = 1.0, sample_rate: int = 8000) -> bytes:
    """A tiny WAV, written by hand so the test does not need soundfile."""
    import io
    import struct
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for i in range(int(seconds * sample_rate)):
            value = int(0.2 * 32767 * np.sin(2 * np.pi * 440.0 * i / sample_rate))
            frames += struct.pack("<h", value)
        handle.writeframes(bytes(frames))
    return buffer.getvalue()


@pytest.fixture()
def client(monkeypatch):
    from app import server

    fake = FakeEngine()
    monkeypatch.setattr(server, "engine", fake)
    with TestClient(server.app) as test_client:
        test_client.fake_engine = fake
        yield test_client


def _upload(client, **fields):
    data = {"file": ("a.wav", wav_bytes(), "audio/wav")}
    return client.post("/v1/audio/transcriptions", data=fields, files=data)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["loaded"] is True
    assert body["language"] == "vi"


def test_metrics(client):
    body = client.get("/metrics").json()
    assert "requests_total" in body
    assert body["model"]


def test_models_endpoints(client):
    listing = client.get("/v1/models").json()
    assert listing["object"] == "list"
    assert listing["data"][0]["id"] == "QUAN-Transcribe-1.0"
    assert client.get("/v1/models/QUAN-Transcribe-1.0").status_code == 200
    # Aliases OpenAI clients commonly send.
    assert client.get("/v1/models/whisper-1").status_code == 200


def test_models_rejects_unknown(client):
    response = client.get("/v1/models/gpt-4o")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_transcribe_json(client):
    response = _upload(client)
    assert response.status_code == 200
    assert response.json() == {"text": "xin chào"}


def test_transcribe_text_and_srt(client):
    text = _upload(client, response_format="text")
    assert text.headers["content-type"].startswith("text/plain")
    assert text.text == "xin chào"

    srt = _upload(client, response_format="srt")
    assert "00:00:00,000 --> 00:00:01,000" in srt.text

    vtt = _upload(client, response_format="vtt")
    assert vtt.text.startswith("WEBVTT")


def test_transcribe_verbose_json(client):
    body = _upload(client, response_format="verbose_json").json()
    assert body["text"] == "xin chào"
    assert body["segments"][0]["text"] == " xin chào"
    assert body["duration"] > 0


def test_transcribe_forwards_extensions(client):
    _upload(client, left_chunks="2", chunk_seconds="12.5", beam_width="8")
    call = client.fake_engine.calls[-1]
    assert call["left_chunks"] == 2
    assert call["chunk_seconds"] == 12.5
    assert call["beam_width"] == 8


def test_transcribe_rejects_bad_format(client):
    response = _upload(client, response_format="yaml")
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "response_format"


def test_transcribe_rejects_bad_language(client):
    response = _upload(client, language="en")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_language"


def test_transcribe_rejects_empty_file(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_audio_file"


def test_transcribe_rejects_undecodable_audio(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("junk.wav", b"not audio at all", "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_audio_file"


def test_missing_file_is_an_openai_error(client):
    response = client.post("/v1/audio/transcriptions", data={"model": "x"})
    assert response.status_code == 400
    assert "error" in response.json()


def test_unknown_route_uses_the_error_envelope(client):
    body = client.get("/nope").json()
    assert body["error"]["code"] == "not_found"


def test_streaming_sse(client):
    response = _upload(client, stream="true")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    kinds = [e["type"] for e in events]
    assert "transcript.text.delta" in kinds
    assert "transcript.text.segment" in kinds
    assert kinds[-1] == "transcript.text.done"
    assert events[-1]["text"] == "xin chào"


def test_streaming_drops_partials_when_asked(client):
    response = _upload(client, stream="true", stream_partials="false")
    kinds = [
        json.loads(line[6:])["type"]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert "transcript.text.delta" not in kinds
    assert "transcript.text.segment" in kinds

def test_streaming_forwards_lm_options(client):
    from app import server

    _upload(client, stream="true", use_lm="true", beam_width="8")
    assert server.engine.calls[-1]["use_lm"] is True
    assert server.engine.calls[-1]["beam_width"] == 8

def test_non_streaming_forwards_lm_options(client):
    from app import server

    _upload(client, use_lm="false", beam_width="4")
    assert server.engine.calls[-1]["use_lm"] is False
    assert server.engine.calls[-1]["beam_width"] == 4


def test_streaming_surfaces_engine_errors(monkeypatch):
    from app import server

    class Exploding(FakeEngine):
        def iter_stream(self, wav, **kwargs):
            raise RuntimeError("cuda out of memory")
            yield  # pragma: no cover

    fake = Exploding()
    monkeypatch.setattr(server, "engine", fake)
    with TestClient(server.app) as test_client:
        response = test_client.post(
            "/v1/audio/transcriptions",
            data={"stream": "true"},
            files={"file": ("a.wav", wav_bytes(), "audio/wav")},
        )
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[0]["type"] == "error"
    assert "cuda out of memory" in events[0]["error"]["message"]


def test_websocket_live_transcription(client):
    with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        assert created["sample_rate"] == SAMPLE_RATE
        # 0.6 s is past one 0.48 s step, so the fake session emits a finished
        # utterance rather than a partial.
        ws.send_bytes(np.zeros(int(0.6 * SAMPLE_RATE), dtype="<f4").tobytes())
        event = ws.receive_json()
        assert event["type"] == "transcript.text.segment"
        assert event["text"] == "xin chào"
        ws.send_text("stop")
        done = ws.receive_json()
        assert done["type"] == "transcript.text.done"
        assert done["final"] is True
        assert ws.receive_json()["type"] == "session.closed"


def test_websocket_pings_and_rejects_unknown(client):
    with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
        ws.receive_json()
        ws.send_text("ping")
        assert ws.receive_json()["type"] == "pong"
        ws.send_text("banana")
        error = ws.receive_json()
        assert error["type"] == "error"
        ws.send_text("stop")
        ws.receive_json()
        ws.receive_json()


def test_websocket_rejects_bad_format(client):
    with client.websocket_connect(
        "/v1/audio/transcriptions/stream?format=mp3"
    ) as ws:
        error = ws.receive_json()
        assert error["type"] == "error"
        assert "format" in error["error"]["message"]


def test_websocket_pcm16(client):
    with client.websocket_connect(
        "/v1/audio/transcriptions/stream?format=pcm16"
    ) as ws:
        assert ws.receive_json()["format"] == "pcm16"
        ws.send_bytes(np.zeros(int(0.6 * SAMPLE_RATE), dtype="<i2").tobytes())
        assert ws.receive_json()["type"] in {
            "transcript.text.delta",
            "transcript.text.segment",
        }


def test_websocket_handles_short_trailing_byte(client):
    """A chunked sender can leave a partial float32 at the end of a frame."""
    with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
        ws.receive_json()
        payload = np.zeros(1000, dtype="<f4").tobytes() + b"\x00\x01"
        ws.send_bytes(payload)
        ws.send_text("stop")
        # The stray byte must be ignored, not raise: either a transcript event
        # or the closing frame arrives first.
        first = ws.receive_json()
        assert first["type"] in {"transcript.text.delta", "transcript.text.done"}

def test_websocket_forwards_lm_options(client):
    from app import server

    with client.websocket_connect(
        "/v1/audio/transcriptions/stream?use_lm=false&beam_width=6"
    ) as ws:
        created = ws.receive_json()
        assert created["use_lm"] is False
        assert created["beam_width"] == 6
        ws.send_text("stop")
        ws.receive_json()
    call = server.engine.calls[-1]
    assert call["use_lm"] is False
    assert call["beam_width"] == 6

def test_websocket_rejects_lm_when_none_loaded(client):
    with client.websocket_connect(
        "/v1/audio/transcriptions/stream?use_lm=true"
    ) as ws:
        error = ws.receive_json()
        assert error["type"] == "error"
        assert "no KenLM model is loaded" in error["error"]["message"]

def test_websocket_rejects_bad_beam_width(client):
    with client.websocket_connect(
        "/v1/audio/transcriptions/stream?beam_width=wide"
    ) as ws:
        error = ws.receive_json()
        assert error["type"] == "error"
        assert "beam_width must be an integer" in error["error"]["message"]


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "QUAN-ASR" in response.text


def test_api_key_enforced(monkeypatch):
    from app import server

    monkeypatch.setattr(server, "engine", FakeEngine())
    monkeypatch.setattr(
        server, "settings", replace(server.settings, api_key="sk-test")
    )
    with TestClient(server.app) as client:
        assert client.get("/health").status_code == 200
        denied = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", wav_bytes(), "audio/wav")},
        )
        assert denied.status_code == 401
        assert denied.json()["error"]["code"] == "invalid_api_key"

        allowed = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer sk-test"},
            files={"file": ("a.wav", wav_bytes(), "audio/wav")},
        )
        assert allowed.status_code == 200

"""Vietnamese punctuation and capitalization restoration with vibert-capu.

Why this exists
---------------
``app/postprocess.py`` restores punctuation with regexes: a question-word list
and a hardcoded dictionary of place names and acronyms. That approach has a
ceiling it cannot pass. It can only ever append ``.`` or ``?`` at the end of a
segment, so a comma or a colon is unreachable, and any proper noun outside the
dictionary stays lowercase.

``dragonSwing/vibert-capu`` is a Vietnamese ViBERT fine-tuned on 5.6M OSCAR
samples for exactly this task: it predicts a GECToR-style edit per word, which
covers ``. , : ?`` and complex casing (``YouTube``, ``MobiFone``). Measured F1
on 10k held-out samples: ``.`` 0.82, ``,`` 0.71, ``:`` 0.64, ``?`` 0.78,
Upper 0.89, Complex-Upper 0.88.

This module wraps the ONNX export (``welcomyou/vibert-capu-onnx``) rather than
the original PyTorch checkpoint, for three reasons:

Where the weights come from
---------------------------
``scripts/fetch_punctuator.py`` vendors three files — ``config.json``,
``vocab.txt`` and one ONNX graph — into ``models/vibert-capu/``. That directory
is checked *before* the Hub, so a deployment that must not reach the network at
runtime, or an image with the weights baked in, never downloads anything. A
source checkout that never ran the script still works: the default Hub id is
used as a fallback. See :func:`_resolve_model`.

* ``onnxruntime`` is already a dependency of the ``vad`` extra, so the CPU image
  gains no new runtime. The PyTorch checkpoint would need the repo's own
  ``gec_model.py`` and ``vocabulary.py`` alongside it.
* The int8 build is 110 MB against 438 MB for fp32, at ~99% of the accuracy.
* Cold start is ~0.8 s against ~6 s.

The original checkpoint is not a plain transformers model — its ``config.json``
declares ``vocab_size: 15`` and ``num_detect_classes: 4``, so
``AutoModelForTokenClassification`` cannot load it. The ONNX export bakes in the
``Seq2Labels`` head, which is why this module only needs a tokenizer.

Two things make this safe to add to a serving path:

* **Lazy and optional.** ``onnxruntime`` and ``transformers`` are imported on
  first use, so the process still starts instantly and neither package is
  required when the punctuator is off. A missing package degrades to the regex
  path in ``app/postprocess.py`` instead of failing the request.
* **Never on the model thread.** This runs on request threads, like the VAD, so
  it can never delay a decode.

Licence
-------
The weights are **CC-BY-SA-4.0**, inherited from ``dragonSwing/vibert-capu``.
That is a share-alike licence and it is *not* the Apache-2.0 the rest of this
repository uses. It applies to the model weights, not to this wrapper.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: The sentinel the model expects in front of every sequence. It is not in
#: ``vocab.txt``; the checkpoint's ``special_tokens_fix`` adds it at load time.
START_TOKEN = "$START"

#: The Hub repo the vendored copy is taken from. ``scripts/fetch_punctuator.py``
#: downloads exactly the files this module reads.
HUB_REPO = "welcomyou/vibert-capu-onnx"

#: Vendored weights, relative to the repository root. Checked before the Hub.
VENDORED_DIR = os.path.join("models", "vibert-capu")

#: The ONNX graph per precision. int8 is 116 MB against 459 MB for fp32.
WEIGHTS = {"int8": "vibert-capu.int8.onnx", "fp32": "vibert-capu.onnx"}

#: GECToR edit actions, in the order ``vocabulary/labels.txt`` lists them. The
#: index into this tuple is the model's output class.
ACTIONS: tuple[str, ...] = (
    "$KEEP",
    "$TRANSFORM_CASE_CAPITAL",
    "$APPEND_,",
    "$APPEND_.",
    "$TRANSFORM_VERB_VB_VBN",
    "$TRANSFORM_CASE_UPPER",
    "$APPEND_:",
    "$APPEND_?",
    "$TRANSFORM_VERB_VB_VBC",
    "$TRANSFORM_CASE_LOWER",
    "$TRANSFORM_CASE_CAPITAL_1",
    "$TRANSFORM_CASE_UPPER_-1",
    "$MERGE_SPACE",
    "@@UNKNOWN@@",
    "@@PADDING@@",
)

#: Index of ``$KEEP`` in :data:`ACTIONS` — the "change nothing" class.
KEEP = 0

#: Labels that carry no instruction and must never be applied.
_INERT = frozenset({"$KEEP", "@@UNKNOWN@@", "@@PADDING@@"})

#: The model leaves a space before the mark it appends (``"nam ,"``); the
#: reference implementation closes that gap as a final pass.
_TIGHTEN = re.compile(r"\s+([.,:?])")


@dataclass(frozen=True)
class PunctuatorConfig:
    """Tuning for :class:`Punctuator`."""

    enabled: bool = False
    #: Hugging Face repo id, or a local directory holding the ONNX export.
    model: str = "welcomyou/vibert-capu-onnx"
    #: Prefer the int8 build: 110 MB against 438 MB, ~99% of the accuracy.
    int8: bool = True
    #: Minimum error probability before an edit is applied. ``0.0`` matches the
    #: reference implementation and applies every non-``$KEEP`` action.
    min_error_probability: float = 0.0
    #: Words per forward pass. The graph accepts up to 512 tokens; longer text
    #: is cut into windows of this many words and the pieces are rejoined.
    max_words: int = 200


def _apply_case(token: str, action: str) -> str:
    """Apply a ``$TRANSFORM_CASE_*`` action to one token.

    The suffix checks are ordered most-specific first: ``$TRANSFORM_CASE_UPPER``
    is a prefix of ``$TRANSFORM_CASE_UPPER_-1``, and ``$TRANSFORM_CASE_CAPITAL``
    of ``$TRANSFORM_CASE_CAPITAL_1``.
    """
    if action.endswith("LOWER"):
        return token.lower()
    if action.endswith("UPPER_-1"):
        return token[:-1].upper() + token[-1:]
    if action.endswith("UPPER"):
        return token.upper()
    if action.endswith("CAPITAL_1"):
        return token[:1] + token[1:].capitalize()
    if action.endswith("CAPITAL"):
        return token.capitalize()
    return token


def _resolve_merges(tokens: list[str]) -> list[str]:
    """Collapse ``$MERGE_*`` markers into the tokens they join.

    ``$MERGE_SPACE`` joins two words into one; ``$MERGE_HYPHEN`` joins them with
    a hyphen. A marker at either end has nothing to join and is dropped.
    """
    if not any(token.startswith("$MERGE_") for token in tokens):
        return tokens
    if tokens and tokens[0].startswith("$MERGE_"):
        tokens = tokens[1:]
    if tokens and tokens[-1].startswith("$MERGE_"):
        tokens = tokens[:-1]

    line = " ".join(tokens)
    line = line.replace(" $MERGE_HYPHEN ", "-")
    line = line.replace(" $MERGE_SPACE ", "")
    line = re.sub(r"([.,:?]\s+)+", r"\1", line)
    return line.split()


def edit_for(index: int, label: str) -> tuple[int, int, str] | None:
    """Turn one predicted label at padded ``index`` into a token-space edit.

    Mirrors ``GecBERTModel.get_token_action``. The padded sequence carries the
    ``$START`` sentinel at position 0, so every index shifts down by one to
    reach token coordinates. ``$APPEND_*`` and ``$MERGE_*`` act *after* the
    word, which is why their span is empty (``start == end``).

    Returns ``None`` for a label that carries no instruction.
    """
    if label in _INERT:
        return None
    if label.startswith("$APPEND_"):
        return index, index, label[len("$APPEND_") :]
    if label.startswith("$MERGE_"):
        return index, index, label
    if label == "$DELETE":
        return index - 1, index, ""
    if label.startswith("$TRANSFORM_") or label.startswith("$REPLACE_"):
        return index - 1, index, label
    return None


def apply_edits(tokens: list[str], edits: list[tuple[int, int, str]]) -> list[str]:
    """Apply GECToR edits to ``tokens``, returning the corrected list.

    ``edits`` are ``(start, end, action)`` triples in token coordinates, where
    ``start == end`` means "insert here". This mirrors
    ``get_target_sent_by_edits`` from the checkpoint's ``utils.py``, including
    the running ``shift`` that keeps later indices valid as tokens are inserted
    and deleted.
    """
    target = list(tokens)
    shift = 0
    #: Marks already appended, keyed by the token index they follow. The model
    #: can fire the same append on two adjacent words, and applying both would
    #: turn ``"a b"`` into ``"a . b ."``.
    appended: dict[int, str] = {}
    for start, end, action in edits:
        position = start + shift
        if start < 0 or position > len(target):
            continue
        source = target[position] if position < len(target) else ""

        if action == "":
            if position < len(target):
                del target[position]
                shift -= 1
        elif start == end:
            # ``$MERGE_*`` arrives here too: the reference inserts the marker as
            # a literal token and resolves it in ``_resolve_merges``. Everything
            # else in this branch is an append, already stripped of its prefix
            # by :func:`edit_for`.
            word = action.removeprefix("$APPEND_")
            if not action.startswith("$MERGE_"):
                if appended.get(start - 1) == word:
                    continue
                appended[start] = word
            # A mark that is already the neighbouring token is a no-op.
            if (position < len(target) and target[position] == word) or (
                position > 0 and target[position - 1] == word
            ):
                continue
            target[position:position] = [word]
            shift += 1
        elif action.startswith("$TRANSFORM_CASE_"):
            target[position] = _apply_case(source, action)
        elif action.startswith("$TRANSFORM_"):
            # Verb-form transforms are English-only. The Vietnamese checkpoint
            # never emits them, but a stray one must not corrupt the token.
            continue
        elif start == end - 1:
            target[position] = action.removeprefix("$REPLACE_")
    return _resolve_merges(target)


def _resolve_model(model: str, int8: bool) -> str:
    """Return a directory holding the ONNX graph, downloading only as a last resort.

    Resolution order:

    1. ``model`` itself, when it is a directory (an explicit override always
       wins, so ``PUNCTUATOR_MODEL=/opt/models/vibert-capu`` is honoured).
    2. The vendored ``models/vibert-capu`` beside the package, which is what
       ``scripts/fetch_punctuator.py`` writes and what the Docker images bake in.
    3. ``snapshot_download`` of :data:`HUB_REPO`, for a source checkout that
       never ran the fetch script.

    A vendored directory that is missing the requested precision falls through
    to the Hub rather than failing, so an int8-only vendor still serves an fp32
    request.
    """
    filename = WEIGHTS["int8" if int8 else "fp32"]
    if os.path.isdir(model):
        return model

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (os.path.join(root, VENDORED_DIR), VENDORED_DIR):
        if os.path.exists(os.path.join(candidate, filename)):
            return candidate

    from huggingface_hub import snapshot_download

    return snapshot_download(HUB_REPO)


class Punctuator:
    """Lazily-loaded vibert-capu ONNX punctuator, safe to share across threads.

    The ONNX session is loaded on first use behind a lock, so a burst of
    concurrent requests triggers exactly one load. Inference is serialised with
    a second lock: the session itself is thread-safe, but the tokenizer is not
    documented as such, and a short segment costs ~20 ms so the serialisation
    is not a bottleneck.
    """

    def __init__(self, config: PunctuatorConfig) -> None:
        self.config = config
        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        self._lock = threading.Lock()
        self.backend: str | None = None
        # Counters, mirrored into /metrics.
        self.calls = 0
        self.seconds = 0.0
        self.words = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def loaded(self) -> bool:
        return self._session is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> tuple[Any, Any]:
        """Load the ONNX session and tokenizer on first use."""
        if self._session is not None:
            return self._session, self._tokenizer
        with self._load_lock:
            if self._session is None:
                self._session, self._tokenizer = self._load()
        return self._session, self._tokenizer

    def _load(self) -> tuple[Any, Any]:
            started = time.perf_counter()
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
            except ImportError as exc:
                self._load_error = (
                    f"{exc.name} is not installed; install the 'punctuate' extra "
                    "(pip install '.[punctuate]') or set USE_PUNCTUATOR=false"
                )
                logger.warning("%s", self._load_error)
                raise RuntimeError(self._load_error) from exc

            # Resolve the model directory using the new resolution logic.
            path = _resolve_model(self.config.model, self.config.int8)
            filename = WEIGHTS["int8" if self.config.int8 else "fp32"]
            onnx_path = os.path.join(path, filename)
            if not os.path.exists(onnx_path):
                raise RuntimeError(f"{onnx_path} is missing from {path}")

            tokenizer = AutoTokenizer.from_pretrained(
                path, do_basic_tokenize=False, model_max_length=1024
            )
            # ``special_tokens_fix`` in the checkpoint config: the model was trained
            # with a sentinel in front of every sequence, and it is not in vocab.txt.
            tokenizer.add_tokens([START_TOKEN])

            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.backend = "int8" if self.config.int8 else "fp32"
            logger.info(
                "vibert-capu punctuator ready (%s) in %.2fs",
                self.backend,
                time.perf_counter() - started,
            )
            return session, tokenizer

    # -- inference -----------------------------------------------------------

    def restore(self, text: str) -> str:
        """Add punctuation and capitalization to one unpunctuated segment."""
        words = text.split()
        if not words:
            return text

        session, tokenizer = self.load()
        started = time.perf_counter()
        try:
            restored: list[str] = []
            for start in range(0, len(words), self.config.max_words):
                window = words[start : start + self.config.max_words]
                restored.extend(self._restore_window(window, session, tokenizer))
        finally:
            self.seconds += time.perf_counter() - started
            self.calls += 1
            self.words += len(words)

        return _TIGHTEN.sub(r"\1", " ".join(restored))

    def _restore_window(
        self, words: list[str], session: Any, tokenizer: Any
    ) -> list[str]:
        """Run one forward pass over at most ``max_words`` words."""
        import numpy as np

        sequence = [START_TOKEN, *words]
        encoded = tokenizer(
            [sequence],
            return_tensors="np",
            padding=True,
            is_split_into_words=True,
            truncation=True,
            add_special_tokens=False,
        )

        # ``input_offsets`` is the index of each word's first subword, which is
        # how the model maps its per-word output back onto the input.
        word_ids = encoded.word_ids(0)
        offsets = [0]
        for index in range(1, len(word_ids)):
            if word_ids[index] != word_ids[index - 1]:
                offsets.append(index)

        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = np.zeros_like(encoded["input_ids"])
        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
            "token_type_ids": token_type_ids.astype(np.int64),
            "input_offsets": np.array([offsets], dtype=np.int64),
        }

        with self._lock:
            logits, detect_logits = session.run(None, inputs)

        # ``detect_logits[:, :, 1]`` is P(INCORRECT); the reference gates the
        # whole sentence on its maximum. At the default 0.0 this never fires.
        if float(detect_logits[0, :, 1].max()) < self.config.min_error_probability:
            return list(words)

        classes = logits[0].argmax(-1)
        probabilities = logits[0].max(-1)
        edits: list[tuple[int, int, str]] = []
        for index, predicted in enumerate(classes):
            if predicted == KEEP:
                continue
            if probabilities[index] < self.config.min_error_probability:
                continue
            edit = edit_for(index, ACTIONS[predicted])
            if edit is not None:
                edits.append(edit)

        return apply_edits(list(words), edits)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "loaded": self.loaded,
            "backend": self.backend,
            "model": self.config.model,
            "int8": self.config.int8,
            "calls": self.calls,
            "seconds_total": round(self.seconds, 3),
            "words_total": self.words,
            "load_error": self._load_error,
        }

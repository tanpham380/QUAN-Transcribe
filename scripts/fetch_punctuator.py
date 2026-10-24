#!/usr/bin/env python
"""Download the vibert-capu ONNX punctuator into a local directory.

``app/punctuate.py`` can pull the model from the Hub on first use, but a
deployment that must not reach the network at runtime — or a Docker build that
wants the weights baked into the image — needs them on disk ahead of time.

Only three files are needed: ``config.json`` and ``vocab.txt`` for the
tokenizer, and one ONNX graph. The upstream repo also ships the PyTorch
checkpoint's own ``gec_model.py``/``vocabulary.py``, a 0.95 MB verb-form
vocabulary and the fp32 graph; this wrapper uses none of them.

    python scripts/fetch_punctuator.py
    python scripts/fetch_punctuator.py --precision fp32
    python scripts/fetch_punctuator.py --out /opt/models/vibert-capu

The weights are CC-BY-SA-4.0, inherited from ``dragonSwing/vibert-capu``; the
script writes a ``SOURCE.txt`` beside them so the attribution travels with the
copy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "models", "vibert-capu")
DEFAULT_REPO = "welcomyou/vibert-capu-onnx"

#: What the tokenizer needs. ``config.json`` is what makes ``AutoTokenizer``
#: pick ``BertTokenizer``; ``vocab.txt`` is the vocabulary itself.
TOKENIZER_FILES = ("config.json", "vocab.txt")

#: The ONNX graph per precision. int8 is 116 MB against 459 MB for fp32, at
#: ~99% of the accuracy.
WEIGHTS = {"int8": "vibert-capu.int8.onnx", "fp32": "vibert-capu.onnx"}


def fetch(repo: str, out: str, precision: str, revision: str | None) -> list[str]:
    """Download the three files into ``out`` and return their names."""
    from huggingface_hub import hf_hub_download

    wanted = [*TOKENIZER_FILES, WEIGHTS[precision]]
    os.makedirs(out, exist_ok=True)
    for name in wanted:
        print(f"  {name}", flush=True)
        hf_hub_download(repo_id=repo, filename=name, revision=revision, local_dir=out)
    # ``local_dir`` keeps a metadata cache beside the files. It is not part of
    # the export and would only confuse a reader of the directory.
    shutil.rmtree(os.path.join(out, ".cache"), ignore_errors=True)
    return wanted


def write_source(out: str, repo: str, revision: str | None, files: list[str]) -> None:
    """Record where the weights came from, for the share-alike attribution."""
    from huggingface_hub import HfApi

    try:
        sha = HfApi().model_info(repo, revision=revision).sha
    except Exception:
        sha = revision or "unknown"
    lines = [
        f"repo: {repo}",
        f"revision: {sha}",
        f"files: {', '.join(files)}",
        "license: CC-BY-SA-4.0 (inherited from dragonSwing/vibert-capu)",
        f"fetched: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}",
    ]
    with open(os.path.join(out, "SOURCE.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Hugging Face repo id (default {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="destination directory "
        f"(default {os.path.relpath(DEFAULT_OUT, REPO_ROOT)})",
    )
    parser.add_argument(
        "--precision",
        choices=sorted(WEIGHTS),
        default="int8",
        help="int8 is 116 MB, fp32 459 MB (default int8)",
    )
    parser.add_argument("--revision", default=None, help="pin a branch, tag or commit")
    args = parser.parse_args(argv)

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print(
            "huggingface_hub is not installed; pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    print(f"fetching {args.repo} ({args.precision}) into {args.out}")
    files = fetch(args.repo, args.out, args.precision, args.revision)
    write_source(args.out, args.repo, args.revision, files)
    total = sum(os.path.getsize(os.path.join(args.out, name)) for name in files)
    print(f"done: {total / 1e6:.1f} MB in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

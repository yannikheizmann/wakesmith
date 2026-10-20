"""Download pinned files with resumable, atomic transfers."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Callable
from pathlib import Path

from wakesmith.config import Args

from ._assets import FEATURES, KOKORO, MLS, MUSAN, OPENWAKEWORD, PIPER, RIRS, VOXCPM, Assets

HF = "https://huggingface.co"


def _session():
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = "wakesmith"
    if token := os.environ.get("HF_TOKEN"):
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def _repo_files(repo: str, revision: str, kind: str, patterns: list[str]) -> list[tuple[str, int]]:
    url, items, session = f"{HF}/api/{kind}s/{repo}/tree/{revision}?recursive=true", [], _session()
    while url:
        response = session.get(url, timeout=60)
        response.raise_for_status()
        items += [i for i in response.json() if i.get("type") == "file"]
        url = response.links.get("next", {}).get("url")
    return [
        (i["path"], i.get("size", 0))
        for i in items
        if any(fnmatch.fnmatch(i["path"], p) for p in patterns)
    ]


def download(url: str, target: Path, advance: Callable[[int], None] | None = None) -> None:
    """Fetch ``url`` into ``target``, resuming a partial file and publishing atomically."""
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with _session().get(url, stream=True, timeout=60, headers=headers) as response:
        if response.status_code == 416:
            partial.replace(target)
            return
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
        with open(partial, "ab" if offset else "wb") as out:
            for chunk in response.iter_content(chunk_size=1 << 20):
                out.write(chunk)
                if advance:
                    advance(len(chunk))
    if partial.stat().st_size == 0:
        raise RuntimeError(f"empty download: {url}")
    partial.replace(target)


def plan(assets: Assets, args: Args) -> list[tuple[str, Path, int]]:
    """List every missing ``(url, target, size)`` that a run needs."""
    items: list[tuple[str, Path, int]] = []

    def hub(source: tuple[str, str, str], patterns: list[str], root: Path) -> None:
        repo, revision, kind = source
        prefix = "" if kind == "model" else "datasets/"
        for path, size in _repo_files(repo, revision, kind, patterns):
            if not (root / path).exists():
                items.append((f"{HF}/{prefix}{repo}/resolve/{revision}/{path}", root / path, size))

    hub(FEATURES, [assets.acav.name, assets.validation.name], assets.acav.parent)
    hub(RIRS, ["16khz/*.wav"], assets.rirs.parent)
    hub(MUSAN, ["music/*", "noise/*", "speech/*"], assets.musan)
    hub(KOKORO, ["config.json", "kokoro-v1_0.pth", "voices/*.pt"], assets.kokoro)
    if args.voxcpm.enabled:
        hub(VOXCPM, ["*"], assets.voxcpm)
    if args.negative_speech.language:
        from wakesmith.config import LANGUAGES

        subset = LANGUAGES[args.negative_speech.language].mls
        shards = [
            f"{subset}/train-{i:05d}-of-*.parquet" for i in range(args.negative_speech.train_shards)
        ]
        hub(MLS, [*shards, f"{subset}/dev-*.parquet", f"{subset}/test-*.parquet"], assets.mls)
    if args.language == "en":
        for suffix, target in (
            (".state_dict.pt", assets.piper),
            (".config.json", assets.piper.with_suffix(".json")),
        ):
            if not target.exists():
                items.append((PIPER + suffix, target, 0))
    for name in ("melspectrogram.onnx", "embedding_model.onnx"):
        if not (assets.openwakeword / name).exists():
            items.append((f"{OPENWAKEWORD}/{name}", assets.openwakeword / name, 0))
    return [item for item in items if not item[1].name.startswith(".")]

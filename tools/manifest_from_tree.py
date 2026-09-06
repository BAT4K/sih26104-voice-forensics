"""Build a cross-session manifest from a speaker/session folder tree.

Most speech corpora already carry the structure cross-session verification
needs - a speaker directory containing one directory per recording session:

    <root>/<speaker>/<session>/*.wav
    <root>/<speaker>/<session>/*.flac

LibriSpeech (speaker/chapter), VoxCeleb (id/video) and most in-house call
archives all look like this. This walks such a tree and writes the manifest
that ``tools/eval_crosssession.py`` consumes.

    python tools/manifest_from_tree.py --root corpus/LibriSpeech/dev-clean \
        --out data/librispeech.jsonl --max-per-session 2
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("tree")

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def build(
    root: Path, out: Path, max_per_session: int = 2, min_sessions: int = 2,
    max_speakers: int = 0, language: str = "unknown",
) -> int:
    """Walk the tree and write one manifest line per selected clip.

    Args:
        root: Directory containing one folder per speaker.
        out: Manifest path to write.
        max_per_session: Clips to keep from each session. Extra clips from the
            same session add no cross-session information - they share a room,
            a microphone and a channel - so a small number keeps the run fast
            without weakening the protocol.
        min_sessions: Speakers with fewer sessions are dropped. Two is the
            minimum at which a leave-one-session-out trial exists at all.
        max_speakers: Cap for a quick run. 0 means no cap.
        language: Recorded in the manifest for later per-language breakdowns.

    Returns:
        Number of clips written.
    """
    by_speaker: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        rel = path.relative_to(root).parts
        if len(rel) < 3:
            continue  # need speaker/session/file
        by_speaker[rel[0]][rel[1]].append(path)

    usable = {s: v for s, v in by_speaker.items() if len(v) >= min_sessions}
    dropped = len(by_speaker) - len(usable)
    if dropped:
        log.info("%d speaker(s) dropped for having fewer than %d sessions", dropped, min_sessions)

    names = sorted(usable)
    if max_speakers:
        names = names[:max_speakers]

    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out, "w", encoding="utf-8") as fh:
        for speaker in names:
            for session, files in sorted(usable[speaker].items()):
                for f in files[:max_per_session]:
                    fh.write(json.dumps({
                        "path": str(f), "label": "real", "speaker": speaker,
                        "source_url": f"{speaker}/{session}", "context": session,
                        "language": language,
                    }) + "\n")
                    written += 1

    sessions = sum(len(usable[s]) for s in names)
    log.info("%d speakers, %d sessions, %d clips -> %s", len(names), sessions, written, out)
    log.info("%.1f sessions per speaker", sessions / max(len(names), 1))
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Manifest from a speaker/session tree.")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-per-session", type=int, default=2)
    ap.add_argument("--min-sessions", type=int, default=2)
    ap.add_argument("--max-speakers", type=int, default=0)
    ap.add_argument("--language", default="unknown")
    args = ap.parse_args()
    if not args.root.is_dir():
        ap.error(f"{args.root} is not a directory")
    build(args.root, args.out, args.max_per_session, args.min_sessions,
          args.max_speakers, args.language)


if __name__ == "__main__":
    main()

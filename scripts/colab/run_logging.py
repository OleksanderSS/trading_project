"""Capture everything a training run says, into one file.

The Colab trainer reports progress with print(), and print() goes to the
notebook's output and nowhere else. Library messages go through
ProjectLogger to logs/system.log. So a run's story was split in two, with
the half that says what is actually happening -- tickers, models, per-window
metrics -- not written down at all. When a run took two hours and ended in
"0 predictions", the only record was whatever was still scrolled into the
cell.

`start_run_log()` puts all of it in one place:

  - stdout and stderr are TEED, not redirected: the notebook keeps showing
    progress live, and the same bytes land in the file. Redirecting would
    leave the operator staring at a blank cell for two hours.
  - a FileHandler joins the root logger, so anything logged by src/ modules
    lands in the same file, interleaved in real time with the prints.
  - the file is flushed on every write. A run that dies takes its buffer
    with it otherwise, which is exactly the run whose log is wanted.

Works the same in Colab, in an IDE-hosted notebook, and in a plain shell.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


class _Tee:
    """Write to two streams. Not a redirect -- both keep receiving."""

    def __init__(self, primary, mirror):
        self._primary = primary
        self._mirror = mirror

    def write(self, text: str) -> int:
        self._primary.write(text)
        # Flushed per write: a crashed run must not take the tail of its own
        # log with it, and that is the log anyone actually needs.
        self._primary.flush()
        self._mirror.write(text)
        self._mirror.flush()
        return len(text)

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        # tqdm and friends ask. Answering honestly for the real stream keeps
        # progress bars behaving as they would without the tee.
        return getattr(self._primary, "isatty", lambda: False)()

    def __getattr__(self, name):
        return getattr(self._primary, name)


_ACTIVE: dict[str, object] = {}


def start_run_log(path: str | Path | None = None, name: str = "colab_run") -> Path:
    """Begin capturing stdout, stderr and logging to one file.

    Returns the log's path. Calling twice replaces the previous capture
    rather than stacking a second tee on top of the first -- nested tees
    duplicate every line once per call, which turns a long log into an
    unreadable one.
    """
    stop_run_log()

    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(f"{name}_{stamp}.log")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handle = open(path, "w", encoding="utf-8", buffering=1)
    handle.write(f"=== run log opened {datetime.now().isoformat()} ===\n")

    _ACTIVE["file"] = handle
    _ACTIVE["stdout"] = sys.stdout
    _ACTIVE["stderr"] = sys.stderr
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)

    # StreamHandler onto the handle already open, NOT a second FileHandler on
    # the same path. Two independent handles to one file interleave their
    # buffers and shred each other's lines -- the first version of this did
    # exactly that, and a logged message came out of the file as its own last
    # seven characters.
    file_handler = logging.StreamHandler(handle)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(logging.INFO)
    root = logging.getLogger()
    # The trainer calls ProjectLogger, which configures the root logger for
    # its own purposes; attaching here rather than replacing leaves that
    # alone and simply adds a second destination.
    root.addHandler(file_handler)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    _ACTIVE["handler"] = file_handler

    print(f"📝 Run log: {path.resolve()}")
    return path


def stop_run_log() -> None:
    """Restore the streams and close the file. Safe to call when inactive."""
    handler = _ACTIVE.pop("handler", None)
    if handler is not None:
        logging.getLogger().removeHandler(handler)
        handler.close()
    for stream in ("stdout", "stderr"):
        original = _ACTIVE.pop(stream, None)
        if original is not None:
            setattr(sys, stream, original)
    handle = _ACTIVE.pop("file", None)
    if handle is not None:
        handle.write(f"=== run log closed {datetime.now().isoformat()} ===\n")
        handle.close()

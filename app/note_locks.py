"""Who is writing in a note right now, so nobody else types over them.

Kept in memory, which works because the app is one process (see CLAUDE.md, gunicorn). A browser keeps its lock by
claiming it again every few seconds, so a lock whose owner lost connection or closed the page ends by itself after
TTL_S. That person's unsaved text is then caught when it arrives: saves check what the note was based on."""

import threading
import time

TTL_S = 25

_locks = {}  # (project_id, paper_id) -> (client, name, expires)
_guard = threading.Lock()


def holder(project_id: int, paper_id: int, client: str):
    """The name of whoever else holds this note ("" if unknown), or None if it is free or this client's."""
    with _guard:
        lock = _locks.get((project_id, paper_id))
        if lock is None or lock[2] < time.monotonic():
            _locks.pop((project_id, paper_id), None)
            return None
        return None if lock[0] == client else lock[1]


def claim(project_id: int, paper_id: int, client: str, name: str):
    """Take or renew the lock. Returns the same as holder(): None if this client has it now."""
    other = holder(project_id, paper_id, client)
    if other is None:
        with _guard:
            _locks[(project_id, paper_id)] = (client, name, time.monotonic() + TTL_S)
    return other


def release(project_id: int, paper_id: int, client: str) -> None:
    with _guard:
        lock = _locks.get((project_id, paper_id))
        if lock is not None and lock[0] == client:
            del _locks[(project_id, paper_id)]

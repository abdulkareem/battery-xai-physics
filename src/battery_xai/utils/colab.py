"""Google Colab helpers."""
from __future__ import annotations

from pathlib import Path


def maybe_mount_drive(enabled: bool = True, mount_point: str = "/content/drive") -> Path | None:
    """Mount Google Drive in Colab when available.

    Returns the mount path if mounted; returns ``None`` outside Colab or when
    mounting is disabled.
    """

    if not enabled:
        return None
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        return None
    drive.mount(mount_point)
    return Path(mount_point)

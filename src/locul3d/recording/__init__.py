"""Locul3D video recording — offscreen-FBO capture → ffmpeg → mp4.

Public surface:

    from locul3d.recording.recorder import VideoRecorder
    from locul3d.recording.encoders import (
        list_available_encoders,
        select_encoder,
        EncoderUnavailable,
    )
"""

from .encoders import (  # noqa: F401
    EncoderUnavailable,
    list_available_encoders,
    select_encoder,
)
from .recorder import VideoRecorder  # noqa: F401

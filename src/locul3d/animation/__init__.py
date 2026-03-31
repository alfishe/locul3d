"""Animation engine for the Locul3D Remote Control API.

Optional, detachable package.  The viewer/editor works without it.
"""

from .engine import AnimationEngine


def create_engine(viewport, dispatcher=None):
    """Factory function — creates and returns an AnimationEngine."""
    return AnimationEngine(viewport, dispatcher=dispatcher)

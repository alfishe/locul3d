"""Signal utilities."""

from PySide6.QtCore import Signal


class Connection:
    """Helper for managing Qt signal connections."""
    
    def __init__(self, signal, slot):
        self.signal = signal
        self.slot = slot
        self.connected = True
    
    def disconnect(self):
        if self.connected:
            self.signal.disconnect(self.slot)
            self.connected = False

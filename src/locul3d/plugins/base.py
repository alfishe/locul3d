"""Plugin base classes for extensible functionality."""

from abc import ABC, abstractmethod
from typing import Optional


class Plugin(ABC):
    """Base class for all plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass


class ImporterPlugin(Plugin):
    """Base class for file import plugins."""

    @property
    @abstractmethod
    def file_extensions(self) -> list[str]:
        """List of file extensions this plugin handles (e.g., ['.ply', '.obj'])."""
        pass

    @abstractmethod
    def can_import(self, file_path: str) -> bool:
        """Check if this plugin can import the given file."""
        pass

    @abstractmethod
    def import_file(self, file_path: str) -> Optional['LayerData']:
        """Import file and return LayerData object."""
        pass


class ToolPlugin(Plugin):
    """Base class for annotation tool plugins."""

    @property
    @abstractmethod
    def tool_id(self) -> str:
        """Unique identifier for this tool (e.g., 'select', 'move')."""
        pass

    @property
    @abstractmethod
    def shortcut(self) -> Optional[str]:
        """Keyboard shortcut (e.g., 'Q', 'G')."""
        pass

    @abstractmethod
    def activate(self, viewport):
        """Activate tool in given viewport."""
        pass

    @abstractmethod
    def deactivate(self, viewport):
        """Deactivate tool in given viewport."""
        pass

    @abstractmethod
    def handle_mouse_press(self, event, viewport):
        """Handle mouse press event."""
        pass

    @abstractmethod
    def handle_mouse_move(self, event, viewport):
        """Handle mouse move event."""
        pass

    @abstractmethod
    def handle_mouse_release(self, event, viewport):
        """Handle mouse release event."""
        pass


class PluginManager:
    """Manages plugin registration and discovery."""

    def __init__(self):
        self._importers: dict[str, ImporterPlugin] = {}
        self._tools: dict[str, ToolPlugin] = {}
        self._default_importer: Optional[str] = None

    def register_importer(self, plugin: ImporterPlugin):
        """Register a file importer plugin."""
        for ext in plugin.file_extensions:
            self._importers[ext] = plugin

    def register_tool(self, plugin: ToolPlugin):
        """Register a tool plugin."""
        self._tools[plugin.tool_id] = plugin

    def get_importer(self, file_path: str) -> Optional[ImporterPlugin]:
        """Get appropriate importer for given file."""
        from pathlib import Path
        ext = Path(file_path).suffix.lower()
        return self._importers.get(ext)

    def get_tool(self, tool_id: str) -> Optional[ToolPlugin]:
        """Get tool by ID."""
        return self._tools.get(tool_id)

    def list_importers(self) -> list[str]:
        """List all registered importers."""
        return [p.name for p in self._importers.values()]

    def list_tools(self) -> list[str]:
        """List all registered tools."""
        return [p.name for p in self._tools.values()]

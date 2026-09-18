"""studio_mcp — Local AI Studio MCP sunucusu (Faz 12): stdio + streamable HTTP.

Araçlar Studio'nun `/v1` REST API'sini çağırır (`StudioClient`); sunucu API
sürecine mount edilebilir (`create_server` + `MCPServer.streamable_http_app`)
ya da `studio-mcp` komutuyla ayrı bir süreç/makine olarak çalışır.
"""

from .client import StudioApiError, StudioClient, StudioTimeout
from .server import create_server

__version__ = "0.1.0"

__all__ = ["StudioApiError", "StudioClient", "StudioTimeout", "create_server"]

"""MCP (Model Context Protocol) stdio server for Laya (optional extra).

Install the extra to use it:

    pip install "laya-multimodal[mcp]"

Then run the server:

    laya-multimodal-mcp-server            # console script
    python -m laya_multimodal.mcp.server  # equivalent module form

The submodules ``device``, ``remote`` and ``tools`` are importable without the
``mcp`` dependency; only ``server`` requires the extra.
"""

__all__ = ["device", "remote", "server", "tools"]

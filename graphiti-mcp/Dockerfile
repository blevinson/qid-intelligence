FROM ghcr.io/blevinson/graphiti-mcp:claude-code-2026-05-01d

# QIDP-232: pass host="0.0.0.0" to FastMCP() so DNS rebinding protection
# does not activate with localhost-only allowed_hosts for cluster-internal access.
COPY graphiti_mcp_server.py /app/src/graphiti_mcp_server.py

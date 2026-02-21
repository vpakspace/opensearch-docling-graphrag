#!/usr/bin/env python3
"""Launch the OpenSearch Docling GraphRAG API server."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=8508,
        reload=False,
        log_level="info",
    )

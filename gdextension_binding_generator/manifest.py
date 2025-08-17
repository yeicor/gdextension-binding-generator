
import sys
import os
import platform
import shlex
from datetime import datetime, timezone
try:
    # Python 3.8+
    from importlib import metadata as importlib_metadata  # type: ignore
except Exception:
    importlib_metadata = None  # type: ignore

import json
from typing import Sequence
from .generate_bindings import GenerationContext
from .models import ClassInfo
from .utils import write_text

import logging
logger = logging.getLogger(__name__)

def emit_manifest(ctx: GenerationContext, classes: Sequence[ClassInfo]) -> None:
    """
    Emit a JSON manifest of the generation inputs, including generator metadata,
    version, and full command-line invocation. Useful for debugging and testing.
    """
    # Lazy imports to keep module import-time light

    # Resolve generator version as best as possible
    generator_version = None
    if importlib_metadata is not None:
        for dist_name in (
            "gdextension-binding-generator",
            "gdextension_binding_generator",
        ):
            try:
                generator_version = importlib_metadata.version(dist_name)  # type: ignore[attr-defined]
                if generator_version:
                    break
            except Exception:
                pass
    if not generator_version:
        # Fallback: try package-level __version__
        try:
            import importlib
            top_pkg = (__package__ or "").split(".")[0]
            if top_pkg:
                pkg_mod = importlib.import_module(top_pkg)
                generator_version = getattr(pkg_mod, "__version__", None)
        except Exception:
            generator_version = None

    # Command line info
    argv = list(getattr(sys, "argv", []) or [])
    command_line = " ".join(shlex.quote(a) for a in argv) if argv else ""

    # Environment / runtime info
    now_utc = datetime.now(timezone.utc).isoformat()
    env_info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
        "timestamp_utc": now_utc,
    }

    manifest = {
        # Generator metadata
        "generator": {
            "name": "gdextension-binding-generator",
            "url": "https://github.com/yeicor/gdextension-binding-generator",
            "version": generator_version or "unknown",
        },

        # Invocation and environment details
        "invocation": {
            "argv": argv,
            "command_line": command_line,
        },
        "environment": env_info,

        # Main configuration snapshot
        "prefix": ctx.prefix,
        "output_dir": str(ctx.output_dir),
        "classes_dir": str(ctx.classes_dir),
        "class_count": len(classes),
        "classes": [c.to_dict() for c in classes],
    }

    manifest_path = ctx.output_dir / "manifest.json"
    content = json.dumps(manifest, indent=2)
    try:
        write_text(manifest_path, content, dry_run=ctx.dry_run)
    except Exception:
        logger.exception("Failed to write manifest to %s; continuing without manifest", manifest_path)

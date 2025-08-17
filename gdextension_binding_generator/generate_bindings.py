#!/usr/bin/env python3
"""
GDExtension C++ binding generator (production-ready, modular)

This entrypoint wires together:
- Parsing (libclang-based) to discover classes and detailed method info
- Emitting (Jinja2-based) to generate Godot 4.4 GDExtension wrapper sources

Outputs:
- <output_dir>/register_types.h
- <output_dir>/register_types.cpp
- <output_dir>/classes/<WrapperName>.h
- <output_dir>/classes/<WrapperName>.cpp
- <optional> <output_dir>/manifest.json (for introspection)

Usage (example):
  python -m gdextension_binding_generator.generate_bindings \
    --prefix OCC_ \
    --headers path/to/opencascade/include \
    --headers path/to/other/headers \
    --clang-args "-Ipath/to/opencascade/include -std=c++17" \
    --output-dir src/generated

Notes:
- You need libclang and Jinja2 installed in your Python environment.
- For libclang discovery issues, ensure your environment can locate the libclang shared library.
"""

from __future__ import annotations

import argparse
import re
import sys
import shlex
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import logging

logger = logging.getLogger(__name__)

# Local modules
from .models import GenerationContext
from .utils import TemplateRenderer, configure_logging
from .manifest import emit_manifest
from .parsing.clang_parser import collect_classes_from_headers
from .emitters.godot_emitter import GodotEmitter, EmitterConfig
from .emitters.godot_variant_emitter import GodotVariantEmitter, VariantEmitterConfig
from .type_mapping import MappingConfig, TypeMapper


# --------------------------
# Helpers
# --------------------------

def discover_header_files(paths: List[str]) -> List[Path]:
    """
    Expand files and directories into a unique, sorted list of header files.
    """
    results: List[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_file() and pp.suffix.lower() in (".h", ".hpp", ".hh", ".hxx"):
            results.append(pp.resolve())
        elif pp.is_dir():
            for ext in ("*.h", "*.hpp", "*.hh", "*.hxx"):
                results.extend(sorted(Path(pp).rglob(ext)))
        else:
            logger.warning("Skipping non-existent path: %s", p)

    # De-duplicate preserving order
    seen: set[str] = set()
    unique: List[Path] = []
    for f in results:
        s = str(f.resolve())
        if s in seen:
            continue
        seen.add(s)
        unique.append(Path(s))
    return unique


# --------------------------
# CLI
# --------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Godot 4.4 GDExtension bindings (modular, production-ready)")

    p.add_argument(
        "--headers",
        action="append",
        default=[],
        help="Header file or directory to parse (repeatable). If a directory is given, all .h/.hpp/.hh/.hxx files are parsed.",
    )
    p.add_argument(
        "--clang-args",
        default="",
        help="Additional clang arguments (e.g., -I/path/include -DDEFINE=1 -std=c++17)",
    )
    p.add_argument(
        "--exclude-regex",
        default="",
        help="Regex to exclude classes by name (spelling).",
    )
    p.add_argument(
        "--include-filter",
        action="append",
        default=[],
        help="Only include classes whose definition file path starts with any of these prefixes. Repeatable.",
    )
    p.add_argument(
        "--output-dir",
        default="generated",
        help="Output directory for generated code (root for register_types, and classes/).",
    )
    p.add_argument(
        "--templates-dir",
        default=None,
        help="Optional templates directory. If omitted, package templates and embedded defaults are used.",
    )
    p.add_argument(
        "--godot-include",
        action="append",
        default=[],
        help="Path(s) to godot-cpp includes (metadata only; not used directly by the generator).",
    )
    p.add_argument(
        "--prefix",
        default="gdextension_binding_generator_prefix_",
        help="Prefix to add to generated wrapper class names (e.g., OCC_).",
    )
    p.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not emit the JSON manifest alongside generated sources.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run discovery and report classes/methods without writing files.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for DEBUG). Repeat for more detail."
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="Decrease verbosity (-q for WARNING, -qq for ERROR)."
    )
    p.add_argument(
        "--log-level",
        choices=["CRITICAL","ERROR","WARNING","INFO","DEBUG","NOTSET","critical","error","warning","info","debug","notset"],
        default=None,
        help="Explicit log level (overrides -v/-q)."
    )
    p.add_argument(
        "--log-format",
        default="%(levelname)s: %(message)s",
        help="Logging format string."
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="Optional file to write logs to."
    )
    # Variant-aware emitter options
    p.add_argument(
        "--variant-api",
        action="store_true",
        help="Use Variant-aware emitter: only expose methods with Variant-compatible types and conversions."
    )
    p.add_argument(
        "--no-opaque-handles",
        action="store_true",
        help="When using --variant-api, disable exposing unknown pointer returns as opaque uint64_t handles."
    )
    p.add_argument(
        "--no-std-string",
        action="store_true",
        help="When using --variant-api, disable std::string <-> godot::String mapping."
    )

    return p.parse_args(argv)


# --------------------------
# Main
# --------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ns = parse_args(argv)

    # Configure logging as early as possible
    if getattr(ns, "log_level", None):
        level_name = str(ns.log_level).upper()
        level = getattr(logging, level_name, logging.INFO)
    else:
        if getattr(ns, "verbose", 0) >= 1:
            level = logging.DEBUG
        elif getattr(ns, "quiet", 0) >= 2:
            level = logging.ERROR
        elif getattr(ns, "quiet", 0) == 1:
            level = logging.WARNING
        else:
            level = logging.INFO

    configure_logging(
        level=level,
        to_file=ns.log_file,
        fmt=getattr(ns, "log_format", "%(levelname)s: %(message)s"),
    )

    out_dir = Path(ns.output_dir).resolve()
    classes_dir = out_dir / "classes"

    ctx = GenerationContext(
        output_dir=out_dir,
        classes_dir=classes_dir,
        templates_dir=Path(ns.templates_dir).resolve() if ns.templates_dir else None,
        godot_includes=[str(Path(p).resolve()) for p in ns.godot_include],
        prefix=ns.prefix,
        dry_run=ns.dry_run,
    )

    # Initialize renderer (layered: user dir -> package templates -> embedded defaults)
    try:
        renderer = TemplateRenderer(ctx.templates_dir)
    except Exception as e:
        logger.exception("Failed to initialize templating")
        return 1

    # Discover headers
    headers = discover_header_files(ns.headers)
    if not headers:
        logger.error("No headers found to parse. Provide --headers.")
        return 2

    # Prepare parsing parameters
    try:
        clang_args = shlex.split(ns.clang_args) if ns.clang_args else []
    except Exception as ex:
        # Fallback if shlex fails due to platform-specific quoting
        logger.warning("Falling back to naive clang args split due to parsing error: %s", ex)
        clang_args = [a for a in ns.clang_args.split(" ") if a.strip()]

    exclude_re = re.compile(ns.exclude_regex) if ns.exclude_regex else None

    # Parse classes and methods
    try:
        classes = collect_classes_from_headers(
            headers=headers,
            clang_args=clang_args,
            include_filters=ns.include_filter or None,
            exclude_class_regex=exclude_re,
            prefix=ctx.prefix,
            emit_diagnostics=True,
        )
    except Exception:
        logger.exception("Failed to collect classes")
        return 3

    # Report discovery
    logger.info("Discovered %d class(es)", len(classes))
    for c in classes:
        inst_count = sum(1 for m in c.methods if m.kind.name in ("INSTANCE", "OPERATOR"))
        stat_count = sum(1 for m in c.methods if m.kind.name == "STATIC")
        logger.debug("Class %s (-> wrapper %s) methods: instance=%d, static=%d", c.qualified_name, c.wrapper_name, inst_count, stat_count)

    if ctx.dry_run:
        logger.info("Dry-run complete (no files written).")
        return 0

    # Emit sources
    try:
        if getattr(ns, "variant_api", False):
            # Configure TypeMapper based on CLI
            cfg = MappingConfig(
                prefix=ctx.prefix,
                enable_opaque_handles=not getattr(ns, "no_opaque_handles", False),
                use_wrapped_param_bridge=True,
                enable_std_string=not getattr(ns, "no_std_string", False),
            )
            mapper = TypeMapper.from_classes(classes, config=cfg)
            ve_cfg = VariantEmitterConfig()
            emitter = GodotVariantEmitter(
                ctx=ctx,
                renderer=renderer,
                config=ve_cfg,
                mapper=mapper,
            )
        else:
            emitter = GodotEmitter(
                ctx=ctx,
                renderer=renderer,
                config=EmitterConfig(),
            )
        emitter.emit(classes)
    except Exception:
        logger.exception("Failed to generate files")
        return 4



    # Optional: emit a JSON manifest of the generation data for debugging/inspection.
    if not ns.no_manifest:
        try:
            emit_manifest(ctx, classes)
        except Exception:
            logger.exception("Failed to emit generation manifest")
            return 5

    return 0


if __name__ == "__main__":
    sys.exit(main())

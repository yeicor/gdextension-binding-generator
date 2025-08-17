#!/usr/bin/env python3
"""
Emitter module for generating Godot GDExtension wrapper sources.

This module takes the parsed data model (classes, methods, etc.) and uses a
Jinja2-based renderer to emit:

- register_types.h
- register_types.cpp
- classes/<WrapperName>.h
- classes/<WrapperName>.cpp

Design goals:
- Clean separation of concerns from parsing and data modeling.
- Production-grade file writing (atomic, idempotent).
- Configurable template names.
- Optional manifest output for debugging and introspection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..models import ClassInfo, GenerationContext, build_template_context
from ..utils import TemplateRenderer, ensure_dir, write_text

logger = logging.getLogger(__name__)


# --------------------------
# Configuration
# --------------------------

@dataclass(frozen=True)
class EmitterConfig:
    """
    Configuration for the Godot code emitter.

    You can override template names to use custom ones in your templates directory
    or embedded defaults (see utils.TemplateRenderer for fallback layering).
    """
    register_types_header_template: str = "register_types.h.j2"
    register_types_source_template: str = "register_types.cpp.j2"
    class_header_template: str = "class_header.h.j2"
    class_source_template: str = "class_source.cpp.j2"


# --------------------------
# Emitter
# --------------------------

class GodotEmitter:
    """
    Emit Godot GDExtension wrapper code from parsed class metadata.

    Usage:
        emitter = GodotEmitter(ctx, renderer, config)
        emitter.emit(all_classes)

    Note:
        This emitter renders the full discovered API surface. If you only want to expose
        methods whose argument and return types are Variant-compatible (with automatic
        conversions such as String bridging, wrapped-class impl bridging, and optional
        opaque handles), consider using GodotVariantEmitter from
        gdextension_binding_generator.emitters.godot_variant_emitter.
    """

    def __init__(self, ctx: GenerationContext, renderer: TemplateRenderer, config: Optional[EmitterConfig] = None) -> None:
        self.ctx = ctx
        self.renderer = renderer
        self.config = config or EmitterConfig()

    # ---- Public API ----

    def emit(self, classes: Sequence[ClassInfo]) -> None:
        """
        Generate all outputs: register_types files, class headers/sources, and optional manifest.
        """
        # Ensure directories exist (idempotent)
        ensure_dir(self.ctx.output_dir)
        ensure_dir(self.ctx.classes_dir)

        # Emit register_types
        self._emit_register_types(classes)

        # Emit class wrappers
        for cls in classes:
            self._emit_class(cls)

        logger.info("Generation complete under: %s", self.ctx.output_dir)

    # ---- Internals ----

    def _emit_register_types(self, classes: Sequence[ClassInfo]) -> None:
        """
        Render and write register_types.h/.cpp using the configured templates.
        """
        context = build_template_context(self.ctx.prefix, classes)

        try:
            header_content = self.renderer.render(self.config.register_types_header_template, context)
            source_content = self.renderer.render(self.config.register_types_source_template, context)
        except Exception:
            logger.exception("Failed to render register_types templates; aborting generation")
            raise

        header_path = self.ctx.output_dir / "register_types.h"
        source_path = self.ctx.output_dir / "register_types.cpp"

        try:
            write_text(header_path, header_content, dry_run=self.ctx.dry_run)
            write_text(source_path, source_content, dry_run=self.ctx.dry_run)
        except Exception:
            logger.exception("Failed to write register_types files to %s", self.ctx.output_dir)
            raise

    def _emit_class(self, cls: ClassInfo) -> None:
        """
        Render and write the header/source for a single class.
        """
        # Keep class context compact and stable for templates
        cdict = cls.to_dict()
        context: Dict = {"cls": cdict, "prefix": self.ctx.prefix}

        # Warn if we are generating a minimal skeleton without API surface
        try:
            ctor_count = len(cdict.get("constructors", []))
            inst_count = len(cdict.get("instance_methods", []))
            static_count = len(cdict.get("static_methods", []))
        except Exception:
            ctor_count = inst_count = static_count = 0
        if (ctor_count + inst_count + static_count) == 0:
            logger.warning("Wrapper %s for native %s has no constructors or methods; generating minimal skeleton", cls.wrapper_name, cls.qualified_name)

        try:
            header_content = self.renderer.render(self.config.class_header_template, context)
            source_content = self.renderer.render(self.config.class_source_template, context)
        except Exception:
            logger.exception("Failed to render templates for wrapper %s; skipping this class", cls.wrapper_name)
            return

        header_path = self.ctx.classes_dir / f"{cls.wrapper_name}.h"
        source_path = self.ctx.classes_dir / f"{cls.wrapper_name}.cpp"

        try:
            write_text(header_path, header_content, dry_run=self.ctx.dry_run)
            write_text(source_path, source_content, dry_run=self.ctx.dry_run)
        except Exception:
            logger.exception("Failed to write wrapper files for %s; skipping this class", cls.wrapper_name)

__all__ = [
    "EmitterConfig",
    "GodotEmitter",
]

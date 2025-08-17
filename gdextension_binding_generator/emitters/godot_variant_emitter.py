#!/usr/bin/env python3
"""
Variant-aware emitter for Godot GDExtension wrappers.

This emitter uses the TypeMapper to:
- Only expose methods whose argument and return types are compatible with Godot Variant,
  either natively (int, float, bool), via conversions (char* <-> godot::String,
  std::string <-> godot::String), or by bridging wrapped classes (Ref<Wrapper> <-> impl*).
- Optionally expose opaque handles (uint64_t) for unknown pointer returns.
- Render per-call conversion code (temporaries, to-native expressions) and return conversions.

Outputs:
- <output_dir>/register_types.h (using existing template)
- <output_dir>/register_types.cpp (using existing template)
- <output_dir>/classes/<WrapperName>.h (mapped signatures)
- <output_dir>/classes/<WrapperName>.cpp (mapped implementations and conversions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import logging
import re

from ..models import ClassInfo, MethodInfo, MethodKind, build_template_context
from ..utils import TemplateRenderer, ensure_dir, write_text, sanitize_identifier
from ..type_mapping import (
    TypeMapper,
    default_occt_mapper,
    MappedKind,
    MappedMethod,
    MappedParameter,
)

logger = logging.getLogger(__name__)


# --------------------------
# Configuration
# --------------------------

@dataclass(frozen=True)
class VariantEmitterConfig:
    """
    Configuration for the Variant-aware Godot emitter.
    """
    register_types_header_template: str = "register_types.h.j2"
    register_types_source_template: str = "register_types.cpp.j2"
    # Whether to generate constructor wrappers (construct(), construct_1(), ...)
    emit_constructor_wrappers: bool = True
    # Whether to include static methods
    emit_static_methods: bool = True
    # If True, classes with zero supported methods are still emitted as skeletons
    emit_empty_classes: bool = True


# --------------------------
# Emitter
# --------------------------

class GodotVariantEmitter:
    """
    Emit Godot GDExtension wrapper code using TypeMapper to ensure Variant-compatible API.

    Usage:
        emitter = GodotVariantEmitter(ctx, renderer, config=config, mapper=mapper)
        emitter.emit(all_classes)
    """

    def __init__(
        self,
        ctx,
        renderer: TemplateRenderer,
        config: Optional[VariantEmitterConfig] = None,
        mapper: Optional[TypeMapper] = None,
    ) -> None:
        self.ctx = ctx
        self.renderer = renderer
        self.config = config or VariantEmitterConfig()
        self._mapper = mapper  # Can be None; if so, built in emit()
        # Emit opaque_handle_registry.h only once per run when needed.
        self._opaque_emitted = False

    # ---- Public API ----

    def emit(self, classes: Sequence[ClassInfo]) -> None:
        """
        Generate all outputs: register_types files and class headers/sources using mapped signatures.
        """
        ensure_dir(self.ctx.output_dir)
        ensure_dir(self.ctx.classes_dir)

        # Build TypeMapper if not provided
        mapper = self._mapper or default_occt_mapper(classes, prefix=self.ctx.prefix)

        # Emit register_types using existing templates (class list unaffected by mapping)
        self._emit_register_types(classes)

        # Emit class wrappers
        for ci in classes:
            self._emit_class(ci, mapper)

        logger.info("Variant-aware generation complete under: %s", self.ctx.output_dir)

    # ---- Internals ----

    def _emit_register_types(self, classes: Sequence[ClassInfo]) -> None:
        """
        Render and write register_types.h/.cpp using the configured templates.
        """
        context = build_template_context(self.ctx.prefix, classes)
        header_content = self.renderer.render(self.config.register_types_header_template, context)
        source_content = self.renderer.render(self.config.register_types_source_template, context)

        write_text(self.ctx.output_dir / "register_types.h", header_content, dry_run=self.ctx.dry_run)
        write_text(self.ctx.output_dir / "register_types.cpp", source_content, dry_run=self.ctx.dry_run)

    def _emit_class(self, ci: ClassInfo, mapper: TypeMapper) -> None:
        """
        Generate mapped header and source for a single class.
        """
        # Filter and map methods
        mapped_instance: List[MappedMethod] = []
        mapped_static: List[MappedMethod] = []

        for m in ci.methods:
            if m.kind in (MethodKind.CONSTRUCTOR, MethodKind.DESTRUCTOR):
                continue
            mm = mapper.map_method(ci, m)
            if not mm.supported:
                logger.debug("Skipping unsupported method: %s::%s (%s)", ci.qualified_name, m.name, mm.reason or "")
                continue
            if m.kind == MethodKind.STATIC and self.config.emit_static_methods:
                mapped_static.append(mm)
            elif m.kind in (MethodKind.INSTANCE, MethodKind.OPERATOR):
                mapped_instance.append(mm)

        # Constructors: map parameters only and emit construct wrappers if enabled
        mapped_ctors: List[Tuple[MethodInfo, List[Tuple[str, str]], List[str], List[str], List[str]]] = []
        if self.config.emit_constructor_wrappers:
            for ctor in ci.constructors:
                params_sig, pre_lines, arg_exprs, d_method_args = self._map_parameters_signature(mapper, ctor)
                if params_sig is None:
                    logger.debug("Skipping unsupported constructor: %s::%s", ci.qualified_name, ctor.cpp_signature)
                    continue
                mapped_ctors.append((ctor, params_sig, pre_lines, arg_exprs, d_method_args))

        if not mapped_instance and not mapped_static and not mapped_ctors and not self.config.emit_empty_classes:
            logger.info("No supported methods for %s; skipping wrapper generation", ci.qualified_name)
            return

        # Build variant mapping structure for templates
        variant = {
            "constructors": [],
            "instance_methods": [],
            "static_methods": [],
        }
        # Constructors (mapped)
        for ctor, params_sig, pre_lines, arg_exprs, _ in mapped_ctors:
            suffix = f"_{ctor.overload_index}" if ctor.overload_index is not None and ctor.overload_index > 0 else ""
            variant["constructors"].append({
                "name": ctor.name,
                "suffix": suffix,
                "params": [{"type": t, "name": n} for (t, n) in params_sig],
                "pre_lines": list(pre_lines),
                "call_args": list(arg_exprs),
                "return_type": "bool",
                "call_expr": None,
                "native_name": ctor.name,
                "return_transform": None,
                "default_return": "false",
            })
        # Instance methods (mapped)
        for mm in mapped_instance:
            # Build sanitized params and conversion/call data
            san_params = []
            pre_lines = []
            call_args = []
            for (ptype, pname), mp in zip(mm.exposed_param_list, mm.exposed_params):
                sname = sanitize_identifier(pname)
                san_params.append({"type": ptype, "name": sname})
                for pl in mp.mapping.pre_call_lines:
                    pre_lines.append(pl.replace("{var}", sname))
                if mp.mapping.to_native_expr:
                    call_args.append(mp.mapping.to_native_expr.replace("{var}", sname))
                else:
                    call_args.append(sname)
            ret_pre = list(mm.exposed_return.mapping.pre_call_lines)
            variant["instance_methods"].append({
                "name": mm.method_name,
                "native_name": mm.method_name,
                "exposed_name": mm.exposed_name,
                "return_type": mm.exposed_return.mapping.exposed_spelling,
                "params": san_params,
                "is_const": mm.is_const,
                "pre_lines": pre_lines + ret_pre,
                "call_args": call_args,
                "return_transform": mm.exposed_return.mapping.from_native_expr or None,
                "default_return": mm.default_return_expr,
                "call_expr": None,
            })
        # Static methods (mapped)
        for mm in mapped_static:
            # Build sanitized params and conversion/call data
            san_params = []
            pre_lines = []
            call_args = []
            for (ptype, pname), mp in zip(mm.exposed_param_list, mm.exposed_params):
                sname = sanitize_identifier(pname)
                san_params.append({"type": ptype, "name": sname})
                for pl in mp.mapping.pre_call_lines:
                    pre_lines.append(pl.replace("{var}", sname))
                if mp.mapping.to_native_expr:
                    call_args.append(mp.mapping.to_native_expr.replace("{var}", sname))
                else:
                    call_args.append(sname)
            ret_pre = list(mm.exposed_return.mapping.pre_call_lines)
            variant["static_methods"].append({
                "name": mm.method_name,
                "native_name": mm.method_name,
                "exposed_name": mm.exposed_name,
                "return_type": mm.exposed_return.mapping.exposed_spelling,
                "params": san_params,
                "pre_lines": pre_lines + ret_pre,
                "call_args": call_args,
                "return_transform": mm.exposed_return.mapping.from_native_expr or None,
                "call_expr": None,
            })

        context = {"cls": ci.to_dict(), "prefix": self.ctx.prefix, "variant": variant}
        header_text = self.renderer.render("variant_class_header.h.j2", context)
        source_text = self.renderer.render("variant_class_source.cpp.j2", context)

        # If any mapped methods use opaque handles, ensure registry header exists (once)
        handles_used = uses_opaque_handles(mapped_instance, mapped_static)
        if handles_used and not self._opaque_emitted:
            try:
                registry_hdr = self.renderer.render("opaque_handle_registry.h.j2", {})
                write_text(self.ctx.classes_dir / "opaque_handle_registry.h", registry_hdr, dry_run=self.ctx.dry_run)
                self._opaque_emitted = True
            except Exception:
                logger.exception("Failed to render/write opaque_handle_registry.h; continuing")

        # If handles are used in this class, include the registry header from the generated source
        if handles_used:
            marker = f'#include "{ci.wrapper_name}.h"'
            include_line = marker + '\n#include "opaque_handle_registry.h"'
            if marker in source_text:
                source_text = source_text.replace(marker, include_line, 1)
            else:
                source_text = '#include "opaque_handle_registry.h"\n' + source_text

        write_text(self.ctx.classes_dir / f"{ci.wrapper_name}.h", header_text, dry_run=self.ctx.dry_run)
        write_text(self.ctx.classes_dir / f"{ci.wrapper_name}.cpp", source_text, dry_run=self.ctx.dry_run)

    # ---- Parameter mapping helpers ----

    def _map_parameters_signature(
        self,
        mapper: TypeMapper,
        m: MethodInfo,
    ) -> Tuple[Optional[List[Tuple[str, str]]], List[str], List[str], List[str]]:
        """
        For a method or constructor:
        - Returns:
          - params_sig: list of (exposed_type, name) for signature, or None if unsupported
          - pre_lines: list of C++ lines to emit before the call
          - arg_exprs: list of expressions for native call-site
          - d_method_args: list of sanitized names for D_METHOD
        """
        params_sig: List[Tuple[str, str]] = []
        pre_lines: List[str] = []
        arg_exprs: List[str] = []
        d_method_args: List[str] = []

        for i, p in enumerate(m.parameters, start=1):
            pname = p.name or f"arg{i}"
            sname = sanitize_identifier(pname)
            mp = mapper._map_parameter(p.cpp_type, pname)  # Using internal mapping for parameters
            if mp.kind == MappedKind.UNSUPPORTED:
                return None, [], [], []
            params_sig.append((mp.exposed_spelling, sname))

            # Pre-call conversions (use sanitized name)
            for line in mp.pre_call_lines:
                pre_lines.append(line.replace("{var}", sname))

            # Native argument expression (use sanitized name)
            if mp.to_native_expr:
                arg_exprs.append(mp.to_native_expr.replace("{var}", sname))
            else:
                arg_exprs.append(sname)

            d_method_args.append(sname)

        return params_sig, pre_lines, arg_exprs, d_method_args

# ---- Utilities ----

def collect_forward_decl_wrappers(
    instance_methods: List[MappedMethod],
    static_methods: List[MappedMethod],
) -> List[str]:
    used: set[str] = set()
    for mm in list(instance_methods) + list(static_methods):
        # Params
        for mp in mm.exposed_params:
            _accumulate_wrapper_from_spelling(mp.mapping.exposed_spelling, used)
        # Return
        _accumulate_wrapper_from_spelling(mm.exposed_return.mapping.exposed_spelling, used)
    return list(used)

def _accumulate_wrapper_from_spelling(spelling: str, out: set) -> None:
    # Look for godot::Ref<WrapperName>
    m = re.match(r"\s*godot::Ref<\s*([A-Za-z_]\w*)\s*>\s*$", spelling or "")
    if m:
        out.add(m.group(1))

def uses_opaque_handles(
    instance_methods: List[MappedMethod],
    static_methods: List[MappedMethod],
) -> bool:
    def uses_in_method(mm: MappedMethod) -> bool:
        if mm.exposed_return.mapping.exposed_spelling.strip() == "uint64_t":
            return True
        for mp in mm.exposed_params:
            if mp.mapping.exposed_spelling.strip() == "uint64_t":
                return True
        return False

    return any(uses_in_method(mm) for mm in list(instance_methods) + list(static_methods))

def build_param_call_data(
    params: List[MappedParameter],
) -> Tuple[List[str], List[str]]:
    """
    Build pre-call conversion lines and native argument expressions for given mapped parameters.
    """
    pre_lines: List[str] = []
    native_args: List[str] = []
    for mp in params:
        name = mp.name
        # Pre lines with placeholder substitution
        for pl in mp.mapping.pre_call_lines:
            pre_lines.append(pl.replace("{var}", name))
        # Arg expression
        if mp.mapping.to_native_expr:
            native_args.append(mp.mapping.to_native_expr.replace("{var}", name))
        else:
            native_args.append(name)
    return pre_lines, native_args


__all__ = [
    "VariantEmitterConfig",
    "GodotVariantEmitter",
]

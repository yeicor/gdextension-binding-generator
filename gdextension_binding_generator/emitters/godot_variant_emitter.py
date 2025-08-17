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

        header_text = self._render_header(ci, mapped_instance, mapped_static, mapped_ctors)
        source_text = self._render_source(ci, mapped_instance, mapped_static, mapped_ctors)

        # If any mapped methods use opaque handles, emit the registry header alongside classes
        if uses_opaque_handles(mapped_instance, mapped_static):
            try:
                registry_hdr = self.renderer.render("opaque_handle_registry.h.j2", {})
                write_text(self.ctx.classes_dir / "opaque_handle_registry.h", registry_hdr, dry_run=self.ctx.dry_run)
            except Exception:
                logger.exception("Failed to render/write opaque_handle_registry.h; continuing")

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
            mp = mapper._map_parameter(p.cpp_type, pname)  # Using internal mapping for parameters
            if mp.kind == MappedKind.UNSUPPORTED:
                return None, [], [], []
            params_sig.append((mp.exposed_spelling, pname))

            # Pre-call conversions
            for line in mp.pre_call_lines:
                pre_lines.append(line.replace("{var}", pname))

            # Native argument expression
            if mp.to_native_expr:
                arg_exprs.append(mp.to_native_expr.replace("{var}", pname))
            else:
                arg_exprs.append(pname)

            d_method_args.append(sanitize_identifier(pname))

        return params_sig, pre_lines, arg_exprs, d_method_args

    # ---- Code rendering ----

    def _render_header(
        self,
        ci: ClassInfo,
        instance_methods: List[MappedMethod],
        static_methods: List[MappedMethod],
        mapped_ctors: List[Tuple[MethodInfo, List[Tuple[str, str]], List[str], List[str], List[str]]],
    ) -> str:
        """
        Generate the header content for a class using mapped signatures.
        """
        lines: List[str] = []
        lines.append("#pragma once")
        lines.append("/**")
        lines.append(" * This file is generated by the GDExtension binding generator (Variant-aware).")
        lines.append(" *")
        lines.append(f" * Wrapper: {ci.wrapper_name}")
        lines.append(f" * Native : {ci.qualified_name}")
        lines.append(" */")
        lines.append("")
        lines.append("#include <godot_cpp/classes/ref_counted.hpp>")
        lines.append("#include <godot_cpp/core/class_db.hpp>")
        # Include for opaque handles if used
        if uses_opaque_handles(instance_methods, static_methods):
            lines.append("#include <cstdint>")
        lines.append("")
        lines.append(f"// Include the native library header that defines {ci.qualified_name}.")
        lines.append(f"#include <{ci.include_header}>")
        lines.append("")

        # Collect forward declarations for wrapper types used as Ref<Wrapper>
        fwd_wrappers = collect_forward_decl_wrappers(instance_methods, static_methods)
        lines.append("namespace godot {")
        lines.append("")
        for w in sorted(fwd_wrappers):
            # Avoid forward declaring self
            if w != ci.wrapper_name:
                lines.append(f"class {w};")
        if fwd_wrappers:
            lines.append("")

        lines.append(f"class {ci.wrapper_name} : public RefCounted {{")
        lines.append(f"    GDCLASS({ci.wrapper_name}, RefCounted)")
        lines.append("")
        lines.append("  private:")
        lines.append("    // NOTE: Adjust ownership/lifetime as needed for your library.")
        lines.append("    // This skeleton uses a raw pointer; consider smart pointers or handles.")
        abs_q = f"::{ci.qualified_name}" if not ci.qualified_name.startswith('::') else ci.qualified_name
        lines.append(f"    {abs_q}* impl = nullptr;")
        lines.append("    bool owns_impl = true;")
        lines.append("")
        lines.append("  protected:")
        lines.append("    static void _bind_methods();")
        lines.append("")
        lines.append("  public:")
        lines.append(f"    {ci.wrapper_name}();")
        lines.append(f"    ~{ci.wrapper_name}();")
        lines.append("")
        lines.append("    // Helper API")
        lines.append("    bool is_valid() const;")
        lines.append("")
        lines.append("    // Ownership and impl management")
        lines.append("    void reset();")
        lines.append(f"    void set_impl({abs_q}* p, bool p_take_ownership = false);")
        lines.append(f"    {abs_q}* get_impl() const;")
        lines.append("    void set_owns_impl(bool p_own);")
        lines.append("    bool get_owns_impl() const;")
        lines.append("")

        # Constructors wrappers
        if mapped_ctors:
            lines.append("    // Construct the underlying native object using discovered constructors")
            for ctor, params_sig, _, _, _ in mapped_ctors:
                suffix = f"_{ctor.overload_index}" if ctor.overload_index is not None and ctor.overload_index > 0 else ""
                params_text = ", ".join(f"{t} {n}" for t, n in params_sig)
                lines.append(f"    // {ctor.cpp_signature}")
                lines.append(f"    bool construct{suffix}({params_text});")
            lines.append("")

        # Instance methods
        if instance_methods:
            lines.append("    // Discovered instance methods (Variant-compatible)")
            for mm in instance_methods:
                params_text = ", ".join(f"{t} {n}" for t, n in mm.exposed_param_list)
                const_q = " const" if mm.is_const else ""
                lines.append(f"    // native: {mm.method_name}")
                lines.append(f"    {mm.exposed_return.mapping.exposed_spelling} {mm.exposed_name}({params_text}){const_q};")
            lines.append("")

        # Static methods
        if static_methods:
            lines.append("    // Discovered static methods (Variant-compatible)")
            for mm in static_methods:
                params_text = ", ".join(f"{t} {n}" for t, n in mm.exposed_param_list)
                lines.append(f"    // native static: {mm.method_name}")
                lines.append(f"    static {mm.exposed_return.mapping.exposed_spelling} {mm.exposed_name}({params_text});")
            lines.append("")

        lines.append("};")
        lines.append("")
        lines.append("} // namespace godot")
        lines.append("")
        return "\n".join(lines)

    def _render_source(
        self,
        ci: ClassInfo,
        instance_methods: List[MappedMethod],
        static_methods: List[MappedMethod],
        mapped_ctors: List[Tuple[MethodInfo, List[Tuple[str, str]], List[str], List[str], List[str]]],
    ) -> str:
        """
        Generate the source content for a class with mapped method implementations.
        """
        lines: List[str] = []
        lines.append("/**")
        lines.append(" * This file is generated by the GDExtension binding generator (Variant-aware).")
        lines.append(" *")
        lines.append(f" * Native : {ci.qualified_name}")
        lines.append(f" * Wrapper: {ci.wrapper_name}")
        lines.append(" */")
        lines.append("")
        lines.append(f'#include "{ci.wrapper_name}.h"')
        if uses_opaque_handles(instance_methods, static_methods):
            lines.append('#include "opaque_handle_registry.h"')
        lines.append("")
        lines.append("using namespace godot;")
        lines.append("")
        lines.append(f"void {ci.wrapper_name}::_bind_methods() {{")
        lines.append("    // Helper methods")
        lines.append(f"    ClassDB::bind_method(D_METHOD(\"is_valid\"), &{ci.wrapper_name}::is_valid);")
        lines.append(f"    ClassDB::bind_method(D_METHOD(\"reset\"), &{ci.wrapper_name}::reset);")
        lines.append(f"    ClassDB::bind_method(D_METHOD(\"set_owns_impl\", \"own\"), &{ci.wrapper_name}::set_owns_impl);")
        lines.append(f"    ClassDB::bind_method(D_METHOD(\"get_owns_impl\"), &{ci.wrapper_name}::get_owns_impl);")
        lines.append("")
        # Constructor bindings
        for ctor, _, _, _, d_method_args in mapped_ctors:
            suffix = f"_{ctor.overload_index}" if ctor.overload_index is not None and ctor.overload_index > 0 else ""
            args_list = ", ".join([f"\"{a}\"" for a in d_method_args])
            if args_list:
                lines.append(f"    ClassDB::bind_method(D_METHOD(\"construct{suffix}\", {args_list}), &{ci.wrapper_name}::construct{suffix});")
            else:
                lines.append(f"    ClassDB::bind_method(D_METHOD(\"construct{suffix}\"), &{ci.wrapper_name}::construct{suffix});")
        if mapped_ctors:
            lines.append("")
        # Instance methods
        for mm in instance_methods:
            args_list = ", ".join([f"\"{sanitize_identifier(n)}\"" for _, n in mm.exposed_param_list])
            if args_list:
                lines.append(f"    ClassDB::bind_method(D_METHOD(\"{mm.exposed_name}\", {args_list}), &{ci.wrapper_name}::{mm.exposed_name});")
            else:
                lines.append(f"    ClassDB::bind_method(D_METHOD(\"{mm.exposed_name}\"), &{ci.wrapper_name}::{mm.exposed_name});")
        if instance_methods:
            lines.append("")
        # Static methods
        for mm in static_methods:
            args_list = ", ".join([f"\"{sanitize_identifier(n)}\"" for _, n in mm.exposed_param_list])
            if args_list:
                lines.append(f"    ClassDB::bind_static_method({ci.wrapper_name}::get_class_static(), D_METHOD(\"{mm.exposed_name}\", {args_list}), &{ci.wrapper_name}::{mm.exposed_name});")
            else:
                lines.append(f"    ClassDB::bind_static_method({ci.wrapper_name}::get_class_static(), D_METHOD(\"{mm.exposed_name}\"), &{ci.wrapper_name}::{mm.exposed_name});")
        lines.append("}")
        lines.append("")

        # Helper/ownership implementations
        abs_q = f"::{ci.qualified_name}" if not ci.qualified_name.startswith('::') else ci.qualified_name

        lines.append(f"bool {ci.wrapper_name}::is_valid() const {{")
        lines.append("    return impl != nullptr;")
        lines.append("}")
        lines.append("")
        lines.append(f"void {ci.wrapper_name}::reset() {{")
        lines.append("    if (impl && owns_impl) {")
        lines.append("        delete impl;")
        lines.append("    }")
        lines.append("    impl = nullptr;")
        lines.append("}")
        lines.append("")
        lines.append(f"void {ci.wrapper_name}::set_impl({abs_q}* p, bool p_take_ownership) {{")
        lines.append("    if (impl && owns_impl) {")
        lines.append("        delete impl;")
        lines.append("    }")
        lines.append("    impl = p;")
        lines.append("    owns_impl = p_take_ownership;")
        lines.append("}")
        lines.append("")
        lines.append(f"{abs_q}* {ci.wrapper_name}::get_impl() const {{")
        lines.append("    return impl;")
        lines.append("}")
        lines.append("")
        lines.append(f"void {ci.wrapper_name}::set_owns_impl(bool p_own) {{")
        lines.append("    owns_impl = p_own;")
        lines.append("}")
        lines.append("")
        lines.append(f"bool {ci.wrapper_name}::get_owns_impl() const {{")
        lines.append("    return owns_impl;")
        lines.append("}")
        lines.append("")

        # Constructor wrappers
        for ctor, params_sig, pre_lines, arg_exprs, _ in mapped_ctors:
            suffix = f"_{ctor.overload_index}" if ctor.overload_index is not None and ctor.overload_index > 0 else ""
            params_text = ", ".join(f"{t} {n}" for t, n in params_sig)
            lines.append(f"// {ctor.cpp_signature}")
            lines.append(f"bool {ci.wrapper_name}::construct{suffix}({params_text}) {{")
            lines.append("    if (impl && owns_impl) {")
            lines.append("        delete impl;")
            lines.append("    }")
            # Pre-call lines
            for pl in pre_lines:
                lines.append(f"    {pl}")
            call_args = ", ".join(arg_exprs)
            lines.append(f"    impl = new {abs_q}({call_args});")
            lines.append("    return impl != nullptr;")
            lines.append("}")
            lines.append("")

        # Instance methods
        for mm in instance_methods:
            params_text = ", ".join(f"{t} {n}" for t, n in mm.exposed_param_list)
            const_q = " const" if mm.is_const else ""
            lines.append(f"// native: {mm.method_name}")
            lines.append(f"{mm.exposed_return.mapping.exposed_spelling} {ci.wrapper_name}::{mm.exposed_name}({params_text}){const_q} {{")
            # Impl check
            if mm.has_return:
                default_expr = mm.default_return_expr or "{}"
                lines.append("    if (!impl) {")
                lines.append(f"        return {default_expr};")
                lines.append("    }")
            else:
                lines.append("    if (!impl) {")
                lines.append("        return;")
                lines.append("    }")
            # Pre-call conversions per parameter (recompute to replace placeholders safely)
            param_pre_lines, native_args = build_param_call_data(mm.exposed_params)
            for pl in param_pre_lines:
                lines.append(f"    {pl}")
            # Return pre-call lines if any
            for pl in mm.exposed_return.mapping.pre_call_lines:
                lines.append(f"    {pl}")
            call_expr = f"impl->{mm.method_name}({', '.join(native_args)})"
            # Return or void handling
            ret_map = mm.exposed_return.mapping
            if ret_map.kind == MappedKind.VOID:
                lines.append(f"    {call_expr};")
            else:
                if ret_map.from_native_expr:
                    if ret_map.from_native_expr.lstrip().startswith("{"):
                        # Full block form; substitute {expr}
                        block = ret_map.from_native_expr.replace("{expr}", call_expr)
                        for ln in block.splitlines():
                            lines.append(f"    {ln}")
                    else:
                        expr = ret_map.from_native_expr.replace("{expr}", call_expr)
                        lines.append(f"    return {expr};")
                else:
                    lines.append(f"    return {call_expr};")
            lines.append("}")
            lines.append("")

        # Static methods
        for mm in static_methods:
            params_text = ", ".join(f"{t} {n}" for t, n in mm.exposed_param_list)
            lines.append(f"// native static: {mm.method_name}")
            lines.append(f"{mm.exposed_return.mapping.exposed_spelling} {ci.wrapper_name}::{mm.exposed_name}({params_text}) {{")
            # Pre-call conversions per parameter
            param_pre_lines, native_args = build_param_call_data(mm.exposed_params)
            for pl in param_pre_lines:
                lines.append(f"    {pl}")
            for pl in mm.exposed_return.mapping.pre_call_lines:
                lines.append(f"    {pl}")
            call_expr = f"{abs_q}::{mm.method_name}({', '.join(native_args)})"
            ret_map = mm.exposed_return.mapping
            if ret_map.kind == MappedKind.VOID:
                lines.append(f"    {call_expr};")
            else:
                if ret_map.from_native_expr:
                    if ret_map.from_native_expr.lstrip().startswith("{"):
                        block = ret_map.from_native_expr.replace("{expr}", call_expr)
                        for ln in block.splitlines():
                            lines.append(f"    {ln}")
                    else:
                        expr = ret_map.from_native_expr.replace("{expr}", call_expr)
                        lines.append(f"    return {expr};")
                else:
                    lines.append(f"    return {call_expr};")
            lines.append("}")
            lines.append("")

        # Ctors/dtor
        lines.append(f"{ci.wrapper_name}::{ci.wrapper_name}() {{")
        lines.append("    // Do not auto-construct; use construct(...) wrappers to initialize impl.")
        lines.append("    impl = nullptr;")
        lines.append("    owns_impl = true;")
        lines.append("}")
        lines.append("")
        lines.append(f"{ci.wrapper_name}::~{ci.wrapper_name}() {{")
        lines.append("    if (impl && owns_impl) {")
        lines.append("        delete impl;")
        lines.append("    }")
        lines.append("    impl = nullptr;")
        lines.append("}")
        lines.append("")

        return "\n".join(lines)

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

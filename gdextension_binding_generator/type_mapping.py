#!/usr/bin/env python3
"""
Type mapping for Godot GDExtension wrappers.

This module analyzes parsed C++ types and methods (from `models.py`) and computes
how to expose them to Godot in a Variant-friendly way. It provides:

- A catalog of common mappings (C++ <-> Godot) for scalars and strings
- Bridging rules for wrapped classes (Ref<Wrapper> <-> native_impl*)
- A conservative "opaque handle" strategy for unknown pointer-returning methods
- Per-method mapping decisions, pre/post-call conversion snippets, and reasons
  for unsupported signatures

Typical usage (high level):

    from .models import ClassInfo, MethodInfo
    from .type_mapping import TypeMapper, MappingConfig

    mapper = TypeMapper.from_classes(
        classes, config=MappingConfig(prefix="OCC_")
    )

    for ci in classes:
        for mi in ci.methods:
            mapped = mapper.map_method(ci, mi)
            if not mapped.supported:
                # skip emitting this method; mapped.reason provides details
                continue
            # Use mapped.exposed_params / mapped.exposed_return to generate
            # wrapper signatures, D_METHOD() names, conversion snippets, etc.

Design notes:
- This module is independent from the Jinja templates. Emitters can opt-in to
  use these mappings to only generate supported methods and wire conversions.
- The conversion snippets are C++ fragments meant to be used by templates.
- For OpenCASCADE, strings are frequently `const char *` or OCCT-specific types.
  We conservatively support `const char*` and `std::string` -> `godot::String`.
  You can extend `MappingConfig.custom_rules` for OCCT-specific string types
  (e.g., TCollection_AsciiString) as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple
import re

from .models import ClassInfo, MethodInfo, MethodKind, CppType


# --------------------------
# Helpers
# --------------------------

def _strip_cv_and_class_kw(spelling: str) -> str:
    """
    Normalize a C++ type spelling to aid mapping heuristics:
    - Remove leading 'const', 'class', 'struct'
    - Collapse repeated spaces
    - Keep pointer/reference symbols for higher-level logic.
    """
    s = spelling or ""
    s = s.strip()
    # Remove some leading qualifiers/keywords conservatively (only at boundaries)
    for kw in ("const ", "class ", "struct ", "enum "):
        if s.startswith(kw):
            s = s[len(kw):].lstrip()
    # Normalize spacing around * and &
    s = s.replace(" &", "&").replace("& ", "&")
    s = s.replace(" *", "*").replace("* ", "*")
    # Collapse internal multiple spaces
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def _base_identifier(spelling: str) -> str:
    """
    Extract a best-effort "base identifier" for a type spelling, ignoring pointers/references and CV (const/volatile).
    Example:
      'const opencascade::TopAbs_Shape*&' -> 'opencascade::TopAbs_Shape'
      'const char*' -> 'char'
      'char const*' -> 'char'
      'std::basic_string<char>' -> 'std::basic_string<char>'
    """
    s = _strip_cv_and_class_kw(spelling)
    # Remove pointer/reference markers
    s = s.replace("&", "").replace("*", " ").strip()
    # Remove all const/volatile tokens regardless of position
    tokens = [tok for tok in s.split() if tok not in ("const", "volatile")]
    s = " ".join(tokens).strip()
    # Collapse spaces
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def _absolute_qualified_name(qualified: str) -> str:
    """
    Ensure a '::'-prefixed absolute qualified name for class-like identifiers.
    """
    if not qualified:
        return qualified
    return qualified if qualified.startswith("::") else f"::{qualified}"


def _is_integral(t: str) -> bool:
    ts = _strip_cv_and_class_kw(t)
    return ts in {
        "bool",
        "char",
        "signed char",
        "unsigned char",
        "short",
        "short int",
        "unsigned short",
        "unsigned short int",
        "int",
        "unsigned",
        "unsigned int",
        "long",
        "long int",
        "unsigned long",
        "unsigned long int",
        "long long",
        "long long int",
        "unsigned long long",
        "unsigned long long int",
        "size_t",
        "ptrdiff_t",
        "intptr_t",
        "uintptr_t",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
        "int64_t",
        "uint64_t",
    }


def _is_floating(t: str) -> bool:
    ts = _strip_cv_and_class_kw(t)
    return ts in {"float", "double", "long double"}


def _normalize_spaces(s: str) -> str:
    s = s.replace(" &", "&").replace("& ", "&")
    s = s.replace(" *", "*").replace("* ", "*")
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()

def _pointer_left_part(sp: str) -> str:
    idx = sp.find("*")
    return sp if idx < 0 else sp[:idx]

def _is_c_char_ptr(t: CppType) -> bool:
    """
    Detect any char* (const or non-const) using pointer flag and base identifier.
    """
    return bool(t.is_pointer and _base_identifier(t.spelling) == "char")

def _is_c_char_ptr_const(t: CppType) -> bool:
    """
    Detect char const* / const char* (pointer-to-const char), not const pointer to char.
    """
    if not _is_c_char_ptr(t):
        return False
    s = _normalize_spaces(_strip_cv_and_class_kw(t.spelling))
    left = _pointer_left_part(s)
    toks = left.split()
    # If 'const' appears before the first '*', it's pointer-to-const
    return "const" in toks


def _is_std_string(spelling: str) -> bool:
    s = _strip_cv_and_class_kw(spelling)
    return (
        s == "std::string"
        or s.startswith("std::__cxx11::basic_string<char")
        or s.startswith("std::basic_string<char")
    )

def _is_primitive_base_type(name: str) -> bool:
    """
    Return True if the base identifier corresponds to a Godot-supported primitive scalar.
    """
    n = _strip_cv_and_class_kw(name)
    return _is_integral(n) or _is_floating(n) or n in {"bool", "char", "wchar_t"}

def _is_pointer_to_primitive(t: CppType) -> bool:
    """
    True for pointers to primitive scalar types (e.g., double*, int*), which should not use opaque handles.
    """
    return bool(t.is_pointer and _is_primitive_base_type(_base_identifier(t.spelling)))

def _match_occt_handle_type(spelling: str) -> Optional[str]:
    """
    Detect OCCT handle smart pointers, e.g. 'opencascade::handle<T>' or 'Handle<T>'.
    Returns the canonical handle type string if matched, else None.
    """
    s = _strip_cv_and_class_kw(spelling or "")
    ss = re.sub(r"\s+", "", s)  # remove spaces
    # Common patterns (case-insensitive for 'handle')
    # e.g., 'opencascade::handle<Geom_Curve>' or 'Handle<Geom_Curve>'
    m = re.search(r"(?:^|::)(?:opencascade::)?handle<([^>]+)>", ss, re.IGNORECASE)
    if m:
        # Reconstruct handle type using the original (non-space-stripped) substring
        inner = m.group(1)
        # Try to rebuild a normalized handle type string
        # Prefer the lowercase opencascade::handle form for canonicalization
        return f"opencascade::handle<{inner}>"
    return None


# --------------------------
# Mapping model
# --------------------------

class MappedKind(Enum):
    PRIMITIVE = auto()
    STRING = auto()
    WRAPPED_CLASS = auto()
    OPAQUE_HANDLE = auto()
    VOID = auto()
    UNSUPPORTED = auto()


@dataclass
class MappedType:
    """
    Describes a single type mapping between 'native' and 'exposed' (Godot-facing).
    """
    native_spelling: str
    exposed_spelling: str
    kind: MappedKind
    # For call-site conversions:
    # - 'to_native_expr' is a C++ expression template that receives "{var}" placeholder
    #   (the name of the exposed variable). It must evaluate to the 'native' value passed
    #   to the underlying impl method.
    # - 'pre_call_lines' are statements to be declared before the call (e.g., char buffers).
    # - 'post_call_lines' are statements to run after the call (rarely needed).
    # - For return values, use 'from_native_expr' providing "{expr}" placeholder representing
    #   the native expression being converted to the exposed return.
    to_native_expr: Optional[str] = None
    from_native_expr: Optional[str] = None
    pre_call_lines: List[str] = field(default_factory=list)
    post_call_lines: List[str] = field(default_factory=list)
    # Default expression for exposed return value if the underlying impl is unavailable.
    # Example: "godot::String()", "0", "false", "godot::Ref<MyWrapper>()"
    default_exposed_value: Optional[str] = None
    # Optional notes for emitters
    notes: Optional[str] = None


@dataclass
class MappedParameter:
    name: str
    mapping: MappedType


@dataclass
class MappedReturn:
    mapping: MappedType


@dataclass
class MappedMethod:
    """
    Full mapping for a method, ready for code emission by templates.
    """
    cls_native_qname: str
    cls_wrapper_name: str
    method_name: str
    is_const: bool
    is_static: bool
    exposed_params: List[MappedParameter]
    exposed_return: MappedReturn
    supported: bool
    reason: Optional[str] = None
    # Exposed method name (after overload disambiguation) and overload index (if any)
    exposed_name: str = ""
    overload_index: Optional[int] = None
    # Aggregate pre/post-call statements (emit in source)
    pre_call_lines: List[str] = field(default_factory=list)
    post_call_lines: List[str] = field(default_factory=list)

    @property
    def exposed_param_list(self) -> List[Tuple[str, str]]:
        """
        Return a list of (type, name) for exposed wrapper signature.
        """
        return [(p.mapping.exposed_spelling, p.name) for p in self.exposed_params]

    @property
    def has_return(self) -> bool:
        return self.exposed_return.mapping.kind != MappedKind.VOID

    @property
    def default_return_expr(self) -> Optional[str]:
        """
        Default expression to return when impl is null (for instance methods).
        """
        return self.exposed_return.mapping.default_exposed_value


# --------------------------
# Configuration
# --------------------------

@dataclass
class MappingRule:
    """
    Rule for custom type mapping. Use either constant strings or simple templates.
    Placeholders:
      - In to_native_expr/from_native_expr use "{var}" or "{expr}" as explained in MappedType.
    """
    match: str  # normalized base type identifier to match (e.g., "::opencascade::Foo" or "TCollection_AsciiString")
    exposed_spelling: str
    kind: MappedKind
    to_native_expr: Optional[str] = None
    from_native_expr: Optional[str] = None
    pre_call_lines: List[str] = field(default_factory=list)
    post_call_lines: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class MappingConfig:
    """
    Settings and extensions for the type-mapper.
    """
    # Prefix applied to wrapper class names (must match GenerationContext.prefix)
    prefix: str = ""
    # Unknown pointer returns: either map to opaque handles (uint64_t) or mark unsupported.
    enable_opaque_handles: bool = True
    # For wrapped-classes parameters, prefer passing Ref<Wrapper> instead of raw native pointer.
    use_wrapped_param_bridge: bool = True
    # Enable std::string mapping to godot::String
    enable_std_string: bool = True
    # Custom rules keyed by normalized base identifier (no pointers/refs, no const)
    custom_rules: Dict[str, MappingRule] = field(default_factory=dict)

    def with_rule(self, rule: MappingRule) -> "MappingConfig":
        self.custom_rules[_base_identifier(rule.match)] = rule
        return self


# --------------------------
# Mapper
# --------------------------

class TypeMapper:
    """
    Orchestrates type mapping decisions using known wrapped classes and config.

    Build with:
      - from_classes(classes, config): uses parsed ClassInfo to populate known wrapped types
      - or TypeMapper(known_wrapped_map, config)
    """

    def __init__(self, known_wrapped: Dict[str, str], config: Optional[MappingConfig] = None) -> None:
        # Map absolute native qname -> wrapper name
        self.known_wrapped: Dict[str, str] = { _absolute_qualified_name(k): v for k, v in known_wrapped.items() }
        self.config = config or MappingConfig()

    @staticmethod
    def from_classes(classes: Sequence[ClassInfo], config: Optional[MappingConfig] = None) -> "TypeMapper":
        km: Dict[str, str] = {}
        for ci in classes:
            q = ci.qualified_name
            wrapper = ci.wrapper_name
            km[_absolute_qualified_name(q)] = wrapper
        return TypeMapper(km, config=config)

    # ---- Public API ----

    def map_method(self, cls: ClassInfo, m: MethodInfo) -> MappedMethod:
        """
        Compute an exposed signature and conversion snippets for a single method.
        If unsupported, supported=False with 'reason'.
        """
        is_static = m.kind == MethodKind.STATIC
        exposed_params: List[MappedParameter] = []
        pre_lines: List[str] = []
        post_lines: List[str] = []

        # Parameters
        for i, p in enumerate(m.parameters, start=1):
            pname = p.name or f"arg{i}"
            pmapping = self._map_parameter(p.cpp_type, pname)
            if pmapping.kind == MappedKind.UNSUPPORTED:
                return MappedMethod(
                    cls_native_qname=cls.qualified_name,
                    cls_wrapper_name=cls.wrapper_name,
                    method_name=m.name,
                    is_const=m.is_const,
                    is_static=is_static,
                    exposed_params=[],
                    exposed_return=MappedReturn(mapping=self._void_mapping()),
                    supported=False,
                    reason=f"Unsupported parameter '{pname}' type '{p.cpp_type.spelling}'",
                    exposed_name=m.exposed_name,
                    overload_index=m.overload_index,
                )
            exposed_params.append(MappedParameter(name=pname, mapping=pmapping))
            pre_lines.extend(pmapping.pre_call_lines)
            post_lines.extend(pmapping.post_call_lines)

        # Return type
        rmapping = self._map_return(m.return_type)
        if rmapping.kind == MappedKind.UNSUPPORTED:
            return MappedMethod(
                cls_native_qname=cls.qualified_name,
                cls_wrapper_name=cls.wrapper_name,
                method_name=m.name,
                is_const=m.is_const,
                is_static=is_static,
                exposed_params=exposed_params,
                exposed_return=MappedReturn(mapping=rmapping),
                supported=False,
                reason=f"Unsupported return type '{m.return_type.spelling}'",
                pre_call_lines=pre_lines,
                post_call_lines=post_lines,
                exposed_name=m.exposed_name,
                overload_index=m.overload_index,
            )

        return MappedMethod(
            cls_native_qname=cls.qualified_name,
            cls_wrapper_name=cls.wrapper_name,
            method_name=m.name,
            is_const=m.is_const,
            is_static=is_static,
            exposed_params=exposed_params,
            exposed_return=MappedReturn(mapping=rmapping),
            supported=True,
            reason=None,
            pre_call_lines=pre_lines + rmapping.pre_call_lines,
            post_call_lines=post_lines + rmapping.post_call_lines,
            exposed_name=m.exposed_name,
            overload_index=m.overload_index,
        )

    # ---- Primitive/string/wrapper/opaque rules ----

    def _void_mapping(self) -> MappedType:
        return MappedType(native_spelling="void", exposed_spelling="void", kind=MappedKind.VOID)

    def _primitive_mapping(self, t: CppType) -> MappedType:
        s = _strip_cv_and_class_kw(t.spelling)
        # Compute a reasonable default expression for exposed return values
        default_expr: Optional[str] = None
        if s == "bool":
            default_expr = "false"
        elif _is_integral(s):
            default_expr = "0"
        elif _is_floating(s):
            default_expr = "0.0"
        # Keep primitive as-is in wrapper signature (GDExtension supports these)
        return MappedType(native_spelling=s, exposed_spelling=s, kind=MappedKind.PRIMITIVE, default_exposed_value=default_expr)

    def _string_param_mapping_c_char_ptr(self, t: CppType, pname: str) -> MappedType:
        """
        Map exposed `godot::String` to native `const char*`. Requires temporaries to keep
        the C string alive for the call duration.
        """
        pre = [
            f"godot::CharString __{pname}_utf8 = {pname}.utf8();",
            f"const char* __{pname}_cstr = __{pname}_utf8.get_data();",
        ]
        return MappedType(
            native_spelling=_strip_cv_and_class_kw(t.spelling),
            exposed_spelling="godot::String",
            kind=MappedKind.STRING,
            to_native_expr=f"__{pname}_cstr",
            pre_call_lines=pre,
            notes="godot::String -> const char* conversion uses temporary CharString",
        )

    def _string_param_mapping_std_string(self, t: CppType, pname: str) -> MappedType:
        """
        Map exposed `godot::String` to native `std::string`. Use UTF-8 encoding.
        """
        pre = [
            f"godot::CharString __{pname}_utf8 = {pname}.utf8();",
            f"std::string __{pname}_std(__{pname}_utf8.get_data());",
        ]
        return MappedType(
            native_spelling=_strip_cv_and_class_kw(t.spelling),
            exposed_spelling="godot::String",
            kind=MappedKind.STRING,
            to_native_expr=f"__{pname}_std",
            pre_call_lines=pre,
            notes="godot::String -> std::string via UTF-8",
        )

    def _string_return_mapping_c_char_ptr(self, t: CppType) -> MappedType:
        """
        Map native `const char*` (or char*) to exposed `godot::String`.
        Assumes the returned pointer remains valid for the conversion expression.
        """
        return MappedType(
            native_spelling=_strip_cv_and_class_kw(t.spelling),
            exposed_spelling="godot::String",
            kind=MappedKind.STRING,
            from_native_expr='godot::String::utf8({expr})',
            default_exposed_value="godot::String()",
            notes="Assumes returned char* is valid and null-terminated",
        )

    def _string_return_mapping_std_string(self, t: CppType) -> MappedType:
        """
        Map native `std::string` to exposed `godot::String`.
        """
        return MappedType(
            native_spelling=_strip_cv_and_class_kw(t.spelling),
            exposed_spelling="godot::String",
            kind=MappedKind.STRING,
            from_native_expr='godot::String::utf8({expr}.c_str())',
            default_exposed_value="godot::String()",
            notes="std::string -> String via UTF-8",
        )

    def _wrapped_param_mapping_ptr(self, t: CppType, pname: str, native_base: str, wrapper_name: str) -> MappedType:
        """
        Expose `Ref<Wrapper>` while passing native pointer/reference to impl via __ocgd_get_impl().
        """
        pre: List[str] = []
        if t.is_reference:
            expr = f"*reinterpret_cast<{native_base}*>(__ocgd_get_impl({{var}}.ptr()))"
        else:
            expr = f"reinterpret_cast<{native_base}*>(__ocgd_get_impl({{var}}.ptr()))"
        return MappedType(
            native_spelling=_strip_cv_and_class_kw(t.spelling),
            exposed_spelling=f"godot::Ref<{wrapper_name}>",
            kind=MappedKind.WRAPPED_CLASS,
            to_native_expr=expr,
            pre_call_lines=pre,
            notes=f"Bridge Ref<{wrapper_name}> -> {native_base}{'&' if t.is_reference else '*'} via __ocgd_get_impl()",
        )

    def _wrapped_return_mapping_ptr(self, t: CppType, native_base: str, wrapper_name: str) -> MappedType:
        """
        Expose `Ref<Wrapper>` and wrap a returned `native*` with set_impl(ptr, false).
        Ownership is not taken by default to avoid double delete.
        """
        expr = f"__ocgd_make_from_impl(reinterpret_cast<void*>({{expr}}))"
        return MappedType(
            native_spelling=_strip_cv_and_class_kw(t.spelling),
            exposed_spelling=f"godot::Ref<{wrapper_name}>",
            kind=MappedKind.WRAPPED_CLASS,
            from_native_expr=expr,
            default_exposed_value=f"godot::Ref<{wrapper_name}>()",
            notes=f"Wrap returned {native_base}* into Ref<{wrapper_name}> via __ocgd_make_from_impl()",
        )

    def _opaque_handle_return_mapping_ptr(self, t: CppType) -> MappedType:
        """
        Expose an opaque handle (uint64_t) for unknown native pointers.
        This requires a runtime handle registry to be useful across API calls,
        which the templates/runtime must provide.

        Suggested registry API (not implemented here):
          template <class T> struct OpaqueHandleRegistry {
              static uint64_t put(T* ptr);
              static T* get(uint64_t handle);
              static void release(uint64_t handle);
          };
        """
        base_norm = _base_identifier(t.spelling)
        return MappedType(
            native_spelling=_strip_cv_and_class_kw(t.spelling),
            exposed_spelling="uint64_t",
            kind=MappedKind.OPAQUE_HANDLE,
            from_native_expr=f"OpaqueHandleRegistry<{base_norm}>::put({{expr}})",
            default_exposed_value="0",
            notes=f"Opaque pointer -> handle mapping for {base_norm}* (requires runtime registry)",
        )

    # ---- Parameter mapping ----

    def _map_parameter(self, t: CppType, pname: str) -> MappedType:
        # Void parameter: not expected
        if _strip_cv_and_class_kw(t.spelling) == "void":
            return MappedType(native_spelling="void", exposed_spelling="void", kind=MappedKind.UNSUPPORTED, notes="void parameter is invalid")

        # Primitive scalars are directly supported by Godot
        if _is_integral(t.spelling) or _is_floating(t.spelling):
            return self._primitive_mapping(t)

        # Godot String mapping only for const char* (pointer-to-const)
        if _is_c_char_ptr_const(t):
            return self._string_param_mapping_c_char_ptr(t, pname)
        # Writable char* buffers should not be mapped to String; expose as intptr_t
        if _is_c_char_ptr(t):
            return MappedType(
                native_spelling=_strip_cv_and_class_kw(t.spelling),
                exposed_spelling="intptr_t",
                kind=MappedKind.PRIMITIVE,
                to_native_expr="reinterpret_cast<char*>(static_cast<intptr_t>({var}))",
                default_exposed_value="0",
                notes="Map writable char* parameter to intptr_t for Variant API",
            )

        # std::string mapping if enabled
        if self.config.enable_std_string and _is_std_string(t.spelling):
            return self._string_param_mapping_std_string(t, pname)

        # OCCT handle<T> parameter mapping -> opaque handle (uint64_t)
        # Pass native as a dereferenced handle from the registry
        handle_type = _match_occt_handle_type(t.spelling)
        if handle_type:
            return MappedType(
                native_spelling=_strip_cv_and_class_kw(t.spelling),
                exposed_spelling="uint64_t",
                kind=MappedKind.OPAQUE_HANDLE,
                to_native_expr=f"*OpaqueHandleRegistry<{handle_type}>::get({{var}})",
                notes=f"OCCT handle parameter mapped to opaque handle (registry of {handle_type})",
            )

        # void* parameter mapping -> intptr_t exposed
        # Ensures typedef aliases like Standard_Address (void*) are exposed as intptr_t
        if t.is_pointer and _base_identifier(t.spelling) == "void":
            return MappedType(
                native_spelling=_strip_cv_and_class_kw(t.spelling),
                exposed_spelling="intptr_t",
                kind=MappedKind.PRIMITIVE,
                to_native_expr="reinterpret_cast<void*>(static_cast<intptr_t>({var}))",
                default_exposed_value="0",
                notes="Map void* parameter to intptr_t for Variant API"
            )

        # Custom rule by base identifier
        custom = self._match_custom_rule(t)
        if custom:
            return MappedType(
                native_spelling=_strip_cv_and_class_kw(t.spelling),
                exposed_spelling=custom.exposed_spelling,
                kind=custom.kind,
                to_native_expr=custom.to_native_expr,
                from_native_expr=custom.from_native_expr,
                pre_call_lines=list(custom.pre_call_lines),
                post_call_lines=list(custom.post_call_lines),
                notes=custom.notes,
            )

        # Wrapped class bridging (pointer/reference parameters)
        base = _base_identifier(t.spelling)
        base_abs = _absolute_qualified_name(base)
        if (t.is_pointer or t.is_reference) and base_abs in self.known_wrapped and self.config.use_wrapped_param_bridge:
            wrapper = self.known_wrapped[base_abs]
            return self._wrapped_param_mapping_ptr(t, pname, base_abs, wrapper)

        # Unknown pointer/reference param: map pointers to opaque handles if enabled; references unsupported
        if t.is_pointer:
            # Avoid opaque handles for pointers to primitive scalars (e.g., double*, int*): use intptr_t instead
            if _is_pointer_to_primitive(t):
                return MappedType(
                    native_spelling=_strip_cv_and_class_kw(t.spelling),
                    exposed_spelling="intptr_t",
                    kind=MappedKind.PRIMITIVE,
                    to_native_expr="reinterpret_cast<{}*>(static_cast<intptr_t>({{var}}))".format(_base_identifier(t.spelling)),
                    default_exposed_value="0",
                    notes="Pointer to primitive mapped to intptr_t (no opaque handle needed)",
                )
            if self.config.enable_opaque_handles:
                base_norm = _base_identifier(t.spelling)
                return MappedType(
                    native_spelling=_strip_cv_and_class_kw(t.spelling),
                    exposed_spelling="uint64_t",
                    kind=MappedKind.OPAQUE_HANDLE,
                    to_native_expr=f"OpaqueHandleRegistry<{base_norm}>::get({{var}})",
                    notes=f"Opaque handle -> pointer mapping for {base_norm}* (requires runtime registry)",
                )
            return MappedType(
                native_spelling=_strip_cv_and_class_kw(t.spelling),
                exposed_spelling="<unsupported>",
                kind=MappedKind.UNSUPPORTED,
                notes="Pointer parameter to unknown type is not supported",
            )
        if t.is_reference:
            return MappedType(
                native_spelling=_strip_cv_and_class_kw(t.spelling),
                exposed_spelling="<unsupported>",
                kind=MappedKind.UNSUPPORTED,
                notes="Reference parameter to unknown type is not supported",
            )

        # Fallback unsupported
        return MappedType(
            native_spelling=_strip_cv_and_class_kw(t.spelling),
            exposed_spelling="<unsupported>",
            kind=MappedKind.UNSUPPORTED,
            notes="Type not recognized as Variant-compatible or convertible",
        )

    # ---- Return mapping ----

    def _map_return(self, t: CppType) -> MappedType:
        # void
        if _strip_cv_and_class_kw(t.spelling) == "void":
            return self._void_mapping()

        # Primitive scalars: direct
        if _is_integral(t.spelling) or _is_floating(t.spelling):
            return self._primitive_mapping(t)

        # Strings
        if _is_c_char_ptr(t):
            return self._string_return_mapping_c_char_ptr(t)
        if self.config.enable_std_string and _is_std_string(t.spelling):
            return self._string_return_mapping_std_string(t)

        # OCCT handle<T> return mapping -> opaque handle (uint64_t)
        handle_type = _match_occt_handle_type(t.spelling)
        if handle_type:
            return MappedType(
                native_spelling=_strip_cv_and_class_kw(t.spelling),
                exposed_spelling="uint64_t",
                kind=MappedKind.OPAQUE_HANDLE,
                from_native_expr=f"OpaqueHandleRegistry<{handle_type}>::put(new {handle_type}({{expr}}))",
                default_exposed_value="0",
                notes=f"OCCT handle return mapped to opaque handle (registry of {handle_type})",
            )

        # void* return mapping -> intptr_t exposed
        # Ensures typedef aliases like Standard_Address (void*) are exposed as intptr_t
        if t.is_pointer and _base_identifier(t.spelling) == "void":
            return MappedType(
                native_spelling=_strip_cv_and_class_kw(t.spelling),
                exposed_spelling="intptr_t",
                kind=MappedKind.PRIMITIVE,
                from_native_expr="reinterpret_cast<intptr_t>({expr})",
                default_exposed_value="0",
                notes="Map void* return to intptr_t for Variant API"
            )

        # Custom rule
        custom = self._match_custom_rule(t)
        if custom:
            return MappedType(
                native_spelling=_strip_cv_and_class_kw(t.spelling),
                exposed_spelling=custom.exposed_spelling,
                kind=custom.kind,
                to_native_expr=custom.to_native_expr,
                from_native_expr=custom.from_native_expr,
                pre_call_lines=list(custom.pre_call_lines),
                post_call_lines=list(custom.post_call_lines),
                notes=custom.notes,
            )

        # Wrapped class pointer returns
        base = _base_identifier(t.spelling)
        base_abs = _absolute_qualified_name(base)
        if t.is_pointer and base_abs in self.known_wrapped:
            wrapper = self.known_wrapped[base_abs]
            return self._wrapped_return_mapping_ptr(t, base_abs, wrapper)

        # Unknown pointer return -> opaque handle if enabled (but avoid primitives)
        if t.is_pointer and self.config.enable_opaque_handles:
            if _is_pointer_to_primitive(t):
                return MappedType(
                    native_spelling=_strip_cv_and_class_kw(t.spelling),
                    exposed_spelling="intptr_t",
                    kind=MappedKind.PRIMITIVE,
                    from_native_expr="reinterpret_cast<intptr_t>({expr})",
                    default_exposed_value="0",
                    notes="Pointer to primitive mapped to intptr_t (no opaque handle needed)",
                )
            return self._opaque_handle_return_mapping_ptr(t)

        # Unknown reference or value class return: not supported (copy/ownership unclear)
        if t.is_reference or ("::" in base and not t.is_pointer):
            return MappedType(
                native_spelling=_strip_cv_and_class_kw(t.spelling),
                exposed_spelling="<unsupported>",
                kind=MappedKind.UNSUPPORTED,
                notes="Returning complex class by value/reference is not supported without explicit mapping",
            )

        # Fallback unsupported
        return MappedType(
            native_spelling=_strip_cv_and_class_kw(t.spelling),
            exposed_spelling="<unsupported>",
            kind=MappedKind.UNSUPPORTED,
            notes="Return type not recognized as Variant-compatible or convertible",
        )

    # ---- Custom rules matching ----

    def _match_custom_rule(self, t: CppType) -> Optional[MappingRule]:
        base = _base_identifier(t.spelling)
        return self.config.custom_rules.get(base)


# --------------------------
# Convenience utilities
# --------------------------

def supported_mapped_methods(mapper: TypeMapper, cls: ClassInfo) -> List[MappedMethod]:
    """
    Helper: get only supported mapped methods for a class, skipping constructors/destructors.
    """
    out: List[MappedMethod] = []
    for m in cls.methods:
        if m.kind in (MethodKind.CONSTRUCTOR, MethodKind.DESTRUCTOR):
            continue
        mm = mapper.map_method(cls, m)
        if mm.supported:
            out.append(mm)
    return out


def build_known_wrapped_map(classes: Sequence[ClassInfo]) -> Dict[str, str]:
    """
    Map native absolute qualified name -> wrapper name, for quick bridging lookups.
    """
    m: Dict[str, str] = {}
    for ci in classes:
        m[_absolute_qualified_name(ci.qualified_name)] = ci.wrapper_name
    return m


# --------------------------
# OCCT-friendly defaults
# --------------------------

def default_occt_mapper(classes: Sequence[ClassInfo], prefix: str) -> TypeMapper:
    """
    Construct a TypeMapper with defaults friendly to typical OpenCASCADE codebases:
    - Primitive scalars
    - const char* and std::string <-> godot::String
    - Wrapped class pointer bridging (Ref<Wrapper> <-> native*)
    - Opaque handle mapping for unknown pointer returns

    Extend with config.with_rule(...) for project-specific types (e.g., TCollection_AsciiString).
    """
    cfg = MappingConfig(
        prefix=prefix,
        enable_opaque_handles=True,
        use_wrapped_param_bridge=True,
        enable_std_string=True,
    )
    # Example: map OCCT Standard_CString (typedef const char*) explicitly (optional)
    cfg.with_rule(MappingRule(
        match="Standard_CString",
        exposed_spelling="godot::String",
        kind=MappedKind.STRING,
        to_native_expr="({var}.utf8().get_data())",
        from_native_expr="godot::String::utf8({expr})",
        notes="OCCT Standard_CString <-> String (UTF-8)",
    ))
    # Example: map OCCT handle patterns manually in templates if needed.

    return TypeMapper.from_classes(classes, config=cfg)


__all__ = [
    "MappedKind",
    "MappedType",
    "MappedParameter",
    "MappedReturn",
    "MappedMethod",
    "MappingRule",
    "MappingConfig",
    "TypeMapper",
    "supported_mapped_methods",
    "build_known_wrapped_map",
    "default_occt_mapper",
]

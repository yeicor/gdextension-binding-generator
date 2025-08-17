#!/usr/bin/env python3
"""
Clang-based parsing for collecting C++ class and method metadata.

This module traverses C/C++ headers using libclang and produces a rich,
template-friendly intermediate representation of classes and their methods.

Key features:
- Robust class discovery with include path filters and exclusion regex.
- Detailed method info (return type, parameters, const/static/virtual/pure).
- Basic base class extraction.
- Overload disambiguation to expose unique method names for binding.
- Deduplication via Clang USR.

Intended consumers:
- Emitters/templates to generate Godot 4.4 GDExtension wrapper sources.

Requirements:
- Python clang bindings (pip install clang)
- libclang available on the system or discoverable via standard mechanisms
"""

from __future__ import annotations

# removed unused import: os

import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from clang import cindex  # type: ignore
except Exception:  # pragma: no cover
    cindex = None  # Lazy error on use

from ..models import (
    BaseClassRef,
    ClassInfo,
    CppType,
    MethodInfo,
    MethodKind,
    ParameterInfo,
    sanitize_identifier,
)


# --------------------------
# libclang setup
# --------------------------

def ensure_libclang_loaded() -> None:
    """
    Ensure clang.cindex is importable. This function doesn't try to set a library path,
    but provides a single point to improve discovery in future.
    """
    if cindex is None:
        raise RuntimeError(
            "libclang (clang.cindex) is not available. Install clang Python bindings "
            "(e.g., pip install clang) and ensure libclang is discoverable."
        )


def _create_index():
    ensure_libclang_loaded()
    idx_cls = getattr(cindex, "Index", None)
    if idx_cls is None:
        raise RuntimeError("clang.cindex.Index is unavailable even though libclang is loaded")
    return idx_cls.create()


def parse_translation_unit(header: Path, clang_args: List[str]):
    """
    Parse a single header into a TranslationUnit with conservative options suitable
    for faster traversal and minimal preprocessing effects.
    """
    idx = _create_index()
    args = list(clang_args)
    # Silence warnings from system headers
    if not any(a.startswith("-W") for a in args):
        args.extend(["-Wno-everything"])
    tu = idx.parse(
        str(header),
        args=args,
        options=(
            getattr(getattr(cindex, "TranslationUnit", None), "PARSE_SKIP_FUNCTION_BODIES", 0)
            | getattr(getattr(cindex, "TranslationUnit", None), "PARSE_INCOMPLETE", 0)
            | getattr(getattr(cindex, "TranslationUnit", None), "PARSE_DETAILED_PROCESSING_RECORD", 0)
        ),
    )
    return tu


# --------------------------
# Helpers
# --------------------------

_SYSTEM_DIR_PREFIXES: Tuple[str, ...] = ("/usr/include", "/usr/local/include")

def _is_system_location(loc: Any) -> bool:
    try:
        f = getattr(loc, "file", None)
        if f is None:
            return True
        p = Path(str(f.name))
        return any(str(p).startswith(sd) for sd in _SYSTEM_DIR_PREFIXES)
    except Exception:
        return True


def _collect_namespace(cursor: Any) -> List[str]:
    """
    Collect namespaces for a declaration cursor by walking semantic parents.
    """
    ns: List[str] = []
    cur = getattr(cursor, "semantic_parent", None)
    while cur is not None and getattr(getattr(cur, "kind", None), "name", "") in (
        "NAMESPACE",
        "CLASS_DECL",
        "STRUCT_DECL",
        "CLASS_TEMPLATE",
    ):
        if getattr(getattr(cur, "kind", None), "name", "") == "NAMESPACE" and cur.spelling:
            ns.append(cur.spelling)
        cur = cur.semantic_parent
    ns.reverse()
    return ns


def _qualified_name_from_cursor(decl: Any) -> str:
    """
    Build a fully qualified name (including namespaces and enclosing classes) for a declaration cursor.
    """
    try:
        base = getattr(decl, "spelling", None) or ""
        if not base:
            return ""
        parts: List[str] = []
        parent = getattr(decl, "semantic_parent", None)
        while parent is not None and getattr(getattr(parent, "kind", None), "name", "") in (
            "NAMESPACE",
            "CLASS_DECL",
            "STRUCT_DECL",
            "CLASS_TEMPLATE",
            "ENUM_DECL",
        ):
            if getattr(parent, "spelling", None):
                parts.append(parent.spelling)
            parent = getattr(parent, "semantic_parent", None)
        parts.reverse()
        return "::".join(parts + [base]) if parts else base
    except Exception:
        return ""

def _fully_qualified_type_spelling(tp: Any) -> str:
    """
    Return a canonicalized type spelling that resolves typedefs (e.g., Standard_Address -> void*)
    and preserves pointer/reference/const qualifiers by leveraging Clang's canonical type.
    """
    try:
        get_canon = getattr(tp, "get_canonical", None)
        ct = get_canon() if callable(get_canon) else getattr(tp, "canonical", None)
        base = ct or tp
        spelling = getattr(base, "spelling", None) or "void"
        return spelling
    except Exception:
        base = tp
    spelling = getattr(base, "spelling", None) or "void"
    try:
        decl = getattr(base, "get_declaration", lambda: None)()
    except Exception:
        decl = None
    if decl is not None and getattr(decl, "spelling", None):
        spelling = _qualified_name_from_cursor(decl)
    return spelling

def _cpp_type_from_clang_type(tp: Any) -> CppType:
    """
    Convert a clang Type to a CppType with best effort. We primarily rely on
    the type spelling; detection of pointer/reference/const is delegated to
    the CppType.from_spelling heuristic. The spelling is normalized to a
    fully qualified form when possible.
    """
    spelling = _fully_qualified_type_spelling(tp)
    return CppType.from_spelling(spelling)


def _method_kind_from_cursor(node: Any) -> MethodKind:
    kind_name = getattr(getattr(node, "kind", None), "name", "")
    if kind_name == "CONSTRUCTOR":
        return MethodKind.CONSTRUCTOR
    if kind_name == "DESTRUCTOR":
        return MethodKind.DESTRUCTOR
    if kind_name == "CXX_METHOD":
        # Operators are also CXX_METHOD with spelling like "operator+"
        if getattr(node, "spelling", None) and str(node.spelling).startswith("operator"):
            return MethodKind.OPERATOR
        return MethodKind.INSTANCE
    return MethodKind.INSTANCE


def _should_consider_location(node: Any, include_filters: Optional[List[str]]) -> bool:
    """
    Use file path filtering to decide whether to consider a node for collection.
    If filters are provided, only accept nodes whose file path starts with any filter.
    Otherwise, exclude system header locations by default.
    """
    loc = getattr(node, "location", None)
    if loc is None or loc.file is None:
        return node.kind.name in ("NAMESPACE", "TRANSLATION_UNIT") # Allow namespaces and TUs without file
    fpath = str(Path(str(loc.file.name)).resolve())

    if include_filters:
        return any(fpath.startswith(str(Path(f).resolve())) for f in include_filters)
    return not _is_system_location(loc)


def _best_header_for_cursor(node: Any) -> str:
    try:
        return node.location.file.name  # type: ignore[attr-defined]
    except Exception:
        return ""


def _parse_parameters(node: Any) -> List[ParameterInfo]:
    """
    Extract parameters from a method/constructor cursor.
    """
    params: List[ParameterInfo] = []
    try:
        for i, p in enumerate(getattr(node, "get_arguments", lambda: [])(), start=1):
            name = p.spelling or f"arg{i}"
            cpp_t = _cpp_type_from_clang_type(p.type)
            params.append(ParameterInfo(name=name, cpp_type=cpp_t, default_value=None))
    except Exception as ex:
        # Fall back to empty parameter list on unexpected cursor behavior.
        logger.warning("Failed to parse parameters for node '%s'; emitting empty parameter list: %s", getattr(node, "spelling", None) or "<unnamed>", ex)
    return params


def _parse_method(
    node: Any,
    class_qname: str,
) -> Optional[MethodInfo]:
    """
    Convert a CXX_METHOD/CONSTRUCTOR/DESTRUCTOR or FUNCTION_TEMPLATE to MethodInfo.
    """
    # normalized node may differ for templates
    if getattr(getattr(node, "kind", None), "name", "") == "FUNCTION_TEMPLATE":
        logger.warning("Skipping function template '%s' in class '%s' (not supported by generator)", getattr(node, "spelling", None) or "<unnamed>", class_qname or "<global>")
        return None

    kind = _method_kind_from_cursor(node)
    name = node.spelling or ""
    usr = ""
    try:
        usr = node.get_usr() or ""
    except Exception:
        pass

    # Static detection only applies to CXX_METHOD
    is_static = False
    try:
        is_static = bool(getattr(node, "is_static_method")())
    except Exception:
        is_static = False

    # Const detection applies to instance methods
    is_const = False
    try:
        is_const = bool(getattr(node, "is_const_method")())
    except Exception:
        is_const = False

    is_virtual = False
    is_pure_virtual = False
    try:
        is_virtual = bool(getattr(node, "is_virtual_method")())
        is_pure_virtual = bool(getattr(node, "is_pure_virtual_method")())
    except Exception:
        pass

    # Return type: constructors/destructors are modeled as void for emission purposes.
    if kind in (MethodKind.CONSTRUCTOR, MethodKind.DESTRUCTOR):
        ret = CppType.from_spelling("void")
    else:
        try:
            ret = _cpp_type_from_clang_type(node.result_type)
        except Exception as ex:
            logger.warning("Failed to resolve return type for %s; defaulting to void: %s", f"{class_qname}::{name}" if class_qname else name, ex)
            ret = CppType.from_spelling("void")

    params = _parse_parameters(node)

    # Determine MethodKind.STATIC when possible
    concrete_kind = kind
    if kind == MethodKind.INSTANCE and is_static:
        concrete_kind = MethodKind.STATIC

    mi = MethodInfo(
        name=name,
        return_type=ret,
        parameters=params,
        is_const=is_const,
        is_virtual=is_virtual,
        is_pure_virtual=is_pure_virtual,
        is_explicit=False,  # Clang API for explicit is not exposed here reliably
        is_noexcept=False,  # Not easily retrievable from clang.cindex
        kind=concrete_kind,
        qualified_name=f"{class_qname}::{name}" if class_qname else name,
        usr=usr or "",
    )
    return mi


def _parse_base_specifiers(node: Any) -> List[BaseClassRef]:
    bases: List[BaseClassRef] = []
    for c in getattr(node, "get_children", lambda: [])():
        if getattr(getattr(c, "kind", None), "name", "") == "CXX_BASE_SPECIFIER":
            qn = ""
            try:
                if getattr(c, "type", None) and getattr(c.type, "spelling", None):
                    qn = _fully_qualified_type_spelling(c.type)
                elif getattr(c, "spelling", None):
                    qn = c.spelling
            except Exception as ex:
                logger.warning("Failed to parse base specifier for class node '%s': %s", getattr(node, "spelling", None) or "<unnamed>", ex)
            if not qn:
                continue
            # Access specifier (public/protected/private)
            access = "public"
            try:
                acc = getattr(c, "access_specifier", None)  # type: ignore[attr-defined]
                if acc and getattr(acc, "name", None):
                    access = acc.name.lower()
            except Exception as ex:
                logger.warning("Failed to determine base access specifier for '%s'; defaulting to 'public': %s", qn or "<unknown>", ex)
            # Virtual inheritance
            is_virtual = False
            try:
                is_virtual = bool(getattr(c, "is_virtual_base"))  # may not exist
            except Exception as ex:
                logger.warning("Failed to determine virtual inheritance for base '%s': %s", qn or "<unknown>", ex)
            bases.append(BaseClassRef(qualified_name=qn, is_virtual=is_virtual, access=access))
    return bases


def _collect_class_decls(
    tu: object,
    include_filters: Optional[List[str]],
    exclude_class_regex: Optional[object],
    prefix: str,
):
    """
    Traverse the TU and collect ClassInfo entries including methods.
    """
    class_map: Dict[str, ClassInfo] = {}

    def visit(node) -> None:
        try:
            # Respect file filters and avoid system locations unless overridden
            if not _should_consider_location(node, include_filters):
                # Warn when include filters are active and a class/struct is excluded
                if include_filters:
                    kind_name = getattr(getattr(node, "kind", None), "name", "")
                    if kind_name in ("CLASS_DECL", "STRUCT_DECL"):
                        try:
                            loc_file = node.location.file.name if node.location and node.location.file else "<unknown>"
                        except Exception:
                            loc_file = "<unknown>"
                        logger.warning("Excluding %s '%s' from %s due to include filters", kind_name, node.spelling or "<unnamed>", loc_file)
                return

            kind_name = getattr(getattr(node, "kind", None), "name", "")
            if kind_name in ("CLASS_DECL", "STRUCT_DECL"):
                if not node.is_definition():
                    # Skip forward declarations (note: logic preserved)
                    try:
                        loc_file = node.location.file.name if node.location and node.location.file else "<unknown>"
                    except Exception:
                        loc_file = "<unknown>"
                    logger.warning("Skipping forward declaration of '%s' at %s", node.spelling or "<unnamed>", loc_file)
                    return
                if not node.spelling:  # anonymous
                    logger.warning("Skipping anonymous class/struct declaration")
                    return

                # Skip non-public nested classes/structs (e.g., private nested helpers)
                try:
                    parent = getattr(node, "semantic_parent", None)
                    parent_kind = getattr(getattr(parent, "kind", None), "name", "")
                    is_nested = parent_kind in ("CLASS_DECL", "STRUCT_DECL", "CLASS_TEMPLATE")
                    acc = getattr(node, "access_specifier", None)  # may be None
                    acc_name = getattr(acc, "name", "NONE")
                    if is_nested and acc_name != "PUBLIC":
                        logger.info("Skipping non-public nested %s '%s' (access=%s)", kind_name.lower(), node.spelling or "<unnamed>", acc_name)
                        return
                except Exception as e:
                    logger.warning("Failed to determine access specifier for node '%s'; including it; error: %s", getattr(node, "spelling", None) or "<unnamed>", e)

                usr = ""
                try:
                    usr = node.get_usr() or ""
                except Exception:
                    usr = ""

                name = node.spelling or ""
                if not name:
                    # Skip nameless
                    pass
                else:
                    should_exclude = False
                    if exclude_class_regex:
                        try:
                            pat = getattr(exclude_class_regex, "pattern", None) or str(exclude_class_regex)
                            if re.search(pat, name):
                                should_exclude = True
                        except Exception as ex:
                            logger.warning("Error evaluating exclude_class_regex on '%s': %s", name, ex)
                    if should_exclude:
                        logger.warning("Excluding class '%s' due to exclude regex", name)
                    else:
                        # Determine qualified name and header
                        ns = _collect_namespace(node)
                        header_file = _best_header_for_cursor(node)
                        ci = ClassInfo(
                            name=name,
                            namespaces=ns,
                            header=header_file,
                            cursor_usr=usr,
                            wrapper_prefix=prefix,
                        )

                        # Parse base classes
                        ci.bases = _parse_base_specifiers(node)

                        # Parse methods under this class (walk only direct children for performance)
                        for c in getattr(node, "get_children", lambda: [])():
                            ck = getattr(getattr(c, "kind", None), "name", "")
                            if ck in ("CXX_METHOD", "CONSTRUCTOR", "DESTRUCTOR"):
                                mi = _parse_method(c, ci.qualified_name)
                                if mi:
                                    ci.methods.append(mi)

                        # Deduplicate by USR if available; otherwise by qualified name
                        key = usr or ci.qualified_name
                        if key not in class_map:
                            class_map[key] = ci
                        else:
                            # Merge methods if the class was encountered multiple times
                            existing = class_map[key]
                            existing.methods.extend(ci.methods)
                            # Prefer a concrete header path
                            if not existing.header and ci.header:
                                existing.header = ci.header

            # Recurse
            for c in node.get_children():
                visit(c)

        except Exception as ex:
            # Be tolerant of unexpected nodes; do not abort traversal
            logger.warning("Unexpected error while visiting AST node kind=%s: %s", getattr(getattr(node, "kind", None), "name", None), ex)

    root = getattr(tu, "cursor", None)
    if root is not None:
        visit(root)

    # Disambiguate overloaded methods per class
    for ci in class_map.values():
        _assign_overload_indices(ci)

    classes = sorted(class_map.values(), key=lambda c: (c.namespaces, c.name))
    return classes


def _assign_overload_indices(ci: ClassInfo) -> None:
    """
    Detect overloaded methods by name within instance/static groups, assign deterministic indices.
    The first overload keeps base name (index 0), subsequent overloads get suffixes (_1, _2, ...).
    """
    # Group by (kind_group, base_name)
    # kind_group distinguishes instance and static; we avoid mixing them in overload sets
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, m in enumerate(ci.methods):
        if m.kind not in (MethodKind.INSTANCE, MethodKind.STATIC, MethodKind.OPERATOR):
            continue
        # Operators also overload; treat "operator_plus" style exposure; leave as-is for now
        base_name = sanitize_identifier(m.name)
        kind_group = "static" if m.kind == MethodKind.STATIC else "instance"
        groups.setdefault((kind_group, base_name), []).append(i)

    for _, idxs in groups.items():
        if len(idxs) <= 1:
            # Single method: not overloaded, leave overload_index as None
            continue
        # Multiple overloads: assign indices deterministically by signature
        # Create stable ordering using the signature_key/hash
        idxs_sorted = sorted(idxs, key=lambda j: ci.methods[j].signature_key)
        for order, j in enumerate(idxs_sorted):
            ci.methods[j].overload_index = order


# --------------------------
# Public API
# --------------------------

def collect_classes_from_headers(
    headers: Iterable[Path],
    clang_args: List[str],
    include_filters: Optional[List[str]] = None,
    exclude_class_regex: Optional[object] = None,
    prefix: str = "",
    emit_diagnostics: bool = True,
) -> List[ClassInfo]:
    """
    Parse headers and return a list of ClassInfo with detailed method info.

    Parameters:
    - headers: List of header files to parse (files only; directories must be expanded by the caller).
    - clang_args: Command line arguments for clang (include paths, defines, -std, etc.).
    - include_filters: If provided, only classes whose definition file starts with one of these prefixes are included.
    - exclude_class_regex: Regular expression to exclude classes by name.
    - prefix: Wrapper class name prefix (e.g., "OCC_").
    - emit_diagnostics: Whether to print clang diagnostics to stderr.

    Returns:
    - Sorted list of ClassInfo instances.
    """
    ensure_libclang_loaded()

    filters = [str(Path(f).resolve()) for f in (include_filters or [])]
    class_map: Dict[str, ClassInfo] = {}

    for header in headers:
        tu = parse_translation_unit(header, clang_args)
        if emit_diagnostics:
            for diag in tu.diagnostics:
                logger.warning("[clang] %s", diag)

        classes = _collect_class_decls(
            tu=tu,
            include_filters=filters if filters else None,
            exclude_class_regex=exclude_class_regex,
            prefix=prefix,
        )
        for c in classes:
            key = c.cursor_usr or c.qualified_name
            if key in class_map:
                # Merge methods/bases carefully, avoiding duplicates by USR when present
                existing = class_map[key]
                existing.methods.extend(c.methods)
                existing.bases.extend(c.bases)
                if not existing.header and c.header:
                    existing.header = c.header
            else:
                class_map[key] = c

    # After merging across TUs, reassign overload indices for each class
    for ci in class_map.values():
        _assign_overload_indices(ci)

    return sorted(class_map.values(), key=lambda c: (c.namespaces, c.name))


__all__ = [
    "collect_classes_from_headers",
    "parse_translation_unit",
    "ensure_libclang_loaded",
]

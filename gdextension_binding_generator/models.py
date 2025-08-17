#!/usr/bin/env python3
"""
Data models for the GDExtension binding generator.

This module provides strongly-typed, serializable data structures to describe:
- C++ types (lightweight parsing of pointers/references/const)
- Function parameters
- Methods (instance/static/constructor/destructor)
- Classes (namespaces, headers, methods, etc.)
- Generation context (paths, flags, prefixes)

The models are designed to be consumed by:
- The parsing layer (to populate instances)
- The emitters/templates (Jinja2) to render wrapper source code
- Future transformation layers (e.g., type mapping, filtering, renaming)

The goal is to keep all binding-relevant metadata in one place and to offer
convenient helpers that make templates simpler and safer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from .utils import sanitize_identifier, stable_signature_hash, camel_to_snake

# --------------------------
# C++ Type model
# --------------------------

@dataclass(frozen=True)
class CppType:
    """
    Lightweight representation of a C++ type for emission and mapping purposes.

    This is intentionally simple — it keeps enough information to:
    - Generate readable signatures
    - Make basic decisions (pointer/reference/const)
    - Support downstream mapping (e.g., to Godot Variant-compatible types)

    For complex cases (templates, namespaces), keep the spelling string intact.
    """
    spelling: str
    is_const: bool = False
    is_pointer: bool = False
    is_reference: bool = False

    @staticmethod
    def from_spelling(spelling: str) -> CppType:
        """
        Parse a C++ type spelling heuristically into a CppType.
        Non-destructive: preserves the original spelling in `canonical` while
        extracting common modifiers.
        """
        is_const = "const" in spelling
        is_pointer = "*" in spelling
        is_reference = "&" in spelling
        return CppType(
            spelling=spelling,
            is_const=is_const,
            is_pointer=is_pointer,
            is_reference=is_reference,
        )

    def to_dict(self) -> Dict:
        return {
            "spelling": self.spelling,
            "is_const": self.is_const,
            "is_pointer": self.is_pointer,
            "is_reference": self.is_reference,
        }


# --------------------------
# Method/Parameter models
# --------------------------

class MethodKind(Enum):
    CONSTRUCTOR = auto()
    DESTRUCTOR = auto()
    INSTANCE = auto()
    STATIC = auto()
    OPERATOR = auto()

@dataclass
class ParameterInfo:
    name: str
    cpp_type: CppType
    default_value: Optional[str] = None
    is_variadic: bool = False

    @property
    def exposed_name(self) -> str:
        """
        Name as exposed to Godot (snake_case, sanitized).
        """
        return sanitize_identifier(self.name or "arg")

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "exposed_name": self.exposed_name,
            "cpp_type": self.cpp_type.to_dict(),
            "default_value": self.default_value,
            "is_variadic": self.is_variadic,
        }

@dataclass
class MethodInfo:
    """
    Description of a class method. Overloads are represented as separate instances.
    """
    name: str
    return_type: CppType
    parameters: List[ParameterInfo] = field(default_factory=list)
    is_const: bool = False
    is_virtual: bool = False
    is_pure_virtual: bool = False
    is_explicit: bool = False
    is_noexcept: bool = False
    kind: MethodKind = MethodKind.INSTANCE
    qualified_name: str = ""  # e.g., "ns::Class::method"
    usr: str = ""  # clang USR for deduplication if available

    # Overload resolution helpers
    overload_index: Optional[int] = None  # set by parser if needed
    _explicit_exposed_name: Optional[str] = None  # allow renaming strategies upstream

    @property
    def cpp_signature(self) -> str:
        """
        Human-friendly C++ signature string (without default values),
        used for diagnostics and for stable hashing.
        """
        params = ", ".join(p.cpp_type.spelling for p in self.parameters)
        const_q = " const" if self.is_const and self.kind == MethodKind.INSTANCE else ""
        qual = const_q
        return f"{self.name}({params}){qual} -> {self.return_type.spelling}"

    @property
    def signature_key(self) -> str:
        """
        A unique key suitable for hashing, insensitive to whitespace and qualifiers noise.
        """
        parts = [self.name, "|"]
        for p in self.parameters:
            parts.append(p.cpp_type.spelling)
            parts.append(",")
        parts.append("|C" if self.is_const else "|")
        parts.append(f"->{self.return_type.spelling}")
        return "".join(parts)

    @property
    def signature_hash(self) -> str:
        return stable_signature_hash(self.signature_key)

    @property
    def exposed_name(self) -> str:
        """
        Name used for Godot exposure and wrapper method, disambiguated if overloaded.
        Templates or emitters can use this safely for ClassDB::bind_method and method names.
        """
        base = sanitize_identifier(self._explicit_exposed_name or self.name)
        # For constructors/destructors we don't expose as callable methods by default.
        if self.kind in (MethodKind.CONSTRUCTOR, MethodKind.DESTRUCTOR):
            return base
        # If this method would collide by name (upstream can set overload_index), add suffix
        if self.overload_index is not None and self.overload_index > 0:
            return f"{base}_{self.overload_index}"
        # As a fallback, if an emitter wishes to always disambiguate, it can use signature_hash
        return base

    @property
    def d_method_args(self) -> List[str]:
        """
        Argument names for D_METHOD(...) binding (exposed/snake_case).
        """
        return [p.exposed_name for p in self.parameters]

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "exposed_name": self.exposed_name,
            "qualified_name": self.qualified_name or self.name,
            "usr": self.usr,
            "kind": self.kind.name,
            "is_const": self.is_const,
            "is_virtual": self.is_virtual,
            "is_pure_virtual": self.is_pure_virtual,
            "is_explicit": self.is_explicit,
            "is_noexcept": self.is_noexcept,
            "return_type": self.return_type.to_dict(),
            "parameters": [p.to_dict() for p in self.parameters],
            "cpp_signature": self.cpp_signature,
            "signature_hash": self.signature_hash,
            "d_method_args": self.d_method_args,
            "overload_index": self.overload_index,
        }


# --------------------------
# Class model
# --------------------------

@dataclass
class BaseClassRef:
    qualified_name: str
    is_virtual: bool = False
    access: str = "public"  # "public"/"protected"/"private"

    def to_dict(self) -> Dict:
        return {
            "qualified_name": self.qualified_name,
            "is_virtual": self.is_virtual,
            "access": self.access,
        }

@dataclass
class ClassInfo:
    name: str
    namespaces: List[str] = field(default_factory=list)
    header: str = ""
    cursor_usr: str = ""  # clang USR to deduplicate
    wrapper_prefix: str = ""  # e.g., "OCC_"
    bases: List[BaseClassRef] = field(default_factory=list)
    methods: List[MethodInfo] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        if self.namespaces:
            return "::".join(self.namespaces + [self.name])
        return self.name

    @property
    def wrapper_name(self) -> str:
        return f"{self.wrapper_prefix}{self.name}" if self.wrapper_prefix else self.name

    @property
    def include_header(self) -> str:
        """
        Return the best header include path for this class.
        Prefer the one discovered by libclang; otherwise, guess from the fully qualified name.
        """
        if self.header:
            return self.header
        guess = self.qualified_name.replace("::", "/") + ".h"
        return guess

    @property
    def constructors(self) -> List[MethodInfo]:
        return [m for m in self.methods if m.kind == MethodKind.CONSTRUCTOR]

    @property
    def destructors(self) -> List[MethodInfo]:
        return [m for m in self.methods if m.kind == MethodKind.DESTRUCTOR]

    @property
    def instance_methods(self) -> List[MethodInfo]:
        return [m for m in self.methods if m.kind == MethodKind.INSTANCE]

    @property
    def static_methods(self) -> List[MethodInfo]:
        return [m for m in self.methods if m.kind == MethodKind.STATIC]

    def to_dict(self) -> Dict:
        """
        Serialize to a template-friendly dictionary.
        """
        return {
            "name": self.name,
            "namespaces": list(self.namespaces),
            "qualified_name": self.qualified_name,
            "absolute_qualified_name": ("::" + self.qualified_name) if not self.qualified_name.startswith("::") else self.qualified_name,
            "header": self.include_header,
            "wrapper_name": self.wrapper_name,
            "bases": [b.to_dict() for b in self.bases],
            "methods": [m.to_dict() for m in self.methods],
            "constructors": [m.to_dict() for m in self.constructors],
            "destructors": [m.to_dict() for m in self.destructors],
            "instance_methods": [m.to_dict() for m in self.instance_methods],
            "static_methods": [m.to_dict() for m in self.static_methods],
        }


# --------------------------
# Generation context
# --------------------------

@dataclass
class GenerationContext:
    """
    Parameters for a single generation run.

    Paths are absolute. Emitters should rely on these rather than guessing.
    """
    output_dir: Path
    classes_dir: Path
    templates_dir: Optional[Path]
    godot_includes: List[str]
    prefix: str
    dry_run: bool = False

    def to_dict(self) -> Dict:
        return {
            "output_dir": str(self.output_dir),
            "classes_dir": str(self.classes_dir),
            "templates_dir": str(self.templates_dir) if self.templates_dir else None,
            "godot_includes": list(self.godot_includes),
            "prefix": self.prefix,
            "dry_run": self.dry_run,
        }


# --------------------------
# Template convenience helpers
# --------------------------

def build_template_context(prefix: str, classes: Iterable[ClassInfo]) -> Dict:
    """
    Produce a flattened context dict to pass to Jinja2 templates.
    Keeps the surface small and stable across templates.
    """
    return {
        "prefix": prefix,
        "classes": [c.to_dict() for c in classes],
    }


__all__ = [
    "CppType",
    "ParameterInfo",
    "MethodInfo",
    "MethodKind",
    "BaseClassRef",
    "ClassInfo",
    "GenerationContext",
    "camel_to_snake",
    "sanitize_identifier",
    "stable_signature_hash",
    "build_template_context",
]

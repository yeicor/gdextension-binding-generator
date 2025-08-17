#!/usr/bin/env python3
"""
Utilities for templating (Jinja2) and file I/O for the GDExtension binding generator.

This module provides:
- Robust, layered Jinja2 environment creation with user templates, package templates,
  and embedded templates as fallback.
- Helpful template filters and globals to make Jinja usage concise and safe.
- Production-grade file writing helpers (atomic writes, newline normalization, idempotency).

The goal is to keep the rest of the codebase clean and focused on parsing and emission logic.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, TextIO
import logging

logger = logging.getLogger(__name__)

def configure_logging(
    level: Optional[Union[int, str]] = None,
    *,
    to_file: Optional[Union[str, Path]] = None,
    fmt: Optional[str] = None,
    stream: Optional[TextIO] = None,
    propagate_package_loggers: bool = True,
) -> None:
    """
    Configure project-wide logging with consistent formatting and optional file output.

    Parameters:
    - level: int or name (e.g., 'INFO', 'DEBUG'). Defaults to INFO.
    - to_file: path to a log file; if provided, logs are also written there.
    - fmt: logging format string. Defaults to '%(levelname)s: %(message)s'.
    - stream: stream for console logs (defaults to sys.stderr).
    - propagate_package_loggers: whether the 'gdextension_binding_generator' logger propagates to root.
    """
    # Resolve level
    if level is None:
        resolved_level = logging.INFO
    elif isinstance(level, str):
        resolved_level = getattr(logging, level.upper(), logging.INFO)
    else:
        resolved_level = int(level)

    log_format = fmt or "%(levelname)s: %(message)s"
    stream = stream or sys.stderr

    # Reset root handlers for deterministic setup
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(resolved_level)

    handlers: List[logging.Handler] = []

    stream_handler = logging.StreamHandler(stream)
    stream_handler.setLevel(resolved_level)
    stream_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(stream_handler)

    if to_file:
        file_handler = logging.FileHandler(str(to_file), mode="w")
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    for h in handlers:
        root.addHandler(h)

    # Package logger configuration
    pkg_logger = logging.getLogger("gdextension_binding_generator")
    pkg_logger.setLevel(resolved_level)
    pkg_logger.propagate = propagate_package_loggers

    # Ensure our module-level logger picks up new config
    global logger
    logger = logging.getLogger(__name__)

try:
    from jinja2 import (
        ChoiceLoader,
        DictLoader,
        Environment,
        FileSystemLoader,
        PackageLoader,
        StrictUndefined,
        TemplateNotFound,
    )
except Exception as e:  # pragma: no cover
    Environment = None  # type: ignore
    ChoiceLoader = None  # type: ignore
    DictLoader = None  # type: ignore
    FileSystemLoader = None  # type: ignore
    PackageLoader = None  # type: ignore
    StrictUndefined = None  # type: ignore
    TemplateNotFound = Exception  # type: ignore

# Prefer importing helpers from models to keep logic consistent.
try:
    from .models import camel_to_snake, sanitize_identifier
except Exception:
    # Fallback minimal implementations (avoid circular deps if needed).
    def camel_to_snake(name: str) -> str:
        out: List[str] = []
        prev_lower = False
        prev_char = ""
        for ch in name:
            if ch.isupper() and (prev_lower or (prev_char and prev_char.isalpha() and prev_char != "_")):
                out.append("_")
            out.append(ch.lower())
            prev_lower = ch.islower()
            prev_char = ch
        return "".join(out)

    def sanitize_identifier(name: str) -> str:
        snake = camel_to_snake(name)
        reserved = {
            "class",
            "enum",
            "struct",
            "union",
            "template",
            "operator",
            "signal",
            "var",
            "const",
            "static",
            "void",
            "int",
            "float",
            "bool",
            "true",
            "false",
            "self",
            "super",
            "_init",
            "_ready",
            "_process",
            "_physics_process",
        }
        if snake in reserved:
            return f"{snake}_"
        return snake

# ----------------------------------------
# Jinja environment helpers
# ----------------------------------------

class TemplateRenderer:
    """
    A thin wrapper over a Jinja2 Environment with layered loaders and useful filters.
    - templates_dir: user-provided templates directory (highest precedence)
    - package templates: gdextension_binding_generator/templates (if installed)
    """

    def __init__(self, templates_dir: Optional[Path]) -> None:
        if (
            Environment is None
            or ChoiceLoader is None
            or FileSystemLoader is None
            or PackageLoader is None
            or StrictUndefined is None
        ):
            raise RuntimeError("jinja2 is not available or failed to import components. Install with: pip install Jinja2")

        loaders: List[Any] = []

        # 1) User-provided directory
        if templates_dir:
            p = Path(templates_dir)
            if p.is_dir():
                loaders.append(FileSystemLoader(str(p)))

        # 2) Package templates (installed alongside this module)
        # Prefer PackageLoader if available, else attempt a filesystem path.
        try:
            loaders.append(PackageLoader("gdextension_binding_generator", "templates"))
        except Exception:
            pkg_templates_fs = Path(__file__).parent / "templates"
            if pkg_templates_fs.is_dir():
                loaders.append(FileSystemLoader(str(pkg_templates_fs)))

        self.env = Environment(
            loader=ChoiceLoader(loaders),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )

        self._register_filters()
        self._register_globals()

    # ---- Filters and globals registration ----

    def _register_filters(self) -> None:
        self.env.filters["to_snake"] = camel_to_snake
        self.env.filters["sanitize"] = sanitize_identifier
        self.env.filters["cpp_param_list"] = _filter_cpp_param_list
        self.env.filters["d_method_args"] = _filter_d_method_args
        self.env.filters["bind_method_line"] = _filter_bind_method_line

    def _register_globals(self) -> None:
        self.env.globals["hasattr"] = hasattr
        self.env.globals["getattr"] = getattr
        self.env.globals["len"] = len

    # ---- Rendering ----

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound as e:
            raise RuntimeError(f"Template not found: {template_name}") from e
        return template.render(**context)


# ----------------------------------------
# Template filter implementations
# ----------------------------------------

def _as_method_dict(m: Any) -> Dict[str, Any]:
    """
    Normalize either a dataclass-like object or a dict into a dict with expected keys.
    """
    if isinstance(m, dict):
        return m
    # Try attribute access
    try:
        return {
            "name": getattr(m, "name"),
            "exposed_name": getattr(m, "exposed_name", getattr(m, "name")),
            "parameters": getattr(m, "parameters", []),
            "is_const": getattr(m, "is_const", False),
            "return_type": getattr(m, "return_type", {"canonical": "void"}),
            "d_method_args": getattr(m, "d_method_args", []),
        }
    except Exception:
        raise TypeError("Unsupported method value for template filters; expected dict or object with attributes")

def _as_param_list(plist: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(plist, dict):
        # single param dict
        plist = [plist]
    for p in plist or []:
        if isinstance(p, dict):
            out.append(p)
            continue
        # Dataclass-like param
        try:
            out.append(
                {
                    "name": getattr(p, "name", None),
                    "cpp_type": getattr(p, "cpp_type", {"canonical": "void"}),
                }
            )
        except Exception:
            raise TypeError("Unsupported parameter value; expected dict or object with attributes")
    return out

def _filter_cpp_param_list(method_like: Any) -> str:
    """
    Render a C++ parameter list from a method-like object/dict:
    'Type1 a, const T& b'
    """
    m = _as_method_dict(method_like)
    params = _as_param_list(m.get("parameters"))
    parts: List[str] = []
    for i, p in enumerate(params, start=1):
        ptype = p.get("cpp_type", {})
        tcanon = ptype.get("canonical", "void")
        pname = p.get("name") or f"arg{i}"
        parts.append(f"{tcanon} {pname}")
    return ", ".join(parts)

def _filter_d_method_args(method_like: Any) -> List[str]:
    """
    Return the list of argument names to feed D_METHOD(...)
    """
    m = _as_method_dict(method_like)
    names = []
    dnames = m.get("d_method_args")
    if isinstance(dnames, list) and dnames:
        return list(dnames)
    # Fallback: derive from parameters
    params = _as_param_list(m.get("parameters"))
    for i, p in enumerate(params, start=1):
        pname = p.get("name") or f"arg{i}"
        names.append(sanitize_identifier(pname))
    return names

def _filter_bind_method_line(method_like: Any, wrapper_name: str, is_static: bool = False) -> str:
    """
    Produce a full C++ bind method line for use in templates.
    Example (instance):
      ClassDB::bind_method(D_METHOD("foo", "a", "b"), &Wrapper::foo);
    Example (static):
      ClassDB::bind_static_method(Wrapper::get_class_static(), D_METHOD("bar"), &Wrapper::bar);
    """
    m = _as_method_dict(method_like)
    exposed = sanitize_identifier(m.get("exposed_name") or m.get("name", ""))
    args_list = _filter_d_method_args(m)
    d_method = ', '.join([f"\"{exposed}\""] + [f"\"{a}\"" for a in args_list])
    if is_static:
        return f'ClassDB::bind_static_method({wrapper_name}::get_class_static(), D_METHOD({d_method}), &{wrapper_name}::{exposed});'
    return f'ClassDB::bind_method(D_METHOD({d_method}), &{wrapper_name}::{exposed});'


# ----------------------------------------
# File I/O helpers
# ----------------------------------------

def ensure_dir(p: Path) -> None:
    """
    Ensure directory exists (mkdir -p).
    """
    Path(p).mkdir(parents=True, exist_ok=True)

def normalize_newlines(text: str) -> str:
    """
    Normalize to Unix newlines for reproducible diffs and consistent build environments.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")

def _read_text_if_exists(path: Path, encoding: str = "utf-8") -> Optional[str]:
    try:
        with open(path, "r", encoding=encoding, newline="") as f:
            return f.read()
    except FileNotFoundError:
        return None

def atomic_write_text(
    path: Path,
    content: str,
    encoding: str = "utf-8",
    make_parents: bool = True,
    mode: Optional[int] = 0o644,
    log: bool = True,
    only_if_changed: bool = True,
) -> bool:
    """
    Write text atomically to the given path:
    - Optionally avoid writing if the content is unchanged.
    - Write to a temp file in the same directory and os.replace to final path.
    - Set POSIX file mode if provided.

    Returns True if a write occurred, False if skipped due to idempotency.
    """
    content = normalize_newlines(content)
    if make_parents:
        ensure_dir(path.parent)

    # Skip write if unchanged
    if only_if_changed:
        old = _read_text_if_exists(path, encoding=encoding)
        if old is not None and normalize_newlines(old) == content:
            if log:
                logger.debug(f"[skip] {path} (unchanged)")
            return False

    dir_fd = None
    tmp_path = None
    try:
        # Create tmp file in same directory for atomic replace
        fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as f:
            f.write(content)
        if mode is not None:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
        if log:
            logger.info(f"[write] {path}")
        return True
    finally:
        # Cleanup temp if exception occurred before replace
        if tmp_path and os.path.exists(tmp_path) and not os.path.samefile(tmp_path, path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def write_text(
    path: Path,
    content: str,
    encoding: str = "utf-8",
    dry_run: bool = False,
    log: bool = True,
) -> None:
    """
    Convenience wrapper over atomic_write_text with optional dry-run support.
    """
    if dry_run:
        if log:
            logger.info(f"[dry-run] write {path}")
        return
    atomic_write_text(path, content, encoding=encoding, log=log)


__all__ = [
    "TemplateRenderer",
    "configure_logging",
    "ensure_dir",
    "normalize_newlines",
    "atomic_write_text",
    "write_text",
]

# GDExtension Binding Generator (Godot 4.4)

This repository contains a modular, maintainable, and fully automated binding generator to scaffold Godot 4.4 GDExtension wrappers for large C/C++ libraries (e.g., OpenCASCADE). It discovers classes and methods from your headers via libclang and emits:

- generated/register_types.h
- generated/register_types.cpp
- generated/classes/<WrapperName>.h
- generated/classes/<WrapperName>.cpp

The generated wrappers are minimal skeletons you can grow incrementally. They compile cleanly with godot-cpp and register as RefCounted classes in Godot, ready for your custom API.

---

## Why this approach

- Automated: Point the generator at your headers to produce valid C++ sources.
- General: Works with most C++ libraries (no special build system or annotations required).
- Maintainable: Modular architecture (models, parsing, emitters, utils) + Jinja2 templating.
- Production-ready: Layered templates (user dir → package templates → embedded defaults), atomic writes, optional JSON manifest.

---

## Quick start

1) Prerequisites
- Python 3.8+
- Clang and libclang (system package; ensure libclang is discoverable by the Python bindings)
- Python packages: clang, Jinja2
- A Godot 4.4-compatible godot-cpp in your extension project (to build the generated code later)

Install Python packages:
- pip install clang Jinja2

If the Python clang bindings cannot locate libclang automatically, set an environment variable that your platform recognizes (for example, many environments honor something like LIBCLANG_PATH pointing to the directory containing libclang.*). Consult your OS or LLVM/Clang distribution documentation.

2) Directory layout
- gdextension_binding_generator/generate_bindings.py (entrypoint)
- gdextension_binding_generator/{models.py, parsing/, emitters/, templates/} (modular architecture)
- templates/ (optional; user overrides)
- generated/ (output goes here by default)

3) Run the generator
- Basic example (dry run):
  - python -m gdextension_binding_generator.generate_bindings --prefix OCC_ --headers /path/to/headers --clang-args "-I/path/to/headers -std=c++17" --dry-run
- Actual generation:
  - python -m gdextension_binding_generator.generate_bindings --prefix OCC_ --headers /path/to/headers --clang-args "-I/path/to/headers -std=c++17"

Notes:
- You can pass multiple --headers arguments (files or directories). Directories are scanned recursively for .h/.hpp/.hh/.hxx files.
- Use --include-filter to constrain which headers are considered by absolute path prefix.
- Use --exclude-regex to skip classes by name.
- Use --prefix to prepend a string to generated wrapper classes and init symbols (e.g., OCC_).
- If you provide --templates-dir, those templates take precedence over package templates and embedded defaults.

4) Generated files
- generated/register_types.h
- generated/register_types.cpp
- generated/classes/<WrapperName>.h
- generated/classes/<WrapperName>.cpp

The register_types files expose:
- <prefix>initialize_module
- <prefix>uninitialize_module
- An entry point function named <prefix>library_init

5) Wire into your GDExtension project
- Add the generated files to your build (e.g., via SCons).
- Ensure your .gdextension file’s [configuration] entry_symbol matches the generated one:
  - entry_symbol = "<prefix>library_init"
- Ensure the godot-cpp include and link paths are correct in your build configuration.

6) Open Godot and test
- After compiling your extension, launch your Godot project and confirm your wrapper classes appear and can be instantiated (they derive from RefCounted).
- The wrappers declare and bind discovered methods automatically. Implement the bodies to call into your native library as needed; is_valid() is provided by default.

---

## Command-line reference

- --prefix: Prefix applied to each generated wrapper class and to the init symbols (<prefix>initialize_module, <prefix>library_init).
- --headers: Repeatable. Files or directories of headers to parse.
- --clang-args: Extra args for libclang (e.g., -Iinclude -DDEFINE=1 -std=c++17).
- --include-filter: Repeatable. Only include classes whose definition path starts with any of these prefixes.
- --exclude-regex: Regex to exclude classes by name.
- --output-dir: Root output directory (default: generated).
- --templates-dir: Optional path to a directory containing template files. If omitted, package templates and embedded defaults are used.
- --godot-include: Repeatable. Paths to godot-cpp includes (metadata only; not used by the generator logic).
- --no-manifest: Do not emit a JSON manifest of the discovered classes/methods.
- --dry-run: Perform discovery and report classes and methods without writing files.

---

## What gets generated

- A RefCounted wrapper class per discovered C++ class. The header includes the external library’s header (as discovered by libclang). The class contains:
  - A raw impl pointer (adjust to your library’s ownership model)
  - A static _bind_methods() stub
  - A simple is_valid() method
  - Declarations for discovered instance and static methods with Godot-friendly names
- The .cpp implements:
  - ClassDB::bind_method lines for instance methods and bind_static_method for static ones
  - Method bodies as stubs that you can wire to the underlying native calls
- register_types that:
  - Registers each wrapper class with GDREGISTER_RUNTIME_CLASS
  - Exposes <prefix>initialize_module and <prefix>uninitialize_module
  - Defines the <prefix>library_init entry point

This baseline is designed for correctness and simplicity. Expand it to fit your project’s needs.

---

## Customizing and extending

Typical next steps:
- Add construction patterns and factory helpers for impl (e.g., from handles or factory functions in your library).
- Bind methods, properties, and signals:
  - Method discovery is on by default (return type, parameters, constness, staticness, overloads). Overloads are disambiguated with stable suffixes when necessary.
  - Map C++ types to Godot Variant-compatible types as you evolve the generator.
  - Extend templates to customize naming, filtering, or to emit ADD_PROPERTY and ADD_SIGNAL.
- Manage lifetime and ownership explicitly:
  - Replace the raw pointer with smart pointers or external references as needed.
  - Consider disabling the default destructor or making ownership opt-in.

Template strategy:
- Copy the embedded templates into templates/ and adjust:
  - register_types.h.j2
  - register_types.cpp.j2
  - class_header.h.j2
  - class_source.cpp.j2
- Add new templates for more advanced generation (e.g., per-method emission) and invoke them from the generator as you expand the pipeline.

---

## Tips for OpenCASCADE

- Include OpenCASCADE headers with appropriate -I flags in --clang-args.
- Use --include-filter to limit parsing to the modules you need first (e.g., the TKG2d/TKG3d/ TKGeomBase ranges).
- Use --exclude-regex for internal or template-heavy types that you do not want to wrap initially.
- Start with a narrow slice, verify compilation, then expand.

---

## Troubleshooting

- Clang diagnostics:
  - The generator prints libclang diagnostics to stderr. If it reports missing headers, update --clang-args with the required -I paths and definitions.
- libclang not found:
  - Ensure the Python clang bindings can locate the libclang shared library. Setting an environment variable pointing to the directory containing libclang.* is a common solution, depending on your platform.
- No classes discovered:
  - Verify your --headers paths and --include-filter prefixes.
  - Confirm your C++ standard and macro definitions in --clang-args match the headers’ requirements.

---

## FAQ

- Is this tied to OpenCASCADE?
  - No. It targets any C/C++ codebase that libclang can parse.
- Why not parse with a regex or simpler parser?
  - Robust C++ parsing requires a real parser. libclang provides stable AST introspection with preprocessing.
- Why RefCounted as the base wrapper?
  - It’s a safe, editor-friendly base in Godot. You can switch to Object or more specific types as appropriate.

---

## License

You can use this generator as you see fit in your projects. If you share improvements, consider contributing them back to help others.
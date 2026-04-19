# LensDiff Build Notes

`LensDiff` uses a standalone single-plugin layout:

- local OFX SDK under `source/ofx-sdk`
- staged bundle under `source/bundle/LensDiff.ofx.bundle`
- Windows helper wrappers `configure_vc.bat` and `build_vc.bat`
- Linux preset-based staging under `source/bundle/LensDiff.ofx.bundle/Contents/Linux-x86-64`

## Windows

From `source`:

```bat
configure_vc.bat
build_vc.bat
```

The wrappers expect a Visual Studio 2026 installation with CMake and Ninja.

## Linux

From `source`:

```bash
cmake --preset linux-ninja-release
cmake --build --preset linux-build-release
```

The staged Linux plugin lands at:

```text
bundle/LensDiff.ofx.bundle/Contents/Linux-x86-64/LensDiff.ofx
```

Linux GPU deployment expects a system NVIDIA driver and CUDA runtime on GPU hosts. The portable bundle does not ship NVIDIA CUDA user-space libraries, and the CPU reference path remains the safe fallback when CUDA is unavailable.

For the Linux custom-aperture import button, the host machine should provide either `zenity` or `kdialog`.

## Presets

`CMakePresets.json` includes:

- `vs2026-x64`
- `linux-ninja-release`
- `linux-ninja-debug`
- `macos-ninja-release`

## Current Backend Status

- CPU reference path is active and remains the correctness anchor.
- CUDA is active on supported non-macOS builds and is the intended production GPU path on Linux and Windows.
- Metal is active on macOS builds.
- Linux now has a supported preset-based build path and a portable-bundle CI workflow, but real host validation in Resolve Linux is still the release gate.

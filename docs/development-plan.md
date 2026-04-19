# LensDiff Development Plan

## Purpose And Design Goals
- Build `LensDiff` as an OFX diffraction effect whose ground truth is a pupil-derived PSF pipeline rather than heuristic glow logic.
- Keep the system layered: OFX host glue, render-state translation, PSF and spectrum core, and backend execution.
- Treat `Selection` as a policy layer that decides what image energy enters the optical model. It is not part of the optics.
- Preserve a CPU reference path as the first correctness anchor, but do not treat it as a practical shipping render path for this effect; production priority belongs to CUDA and Metal.
- Keep LensDiff aligned to the incoherent diffraction-imaging branch described by Goodman and Rafael de la Fuente: pupil-derived PSF or OTF acting on selected highlight irradiance. Full coherent angular-spectrum propagation remains a separate experimental branch, not part of the main highlight-diffraction product path.
- Keep this document current with every architectural, mathematical, validation, or stage change.

## Current Stage
- Milestone: `v2`
- Status: `in_progress`
- Focus:
  - extend the shared spectral-bank path from the existing 5-bin mode to an explicit 9-bin mode
  - keep CPU, CUDA, and Metal on the same spectral mapping and wavelength-bank contract
  - preserve the existing `Mono`, `Tristimulus`, and `Spectral 5` looks while adding a denser `Spectral 9` option
  - treat custom aperture import as already landed and begin the next `v2` optics work with an opt-in non-flat phase scaffold
  - keep experimental coherent or near-field propagation out of the main product path until the incoherent imaging branch stays production-stable
- Exit Criteria:
  - CPU, CUDA, and Metal all accept the same four spectral modes: `Mono`, `Tristimulus`, `Spectral 5`, and `Spectral 9`
  - `Spectral 9` uses a shared 9-wavelength bank without regressing the existing 3-bin or 5-bin modes
  - Resolve validation confirms CPU/CUDA parity for `Spectral 9` in both debug and final views
  - the denser spectral bank stays luminance-disciplined under `Natural`, `Cyan-Magenta`, and `Warm-Cool`
  - the plugin-local roadmap now carries an opt-in non-flat phase path without regressing flat-phase behavior
- Immediate Next Steps:
  - validate the new `Spectral 9` mode on deterministic synthetic plates and Resolve footage
  - confirm CPU/CUDA parity for the 9-bin path, especially in debug PSF and final composite views
  - verify that `Spectral 5` remains visually stable after widening the shared spectral-bank limit
  - validate the first non-flat phase controls against known defocus, astigmatism, coma, and spherical-aberration expectations
  - decide whether the next `v2` code change should deepen the phase model or return to 9-bin tuning
  - keep Metal parity checks focused and structural until broader macOS testing is available

## Stable Architectural Principles
- The PSF engine is the truth. Creative controls may reshape PSF-derived structure, spectral mapping, and composite policy, but may not bypass the model with extra additive glow terms.
- All composite behavior must remain derivable from this sequence:
  - `selected = mask * src`
  - `retained = corePreserve * selected`
  - `redistributedInput = (1 - corePreserve) * selected`
  - `base = src - coreCompensation * redistributedInput`
  - `effect = convolve(redistributedInput, psfBank)`
  - `out = base + effectGain * effect`
- `Selection` is a policy layer, not part of the optical model.
- `anisotropyEmphasis` preserves exact unit PSF energy after reshaping. It does not preserve the core integral.
- `corePSF` and `structurePSF` define the primary artistic redistribution between isotropic and directional energy.
- Finite PSF support is a numerical boundary, not an optical shape control; resampled kernels may be support-tapered near their maximum radius to prevent square crop artefacts under aggressive settings.
- Raw PSF synthesis may use a larger internal FFT grid than the user-facing pupil raster so the exported kernel bank stays stable under aggressive scales and large support radii.
- Final composite may automatically reduce local subtraction to enforce a minimum selected-core floor; this safety lives in composite policy and does not alter PSF synthesis.
- `spectrumForce=0` must equal exactly `Natural`.
- `spectrumForce=1` must equal exactly the selected style map.
- With `chromaticAffectsLuma=off`, spectral styling may change hue and chroma but must not materially change effect luminance.

## Implementation Roadmap
- `v1A`
  - scaffold the plugin tree from the Chromaspace build skeleton
  - implement CPU pupil synthesis, FFT-derived PSF generation, highlight selection, convolution, and debug views
  - expose policy, optics, spectrum architecture, and composite controls
  - wire ROI expansion, identity handling, and cache seams
- `v1B`
  - add the tristimulus spectral path on the same PSF-bank architecture
  - implement natural, cyan-magenta, and warm-cool spectrum mappings
  - validate luminance-normalized spectral styling and chromatic luma coupling
- `v2`
  - extend the spectral-bank path beyond the existing 5-bin mode and validate the denser bank in-host
  - keep the shipped custom aperture import path and deepen non-flat phase support from the new opt-in scaffold
  - evaluate an explicit experimental coherent-propagation branch only after the main incoherent imaging path is production-stable
  - evaluate near-field or propagation experiments behind explicit opt-in controls

## Current Scope
- New plugin root: `OFX-Workshop/LensDiff`
- Canonical document: `LensDiff/docs/development-plan.md`
- Synthetic validation workflow:
  - `tools/generate_synthetic_sources.py`
  - `tools/synthetic_requirements.txt`
  - `docs/synthetic-sources.md`
  - generated output under `testdata/synthetic`
- Source layout:
  - `source/src/LensDiff.cpp`
  - `source/src/core/*`
  - `source/src/cuda/*`
  - `source/src/metal/*`
- Public groups:
  - `Selection`
  - `Optics`
  - `Phase`
  - `Spectrum`
  - `Composite`
  - `Defaults & Presets`
  - `Support`
  - `Optics` now includes `Circle`, `Polygon`, `Petals`, `Spiral`, `Hexagon`, `Square Grid`, and `Snowflake` as the startup-visible built-ins, with a user-driven injected custom aperture entry backed by a `Custom` import button and hidden file-path storage for imported grayscale or alpha masks with shared `Normalize` and `Invert` controls
  - `Phase` now owns the `Enable Phase` toggle plus nested `Primary`, `Higher Order`, `Field Variation`, `Field Higher Order`, `Chromatic`, and `Finishing` sections so the phase-facing UI reads as one coherent feature instead of being scattered through optics
  - The host UI now follows a stricter contextual-visibility policy: split-only composite controls stay hidden outside `Split`, `Core Compensation` stays hidden outside `Augment`, mono spectrum hides style-only controls, vane thickness stays hidden until vanes exist, and the phase UI keeps legacy duplicate wrappers hidden as compatibility-only params while surfacing the directly effective coefficients first
  - `Diffraction Scale`, `Kernel Radius`, `Creative Fringe`, and `Scatter Radius` now author in percent of the full frame short side, with runtime resolution into pixels done from source bounds so UHD-tuned presets keep their relative look at 1080p, UHD, and larger formats
  - `Defaults & Presets` mirrors the Chromaspace-style single-file preset workflow with a protected `Default`, shipped core presets for `Star Diff`, `Iris`, `Arrow Head`, `Snowflake`, `Fogy`, and `Imperfection`, JSON-backed user presets, and immediate menu refresh after save/update/rename/delete operations
  - `Support` currently exposes a read-only `Version` label so builds can report their packaged LensDiff version directly in-host
- Current backend status:
  - CPU reference: active and Windows-build-verified; retained as the debugging and parity anchor rather than a practical production renderer
  - CUDA: active host-CUDA render path built with CUDA kernels plus cuFFT; this is the primary production-performance target on Windows, and it now builds the raw shifted PSF and reference PSF on CUDA before feeding the shared host-side kernel shaping and cache finalization path
  - Metal: correctness-first host-Metal render path implemented with Metal compute kernels and direct GPU convolution; structurally aligned to the shared spectral mapping contract and now routed through its own explicit PSF-bank construction seam so a future Metal-side raw-PSF build can replace the shared host builder cleanly
  - Linux build/package path: supported through the `linux-ninja-*` CMake presets plus a portable-bundle GitHub Actions workflow that stages `Contents/Linux-x86-64/LensDiff.ofx`, verifies unresolved shared-library deps with `ldd`, and packages a `.tar.gz` artifact without bundling NVIDIA CUDA runtime libraries
  - Linux UI helpers: the custom aperture picker now uses Linux desktop helpers (`zenity` first, then `kdialog`) rather than failing unconditionally outside Windows/macOS
  - Diagnostics: `LENSDIFF_LOG` enables backend and fallback logging, and `LENSDIFF_TIMING` adds per-render timing to the same host-wrapper log output
  - `Pupil Resolution`: now exposed as explicit power-of-two choices through a visible compatibility wrapper while the legacy backing param remains hidden for project safety

## Deferred Features
- Real Metal FFT execution and Metal-side raw PSF synthesis
- Wider GPU ownership of PSF-bank shaping beyond the current CUDA raw-PSF synthesis stage
- 5-9 wavelength multi-bin spectral rendering
- Further tuning of the CIE-informed creative style endpoints
- Broader non-power-of-two FFT support beyond the current explicit power-of-two pupil-resolution choices
- Non-flat phase aberrations
- Experimental coherent propagation modes derived from angular-spectrum or Fresnel screen-distance simulation workflows
- Near-field propagation experiments
- Broader OCIO and transfer-function handling

## Open Questions And Risks
- The diffraction scale parameterization now uses a measured clear-circle first-minimum reference, but it still needs analytic validation against circular-pupil first-zero behavior and practical host-side tuning.
- Preserve-mode exactness is straightforward for a normalized kernel, but the interaction with split core and structure shaping still needs explicit regression tests.
- Resolve GPU host-pointer behavior differs from CPU and tiled hosts and must be treated as a backend integration problem, not as a math-layer concern.
- Metal support on Intel and Apple silicon must avoid APIs that silently narrow platform coverage.
- The current shared spectral mapping is now structurally aligned across backends, but its creative style endpoints still should not be treated as the final color science target for `v1B`.
- The first CUDA pass builds and links cleanly, but still needs host validation for image correctness, parity, and stream-lifetime behavior in Resolve.
- The first Metal pass is intentionally correctness-first and uses direct GPU convolution rather than FFT, so it still needs macOS validation for correctness, render-window behavior, and acceptable performance on both Intel and Apple silicon.
- Metal now consumes the shared spectral mapping matrices structurally, but macOS validation for parity and host integration is still deferred.
- Linux now has explicit build/package support, but Resolve-on-Linux validation still needs to confirm plugin loading, CUDA routing, aperture-dialog behavior, and long-session stability before Linux can be treated as release-ready rather than CI-ready.
- The current shared FFT path only guarantees correct pupil synthesis on power-of-two grids, so the present `Pupil Resolution` control is intentionally quantized internally until the UI or FFT implementation is widened.
- The new automatic selected-core safety floor is intended to prevent visually unacceptable black-hole centers, but it slightly relaxes exact local preserve-mode subtraction and still needs host-side tuning validation on real footage.
- Aggressive settings can expose the finite PSF support boundary, so the shared kernel build now needs validation that its circular support taper suppresses square veils without materially blunting genuine aperture-driven structure.
- The shared PSF bank now oversynthesizes and adaptively crops support for robustness, but those heuristics still need validation to ensure they do not trim legitimate long tails or spikes too aggressively.
- Before moving PSF-bank construction to CUDA, the remaining broad-halo shape behavior under aggressive settings still needs one clean discrimination pass so genuine polygon/obstruction envelopes are not mistaken for numerical artifacts, and vice versa.
- CUDA now synthesizes the raw shifted PSFs for active CUDA renders, but repeated-host validation still needs to confirm that the shared host-side kernel shaping produces the same bank as the CPU-only reference path across ordinary and aggressive settings.
- The diffractsim research set mixes two different problem classes: coherent field propagation and incoherent diffraction-limited imaging. LensDiff must keep those separated so highlight diffraction for photographed footage stays grounded in the incoherent PSF or OTF model instead of drifting into lab-style propagation visualizations.
- The first custom-aperture import pass intentionally treats the loaded image as an amplitude mask in the existing pupil model, which is the right fit for LensDiff, but it still needs UX validation around path failures, aspect-fit expectations, and whether the new inversion and normalization controls are sufficient in practice.
- Runtime mutation of OFX choice-menu options appears to be a host-risk area for Resolve startup stability, so the current design should avoid constructor-time or startup-time choice mutation and limit custom-menu injection to explicit user file-selection events.
- The first expanded built-in set (`Star`, `Spiral`, `Hexagon`, `Square Grid`, `Snowflake`) is analytic and intentionally stays inside the same incoherent pupil pipeline, but it still needs look-validation to confirm the more graphic masks feel useful and not merely decorative under aggressive settings.

## Technical Lessons Learned
### OFX Host
- Symptom: The first Windows build failed in the OFX wrapper even though the CPU diffraction core compiled cleanly.
- Cause: `ChoiceParam` time sampling needs the explicit out-parameter form, and helper descriptors must use the correct OFX descriptor types.
- Rule: Treat the host wrapper as a strict API boundary; verify descriptor types and time-sampled param accessors against the local OFX SDK instead of assuming convenience overloads exist.
- Verification: Updated `LensDiff.cpp` to use the correct descriptor types and explicit `ChoiceParam::getValueAtTime(time, value)` access, after which the Windows build succeeded.

- Symptom: Once an OFX plugin advertises CUDA render support, CPU fallback is no longer a safe universal escape hatch.
- Cause: GPU-enabled hosts may provide device pointers in `getPixelData()` rather than host-readable buffers.
- Rule: When CUDA render is enabled by the host, use the true CUDA path or fail explicitly; only allow CPU fallback when the host has not enabled GPU render for that frame.
- Verification: `LensDiff.cpp` now blocks CPU fallback when Resolve enables CUDA render and routes `Auto` to CUDA on supported Windows builds.

- Symptom: It was too easy to lose track of which backend actually rendered a frame or whether a fallback had occurred.
- Cause: Backend selection and failure reasons were implicit in control flow and not surfaced through any diagnostics path.
- Rule: Keep backend selection, fallback behavior, and optional timing visible at the host-wrapper layer so parity work can be debugged without guessing from the image alone.
- Verification: `LensDiff.cpp` now honors `LENSDIFF_LOG` and `LENSDIFF_TIMING`, logging selected backend, render geometry, debug mode, transfer mode, fallback notes, and render timing through `OFX::Log`.

### FFT And PSF Math
- Symptom: Early diffraction plans drifted toward "just add glow" shortcuts.
- Cause: The composite path was not constrained tightly enough around redistribution.
- Rule: Keep all effect behavior derivable from `selection -> redistribution -> convolution -> composite`; do not add external glow terms.
- Verification: Captured in the stable architectural principles and enforced by the v1A implementation contract.

- Symptom: The first tristimulus pass duplicated spectral mapping logic across CPU, CUDA, and Metal and drifted toward hand-tuned matrices.
- Cause: Spectral styling landed before a shared mapping contract existed.
- Rule: Build spectral mapping once from shared CIE-informed configuration and feed the same matrices into every active backend instead of hard-coding per-backend style math.
- Verification: Added `LensDiffSpectrum.*`, rewired CPU/CUDA to use the shared mapping config, and synced the Metal spectral kernel interface to the same host-provided matrices.

- Symptom: `diffractionScalePx` was meaningful in practice but still depended on a fixed internal bridge factor.
- Cause: The first implementation scaled the PSF with a hard-coded constant instead of measuring the raw circular-reference PSF produced by the current pupil rasterization and FFT path.
- Rule: Calibrate diffraction scale from the measured first minimum of a clear circular reference pupil built through the same PSF pipeline, then apply wavelength scaling on top of that reference.
- Verification: `LensDiffCpuReference.cpp` now measures a clear-circle reference PSF during PSF-bank rebuilds and derives the resampling scale from that measured radius.

- Symptom: Non-power-of-two `Pupil Resolution` values could make the effect disappear or behave erratically, and even-size impulse responses could look slightly off-center.
- Cause: The shared FFT implementation is radix-2 only, while the UI exposed arbitrary integer pupil sizes, and shifted even-size PSF grids were being sampled around a half-pixel center.
- Rule: Quantize the current pupil synthesis path to power-of-two sizes and sample shifted even-size FFT outputs around the integer Nyquist-centered origin; do not expose arbitrary FFT sizes unless the core truly supports them.
- Verification: `LensDiff.cpp` now quantizes `Pupil Resolution` at parameter resolution time, and `LensDiffCpuReference.cpp` reinforces that quantization in the shared core while using an even-size-aware center for raw PSF analysis and resampling.

- Symptom: A hidden power-of-two snap in the UI makes the control feel unpredictable even when the underlying FFT constraint is legitimate.
- Cause: The host-facing optics panel still exposed an integer slider after the core had already been constrained to radix-2 synthesis.
- Rule: Make FFT constraints explicit in the UI once they are known, instead of preserving a fake arbitrary-resolution control that silently maps to a smaller supported set.
- Verification: `LensDiff.cpp` now exposes `Pupil Resolution` as a choice among `64`, `128`, `256`, `512`, and `1024` rather than an arbitrary integer param.

- Symptom: Swapping an OFX param from integer to choice under the same param name can destabilize host project loading.
- Cause: OFX hosts may persist both param identity and param type across plugin upgrades, so a type change under the same name is a compatibility break rather than a harmless UI refactor.
- Rule: Preserve legacy param names and types for project compatibility, then layer new UI behavior on top with new param names or hidden compatibility shims.
- Verification: `LensDiff.cpp` now keeps the original `pupilResolution` int param hidden as the backing value and exposes the user-facing power-of-two UI through the new `pupilResolutionChoice` param.

- Symptom: Some real highlights could develop unnaturally dark or black-looking centers even though the PSF itself remained positive.
- Cause: Preserve-mode composite subtracts redistributed source energy locally before adding the convolved effect back, so extended or clipped highlights can underfill the center when the redistributed effect spreads energy outward.
- Rule: Keep the PSF and redistribution model intact, but enforce an automatic minimum selected-core floor in the final composite so pathological black-hole centers cannot appear in practical usage.
- Verification: CPU, CUDA, and Metal composite paths now clamp the final result against a minimum selected-core fraction derived from the selected source before output encoding.

- Symptom: Large or aggressively amplified highlights could reveal a square-looking outer veil even when the pupil and PSF center structure were otherwise plausible.
- Cause: The shared PSF was being resampled into a finite square support with a hard boundary, so low-energy tails near the kernel limit could leak the square crop into the final convolution.
- Rule: Treat the support boundary as a numerical window, not an optical feature; taper resampled PSFs smoothly to zero near the maximum support radius so extreme settings do not expose a square kernel box.
- Verification: `LensDiffCpuReference.cpp` now applies a circular support taper before kernel normalization, and all active backends inherit the same host-built PSF bank.

- Symptom: Even after support tapering, aggressive settings could still expose boxy diffraction envelopes because the raw PSF synthesis grid and fixed exported support were too tightly coupled to the UI limits.
- Cause: The user-facing pupil raster was also the raw FFT synthesis grid, and kernels always exported at the full requested maximum radius whether or not the tail energy justified that support.
- Rule: Decouple internal raw PSF synthesis from the UI pupil resolution and adaptively crop exported kernel support from the actual kernel energy and residual tail strength, so robustness comes from better support estimation rather than cosmetic clamping.
- Verification: `LensDiffCpuReference.cpp` now zero-pads pupil synthesis onto a larger internal FFT grid and crops each exported kernel to an adaptive support radius before caching.

- Symptom: Built-in analytic iris controls alone cannot cover the range of aperture masks explored in the diffractsim references.
- Cause: Many of the target examples are fundamentally image-defined amplitude masks rather than simple circles or polygons.
- Rule: Bring custom masks into LensDiff as amplitude pupils that feed the same incoherent PSF-bank pipeline, preserving aspect ratio inside the pupil and treating white-plus-alpha as transmissive amplitude.
- Verification: `LensDiffApertureImage.*`, `LensDiff.cpp`, and `LensDiffCpuReference.cpp` now add a `Custom` aperture mode backed by imported image masks on the shared pupil-building path.

- Symptom: Users do not want a permanently visible `Custom` entry, but unrestricted runtime choice mutation can destabilize OFX host startup.
- Cause: The risky part is not dynamic menu mutation in itself, but doing it during startup or instance restoration instead of in direct response to a user edit.
- Rule: Keep the startup aperture menu static and built-in-only, then inject and select the custom aperture entry only after the user explicitly chooses a file through the `Custom` import button and native file picker.
- Rule: Never mutate the aperture choice list in response to ordinary aperture-menu interaction; restrict custom-entry injection to the explicit import-button path to reduce Resolve host-instability risk.
- Verification: `LensDiff.cpp` now mutates `apertureMode` only from `changedParam(customAperturePath)` when the change reason is `eChangeUserEdit`, avoiding constructor-time menu mutation while still removing `Custom` from the default menu.

- Symptom: Imported masks often need external cleanup before they behave as useful pupils, especially when black/white polarity or limited value range does not match the expected transmission convention.
- Cause: The first custom-mask pass assumed the source mask was already authored in the correct polarity and full 0..1 range.
- Rule: Keep imported masks on the same amplitude-pupil path, but add explicit shared `Normalize` and `Invert` controls so common mask-prep corrections happen inside LensDiff rather than outside it.
- Verification: `LensDiff.cpp`, `LensDiffTypes.h`, and `LensDiffCpuReference.cpp` now expose `Custom Normalize` and `Custom Invert`, feed them into the cache key, and apply them before custom-mask resampling.

- Symptom: Relying only on `Circle`, `Polygon`, and imported masks makes it slower to iterate toward some of the reference pupils the user cares about.
- Cause: Several target looks, especially star-like and spiral-iris masks, are common enough to deserve built-in analytic options rather than always requiring image import.
- Rule: Add more built-in pupil modes only when they can be expressed as amplitude masks inside the same shared incoherent pupil builder so all backends inherit them automatically.
- Verification: `LensDiff.cpp` and `LensDiffCpuReference.cpp` now expose and implement analytic `Star`, `Spiral`, `Hexagon`, `Square Grid`, and `Snowflake` aperture modes alongside `Circle`, `Polygon`, and `Custom`.

### CUDA
- Symptom: The CUDA backend needs the same optical truth as the CPU path, not a second divergent PSF implementation.
- Cause: Rebuilding pupil and kernel synthesis separately on the GPU would increase drift and debugging cost.
- Rule: Keep PSF-bank synthesis shared and cached on the host side, then run highlight extraction, convolution, and compositing as true CUDA work on host CUDA buffers.
- Verification: `LensDiffCuda.cu` now reuses the shared PSF bank and performs device-side decode, selection, convolution, energy handling, and final composite.

- Symptom: Shared CPU-side PSF-bank rebuilds became the practical bottleneck once the render path itself moved to CUDA.
- Cause: Raw pupil-to-FFT synthesis still ran entirely on the host even when the active render backend was CUDA.
- Rule: Move the heavy raw pupil-to-shifted-PSF FFT stage onto CUDA first, but keep pupil generation, kernel shaping, and final cache assembly on the shared host path until parity is firmly established.
- Verification: `LensDiffCuda.cu` now builds the raw shifted PSF and clear-circle reference PSF with CUDA plus cuFFT, then finalizes the cached kernel bank through the shared host-side shaping code.

### Metal
- Symptom: Jumping straight from a stub to a Metal FFT path would have created too many moving pieces to validate safely on a non-macOS workstation.
- Cause: Host-Metal buffer handling, render-window addressing, transfer parity, and convolution math all needed to be proven before optimizing the backend architecture.
- Rule: Land Metal in two phases: first a correctness-first GPU path with explicit buffer handling and direct convolution, then replace the hot path with FFT once parity is confirmed on real macOS hosts.
- Verification: `LensDiffMetal.mm` now compiles a dedicated Metal kernel library at runtime and routes selection, convolution, spectral mapping, preserve-mode scaling, and final packing through host Metal buffers.

- Symptom: CUDA gained an explicit raw-PSF construction seam, but Metal was still calling the shared PSF-cache builder directly.
- Cause: The first Metal pass focused on correctness-first direct convolution and never needed its own cache-build gate.
- Rule: Keep Metal on the same PSF-bank construction interface as CUDA even before Metal-side FFT work lands, so backend-specific raw-PSF synthesis can be added later without disturbing render-path behavior.
- Verification: `LensDiffMetal.mm` now routes cache creation through `ensurePsfBankMetal(...)`, which validates the shared cache key and centralizes the future replacement point for Metal-side PSF building.

### Performance
- Symptom: PSF synthesis is much more expensive than parameter reads or simple pixel loops.
- Cause: FFT-derived kernels and debug artifacts are resolution-dependent and expensive to rebuild unnecessarily.
- Rule: Cache PSF-bank artifacts by optics and spectrum key; do not treat pupil synthesis as a per-pixel concern.
- Verification: Cache key and cache struct are part of the first implementation pass.

### Validation
- Symptom: Complex optical effects can look plausible while being mathematically wrong.
- Cause: Visual spot checks alone hide normalization, wraparound, and parity errors.
- Rule: Prefer impulse, PSF sum, OTF relation, exposure invariance, and backend parity tests before look tuning.
- Verification: Included in the validation matrix below.

- Symptom: Ad hoc footage is useful for beauty checks, but poor at isolating PSF, threshold, and backend regressions.
- Cause: Real scenes mix too many variables at once and hide simple correctness failures behind production texture.
- Rule: Maintain deterministic synthetic sources for impulse, exposure, spectrum, edge, and odd-dimension checks, and use them before creative look tuning.
- Verification: Added `tools/generate_synthetic_sources.py`, the supporting requirements/doc files, and the `testdata/synthetic` output path.

- Symptom: Purely diagnostic plates are good for correctness, but they are not enough to tune default looks or judge whether creative spectrum endpoints feel cinematic.
- Cause: Endpoint tuning needs controlled but more scene-like light arrangements without giving up determinism.
- Rule: Extend the synthetic suite with beauty-oriented but still deterministic practical-light scenes and directional stress probes before moving to live-action tuning footage.
- Verification: Added advanced generator outputs for annular layouts, vane probes, diagonal glints, and a controlled practical-light beauty scene.

- Symptom: A Linux CMake preset by itself made LensDiff look portable, but it did not prove that the staged OFX bundle, runtime dialog helpers, or shared-library closure were actually ready for host use.
- Cause: Build scaffolding existed before Linux artifact verification, Linux-native file-picker support, and portable-bundle CI were treated as release-facing requirements.
- Rule: Treat Linux readiness as three separate layers: preset-based build, verified staged bundle plus dependency closure, and manual Resolve-on-Linux validation.
- Verification: Added Linux desktop-helper support for the custom aperture picker, documented the Linux build/deployment policy, and introduced a Linux CI workflow that builds, stages, verifies, and packages `LensDiff.ofx.bundle`.

## Math And Research Verification Notes
- Claim: Highlight diffraction for photographed footage should use an incoherent imaging model based on PSF convolution rather than full-scene coherent propagation.
- Source Or Paper: Rafael de la Fuente's Fourier-optics article, informed by Goodman-style Fourier optics references.
- Implementation Consequence: `LensDiff` uses pupil synthesis -> FFT-derived PSF -> highlight-only convolution as the main render path.
- Validation: Architectural review and initial v1A implementation design.
- Confidence: `tested`

- Claim: The diffractsim lens examples and article cover both coherent propagation through lenses and incoherent diffraction-limited imaging, and those should not be collapsed into one render mode.
- Source Or Paper: Rafael de la Fuente's article "Simulating Light Diffraction with Lenses - Visualizing Fourier Optics", local `diffractsim-2.2.3` package, and the `optical_imaging_system_using_convolution__polychromatic_incoherent.py` example.
- Implementation Consequence: LensDiff's mainline feature set should continue following the incoherent imaging path, while screen-distance lens animations, Fourier-plane propagation views, and other coherent field simulations remain explicit future experimental modes rather than silent scope creep in `v1B`.
- Validation: Local diffractsim source review confirmed that the package contains both angular-spectrum propagation workflows and a separate incoherent convolution-based imaging example.
- Confidence: `tested`

- Claim: The OTF is the Fourier transform of the PSF and is linked to the pupil through autocorrelation relations.
- Source Or Paper: Williams and Becklund-style OTF treatment; see also the Hopkins-style relation referenced during planning.
- Implementation Consequence: OTF debug views should be derived from the PSF, not invented separately.
- Validation: OTF debug output is computed from the cached PSF bank in the CPU reference path.
- Confidence: `tested`

- Claim: `anisotropyEmphasis` should preserve exact unit-energy normalization after reshaping.
- Source Or Paper: Internal design requirement derived from preserve-mode stability goals.
- Implementation Consequence: Every reshaped PSF is renormalized to sum exactly to 1 before use.
- Validation: CPU implementation and validation matrix checks.
- Confidence: `assumed`

- Claim: Input transfer handling must decode to linear before selection and convolution, then re-encode the final output to the chosen input transfer.
- Source Or Paper: Internal design requirement, with DaVinci Intermediate decode behavior reused from `common/color/ColorManagement.cpp`.
- Implementation Consequence: `LensDiff` now exposes an `Input Transfer` selector with `Linear` and `DaVinci Intermediate`; CPU and CUDA both process internally in linear light.
- Validation: Windows build verification complete; host-image validation is still pending.
- Confidence: `tested`

- Claim: Creative spectrum styles must be luminance-normalized when `chromaticAffectsLuma=off`.
- Source Or Paper: Internal design requirement from the v1.3 plan.
- Implementation Consequence: Style mapping endpoints are constrained so spectrum-force changes hue and chroma without unintended brightness drift.
- Validation: CPU, CUDA, and Metal now build from the same shared CIE-informed basis structurally, but still need Resolve-side regression tests and possible endpoint tuning.
- Confidence: `tested`

- Claim: `diffractionScalePx` should map to a clear circular reference radius measured from the plugin's own pupil-to-PSF pipeline rather than a fixed implementation constant.
- Source Or Paper: Internal design requirement derived from the v1.3 plan's "clear circular pupil at 550 nm" interpretation.
- Implementation Consequence: PSF-bank rebuilds now measure the first radial minimum of a clear circular reference PSF and use that radius to calibrate resampling for every wavelength bin.
- Validation: Windows build verification complete; host-image validation and analytic comparison against Airy first-zero expectations are still pending.
- Confidence: `tested`

- Claim: The current shared FFT path is only valid on power-of-two pupil grids, and even-size shifted FFT outputs must be sampled about the integer-centered shifted origin rather than a half-pixel midpoint.
- Source Or Paper: Internal implementation constraint of the current radix-2 FFT and standard even-size `fftshift` indexing behavior.
- Implementation Consequence: `Pupil Resolution` is quantized internally to power-of-two values for now, and raw PSF resampling plus radial analysis use an even-size-aware center coordinate.
- Validation: Host testing surfaced disappearing non-power-of-two pupil sizes and slight centering drift; the shared CPU core and host param resolution were updated accordingly.
- Confidence: `tested`

- Claim: A practical highlight-diffraction composite should not be allowed to form visually black core holes, even if local redistribution underfills the center.
- Source Or Paper: Internal safety requirement derived from the v1.3 design goal that composite policy may shape application without replacing the PSF truth.
- Implementation Consequence: Final composite now enforces an automatic minimum selected-core floor after redistribution and convolution but before output encoding.
- Validation: Added the safeguard consistently to CPU, CUDA, and Metal composite paths; host-image tuning validation is still pending.
- Confidence: `tested`

## Validation Matrix
- Validate PSF truth first:
  - delta input reproduces the PSF
  - PSF sums to 1
  - circular pupils show Airy-like behavior
  - annular and vane pupils produce expected structural changes
  - OTF from FFT(PSF) matches the reference relation within tolerance
- Validate energy behavior in both composite modes:
  - `Preserve` must conserve selected highlight energy numerically
  - `Augment` must deviate only by the explicit gain difference
- Validate `anisotropyEmphasis`:
  - total PSF energy remains exactly 1 after reshaping
  - no negative kernel values survive
  - exposure changes do not alter the control except through selection policy
- Validate creative spectrum behavior:
  - `spectrumForce` moves monotonically from `Natural` to the selected style
  - `spectrumForce=0` matches `Natural`
  - `spectrumForce=1` matches the selected style
  - with `chromaticAffectsLuma=off`, effect luminance remains constant within tolerance
- Validate exposure invariance:
  - scale linear source exposure up and down
  - confirm PSF geometry is unchanged
  - verify only threshold policy changes
- Validate backend parity across CPU, CUDA, and Metal on:
  - impulse tests
  - polygon pupils
  - annular pupils
  - odd image sizes
  - non-zero origins
  - cropped render windows
  - animated parameters
  - repeated renders

## Stage Changelog
- 2026-04-18
  - Moved the old CPU-only phase suite features onto the backend render paths so field variation, scatter, creative fringe, and the extra phase debug views no longer force a full CPU reference render on GPU-capable hosts.
  - Extended the CUDA PSF-bank builder to synthesize 3x3 field-zone caches directly from resolved field-phase params, then blended the per-zone GPU renders with the same bilinear zone weighting used by the CPU reference.
  - Mirrored that field-zone accumulation strategy in Metal and added backend-local post passes for scatter blur and creative fringe so both GPU backends now follow the same user-facing feature contract.
- 2026-04-17
  - Reorganized the host UI so `Phase` is a dedicated top-level group with contextual subgroups for primary terms, higher-order terms, field variation, chromatic controls, and finishing controls rather than a flat optics spillover.
  - Hid inert controls more aggressively across the plugin: `Split`-only composite controls now live under a contextual subgroup, `Core Compensation` now hides outside `Augment`, mono spectrum hides style-only controls, and optics-only details such as vane thickness now hide until their parent feature is active.
  - Kept the legacy phase macro wrappers alive only as hidden compatibility shims so saved presets and older JSON still round-trip, while the public UI now surfaces the directly effective coefficients such as `Astig 0` and `Astig 45` first.
- 2026-04-17
  - Replaced the first-pass raw phase sliders with a single structured Zernike-based phase system, exposed as one advanced control set behind an `Enable Phase` toggle and shared normalized coefficients for defocus, astigmatism, coma, spherical, and trefoil terms.
  - Centralized the non-flat phase basis in a new shared core layer so CPU, CUDA, and Metal all synthesize the same phase-aware pupil contract instead of maintaining backend-local phase math.
  - Added a dedicated `Phase` debug view and carried the cached pupil phase map through all backends so phase validation can happen directly in-host rather than being inferred only from beauty output.
- 2026-04-17
  - Reworked `Spectral 9` so the canonical 9-bin mode now derives its natural and styled color columns from the balanced `Spectral 5` anchor set instead of treating every extra wavelength sample as a full new narrowband hue.
  - Kept the denser 9-bin PSF geometry, but pulled the visible color response closer to the established `Spectral 5` look after Resolve feedback showed the previous dense-bank mapping still leaning too cyan-blue.
  - Fixed a backend parity bug in the CUDA and Metal spectral mappers where only the first 5 spectral planes were being consumed during RGB reconstruction, which made `Spectral 9` disproportionately blue by dropping the warm-side bins on GPU.
- 2026-04-16
  - Tuned `Spectral 9` so the dense-bank natural and styled sums are balanced toward the established `Spectral 5` reference instead of drifting cooler under the wider wavelength set.
- 2026-04-16
  - Promoted the saved `Fuzzy` look into the shipped built-in preset catalog as `Arrow Head`, so it behaves like the existing native presets, including reserved-name protection and duplicate cleanup on load.
- 2026-04-13
  - Added an opt-in non-flat pupil phase scaffold with defocus, astigmatism, coma, and spherical controls carried through presets, cache keys, CPU raw-PSF synthesis, CUDA raw-PSF synthesis, and the Metal path's shared raw-PSF builder.
  - Kept diffraction-scale calibration anchored to a flat clear-circle reference so the first aberration pass changes PSF shape without redefining the scale reference.
- 2026-04-13
  - Started `v2` explicitly by widening the shared spectral-bank capacity from 5 bins to 9 bins and adding a new `Spectral 9` mode on the same CPU/CUDA/Metal contract.
  - Kept the previous 5-bin path as an explicit `Spectral 5` mode so existing looks and saved presets remain behaviorally stable while denser spectral work begins.
- 2026-04-13
  - Promoted the saved `Star Diff` and `Iris` looks into shipped core presets so fresh LensDiff installs expose them by default without relying on user JSON state.
  - Reserved the shipped core preset names against future user save/rename operations, matching the existing protected `Default` behavior.
  - Added startup de-duplication for exact user-preset copies of the shipped core presets and labeled any remaining legacy name collisions as `(... User)` in the preset menu for cleanup clarity.
- 2026-04-03: Added a dedicated 5-bin `Spectral` mode on top of the existing `Mono` and `Tristimulus` paths, extended the shared spectrum matrices to 5-bin mapping, and structurally synced CUDA and Metal to the same multi-bin contract so the next stage can validate look and parity instead of redesigning backend plumbing.
- 2026-04-03: Rebalanced the `Cyan-Magenta` and `Warm-Cool` creative spectrum endpoints again, pulling both styles closer to `Natural` after real-footage feedback showed each still leaning too strongly toward one side at high force and saturation.
- 2026-04-04
  - Added a new `Support` UI group with a read-only `Version` label so in-host builds report the active LensDiff version directly in the plugin controls.
  - Updated the OFX descriptor label and plugin grouping so hosts list the plugin as `LensDiff v0.1.16Beta` under `Moaz Elgabry`, matching the Chromaspace-style menu presentation.
  - Investigated warmth drift in the new 5-bin `Spectral` mode versus `Tristimulus` and confirmed the shared mapping had only been explicitly white-balanced on the 3-bin path.
  - Added a generalized neutral-balance pass for non-3-bin spectral mappings so the 5-bin `Natural` response no longer carries an avoidable warm bias relative to `Tristimulus`.
  - Fixed a Metal parity bug where the host-side composite parameter struct had fallen out of sync with the shader's selected-core safety-floor input, and tightened Metal spectral setup to use the shared resolved spectral bin count.
  - Added a Chromaspace-style LensDiff preset manager with a protected `Default`, JSON-backed user presets, menu-level `Custom` / `Modified` state, and push-button save/update/rename/delete actions that rebuild the preset menu immediately after each mutation.
  - Wired saved defaults into LensDiff's descriptor defaults so `Save Defaults` affects new instances after the next plugin or host restart, including imported custom-aperture state when that mode is the active saved look.
- 2026-04-01
  - Landed the initial `LensDiff` plugin tree under `OFX-Workshop`.
  - Added the canonical `development-plan.md` document and the layered source layout.
  - Established the v1A CPU-first implementation target with explicit CUDA and Metal seams.
  - Deferred real GPU FFT execution, multi-bin spectral rendering, and advanced optical features to later stages.
  - Identified key risks around diffraction scale calibration, preserve-mode validation, and host GPU integration.
- 2026-04-01
  - Built `LensDiff.ofx` successfully on Windows and staged the bundle under `source/bundle/LensDiff.ofx.bundle`.
  - Confirmed the CPU reference path, OFX parameter groups, ROI expansion, identity wiring, and debug-view plumbing compile together.
  - Kept CUDA as a compile-valid stub with CPU fallback rather than advertising incomplete GPU execution as production-ready.
  - Marked the current spectrum mapping as provisional so `v1B` can replace it with CIE-fitted mappings without design drift.
- 2026-04-01
  - Added an `Input Transfer` selector with `Linear` and `DaVinci Intermediate`, decoding internally to linear and re-encoding the final output.
  - Reused `OFX-Workshop/common/color/ColorManagement.cpp` for shared transfer metadata and DaVinci Intermediate decode behavior.
  - Replaced the CUDA stub with a true device render path using CUDA kernels plus cuFFT on host CUDA buffers.
  - Updated host routing so `Auto` selects CUDA when Resolve enables CUDA render, while CPU fallback remains available only for non-GPU host renders.
- 2026-04-02
  - Replaced the Metal stub with a correctness-first host-Metal render path using runtime-compiled Metal compute kernels.
  - Added direct GPU selection, mono and tristimulus convolution, spectral mapping, preserve-mode scaling, static debug views, and final compositing on Metal buffers.
  - Enabled OFX Metal render support on macOS so `Auto` can route to the Metal backend when the host provides a Metal command queue.
  - Deferred Metal FFT optimization until after macOS parity and performance validation.
- 2026-04-02
  - Moved active development into `v1B` by introducing `LensDiffSpectrum.*` as a shared spectral-mapping layer.
  - Replaced the provisional CPU and CUDA spectral matrices with a CIE-informed basis derived from the plugin's active wavelength bins.
  - Kept the creative style endpoints monotonic while balancing equal-energy tristimulus bins back toward a neutral natural response.
  - Deferred Metal spectral-sync validation with the rest of the macOS testing backlog.
- 2026-04-02
  - Synced the Metal backend to the shared `LensDiffSpectrum` matrix contract so CPU, CUDA, and Metal now share the same spectral mapping architecture.
  - Replaced the old fixed diffraction-scale bridge with a measured clear-circle reference radius derived from the PSF-bank build itself.
  - Kept Metal host validation deferred while still aligning its shader inputs and host-side parameters with the active `v1B` contracts.
  - Left further creative style tuning and real-host parity checks as the next `v1B` work items.
- 2026-04-02
  - Added host-wrapper diagnostics for backend selection, fallback reporting, and optional per-render timing via `LENSDIFF_LOG` and `LENSDIFF_TIMING`.
  - Kept the diagnostics logic in the OFX host layer so backend execution and core math remain unchanged.
  - Confirmed the Windows build still succeeds after the diagnostics pass.
  - Pointed the next `v1B` step toward creative style tuning and CPU/CUDA validation using the new logs.
- 2026-04-02
  - Added a Python-based synthetic-source generator for deterministic float32 TIFF validation plates.
  - Documented the workflow and pinned the lightweight `numpy` and `tifffile` dependencies for reproducible source generation.
  - Defined a starter suite covering impulse, edge, grid, exposure, spectrum, slit, and odd-dimension checks.
  - Shifted the validation workflow toward generated plates before footage-based creative tuning.
- 2026-04-02
  - Extended the synthetic generator with advanced annular, vane, diagonal-glint, and practical-light beauty plates.
  - Kept the new plates deterministic so they can serve both correctness checks and early creative endpoint tuning.
  - Updated the synthetic-source docs to describe the expanded suite.
  - Strengthened the validation path so look tuning can stay controlled before relying on live-action footage.
- 2026-04-02
  - Quantized `Pupil Resolution` internally to the nearest power of two in the host layer and reinforced the same rule in the shared PSF core.
  - Fixed raw PSF centering assumptions for even-size shifted FFT grids so impulse responses and scale calibration stop drifting around a half-pixel midpoint.
  - Captured the current limitation explicitly: for now the control is quantized internally, and the planned cleanup is to expose discrete power-of-two UI choices or broaden FFT support beyond power-of-two grids.
- 2026-04-02
  - Added an automatic selected-core safety floor to the final composite so pathological black-hole centers cannot form even when local redistribution underfills the source highlight center.
  - Kept the safeguard in composite policy rather than PSF synthesis so the optical model stays unchanged.
  - Synced the new safety rule across CPU, CUDA, and Metal paths and recorded the resulting preserve-mode tradeoff explicitly in the plan.
- 2026-04-02
  - Added a circular support taper to the shared resampled PSF kernel so finite kernel boundaries stop leaking square crop artifacts into extreme highlight blooms.
  - Kept the taper in the shared PSF-bank build, which means CPU, CUDA, and Metal all inherit the same robustness fix automatically.
  - Recorded the numerical-support rationale explicitly so this remains a documented stability rule rather than an unexplained look tweak.
- 2026-04-02
  - Decoupled internal raw PSF synthesis from the UI pupil raster by zero-padding onto a larger internal FFT grid.
  - Added adaptive kernel-support cropping based on captured energy and residual tail strength so exported PSFs stop carrying oversized low-energy box support by default.
  - Kept the change inside the shared PSF-bank build so CPU, CUDA, and Metal all continue to consume the same kernels and diagnostics stay comparable.
- 2026-04-02
  - Reaffirmed that the CPU backend is a correctness and parity anchor, not a practical shipping renderer for LensDiff's target workloads.
  - Prioritized one more aggressive-setting correctness pass on shared PSF/support behavior ahead of GPU PSF-bank construction.
  - Recorded CUDA-side PSF-bank construction as the next performance optimization after that validation gate, while keeping CPU synthesis as the oracle path.
- 2026-04-02
  - Reviewed the local diffractsim package and Rafael de la Fuente's Fourier-optics article against LensDiff's current architecture.
  - Confirmed that LensDiff already tracks the incoherent imaging methodology from that research, while many diffractsim showcase examples belong to a separate coherent propagation problem class.
  - Recorded explicit scope guidance so future work can pursue coherent propagation as an opt-in experimental branch without derailing the main highlight-diffraction roadmap.
- 2026-04-02
  - Moved the raw pupil-to-shifted-PSF synthesis stage onto CUDA for active CUDA renders while preserving the shared host-side kernel shaping and cache finalization path.
  - Kept CPU synthesis as the oracle and fallback path so the numerical truth of the PSF bank still lives in one shared shaping contract.
  - Set the next validation target to CPU/CUDA parity for the rebuilt PSF bank and real-host profiling of repeated optics changes.
- 2026-04-02
  - Added an explicit Metal-side PSF-bank construction gate so Metal now follows the same cache-build seam as CUDA.
  - Kept the current Metal implementation on the shared host-side PSF builder while reserving that new gate as the clean replacement point for a future Metal raw-PSF path.
  - Updated the plan to treat real Metal FFT and Metal-side raw PSF synthesis as a later validated optimization stage rather than hidden behavior.
- 2026-04-02
  - Started the first custom-aperture feature by adding a `Custom` optics mode plus a file-path parameter for imported masks.
  - Added platform image loading on the shared pupil path so imported grayscale or alpha masks become amplitude pupils inside the existing PSF-bank architecture.
  - Kept the scope disciplined: imported masks still feed the incoherent pupil -> PSF -> convolution pipeline rather than introducing a separate coherent-propagation mode.
- 2026-04-02
  - Extended the built-in analytic aperture set with `Star` and `Spiral` modes inspired by the reference masks under discussion.
  - Kept both new modes inside the shared pupil builder so CPU, CUDA, and Metal all inherit the same pupil-driven PSF behavior automatically.
  - Left honeycomb, gratings, and other more specialized mask families to either `Custom` import or a later built-in expansion pass.
- 2026-04-02
  - Extended the built-in analytic aperture set again with `Hexagon`, `Square Grid`, and `Snowflake` modes to cover more of the referenced pupil-mask family without requiring image import.
  - Kept the new masks on the same shared analytic pupil path so all active backends inherit them automatically and cache behavior remains unchanged.
  - Left more specialized families such as honeycomb arrays and other repeated lattices as either `Custom` import or a later dedicated built-in pass.
- 2026-04-02
  - Reworked the custom-aperture UX around a dedicated `Custom` import button, hidden file-path storage, and a user-driven dynamic menu entry instead of a permanently visible `Custom` item.
  - Restricted the dynamic menu injection to explicit user file-selection events so Resolve startup no longer depends on constructor-time or restoration-time choice mutation.
  - Kept auto-selection of the newly loaded custom aperture while leaving the default startup menu free of custom entries.
- 2026-04-02
  - Added shared `Custom Normalize` and `Custom Invert` controls so imported aperture masks can be corrected inside LensDiff without external preprocessing.
  - Wired those controls into the shared custom pupil builder and PSF-cache key so CPU, CUDA, and Metal continue to consume the same resolved custom aperture.
  - Kept the new controls disabled until a custom aperture file is actually loaded so the optics UI stays focused for non-custom users.
- 2026-04-03
  - Softened the creative spectral endpoint maps again and made them explicitly relative to `Natural` so `Cyan-Magenta` and `Warm-Cool` stay stylized without pulling the whole diffraction field too hard toward one endpoint.
  - Kept the luminance-normalized and monotonic blend behavior unchanged so the endpoint tuning remains grounded in the same spectrum architecture.
  - Renamed the former `Star` pupil label to `Petals` in the public optics UI to better match the actual analytic shape.
- 2026-04-03
  - Added an explicit `Custom Status` optics field so imported aperture masks report load success and image dimensions instead of failing silently.
  - Kept the status field hidden unless a loaded custom aperture is actively selected, matching the contextual behavior of the normalize and invert controls.
  - Left render-time custom-aperture behavior unchanged so this stage improves diagnostics without perturbing backend parity.
- 2026-04-02
  - Replaced the old arbitrary `Pupil Resolution` integer control with an explicit power-of-two choice so the optics UI now matches the actual FFT constraints.
  - Removed the last hidden host-side pupil-resolution quantization from the user-facing control path while keeping the same shared radix-2 synthesis contract underneath.
  - Narrowed the deferred work from "make the UI honest" to the deeper future option of adding true non-power-of-two FFT support.
- 2026-04-02
  - Adjusted the new `Pupil Resolution` UI to restore OFX project compatibility after recognizing that retyping the existing param name from int to choice was host-risky.
  - Kept the original `pupilResolution` int param as a hidden compatibility backing field and introduced `pupilResolutionChoice` as the visible power-of-two selector.
  - Preserved the same explicit FFT-size UI while avoiding a likely Resolve crash path on older saved LensDiff instances.
- 2026-04-18
  - Added shared custom-aperture image decode caching keyed by file path and last-write timestamp so repeated optics edits stop reloading the same mask from disk across CPU, CUDA, and Metal paths.
  - Added cached static-debug image reuse plus stage-level `LENSDIFF_TIMING` logging for custom aperture loads, PSF-bank global rebuilds, field-zone rebuilds, and static debug generation.
  - Reused a shared cached reference clear-circle raw PSF across backends, moved CUDA cropped-kernel taper and normalization onto the device, and reduced Metal PSF-bank command-buffer waits by batching dependent compute passes into fewer synchronized submissions.
- 2026-04-18
  - Moved CUDA pupil-amplitude and phase-wave raster generation onto the device so the raw-PSF build no longer uploads host-built optical intermediates before the cuFFT stage.
  - Reworked Metal optical prep so pupil and phase rasters are built on GPU too, added a direct Metal row-column FFT path for power-of-two raw-PSF grids, and added a GPU Bluestein path for non-power-of-two raw-PSF grids before the existing GPU shaping/composite stages.
  - Kept the internal raw-PSF sizing rule unchanged so the visual appearance of the evaluated presets does not drift under a hidden power-of-two snap, while still targeting full GPU ownership of raw-PSF synthesis on both CUDA and Metal.

#pragma once

#include "ColorManagement.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

constexpr int kLensDiffMaxSpectralBins = 9;

enum class LensDiffBackendType {
    Auto,
    CpuReference,
    Cuda,
    Metal,
};

enum class LensDiffExtractionMode {
    MaxRgb,
    Luma,
};

enum class LensDiffApertureMode {
    Circle,
    Polygon,
    Star,
    Spiral,
    Custom,
    Hexagon,
    SquareGrid,
    Snowflake,
};

enum class LensDiffApodizationMode {
    Flat,
    Cosine,
    Gaussian,
};

enum class LensDiffSpectralMode {
    Mono,
    Tristimulus,
    Spectral5,
    Spectral9,
};

enum class LensDiffSpectrumStyle {
    Natural,
    CyanMagenta,
    WarmCool,
};

enum class LensDiffLookMode {
    Physical,
    Split,
};

enum class LensDiffEnergyMode {
    Preserve,
    Augment,
};

enum class LensDiffDebugView {
    Final,
    Selection,
    Pupil,
    Psf,
    Otf,
    Core,
    Structure,
    Effect,
    Phase,
    PhaseEdge,
    FieldPsf,
    ChromaticSplit,
    CreativeFringe,
    Scatter,
};

enum class LensDiffInputTransfer {
    Linear = 0,
    DavinciIntermediate,
};

enum class LensDiffPhaseMode {
    Off,
    Enabled,
};

struct LensDiffImageRect {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;

    int width() const { return x2 - x1; }
    int height() const { return y2 - y1; }
};

struct LensDiffImageView {
    void* data = nullptr;
    std::ptrdiff_t rowBytes = 0;
    LensDiffImageRect bounds {};
    int originX = 0;
    int originY = 0;
    int components = 4;
    int bytesPerComponent = 4;
};

struct LensDiffRenderRequest {
    LensDiffImageRect renderWindow {};
    LensDiffImageRect frameBounds {};
    LensDiffImageView src {};
    LensDiffImageView dst {};
    LensDiffBackendType requestedBackend = LensDiffBackendType::Auto;
    LensDiffBackendType selectedBackend = LensDiffBackendType::CpuReference;
    int frameShortSidePx = 0;
    int estimatedSupportRadiusPx = 0;
    bool hostEnabledCudaRender = false;
    bool hostEnabledMetalRender = false;
    void* cudaStream = nullptr;
    void* metalCommandQueue = nullptr;
};

struct LensDiffParams {
    LensDiffExtractionMode extractionMode = LensDiffExtractionMode::MaxRgb;
    LensDiffApertureMode apertureMode = LensDiffApertureMode::Circle;
    LensDiffApodizationMode apodizationMode = LensDiffApodizationMode::Flat;
    LensDiffSpectralMode spectralMode = LensDiffSpectralMode::Mono;
    LensDiffSpectrumStyle spectrumStyle = LensDiffSpectrumStyle::Natural;
    LensDiffLookMode lookMode = LensDiffLookMode::Split;
    LensDiffEnergyMode energyMode = LensDiffEnergyMode::Preserve;
    LensDiffDebugView debugView = LensDiffDebugView::Final;
    LensDiffInputTransfer inputTransfer = LensDiffInputTransfer::Linear;
    bool resolutionAware = false;

    double threshold = 4.0;
    double softnessStops = 1.0;
    double pointEmphasis = 0.0;
    double corePreserve = 0.1;

    int bladeCount = 6;
    double roundness = 0.5;
    double rotationDeg = 0.0;
    double centralObstruction = 0.0;
    int vaneCount = 0;
    double vaneThickness = 0.01;
    double diffractionScalePx = 24.0;
    int pupilResolution = 256;
    double maxKernelRadiusPx = 64.0;
    int frameShortSidePx = 2160;
    std::string customAperturePath;
    bool customApertureNormalize = true;
    bool customApertureInvert = false;
    LensDiffPhaseMode phaseMode = LensDiffPhaseMode::Off;
    double phaseDefocus = 0.0;
    double phaseAstigmatism0 = 0.0;
    double phaseAstigmatism45 = 0.0;
    double phaseComaX = 0.0;
    double phaseComaY = 0.0;
    double phaseSpherical = 0.0;
    double phaseTrefoilX = 0.0;
    double phaseTrefoilY = 0.0;
    double phaseSecondaryAstigmatism0 = 0.0;
    double phaseSecondaryAstigmatism45 = 0.0;
    double phaseQuadrafoil0 = 0.0;
    double phaseQuadrafoil45 = 0.0;
    double phaseSecondaryComaX = 0.0;
    double phaseSecondaryComaY = 0.0;
    double pupilDecenterX = 0.0;
    double pupilDecenterY = 0.0;
    double phaseFieldStrength = 0.0;
    double phaseFieldEdgeBias = 0.0;
    double phaseFieldDefocus = 0.0;
    double phaseFieldAstigRadial = 0.0;
    double phaseFieldAstigTangential = 0.0;
    double phaseFieldComaRadial = 0.0;
    double phaseFieldComaTangential = 0.0;
    double phaseFieldSpherical = 0.0;
    double phaseFieldTrefoilRadial = 0.0;
    double phaseFieldTrefoilTangential = 0.0;
    double phaseFieldSecondaryAstigRadial = 0.0;
    double phaseFieldSecondaryAstigTangential = 0.0;
    double phaseFieldQuadrafoilRadial = 0.0;
    double phaseFieldQuadrafoilTangential = 0.0;
    double phaseFieldSecondaryComaRadial = 0.0;
    double phaseFieldSecondaryComaTangential = 0.0;

    double spectrumForce = 0.0;
    double spectrumSaturation = 1.0;
    bool chromaticAffectsLuma = false;
    double chromaticFocus = 0.0;
    double chromaticSpread = 0.0;
    double creativeFringe = 0.0;
    double scatterAmount = 0.0;
    double scatterRadius = 0.0;

    double effectGain = 1.0;
    double coreCompensation = 1.0;
    double anisotropyEmphasis = 0.0;
    double coreGain = 1.0;
    double structureGain = 1.0;
    double coreShoulder = 0.0;
    double structureShoulder = 0.0;
};

struct LensDiffSpectrumConfig {
    int binCount = 0;
    std::array<float, kLensDiffMaxSpectralBins * 3> naturalMatrix {};
    std::array<float, kLensDiffMaxSpectralBins * 3> styleMatrix {};
};

struct LensDiffPsfBankKey {
    LensDiffApertureMode apertureMode = LensDiffApertureMode::Circle;
    LensDiffApodizationMode apodizationMode = LensDiffApodizationMode::Flat;
    LensDiffSpectralMode spectralMode = LensDiffSpectralMode::Mono;
    int bladeCount = 0;
    int vaneCount = 0;
    int pupilResolution = 0;
    int frameShortSidePx = 0;
    std::string customAperturePath;
    bool customApertureNormalize = true;
    bool customApertureInvert = false;
    double roundness = 0.0;
    double rotationDeg = 0.0;
    double centralObstruction = 0.0;
    double vaneThickness = 0.0;
    double maxKernelRadiusPx = 0.0;
    double diffractionScalePx = 0.0;
    double anisotropyEmphasis = 0.0;
    double phaseDefocus = 0.0;
    double phaseAstigmatism0 = 0.0;
    double phaseAstigmatism45 = 0.0;
    double phaseComaX = 0.0;
    double phaseComaY = 0.0;
    double phaseSpherical = 0.0;
    double phaseTrefoilX = 0.0;
    double phaseTrefoilY = 0.0;
    double phaseSecondaryAstigmatism0 = 0.0;
    double phaseSecondaryAstigmatism45 = 0.0;
    double phaseQuadrafoil0 = 0.0;
    double phaseQuadrafoil45 = 0.0;
    double phaseSecondaryComaX = 0.0;
    double phaseSecondaryComaY = 0.0;
    double pupilDecenterX = 0.0;
    double pupilDecenterY = 0.0;
    double chromaticFocus = 0.0;
    double chromaticSpread = 0.0;

    bool operator==(const LensDiffPsfBankKey& other) const {
        return apertureMode == other.apertureMode &&
               apodizationMode == other.apodizationMode &&
               spectralMode == other.spectralMode &&
               bladeCount == other.bladeCount &&
               vaneCount == other.vaneCount &&
               pupilResolution == other.pupilResolution &&
               frameShortSidePx == other.frameShortSidePx &&
               customAperturePath == other.customAperturePath &&
               customApertureNormalize == other.customApertureNormalize &&
               customApertureInvert == other.customApertureInvert &&
               roundness == other.roundness &&
               rotationDeg == other.rotationDeg &&
               centralObstruction == other.centralObstruction &&
               vaneThickness == other.vaneThickness &&
               maxKernelRadiusPx == other.maxKernelRadiusPx &&
               diffractionScalePx == other.diffractionScalePx &&
               anisotropyEmphasis == other.anisotropyEmphasis &&
               phaseDefocus == other.phaseDefocus &&
               phaseAstigmatism0 == other.phaseAstigmatism0 &&
               phaseAstigmatism45 == other.phaseAstigmatism45 &&
               phaseComaX == other.phaseComaX &&
               phaseComaY == other.phaseComaY &&
               phaseSpherical == other.phaseSpherical &&
               phaseTrefoilX == other.phaseTrefoilX &&
               phaseTrefoilY == other.phaseTrefoilY &&
               phaseSecondaryAstigmatism0 == other.phaseSecondaryAstigmatism0 &&
               phaseSecondaryAstigmatism45 == other.phaseSecondaryAstigmatism45 &&
               phaseQuadrafoil0 == other.phaseQuadrafoil0 &&
               phaseQuadrafoil45 == other.phaseQuadrafoil45 &&
               phaseSecondaryComaX == other.phaseSecondaryComaX &&
               phaseSecondaryComaY == other.phaseSecondaryComaY &&
               pupilDecenterX == other.pupilDecenterX &&
               pupilDecenterY == other.pupilDecenterY &&
               chromaticFocus == other.chromaticFocus &&
               chromaticSpread == other.chromaticSpread;
    }
};

constexpr int kLensDiffResolutionAwareReferenceShortSidePx = 2160;

inline int ResolveLensDiffFrameShortSidePx(const LensDiffParams& params) {
    return std::max(1, params.frameShortSidePx);
}

inline int ResolveLensDiffOpticsShortSidePx(const LensDiffParams& params) {
    return params.resolutionAware ? kLensDiffResolutionAwareReferenceShortSidePx
                                  : ResolveLensDiffFrameShortSidePx(params);
}

inline double ResolveLensDiffEffectWorkingScale(const LensDiffParams& params) {
    return static_cast<double>(ResolveLensDiffOpticsShortSidePx(params)) /
           static_cast<double>(ResolveLensDiffFrameShortSidePx(params));
}

inline double ResolveLensDiffAuthoringPercentToPixels(double percent, int frameShortSidePx) {
    return std::max(0.0, percent) * 0.01 * static_cast<double>(std::max(1, frameShortSidePx));
}

inline double ResolveLensDiffDiffractionScalePx(const LensDiffParams& params) {
    return ResolveLensDiffAuthoringPercentToPixels(params.diffractionScalePx, ResolveLensDiffOpticsShortSidePx(params));
}

inline int ResolveLensDiffMaxKernelRadiusPx(const LensDiffParams& params) {
    return std::max(0, static_cast<int>(std::ceil(
                            ResolveLensDiffAuthoringPercentToPixels(params.maxKernelRadiusPx,
                                                                   ResolveLensDiffOpticsShortSidePx(params)))));
}

inline double ResolveLensDiffCreativeFringePx(const LensDiffParams& params) {
    return ResolveLensDiffAuthoringPercentToPixels(params.creativeFringe, ResolveLensDiffOpticsShortSidePx(params));
}

inline double ResolveLensDiffScatterRadiusPx(const LensDiffParams& params) {
    return ResolveLensDiffAuthoringPercentToPixels(params.scatterRadius, ResolveLensDiffOpticsShortSidePx(params));
}

struct LensDiffFieldKey {
    double phaseFieldStrength = 0.0;
    double phaseFieldEdgeBias = 0.0;
    double phaseFieldDefocus = 0.0;
    double phaseFieldAstigRadial = 0.0;
    double phaseFieldAstigTangential = 0.0;
    double phaseFieldComaRadial = 0.0;
    double phaseFieldComaTangential = 0.0;
    double phaseFieldSpherical = 0.0;
    double phaseFieldTrefoilRadial = 0.0;
    double phaseFieldTrefoilTangential = 0.0;
    double phaseFieldSecondaryAstigRadial = 0.0;
    double phaseFieldSecondaryAstigTangential = 0.0;
    double phaseFieldQuadrafoilRadial = 0.0;
    double phaseFieldQuadrafoilTangential = 0.0;
    double phaseFieldSecondaryComaRadial = 0.0;
    double phaseFieldSecondaryComaTangential = 0.0;

    bool operator==(const LensDiffFieldKey& other) const {
        return phaseFieldStrength == other.phaseFieldStrength &&
               phaseFieldEdgeBias == other.phaseFieldEdgeBias &&
               phaseFieldDefocus == other.phaseFieldDefocus &&
               phaseFieldAstigRadial == other.phaseFieldAstigRadial &&
               phaseFieldAstigTangential == other.phaseFieldAstigTangential &&
               phaseFieldComaRadial == other.phaseFieldComaRadial &&
               phaseFieldComaTangential == other.phaseFieldComaTangential &&
               phaseFieldSpherical == other.phaseFieldSpherical &&
               phaseFieldTrefoilRadial == other.phaseFieldTrefoilRadial &&
               phaseFieldTrefoilTangential == other.phaseFieldTrefoilTangential &&
               phaseFieldSecondaryAstigRadial == other.phaseFieldSecondaryAstigRadial &&
               phaseFieldSecondaryAstigTangential == other.phaseFieldSecondaryAstigTangential &&
               phaseFieldQuadrafoilRadial == other.phaseFieldQuadrafoilRadial &&
               phaseFieldQuadrafoilTangential == other.phaseFieldQuadrafoilTangential &&
               phaseFieldSecondaryComaRadial == other.phaseFieldSecondaryComaRadial &&
               phaseFieldSecondaryComaTangential == other.phaseFieldSecondaryComaTangential;
    }
};

struct LensDiffKernel {
    int size = 0;
    std::vector<float> values;
};

struct LensDiffPsfBin {
    float wavelengthNm = 550.0f;
    LensDiffKernel full {};
    LensDiffKernel core {};
    LensDiffKernel structure {};
};

struct LensDiffFieldZoneCache {
    int zoneX = 0;
    int zoneY = 0;
    float normalizedX = 0.0f;
    float normalizedY = 0.0f;
    float radialNorm = 0.0f;
    LensDiffPsfBankKey key {};
    LensDiffParams resolvedParams {};
    std::vector<LensDiffPsfBin> bins;
    int supportRadiusPx = 0;
    int pupilDisplaySize = 0;
    std::vector<float> pupilDisplay;
    int phaseDisplaySize = 0;
    std::vector<float> phaseDisplay;
};

struct LensDiffPsfBankCache {
    bool valid = false;
    LensDiffPsfBankKey key {};
    LensDiffFieldKey fieldKey {};
    std::vector<LensDiffPsfBin> bins;
    int supportRadiusPx = 0;
    int pupilDisplaySize = 0;
    std::vector<float> pupilDisplay;
    int phaseDisplaySize = 0;
    std::vector<float> phaseDisplay;
    int fieldGridSize = 0;
    std::vector<LensDiffFieldZoneCache> fieldZones;
    int staticDebugView = -1;
    int staticDebugWidth = 0;
    int staticDebugHeight = 0;
    std::vector<float> staticDebugPixels;
};

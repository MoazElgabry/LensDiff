#include "ofxsImageEffect.h"
#include "ofxsSupportPrivate.h"

#include "core/LensDiffApertureImage.h"
#include "core/LensDiffFileDialog.h"
#include "core/LensDiffCpuReference.h"
#include "core/LensDiffDiagnostics.h"
#include "core/LensDiffPhase.h"
#include "core/LensDiffTypes.h"
#include "cuda/LensDiffCuda.h"
#ifdef __APPLE__
#include "metal/LensDiffMetal.h"
#endif
#include "ofxsLog.h"

#include <algorithm>
#include <chrono>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <spawn.h>
extern char** environ;
#endif

namespace {

using namespace OFX;

constexpr const char* kPluginName = "LensDiff";
constexpr const char* kPluginGrouping = "Moaz Elgabry";
constexpr const char* kPluginDescription =
    "FFT-driven lens diffraction effect with pupil-derived PSF synthesis, policy-based highlight selection, and a CPU reference core.";
constexpr const char* kPluginIdentifier = "com.moazelgabry.LensDiff";
constexpr int kPluginVersionMajor = 0;
constexpr int kPluginVersionMinor = 2;
constexpr const char* kPluginVersionLabel = "v0.2.7";
constexpr const char* kPluginDisplayVersion = "0.2.7";
constexpr const char* kWebsiteUrl = "https://moazelgabry.com";

// LensDiff's FFT convolution depends on neighboring pixels and stable frame geometry.
// Advertising tile support lets hosts submit partial image bounds that are not a
// complete, deterministic input for the current render path.
constexpr bool kSupportsTiles = false;
constexpr bool kSupportsMultiResolution = true;
constexpr bool kSupportsMultipleClipPARs = false;
constexpr int kDefaultApertureChoiceIndex = 1;
constexpr int kLastBuiltinApertureChoiceIndex = 6;
constexpr int kCustomApertureChoiceIndex = 7;
constexpr const char* kStaticCustomApertureOptionLabel = "Custom";
constexpr std::array<int, 5> kPupilResolutionOptions = {64, 128, 256, 512, 1024};

const std::array<const char*, 7> kBuiltinApertureOptionLabels = {
    "Circle",
    "Polygon",
    "Petals",
    "Spiral",
    "Hexagon",
    "Square Grid",
    "Snowflake",
};

constexpr const char* kLensDiffPresetDefaultName = "Default";
constexpr const char* kLensDiffPresetCustomLabel = "(Custom)";

void openUrl(const std::string& url) {
#if defined(_WIN32)
    ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#elif defined(__APPLE__)
    pid_t pid = 0;
    const char* argv[] = {"open", url.c_str(), nullptr};
    posix_spawnp(&pid, "open", nullptr, nullptr, const_cast<char* const*>(argv), environ);
#else
    pid_t pid = 0;
    const char* argv[] = {"xdg-open", url.c_str(), nullptr};
    posix_spawnp(&pid, "xdg-open", nullptr, nullptr, const_cast<char* const*>(argv), environ);
#endif
}

struct LensDiffPresetValues {
    bool simpleMode = false;
    bool resolutionAware = true;
    int inputTransfer = 0;
    double threshold = 4.0;
    double softnessStops = 1.0;
    int extractionMode = 0;
    double pointEmphasis = 0.0;
    double corePreserve = 0.1;

    int apertureMode = static_cast<int>(LensDiffApertureMode::Polygon);
    std::string customAperturePath;
    bool customApertureNormalize = true;
    bool customApertureInvert = false;
    int bladeCount = 6;
    double roundness = 0.5;
    double rotationDeg = 0.0;
    double centralObstruction = 0.0;
    int vaneCount = 0;
    double vaneThickness = 0.01;
    int apodizationMode = 0;
    double diffractionScalePx = 24.0;
    int pupilResolution = 256;
    double maxKernelRadiusPx = 64.0;
    bool phaseEnabled = false;
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

    int spectralMode = 0;
    int spectrumStyle = 0;
    double spectrumForce = 0.0;
    double spectrumSaturation = 1.0;
    bool chromaticAffectsLuma = false;
    double chromaticFocus = 0.0;
    double chromaticSpread = 0.0;
    double creativeFringe = 0.0;
    double scatterAmount = 0.0;
    double scatterRadius = 0.0;

    int lookMode = 1;
    int energyMode = 0;
    double effectGain = 1.0;
    double coreCompensation = 1.0;
    double anisotropyEmphasis = 0.0;
    double coreGain = 1.0;
    double structureGain = 1.0;
    double coreShoulder = 0.0;
    double structureShoulder = 0.0;
    int backendPreference = 0;
    int debugView = 0;
};

struct LensDiffUserPreset {
    std::string id;
    std::string name;
    std::string createdAtUtc;
    std::string updatedAtUtc;
    LensDiffPresetValues values {};
};

struct LensDiffPresetStore {
    bool loaded = false;
    LensDiffUserPreset defaultPreset {};
    std::vector<LensDiffUserPreset> userPresets;
};

struct LensDiffBuiltinPreset {
    const char* name = "";
    LensDiffPresetValues values {};
};

const std::array<LensDiffBuiltinPreset, 6>& lensdiffBuiltinPresets();

LensDiffPresetStore& lensdiffPresetStore() {
    static LensDiffPresetStore store;
    return store;
}

std::mutex& lensdiffPresetMutex() {
    static std::mutex mutex;
    return mutex;
}

class BoolScope {
public:
    explicit BoolScope(bool& value) : value_(value) { value_ = true; }
    ~BoolScope() { value_ = false; }

private:
    bool& value_;
};

template <typename T>
bool approxEqual(T a, T b, T epsilon) {
    return std::abs(a - b) <= epsilon;
}

constexpr double kLensDiffPi = 3.14159265358979323846;
constexpr double kLensDiffDefocusNormalization = 1.7320508075688772;
constexpr double kLensDiffAstigNormalization = 2.4494897427831781;
constexpr double kLensDiffComaNormalization = 2.8284271247461901;
constexpr double kLensDiffSphericalNormalization = 2.2360679774997898;
double lensdiffAuthoringPercentToPixels(double percent, int shortSidePx) {
    return std::max(0.0, percent) * 0.01 * static_cast<double>(std::max(1, shortSidePx));
}

int lensdiffAuthoringRadiusPercentToPixels(double percent, int shortSidePx) {
    return std::max(0, static_cast<int>(std::ceil(lensdiffAuthoringPercentToPixels(percent, shortSidePx))));
}

int lensdiffShortSidePx(const LensDiffImageRect& rect) {
    return std::max(1, std::min(rect.width(), rect.height()));
}

int lensdiffShortSidePx(const OfxRectI& rect) {
    return std::max(1, std::min(rect.x2 - rect.x1, rect.y2 - rect.y1));
}

int lensdiffShortSidePx(const OfxRectD& rect) {
    const double width = std::max(0.0, rect.x2 - rect.x1);
    const double height = std::max(0.0, rect.y2 - rect.y1);
    return std::max(1, static_cast<int>(std::ceil(std::min(width, height))));
}

int lensdiffShortSidePx(const OfxRectD& rect, const OfxPointD& renderScale) {
    const double width = std::max(0.0, rect.x2 - rect.x1) * std::max(0.0, renderScale.x);
    const double height = std::max(0.0, rect.y2 - rect.y1) * std::max(0.0, renderScale.y);
    return std::max(1, static_cast<int>(std::ceil(std::min(width, height))));
}

LensDiffImageRect lensdiffScaledImageRect(const OfxRectD& rect, const OfxPointD& renderScale) {
    const double scaleX = std::max(0.0, renderScale.x);
    const double scaleY = std::max(0.0, renderScale.y);
    return {
        static_cast<int>(std::floor(rect.x1 * scaleX)),
        static_cast<int>(std::floor(rect.y1 * scaleY)),
        static_cast<int>(std::ceil(rect.x2 * scaleX)),
        static_cast<int>(std::ceil(rect.y2 * scaleY)),
    };
}

double normalizePhaseCoefficient(double value, double normalization) {
    return normalization != 0.0 ? (value / normalization) : value;
}

double phasePairMagnitude(double x, double y) {
    return std::sqrt(x * x + y * y);
}

double phasePairAngleDeg(double x, double y, int harmonic) {
    if (harmonic == 0) return 0.0;
    if (std::abs(x) <= 1e-12 && std::abs(y) <= 1e-12) return 0.0;
    return std::atan2(y, x) * (180.0 / kLensDiffPi) / static_cast<double>(harmonic);
}

void resolveSimplePhasePair(double magnitude,
                            double angleDeg,
                            int harmonic,
                            double* outX,
                            double* outY) {
    if (!outX || !outY) return;
    const double theta = angleDeg * (kLensDiffPi / 180.0);
    const double phase = static_cast<double>(harmonic) * theta;
    *outX = magnitude * std::cos(phase);
    *outY = magnitude * std::sin(phase);
}

bool lensdiffNeedsCpuOnlyPhaseSuite(const LensDiffParams& params) {
    (void)params;
    return false;
}

std::string normalizePresetNameKey(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    bool inWhitespace = false;
    for (unsigned char uc : in) {
        if (std::isspace(uc)) {
            inWhitespace = true;
            continue;
        }
        if (inWhitespace && !out.empty()) out.push_back(' ');
        inWhitespace = false;
        out.push_back(static_cast<char>(std::tolower(uc)));
    }
    while (!out.empty() && out.front() == ' ') out.erase(out.begin());
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

int lensdiffBuiltinPresetIndexByName(const std::string& name) {
    const std::string key = normalizePresetNameKey(name);
    if (key.empty()) return -1;
    const auto& presets = lensdiffBuiltinPresets();
    for (int i = 0; i < static_cast<int>(presets.size()); ++i) {
        if (normalizePresetNameKey(presets[static_cast<std::size_t>(i)].name) == key) return i;
    }
    return -1;
}

const LensDiffBuiltinPreset* lensdiffBuiltinPresetByIndex(int index) {
    const auto& presets = lensdiffBuiltinPresets();
    if (index < 0 || index >= static_cast<int>(presets.size())) return nullptr;
    return &presets[static_cast<std::size_t>(index)];
}

const LensDiffBuiltinPreset* lensdiffBuiltinPresetByName(const std::string& name) {
    return lensdiffBuiltinPresetByIndex(lensdiffBuiltinPresetIndexByName(name));
}

std::string lensdiffReservedPresetNameMessage(const std::string& name) {
    const std::string key = normalizePresetNameKey(name);
    if (key == "default") {
        return "The preset name 'Default' is reserved. Use 'Save Defaults' to overwrite the protected Default preset.";
    }
    if (const auto* builtin = lensdiffBuiltinPresetByName(name)) {
        return "The preset name '" + std::string(builtin->name) +
               "' is reserved by a built-in LensDiff preset. Choose another name.";
    }
    return "That preset name is reserved.";
}

std::string lensdiffMenuLabelForUserPresetName(const std::string& name) {
    if (lensdiffBuiltinPresetByName(name)) return name + " (User)";
    return name;
}

std::string sanitizePresetName(const std::string& in, const char* fallback) {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        if (c == '\n' || c == '\r' || c == '\t') continue;
        out.push_back(c);
    }
    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.front()))) out.erase(out.begin());
    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) out.pop_back();
    if (out.empty()) out = fallback ? std::string(fallback) : std::string("Preset");
    if (out.size() > 96) out.resize(96);
    return out;
}

std::string makePresetId(const std::string& prefix) {
    static std::atomic<unsigned long> counter{1};
    std::ostringstream os;
    os << prefix << '_' << std::time(nullptr) << '_' << counter.fetch_add(1, std::memory_order_relaxed);
    return os.str();
}

std::string nowUtcIso8601() {
    const std::time_t now = std::time(nullptr);
    std::tm tmUtc {};
#ifdef _WIN32
    gmtime_s(&tmUtc, &now);
#else
    gmtime_r(&now, &tmUtc);
#endif
    char buffer[32] = {};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tmUtc);
    return buffer;
}

std::filesystem::path lensdiffPresetDirPath() {
#ifdef _WIN32
    const char* base = std::getenv("APPDATA");
    if (!base || !*base) base = std::getenv("LOCALAPPDATA");
    if (base && *base) return std::filesystem::path(base) / "LensDiff";
#elif defined(__APPLE__)
    const char* home = std::getenv("HOME");
    if (home && *home) return std::filesystem::path(home) / "Library" / "Application Support" / "LensDiff";
#else
    const char* home = std::getenv("HOME");
    if (home && *home) return std::filesystem::path(home) / ".config" / "LensDiff";
#endif
    return std::filesystem::path(".");
}

std::filesystem::path lensdiffPresetFilePath() {
    return lensdiffPresetDirPath() / "presets_v1.json";
}

LensDiffPresetValues lensdiffFactoryPresetValues() {
    LensDiffPresetValues values {};
    values.resolutionAware = true;
    values.simpleMode = true;
    values.inputTransfer = 1;
    values.threshold = 4.0;
    values.softnessStops = 3.0;
    values.extractionMode = 0;
    values.pointEmphasis = 0.0;
    values.corePreserve = 0.8;

    values.apertureMode = static_cast<int>(LensDiffApertureMode::Polygon);
    values.customAperturePath.clear();
    values.customApertureNormalize = true;
    values.customApertureInvert = false;
    values.bladeCount = 5;
    values.roundness = 0.5;
    values.rotationDeg = 0.0;
    values.centralObstruction = 0.6;
    values.vaneCount = 4;
    values.vaneThickness = 0.055;
    values.apodizationMode = 0;
    values.diffractionScalePx = 6.94444444444444;
    values.pupilResolution = 600;
    values.maxKernelRadiusPx = 17.1296296296296;
    values.phaseEnabled = false;
    values.phaseDefocus = 0.0;
    values.phaseAstigmatism0 = 0.0;
    values.phaseAstigmatism45 = 0.0;
    values.phaseComaX = 0.0;
    values.phaseComaY = 0.0;
    values.phaseSpherical = 0.0;
    values.phaseTrefoilX = 0.0;
    values.phaseTrefoilY = 0.0;
    values.phaseSecondaryAstigmatism0 = 0.0;
    values.phaseSecondaryAstigmatism45 = 0.0;
    values.phaseQuadrafoil0 = 0.0;
    values.phaseQuadrafoil45 = 0.0;
    values.phaseSecondaryComaX = 0.0;
    values.phaseSecondaryComaY = 0.0;
    values.pupilDecenterX = 0.0;
    values.pupilDecenterY = 0.0;
    values.phaseFieldStrength = 0.0;
    values.phaseFieldEdgeBias = 0.0;
    values.phaseFieldDefocus = 0.0;
    values.phaseFieldAstigRadial = 0.0;
    values.phaseFieldAstigTangential = 0.0;
    values.phaseFieldComaRadial = 0.0;
    values.phaseFieldComaTangential = 0.0;
    values.phaseFieldSpherical = 0.0;
    values.phaseFieldTrefoilRadial = 0.0;
    values.phaseFieldTrefoilTangential = 0.0;
    values.phaseFieldSecondaryAstigRadial = 0.0;
    values.phaseFieldSecondaryAstigTangential = 0.0;
    values.phaseFieldQuadrafoilRadial = 0.0;
    values.phaseFieldQuadrafoilTangential = 0.0;
    values.phaseFieldSecondaryComaRadial = 0.0;
    values.phaseFieldSecondaryComaTangential = 0.0;

    values.spectralMode = static_cast<int>(LensDiffSpectralMode::Mono);
    values.spectrumStyle = 1;
    values.spectrumForce = 0.7;
    values.spectrumSaturation = 1.0;
    values.chromaticAffectsLuma = true;
    values.chromaticFocus = 0.0;
    values.chromaticSpread = 0.0;
    values.creativeFringe = 0.0;
    values.scatterAmount = 0.0;
    values.scatterRadius = 0.740740740740741;

    values.lookMode = 0;
    values.energyMode = 0;
    values.effectGain = 1.0;
    values.coreCompensation = 1.0;
    values.anisotropyEmphasis = 0.0;
    values.coreGain = 1.0;
    values.structureGain = 1.0;
    values.coreShoulder = 0.0;
    values.structureShoulder = 0.0;
    values.backendPreference = 0;
    values.debugView = 0;
    return values;
}

LensDiffPresetValues lensdiffStarDiffPresetValues() {
    LensDiffPresetValues values = lensdiffFactoryPresetValues();
    values.threshold = 5.0;
    values.softnessStops = 2.99975824356079;
    values.corePreserve = 0.604651153087616;
    values.apertureMode = 1;
    values.bladeCount = 16;
    values.roundness = 0.0;
    values.rotationDeg = 60.0;
    values.centralObstruction = 0.0;
    values.vaneCount = 2;
    values.vaneThickness = 0.0945736467838287;
    values.diffractionScalePx = 11.8518518518519;
    values.pupilResolution = 674;
    values.maxKernelRadiusPx = 16.6666666666667;
    values.spectralMode = 2;
    values.spectrumStyle = 1;
    values.spectrumForce = 0.596899211406708;
    values.spectrumSaturation = 0.9;
    values.chromaticAffectsLuma = true;
    values.lookMode = 0;
    values.energyMode = 0;
    values.effectGain = 0.961240291595459;
    values.coreCompensation = 1.0;
    values.anisotropyEmphasis = 0.0;
    values.coreGain = 1.0;
    values.structureGain = 1.0;
    values.coreShoulder = 0.0;
    values.structureShoulder = 0.0;
    values.backendPreference = 0;
    values.debugView = 0;
    return values;
}

LensDiffPresetValues lensdiffIrisPresetValues() {
    LensDiffPresetValues values = lensdiffFactoryPresetValues();
    values.threshold = 5.0;
    values.softnessStops = 2.99975824356079;
    values.corePreserve = 0.78294575214386;
    values.apertureMode = 1;
    values.bladeCount = 7;
    values.roundness = 0.22480620443821;
    values.rotationDeg = -113.023254394531;
    values.centralObstruction = 0.0810077488422394;
    values.vaneCount = 4;
    values.vaneThickness = 0.0945736467838287;
    values.diffractionScalePx = 11.8518518518519;
    values.pupilResolution = 592;
    values.maxKernelRadiusPx = 16.6666666666667;
    values.spectralMode = 2;
    values.spectrumStyle = 1;
    values.spectrumForce = 0.937984466552734;
    values.spectrumSaturation = 0.0;
    values.chromaticAffectsLuma = true;
    values.lookMode = 1;
    values.energyMode = 1;
    values.effectGain = 0.217054262757301;
    values.coreCompensation = 1.0;
    values.anisotropyEmphasis = 0.0;
    values.coreGain = 1.79844963550568;
    values.structureGain = 0.527131795883179;
    values.coreShoulder = 0.0;
    values.structureShoulder = 0.0;
    values.backendPreference = 0;
    values.debugView = 0;
    return values;
}

LensDiffPresetValues lensdiffArrowHeadPresetValues() {
    LensDiffPresetValues values = lensdiffFactoryPresetValues();
    values.threshold = 5.0;
    values.softnessStops = 6.0;
    values.pointEmphasis = 1.0;
    values.corePreserve = 0.550387620925903;
    values.apertureMode = 1;
    values.bladeCount = 6;
    values.roundness = 0.581395328044891;
    values.rotationDeg = 12.5581398010254;
    values.centralObstruction = 0.405038744211197;
    values.vaneCount = 2;
    values.vaneThickness = 0.0193798448890448;
    values.apodizationMode = 1;
    values.diffractionScalePx = 3.28452498824508;
    values.pupilResolution = 592;
    values.maxKernelRadiusPx = 19.1666666666667;
    values.phaseEnabled = true;
    values.phaseDefocus = normalizePhaseCoefficient(-0.387596905231476, kLensDiffDefocusNormalization);
    values.phaseAstigmatism0 = normalizePhaseCoefficient(-2.0, kLensDiffAstigNormalization);
    values.phaseAstigmatism45 = normalizePhaseCoefficient(2.0, kLensDiffAstigNormalization);
    values.phaseComaX = normalizePhaseCoefficient(1.00775194168091, kLensDiffComaNormalization);
    values.phaseComaY = normalizePhaseCoefficient(-0.69767439365387, kLensDiffComaNormalization);
    values.phaseSpherical = normalizePhaseCoefficient(0.976744174957275, kLensDiffSphericalNormalization);
    values.spectralMode = 2;
    values.spectrumStyle = 2;
    values.spectrumForce = 0.844961225986481;
    values.spectrumSaturation = 0.775193810462952;
    values.chromaticAffectsLuma = true;
    values.lookMode = 0;
    values.energyMode = 0;
    values.effectGain = 1.0;
    values.coreCompensation = 0.0;
    values.anisotropyEmphasis = 0.0;
    values.coreGain = 1.0;
    values.structureGain = 1.0;
    values.coreShoulder = 0.0;
    values.structureShoulder = 0.0;
    values.backendPreference = 0;
    values.debugView = 0;
    return values;
}

LensDiffPresetValues lensdiffSnowflakePresetValues() {
    LensDiffPresetValues values = lensdiffFactoryPresetValues();
    values.threshold = 1.5;
    values.softnessStops = 5.0;
    values.pointEmphasis = 1.0;
    values.corePreserve = 0.79990541934967;
    values.apertureMode = static_cast<int>(LensDiffApertureMode::Snowflake);
    values.bladeCount = 4;
    values.roundness = 0.0;
    values.rotationDeg = 6.97674417495728;
    values.centralObstruction = 0.154651165008545;
    values.vaneCount = 1;
    values.vaneThickness = 0.077519379556179;
    values.apodizationMode = 0;
    values.diffractionScalePx = 4.743037753635;
    values.pupilResolution = 533;
    values.maxKernelRadiusPx = 20.0462962962963;
    values.phaseEnabled = true;
    values.phaseFieldStrength = 0.0;
    values.phaseFieldEdgeBias = 0.0;
    values.phaseFieldDefocus = 0.387596905231476;
    values.phaseFieldAstigRadial = 0.666666686534882;
    values.phaseFieldAstigTangential = 0.0;
    values.phaseFieldComaRadial = 0.0;
    values.phaseFieldComaTangential = 0.0;
    values.phaseFieldSpherical = 0.0;
    values.phaseFieldTrefoilRadial = 0.0;
    values.phaseFieldTrefoilTangential = 0.0;
    values.phaseFieldSecondaryAstigRadial = 0.0;
    values.phaseFieldSecondaryAstigTangential = 0.0;
    values.phaseFieldQuadrafoilRadial = 0.0;
    values.phaseFieldQuadrafoilTangential = 0.0;
    values.phaseFieldSecondaryComaRadial = 0.0;
    values.phaseFieldSecondaryComaTangential = 0.0;
    values.spectralMode = 3;
    values.spectrumStyle = 1;
    values.spectrumForce = 1.0;
    values.spectrumSaturation = 0.852713167667389;
    values.chromaticAffectsLuma = true;
    values.creativeFringe = 0.103359178260521;
    values.scatterAmount = 0.232558146119118;
    values.scatterRadius = 5.51248938949018;
    values.lookMode = 0;
    values.energyMode = 0;
    values.effectGain = 0.649999976158142;
    values.coreCompensation = 1.0;
    values.anisotropyEmphasis = 0.5;
    values.coreGain = 1.0;
    values.structureGain = 1.0;
    values.coreShoulder = 0.0;
    values.structureShoulder = 0.0;
    values.backendPreference = 0;
    values.debugView = 0;
    return values;
}

LensDiffPresetValues lensdiffFogyPresetValues() {
    LensDiffPresetValues values = lensdiffFactoryPresetValues();
    values.threshold = 6.00356578826904;
    values.softnessStops = 3.5;
    values.pointEmphasis = 0.0;
    values.corePreserve = 0.79990541934967;
    values.apertureMode = static_cast<int>(LensDiffApertureMode::Spiral);
    values.bladeCount = 3;
    values.roundness = 0.5;
    values.rotationDeg = 60.0;
    values.centralObstruction = 0.600340306758881;
    values.vaneCount = 4;
    values.vaneThickness = 0.0545635670423508;
    values.apodizationMode = 0;
    values.diffractionScalePx = 23.7037037037037;
    values.pupilResolution = 845;
    values.maxKernelRadiusPx = 47.4074074074074;
    values.phaseEnabled = true;
    values.phaseDefocus = 0.170542642474174;
    values.spectralMode = 0;
    values.spectrumStyle = 1;
    values.spectrumForce = 0.69981861114502;
    values.spectrumSaturation = 1.0;
    values.chromaticAffectsLuma = true;
    values.creativeFringe = 0.0;
    values.scatterAmount = 0.806201577186584;
    values.scatterRadius = 8.81998274061417;
    values.lookMode = 0;
    values.energyMode = 0;
    values.effectGain = 1.0;
    values.coreCompensation = 1.0;
    values.anisotropyEmphasis = 0.0;
    values.coreGain = 1.0;
    values.structureGain = 1.0;
    values.coreShoulder = 0.0;
    values.structureShoulder = 0.0;
    values.backendPreference = 0;
    values.debugView = 0;
    return values;
}

LensDiffPresetValues lensdiffImperfectionPresetValues() {
    LensDiffPresetValues values = lensdiffFactoryPresetValues();
    values.threshold = 4.0;
    values.softnessStops = 2.99975824356079;
    values.pointEmphasis = 1.0;
    values.corePreserve = 0.860465109348297;
    values.apertureMode = static_cast<int>(LensDiffApertureMode::SquareGrid);
    values.bladeCount = 9;
    values.roundness = 0.527131795883179;
    values.rotationDeg = 60.0;
    values.centralObstruction = 0.449224799871445;
    values.vaneCount = 4;
    values.vaneThickness = 0.0519379861652851;
    values.apodizationMode = 0;
    values.diffractionScalePx = 0.852031266247782;
    values.pupilResolution = 741;
    values.maxKernelRadiusPx = 22.5925925925926;
    values.phaseEnabled = true;
    values.phaseDefocus = 0.0;
    values.phaseAstigmatism0 = -0.356589138507843;
    values.phaseAstigmatism45 = -0.573643386363983;
    values.phaseComaX = 0.2945736348629;
    values.phaseComaY = 0.387596905231476;
    values.phaseSpherical = 0.0;
    values.phaseTrefoilX = -0.542635679244995;
    values.phaseTrefoilY = 0.573643386363983;
    values.phaseSecondaryAstigmatism0 = 0.821705400943756;
    values.phaseSecondaryAstigmatism45 = 0.0;
    values.phaseQuadrafoil0 = -1.65891468524933;
    values.phaseQuadrafoil45 = -1.56589150428772;
    values.phaseSecondaryComaX = 0.387596905231476;
    values.phaseSecondaryComaY = 0.883720934391022;
    values.pupilDecenterX = -0.0193798448890448;
    values.pupilDecenterY = -0.0348837226629257;
    values.phaseFieldStrength = 0.0;
    values.phaseFieldEdgeBias = 0.379844963550568;
    values.phaseFieldDefocus = 0.0;
    values.phaseFieldAstigRadial = 1.44186043739319;
    values.phaseFieldAstigTangential = 0.0;
    values.phaseFieldComaRadial = 0.0;
    values.phaseFieldComaTangential = 0.0;
    values.phaseFieldSpherical = 0.0;
    values.phaseFieldTrefoilRadial = 0.0;
    values.phaseFieldTrefoilTangential = 0.0;
    values.phaseFieldSecondaryAstigRadial = 0.0;
    values.phaseFieldSecondaryAstigTangential = 0.0;
    values.phaseFieldQuadrafoilRadial = 0.0;
    values.phaseFieldQuadrafoilTangential = 0.0;
    values.phaseFieldSecondaryComaRadial = 0.0;
    values.phaseFieldSecondaryComaTangential = 0.0;
    values.spectralMode = 3;
    values.spectrumStyle = 1;
    values.spectrumForce = 0.69981861114502;
    values.spectrumSaturation = 0.325581401586533;
    values.chromaticAffectsLuma = true;
    values.chromaticFocus = 0.0;
    values.chromaticSpread = 0.0;
    values.creativeFringe = 0.054550673122759;
    values.scatterAmount = 0.325581401586533;
    values.scatterRadius = 3.30749370433666;
    values.lookMode = 0;
    values.energyMode = 0;
    values.effectGain = 1.0;
    values.coreCompensation = 1.0;
    values.anisotropyEmphasis = 0.209302321076393;
    values.coreGain = 1.0;
    values.structureGain = 1.0;
    values.coreShoulder = 0.0;
    values.structureShoulder = 0.0;
    values.backendPreference = 0;
    values.debugView = 0;
    return values;
}

const std::array<LensDiffBuiltinPreset, 6>& lensdiffBuiltinPresets() {
    static const std::array<LensDiffBuiltinPreset, 6> presets = {{
        {"Star Diff", lensdiffStarDiffPresetValues()},
        {"Iris", lensdiffIrisPresetValues()},
        {"Arrow Head", lensdiffArrowHeadPresetValues()},
        {"Snowflake", lensdiffSnowflakePresetValues()},
        {"Fogy", lensdiffFogyPresetValues()},
        {"Imperfection", lensdiffImperfectionPresetValues()},
    }};
    return presets;
}

bool lensdiffPresetValuesEqual(const LensDiffPresetValues& a, const LensDiffPresetValues& b) {
    const bool phaseEqual =
        a.phaseEnabled == b.phaseEnabled &&
        (!a.phaseEnabled ||
         (approxEqual(a.phaseDefocus, b.phaseDefocus, 1e-6) &&
          approxEqual(a.phaseAstigmatism0, b.phaseAstigmatism0, 1e-6) &&
          approxEqual(a.phaseAstigmatism45, b.phaseAstigmatism45, 1e-6) &&
          approxEqual(a.phaseComaX, b.phaseComaX, 1e-6) &&
          approxEqual(a.phaseComaY, b.phaseComaY, 1e-6) &&
          approxEqual(a.phaseSpherical, b.phaseSpherical, 1e-6) &&
          approxEqual(a.phaseTrefoilX, b.phaseTrefoilX, 1e-6) &&
          approxEqual(a.phaseTrefoilY, b.phaseTrefoilY, 1e-6) &&
          approxEqual(a.phaseSecondaryAstigmatism0, b.phaseSecondaryAstigmatism0, 1e-6) &&
          approxEqual(a.phaseSecondaryAstigmatism45, b.phaseSecondaryAstigmatism45, 1e-6) &&
          approxEqual(a.phaseQuadrafoil0, b.phaseQuadrafoil0, 1e-6) &&
          approxEqual(a.phaseQuadrafoil45, b.phaseQuadrafoil45, 1e-6) &&
          approxEqual(a.phaseSecondaryComaX, b.phaseSecondaryComaX, 1e-6) &&
          approxEqual(a.phaseSecondaryComaY, b.phaseSecondaryComaY, 1e-6) &&
          approxEqual(a.pupilDecenterX, b.pupilDecenterX, 1e-6) &&
          approxEqual(a.pupilDecenterY, b.pupilDecenterY, 1e-6) &&
          approxEqual(a.phaseFieldStrength, b.phaseFieldStrength, 1e-6) &&
          approxEqual(a.phaseFieldEdgeBias, b.phaseFieldEdgeBias, 1e-6) &&
          approxEqual(a.phaseFieldDefocus, b.phaseFieldDefocus, 1e-6) &&
          approxEqual(a.phaseFieldAstigRadial, b.phaseFieldAstigRadial, 1e-6) &&
          approxEqual(a.phaseFieldAstigTangential, b.phaseFieldAstigTangential, 1e-6) &&
          approxEqual(a.phaseFieldComaRadial, b.phaseFieldComaRadial, 1e-6) &&
          approxEqual(a.phaseFieldComaTangential, b.phaseFieldComaTangential, 1e-6) &&
          approxEqual(a.phaseFieldSpherical, b.phaseFieldSpherical, 1e-6) &&
          approxEqual(a.phaseFieldTrefoilRadial, b.phaseFieldTrefoilRadial, 1e-6) &&
          approxEqual(a.phaseFieldTrefoilTangential, b.phaseFieldTrefoilTangential, 1e-6) &&
          approxEqual(a.phaseFieldSecondaryAstigRadial, b.phaseFieldSecondaryAstigRadial, 1e-6) &&
          approxEqual(a.phaseFieldSecondaryAstigTangential, b.phaseFieldSecondaryAstigTangential, 1e-6) &&
          approxEqual(a.phaseFieldQuadrafoilRadial, b.phaseFieldQuadrafoilRadial, 1e-6) &&
          approxEqual(a.phaseFieldQuadrafoilTangential, b.phaseFieldQuadrafoilTangential, 1e-6) &&
          approxEqual(a.phaseFieldSecondaryComaRadial, b.phaseFieldSecondaryComaRadial, 1e-6) &&
          approxEqual(a.phaseFieldSecondaryComaTangential, b.phaseFieldSecondaryComaTangential, 1e-6)));
    return a.inputTransfer == b.inputTransfer &&
           approxEqual(a.threshold, b.threshold, 1e-6) &&
           approxEqual(a.softnessStops, b.softnessStops, 1e-6) &&
           a.extractionMode == b.extractionMode &&
           approxEqual(a.pointEmphasis, b.pointEmphasis, 1e-6) &&
           approxEqual(a.corePreserve, b.corePreserve, 1e-6) &&
           a.apertureMode == b.apertureMode &&
           a.customAperturePath == b.customAperturePath &&
           a.customApertureNormalize == b.customApertureNormalize &&
           a.customApertureInvert == b.customApertureInvert &&
           a.bladeCount == b.bladeCount &&
           approxEqual(a.roundness, b.roundness, 1e-6) &&
           approxEqual(a.rotationDeg, b.rotationDeg, 1e-6) &&
           approxEqual(a.centralObstruction, b.centralObstruction, 1e-6) &&
           a.vaneCount == b.vaneCount &&
           approxEqual(a.vaneThickness, b.vaneThickness, 1e-6) &&
           a.apodizationMode == b.apodizationMode &&
           approxEqual(a.diffractionScalePx, b.diffractionScalePx, 1e-6) &&
           a.pupilResolution == b.pupilResolution &&
           approxEqual(a.maxKernelRadiusPx, b.maxKernelRadiusPx, 1e-6) &&
           phaseEqual &&
           a.spectralMode == b.spectralMode &&
           a.spectrumStyle == b.spectrumStyle &&
           approxEqual(a.spectrumForce, b.spectrumForce, 1e-6) &&
           approxEqual(a.spectrumSaturation, b.spectrumSaturation, 1e-6) &&
           a.chromaticAffectsLuma == b.chromaticAffectsLuma &&
           approxEqual(a.chromaticFocus, b.chromaticFocus, 1e-6) &&
           approxEqual(a.chromaticSpread, b.chromaticSpread, 1e-6) &&
           approxEqual(a.creativeFringe, b.creativeFringe, 1e-6) &&
           approxEqual(a.scatterAmount, b.scatterAmount, 1e-6) &&
           approxEqual(a.scatterRadius, b.scatterRadius, 1e-6) &&
           a.lookMode == b.lookMode &&
           a.energyMode == b.energyMode &&
           a.resolutionAware == b.resolutionAware &&
           approxEqual(a.effectGain, b.effectGain, 1e-6) &&
           approxEqual(a.coreCompensation, b.coreCompensation, 1e-6) &&
           approxEqual(a.anisotropyEmphasis, b.anisotropyEmphasis, 1e-6) &&
           approxEqual(a.coreGain, b.coreGain, 1e-6) &&
           approxEqual(a.structureGain, b.structureGain, 1e-6) &&
           approxEqual(a.coreShoulder, b.coreShoulder, 1e-6) &&
           approxEqual(a.structureShoulder, b.structureShoulder, 1e-6) &&
           a.backendPreference == b.backendPreference &&
           a.debugView == b.debugView;
}

LensDiffPresetValues lensdiffPresetValuesForDirtyComparison(LensDiffPresetValues values) {
    values.effectGain = 0.0;
    return values;
}

bool lensdiffPresetNameReserved(const std::string& name) {
    const std::string key = normalizePresetNameKey(name);
    return key == "default" || lensdiffBuiltinPresetIndexByName(name) >= 0;
}

bool lensdiffPresetUserNameExistsLocked(const std::string& name,
                                        const std::string* ignoreId = nullptr) {
    const std::string key = normalizePresetNameKey(name);
    if (key.empty() || key == "default") return key == "default";
    for (const auto& preset : lensdiffPresetStore().userPresets) {
        if (ignoreId && !ignoreId->empty() && preset.id == *ignoreId) continue;
        if (normalizePresetNameKey(preset.name) == key) return true;
    }
    return false;
}

int lensdiffUserPresetIndexByNameLocked(const std::string& name) {
    const std::string key = normalizePresetNameKey(name);
    const auto& presets = lensdiffPresetStore().userPresets;
    for (int i = 0; i < static_cast<int>(presets.size()); ++i) {
        if (normalizePresetNameKey(presets[static_cast<std::size_t>(i)].name) == key) return i;
    }
    return -1;
}

std::string jsonEscape(const std::string& s) {
    std::ostringstream os;
    for (char c : s) {
        switch (c) {
            case '\\': os << "\\\\"; break;
            case '"': os << "\\\""; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default: os << c; break;
        }
    }
    return os.str();
}

bool extractJsonStringField(const std::string& json, const std::string& key, std::string* out) {
    if (!out) return false;
    const std::string token = "\"" + key + "\":\"";
    const std::size_t start = json.find(token);
    if (start == std::string::npos) return false;
    bool escaped = false;
    std::string value;
    for (std::size_t i = start + token.size(); i < json.size(); ++i) {
        const char c = json[i];
        if (escaped) {
            switch (c) {
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                default: value.push_back(c); break;
            }
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            *out = value;
            return true;
        }
        value.push_back(c);
    }
    return false;
}

bool extractJsonBoolField(const std::string& json, const std::string& key, bool* out) {
    if (!out) return false;
    const std::string token = "\"" + key + "\":";
    const std::size_t start = json.find(token);
    if (start == std::string::npos) return false;
    std::size_t pos = start + token.size();
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (json.compare(pos, 4, "true") == 0) {
        *out = true;
        return true;
    }
    if (json.compare(pos, 5, "false") == 0) {
        *out = false;
        return true;
    }
    return false;
}

bool extractJsonNumberFieldText(const std::string& json, const std::string& key, std::string* out) {
    if (!out) return false;
    const std::string token = "\"" + key + "\":";
    const std::size_t start = json.find(token);
    if (start == std::string::npos) return false;
    std::size_t pos = start + token.size();
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    std::size_t end = pos;
    while (end < json.size()) {
        const char c = json[end];
        if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')) {
            break;
        }
        ++end;
    }
    if (end <= pos) return false;
    *out = json.substr(pos, end - pos);
    return true;
}

bool extractJsonIntField(const std::string& json, const std::string& key, int* out) {
    if (!out) return false;
    std::string text;
    if (!extractJsonNumberFieldText(json, key, &text)) return false;
    try {
        *out = std::stoi(text);
        return true;
    } catch (...) {
        return false;
    }
}

bool extractJsonDoubleField(const std::string& json, const std::string& key, double* out) {
    if (!out) return false;
    std::string text;
    if (!extractJsonNumberFieldText(json, key, &text)) return false;
    try {
        *out = std::stod(text);
        return true;
    } catch (...) {
        return false;
    }
}

bool extractJsonStructuredField(const std::string& json,
                                const std::string& key,
                                char openChar,
                                char closeChar,
                                std::string* out) {
    if (!out) return false;
    const std::string token = "\"" + key + "\":";
    const std::size_t start = json.find(token);
    if (start == std::string::npos) return false;
    std::size_t pos = start + token.size();
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size() || json[pos] != openChar) return false;
    int depth = 0;
    bool inString = false;
    bool escaped = false;
    std::size_t end = pos;
    for (; end < json.size(); ++end) {
        const char c = json[end];
        if (inString) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                inString = false;
            }
            continue;
        }
        if (c == '"') {
            inString = true;
            continue;
        }
        if (c == openChar) {
            ++depth;
        } else if (c == closeChar) {
            --depth;
            if (depth == 0) {
                *out = json.substr(pos, end - pos + 1);
                return true;
            }
        }
    }
    return false;
}

bool extractJsonObjectField(const std::string& json, const std::string& key, std::string* out) {
    return extractJsonStructuredField(json, key, '{', '}', out);
}

bool extractJsonArrayField(const std::string& json, const std::string& key, std::string* out) {
    return extractJsonStructuredField(json, key, '[', ']', out);
}

std::vector<std::string> extractJsonObjectsFromArray(const std::string& arrayJson) {
    std::vector<std::string> out;
    bool inString = false;
    bool escaped = false;
    int depth = 0;
    std::size_t objectStart = std::string::npos;
    for (std::size_t i = 0; i < arrayJson.size(); ++i) {
        const char c = arrayJson[i];
        if (inString) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                inString = false;
            }
            continue;
        }
        if (c == '"') {
            inString = true;
            continue;
        }
        if (c == '{') {
            if (depth == 0) objectStart = i;
            ++depth;
            continue;
        }
        if (c == '}') {
            --depth;
            if (depth == 0 && objectStart != std::string::npos) {
                out.push_back(arrayJson.substr(objectStart, i - objectStart + 1));
                objectStart = std::string::npos;
            }
        }
    }
    return out;
}

std::string lensdiffPresetValuesAsJson(const LensDiffPresetValues& values) {
    std::ostringstream os;
    os << "{";
    os << "\"simpleMode\":" << (values.simpleMode ? "true" : "false") << ",";
    os << "\"inputTransfer\":" << values.inputTransfer << ",";
    os << "\"resolutionAware\":" << (values.resolutionAware ? "true" : "false") << ",";
    os << "\"threshold\":" << std::setprecision(15) << values.threshold << ",";
    os << "\"softnessStops\":" << std::setprecision(15) << values.softnessStops << ",";
    os << "\"extractionMode\":" << values.extractionMode << ",";
    os << "\"pointEmphasis\":" << std::setprecision(15) << values.pointEmphasis << ",";
    os << "\"corePreserve\":" << std::setprecision(15) << values.corePreserve << ",";
    os << "\"apertureMode\":" << values.apertureMode << ",";
    os << "\"customAperturePath\":\"" << jsonEscape(values.customAperturePath) << "\",";
    os << "\"customApertureNormalize\":" << (values.customApertureNormalize ? "true" : "false") << ",";
    os << "\"customApertureInvert\":" << (values.customApertureInvert ? "true" : "false") << ",";
    os << "\"bladeCount\":" << values.bladeCount << ",";
    os << "\"roundness\":" << std::setprecision(15) << values.roundness << ",";
    os << "\"rotationDeg\":" << std::setprecision(15) << values.rotationDeg << ",";
    os << "\"centralObstruction\":" << std::setprecision(15) << values.centralObstruction << ",";
    os << "\"vaneCount\":" << values.vaneCount << ",";
    os << "\"vaneThickness\":" << std::setprecision(15) << values.vaneThickness << ",";
    os << "\"apodizationMode\":" << values.apodizationMode << ",";
    os << "\"diffractionScalePx\":" << std::setprecision(15) << values.diffractionScalePx << ",";
    os << "\"pupilResolution\":" << values.pupilResolution << ",";
    os << "\"maxKernelRadiusPx\":" << std::setprecision(15) << values.maxKernelRadiusPx << ",";
    os << "\"phaseEnabled\":" << (values.phaseEnabled ? "true" : "false") << ",";
    os << "\"phaseDefocus\":" << std::setprecision(15) << values.phaseDefocus << ",";
    os << "\"phaseAstigmatism0\":" << std::setprecision(15) << values.phaseAstigmatism0 << ",";
    os << "\"phaseAstigmatism45\":" << std::setprecision(15) << values.phaseAstigmatism45 << ",";
    os << "\"phaseComaX\":" << std::setprecision(15) << values.phaseComaX << ",";
    os << "\"phaseComaY\":" << std::setprecision(15) << values.phaseComaY << ",";
    os << "\"phaseSpherical\":" << std::setprecision(15) << values.phaseSpherical << ",";
    os << "\"phaseTrefoilX\":" << std::setprecision(15) << values.phaseTrefoilX << ",";
    os << "\"phaseTrefoilY\":" << std::setprecision(15) << values.phaseTrefoilY << ",";
    os << "\"phaseSecondaryAstigmatism0\":" << std::setprecision(15) << values.phaseSecondaryAstigmatism0 << ",";
    os << "\"phaseSecondaryAstigmatism45\":" << std::setprecision(15) << values.phaseSecondaryAstigmatism45 << ",";
    os << "\"phaseQuadrafoil0\":" << std::setprecision(15) << values.phaseQuadrafoil0 << ",";
    os << "\"phaseQuadrafoil45\":" << std::setprecision(15) << values.phaseQuadrafoil45 << ",";
    os << "\"phaseSecondaryComaX\":" << std::setprecision(15) << values.phaseSecondaryComaX << ",";
    os << "\"phaseSecondaryComaY\":" << std::setprecision(15) << values.phaseSecondaryComaY << ",";
    os << "\"pupilDecenterX\":" << std::setprecision(15) << values.pupilDecenterX << ",";
    os << "\"pupilDecenterY\":" << std::setprecision(15) << values.pupilDecenterY << ",";
    os << "\"phaseFieldStrength\":" << std::setprecision(15) << values.phaseFieldStrength << ",";
    os << "\"phaseFieldEdgeBias\":" << std::setprecision(15) << values.phaseFieldEdgeBias << ",";
    os << "\"phaseFieldDefocus\":" << std::setprecision(15) << values.phaseFieldDefocus << ",";
    os << "\"phaseFieldAstigRadial\":" << std::setprecision(15) << values.phaseFieldAstigRadial << ",";
    os << "\"phaseFieldAstigTangential\":" << std::setprecision(15) << values.phaseFieldAstigTangential << ",";
    os << "\"phaseFieldComaRadial\":" << std::setprecision(15) << values.phaseFieldComaRadial << ",";
    os << "\"phaseFieldComaTangential\":" << std::setprecision(15) << values.phaseFieldComaTangential << ",";
    os << "\"phaseFieldSpherical\":" << std::setprecision(15) << values.phaseFieldSpherical << ",";
    os << "\"phaseFieldTrefoilRadial\":" << std::setprecision(15) << values.phaseFieldTrefoilRadial << ",";
    os << "\"phaseFieldTrefoilTangential\":" << std::setprecision(15) << values.phaseFieldTrefoilTangential << ",";
    os << "\"phaseFieldSecondaryAstigRadial\":" << std::setprecision(15) << values.phaseFieldSecondaryAstigRadial << ",";
    os << "\"phaseFieldSecondaryAstigTangential\":" << std::setprecision(15) << values.phaseFieldSecondaryAstigTangential << ",";
    os << "\"phaseFieldQuadrafoilRadial\":" << std::setprecision(15) << values.phaseFieldQuadrafoilRadial << ",";
    os << "\"phaseFieldQuadrafoilTangential\":" << std::setprecision(15) << values.phaseFieldQuadrafoilTangential << ",";
    os << "\"phaseFieldSecondaryComaRadial\":" << std::setprecision(15) << values.phaseFieldSecondaryComaRadial << ",";
    os << "\"phaseFieldSecondaryComaTangential\":" << std::setprecision(15) << values.phaseFieldSecondaryComaTangential << ",";
    os << "\"spectralMode\":" << values.spectralMode << ",";
    os << "\"spectrumStyle\":" << values.spectrumStyle << ",";
    os << "\"spectrumForce\":" << std::setprecision(15) << values.spectrumForce << ",";
    os << "\"spectrumSaturation\":" << std::setprecision(15) << values.spectrumSaturation << ",";
    os << "\"chromaticAffectsLuma\":" << (values.chromaticAffectsLuma ? "true" : "false") << ",";
    os << "\"chromaticFocus\":" << std::setprecision(15) << values.chromaticFocus << ",";
    os << "\"chromaticSpread\":" << std::setprecision(15) << values.chromaticSpread << ",";
    os << "\"creativeFringe\":" << std::setprecision(15) << values.creativeFringe << ",";
    os << "\"scatterAmount\":" << std::setprecision(15) << values.scatterAmount << ",";
    os << "\"scatterRadius\":" << std::setprecision(15) << values.scatterRadius << ",";
    os << "\"lookMode\":" << values.lookMode << ",";
    os << "\"energyMode\":" << values.energyMode << ",";
    os << "\"effectGain\":" << std::setprecision(15) << values.effectGain << ",";
    os << "\"coreCompensation\":" << std::setprecision(15) << values.coreCompensation << ",";
    os << "\"anisotropyEmphasis\":" << std::setprecision(15) << values.anisotropyEmphasis << ",";
    os << "\"coreGain\":" << std::setprecision(15) << values.coreGain << ",";
    os << "\"structureGain\":" << std::setprecision(15) << values.structureGain << ",";
    os << "\"coreShoulder\":" << std::setprecision(15) << values.coreShoulder << ",";
    os << "\"structureShoulder\":" << std::setprecision(15) << values.structureShoulder << ",";
    os << "\"backendPreference\":" << values.backendPreference << ",";
    os << "\"debugView\":" << values.debugView;
    os << "}";
    return os.str();
}

bool parseLensDiffPresetValuesFromJson(const std::string& json, LensDiffPresetValues* out) {
    if (!out) return false;
    LensDiffPresetValues values = lensdiffFactoryPresetValues();
    (void)extractJsonBoolField(json, "simpleMode", &values.simpleMode);
    (void)extractJsonIntField(json, "inputTransfer", &values.inputTransfer);
    (void)extractJsonBoolField(json, "resolutionAware", &values.resolutionAware);
    (void)extractJsonDoubleField(json, "threshold", &values.threshold);
    (void)extractJsonDoubleField(json, "softnessStops", &values.softnessStops);
    (void)extractJsonIntField(json, "extractionMode", &values.extractionMode);
    (void)extractJsonDoubleField(json, "pointEmphasis", &values.pointEmphasis);
    (void)extractJsonDoubleField(json, "corePreserve", &values.corePreserve);
    (void)extractJsonIntField(json, "apertureMode", &values.apertureMode);
    (void)extractJsonStringField(json, "customAperturePath", &values.customAperturePath);
    (void)extractJsonBoolField(json, "customApertureNormalize", &values.customApertureNormalize);
    (void)extractJsonBoolField(json, "customApertureInvert", &values.customApertureInvert);
    (void)extractJsonIntField(json, "bladeCount", &values.bladeCount);
    (void)extractJsonDoubleField(json, "roundness", &values.roundness);
    (void)extractJsonDoubleField(json, "rotationDeg", &values.rotationDeg);
    (void)extractJsonDoubleField(json, "centralObstruction", &values.centralObstruction);
    (void)extractJsonIntField(json, "vaneCount", &values.vaneCount);
    (void)extractJsonDoubleField(json, "vaneThickness", &values.vaneThickness);
    (void)extractJsonIntField(json, "apodizationMode", &values.apodizationMode);
    (void)extractJsonDoubleField(json, "diffractionScalePx", &values.diffractionScalePx);
    if (!extractJsonIntField(json, "pupilResolution", &values.pupilResolution)) {
        int legacyPupilResolutionChoice = 2;
        if (extractJsonIntField(json, "pupilResolutionChoice", &legacyPupilResolutionChoice)) {
            if (legacyPupilResolutionChoice < 0 || legacyPupilResolutionChoice >= static_cast<int>(kPupilResolutionOptions.size())) {
                legacyPupilResolutionChoice = 2;
            }
            values.pupilResolution = kPupilResolutionOptions[static_cast<std::size_t>(legacyPupilResolutionChoice)];
        }
    }
    (void)extractJsonDoubleField(json, "maxKernelRadiusPx", &values.maxKernelRadiusPx);
    (void)extractJsonBoolField(json, "phaseEnabled", &values.phaseEnabled);
    int legacyPhaseMode = 0;
    if (!values.phaseEnabled && extractJsonIntField(json, "phaseMode", &legacyPhaseMode)) {
        if (legacyPhaseMode == 1) {
            values.phaseEnabled = true;
            double phaseFocus = 0.0;
            double phaseAstigmatism = 0.0;
            double phaseAstigAngleDeg = 0.0;
            double phaseComa = 0.0;
            double phaseComaAngleDeg = 0.0;
            double phaseSimpleSpherical = 0.0;
            (void)extractJsonDoubleField(json, "phaseFocus", &phaseFocus);
            (void)extractJsonDoubleField(json, "phaseAstigmatism", &phaseAstigmatism);
            (void)extractJsonDoubleField(json, "phaseAstigAngleDeg", &phaseAstigAngleDeg);
            (void)extractJsonDoubleField(json, "phaseComa", &phaseComa);
            (void)extractJsonDoubleField(json, "phaseComaAngleDeg", &phaseComaAngleDeg);
            (void)extractJsonDoubleField(json, "phaseSimpleSpherical", &phaseSimpleSpherical);
            values.phaseDefocus = phaseFocus;
            resolveSimplePhasePair(phaseAstigmatism, phaseAstigAngleDeg, 2, &values.phaseAstigmatism0, &values.phaseAstigmatism45);
            resolveSimplePhasePair(phaseComa, phaseComaAngleDeg, 1, &values.phaseComaX, &values.phaseComaY);
            values.phaseSpherical = phaseSimpleSpherical;
        } else if (legacyPhaseMode != 0) {
            // Legacy saved states used multiple phase modes. The live implementation keeps
            // one structured Zernike control set behind a single enable toggle.
            values.phaseEnabled = true;
        }
    }
    (void)extractJsonDoubleField(json, "phaseDefocus", &values.phaseDefocus);
    (void)extractJsonDoubleField(json, "phaseAstigmatism0", &values.phaseAstigmatism0);
    (void)extractJsonDoubleField(json, "phaseAstigmatism45", &values.phaseAstigmatism45);
    (void)extractJsonDoubleField(json, "phaseComaX", &values.phaseComaX);
    (void)extractJsonDoubleField(json, "phaseComaY", &values.phaseComaY);
    (void)extractJsonDoubleField(json, "phaseSpherical", &values.phaseSpherical);
    (void)extractJsonDoubleField(json, "phaseTrefoilX", &values.phaseTrefoilX);
    (void)extractJsonDoubleField(json, "phaseTrefoilY", &values.phaseTrefoilY);
    (void)extractJsonDoubleField(json, "phaseSecondaryAstigmatism0", &values.phaseSecondaryAstigmatism0);
    (void)extractJsonDoubleField(json, "phaseSecondaryAstigmatism45", &values.phaseSecondaryAstigmatism45);
    (void)extractJsonDoubleField(json, "phaseQuadrafoil0", &values.phaseQuadrafoil0);
    (void)extractJsonDoubleField(json, "phaseQuadrafoil45", &values.phaseQuadrafoil45);
    (void)extractJsonDoubleField(json, "phaseSecondaryComaX", &values.phaseSecondaryComaX);
    (void)extractJsonDoubleField(json, "phaseSecondaryComaY", &values.phaseSecondaryComaY);
    (void)extractJsonDoubleField(json, "pupilDecenterX", &values.pupilDecenterX);
    (void)extractJsonDoubleField(json, "pupilDecenterY", &values.pupilDecenterY);
    (void)extractJsonDoubleField(json, "phaseFieldStrength", &values.phaseFieldStrength);
    (void)extractJsonDoubleField(json, "phaseFieldEdgeBias", &values.phaseFieldEdgeBias);
    (void)extractJsonDoubleField(json, "phaseFieldDefocus", &values.phaseFieldDefocus);
    (void)extractJsonDoubleField(json, "phaseFieldAstigRadial", &values.phaseFieldAstigRadial);
    (void)extractJsonDoubleField(json, "phaseFieldAstigTangential", &values.phaseFieldAstigTangential);
    (void)extractJsonDoubleField(json, "phaseFieldComaRadial", &values.phaseFieldComaRadial);
    (void)extractJsonDoubleField(json, "phaseFieldComaTangential", &values.phaseFieldComaTangential);
    (void)extractJsonDoubleField(json, "phaseFieldSpherical", &values.phaseFieldSpherical);
    (void)extractJsonDoubleField(json, "phaseFieldTrefoilRadial", &values.phaseFieldTrefoilRadial);
    (void)extractJsonDoubleField(json, "phaseFieldTrefoilTangential", &values.phaseFieldTrefoilTangential);
    (void)extractJsonDoubleField(json, "phaseFieldSecondaryAstigRadial", &values.phaseFieldSecondaryAstigRadial);
    (void)extractJsonDoubleField(json, "phaseFieldSecondaryAstigTangential", &values.phaseFieldSecondaryAstigTangential);
    (void)extractJsonDoubleField(json, "phaseFieldQuadrafoilRadial", &values.phaseFieldQuadrafoilRadial);
    (void)extractJsonDoubleField(json, "phaseFieldQuadrafoilTangential", &values.phaseFieldQuadrafoilTangential);
    (void)extractJsonDoubleField(json, "phaseFieldSecondaryComaRadial", &values.phaseFieldSecondaryComaRadial);
    (void)extractJsonDoubleField(json, "phaseFieldSecondaryComaTangential", &values.phaseFieldSecondaryComaTangential);
    (void)extractJsonIntField(json, "spectralMode", &values.spectralMode);
    (void)extractJsonIntField(json, "spectrumStyle", &values.spectrumStyle);
    (void)extractJsonDoubleField(json, "spectrumForce", &values.spectrumForce);
    (void)extractJsonDoubleField(json, "spectrumSaturation", &values.spectrumSaturation);
    (void)extractJsonBoolField(json, "chromaticAffectsLuma", &values.chromaticAffectsLuma);
    (void)extractJsonDoubleField(json, "chromaticFocus", &values.chromaticFocus);
    (void)extractJsonDoubleField(json, "chromaticSpread", &values.chromaticSpread);
    (void)extractJsonDoubleField(json, "creativeFringe", &values.creativeFringe);
    (void)extractJsonDoubleField(json, "scatterAmount", &values.scatterAmount);
    (void)extractJsonDoubleField(json, "scatterRadius", &values.scatterRadius);
    (void)extractJsonIntField(json, "lookMode", &values.lookMode);
    (void)extractJsonIntField(json, "energyMode", &values.energyMode);
    (void)extractJsonDoubleField(json, "effectGain", &values.effectGain);
    (void)extractJsonDoubleField(json, "coreCompensation", &values.coreCompensation);
    (void)extractJsonDoubleField(json, "anisotropyEmphasis", &values.anisotropyEmphasis);
    (void)extractJsonDoubleField(json, "coreGain", &values.coreGain);
    (void)extractJsonDoubleField(json, "structureGain", &values.structureGain);
    (void)extractJsonDoubleField(json, "coreShoulder", &values.coreShoulder);
    (void)extractJsonDoubleField(json, "structureShoulder", &values.structureShoulder);
    (void)extractJsonIntField(json, "backendPreference", &values.backendPreference);
    (void)extractJsonIntField(json, "debugView", &values.debugView);
    *out = values;
    return true;
}

void saveLensDiffPresetStoreLocked() {
    const auto path = lensdiffPresetFilePath();
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os.is_open()) return;

    LensDiffPresetStore& store = lensdiffPresetStore();
    os << "{\n";
    os << "  \"schemaVersion\":1,\n";
    os << "  \"updatedAtUtc\":\"" << jsonEscape(nowUtcIso8601()) << "\",\n";
    os << "  \"defaultPreset\":{\n";
    os << "    \"id\":\"" << jsonEscape(store.defaultPreset.id) << "\",\n";
    os << "    \"name\":\"" << jsonEscape(store.defaultPreset.name) << "\",\n";
    os << "    \"updatedAtUtc\":\"" << jsonEscape(store.defaultPreset.updatedAtUtc) << "\",\n";
    os << "    \"values\":" << lensdiffPresetValuesAsJson(store.defaultPreset.values) << "\n";
    os << "  },\n";
    os << "  \"userPresets\":[\n";
    for (std::size_t i = 0; i < store.userPresets.size(); ++i) {
        const auto& preset = store.userPresets[i];
        os << "    {\n";
        os << "      \"id\":\"" << jsonEscape(preset.id) << "\",\n";
        os << "      \"name\":\"" << jsonEscape(preset.name) << "\",\n";
        os << "      \"createdAtUtc\":\"" << jsonEscape(preset.createdAtUtc) << "\",\n";
        os << "      \"updatedAtUtc\":\"" << jsonEscape(preset.updatedAtUtc) << "\",\n";
        os << "      \"values\":" << lensdiffPresetValuesAsJson(preset.values) << "\n";
        os << "    }" << (i + 1 < store.userPresets.size() ? "," : "") << "\n";
    }
    os << "  ]\n";
    os << "}\n";
}

void ensureLensDiffPresetStoreLoadedLocked() {
    LensDiffPresetStore& store = lensdiffPresetStore();
    if (store.loaded) return;
    store = LensDiffPresetStore{};
    store.loaded = true;

    const LensDiffPresetValues factory = lensdiffFactoryPresetValues();
    store.defaultPreset.id = "default";
    store.defaultPreset.name = kLensDiffPresetDefaultName;
    store.defaultPreset.createdAtUtc = nowUtcIso8601();
    store.defaultPreset.updatedAtUtc = store.defaultPreset.createdAtUtc;
    store.defaultPreset.values = factory;

    bool needsSave = false;
    std::ifstream is(lensdiffPresetFilePath(), std::ios::binary);
    if (!is.is_open()) {
        saveLensDiffPresetStoreLocked();
        return;
    }

    std::string json((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    if (json.empty()) {
        saveLensDiffPresetStoreLocked();
        return;
    }

    std::string defaultObj;
    if (extractJsonObjectField(json, "defaultPreset", &defaultObj)) {
        std::string defaultValuesJson;
        if (extractJsonObjectField(defaultObj, "values", &defaultValuesJson)) {
            parseLensDiffPresetValuesFromJson(defaultValuesJson, &store.defaultPreset.values);
        } else {
            needsSave = true;
        }
        std::string updatedAtUtc;
        if (extractJsonStringField(defaultObj, "updatedAtUtc", &updatedAtUtc)) {
            store.defaultPreset.updatedAtUtc = updatedAtUtc;
        }
    } else {
        needsSave = true;
    }

    if (store.defaultPreset.values.simpleMode != factory.simpleMode) {
        store.defaultPreset.values.simpleMode = factory.simpleMode;
        needsSave = true;
    }
    if (store.defaultPreset.values.apertureMode != factory.apertureMode) {
        store.defaultPreset.values.apertureMode = factory.apertureMode;
        needsSave = true;
    }
    if (!approxEqual(store.defaultPreset.values.rotationDeg, factory.rotationDeg, 1e-6)) {
        store.defaultPreset.values.rotationDeg = factory.rotationDeg;
        needsSave = true;
    }
    if (!approxEqual(store.defaultPreset.values.diffractionScalePx, factory.diffractionScalePx, 1e-6)) {
        store.defaultPreset.values.diffractionScalePx = factory.diffractionScalePx;
        needsSave = true;
    }
    if (store.defaultPreset.values.pupilResolution != factory.pupilResolution) {
        store.defaultPreset.values.pupilResolution = factory.pupilResolution;
        needsSave = true;
    }
    if (!approxEqual(store.defaultPreset.values.maxKernelRadiusPx, factory.maxKernelRadiusPx, 1e-6)) {
        store.defaultPreset.values.maxKernelRadiusPx = factory.maxKernelRadiusPx;
        needsSave = true;
    }
    if (store.defaultPreset.values.spectralMode != factory.spectralMode) {
        store.defaultPreset.values.spectralMode = factory.spectralMode;
        needsSave = true;
    }

    std::string arrayJson;
    if (extractJsonArrayField(json, "userPresets", &arrayJson)) {
        for (const std::string& objectJson : extractJsonObjectsFromArray(arrayJson)) {
            LensDiffUserPreset preset{};
            std::string valuesJson;
            std::string name;
            if (!extractJsonStringField(objectJson, "name", &name)) continue;
            if (extractJsonObjectField(objectJson, "values", &valuesJson)) {
                parseLensDiffPresetValuesFromJson(valuesJson, &preset.values);
            }
            if (!extractJsonStringField(objectJson, "id", &preset.id) || preset.id.empty()) {
                preset.id = makePresetId("lensdiff");
                needsSave = true;
            }
            preset.name = name;
            if (normalizePresetNameKey(preset.name) == "default") {
                needsSave = true;
                continue;
            }
            if (lensdiffBuiltinPresetByName(preset.name)) {
                needsSave = true;
                continue;
            }
            if (!extractJsonStringField(objectJson, "createdAtUtc", &preset.createdAtUtc) || preset.createdAtUtc.empty()) {
                preset.createdAtUtc = nowUtcIso8601();
                needsSave = true;
            }
            if (!extractJsonStringField(objectJson, "updatedAtUtc", &preset.updatedAtUtc) || preset.updatedAtUtc.empty()) {
                preset.updatedAtUtc = preset.createdAtUtc;
                needsSave = true;
            }
            store.userPresets.push_back(preset);
        }
    }

    if (needsSave) saveLensDiffPresetStoreLocked();
}

void reloadLensDiffPresetStoreFromDiskLocked() {
    lensdiffPresetStore() = LensDiffPresetStore{};
    ensureLensDiffPresetStoreLoadedLocked();
}

LensDiffPresetValues describeLensDiffDefaultValues() {
    std::lock_guard<std::mutex> lock(lensdiffPresetMutex());
    ensureLensDiffPresetStoreLoadedLocked();
    return lensdiffPresetStore().defaultPreset.values;
}

#ifdef _WIN32
bool confirmLensDiffPresetOverwriteDialog(const std::string& presetName) {
    const std::string message = "Preset '" + presetName + "' already exists. Overwrite?";
    return MessageBoxA(nullptr, message.c_str(), "LensDiff", MB_ICONQUESTION | MB_YESNO) == IDYES;
}

void showLensDiffPresetInfoDialog(const std::string& text) {
    MessageBoxA(nullptr, text.c_str(), "LensDiff", MB_ICONINFORMATION | MB_OK);
}

bool confirmLensDiffPresetDeleteDialog(const std::string& presetName) {
    const std::string message = "Delete preset '" + presetName + "'? This cannot be undone.";
    return MessageBoxA(nullptr, message.c_str(), "LensDiff", MB_ICONWARNING | MB_YESNO) == IDYES;
}
#elif defined(__APPLE__)
std::string execAndReadLensDiff(const std::string& cmd) {
    std::string out;
    FILE* f = popen(cmd.c_str(), "r");
    if (!f) return out;
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), f)) out += buffer;
    pclose(f);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}

bool confirmLensDiffPresetOverwriteDialog(const std::string& presetName) {
    std::string safe = presetName;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd =
        "osascript -e 'button returned of (display dialog \"Preset \\\"" + safe +
        "\\\" already exists. Overwrite?\" buttons {\"Cancel\",\"Overwrite\"} default button \"Overwrite\")' 2>/dev/null";
    return execAndReadLensDiff(cmd) == "Overwrite";
}

void showLensDiffPresetInfoDialog(const std::string& text) {
    std::string safe = text;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd =
        "osascript -e 'display dialog \"" + safe + "\" buttons {\"OK\"} default button \"OK\"' 2>/dev/null";
    (void)execAndReadLensDiff(cmd);
}

bool confirmLensDiffPresetDeleteDialog(const std::string& presetName) {
    std::string safe = presetName;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd =
        "osascript -e 'button returned of (display dialog \"Delete preset \\\"" + safe +
        "\\\"? This cannot be undone.\" buttons {\"Cancel\",\"Delete\"} default button \"Delete\")' 2>/dev/null";
    return execAndReadLensDiff(cmd) == "Delete";
}
#else
bool linuxLensDiffCommandExists(const char* cmd) {
    if (!cmd || !*cmd) return false;
    std::string probe = "command -v ";
    probe += cmd;
    probe += " >/dev/null 2>&1";
    return std::system(probe.c_str()) == 0;
}

bool confirmLensDiffPresetOverwriteDialog(const std::string& presetName) {
    if (linuxLensDiffCommandExists("zenity")) {
        std::string safe = presetName;
        for (char& c : safe) if (c == '"') c = '\'';
        const std::string cmd =
            "zenity --question --title=\"LensDiff\" --text=\"Preset '" + safe + "' already exists. Overwrite?\" 2>/dev/null";
        return std::system(cmd.c_str()) == 0;
    }
    if (linuxLensDiffCommandExists("kdialog")) {
        std::string safe = presetName;
        for (char& c : safe) if (c == '"') c = '\'';
        const std::string cmd =
            "kdialog --warningyesno \"Preset '" + safe + "' already exists. Overwrite?\" 2>/dev/null";
        return std::system(cmd.c_str()) == 0;
    }
    std::fprintf(stderr, "[LensDiff] overwrite confirmation unavailable for preset '%s'.\n", presetName.c_str());
    return false;
}

void showLensDiffPresetInfoDialog(const std::string& text) {
    if (linuxLensDiffCommandExists("zenity")) {
        std::string safe = text;
        for (char& c : safe) if (c == '"') c = '\'';
        const std::string cmd = "zenity --info --title=\"LensDiff\" --text=\"" + safe + "\" 2>/dev/null";
        const int result = std::system(cmd.c_str());
        (void)result;
        return;
    }
    if (linuxLensDiffCommandExists("kdialog")) {
        std::string safe = text;
        for (char& c : safe) if (c == '"') c = '\'';
        const std::string cmd = "kdialog --msgbox \"" + safe + "\" 2>/dev/null";
        const int result = std::system(cmd.c_str());
        (void)result;
        return;
    }
    std::fprintf(stderr, "[LensDiff] %s\n", text.c_str());
}

bool confirmLensDiffPresetDeleteDialog(const std::string& presetName) {
    if (linuxLensDiffCommandExists("zenity")) {
        std::string safe = presetName;
        for (char& c : safe) if (c == '"') c = '\'';
        const std::string cmd =
            "zenity --question --title=\"LensDiff\" --text=\"Delete preset '" + safe + "'? This cannot be undone.\" 2>/dev/null";
        return std::system(cmd.c_str()) == 0;
    }
    if (linuxLensDiffCommandExists("kdialog")) {
        std::string safe = presetName;
        for (char& c : safe) if (c == '"') c = '\'';
        const std::string cmd =
            "kdialog --warningyesno \"Delete preset '" + safe + "'? This cannot be undone.\" 2>/dev/null";
        return std::system(cmd.c_str()) == 0;
    }
    std::fprintf(stderr, "[LensDiff] delete confirmation unavailable for preset '%s'.\n", presetName.c_str());
    return false;
}
#endif

OFX::DoubleParamDescriptor* defineDoubleParam(OFX::ImageEffectDescriptor& desc,
                                              const std::string& name,
                                              const std::string& label,
                                              const std::string& hint,
                                              OFX::GroupParamDescriptor* parent,
                                              double defaultValue,
                                              double minValue,
                                              double maxValue,
                                              double increment) {
    auto* param = desc.defineDoubleParam(name);
    param->setLabels(label, label, label);
    param->setHint(hint);
    param->setDefault(defaultValue);
    param->setRange(minValue, maxValue);
    param->setDisplayRange(minValue, maxValue);
    param->setIncrement(increment);
    param->setParent(*parent);
    return param;
}

OFX::IntParamDescriptor* defineIntParam(OFX::ImageEffectDescriptor& desc,
                                        const std::string& name,
                                        const std::string& label,
                                        const std::string& hint,
                                        OFX::GroupParamDescriptor* parent,
                                        int defaultValue,
                                        int minValue,
                                        int maxValue) {
    auto* param = desc.defineIntParam(name);
    param->setLabels(label, label, label);
    param->setHint(hint);
    param->setDefault(defaultValue);
    param->setRange(minValue, maxValue);
    param->setDisplayRange(minValue, maxValue);
    param->setParent(*parent);
    return param;
}

OFX::BooleanParamDescriptor* defineBooleanParam(OFX::ImageEffectDescriptor& desc,
                                                const std::string& name,
                                                const std::string& label,
                                                const std::string& hint,
                                                OFX::GroupParamDescriptor* parent,
                                                bool defaultValue) {
    auto* param = desc.defineBooleanParam(name);
    param->setLabels(label, label, label);
    param->setHint(hint);
    param->setDefault(defaultValue);
    param->setParent(*parent);
    return param;
}

OFX::ChoiceParamDescriptor* defineChoiceParam(OFX::ImageEffectDescriptor& desc,
                                              const std::string& name,
                                              const std::string& label,
                                              const std::string& hint,
                                              OFX::GroupParamDescriptor* parent,
                                              std::initializer_list<const char*> options,
                                              int defaultIndex) {
    auto* param = desc.defineChoiceParam(name);
    param->setLabel(label);
    param->setHint(hint);
    for (const char* option : options) {
        param->appendOption(option);
    }
    param->setDefault(defaultIndex);
    param->setParent(*parent);
    return param;
}

OFX::StringParamDescriptor* defineStringParam(OFX::ImageEffectDescriptor& desc,
                                              const std::string& name,
                                              const std::string& label,
                                              const std::string& hint,
                                              OFX::GroupParamDescriptor* parent,
                                              const std::string& defaultValue,
                                              OFX::StringTypeEnum stringType,
                                              bool fileMustExist) {
    auto* param = desc.defineStringParam(name);
    param->setLabel(label);
    param->setHint(hint);
    param->setDefault(defaultValue);
    param->setStringType(stringType);
    param->setFilePathExists(fileMustExist);
    param->setParent(*parent);
    return param;
}

OFX::PushButtonParamDescriptor* definePushButtonParam(OFX::ImageEffectDescriptor& desc,
                                                      const std::string& name,
                                                      const std::string& label,
                                                      const std::string& hint,
                                                      OFX::GroupParamDescriptor* parent) {
    auto* param = desc.definePushButtonParam(name);
    param->setLabels(label, label, label);
    param->setHint(hint);
    param->setParent(*parent);
    return param;
}

OFX::StringParamDescriptor* defineLabelParam(OFX::ImageEffectDescriptor& desc,
                                             const std::string& name,
                                             const std::string& label,
                                             const std::string& hint,
                                             OFX::GroupParamDescriptor* parent,
                                             const std::string& value) {
    auto* param = desc.defineStringParam(name);
    param->setLabel(label);
    param->setHint(hint);
    param->setDefault(value);
    param->setStringType(OFX::eStringTypeLabel);
    param->setParent(*parent);
    return param;
}

int getChoiceValueAtTime(OFX::ChoiceParam* param, double time) {
    int value = 0;
    if (param) {
        param->getValueAtTime(time, value);
    }
    return value;
}

std::string getStringValueAtTime(OFX::StringParam* param, double time) {
    std::string value;
    if (param) {
        param->getValueAtTime(time, value);
    }
    return value;
}

LensDiffImageView makeImageView(OFX::Image* image) {
    LensDiffImageView view {};
    if (!image) {
        return view;
    }

    const OfxRectI bounds = image->getBounds();
    view.data = image->getPixelData();
    view.rowBytes = image->getRowBytes();
    view.bounds = {bounds.x1, bounds.y1, bounds.x2, bounds.y2};
    view.originX = bounds.x1;
    view.originY = bounds.y1;
    view.components = 4;
    view.bytesPerComponent = 4;
    return view;
}

LensDiffBackendType backendFromChoice(int index) {
    switch (index) {
        case 1: return LensDiffBackendType::CpuReference;
        case 2: return LensDiffBackendType::Cuda;
        case 3: return LensDiffBackendType::Metal;
        default: return LensDiffBackendType::Auto;
    }
}

LensDiffBackendType resolveRenderBackend(const OFX::RenderArguments& args, int backendChoiceIndex) {
    const LensDiffBackendType preferred = backendFromChoice(backendChoiceIndex);
    if (preferred != LensDiffBackendType::Auto) {
        return preferred;
    }
#if defined(LENSDIFF_HAS_CUDA) && !defined(__APPLE__)
    if (args.isEnabledCudaRender && args.pCudaStream != nullptr) {
        return LensDiffBackendType::Cuda;
    }
#endif
#ifdef __APPLE__
    if (args.isEnabledMetalRender && args.pMetalCmdQ != nullptr) {
        return LensDiffBackendType::Metal;
    }
#endif
    return LensDiffBackendType::CpuReference;
}

LensDiffExtractionMode extractionFromChoice(int index) {
    return index == 1 ? LensDiffExtractionMode::Luma : LensDiffExtractionMode::MaxRgb;
}

bool hasCustomAperturePath(const std::string& path) {
    for (unsigned char ch : path) {
        if (!std::isspace(ch)) {
            return true;
        }
    }
    return false;
}

std::string customApertureDisplayName(const std::string& path) {
    if (!hasCustomAperturePath(path)) {
        return "Custom";
    }

    const std::size_t leafPos = path.find_last_of("/\\");
    std::string leaf = leafPos == std::string::npos ? path : path.substr(leafPos + 1);
    while (!leaf.empty() && std::isspace(static_cast<unsigned char>(leaf.back()))) {
        leaf.pop_back();
    }
    while (!leaf.empty() && std::isspace(static_cast<unsigned char>(leaf.front()))) {
        leaf.erase(leaf.begin());
    }
    if (leaf.empty()) {
        return "Custom";
    }

    const std::size_t extensionPos = leaf.find_last_of('.');
    if (extensionPos != std::string::npos && extensionPos > 0) {
        leaf = leaf.substr(0, extensionPos);
    }
    return leaf.empty() ? "Custom" : leaf;
}

LensDiffApertureMode apertureFromChoice(int index, bool hasCustomAperture) {
    switch (index) {
        case 1: return LensDiffApertureMode::Polygon;
        case 2: return LensDiffApertureMode::Star;
        case 3: return LensDiffApertureMode::Spiral;
        case 4: return LensDiffApertureMode::Hexagon;
        case 5: return LensDiffApertureMode::SquareGrid;
        case 6: return LensDiffApertureMode::Snowflake;
        case kCustomApertureChoiceIndex:
            return hasCustomAperture ? LensDiffApertureMode::Custom : LensDiffApertureMode::Circle;
        default: return LensDiffApertureMode::Circle;
    }
}

int apertureChoiceIndexForPresetMode(LensDiffApertureMode mode, bool hasCustomAperture) {
    switch (mode) {
        case LensDiffApertureMode::Circle: return 0;
        case LensDiffApertureMode::Polygon: return 1;
        case LensDiffApertureMode::Star: return 2;
        case LensDiffApertureMode::Spiral: return 3;
        case LensDiffApertureMode::Hexagon: return 4;
        case LensDiffApertureMode::SquareGrid: return 5;
        case LensDiffApertureMode::Snowflake: return 6;
        case LensDiffApertureMode::Custom:
            return hasCustomAperture ? kCustomApertureChoiceIndex : 0;
        default:
            return 0;
    }
}

LensDiffApertureMode sanitizePresetApertureMode(int storedMode) {
    switch (static_cast<LensDiffApertureMode>(storedMode)) {
        case LensDiffApertureMode::Circle:
        case LensDiffApertureMode::Polygon:
        case LensDiffApertureMode::Star:
        case LensDiffApertureMode::Spiral:
        case LensDiffApertureMode::Custom:
        case LensDiffApertureMode::Hexagon:
        case LensDiffApertureMode::SquareGrid:
        case LensDiffApertureMode::Snowflake:
            return static_cast<LensDiffApertureMode>(storedMode);
        default:
            return LensDiffApertureMode::Circle;
    }
}

LensDiffApodizationMode apodizationFromChoice(int index) {
    switch (index) {
        case 1: return LensDiffApodizationMode::Cosine;
        case 2: return LensDiffApodizationMode::Gaussian;
        default: return LensDiffApodizationMode::Flat;
    }
}

LensDiffSpectralMode spectralFromChoice(int index) {
    switch (index) {
        case 1: return LensDiffSpectralMode::Tristimulus;
        case 2: return LensDiffSpectralMode::Spectral5;
        case 3: return LensDiffSpectralMode::Spectral9;
        default: return LensDiffSpectralMode::Mono;
    }
}

LensDiffSpectrumStyle spectrumStyleFromChoice(int index) {
    switch (index) {
        case 1: return LensDiffSpectrumStyle::CyanMagenta;
        case 2: return LensDiffSpectrumStyle::WarmCool;
        default: return LensDiffSpectrumStyle::Natural;
    }
}

LensDiffLookMode lookModeFromChoice(int index) {
    return index == 1 ? LensDiffLookMode::Split : LensDiffLookMode::Physical;
}

LensDiffEnergyMode energyModeFromChoice(int index) {
    return index == 1 ? LensDiffEnergyMode::Augment : LensDiffEnergyMode::Preserve;
}

LensDiffDebugView debugViewFromChoice(int index) {
    switch (index) {
        case 1: return LensDiffDebugView::Selection;
        case 2: return LensDiffDebugView::Pupil;
        case 3: return LensDiffDebugView::Psf;
        case 4: return LensDiffDebugView::Otf;
        case 5: return LensDiffDebugView::Core;
        case 6: return LensDiffDebugView::Structure;
        case 7: return LensDiffDebugView::Effect;
        case 8: return LensDiffDebugView::Phase;
        case 9: return LensDiffDebugView::PhaseEdge;
        case 10: return LensDiffDebugView::FieldPsf;
        case 11: return LensDiffDebugView::ChromaticSplit;
        case 12: return LensDiffDebugView::CreativeFringe;
        case 13: return LensDiffDebugView::Scatter;
        default: return LensDiffDebugView::Final;
    }
}

LensDiffInputTransfer inputTransferFromChoice(int index) {
    return index == 1 ? LensDiffInputTransfer::DavinciIntermediate : LensDiffInputTransfer::Linear;
}

int pupilResolutionFromChoice(int index) {
    if (index < 0 || index >= static_cast<int>(kPupilResolutionOptions.size())) {
        return 256;
    }
    return kPupilResolutionOptions[static_cast<std::size_t>(index)];
}

int choiceFromPupilResolution(int value) {
    int bestIndex = 0;
    int bestDistance = std::abs(value - kPupilResolutionOptions[0]);
    for (int i = 1; i < static_cast<int>(kPupilResolutionOptions.size()); ++i) {
        const int distance = std::abs(value - kPupilResolutionOptions[static_cast<std::size_t>(i)]);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = i;
        }
    }
    if (bestIndex < 0 || bestIndex >= static_cast<int>(kPupilResolutionOptions.size())) {
        return 2;
    }
    return bestIndex;
}

const char* backendName(LensDiffBackendType backend) {
    switch (backend) {
        case LensDiffBackendType::CpuReference: return "CPU";
        case LensDiffBackendType::Cuda: return "CUDA";
        case LensDiffBackendType::Metal: return "Metal";
        case LensDiffBackendType::Auto:
        default: return "Auto";
    }
}

const char* spectralModeName(LensDiffSpectralMode mode) {
    switch (mode) {
        case LensDiffSpectralMode::Tristimulus: return "Tristimulus";
        case LensDiffSpectralMode::Spectral5: return "Spectral5";
        case LensDiffSpectralMode::Spectral9: return "Spectral9";
        case LensDiffSpectralMode::Mono:
        default:
            return "Mono";
    }
}

const char* debugViewName(LensDiffDebugView view) {
    switch (view) {
        case LensDiffDebugView::Selection: return "Selection";
        case LensDiffDebugView::Pupil: return "Pupil";
        case LensDiffDebugView::Psf: return "PSF";
        case LensDiffDebugView::Otf: return "OTF";
        case LensDiffDebugView::Core: return "Core";
        case LensDiffDebugView::Structure: return "Structure";
        case LensDiffDebugView::Effect: return "Effect";
        case LensDiffDebugView::Phase: return "Phase";
        case LensDiffDebugView::PhaseEdge: return "PhaseEdge";
        case LensDiffDebugView::FieldPsf: return "FieldPSF";
        case LensDiffDebugView::ChromaticSplit: return "ChromaticSplit";
        case LensDiffDebugView::CreativeFringe: return "CreativeFringe";
        case LensDiffDebugView::Scatter: return "Scatter";
        case LensDiffDebugView::Final:
        default: return "Final";
    }
}

const char* inputTransferName(LensDiffInputTransfer transfer) {
    return transfer == LensDiffInputTransfer::DavinciIntermediate ? "DaVinci Intermediate" : "Linear";
}

bool envFlagEnabled(const char* name) {
    return LensDiffEnvFlagEnabled(name);
}

void logLensDiffRender(bool includeTiming,
                       LensDiffBackendType backend,
                       const LensDiffRenderRequest& request,
                       const LensDiffParams& params,
                       const std::string& note,
                       double elapsedMs = 0.0) {
    std::ostringstream ss;
    ss << "[LensDiff] backend=" << backendName(backend)
       << " renderWindow=(" << request.renderWindow.x1 << "," << request.renderWindow.y1 << ")-("
       << request.renderWindow.x2 << "," << request.renderWindow.y2 << ")"
       << " frameBounds=(" << request.frameBounds.x1 << "," << request.frameBounds.y1 << ")-("
       << request.frameBounds.x2 << "," << request.frameBounds.y2 << ")"
       << " srcBounds=(" << request.src.bounds.x1 << "," << request.src.bounds.y1 << ")-("
       << request.src.bounds.x2 << "," << request.src.bounds.y2 << ")"
       << " dstBounds=(" << request.dst.bounds.x1 << "," << request.dst.bounds.y1 << ")-("
       << request.dst.bounds.x2 << "," << request.dst.bounds.y2 << ")"
       << " frameShortSidePx=" << request.frameShortSidePx
       << " opticsShortSidePx=" << ResolveLensDiffOpticsShortSidePx(params)
       << " resolutionAware=" << (params.resolutionAware ? "true" : "false")
        << " diffractionScalePct=" << params.diffractionScalePx
       << " diffractionScalePx=" << ResolveLensDiffDiffractionScalePx(params)
       << " maxKernelRadiusPct=" << params.maxKernelRadiusPx
       << " maxKernelRadiusPx=" << ResolveLensDiffMaxKernelRadiusPx(params)
       << " debug=" << debugViewName(params.debugView)
       << " spectral=" << spectralModeName(params.spectralMode)
       << " inputTransfer=" << inputTransferName(params.inputTransfer);
    if (!note.empty()) {
        ss << " note=" << note;
    }
    if (includeTiming) {
        ss << " elapsedMs=" << elapsedMs;
    }
    const std::string line = ss.str();
    OFX::Log::print("%s\n", line.c_str());
    WriteLensDiffDiagnosticLine(line);
}

class LensDiffEffect : public OFX::ImageEffect {
public:
    struct LensDiffPresetSelection {
        enum class Kind {
            Default,
            Builtin,
            User,
            Custom
        };
        Kind kind = Kind::Default;
        int builtinIndex = -1;
        int userIndex = -1;
        bool modified = false;
    };

    LensDiffEffect(OfxImageEffectHandle handle)
        : OFX::ImageEffect(handle)
        , srcClip_(fetchClip(kOfxImageEffectSimpleSourceClipName))
        , dstClip_(fetchClip(kOfxImageEffectOutputClipName))
        , simpleModeState_(fetchBooleanParam("simpleModeState"))
        , simpleModeToggle_(fetchPushButtonParam("simpleModeToggle"))
        , simpleEffectGain_(fetchDoubleParam("simpleEffectGain"))
        , threshold_(fetchDoubleParam("threshold"))
        , inputTransfer_(fetchChoiceParam("inputTransfer"))
        , softnessStops_(fetchDoubleParam("softnessStops"))
        , pointEmphasis_(fetchDoubleParam("pointEmphasis"))
        , corePreserve_(fetchDoubleParam("corePreserve"))
        , extractionMode_(fetchChoiceParam("extractionMode"))
        , apertureMode_(fetchChoiceParam("apertureMode"))
        , customApertureImport_(fetchPushButtonParam("customApertureImport"))
        , customAperturePath_(fetchStringParam("customAperturePath"))
        , customApertureStatus_(fetchStringParam("customApertureStatus"))
        , customApertureNormalize_(fetchBooleanParam("customApertureNormalize"))
        , customApertureInvert_(fetchBooleanParam("customApertureInvert"))
        , bladeCount_(fetchIntParam("bladeCount"))
        , roundness_(fetchDoubleParam("roundness"))
        , rotationDeg_(fetchDoubleParam("rotationDeg"))
        , centralObstruction_(fetchDoubleParam("centralObstruction"))
        , vaneCount_(fetchIntParam("vaneCount"))
        , vaneThickness_(fetchDoubleParam("vaneThickness"))
        , apodizationMode_(fetchChoiceParam("apodizationMode"))
        , diffractionScalePx_(fetchDoubleParam("diffractionScalePx"))
        , pupilResolution_(fetchIntParam("pupilResolution"))
        , pupilResolutionChoice_(nullptr)
        , maxKernelRadiusPx_(fetchDoubleParam("maxKernelRadiusPx"))
        , phaseEnabled_(fetchBooleanParam("phaseEnabled"))
        , phaseFocus_(fetchDoubleParam("phaseFocus"))
        , phaseAstigmatism_(fetchDoubleParam("phaseAstigmatism"))
        , phaseAstigAngleDeg_(fetchDoubleParam("phaseAstigAngleDeg"))
        , phaseComa_(fetchDoubleParam("phaseComa"))
        , phaseComaAngleDeg_(fetchDoubleParam("phaseComaAngleDeg"))
        , phaseSphericalAmount_(fetchDoubleParam("phaseSphericalAmount"))
        , phaseTrefoil_(fetchDoubleParam("phaseTrefoil"))
        , phaseTrefoilAngleDeg_(fetchDoubleParam("phaseTrefoilAngleDeg"))
        , phaseDefocus_(fetchDoubleParam("phaseDefocus"))
        , phaseAstigmatism0_(fetchDoubleParam("phaseAstigmatism0"))
        , phaseAstigmatism45_(fetchDoubleParam("phaseAstigmatism45"))
        , phaseComaX_(fetchDoubleParam("phaseComaX"))
        , phaseComaY_(fetchDoubleParam("phaseComaY"))
        , phaseSpherical_(fetchDoubleParam("phaseSpherical"))
        , phaseTrefoilX_(fetchDoubleParam("phaseTrefoilX"))
        , phaseTrefoilY_(fetchDoubleParam("phaseTrefoilY"))
        , phaseSecondaryAstigmatism0_(fetchDoubleParam("phaseSecondaryAstigmatism0"))
        , phaseSecondaryAstigmatism45_(fetchDoubleParam("phaseSecondaryAstigmatism45"))
        , phaseQuadrafoil0_(fetchDoubleParam("phaseQuadrafoil0"))
        , phaseQuadrafoil45_(fetchDoubleParam("phaseQuadrafoil45"))
        , phaseSecondaryComaX_(fetchDoubleParam("phaseSecondaryComaX"))
        , phaseSecondaryComaY_(fetchDoubleParam("phaseSecondaryComaY"))
        , pupilDecenterX_(fetchDoubleParam("pupilDecenterX"))
        , pupilDecenterY_(fetchDoubleParam("pupilDecenterY"))
        , phaseFieldStrength_(fetchDoubleParam("phaseFieldStrength"))
        , phaseFieldEdgeBias_(fetchDoubleParam("phaseFieldEdgeBias"))
        , phaseFieldDefocus_(fetchDoubleParam("phaseFieldDefocus"))
        , phaseFieldAstigRadial_(fetchDoubleParam("phaseFieldAstigRadial"))
        , phaseFieldAstigTangential_(fetchDoubleParam("phaseFieldAstigTangential"))
        , phaseFieldComaRadial_(fetchDoubleParam("phaseFieldComaRadial"))
        , phaseFieldComaTangential_(fetchDoubleParam("phaseFieldComaTangential"))
        , phaseFieldSpherical_(fetchDoubleParam("phaseFieldSpherical"))
        , phaseFieldTrefoilRadial_(fetchDoubleParam("phaseFieldTrefoilRadial"))
        , phaseFieldTrefoilTangential_(fetchDoubleParam("phaseFieldTrefoilTangential"))
        , phaseFieldSecondaryAstigRadial_(fetchDoubleParam("phaseFieldSecondaryAstigRadial"))
        , phaseFieldSecondaryAstigTangential_(fetchDoubleParam("phaseFieldSecondaryAstigTangential"))
        , phaseFieldQuadrafoilRadial_(fetchDoubleParam("phaseFieldQuadrafoilRadial"))
        , phaseFieldQuadrafoilTangential_(fetchDoubleParam("phaseFieldQuadrafoilTangential"))
        , phaseFieldSecondaryComaRadial_(fetchDoubleParam("phaseFieldSecondaryComaRadial"))
        , phaseFieldSecondaryComaTangential_(fetchDoubleParam("phaseFieldSecondaryComaTangential"))
        , spectralMode_(fetchChoiceParam("spectralMode"))
        , spectrumStyle_(fetchChoiceParam("spectrumStyle"))
        , spectrumForce_(fetchDoubleParam("spectrumForce"))
        , spectrumSaturation_(fetchDoubleParam("spectrumSaturation"))
        , chromaticAffectsLuma_(fetchBooleanParam("chromaticAffectsLuma"))
        , chromaticFocus_(fetchDoubleParam("chromaticFocus"))
        , chromaticSpread_(fetchDoubleParam("chromaticSpread"))
        , creativeFringe_(fetchDoubleParam("creativeFringe"))
        , scatterAmount_(fetchDoubleParam("scatterAmount"))
        , scatterRadius_(fetchDoubleParam("scatterRadius"))
        , lookMode_(fetchChoiceParam("lookMode"))
        , energyMode_(fetchChoiceParam("energyMode"))
        , effectGain_(fetchDoubleParam("effectGain"))
        , resolutionAware_(fetchBooleanParam("resolutionAware"))
        , coreCompensation_(fetchDoubleParam("coreCompensation"))
        , anisotropyEmphasis_(fetchDoubleParam("anisotropyEmphasis"))
        , coreGain_(fetchDoubleParam("coreGain"))
        , structureGain_(fetchDoubleParam("structureGain"))
        , coreShoulder_(fetchDoubleParam("coreShoulder"))
        , structureShoulder_(fetchDoubleParam("structureShoulder"))
        , backendPreference_(fetchChoiceParam("backendPreference"))
        , debugView_(fetchChoiceParam("debugView"))
        , lensdiffPresetMenu_(nullptr) {
        if (paramExists("pupilResolutionChoice")) {
            pupilResolutionChoice_ = fetchChoiceParam("pupilResolutionChoice");
        }
        if (paramExists("lensdiffPresetMenu")) {
            lensdiffPresetMenu_ = fetchChoiceParam("lensdiffPresetMenu");
        }
        if (paramExists("CompositeGroup")) {
            compositeGroup_ = fetchGroupParam("CompositeGroup");
        }
        if (paramExists("CompositeSplitGroup")) {
            compositeSplitGroup_ = fetchGroupParam("CompositeSplitGroup");
        }
        if (paramExists("PhaseGroup")) {
            phaseGroup_ = fetchGroupParam("PhaseGroup");
        }
        if (paramExists("PhasePrimaryGroup")) {
            phasePrimaryGroup_ = fetchGroupParam("PhasePrimaryGroup");
        }
        if (paramExists("PhaseAdvancedGroup")) {
            phaseAdvancedGroup_ = fetchGroupParam("PhaseAdvancedGroup");
        }
        if (paramExists("PhaseFieldGroup")) {
            phaseFieldGroup_ = fetchGroupParam("PhaseFieldGroup");
        }
        if (paramExists("PhaseFieldPrimaryGroup")) {
            phaseFieldPrimaryGroup_ = fetchGroupParam("PhaseFieldPrimaryGroup");
        }
        if (paramExists("PhaseFieldHigherOrderGroup")) {
            phaseFieldHigherOrderGroup_ = fetchGroupParam("PhaseFieldHigherOrderGroup");
        }
        if (paramExists("PhaseChromaticGroup")) {
            phaseChromaticGroup_ = fetchGroupParam("PhaseChromaticGroup");
        }
        if (paramExists("PhaseFinishingGroup")) {
            phaseFinishingGroup_ = fetchGroupParam("PhaseFinishingGroup");
        }
        refreshLensDiffPresetMenuMetadataFromStore();
        syncPhaseMacrosFromCoefficients();
        syncEffectGainMirror(false);
        refreshDynamicControlVisibility();
        updateCustomApertureStatus();
        syncPupilResolutionUi(false);
    }

    void render(const OFX::RenderArguments& args) override;
    bool isIdentity(const OFX::IsIdentityArguments& args, OFX::Clip*& identityClip, double& identityTime) override;
    bool getRegionOfDefinition(const OFX::RegionOfDefinitionArguments& args, OfxRectD& rod) override;
    void getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args, OFX::RegionOfInterestSetter& rois) override;
    void changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) override;
    void syncPrivateData() override;

private:
    LensDiffParams resolveParams(double time) const;
    LensDiffParams resolveParams(double time, int frameShortSidePx) const;
    LensDiffPresetValues captureCurrentLensDiffPresetValues(double time) const;
    LensDiffPresetValues captureCurrentLensDiffPresetDirtyValues(double time) const;
    void writeLensDiffPresetValuesToParams(const LensDiffPresetValues& values);
    LensDiffPresetSelection matchingLensDiffPresetSelection(double time) const;
    LensDiffPresetSelection matchingLensDiffPresetSelection(const LensDiffPresetValues& current) const;
    LensDiffPresetSelection resolvedLensDiffPresetSelection(double time) const;
    LensDiffPresetSelection resolvedLensDiffPresetSelection(const LensDiffPresetSelection& exact,
                                                           const LensDiffPresetSelection& currentMenuSelection) const;
    LensDiffPresetSelection selectedLensDiffPresetFromMenu(double time) const;
    void rebuildLensDiffPresetMenu(double time, const LensDiffPresetSelection& selection);
    void updateLensDiffPresetActionState(double time);
    void refreshLensDiffPresetMenuMetadataFromStore();
    void syncLensDiffPresetMenuState(double time);
    void syncLensDiffPresetMenuFromDisk(double time,
                                        const std::optional<LensDiffPresetSelection>& preferred = std::nullopt);
    void finalizeLensDiffPresetApplication(double time);
    bool resolveLensDiffPresetSelectionValues(const LensDiffPresetSelection& selection,
                                             LensDiffPresetValues* values,
                                             std::string* presetName = nullptr) const;
    void applyLensDiffPresetSelection(double time,
                                      const LensDiffPresetSelection& selection,
                                      const char* reasonParam);
    void applySelectedLensDiffPreset(double time);
    bool isLensDiffPresetManagedParam(const std::string& paramName) const;
    void writeOpticsPresetValuesToParams(const LensDiffPresetValues& values);
    void writePhasePresetValuesToParams(const LensDiffPresetValues& values);
    void writeSpectrumPresetValuesToParams(const LensDiffPresetValues& values);
    void writeCompositePresetValuesToParams(const LensDiffPresetValues& values);
    void syncCustomApertureUi(bool forceSelectCustom);
    void syncPupilResolutionUi(bool fromChoice);
    void syncPhaseMacrosFromCoefficients();
    void syncPhaseCoefficientsFromMacros();
    void syncEffectGainMirror(bool fromSimpleEffectGain);
    bool isSimpleMode() const;
    bool isSimpleMode(double time) const;
    void updateSimpleModeToggleLabel();
    void updateCustomApertureControlVisibility();
    void updateOpticsControlVisibility();
    void updateSpectrumControlVisibility();
    void updateCompositeControlVisibility();
    void updatePhaseControlVisibility();
    void refreshDynamicControlVisibility();
    void updateCustomApertureStatus();
    void openPhaseGroup();

    OFX::Clip* srcClip_ = nullptr;
    OFX::Clip* dstClip_ = nullptr;

    OFX::BooleanParam* simpleModeState_ = nullptr;
    OFX::PushButtonParam* simpleModeToggle_ = nullptr;
    OFX::DoubleParam* simpleEffectGain_ = nullptr;
    OFX::DoubleParam* threshold_ = nullptr;
    OFX::ChoiceParam* inputTransfer_ = nullptr;
    OFX::DoubleParam* softnessStops_ = nullptr;
    OFX::DoubleParam* pointEmphasis_ = nullptr;
    OFX::DoubleParam* corePreserve_ = nullptr;
    OFX::ChoiceParam* extractionMode_ = nullptr;
    OFX::ChoiceParam* apertureMode_ = nullptr;
    OFX::PushButtonParam* customApertureImport_ = nullptr;
    OFX::StringParam* customAperturePath_ = nullptr;
    OFX::StringParam* customApertureStatus_ = nullptr;
    OFX::BooleanParam* customApertureNormalize_ = nullptr;
    OFX::BooleanParam* customApertureInvert_ = nullptr;
    OFX::IntParam* bladeCount_ = nullptr;
    OFX::DoubleParam* roundness_ = nullptr;
    OFX::DoubleParam* rotationDeg_ = nullptr;
    OFX::DoubleParam* centralObstruction_ = nullptr;
    OFX::IntParam* vaneCount_ = nullptr;
    OFX::DoubleParam* vaneThickness_ = nullptr;
    OFX::ChoiceParam* apodizationMode_ = nullptr;
    OFX::DoubleParam* diffractionScalePx_ = nullptr;
    OFX::IntParam* pupilResolution_ = nullptr;
    OFX::ChoiceParam* pupilResolutionChoice_ = nullptr;
    OFX::DoubleParam* maxKernelRadiusPx_ = nullptr;
    OFX::BooleanParam* phaseEnabled_ = nullptr;
    OFX::DoubleParam* phaseFocus_ = nullptr;
    OFX::DoubleParam* phaseAstigmatism_ = nullptr;
    OFX::DoubleParam* phaseAstigAngleDeg_ = nullptr;
    OFX::DoubleParam* phaseComa_ = nullptr;
    OFX::DoubleParam* phaseComaAngleDeg_ = nullptr;
    OFX::DoubleParam* phaseSphericalAmount_ = nullptr;
    OFX::DoubleParam* phaseTrefoil_ = nullptr;
    OFX::DoubleParam* phaseTrefoilAngleDeg_ = nullptr;
    OFX::DoubleParam* phaseDefocus_ = nullptr;
    OFX::DoubleParam* phaseAstigmatism0_ = nullptr;
    OFX::DoubleParam* phaseAstigmatism45_ = nullptr;
    OFX::DoubleParam* phaseComaX_ = nullptr;
    OFX::DoubleParam* phaseComaY_ = nullptr;
    OFX::DoubleParam* phaseSpherical_ = nullptr;
    OFX::DoubleParam* phaseTrefoilX_ = nullptr;
    OFX::DoubleParam* phaseTrefoilY_ = nullptr;
    OFX::DoubleParam* phaseSecondaryAstigmatism0_ = nullptr;
    OFX::DoubleParam* phaseSecondaryAstigmatism45_ = nullptr;
    OFX::DoubleParam* phaseQuadrafoil0_ = nullptr;
    OFX::DoubleParam* phaseQuadrafoil45_ = nullptr;
    OFX::DoubleParam* phaseSecondaryComaX_ = nullptr;
    OFX::DoubleParam* phaseSecondaryComaY_ = nullptr;
    OFX::DoubleParam* pupilDecenterX_ = nullptr;
    OFX::DoubleParam* pupilDecenterY_ = nullptr;
    OFX::DoubleParam* phaseFieldStrength_ = nullptr;
    OFX::DoubleParam* phaseFieldEdgeBias_ = nullptr;
    OFX::DoubleParam* phaseFieldDefocus_ = nullptr;
    OFX::DoubleParam* phaseFieldAstigRadial_ = nullptr;
    OFX::DoubleParam* phaseFieldAstigTangential_ = nullptr;
    OFX::DoubleParam* phaseFieldComaRadial_ = nullptr;
    OFX::DoubleParam* phaseFieldComaTangential_ = nullptr;
    OFX::DoubleParam* phaseFieldSpherical_ = nullptr;
    OFX::DoubleParam* phaseFieldTrefoilRadial_ = nullptr;
    OFX::DoubleParam* phaseFieldTrefoilTangential_ = nullptr;
    OFX::DoubleParam* phaseFieldSecondaryAstigRadial_ = nullptr;
    OFX::DoubleParam* phaseFieldSecondaryAstigTangential_ = nullptr;
    OFX::DoubleParam* phaseFieldQuadrafoilRadial_ = nullptr;
    OFX::DoubleParam* phaseFieldQuadrafoilTangential_ = nullptr;
    OFX::DoubleParam* phaseFieldSecondaryComaRadial_ = nullptr;
    OFX::DoubleParam* phaseFieldSecondaryComaTangential_ = nullptr;
    OFX::ChoiceParam* spectralMode_ = nullptr;
    OFX::ChoiceParam* spectrumStyle_ = nullptr;
    OFX::DoubleParam* spectrumForce_ = nullptr;
    OFX::DoubleParam* spectrumSaturation_ = nullptr;
    OFX::BooleanParam* chromaticAffectsLuma_ = nullptr;
    OFX::DoubleParam* chromaticFocus_ = nullptr;
    OFX::DoubleParam* chromaticSpread_ = nullptr;
    OFX::DoubleParam* creativeFringe_ = nullptr;
    OFX::DoubleParam* scatterAmount_ = nullptr;
    OFX::DoubleParam* scatterRadius_ = nullptr;
    OFX::ChoiceParam* lookMode_ = nullptr;
    OFX::ChoiceParam* energyMode_ = nullptr;
    OFX::DoubleParam* effectGain_ = nullptr;
    OFX::BooleanParam* resolutionAware_ = nullptr;
    OFX::DoubleParam* coreCompensation_ = nullptr;
    OFX::DoubleParam* anisotropyEmphasis_ = nullptr;
    OFX::DoubleParam* coreGain_ = nullptr;
    OFX::DoubleParam* structureGain_ = nullptr;
    OFX::DoubleParam* coreShoulder_ = nullptr;
    OFX::DoubleParam* structureShoulder_ = nullptr;
    OFX::ChoiceParam* backendPreference_ = nullptr;
    OFX::ChoiceParam* debugView_ = nullptr;
    OFX::ChoiceParam* lensdiffPresetMenu_ = nullptr;
    OFX::GroupParam* compositeGroup_ = nullptr;
    OFX::GroupParam* compositeSplitGroup_ = nullptr;
    OFX::GroupParam* phaseGroup_ = nullptr;
    OFX::GroupParam* phasePrimaryGroup_ = nullptr;
    OFX::GroupParam* phaseAdvancedGroup_ = nullptr;
    OFX::GroupParam* phaseFieldGroup_ = nullptr;
    OFX::GroupParam* phaseFieldPrimaryGroup_ = nullptr;
    OFX::GroupParam* phaseFieldHigherOrderGroup_ = nullptr;
    OFX::GroupParam* phaseChromaticGroup_ = nullptr;
    OFX::GroupParam* phaseFinishingGroup_ = nullptr;

    bool suppressLensDiffPresetChangedHandling_ = false;
    bool suppressPupilResolutionChangedHandling_ = false;
    bool suppressPhaseSyncHandling_ = false;
    bool suppressEffectGainMirrorHandling_ = false;
    int lensdiffPresetMenuUserCount_ = 0;
    bool lensdiffPresetMenuHasCustom_ = false;
    int lensdiffPresetCustomIndex_ = -1;
    bool lensdiffPresetDisplayStateValid_ = false;
    LensDiffPresetSelection lensdiffPresetLastSelectedMenuSelection_ {};
    LensDiffPresetSelection lensdiffPresetLastExactSelection_ {};
    LensDiffPresetSelection lensdiffPresetLastDisplaySelection_ {};
    std::string lensdiffPresetLastDirtyFingerprint_;

    mutable std::mutex cacheMutex_;
    mutable LensDiffPsfBankCache cache_ {};
    mutable std::uint64_t cacheLatestStartedGeneration_ = 0;
    mutable std::uint64_t cacheLatestCommittedGeneration_ = 0;
};

bool lensdiffPresetSelectionsEqual(const LensDiffEffect::LensDiffPresetSelection& a,
                                   const LensDiffEffect::LensDiffPresetSelection& b) {
    return a.kind == b.kind &&
           a.builtinIndex == b.builtinIndex &&
           a.userIndex == b.userIndex &&
           a.modified == b.modified;
}

LensDiffParams LensDiffEffect::resolveParams(double time) const {
    LensDiffParams params {};
    params.threshold = threshold_->getValueAtTime(time);
    params.inputTransfer = inputTransferFromChoice(getChoiceValueAtTime(inputTransfer_, time));
    params.softnessStops = softnessStops_->getValueAtTime(time);
    params.pointEmphasis = pointEmphasis_->getValueAtTime(time);
    params.corePreserve = corePreserve_->getValueAtTime(time);
    params.extractionMode = extractionFromChoice(getChoiceValueAtTime(extractionMode_, time));
    params.customAperturePath = getStringValueAtTime(customAperturePath_, time);
    params.apertureMode = apertureFromChoice(getChoiceValueAtTime(apertureMode_, time),
                                             hasCustomAperturePath(params.customAperturePath));
    params.customApertureNormalize = customApertureNormalize_ ? customApertureNormalize_->getValueAtTime(time) : true;
    params.customApertureInvert = customApertureInvert_ ? customApertureInvert_->getValueAtTime(time) : false;
    params.bladeCount = bladeCount_->getValueAtTime(time);
    params.roundness = roundness_->getValueAtTime(time);
    params.rotationDeg = rotationDeg_->getValueAtTime(time);
    params.centralObstruction = centralObstruction_->getValueAtTime(time);
    params.vaneCount = vaneCount_->getValueAtTime(time);
    params.vaneThickness = vaneThickness_->getValueAtTime(time);
    params.apodizationMode = apodizationFromChoice(getChoiceValueAtTime(apodizationMode_, time));
    params.diffractionScalePx = diffractionScalePx_->getValueAtTime(time);
    params.pupilResolution = pupilResolution_ ? pupilResolution_->getValueAtTime(time) : 256;
    params.maxKernelRadiusPx = maxKernelRadiusPx_ ? maxKernelRadiusPx_->getValueAtTime(time) : 64.0;
    DisableLensDiffPhase(&params);
    const bool phaseSuiteEnabled = phaseEnabled_ && phaseEnabled_->getValueAtTime(time);
    if (phaseSuiteEnabled) {
        params.phaseMode = LensDiffPhaseMode::Enabled;
        params.phaseDefocus = phaseDefocus_ ? phaseDefocus_->getValueAtTime(time) : 0.0;
        params.phaseAstigmatism0 = phaseAstigmatism0_ ? phaseAstigmatism0_->getValueAtTime(time) : 0.0;
        params.phaseAstigmatism45 = phaseAstigmatism45_ ? phaseAstigmatism45_->getValueAtTime(time) : 0.0;
        params.phaseComaX = phaseComaX_ ? phaseComaX_->getValueAtTime(time) : 0.0;
        params.phaseComaY = phaseComaY_ ? phaseComaY_->getValueAtTime(time) : 0.0;
        params.phaseSpherical = phaseSpherical_ ? phaseSpherical_->getValueAtTime(time) : 0.0;
        params.phaseTrefoilX = phaseTrefoilX_ ? phaseTrefoilX_->getValueAtTime(time) : 0.0;
        params.phaseTrefoilY = phaseTrefoilY_ ? phaseTrefoilY_->getValueAtTime(time) : 0.0;
        params.phaseSecondaryAstigmatism0 =
            phaseSecondaryAstigmatism0_ ? phaseSecondaryAstigmatism0_->getValueAtTime(time) : 0.0;
        params.phaseSecondaryAstigmatism45 =
            phaseSecondaryAstigmatism45_ ? phaseSecondaryAstigmatism45_->getValueAtTime(time) : 0.0;
        params.phaseQuadrafoil0 = phaseQuadrafoil0_ ? phaseQuadrafoil0_->getValueAtTime(time) : 0.0;
        params.phaseQuadrafoil45 = phaseQuadrafoil45_ ? phaseQuadrafoil45_->getValueAtTime(time) : 0.0;
        params.phaseSecondaryComaX = phaseSecondaryComaX_ ? phaseSecondaryComaX_->getValueAtTime(time) : 0.0;
        params.phaseSecondaryComaY = phaseSecondaryComaY_ ? phaseSecondaryComaY_->getValueAtTime(time) : 0.0;
        params.pupilDecenterX = pupilDecenterX_ ? pupilDecenterX_->getValueAtTime(time) : 0.0;
        params.pupilDecenterY = pupilDecenterY_ ? pupilDecenterY_->getValueAtTime(time) : 0.0;
        params.phaseFieldStrength = phaseFieldStrength_ ? phaseFieldStrength_->getValueAtTime(time) : 0.0;
        params.phaseFieldEdgeBias = phaseFieldEdgeBias_ ? phaseFieldEdgeBias_->getValueAtTime(time) : 0.0;
        params.phaseFieldDefocus = phaseFieldDefocus_ ? phaseFieldDefocus_->getValueAtTime(time) : 0.0;
        params.phaseFieldAstigRadial = phaseFieldAstigRadial_ ? phaseFieldAstigRadial_->getValueAtTime(time) : 0.0;
        params.phaseFieldAstigTangential =
            phaseFieldAstigTangential_ ? phaseFieldAstigTangential_->getValueAtTime(time) : 0.0;
        params.phaseFieldComaRadial = phaseFieldComaRadial_ ? phaseFieldComaRadial_->getValueAtTime(time) : 0.0;
        params.phaseFieldComaTangential =
            phaseFieldComaTangential_ ? phaseFieldComaTangential_->getValueAtTime(time) : 0.0;
        params.phaseFieldSpherical = phaseFieldSpherical_ ? phaseFieldSpherical_->getValueAtTime(time) : 0.0;
        params.phaseFieldTrefoilRadial =
            phaseFieldTrefoilRadial_ ? phaseFieldTrefoilRadial_->getValueAtTime(time) : 0.0;
        params.phaseFieldTrefoilTangential =
            phaseFieldTrefoilTangential_ ? phaseFieldTrefoilTangential_->getValueAtTime(time) : 0.0;
        params.phaseFieldSecondaryAstigRadial =
            phaseFieldSecondaryAstigRadial_ ? phaseFieldSecondaryAstigRadial_->getValueAtTime(time) : 0.0;
        params.phaseFieldSecondaryAstigTangential =
            phaseFieldSecondaryAstigTangential_ ? phaseFieldSecondaryAstigTangential_->getValueAtTime(time) : 0.0;
        params.phaseFieldQuadrafoilRadial =
            phaseFieldQuadrafoilRadial_ ? phaseFieldQuadrafoilRadial_->getValueAtTime(time) : 0.0;
        params.phaseFieldQuadrafoilTangential =
            phaseFieldQuadrafoilTangential_ ? phaseFieldQuadrafoilTangential_->getValueAtTime(time) : 0.0;
        params.phaseFieldSecondaryComaRadial =
            phaseFieldSecondaryComaRadial_ ? phaseFieldSecondaryComaRadial_->getValueAtTime(time) : 0.0;
        params.phaseFieldSecondaryComaTangential =
            phaseFieldSecondaryComaTangential_ ? phaseFieldSecondaryComaTangential_->getValueAtTime(time) : 0.0;
    }
    params.spectralMode = spectralFromChoice(getChoiceValueAtTime(spectralMode_, time));
    params.spectrumStyle = spectrumStyleFromChoice(getChoiceValueAtTime(spectrumStyle_, time));
    params.spectrumForce = spectrumForce_->getValueAtTime(time);
    params.spectrumSaturation = spectrumSaturation_->getValueAtTime(time);
    params.chromaticAffectsLuma = chromaticAffectsLuma_->getValueAtTime(time);
    if (phaseSuiteEnabled) {
        params.chromaticFocus = chromaticFocus_ ? chromaticFocus_->getValueAtTime(time) : 0.0;
        params.chromaticSpread = chromaticSpread_ ? chromaticSpread_->getValueAtTime(time) : 0.0;
        params.creativeFringe = creativeFringe_ ? creativeFringe_->getValueAtTime(time) : 0.0;
        params.scatterAmount = scatterAmount_ ? scatterAmount_->getValueAtTime(time) : 0.0;
        params.scatterRadius = scatterRadius_ ? scatterRadius_->getValueAtTime(time) : 0.0;
    }
    params.lookMode = lookModeFromChoice(getChoiceValueAtTime(lookMode_, time));
    params.energyMode = energyModeFromChoice(getChoiceValueAtTime(energyMode_, time));
    params.effectGain = effectGain_->getValueAtTime(time);
    params.resolutionAware = resolutionAware_ ? resolutionAware_->getValueAtTime(time) : false;
    params.coreCompensation = coreCompensation_->getValueAtTime(time);
    params.anisotropyEmphasis = anisotropyEmphasis_->getValueAtTime(time);
    params.coreGain = coreGain_->getValueAtTime(time);
    params.structureGain = structureGain_->getValueAtTime(time);
    params.coreShoulder = coreShoulder_->getValueAtTime(time);
    params.structureShoulder = structureShoulder_->getValueAtTime(time);
    params.debugView = debugViewFromChoice(getChoiceValueAtTime(debugView_, time));
    return params;
}

LensDiffParams LensDiffEffect::resolveParams(double time, int frameShortSidePx) const {
    LensDiffParams params = resolveParams(time);
    params.frameShortSidePx = std::max(1, frameShortSidePx);
    return params;
}

LensDiffPresetValues LensDiffEffect::captureCurrentLensDiffPresetValues(double time) const {
    LensDiffPresetValues values = lensdiffFactoryPresetValues();
    values.simpleMode = isSimpleMode(time);
    values.inputTransfer = getChoiceValueAtTime(inputTransfer_, time);
    values.threshold = threshold_ ? threshold_->getValueAtTime(time) : values.threshold;
    values.softnessStops = softnessStops_ ? softnessStops_->getValueAtTime(time) : values.softnessStops;
    values.extractionMode = getChoiceValueAtTime(extractionMode_, time);
    values.pointEmphasis = pointEmphasis_ ? pointEmphasis_->getValueAtTime(time) : values.pointEmphasis;
    values.corePreserve = corePreserve_ ? corePreserve_->getValueAtTime(time) : values.corePreserve;

    const std::string customPath = getStringValueAtTime(customAperturePath_, time);
    const bool hasCustom = hasCustomAperturePath(customPath);
    const LensDiffApertureMode activeApertureMode =
        apertureFromChoice(getChoiceValueAtTime(apertureMode_, time), hasCustom);
    values.apertureMode = static_cast<int>(activeApertureMode);
    if (activeApertureMode == LensDiffApertureMode::Custom) {
        values.customAperturePath = customPath;
        values.customApertureNormalize =
            customApertureNormalize_ ? customApertureNormalize_->getValueAtTime(time) : true;
        values.customApertureInvert =
            customApertureInvert_ ? customApertureInvert_->getValueAtTime(time) : false;
    } else {
        values.customAperturePath.clear();
        values.customApertureNormalize = true;
        values.customApertureInvert = false;
    }

    values.bladeCount = bladeCount_ ? bladeCount_->getValueAtTime(time) : values.bladeCount;
    values.roundness = roundness_ ? roundness_->getValueAtTime(time) : values.roundness;
    values.rotationDeg = rotationDeg_ ? rotationDeg_->getValueAtTime(time) : values.rotationDeg;
    values.centralObstruction = centralObstruction_ ? centralObstruction_->getValueAtTime(time) : values.centralObstruction;
    values.vaneCount = vaneCount_ ? vaneCount_->getValueAtTime(time) : values.vaneCount;
    values.vaneThickness = vaneThickness_ ? vaneThickness_->getValueAtTime(time) : values.vaneThickness;
    values.apodizationMode = getChoiceValueAtTime(apodizationMode_, time);
    values.diffractionScalePx = diffractionScalePx_ ? diffractionScalePx_->getValueAtTime(time) : values.diffractionScalePx;
    values.pupilResolution = pupilResolution_ ? pupilResolution_->getValueAtTime(time) : values.pupilResolution;
    values.maxKernelRadiusPx = maxKernelRadiusPx_ ? maxKernelRadiusPx_->getValueAtTime(time) : values.maxKernelRadiusPx;
    values.phaseEnabled = phaseEnabled_ ? phaseEnabled_->getValueAtTime(time) : values.phaseEnabled;
    values.phaseDefocus = phaseDefocus_ ? phaseDefocus_->getValueAtTime(time) : values.phaseDefocus;
    values.phaseAstigmatism0 = phaseAstigmatism0_ ? phaseAstigmatism0_->getValueAtTime(time) : values.phaseAstigmatism0;
    values.phaseAstigmatism45 = phaseAstigmatism45_ ? phaseAstigmatism45_->getValueAtTime(time) : values.phaseAstigmatism45;
    values.phaseComaX = phaseComaX_ ? phaseComaX_->getValueAtTime(time) : values.phaseComaX;
    values.phaseComaY = phaseComaY_ ? phaseComaY_->getValueAtTime(time) : values.phaseComaY;
    values.phaseSpherical = phaseSpherical_ ? phaseSpherical_->getValueAtTime(time) : values.phaseSpherical;
    values.phaseTrefoilX = phaseTrefoilX_ ? phaseTrefoilX_->getValueAtTime(time) : values.phaseTrefoilX;
    values.phaseTrefoilY = phaseTrefoilY_ ? phaseTrefoilY_->getValueAtTime(time) : values.phaseTrefoilY;
    values.phaseSecondaryAstigmatism0 =
        phaseSecondaryAstigmatism0_ ? phaseSecondaryAstigmatism0_->getValueAtTime(time) : values.phaseSecondaryAstigmatism0;
    values.phaseSecondaryAstigmatism45 =
        phaseSecondaryAstigmatism45_ ? phaseSecondaryAstigmatism45_->getValueAtTime(time) : values.phaseSecondaryAstigmatism45;
    values.phaseQuadrafoil0 = phaseQuadrafoil0_ ? phaseQuadrafoil0_->getValueAtTime(time) : values.phaseQuadrafoil0;
    values.phaseQuadrafoil45 = phaseQuadrafoil45_ ? phaseQuadrafoil45_->getValueAtTime(time) : values.phaseQuadrafoil45;
    values.phaseSecondaryComaX =
        phaseSecondaryComaX_ ? phaseSecondaryComaX_->getValueAtTime(time) : values.phaseSecondaryComaX;
    values.phaseSecondaryComaY =
        phaseSecondaryComaY_ ? phaseSecondaryComaY_->getValueAtTime(time) : values.phaseSecondaryComaY;
    values.pupilDecenterX = pupilDecenterX_ ? pupilDecenterX_->getValueAtTime(time) : values.pupilDecenterX;
    values.pupilDecenterY = pupilDecenterY_ ? pupilDecenterY_->getValueAtTime(time) : values.pupilDecenterY;
    values.phaseFieldStrength = phaseFieldStrength_ ? phaseFieldStrength_->getValueAtTime(time) : values.phaseFieldStrength;
    values.phaseFieldEdgeBias = phaseFieldEdgeBias_ ? phaseFieldEdgeBias_->getValueAtTime(time) : values.phaseFieldEdgeBias;
    values.phaseFieldDefocus = phaseFieldDefocus_ ? phaseFieldDefocus_->getValueAtTime(time) : values.phaseFieldDefocus;
    values.phaseFieldAstigRadial =
        phaseFieldAstigRadial_ ? phaseFieldAstigRadial_->getValueAtTime(time) : values.phaseFieldAstigRadial;
    values.phaseFieldAstigTangential =
        phaseFieldAstigTangential_ ? phaseFieldAstigTangential_->getValueAtTime(time) : values.phaseFieldAstigTangential;
    values.phaseFieldComaRadial =
        phaseFieldComaRadial_ ? phaseFieldComaRadial_->getValueAtTime(time) : values.phaseFieldComaRadial;
    values.phaseFieldComaTangential =
        phaseFieldComaTangential_ ? phaseFieldComaTangential_->getValueAtTime(time) : values.phaseFieldComaTangential;
    values.phaseFieldSpherical =
        phaseFieldSpherical_ ? phaseFieldSpherical_->getValueAtTime(time) : values.phaseFieldSpherical;
    values.phaseFieldTrefoilRadial =
        phaseFieldTrefoilRadial_ ? phaseFieldTrefoilRadial_->getValueAtTime(time) : values.phaseFieldTrefoilRadial;
    values.phaseFieldTrefoilTangential =
        phaseFieldTrefoilTangential_ ? phaseFieldTrefoilTangential_->getValueAtTime(time) : values.phaseFieldTrefoilTangential;
    values.phaseFieldSecondaryAstigRadial =
        phaseFieldSecondaryAstigRadial_ ? phaseFieldSecondaryAstigRadial_->getValueAtTime(time) : values.phaseFieldSecondaryAstigRadial;
    values.phaseFieldSecondaryAstigTangential =
        phaseFieldSecondaryAstigTangential_ ? phaseFieldSecondaryAstigTangential_->getValueAtTime(time) : values.phaseFieldSecondaryAstigTangential;
    values.phaseFieldQuadrafoilRadial =
        phaseFieldQuadrafoilRadial_ ? phaseFieldQuadrafoilRadial_->getValueAtTime(time) : values.phaseFieldQuadrafoilRadial;
    values.phaseFieldQuadrafoilTangential =
        phaseFieldQuadrafoilTangential_ ? phaseFieldQuadrafoilTangential_->getValueAtTime(time) : values.phaseFieldQuadrafoilTangential;
    values.phaseFieldSecondaryComaRadial =
        phaseFieldSecondaryComaRadial_ ? phaseFieldSecondaryComaRadial_->getValueAtTime(time) : values.phaseFieldSecondaryComaRadial;
    values.phaseFieldSecondaryComaTangential =
        phaseFieldSecondaryComaTangential_ ? phaseFieldSecondaryComaTangential_->getValueAtTime(time) : values.phaseFieldSecondaryComaTangential;

    values.spectralMode = getChoiceValueAtTime(spectralMode_, time);
    values.spectrumStyle = getChoiceValueAtTime(spectrumStyle_, time);
    values.spectrumForce = spectrumForce_ ? spectrumForce_->getValueAtTime(time) : values.spectrumForce;
    values.spectrumSaturation = spectrumSaturation_ ? spectrumSaturation_->getValueAtTime(time) : values.spectrumSaturation;
    values.chromaticAffectsLuma =
        chromaticAffectsLuma_ ? chromaticAffectsLuma_->getValueAtTime(time) : values.chromaticAffectsLuma;
    values.chromaticFocus = chromaticFocus_ ? chromaticFocus_->getValueAtTime(time) : values.chromaticFocus;
    values.chromaticSpread = chromaticSpread_ ? chromaticSpread_->getValueAtTime(time) : values.chromaticSpread;
    values.creativeFringe = creativeFringe_ ? creativeFringe_->getValueAtTime(time) : values.creativeFringe;
    values.scatterAmount = scatterAmount_ ? scatterAmount_->getValueAtTime(time) : values.scatterAmount;
    values.scatterRadius = scatterRadius_ ? scatterRadius_->getValueAtTime(time) : values.scatterRadius;

    values.lookMode = getChoiceValueAtTime(lookMode_, time);
    values.energyMode = getChoiceValueAtTime(energyMode_, time);
    values.effectGain = effectGain_ ? effectGain_->getValueAtTime(time) : values.effectGain;
    values.resolutionAware = resolutionAware_ ? resolutionAware_->getValueAtTime(time) : values.resolutionAware;
    values.coreCompensation = coreCompensation_ ? coreCompensation_->getValueAtTime(time) : values.coreCompensation;
    values.anisotropyEmphasis = anisotropyEmphasis_ ? anisotropyEmphasis_->getValueAtTime(time) : values.anisotropyEmphasis;
    values.coreGain = coreGain_ ? coreGain_->getValueAtTime(time) : values.coreGain;
    values.structureGain = structureGain_ ? structureGain_->getValueAtTime(time) : values.structureGain;
    values.coreShoulder = coreShoulder_ ? coreShoulder_->getValueAtTime(time) : values.coreShoulder;
    values.structureShoulder = structureShoulder_ ? structureShoulder_->getValueAtTime(time) : values.structureShoulder;
    values.backendPreference = getChoiceValueAtTime(backendPreference_, time);
    values.debugView = getChoiceValueAtTime(debugView_, time);
    return values;
}

LensDiffPresetValues LensDiffEffect::captureCurrentLensDiffPresetDirtyValues(double time) const {
    return lensdiffPresetValuesForDirtyComparison(captureCurrentLensDiffPresetValues(time));
}

void LensDiffEffect::writeOpticsPresetValuesToParams(const LensDiffPresetValues& values) {
    if (customAperturePath_) customAperturePath_->setValue(values.customAperturePath);
    if (customApertureNormalize_) customApertureNormalize_->setValue(values.customApertureNormalize);
    if (customApertureInvert_) customApertureInvert_->setValue(values.customApertureInvert);

    const LensDiffApertureMode apertureMode = sanitizePresetApertureMode(values.apertureMode);
    if (apertureMode == LensDiffApertureMode::Custom && hasCustomAperturePath(values.customAperturePath)) {
        syncCustomApertureUi(true);
    } else {
        syncCustomApertureUi(false);
        if (apertureMode_) {
            apertureMode_->setValue(apertureChoiceIndexForPresetMode(apertureMode, false));
        }
    }

    if (bladeCount_) bladeCount_->setValue(values.bladeCount);
    if (roundness_) roundness_->setValue(values.roundness);
    if (rotationDeg_) rotationDeg_->setValue(values.rotationDeg);
    if (centralObstruction_) centralObstruction_->setValue(values.centralObstruction);
    if (vaneCount_) vaneCount_->setValue(values.vaneCount);
    if (vaneThickness_) vaneThickness_->setValue(values.vaneThickness);
    if (apodizationMode_) apodizationMode_->setValue(values.apodizationMode);
    if (diffractionScalePx_) diffractionScalePx_->setValue(values.diffractionScalePx);
    if (pupilResolution_) pupilResolution_->setValue(values.pupilResolution);
    syncPupilResolutionUi(false);
    if (maxKernelRadiusPx_) maxKernelRadiusPx_->setValue(values.maxKernelRadiusPx);
}

void LensDiffEffect::writePhasePresetValuesToParams(const LensDiffPresetValues& values) {
    if (phaseEnabled_) phaseEnabled_->setValue(values.phaseEnabled);
    if (phaseDefocus_) phaseDefocus_->setValue(values.phaseDefocus);
    if (phaseAstigmatism0_) phaseAstigmatism0_->setValue(values.phaseAstigmatism0);
    if (phaseAstigmatism45_) phaseAstigmatism45_->setValue(values.phaseAstigmatism45);
    if (phaseComaX_) phaseComaX_->setValue(values.phaseComaX);
    if (phaseComaY_) phaseComaY_->setValue(values.phaseComaY);
    if (phaseSpherical_) phaseSpherical_->setValue(values.phaseSpherical);
    if (phaseTrefoilX_) phaseTrefoilX_->setValue(values.phaseTrefoilX);
    if (phaseTrefoilY_) phaseTrefoilY_->setValue(values.phaseTrefoilY);
    if (phaseSecondaryAstigmatism0_) phaseSecondaryAstigmatism0_->setValue(values.phaseSecondaryAstigmatism0);
    if (phaseSecondaryAstigmatism45_) phaseSecondaryAstigmatism45_->setValue(values.phaseSecondaryAstigmatism45);
    if (phaseQuadrafoil0_) phaseQuadrafoil0_->setValue(values.phaseQuadrafoil0);
    if (phaseQuadrafoil45_) phaseQuadrafoil45_->setValue(values.phaseQuadrafoil45);
    if (phaseSecondaryComaX_) phaseSecondaryComaX_->setValue(values.phaseSecondaryComaX);
    if (phaseSecondaryComaY_) phaseSecondaryComaY_->setValue(values.phaseSecondaryComaY);
    if (pupilDecenterX_) pupilDecenterX_->setValue(values.pupilDecenterX);
    if (pupilDecenterY_) pupilDecenterY_->setValue(values.pupilDecenterY);
    if (phaseFieldStrength_) phaseFieldStrength_->setValue(values.phaseFieldStrength);
    if (phaseFieldEdgeBias_) phaseFieldEdgeBias_->setValue(values.phaseFieldEdgeBias);
    if (phaseFieldDefocus_) phaseFieldDefocus_->setValue(values.phaseFieldDefocus);
    if (phaseFieldAstigRadial_) phaseFieldAstigRadial_->setValue(values.phaseFieldAstigRadial);
    if (phaseFieldAstigTangential_) phaseFieldAstigTangential_->setValue(values.phaseFieldAstigTangential);
    if (phaseFieldComaRadial_) phaseFieldComaRadial_->setValue(values.phaseFieldComaRadial);
    if (phaseFieldComaTangential_) phaseFieldComaTangential_->setValue(values.phaseFieldComaTangential);
    if (phaseFieldSpherical_) phaseFieldSpherical_->setValue(values.phaseFieldSpherical);
    if (phaseFieldTrefoilRadial_) phaseFieldTrefoilRadial_->setValue(values.phaseFieldTrefoilRadial);
    if (phaseFieldTrefoilTangential_) phaseFieldTrefoilTangential_->setValue(values.phaseFieldTrefoilTangential);
    if (phaseFieldSecondaryAstigRadial_) phaseFieldSecondaryAstigRadial_->setValue(values.phaseFieldSecondaryAstigRadial);
    if (phaseFieldSecondaryAstigTangential_) phaseFieldSecondaryAstigTangential_->setValue(values.phaseFieldSecondaryAstigTangential);
    if (phaseFieldQuadrafoilRadial_) phaseFieldQuadrafoilRadial_->setValue(values.phaseFieldQuadrafoilRadial);
    if (phaseFieldQuadrafoilTangential_) phaseFieldQuadrafoilTangential_->setValue(values.phaseFieldQuadrafoilTangential);
    if (phaseFieldSecondaryComaRadial_) phaseFieldSecondaryComaRadial_->setValue(values.phaseFieldSecondaryComaRadial);
    if (phaseFieldSecondaryComaTangential_) phaseFieldSecondaryComaTangential_->setValue(values.phaseFieldSecondaryComaTangential);
    syncPhaseMacrosFromCoefficients();

    if (chromaticFocus_) chromaticFocus_->setValue(values.chromaticFocus);
    if (chromaticSpread_) chromaticSpread_->setValue(values.chromaticSpread);
    if (creativeFringe_) creativeFringe_->setValue(values.creativeFringe);
    if (scatterAmount_) scatterAmount_->setValue(values.scatterAmount);
    if (scatterRadius_) scatterRadius_->setValue(values.scatterRadius);
}

void LensDiffEffect::writeSpectrumPresetValuesToParams(const LensDiffPresetValues& values) {
    if (spectralMode_) spectralMode_->setValue(values.spectralMode);
    if (spectrumStyle_) spectrumStyle_->setValue(values.spectrumStyle);
    if (spectrumForce_) spectrumForce_->setValue(values.spectrumForce);
    if (spectrumSaturation_) spectrumSaturation_->setValue(values.spectrumSaturation);
    if (chromaticAffectsLuma_) chromaticAffectsLuma_->setValue(values.chromaticAffectsLuma);
}

void LensDiffEffect::writeCompositePresetValuesToParams(const LensDiffPresetValues& values) {
    if (lookMode_) lookMode_->setValue(values.lookMode);
    if (energyMode_) energyMode_->setValue(values.energyMode);
    if (effectGain_ || simpleEffectGain_) {
        BoolScope scope(suppressEffectGainMirrorHandling_);
        if (effectGain_) effectGain_->setValue(values.effectGain);
        if (simpleEffectGain_) simpleEffectGain_->setValue(values.effectGain);
    }
    if (resolutionAware_) resolutionAware_->setValue(values.resolutionAware);
    if (coreCompensation_) coreCompensation_->setValue(values.coreCompensation);
    if (anisotropyEmphasis_) anisotropyEmphasis_->setValue(values.anisotropyEmphasis);
    if (coreGain_) coreGain_->setValue(values.coreGain);
    if (structureGain_) structureGain_->setValue(values.structureGain);
    if (coreShoulder_) coreShoulder_->setValue(values.coreShoulder);
    if (structureShoulder_) structureShoulder_->setValue(values.structureShoulder);
}

void LensDiffEffect::writeLensDiffPresetValuesToParams(const LensDiffPresetValues& values) {
    if (inputTransfer_) inputTransfer_->setValue(values.inputTransfer);
    if (threshold_) threshold_->setValue(values.threshold);
    if (softnessStops_) softnessStops_->setValue(values.softnessStops);
    if (extractionMode_) extractionMode_->setValue(values.extractionMode);
    if (pointEmphasis_) pointEmphasis_->setValue(values.pointEmphasis);
    if (corePreserve_) corePreserve_->setValue(values.corePreserve);
    writeOpticsPresetValuesToParams(values);
    writePhasePresetValuesToParams(values);
    writeSpectrumPresetValuesToParams(values);
    writeCompositePresetValuesToParams(values);
    if (backendPreference_) backendPreference_->setValue(values.backendPreference);
    if (debugView_) debugView_->setValue(values.debugView);

    updateCustomApertureControlVisibility();
    updatePhaseControlVisibility();
    updateCustomApertureStatus();
}

LensDiffEffect::LensDiffPresetSelection LensDiffEffect::matchingLensDiffPresetSelection(double time) const {
    return matchingLensDiffPresetSelection(captureCurrentLensDiffPresetDirtyValues(time));
}

LensDiffEffect::LensDiffPresetSelection LensDiffEffect::matchingLensDiffPresetSelection(
    const LensDiffPresetValues& current) const {
    std::lock_guard<std::mutex> lock(lensdiffPresetMutex());
    ensureLensDiffPresetStoreLoadedLocked();
    if (lensdiffPresetValuesEqual(current, lensdiffPresetValuesForDirtyComparison(lensdiffPresetStore().defaultPreset.values))) {
        return {};
    }
    const auto& builtins = lensdiffBuiltinPresets();
    for (int i = 0; i < static_cast<int>(builtins.size()); ++i) {
        if (lensdiffPresetValuesEqual(current,
                                      lensdiffPresetValuesForDirtyComparison(
                                          builtins[static_cast<std::size_t>(i)].values))) {
            LensDiffPresetSelection selection {};
            selection.kind = LensDiffPresetSelection::Kind::Builtin;
            selection.builtinIndex = i;
            return selection;
        }
    }
    for (int i = 0; i < static_cast<int>(lensdiffPresetStore().userPresets.size()); ++i) {
        if (lensdiffPresetValuesEqual(
                current,
                lensdiffPresetValuesForDirtyComparison(
                    lensdiffPresetStore().userPresets[static_cast<std::size_t>(i)].values))) {
            LensDiffPresetSelection selection {};
            selection.kind = LensDiffPresetSelection::Kind::User;
            selection.userIndex = i;
            return selection;
        }
    }
    LensDiffPresetSelection selection {};
    selection.kind = LensDiffPresetSelection::Kind::Custom;
    return selection;
}

LensDiffEffect::LensDiffPresetSelection LensDiffEffect::resolvedLensDiffPresetSelection(double time) const {
    return resolvedLensDiffPresetSelection(matchingLensDiffPresetSelection(time), selectedLensDiffPresetFromMenu(time));
}

LensDiffEffect::LensDiffPresetSelection LensDiffEffect::resolvedLensDiffPresetSelection(
    const LensDiffPresetSelection& exact,
    const LensDiffPresetSelection& currentMenuSelection) const {
    if (exact.kind != LensDiffPresetSelection::Kind::Custom) return exact;
    if (currentMenuSelection.kind == LensDiffPresetSelection::Kind::Default ||
        currentMenuSelection.kind == LensDiffPresetSelection::Kind::Builtin ||
        currentMenuSelection.kind == LensDiffPresetSelection::Kind::User) {
        LensDiffPresetSelection modified = currentMenuSelection;
        modified.modified = true;
        return modified;
    }
    return exact;
}

LensDiffEffect::LensDiffPresetSelection LensDiffEffect::selectedLensDiffPresetFromMenu(double time) const {
    LensDiffPresetSelection selection {};
    if (!lensdiffPresetMenu_) return selection;

    int selectedIndex = 0;
    lensdiffPresetMenu_->getValueAtTime(time, selectedIndex);
    if (selectedIndex == 0) return selection;
    const int builtinCount = static_cast<int>(lensdiffBuiltinPresets().size());
    if (selectedIndex >= 1 && selectedIndex <= builtinCount) {
        selection.kind = LensDiffPresetSelection::Kind::Builtin;
        selection.builtinIndex = selectedIndex - 1;
        return selection;
    }
    if (lensdiffPresetMenuHasCustom_ && selectedIndex == lensdiffPresetCustomIndex_) {
        selection.kind = LensDiffPresetSelection::Kind::Custom;
        return selection;
    }
    const int firstUserIndex = builtinCount + 1;
    if (selectedIndex >= firstUserIndex && selectedIndex < firstUserIndex + lensdiffPresetMenuUserCount_) {
        selection.kind = LensDiffPresetSelection::Kind::User;
        selection.userIndex = selectedIndex - firstUserIndex;
        return selection;
    }
    selection.kind = LensDiffPresetSelection::Kind::Custom;
    return selection;
}

void LensDiffEffect::rebuildLensDiffPresetMenu(double time, const LensDiffPresetSelection& selection) {
    auto* param = fetchChoiceParam("lensdiffPresetMenu");
    if (!param) return;

    param->resetOptions();
    if (selection.kind == LensDiffPresetSelection::Kind::Default && selection.modified) {
        param->appendOption(std::string(kLensDiffPresetDefaultName) + " (Modified)");
    } else {
        param->appendOption(kLensDiffPresetDefaultName);
    }
    const auto& builtins = lensdiffBuiltinPresets();
    for (int i = 0; i < static_cast<int>(builtins.size()); ++i) {
        const std::string builtinName = builtins[static_cast<std::size_t>(i)].name;
        if (selection.kind == LensDiffPresetSelection::Kind::Builtin &&
            selection.modified &&
            selection.builtinIndex == i) {
            param->appendOption(builtinName + " (Modified)");
        } else {
            param->appendOption(builtinName);
        }
    }

    std::vector<std::string> names;
    {
        std::lock_guard<std::mutex> lock(lensdiffPresetMutex());
        ensureLensDiffPresetStoreLoadedLocked();
        names.reserve(lensdiffPresetStore().userPresets.size());
        for (const auto& preset : lensdiffPresetStore().userPresets) {
            names.push_back(lensdiffMenuLabelForUserPresetName(preset.name));
        }
    }
    for (int i = 0; i < static_cast<int>(names.size()); ++i) {
        if (selection.kind == LensDiffPresetSelection::Kind::User && selection.modified && selection.userIndex == i) {
            param->appendOption(names[static_cast<std::size_t>(i)] + " (Modified)");
        } else {
            param->appendOption(names[static_cast<std::size_t>(i)]);
        }
    }

    lensdiffPresetMenuUserCount_ = static_cast<int>(names.size());
    lensdiffPresetMenuHasCustom_ = (selection.kind == LensDiffPresetSelection::Kind::Custom);
    lensdiffPresetCustomIndex_ = -1;
    int selectedIndex = 0;
    if (selection.kind == LensDiffPresetSelection::Kind::Builtin &&
        selection.builtinIndex >= 0 &&
        selection.builtinIndex < static_cast<int>(builtins.size())) {
        selectedIndex = selection.builtinIndex + 1;
    } else if (selection.kind == LensDiffPresetSelection::Kind::User &&
        selection.userIndex >= 0 &&
        selection.userIndex < lensdiffPresetMenuUserCount_) {
        selectedIndex = static_cast<int>(builtins.size()) + selection.userIndex + 1;
    } else if (selection.kind == LensDiffPresetSelection::Kind::Custom) {
        param->appendOption(kLensDiffPresetCustomLabel);
        lensdiffPresetCustomIndex_ = static_cast<int>(builtins.size()) + lensdiffPresetMenuUserCount_ + 1;
        selectedIndex = lensdiffPresetCustomIndex_;
    }
    BoolScope scope(suppressLensDiffPresetChangedHandling_);
    param->setValue(selectedIndex);
}

void LensDiffEffect::refreshLensDiffPresetMenuMetadataFromStore() {
    std::lock_guard<std::mutex> lock(lensdiffPresetMutex());
    ensureLensDiffPresetStoreLoadedLocked();
    lensdiffPresetMenuUserCount_ = static_cast<int>(lensdiffPresetStore().userPresets.size());
    lensdiffPresetMenuHasCustom_ = false;
    lensdiffPresetCustomIndex_ = -1;
}

void LensDiffEffect::updateLensDiffPresetActionState(double time) {
    const LensDiffPresetSelection selection = selectedLensDiffPresetFromMenu(time);
    const bool userSelected = selection.kind == LensDiffPresetSelection::Kind::User;
    const bool presetResetEnabled = selection.kind != LensDiffPresetSelection::Kind::Custom;
    if (auto* p = fetchPushButtonParam("lensdiffPresetUpdate")) p->setEnabled(userSelected);
    if (auto* p = fetchPushButtonParam("lensdiffPresetRename")) p->setEnabled(userSelected);
    if (auto* p = fetchPushButtonParam("lensdiffPresetDelete")) p->setEnabled(userSelected);
    if (auto* p = fetchPushButtonParam("opticsReset")) p->setEnabled(presetResetEnabled);
    if (auto* p = fetchPushButtonParam("phaseReset")) p->setEnabled(presetResetEnabled);
    if (auto* p = fetchPushButtonParam("spectrumReset")) p->setEnabled(presetResetEnabled);
    if (auto* p = fetchPushButtonParam("compositeReset")) p->setEnabled(presetResetEnabled);
}

void LensDiffEffect::syncLensDiffPresetMenuState(double time) {
    const LensDiffPresetValues dirtyValues = captureCurrentLensDiffPresetDirtyValues(time);
    const std::string dirtyFingerprint = lensdiffPresetValuesAsJson(dirtyValues);
    const LensDiffPresetSelection exactSelection = matchingLensDiffPresetSelection(dirtyValues);
    const LensDiffPresetSelection currentMenuSelection = selectedLensDiffPresetFromMenu(time);
    const LensDiffPresetSelection displaySelection =
        resolvedLensDiffPresetSelection(exactSelection, currentMenuSelection);
    const bool stateUnchanged =
        lensdiffPresetDisplayStateValid_ &&
        dirtyFingerprint == lensdiffPresetLastDirtyFingerprint_ &&
        lensdiffPresetSelectionsEqual(currentMenuSelection, lensdiffPresetLastSelectedMenuSelection_) &&
        lensdiffPresetSelectionsEqual(exactSelection, lensdiffPresetLastExactSelection_) &&
        lensdiffPresetSelectionsEqual(displaySelection, lensdiffPresetLastDisplaySelection_);
    if (stateUnchanged) {
        updateLensDiffPresetActionState(time);
        return;
    }
    const bool displayChanged =
        !lensdiffPresetDisplayStateValid_ ||
        !lensdiffPresetSelectionsEqual(displaySelection, lensdiffPresetLastDisplaySelection_);
    if (displayChanged) {
        rebuildLensDiffPresetMenu(time, displaySelection);
    }
    lensdiffPresetDisplayStateValid_ = true;
    lensdiffPresetLastDirtyFingerprint_ = dirtyFingerprint;
    lensdiffPresetLastSelectedMenuSelection_ = currentMenuSelection;
    lensdiffPresetLastExactSelection_ = exactSelection;
    lensdiffPresetLastDisplaySelection_ = displaySelection;
    updateLensDiffPresetActionState(time);
}

void LensDiffEffect::syncLensDiffPresetMenuFromDisk(double time,
                                                    const std::optional<LensDiffPresetSelection>& preferred) {
    {
        std::lock_guard<std::mutex> lock(lensdiffPresetMutex());
        reloadLensDiffPresetStoreFromDiskLocked();
    }
    refreshLensDiffPresetMenuMetadataFromStore();
    const LensDiffPresetSelection selection = preferred ? *preferred : resolvedLensDiffPresetSelection(time);
    rebuildLensDiffPresetMenu(time, selection);
    const LensDiffPresetValues dirtyValues = captureCurrentLensDiffPresetDirtyValues(time);
    lensdiffPresetDisplayStateValid_ = true;
    lensdiffPresetLastDirtyFingerprint_ = lensdiffPresetValuesAsJson(dirtyValues);
    lensdiffPresetLastSelectedMenuSelection_ = selectedLensDiffPresetFromMenu(time);
    lensdiffPresetLastExactSelection_ = matchingLensDiffPresetSelection(dirtyValues);
    lensdiffPresetLastDisplaySelection_ = selection;
    updateLensDiffPresetActionState(time);
}

void LensDiffEffect::finalizeLensDiffPresetApplication(double /*time*/) {
    syncEffectGainMirror(false);
    refreshDynamicControlVisibility();
    updateCustomApertureStatus();
    syncPupilResolutionUi(false);
}

bool LensDiffEffect::resolveLensDiffPresetSelectionValues(const LensDiffPresetSelection& selection,
                                                          LensDiffPresetValues* values,
                                                          std::string* presetName) const {
    if (!values || selection.kind == LensDiffPresetSelection::Kind::Custom) return false;

    *values = lensdiffFactoryPresetValues();
    if (presetName) presetName->clear();

    std::lock_guard<std::mutex> lock(lensdiffPresetMutex());
    ensureLensDiffPresetStoreLoadedLocked();
    if (selection.kind == LensDiffPresetSelection::Kind::Default) {
        *values = lensdiffPresetStore().defaultPreset.values;
        if (presetName) *presetName = lensdiffPresetStore().defaultPreset.name;
        return true;
    }
    if (selection.kind == LensDiffPresetSelection::Kind::Builtin) {
        const auto* builtin = lensdiffBuiltinPresetByIndex(selection.builtinIndex);
        if (!builtin) return false;
        *values = builtin->values;
        if (presetName) *presetName = builtin->name;
        return true;
    }
    if (selection.userIndex >= 0 &&
        selection.userIndex < static_cast<int>(lensdiffPresetStore().userPresets.size())) {
        const auto& preset = lensdiffPresetStore().userPresets[static_cast<std::size_t>(selection.userIndex)];
        *values = preset.values;
        if (presetName) *presetName = preset.name;
        return true;
    }
    return false;
}

void LensDiffEffect::applyLensDiffPresetSelection(double time,
                                                  const LensDiffPresetSelection& selection,
                                                  const char* /*reasonParam*/) {
    if (selection.kind == LensDiffPresetSelection::Kind::Custom) return;

    LensDiffPresetValues values = lensdiffFactoryPresetValues();
    std::string presetName;
    if (!resolveLensDiffPresetSelectionValues(selection, &values, &presetName)) return;

    BoolScope scope(suppressLensDiffPresetChangedHandling_);
    writeLensDiffPresetValuesToParams(values);
    finalizeLensDiffPresetApplication(time);
    syncLensDiffPresetMenuFromDisk(time, selection);
    if (selection.kind == LensDiffPresetSelection::Kind::User) {
        if (auto* p = fetchStringParam("lensdiffPresetName")) p->setValue(presetName);
    }
}

void LensDiffEffect::applySelectedLensDiffPreset(double time) {
    applyLensDiffPresetSelection(time, selectedLensDiffPresetFromMenu(time), "lensdiffPresetMenu");
}

bool LensDiffEffect::isLensDiffPresetManagedParam(const std::string& paramName) const {
    return paramName == "simpleModeState" ||
           paramName == "inputTransfer" ||
           paramName == "threshold" ||
           paramName == "softnessStops" ||
           paramName == "extractionMode" ||
           paramName == "pointEmphasis" ||
           paramName == "corePreserve" ||
           paramName == "apertureMode" ||
           paramName == "customAperturePath" ||
           paramName == "customApertureNormalize" ||
           paramName == "customApertureInvert" ||
           paramName == "bladeCount" ||
           paramName == "roundness" ||
           paramName == "rotationDeg" ||
           paramName == "centralObstruction" ||
           paramName == "vaneCount" ||
           paramName == "vaneThickness" ||
           paramName == "apodizationMode" ||
           paramName == "diffractionScalePx" ||
           paramName == "pupilResolution" ||
           paramName == "pupilResolutionChoice" ||
           paramName == "maxKernelRadiusPx" ||
           paramName == "phaseEnabled" ||
           paramName == "phaseFocus" ||
           paramName == "phaseAstigmatism" ||
           paramName == "phaseAstigAngleDeg" ||
           paramName == "phaseComa" ||
           paramName == "phaseComaAngleDeg" ||
           paramName == "phaseSphericalAmount" ||
           paramName == "phaseTrefoil" ||
           paramName == "phaseTrefoilAngleDeg" ||
           paramName == "phaseDefocus" ||
           paramName == "phaseAstigmatism0" ||
           paramName == "phaseAstigmatism45" ||
           paramName == "phaseComaX" ||
           paramName == "phaseComaY" ||
           paramName == "phaseSpherical" ||
           paramName == "phaseTrefoilX" ||
           paramName == "phaseTrefoilY" ||
           paramName == "phaseSecondaryAstigmatism0" ||
           paramName == "phaseSecondaryAstigmatism45" ||
           paramName == "phaseQuadrafoil0" ||
           paramName == "phaseQuadrafoil45" ||
           paramName == "phaseSecondaryComaX" ||
           paramName == "phaseSecondaryComaY" ||
           paramName == "pupilDecenterX" ||
           paramName == "pupilDecenterY" ||
           paramName == "phaseFieldStrength" ||
           paramName == "phaseFieldEdgeBias" ||
           paramName == "phaseFieldDefocus" ||
           paramName == "phaseFieldAstigRadial" ||
           paramName == "phaseFieldAstigTangential" ||
           paramName == "phaseFieldComaRadial" ||
           paramName == "phaseFieldComaTangential" ||
           paramName == "phaseFieldSpherical" ||
           paramName == "phaseFieldTrefoilRadial" ||
           paramName == "phaseFieldTrefoilTangential" ||
           paramName == "phaseFieldSecondaryAstigRadial" ||
           paramName == "phaseFieldSecondaryAstigTangential" ||
           paramName == "phaseFieldQuadrafoilRadial" ||
           paramName == "phaseFieldQuadrafoilTangential" ||
           paramName == "phaseFieldSecondaryComaRadial" ||
           paramName == "phaseFieldSecondaryComaTangential" ||
           paramName == "spectralMode" ||
           paramName == "spectrumStyle" ||
           paramName == "spectrumForce" ||
           paramName == "spectrumSaturation" ||
           paramName == "chromaticAffectsLuma" ||
           paramName == "chromaticFocus" ||
           paramName == "chromaticSpread" ||
           paramName == "creativeFringe" ||
           paramName == "scatterAmount" ||
           paramName == "scatterRadius" ||
           paramName == "lookMode" ||
           paramName == "energyMode" ||
           paramName == "resolutionAware" ||
           paramName == "coreCompensation" ||
           paramName == "anisotropyEmphasis" ||
           paramName == "coreGain" ||
           paramName == "structureGain" ||
           paramName == "coreShoulder" ||
           paramName == "structureShoulder" ||
           paramName == "backendPreference" ||
           paramName == "debugView";
}

void LensDiffEffect::syncCustomApertureUi(bool forceSelectCustom) {
    if (!apertureMode_ || !customAperturePath_) {
        return;
    }

    int currentChoice = 0;
    apertureMode_->getValue(currentChoice);

    std::string customPath;
    customAperturePath_->getValue(customPath);
    const bool hasCustom = hasCustomAperturePath(customPath);

    if (forceSelectCustom && hasCustom) {
        apertureMode_->setValue(kCustomApertureChoiceIndex);
        updateCustomApertureControlVisibility();
        return;
    }
    if (!hasCustom && currentChoice == kCustomApertureChoiceIndex) {
        apertureMode_->setValue(kDefaultApertureChoiceIndex);
        updateCustomApertureControlVisibility();
        return;
    }
    if (currentChoice >= 0 && currentChoice <= kCustomApertureChoiceIndex) {
        apertureMode_->setValue(currentChoice);
    }
    updateCustomApertureControlVisibility();
}

void LensDiffEffect::syncPupilResolutionUi(bool fromChoice) {
    if (!pupilResolution_ || !pupilResolutionChoice_) {
        return;
    }

    if (fromChoice) {
        int choiceIndex = 2;
        pupilResolutionChoice_->getValue(choiceIndex);
        int currentValue = 256;
        pupilResolution_->getValue(currentValue);
        const int desiredValue = pupilResolutionFromChoice(choiceIndex);
        if (currentValue != desiredValue) {
            BoolScope scope(suppressPupilResolutionChangedHandling_);
            pupilResolution_->setValue(desiredValue);
        }
        return;
    }

    int storedValue = 256;
    pupilResolution_->getValue(storedValue);
    int currentChoice = 2;
    pupilResolutionChoice_->getValue(currentChoice);
    const int desiredChoice = choiceFromPupilResolution(storedValue);
    if (currentChoice != desiredChoice) {
        BoolScope scope(suppressPupilResolutionChangedHandling_);
        pupilResolutionChoice_->setValue(desiredChoice);
    }
}

void LensDiffEffect::syncPhaseMacrosFromCoefficients() {
    if (suppressPhaseSyncHandling_) return;
    BoolScope scope(suppressPhaseSyncHandling_);

    if (phaseFocus_ && phaseDefocus_) {
        phaseFocus_->setValue(phaseDefocus_->getValue());
    }
    if (phaseAstigmatism_ && phaseAstigAngleDeg_ && phaseAstigmatism0_ && phaseAstigmatism45_) {
        const double x = phaseAstigmatism0_->getValue();
        const double y = phaseAstigmatism45_->getValue();
        phaseAstigmatism_->setValue(phasePairMagnitude(x, y));
        phaseAstigAngleDeg_->setValue(phasePairAngleDeg(x, y, 2));
    }
    if (phaseComa_ && phaseComaAngleDeg_ && phaseComaX_ && phaseComaY_) {
        const double x = phaseComaX_->getValue();
        const double y = phaseComaY_->getValue();
        phaseComa_->setValue(phasePairMagnitude(x, y));
        phaseComaAngleDeg_->setValue(phasePairAngleDeg(x, y, 1));
    }
    if (phaseSphericalAmount_ && phaseSpherical_) {
        phaseSphericalAmount_->setValue(phaseSpherical_->getValue());
    }
    if (phaseTrefoil_ && phaseTrefoilAngleDeg_ && phaseTrefoilX_ && phaseTrefoilY_) {
        const double x = phaseTrefoilX_->getValue();
        const double y = phaseTrefoilY_->getValue();
        phaseTrefoil_->setValue(phasePairMagnitude(x, y));
        phaseTrefoilAngleDeg_->setValue(phasePairAngleDeg(x, y, 3));
    }
}

void LensDiffEffect::syncPhaseCoefficientsFromMacros() {
    if (suppressPhaseSyncHandling_) return;
    BoolScope scope(suppressPhaseSyncHandling_);

    if (phaseDefocus_ && phaseFocus_) {
        phaseDefocus_->setValue(phaseFocus_->getValue());
    }
    if (phaseAstigmatism0_ && phaseAstigmatism45_ && phaseAstigmatism_ && phaseAstigAngleDeg_) {
        double x = 0.0;
        double y = 0.0;
        resolveSimplePhasePair(phaseAstigmatism_->getValue(), phaseAstigAngleDeg_->getValue(), 2, &x, &y);
        phaseAstigmatism0_->setValue(x);
        phaseAstigmatism45_->setValue(y);
    }
    if (phaseComaX_ && phaseComaY_ && phaseComa_ && phaseComaAngleDeg_) {
        double x = 0.0;
        double y = 0.0;
        resolveSimplePhasePair(phaseComa_->getValue(), phaseComaAngleDeg_->getValue(), 1, &x, &y);
        phaseComaX_->setValue(x);
        phaseComaY_->setValue(y);
    }
    if (phaseSpherical_ && phaseSphericalAmount_) {
        phaseSpherical_->setValue(phaseSphericalAmount_->getValue());
    }
    if (phaseTrefoilX_ && phaseTrefoilY_ && phaseTrefoil_ && phaseTrefoilAngleDeg_) {
        double x = 0.0;
        double y = 0.0;
        resolveSimplePhasePair(phaseTrefoil_->getValue(), phaseTrefoilAngleDeg_->getValue(), 3, &x, &y);
        phaseTrefoilX_->setValue(x);
        phaseTrefoilY_->setValue(y);
    }
}

void LensDiffEffect::syncEffectGainMirror(bool fromSimpleEffectGain) {
    if (suppressEffectGainMirrorHandling_) return;
    if (!effectGain_ || !simpleEffectGain_) return;
    BoolScope scope(suppressEffectGainMirrorHandling_);
    if (fromSimpleEffectGain) {
        effectGain_->setValue(simpleEffectGain_->getValue());
    } else {
        simpleEffectGain_->setValue(effectGain_->getValue());
    }
}

bool LensDiffEffect::isSimpleMode() const {
    bool simpleMode = false;
    if (simpleModeState_) simpleModeState_->getValue(simpleMode);
    return simpleMode;
}

bool LensDiffEffect::isSimpleMode(double time) const {
    return simpleModeState_ ? simpleModeState_->getValueAtTime(time) : false;
}

void LensDiffEffect::updateSimpleModeToggleLabel() {
    if (!simpleModeToggle_) return;
    simpleModeToggle_->setLabel(isSimpleMode() ? "Switch to Full Mode" : "Switch to Simple Mode");
}

void LensDiffEffect::updateCustomApertureControlVisibility() {
    std::string customPath;
    if (customAperturePath_) {
        customAperturePath_->getValue(customPath);
    }
    int apertureChoice = 0;
    if (apertureMode_) {
        apertureMode_->getValue(apertureChoice);
    }
    const bool hasCustom = hasCustomAperturePath(customPath);
    const bool customActive = hasCustom && apertureChoice == kCustomApertureChoiceIndex;
    if (customApertureNormalize_) {
        customApertureNormalize_->setIsSecret(!customActive);
        customApertureNormalize_->setEnabled(customActive);
    }
    if (customApertureInvert_) {
        customApertureInvert_->setIsSecret(!customActive);
        customApertureInvert_->setEnabled(customActive);
    }
    if (customApertureStatus_) {
        customApertureStatus_->setIsSecret(!customActive);
        customApertureStatus_->setEnabled(false);
    }
}

void LensDiffEffect::updateOpticsControlVisibility() {
    std::string customPath;
    if (customAperturePath_) {
        customAperturePath_->getValue(customPath);
    }

    int apertureChoice = 0;
    if (apertureMode_) {
        apertureMode_->getValue(apertureChoice);
    }

    int vaneCount = 0;
    if (vaneCount_) {
        vaneCount_->getValue(vaneCount);
    }

    const LensDiffApertureMode apertureMode = apertureFromChoice(apertureChoice, hasCustomAperturePath(customPath));
    const bool bladeCountVisible = apertureMode == LensDiffApertureMode::Polygon ||
                                   apertureMode == LensDiffApertureMode::Star ||
                                   apertureMode == LensDiffApertureMode::Spiral ||
                                   apertureMode == LensDiffApertureMode::SquareGrid ||
                                   apertureMode == LensDiffApertureMode::Snowflake;
    const bool roundnessVisible = apertureMode == LensDiffApertureMode::Polygon ||
                                  apertureMode == LensDiffApertureMode::Star ||
                                  apertureMode == LensDiffApertureMode::Spiral ||
                                  apertureMode == LensDiffApertureMode::Hexagon ||
                                  apertureMode == LensDiffApertureMode::SquareGrid ||
                                  apertureMode == LensDiffApertureMode::Snowflake;
    const bool analyticAperture = apertureMode != LensDiffApertureMode::Custom;

    auto setParamState = [](OFX::Param* param, bool visible) {
        if (!param) return;
        param->setIsSecret(!visible);
        param->setEnabled(visible);
    };

    setParamState(bladeCount_, bladeCountVisible);
    setParamState(roundness_, roundnessVisible);
    setParamState(centralObstruction_, analyticAperture);
    setParamState(vaneCount_, analyticAperture);
    setParamState(vaneThickness_, analyticAperture && vaneCount > 0);
}

void LensDiffEffect::updateSpectrumControlVisibility() {
    int spectralMode = 0;
    if (spectralMode_) {
        spectralMode_->getValue(spectralMode);
    }
    const bool styledSpectrumVisible = spectralMode != 0;

    auto setParamState = [](OFX::Param* param, bool visible) {
        if (!param) return;
        param->setIsSecret(!visible);
        param->setEnabled(visible);
    };

    setParamState(spectrumStyle_, styledSpectrumVisible);
    setParamState(spectrumForce_, styledSpectrumVisible);
    setParamState(spectrumSaturation_, styledSpectrumVisible);
    setParamState(chromaticAffectsLuma_, styledSpectrumVisible);
}

void LensDiffEffect::updateCompositeControlVisibility() {
    int lookMode = 0;
    if (lookMode_) {
        lookMode_->getValue(lookMode);
    }
    int energyMode = 0;
    if (energyMode_) {
        energyMode_->getValue(energyMode);
    }

    const bool splitVisible = lookMode == 1;
    const bool augmentVisible = energyMode == 1;

    auto setParamState = [](OFX::Param* param, bool visible) {
        if (!param) return;
        param->setIsSecret(!visible);
        param->setEnabled(visible);
    };

    setParamState(compositeSplitGroup_, splitVisible);
    setParamState(coreGain_, splitVisible);
    setParamState(structureGain_, splitVisible);
    setParamState(coreShoulder_, splitVisible);
    setParamState(structureShoulder_, splitVisible);
    setParamState(coreCompensation_, augmentVisible);
}

void LensDiffEffect::updatePhaseControlVisibility() {
    bool phaseEnabled = false;
    if (phaseEnabled_) phaseEnabled_->getValue(phaseEnabled);

    int spectralMode = 0;
    if (spectralMode_) {
        spectralMode_->getValue(spectralMode);
    }

    auto isEffectivelyNonZero = [](OFX::DoubleParam* param) {
        return param && std::abs(param->getValue()) > 1e-6;
    };

    const bool fieldDetailsVisible = phaseEnabled && isEffectivelyNonZero(phaseFieldStrength_);
    const bool chromaticVisible = phaseEnabled && spectralMode != 0;
    const bool scatterRadiusVisible = phaseEnabled && isEffectivelyNonZero(scatterAmount_);

    auto setPhaseParamState = [](OFX::Param* param, bool visible) {
        if (!param) return;
        param->setIsSecret(!visible);
        param->setEnabled(visible);
    };

    setPhaseParamState(phaseGroup_, phaseEnabled);
    setPhaseParamState(phasePrimaryGroup_, phaseEnabled);
    setPhaseParamState(phaseAdvancedGroup_, phaseEnabled);
    setPhaseParamState(phaseFieldGroup_, phaseEnabled);
    setPhaseParamState(phaseFieldPrimaryGroup_, fieldDetailsVisible);
    setPhaseParamState(phaseFieldHigherOrderGroup_, fieldDetailsVisible);
    setPhaseParamState(phaseChromaticGroup_, chromaticVisible);
    setPhaseParamState(phaseFinishingGroup_, phaseEnabled);

    // Keep the legacy macro wrappers hidden; they still sync for preset compatibility.
    setPhaseParamState(phaseFocus_, false);
    setPhaseParamState(phaseAstigmatism_, false);
    setPhaseParamState(phaseAstigAngleDeg_, false);
    setPhaseParamState(phaseComa_, false);
    setPhaseParamState(phaseComaAngleDeg_, false);
    setPhaseParamState(phaseSphericalAmount_, false);
    setPhaseParamState(phaseTrefoil_, false);
    setPhaseParamState(phaseTrefoilAngleDeg_, false);

    setPhaseParamState(phaseDefocus_, phaseEnabled);
    setPhaseParamState(phaseAstigmatism0_, phaseEnabled);
    setPhaseParamState(phaseAstigmatism45_, phaseEnabled);
    setPhaseParamState(phaseComaX_, phaseEnabled);
    setPhaseParamState(phaseComaY_, phaseEnabled);
    setPhaseParamState(phaseSpherical_, phaseEnabled);

    setPhaseParamState(phaseTrefoilX_, phaseEnabled);
    setPhaseParamState(phaseTrefoilY_, phaseEnabled);
    setPhaseParamState(phaseSecondaryAstigmatism0_, phaseEnabled);
    setPhaseParamState(phaseSecondaryAstigmatism45_, phaseEnabled);
    setPhaseParamState(phaseQuadrafoil0_, phaseEnabled);
    setPhaseParamState(phaseQuadrafoil45_, phaseEnabled);
    setPhaseParamState(phaseSecondaryComaX_, phaseEnabled);
    setPhaseParamState(phaseSecondaryComaY_, phaseEnabled);
    setPhaseParamState(pupilDecenterX_, phaseEnabled);
    setPhaseParamState(pupilDecenterY_, phaseEnabled);

    setPhaseParamState(phaseFieldStrength_, phaseEnabled);
    setPhaseParamState(phaseFieldEdgeBias_, fieldDetailsVisible);
    setPhaseParamState(phaseFieldDefocus_, fieldDetailsVisible);
    setPhaseParamState(phaseFieldAstigRadial_, fieldDetailsVisible);
    setPhaseParamState(phaseFieldAstigTangential_, fieldDetailsVisible);
    setPhaseParamState(phaseFieldComaRadial_, fieldDetailsVisible);
    setPhaseParamState(phaseFieldComaTangential_, fieldDetailsVisible);
    setPhaseParamState(phaseFieldSpherical_, fieldDetailsVisible);
    setPhaseParamState(phaseFieldTrefoilRadial_, fieldDetailsVisible);
    setPhaseParamState(phaseFieldTrefoilTangential_, fieldDetailsVisible);
    setPhaseParamState(phaseFieldSecondaryAstigRadial_, fieldDetailsVisible);
    setPhaseParamState(phaseFieldSecondaryAstigTangential_, fieldDetailsVisible);
    setPhaseParamState(phaseFieldQuadrafoilRadial_, fieldDetailsVisible);
    setPhaseParamState(phaseFieldQuadrafoilTangential_, fieldDetailsVisible);
    setPhaseParamState(phaseFieldSecondaryComaRadial_, fieldDetailsVisible);
    setPhaseParamState(phaseFieldSecondaryComaTangential_, fieldDetailsVisible);

    setPhaseParamState(chromaticFocus_, chromaticVisible);
    setPhaseParamState(chromaticSpread_, chromaticVisible);
    setPhaseParamState(creativeFringe_, phaseEnabled);
    setPhaseParamState(scatterAmount_, phaseEnabled);
    setPhaseParamState(scatterRadius_, scatterRadiusVisible);
}

void LensDiffEffect::refreshDynamicControlVisibility() {
    const bool simpleMode = isSimpleMode();
    updateSimpleModeToggleLabel();
    updateCustomApertureControlVisibility();
    updateOpticsControlVisibility();
    updateSpectrumControlVisibility();
    updateCompositeControlVisibility();
    updatePhaseControlVisibility();

    auto setParamState = [](OFX::Param* param, bool visible) {
        if (!param) return;
        param->setIsSecret(!visible);
        param->setEnabled(visible);
    };

        setParamState(simpleEffectGain_, simpleMode);
        setParamState(effectGain_, !simpleMode);
        setParamState(compositeGroup_, !simpleMode);
        setParamState(resolutionAware_, !simpleMode);

        setParamState(softnessStops_, !simpleMode);
    setParamState(extractionMode_, !simpleMode);
    setParamState(pointEmphasis_, !simpleMode);
    setParamState(apodizationMode_, !simpleMode);
    setParamState(phaseEnabled_, !simpleMode);

    if (!simpleMode) return;

    setParamState(roundness_, false);
    setParamState(phaseGroup_, false);

    setParamState(chromaticAffectsLuma_, false);
}

void LensDiffEffect::openPhaseGroup() {
    OfxParamSetHandle paramSet = nullptr;
    if (OFX::Private::gEffectSuite->getParamSet(getHandle(), &paramSet) != kOfxStatOK || !paramSet) return;

    OfxParamHandle paramHandle = nullptr;
    OfxPropertySetHandle propHandle = nullptr;
    const OfxStatus handleStatus =
        OFX::Private::gParamSuite->paramGetHandle(paramSet, "PhaseGroup", &paramHandle, &propHandle);
    if (handleStatus != kOfxStatOK || !propHandle) return;

    OFX::PropertySet props(propHandle);
    props.propSetInt(kOfxParamPropGroupOpen, 1, false);
}

void LensDiffEffect::updateCustomApertureStatus() {
    if (!customApertureStatus_) {
        return;
    }

    std::string customPath;
    if (customAperturePath_) {
        customAperturePath_->getValue(customPath);
    }

    std::string status = "No custom aperture loaded.";
    if (hasCustomAperturePath(customPath)) {
        LensDiffApertureImage image;
        std::string error;
        if (LoadLensDiffApertureImage(customPath, &image, &error) && image.width > 0 && image.height > 0) {
            std::ostringstream ss;
            ss << "Loaded " << customApertureDisplayName(customPath)
               << " (" << image.width << "x" << image.height << ")";
            status = ss.str();
        } else if (!error.empty()) {
            status = "Custom aperture load failed: " + error;
        } else {
            status = "Custom aperture load failed.";
        }
    }

    customApertureStatus_->setValue(status);
}

bool LensDiffEffect::getRegionOfDefinition(const OFX::RegionOfDefinitionArguments& args, OfxRectD& rod) {
    rod = srcClip_->getRegionOfDefinition(args.time);
    return true;
}

void LensDiffEffect::getRegionsOfInterest(const OFX::RegionsOfInterestArguments& args, OFX::RegionOfInterestSetter& rois) {
    const int frameShortSidePx = lensdiffShortSidePx(dstClip_->getRegionOfDefinition(args.time), args.renderScale);
    const LensDiffParams params = resolveParams(args.time, frameShortSidePx);
    const int kernelExpand = ResolveLensDiffMaxKernelRadiusPx(params);
    const int scatterExpand = std::max(0, static_cast<int>(std::ceil(ResolveLensDiffScatterRadiusPx(params))));
    const int expand = std::max(kernelExpand, scatterExpand);
    OfxRectD roi = args.regionOfInterest;
    roi.x1 -= expand;
    roi.y1 -= expand;
    roi.x2 += expand;
    roi.y2 += expand;
    rois.setRegionOfInterest(*srcClip_, roi);
}

bool LensDiffEffect::isIdentity(const OFX::IsIdentityArguments& args, OFX::Clip*& identityClip, double& identityTime) {
    const LensDiffParams params = resolveParams(args.time);
    if (params.debugView != LensDiffDebugView::Final) {
        return false;
    }
    if (params.effectGain <= 0.0) {
        identityClip = srcClip_;
        identityTime = args.time;
        return true;
    }
    return false;
}

void LensDiffEffect::syncPrivateData() {
    refreshLensDiffPresetMenuMetadataFromStore();
    lensdiffPresetDisplayStateValid_ = false;
    syncPhaseMacrosFromCoefficients();
    syncEffectGainMirror(false);
    refreshDynamicControlVisibility();
    updateCustomApertureStatus();
    syncPupilResolutionUi(false);
    ImageEffect::syncPrivateData();
}

void LensDiffEffect::changedParam(const OFX::InstanceChangedArgs& args, const std::string& paramName) {
    if (suppressLensDiffPresetChangedHandling_) return;
    if (suppressPupilResolutionChangedHandling_ &&
        (paramName == "pupilResolution" || paramName == "pupilResolutionChoice")) {
        return;
    }
    if (suppressEffectGainMirrorHandling_ &&
        (paramName == "effectGain" || paramName == "simpleEffectGain")) {
        return;
    }
    if (suppressPhaseSyncHandling_ &&
        (paramName == "phaseFocus" ||
         paramName == "phaseAstigmatism" ||
         paramName == "phaseAstigAngleDeg" ||
         paramName == "phaseComa" ||
         paramName == "phaseComaAngleDeg" ||
         paramName == "phaseSphericalAmount" ||
         paramName == "phaseTrefoil" ||
         paramName == "phaseTrefoilAngleDeg" ||
         paramName == "phaseDefocus" ||
         paramName == "phaseAstigmatism0" ||
         paramName == "phaseAstigmatism45" ||
         paramName == "phaseComaX" ||
         paramName == "phaseComaY" ||
         paramName == "phaseSpherical" ||
         paramName == "phaseTrefoilX" ||
         paramName == "phaseTrefoilY")) {
        return;
    }
    if (paramName == "lensdiffPresetMenu") {
        applySelectedLensDiffPreset(args.time);
        return;
    }
    if (paramName == "lensdiffWebsite") {
        if (args.reason == OFX::eChangeUserEdit) openUrl(kWebsiteUrl);
        return;
    }
    if (paramName == "simpleModeToggle") {
        if (args.reason == OFX::eChangeUserEdit && simpleModeState_) {
            const bool nextSimpleMode = !isSimpleMode();
            BoolScope scope(suppressLensDiffPresetChangedHandling_);
            simpleModeState_->setValue(nextSimpleMode);
        }
        refreshDynamicControlVisibility();
        syncLensDiffPresetMenuState(args.time);
        return;
    }
    if (paramName == "simpleModeState") {
        refreshDynamicControlVisibility();
        syncLensDiffPresetMenuState(args.time);
        return;
    }
    if (paramName == "simpleEffectGain") {
        syncEffectGainMirror(true);
        refreshDynamicControlVisibility();
        return;
    }
    if (paramName == "effectGain") {
        syncEffectGainMirror(false);
        refreshDynamicControlVisibility();
        return;
    }
    if (paramName == "opticsReset" ||
        paramName == "phaseReset" ||
        paramName == "spectrumReset" ||
        paramName == "compositeReset") {
        LensDiffPresetValues values;
        if (!resolveLensDiffPresetSelectionValues(selectedLensDiffPresetFromMenu(args.time), &values)) return;

        BoolScope scope(suppressLensDiffPresetChangedHandling_);
        if (paramName == "opticsReset") {
            writeOpticsPresetValuesToParams(values);
        } else if (paramName == "phaseReset") {
            writePhasePresetValuesToParams(values);
            if (values.phaseEnabled) openPhaseGroup();
        } else if (paramName == "spectrumReset") {
            writeSpectrumPresetValuesToParams(values);
        } else {
            writeCompositePresetValuesToParams(values);
        }

        refreshDynamicControlVisibility();
        updateCustomApertureStatus();
        syncPupilResolutionUi(false);
        syncLensDiffPresetMenuState(args.time);
        return;
    }
    if (paramName == "lensdiffPresetSave") {
        const std::string name = sanitizePresetName(getStringValueAtTime(fetchStringParam("lensdiffPresetName"), args.time), "Preset");
        if (lensdiffPresetNameReserved(name)) {
            showLensDiffPresetInfoDialog(lensdiffReservedPresetNameMessage(name));
            return;
        }

        LensDiffPresetSelection preferred {};
        preferred.kind = LensDiffPresetSelection::Kind::User;
        {
            std::lock_guard<std::mutex> lock(lensdiffPresetMutex());
            ensureLensDiffPresetStoreLoadedLocked();
            LensDiffPresetValues values = captureCurrentLensDiffPresetValues(args.time);
            values.simpleMode = lensdiffFactoryPresetValues().simpleMode;
            const std::string now = nowUtcIso8601();
            const int existingIndex = lensdiffUserPresetIndexByNameLocked(name);
            if (existingIndex >= 0) {
                if (!confirmLensDiffPresetOverwriteDialog(name)) return;
                auto& preset = lensdiffPresetStore().userPresets[static_cast<std::size_t>(existingIndex)];
                preset.values = values;
                preset.updatedAtUtc = now;
                preferred.userIndex = existingIndex;
            } else {
                LensDiffUserPreset preset {};
                preset.id = makePresetId("lensdiff");
                preset.name = name;
                preset.createdAtUtc = now;
                preset.updatedAtUtc = now;
                preset.values = values;
                lensdiffPresetStore().userPresets.push_back(preset);
                preferred.userIndex = static_cast<int>(lensdiffPresetStore().userPresets.size()) - 1;
            }
            saveLensDiffPresetStoreLocked();
        }
        if (auto* p = fetchStringParam("lensdiffPresetName")) p->setValue(name);
        syncLensDiffPresetMenuFromDisk(args.time, preferred);
        return;
    }
    if (paramName == "lensdiffPresetSaveDefaults") {
        {
            std::lock_guard<std::mutex> lock(lensdiffPresetMutex());
            ensureLensDiffPresetStoreLoadedLocked();
            auto& preset = lensdiffPresetStore().defaultPreset;
            preset.id = "default";
            preset.name = kLensDiffPresetDefaultName;
            if (preset.createdAtUtc.empty()) preset.createdAtUtc = nowUtcIso8601();
            preset.updatedAtUtc = nowUtcIso8601();
            preset.values = captureCurrentLensDiffPresetValues(args.time);
            saveLensDiffPresetStoreLocked();
        }
        LensDiffPresetSelection preferred {};
        preferred.kind = LensDiffPresetSelection::Kind::Default;
        syncLensDiffPresetMenuFromDisk(args.time, preferred);
        showLensDiffPresetInfoDialog("LensDiff defaults saved.\n\nThese new defaults will be used from the next plugin or host restart.");
        return;
    }
    if (paramName == "lensdiffPresetUpdate") {
        const LensDiffPresetSelection selection = selectedLensDiffPresetFromMenu(args.time);
        if (selection.kind != LensDiffPresetSelection::Kind::User || selection.userIndex < 0) return;
        {
            std::lock_guard<std::mutex> lock(lensdiffPresetMutex());
            ensureLensDiffPresetStoreLoadedLocked();
            if (selection.userIndex >= static_cast<int>(lensdiffPresetStore().userPresets.size())) return;
            auto& preset = lensdiffPresetStore().userPresets[static_cast<std::size_t>(selection.userIndex)];
            preset.values = captureCurrentLensDiffPresetValues(args.time);
            preset.values.simpleMode = lensdiffFactoryPresetValues().simpleMode;
            preset.updatedAtUtc = nowUtcIso8601();
            saveLensDiffPresetStoreLocked();
        }
        syncLensDiffPresetMenuFromDisk(args.time, selection);
        return;
    }
    if (paramName == "lensdiffPresetRename") {
        const LensDiffPresetSelection selection = selectedLensDiffPresetFromMenu(args.time);
        if (selection.kind != LensDiffPresetSelection::Kind::User || selection.userIndex < 0) return;
        const std::string newName =
            sanitizePresetName(getStringValueAtTime(fetchStringParam("lensdiffPresetName"), args.time), "Preset");
        if (lensdiffPresetNameReserved(newName)) {
            showLensDiffPresetInfoDialog(lensdiffReservedPresetNameMessage(newName));
            return;
        }
        {
            std::lock_guard<std::mutex> lock(lensdiffPresetMutex());
            ensureLensDiffPresetStoreLoadedLocked();
            if (selection.userIndex >= static_cast<int>(lensdiffPresetStore().userPresets.size())) return;
            auto& preset = lensdiffPresetStore().userPresets[static_cast<std::size_t>(selection.userIndex)];
            if (lensdiffPresetUserNameExistsLocked(newName, &preset.id)) {
                showLensDiffPresetInfoDialog("A LensDiff preset with that name already exists.");
                return;
            }
            preset.name = newName;
            preset.updatedAtUtc = nowUtcIso8601();
            saveLensDiffPresetStoreLocked();
        }
        if (auto* p = fetchStringParam("lensdiffPresetName")) p->setValue(newName);
        syncLensDiffPresetMenuFromDisk(args.time, selection);
        return;
    }
    if (paramName == "lensdiffPresetDelete") {
        const LensDiffPresetSelection selection = selectedLensDiffPresetFromMenu(args.time);
        if (selection.kind != LensDiffPresetSelection::Kind::User || selection.userIndex < 0) return;
        std::string presetName;
        {
            std::lock_guard<std::mutex> lock(lensdiffPresetMutex());
            ensureLensDiffPresetStoreLoadedLocked();
            if (selection.userIndex >= static_cast<int>(lensdiffPresetStore().userPresets.size())) return;
            presetName = lensdiffPresetStore().userPresets[static_cast<std::size_t>(selection.userIndex)].name;
        }
        if (!confirmLensDiffPresetDeleteDialog(presetName)) return;
        {
            std::lock_guard<std::mutex> lock(lensdiffPresetMutex());
            ensureLensDiffPresetStoreLoadedLocked();
            if (selection.userIndex < 0 || selection.userIndex >= static_cast<int>(lensdiffPresetStore().userPresets.size())) return;
            lensdiffPresetStore().userPresets.erase(lensdiffPresetStore().userPresets.begin() + selection.userIndex);
            saveLensDiffPresetStoreLocked();
        }
        syncLensDiffPresetMenuFromDisk(args.time);
        return;
    }
    if (paramName == "pupilResolution") {
        syncPupilResolutionUi(false);
        syncLensDiffPresetMenuState(args.time);
        return;
    }
    if (paramName == "pupilResolutionChoice") {
        syncPupilResolutionUi(true);
        syncLensDiffPresetMenuState(args.time);
        return;
    }
    if (paramName == "customApertureImport") {
        if (args.reason == OFX::eChangeUserEdit && customAperturePath_) {
            std::string selectedPath;
            std::string error;
            if (OpenLensDiffApertureFileDialog(&selectedPath, &error) && hasCustomAperturePath(selectedPath)) {
                customAperturePath_->setValue(selectedPath);
                syncCustomApertureUi(true);
                updateCustomApertureStatus();
            }
        }
        refreshDynamicControlVisibility();
        syncLensDiffPresetMenuState(args.time);
        return;
    }
    if (paramName == "customAperturePath") {
        if (args.reason == OFX::eChangeUserEdit) {
            syncCustomApertureUi(true);
        }
        refreshDynamicControlVisibility();
        updateCustomApertureStatus();
        syncLensDiffPresetMenuState(args.time);
        return;
    }
    if (paramName == "apertureMode") {
        refreshDynamicControlVisibility();
        syncLensDiffPresetMenuState(args.time);
        return;
    }
    if (paramName == "phaseEnabled") {
        if (args.reason == OFX::eChangeUserEdit && phaseEnabled_ && phaseEnabled_->getValueAtTime(args.time)) {
            openPhaseGroup();
        }
        refreshDynamicControlVisibility();
        syncLensDiffPresetMenuState(args.time);
        return;
    }
    if (paramName == "phaseFocus" ||
        paramName == "phaseAstigmatism" ||
        paramName == "phaseAstigAngleDeg" ||
        paramName == "phaseComa" ||
        paramName == "phaseComaAngleDeg" ||
        paramName == "phaseSphericalAmount" ||
        paramName == "phaseTrefoil" ||
        paramName == "phaseTrefoilAngleDeg") {
        syncPhaseCoefficientsFromMacros();
        refreshDynamicControlVisibility();
        syncLensDiffPresetMenuState(args.time);
        return;
    }
    if (paramName == "phaseDefocus" ||
        paramName == "phaseAstigmatism0" ||
        paramName == "phaseAstigmatism45" ||
        paramName == "phaseComaX" ||
        paramName == "phaseComaY" ||
        paramName == "phaseSpherical" ||
        paramName == "phaseTrefoilX" ||
        paramName == "phaseTrefoilY") {
        syncPhaseMacrosFromCoefficients();
        refreshDynamicControlVisibility();
        syncLensDiffPresetMenuState(args.time);
        return;
    }
    if (paramName == "customApertureNormalize" || paramName == "customApertureInvert") {
        refreshDynamicControlVisibility();
        syncLensDiffPresetMenuState(args.time);
        return;
    }
    if (isLensDiffPresetManagedParam(paramName)) {
        refreshDynamicControlVisibility();
        syncLensDiffPresetMenuState(args.time);
    }
}

void LensDiffEffect::render(const OFX::RenderArguments& args) {
    std::unique_ptr<OFX::Image> dst(dstClip_->fetchImage(args.time));
    std::unique_ptr<OFX::Image> src(srcClip_->fetchImage(args.time));
    if (!dst || !src) {
        OFX::throwSuiteStatusException(kOfxStatFailed);
    }
    if (dst->getPixelDepth() != OFX::eBitDepthFloat || dst->getPixelComponents() != OFX::ePixelComponentRGBA) {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
    if (src->getPixelDepth() != OFX::eBitDepthFloat || src->getPixelComponents() != OFX::ePixelComponentRGBA) {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }

    LensDiffRenderRequest request {};
    request.renderWindow = {args.renderWindow.x1, args.renderWindow.y1, args.renderWindow.x2, args.renderWindow.y2};
    request.frameBounds = lensdiffScaledImageRect(dstClip_->getRegionOfDefinition(args.time), args.renderScale);
    request.src = makeImageView(src.get());
    request.dst = makeImageView(dst.get());
    request.hostEnabledCudaRender = args.isEnabledCudaRender;
    request.hostEnabledMetalRender = args.isEnabledMetalRender;
    request.cudaStream = args.pCudaStream;
    request.metalCommandQueue = args.pMetalCmdQ;

    request.frameShortSidePx = lensdiffShortSidePx(dstClip_->getRegionOfDefinition(args.time), args.renderScale);
    const LensDiffParams params = resolveParams(args.time, request.frameShortSidePx);
    request.requestedBackend = resolveRenderBackend(args, getChoiceValueAtTime(backendPreference_, args.time));
    request.estimatedSupportRadiusPx = ResolveLensDiffMaxKernelRadiusPx(params);
    const bool logEnabled = envFlagEnabled("LENSDIFF_LOG");
    const bool timingEnabled = envFlagEnabled("LENSDIFF_TIMING");
    const auto renderStart = std::chrono::steady_clock::now();

    bool rendered = false;
    std::string error;
    LensDiffBackendType attemptedBackend = request.requestedBackend;
    LensDiffBackendType executedBackend = LensDiffBackendType::Auto;
    std::string note;
    const bool cpuOnlyPhaseSuite = lensdiffNeedsCpuOnlyPhaseSuite(params);
    std::lock_guard<std::mutex> lock(cacheMutex_);
    const bool cpuFallbackAllowed =
        cpuOnlyPhaseSuite || (!request.hostEnabledCudaRender && !request.hostEnabledMetalRender);
    if (cpuOnlyPhaseSuite && request.requestedBackend != LensDiffBackendType::CpuReference && cpuFallbackAllowed) {
        attemptedBackend = request.requestedBackend;
        request.requestedBackend = LensDiffBackendType::CpuReference;
        note = "cpu-phase-suite";
    }

#if defined(LENSDIFF_HAS_CUDA)
    if (request.requestedBackend == LensDiffBackendType::Cuda) {
        request.selectedBackend = LensDiffBackendType::Cuda;
        rendered = RunLensDiffCuda(request, params, cache_, &error);
        if (rendered) {
            executedBackend = LensDiffBackendType::Cuda;
        }
    }
#endif
#ifdef __APPLE__
    if (!rendered && request.requestedBackend == LensDiffBackendType::Metal) {
        request.selectedBackend = LensDiffBackendType::Metal;
        if (logEnabled) {
            std::ostringstream metalDispatchNote;
            metalDispatchNote << "renderWindow=(" << request.renderWindow.x1 << "," << request.renderWindow.y1 << ")-("
                              << request.renderWindow.x2 << "," << request.renderWindow.y2 << ")"
                              << " frameBounds=(" << request.frameBounds.x1 << "," << request.frameBounds.y1 << ")-("
                              << request.frameBounds.x2 << "," << request.frameBounds.y2 << ")"
                              << " srcBounds=(" << request.src.bounds.x1 << "," << request.src.bounds.y1 << ")-("
                              << request.src.bounds.x2 << "," << request.src.bounds.y2 << ")"
                              << " dstBounds=(" << request.dst.bounds.x1 << "," << request.dst.bounds.y1 << ")-("
                              << request.dst.bounds.x2 << "," << request.dst.bounds.y2 << ")"
                              << " resolutionAware=" << (params.resolutionAware ? "true" : "false")
                              << " spectral=" << spectralModeName(params.spectralMode)
                              << " debug=" << debugViewName(params.debugView);
            LogLensDiffDiagnosticEvent("host-metal-dispatch", metalDispatchNote.str());
        }
        if (logEnabled) {
            LogLensDiffDiagnosticEvent("host-metal-call-enter", "about-to-call RunLensDiffMetal");
        }
        rendered = RunLensDiffMetal(request, params, cache_, &error);
        if (logEnabled) {
            std::ostringstream metalReturnNote;
            metalReturnNote << "rendered=" << (rendered ? "true" : "false");
            if (!error.empty()) {
                metalReturnNote << " error=" << error;
            }
            LogLensDiffDiagnosticEvent("host-metal-call-return", metalReturnNote.str());
            LogLensDiffDiagnosticEvent("host-metal-return", metalReturnNote.str());
        }
        if (rendered) {
            executedBackend = LensDiffBackendType::Metal;
        }
    }
#endif
    if (!rendered && cpuFallbackAllowed) {
        request.selectedBackend = LensDiffBackendType::CpuReference;
        rendered = RunLensDiffCpuReference(request, params, cache_, &error);
        if (rendered) {
            executedBackend = LensDiffBackendType::CpuReference;
            if (note.empty() && attemptedBackend != LensDiffBackendType::CpuReference) {
                note = "fallback-from-";
                note += backendName(attemptedBackend);
            }
        }
    }

    if (!rendered) {
        if (logEnabled) {
            const auto renderEnd = std::chrono::steady_clock::now();
            const double elapsedMs =
                std::chrono::duration<double, std::milli>(renderEnd - renderStart).count();
            std::string failureNote = error.empty() ? "render-failed" : ("render-failed:" + error);
            logLensDiffRender(timingEnabled, attemptedBackend, request, params, failureNote, elapsedMs);
        }
        OFX::throwSuiteStatusException(kOfxStatFailed);
    }

    if (logEnabled) {
        const auto renderEnd = std::chrono::steady_clock::now();
        const double elapsedMs =
            std::chrono::duration<double, std::milli>(renderEnd - renderStart).count();
        logLensDiffRender(timingEnabled, executedBackend, request, params, note, elapsedMs);
    }
}

class LensDiffFactory : public OFX::PluginFactoryHelper<LensDiffFactory> {
public:
    LensDiffFactory() : OFX::PluginFactoryHelper<LensDiffFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor) {}

    void describe(OFX::ImageEffectDescriptor& desc) override {
        static const std::string nameWithVersion = std::string(kPluginName) + " " + kPluginVersionLabel;
        desc.setLabels(nameWithVersion.c_str(), nameWithVersion.c_str(), nameWithVersion.c_str());
        desc.setPluginGrouping(kPluginGrouping);
        desc.setPluginDescription(kPluginDescription);
        desc.getPropertySet().propSetString(kOfxPropIcon, "", 0, false);
        desc.getPropertySet().propSetString(kOfxPropIcon, "com.moazelgabry.LensDiff.png", 1, false);
        desc.addSupportedContext(OFX::eContextFilter);
        desc.addSupportedBitDepth(OFX::eBitDepthFloat);
        desc.setSingleInstance(false);
        desc.setHostFrameThreading(false);
        desc.setSupportsMultiResolution(kSupportsMultiResolution);
        desc.setSupportsTiles(kSupportsTiles);
        desc.setTemporalClipAccess(false);
        desc.setRenderTwiceAlways(false);
        desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);
#if defined(LENSDIFF_HAS_CUDA) && !defined(__APPLE__)
        desc.setSupportsCudaRender(true);
#endif
#ifdef __APPLE__
        desc.setSupportsMetalRender(true);
#endif
    }

    void describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum) override {
        const LensDiffPresetValues lensdiffDefaultValues = describeLensDiffDefaultValues();
        const LensDiffApertureMode defaultApertureMode = sanitizePresetApertureMode(lensdiffDefaultValues.apertureMode);
        const int defaultApertureChoiceIndex =
            apertureChoiceIndexForPresetMode(defaultApertureMode, hasCustomAperturePath(lensdiffDefaultValues.customAperturePath));

        auto* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
        src->addSupportedComponent(OFX::ePixelComponentRGBA);
        src->setSupportsTiles(kSupportsTiles);
        src->setTemporalClipAccess(false);

        auto* dst = desc.defineClip(kOfxImageEffectOutputClipName);
        dst->addSupportedComponent(OFX::ePixelComponentRGBA);
        dst->setSupportsTiles(kSupportsTiles);

        auto* page = desc.definePageParam("Controls");

        auto* simpleModeState = desc.defineBooleanParam("simpleModeState");
        simpleModeState->setLabels("Simple Mode State", "Simple Mode State", "Simple Mode State");
        simpleModeState->setDefault(lensdiffDefaultValues.simpleMode);
        simpleModeState->setIsSecret(true);

        auto* simpleModeToggle = desc.definePushButtonParam("simpleModeToggle");
        simpleModeToggle->setLabel(lensdiffDefaultValues.simpleMode ? "Switch to Full Mode" : "Switch to Simple Mode");
        simpleModeToggle->setHint("Toggle between the compact Simple UI and the full LensDiff control set.");
        page->addChild(*simpleModeToggle);

        auto* inputTransfer = desc.defineChoiceParam("inputTransfer");
        inputTransfer->setLabel("Input Transfer");
        inputTransfer->setHint("Transfer function assumed at the plugin input before internal linearization.");
        inputTransfer->appendOption("Linear");
        inputTransfer->appendOption("DaVinci Intermediate");
        inputTransfer->setDefault(lensdiffDefaultValues.inputTransfer);
        page->addChild(*inputTransfer);

        auto* lensdiffPresetMenu = desc.defineChoiceParam("lensdiffPresetMenu");
        lensdiffPresetMenu->setLabel("LensDiff Preset");
        lensdiffPresetMenu->setHint("Quickly load the default, built-in, or saved LensDiff presets.");
        lensdiffPresetMenu->appendOption(kLensDiffPresetDefaultName);
        for (const auto& builtin : lensdiffBuiltinPresets()) {
            lensdiffPresetMenu->appendOption(builtin.name);
        }
        {
            std::lock_guard<std::mutex> lock(lensdiffPresetMutex());
            ensureLensDiffPresetStoreLoadedLocked();
            for (const auto& preset : lensdiffPresetStore().userPresets) {
                lensdiffPresetMenu->appendOption(lensdiffMenuLabelForUserPresetName(preset.name));
            }
        }
        lensdiffPresetMenu->setDefault(0);
        page->addChild(*lensdiffPresetMenu);
        auto* simpleEffectGain = desc.defineDoubleParam("simpleEffectGain");
        simpleEffectGain->setLabels("Effect Gain", "Effect Gain", "Effect Gain");
        simpleEffectGain->setHint("Top-level Simple mode mirror of the Composite effect gain.");
        simpleEffectGain->setDefault(lensdiffDefaultValues.effectGain);
        simpleEffectGain->setRange(0.0, 4.0);
        simpleEffectGain->setDisplayRange(0.0, 4.0);
        simpleEffectGain->setIncrement(0.01);
        simpleEffectGain->setIsSecret(!lensdiffDefaultValues.simpleMode);
        page->addChild(*simpleEffectGain);

        auto* selection = desc.defineGroupParam("SelectionGroup");
        selection->setLabels("Selection", "Selection", "Selection");
        selection->setOpen(true);
        page->addChild(*selection);

        auto* optics = desc.defineGroupParam("OpticsGroup");
        optics->setLabels("Optics", "Optics", "Optics");
        optics->setOpen(true);
        page->addChild(*optics);
        page->addChild(*definePushButtonParam(desc, "opticsReset", "Reset Optics", "Reset the optics controls to the currently selected LensDiff preset.", optics));

        auto* phaseEnabled = desc.defineBooleanParam("phaseEnabled");
        phaseEnabled->setLabels("Enable Phase", "Enable Phase", "Enable Phase");
        phaseEnabled->setHint("Enable the pupil phase controls.");
        phaseEnabled->setDefault(lensdiffDefaultValues.phaseEnabled);
        page->addChild(*phaseEnabled);

        auto* phase = desc.defineGroupParam("PhaseGroup");
        phase->setLabels("Phase", "Phase", "Phase");
        phase->setOpen(false);
        page->addChild(*phase);
        page->addChild(*definePushButtonParam(desc, "phaseReset", "Reset Phase", "Reset the phase controls to the currently selected LensDiff preset.", phase));

        auto* spectrum = desc.defineGroupParam("SpectrumGroup");
        spectrum->setLabels("Spectrum", "Spectrum", "Spectrum");
        spectrum->setOpen(false);
        page->addChild(*spectrum);
        page->addChild(*definePushButtonParam(desc, "spectrumReset", "Reset Spectrum", "Reset the spectrum controls to the currently selected LensDiff preset.", spectrum));

        auto* composite = desc.defineGroupParam("CompositeGroup");
        composite->setLabels("Composite", "Composite", "Composite");
        composite->setOpen(true);
        page->addChild(*composite);
        page->addChild(*definePushButtonParam(desc, "compositeReset", "Reset Composite", "Reset the composite controls to the currently selected LensDiff preset.", composite));
        auto* resolutionAware = desc.defineBooleanParam("resolutionAware");
        resolutionAware->setLabels("Resolution-aware", "Resolution-aware", "Resolution-aware");
        resolutionAware->setHint("Run the generated diffraction effect in a UHD-authored internal optics space for more consistent looks across output resolutions while keeping the base image composite on the native plate.");
        resolutionAware->setDefault(lensdiffDefaultValues.resolutionAware);
        page->addChild(*resolutionAware);

        auto* debugView = desc.defineChoiceParam("debugView");
        debugView->setLabel("Rendered View");
        debugView->setHint("Output diagnostic view.");
        debugView->appendOption("Final");
        debugView->appendOption("Selection");
        debugView->appendOption("Pupil");
        debugView->appendOption("PSF");
        debugView->appendOption("OTF");
        debugView->appendOption("Core");
        debugView->appendOption("Structure");
        debugView->appendOption("Effect");
        debugView->appendOption("Phase");
        debugView->appendOption("Phase Edge");
        debugView->appendOption("Field PSF");
        debugView->appendOption("Chromatic Split");
        debugView->appendOption("Creative Fringe");
        debugView->appendOption("Scatter");
        debugView->setDefault(lensdiffDefaultValues.debugView);
        page->addChild(*debugView);

        auto* presets = desc.defineGroupParam("PresetGroup");
        presets->setLabels("Defaults & Presets", "Defaults & Presets", "Defaults & Presets");
        presets->setOpen(false);

        auto* support = desc.defineGroupParam("SupportGroup");
        support->setLabels("Support", "Support", "Support");
        support->setOpen(false);

        auto* lensdiffPresetName = desc.defineStringParam("lensdiffPresetName");
        lensdiffPresetName->setLabel("Preset Name");
        lensdiffPresetName->setDefault("");
        lensdiffPresetName->setParent(*presets);

        auto* lensdiffPresetSave = desc.definePushButtonParam("lensdiffPresetSave");
        lensdiffPresetSave->setLabel("Save Preset");
        lensdiffPresetSave->setParent(*presets);

        auto* lensdiffPresetSaveDefaults = desc.definePushButtonParam("lensdiffPresetSaveDefaults");
        lensdiffPresetSaveDefaults->setLabel("Save Defaults");
        lensdiffPresetSaveDefaults->setParent(*presets);

        auto* lensdiffPresetUpdate = desc.definePushButtonParam("lensdiffPresetUpdate");
        lensdiffPresetUpdate->setLabel("Update Preset");
        lensdiffPresetUpdate->setEnabled(false);
        lensdiffPresetUpdate->setParent(*presets);

        auto* lensdiffPresetRename = desc.definePushButtonParam("lensdiffPresetRename");
        lensdiffPresetRename->setLabel("Rename Preset");
        lensdiffPresetRename->setEnabled(false);
        lensdiffPresetRename->setParent(*presets);

        auto* lensdiffPresetDelete = desc.definePushButtonParam("lensdiffPresetDelete");
        lensdiffPresetDelete->setLabel("Delete Preset");
        lensdiffPresetDelete->setEnabled(false);
        lensdiffPresetDelete->setParent(*presets);

        page->addChild(*defineDoubleParam(desc, "threshold", "Threshold", "Highlight threshold in stops above 18% gray.", selection, lensdiffDefaultValues.threshold, -8.0, 12.0, 0.1));
        page->addChild(*defineDoubleParam(desc, "softnessStops", "Softness", "Threshold softness in stops.", selection, lensdiffDefaultValues.softnessStops, 1.0, 6.0, 0.05));
        page->addChild(*defineChoiceParam(desc, "extractionMode", "Extraction Mode", "Highlight driver metric.", selection, {"Max RGB", "Luma"}, lensdiffDefaultValues.extractionMode));
        page->addChild(*defineDoubleParam(desc, "pointEmphasis", "Point Emphasis", "Boost compact highlights before diffraction.", selection, lensdiffDefaultValues.pointEmphasis, 0.0, 1.0, 0.01));
        page->addChild(*defineDoubleParam(desc, "corePreserve", "Core Preserve", "Fraction of selected highlight energy retained in-place.", selection, lensdiffDefaultValues.corePreserve, 0.0, 1.0, 0.01));

        auto* apertureMode = desc.defineChoiceParam("apertureMode");
        apertureMode->setLabel("Aperture Mode");
        apertureMode->setHint("Base pupil shape.");
        for (const char* label : kBuiltinApertureOptionLabels) {
            apertureMode->appendOption(label);
        }
        apertureMode->appendOption(kStaticCustomApertureOptionLabel);
        apertureMode->setDefault(defaultApertureChoiceIndex);
        apertureMode->setParent(*optics);
        page->addChild(*apertureMode);
        page->addChild(*definePushButtonParam(desc, "customApertureImport", "Custom", "Import a custom aperture image through a native file picker.", optics));
        auto* customAperturePath =
            defineStringParam(desc, "customAperturePath", "Custom Aperture File", "Hidden backing store for the selected custom aperture path.", optics, lensdiffDefaultValues.customAperturePath, OFX::eStringTypeFilePath, true);
        customAperturePath->setIsSecret(true);
        page->addChild(*customAperturePath);
        auto* customStatus = defineStringParam(desc, "customApertureStatus", "Custom Status", "Status for the currently loaded custom aperture.", optics, "No custom aperture loaded.", OFX::eStringTypeSingleLine, false);
        customStatus->setIsSecret(true);
        page->addChild(*customStatus);
        auto* customNormalize = defineBooleanParam(desc, "customApertureNormalize", "Custom Normalize", "Normalize the loaded custom aperture mask to the full 0..1 range before building the pupil. Binary masks may show little or no visible change.", optics, lensdiffDefaultValues.customApertureNormalize);
        customNormalize->setIsSecret(true);
        page->addChild(*customNormalize);
        auto* customInvert = defineBooleanParam(desc, "customApertureInvert", "Custom Invert", "Invert the loaded custom aperture mask so dark areas transmit and bright areas block.", optics, lensdiffDefaultValues.customApertureInvert);
        customInvert->setIsSecret(true);
        page->addChild(*customInvert);
        page->addChild(*defineIntParam(desc, "bladeCount", "Blade Count", "Polygon aperture blade count.", optics, lensdiffDefaultValues.bladeCount, 3, 16));
        page->addChild(*defineDoubleParam(desc, "roundness", "Roundness", "Blend between hard polygon and circular iris.", optics, lensdiffDefaultValues.roundness, 0.0, 1.0, 0.01));
        page->addChild(*defineDoubleParam(desc, "rotationDeg", "Rotation", "Aperture rotation in degrees.", optics, lensdiffDefaultValues.rotationDeg, -180.0, 180.0, 0.1));
        page->addChild(*defineDoubleParam(desc, "centralObstruction", "Central Obstruction", "Central pupil obstruction ratio.", optics, lensdiffDefaultValues.centralObstruction, 0.0, 0.95, 0.01));
        page->addChild(*defineIntParam(desc, "vaneCount", "Vane Count", "Number of support vane lines.", optics, lensdiffDefaultValues.vaneCount, 0, 8));
        page->addChild(*defineDoubleParam(desc, "vaneThickness", "Vane Thickness", "Relative thickness of vane obstructions.", optics, lensdiffDefaultValues.vaneThickness, 0.0, 0.1, 0.001));
        page->addChild(*defineChoiceParam(desc, "apodizationMode", "Apodization", "Radial pupil weighting.", optics, {"Flat", "Cosine", "Gaussian"}, lensdiffDefaultValues.apodizationMode));
        page->addChild(*defineDoubleParam(desc, "diffractionScalePx", "Diffraction Scale %", "PSF scale as a percent of the output frame short side.", optics, lensdiffDefaultValues.diffractionScalePx, 0.185185185185185, 23.7037037037037, 0.025));
        page->addChild(*defineIntParam(desc, "pupilResolution", "Pupil Resolution", "Resolution used to rasterize the pupil before embedding it into the internal FFT workspace.", optics, lensdiffDefaultValues.pupilResolution, 64, 1024));
        auto* legacyPupilResolutionChoice = defineChoiceParam(desc,
                                                              "pupilResolutionChoice",
                                                              "Pupil Resolution (Legacy Choice)",
                                                              "Hidden compatibility choice retained for older saved instances and presets.",
                                                              optics,
                                                               {"64", "128", "256", "512", "1024"},
                                                               choiceFromPupilResolution(lensdiffDefaultValues.pupilResolution));
        legacyPupilResolutionChoice->setIsSecret(true);
        page->addChild(*legacyPupilResolutionChoice);
        page->addChild(*defineDoubleParam(desc, "maxKernelRadiusPx", "Kernel Radius %", "Maximum diffraction support radius as a percent of the output frame short side.", optics, lensdiffDefaultValues.maxKernelRadiusPx, 0.37037037037037, 47.4074074074074, 0.05));
        auto* phasePrimary = desc.defineGroupParam("PhasePrimaryGroup");
        phasePrimary->setLabels("Primary", "Primary", "Primary");
        phasePrimary->setOpen(true);
        phasePrimary->setParent(*phase);
        page->addChild(*phasePrimary);

        auto* phaseField = desc.defineGroupParam("PhaseFieldGroup");
        phaseField->setLabels("Field Variation", "Field Variation", "Field Variation");
        phaseField->setOpen(false);
        phaseField->setParent(*phase);
        page->addChild(*phaseField);

        auto* phaseFieldStrength = defineDoubleParam(desc, "phaseFieldStrength", "Field Strength", "Blend from center aberration to a 3x3 field-varying PSF model.", phaseField, lensdiffDefaultValues.phaseFieldStrength, 0.0, 2.0, 0.01);
        phaseFieldStrength->setIsSecret(true);
        page->addChild(*phaseFieldStrength);

        auto* phaseFieldEdgeBias = defineDoubleParam(desc, "phaseFieldEdgeBias", "Edge Bias", "Bias how quickly field aberration grows toward the edge.", phaseField, lensdiffDefaultValues.phaseFieldEdgeBias, -1.0, 1.0, 0.01);
        phaseFieldEdgeBias->setIsSecret(true);
        page->addChild(*phaseFieldEdgeBias);

        auto* phaseFieldPrimary = desc.defineGroupParam("PhaseFieldPrimaryGroup");
        phaseFieldPrimary->setLabels("Primary", "Primary", "Primary");
        phaseFieldPrimary->setOpen(true);
        phaseFieldPrimary->setParent(*phaseField);
        page->addChild(*phaseFieldPrimary);

        auto* phaseFieldHigherOrder = desc.defineGroupParam("PhaseFieldHigherOrderGroup");
        phaseFieldHigherOrder->setLabels("Field Higher Order", "Field Higher Order", "Field Higher Order");
        phaseFieldHigherOrder->setOpen(false);
        phaseFieldHigherOrder->setParent(*phaseField);
        page->addChild(*phaseFieldHigherOrder);

        auto* phaseChromatic = desc.defineGroupParam("PhaseChromaticGroup");
        phaseChromatic->setLabels("Chromatic", "Chromatic", "Chromatic");
        phaseChromatic->setOpen(false);
        phaseChromatic->setParent(*phase);
        page->addChild(*phaseChromatic);

        auto* phaseFinishing = desc.defineGroupParam("PhaseFinishingGroup");
        phaseFinishing->setLabels("Finishing", "Finishing", "Finishing");
        phaseFinishing->setOpen(false);
        phaseFinishing->setParent(*phase);
        page->addChild(*phaseFinishing);

        auto* phaseFocus = defineDoubleParam(desc, "phaseFocus", "Focus", "Primary defocus control in waves.", phase, lensdiffDefaultValues.phaseDefocus, -2.0, 2.0, 0.01);
        phaseFocus->setIsSecret(true);
        page->addChild(*phaseFocus);
        auto* phaseAstigmatism = defineDoubleParam(desc, "phaseAstigmatism", "Astigmatism", "Primary astigmatism magnitude.", phase, phasePairMagnitude(lensdiffDefaultValues.phaseAstigmatism0, lensdiffDefaultValues.phaseAstigmatism45), 0.0, 2.5, 0.01);
        phaseAstigmatism->setIsSecret(true);
        page->addChild(*phaseAstigmatism);
        auto* phaseAstigAngleDeg = defineDoubleParam(desc, "phaseAstigAngleDeg", "Astig Angle", "Astigmatism angle in degrees.", phase, phasePairAngleDeg(lensdiffDefaultValues.phaseAstigmatism0, lensdiffDefaultValues.phaseAstigmatism45, 2), -180.0, 180.0, 0.1);
        phaseAstigAngleDeg->setIsSecret(true);
        page->addChild(*phaseAstigAngleDeg);
        auto* phaseComa = defineDoubleParam(desc, "phaseComa", "Coma", "Primary coma magnitude.", phase, phasePairMagnitude(lensdiffDefaultValues.phaseComaX, lensdiffDefaultValues.phaseComaY), 0.0, 2.5, 0.01);
        phaseComa->setIsSecret(true);
        page->addChild(*phaseComa);
        auto* phaseComaAngleDeg = defineDoubleParam(desc, "phaseComaAngleDeg", "Coma Angle", "Coma angle in degrees.", phase, phasePairAngleDeg(lensdiffDefaultValues.phaseComaX, lensdiffDefaultValues.phaseComaY, 1), -180.0, 180.0, 0.1);
        phaseComaAngleDeg->setIsSecret(true);
        page->addChild(*phaseComaAngleDeg);
        auto* phaseSphericalAmount = defineDoubleParam(desc, "phaseSphericalAmount", "Spherical", "Primary spherical aberration coefficient.", phase, lensdiffDefaultValues.phaseSpherical, -2.0, 2.0, 0.01);
        phaseSphericalAmount->setIsSecret(true);
        page->addChild(*phaseSphericalAmount);
        auto* phaseTrefoil = defineDoubleParam(desc, "phaseTrefoil", "Trefoil", "Primary trefoil magnitude.", phase, phasePairMagnitude(lensdiffDefaultValues.phaseTrefoilX, lensdiffDefaultValues.phaseTrefoilY), 0.0, 2.5, 0.01);
        phaseTrefoil->setIsSecret(true);
        page->addChild(*phaseTrefoil);
        auto* phaseTrefoilAngleDeg = defineDoubleParam(desc, "phaseTrefoilAngleDeg", "Trefoil Angle", "Trefoil angle in degrees.", phase, phasePairAngleDeg(lensdiffDefaultValues.phaseTrefoilX, lensdiffDefaultValues.phaseTrefoilY, 3), -180.0, 180.0, 0.1);
        phaseTrefoilAngleDeg->setIsSecret(true);
        page->addChild(*phaseTrefoilAngleDeg);
        auto* chromaticFocus = defineDoubleParam(desc, "chromaticFocus", "Chromatic Focus", "Wavelength-dependent phase defocus shift applied before RGB reconstruction.", phaseChromatic, lensdiffDefaultValues.chromaticFocus, -2.0, 2.0, 0.01);
        chromaticFocus->setIsSecret(true);
        page->addChild(*chromaticFocus);
        auto* chromaticSpread = defineDoubleParam(desc, "chromaticSpread", "Physical Chromatic Spread", "Wavelength-dependent PSF scale spread applied in the spectral bank.", phaseChromatic, lensdiffDefaultValues.chromaticSpread, -1.0, 1.0, 0.01);
        chromaticSpread->setIsSecret(true);
        page->addChild(*chromaticSpread);
        auto* creativeFringe = defineDoubleParam(desc, "creativeFringe", "Creative Fringe %", "Post-reconstruction RGB fringe shift as a percent of the output frame short side.", phaseFinishing, lensdiffDefaultValues.creativeFringe, 0.0, 0.37037037037037, 0.0005);
        creativeFringe->setIsSecret(true);
        page->addChild(*creativeFringe);
        auto* scatterAmount = defineDoubleParam(desc, "scatterAmount", "Scatter Amount", "Effect-only veil amount added after redistribution.", phaseFinishing, lensdiffDefaultValues.scatterAmount, 0.0, 2.0, 0.01);
        scatterAmount->setIsSecret(true);
        page->addChild(*scatterAmount);
        auto* scatterRadius = defineDoubleParam(desc, "scatterRadius", "Scatter Radius %", "Global scatter radius as a percent of the output frame short side.", phaseFinishing, lensdiffDefaultValues.scatterRadius, 0.0, 11.8518518518519, 0.025);
        scatterRadius->setIsSecret(true);
        page->addChild(*scatterRadius);

        auto* phaseAdvanced = desc.defineGroupParam("PhaseAdvancedGroup");
        phaseAdvanced->setLabels("Higher Order", "Higher Order", "Higher Order");
        phaseAdvanced->setOpen(false);
        phaseAdvanced->setParent(*phase);
        page->addChild(*phaseAdvanced);

        auto* phaseDefocus = defineDoubleParam(desc, "phaseDefocus", "Defocus", "Resolved Zernike defocus coefficient in waves.", phasePrimary, lensdiffDefaultValues.phaseDefocus, -2.0, 2.0, 0.01);
        phaseDefocus->setIsSecret(true);
        page->addChild(*phaseDefocus);
        auto* phaseAstigmatism0 = defineDoubleParam(desc, "phaseAstigmatism0", "Astig 0", "Zernike astigmatism aligned to the pupil axes.", phasePrimary, lensdiffDefaultValues.phaseAstigmatism0, -2.0, 2.0, 0.01);
        phaseAstigmatism0->setIsSecret(true);
        page->addChild(*phaseAstigmatism0);
        auto* phaseAstigmatism45 = defineDoubleParam(desc, "phaseAstigmatism45", "Astig 45", "Zernike astigmatism on the diagonal axes.", phasePrimary, lensdiffDefaultValues.phaseAstigmatism45, -2.0, 2.0, 0.01);
        phaseAstigmatism45->setIsSecret(true);
        page->addChild(*phaseAstigmatism45);
        auto* phaseComaX = defineDoubleParam(desc, "phaseComaX", "Coma X", "Zernike coma X coefficient.", phasePrimary, lensdiffDefaultValues.phaseComaX, -2.0, 2.0, 0.01);
        phaseComaX->setIsSecret(true);
        page->addChild(*phaseComaX);
        auto* phaseComaY = defineDoubleParam(desc, "phaseComaY", "Coma Y", "Zernike coma Y coefficient.", phasePrimary, lensdiffDefaultValues.phaseComaY, -2.0, 2.0, 0.01);
        phaseComaY->setIsSecret(true);
        page->addChild(*phaseComaY);
        auto* phaseSpherical = defineDoubleParam(desc, "phaseSpherical", "Spherical", "Zernike spherical coefficient.", phasePrimary, lensdiffDefaultValues.phaseSpherical, -2.0, 2.0, 0.01);
        phaseSpherical->setIsSecret(true);
        page->addChild(*phaseSpherical);
        auto* phaseTrefoilX = defineDoubleParam(desc, "phaseTrefoilX", "Trefoil X", "Zernike trefoil X coefficient.", phaseAdvanced, lensdiffDefaultValues.phaseTrefoilX, -2.0, 2.0, 0.01);
        phaseTrefoilX->setIsSecret(true);
        page->addChild(*phaseTrefoilX);
        auto* phaseTrefoilY = defineDoubleParam(desc, "phaseTrefoilY", "Trefoil Y", "Zernike trefoil Y coefficient.", phaseAdvanced, lensdiffDefaultValues.phaseTrefoilY, -2.0, 2.0, 0.01);
        phaseTrefoilY->setIsSecret(true);
        page->addChild(*phaseTrefoilY);
        auto* phaseSecondaryAstigmatism0 = defineDoubleParam(desc, "phaseSecondaryAstigmatism0", "Secondary Astig 0", "Higher-order secondary astigmatism aligned to the pupil axes.", phaseAdvanced, lensdiffDefaultValues.phaseSecondaryAstigmatism0, -2.0, 2.0, 0.01);
        phaseSecondaryAstigmatism0->setIsSecret(true);
        page->addChild(*phaseSecondaryAstigmatism0);
        auto* phaseSecondaryAstigmatism45 = defineDoubleParam(desc, "phaseSecondaryAstigmatism45", "Secondary Astig 45", "Higher-order secondary astigmatism on the diagonal axes.", phaseAdvanced, lensdiffDefaultValues.phaseSecondaryAstigmatism45, -2.0, 2.0, 0.01);
        phaseSecondaryAstigmatism45->setIsSecret(true);
        page->addChild(*phaseSecondaryAstigmatism45);
        auto* phaseQuadrafoil0 = defineDoubleParam(desc, "phaseQuadrafoil0", "Quadrafoil 0", "Higher-order quadrafoil aligned to the pupil axes.", phaseAdvanced, lensdiffDefaultValues.phaseQuadrafoil0, -2.0, 2.0, 0.01);
        phaseQuadrafoil0->setIsSecret(true);
        page->addChild(*phaseQuadrafoil0);
        auto* phaseQuadrafoil45 = defineDoubleParam(desc, "phaseQuadrafoil45", "Quadrafoil 45", "Higher-order quadrafoil on the diagonal axes.", phaseAdvanced, lensdiffDefaultValues.phaseQuadrafoil45, -2.0, 2.0, 0.01);
        phaseQuadrafoil45->setIsSecret(true);
        page->addChild(*phaseQuadrafoil45);
        auto* phaseSecondaryComaX = defineDoubleParam(desc, "phaseSecondaryComaX", "Secondary Coma X", "Higher-order secondary coma X coefficient.", phaseAdvanced, lensdiffDefaultValues.phaseSecondaryComaX, -2.0, 2.0, 0.01);
        phaseSecondaryComaX->setIsSecret(true);
        page->addChild(*phaseSecondaryComaX);
        auto* phaseSecondaryComaY = defineDoubleParam(desc, "phaseSecondaryComaY", "Secondary Coma Y", "Higher-order secondary coma Y coefficient.", phaseAdvanced, lensdiffDefaultValues.phaseSecondaryComaY, -2.0, 2.0, 0.01);
        phaseSecondaryComaY->setIsSecret(true);
        page->addChild(*phaseSecondaryComaY);
        auto* pupilDecenterX = defineDoubleParam(desc, "pupilDecenterX", "Pupil Decenter X", "Horizontal pupil offset in normalized pupil space.", phaseAdvanced, lensdiffDefaultValues.pupilDecenterX, -0.5, 0.5, 0.001);
        pupilDecenterX->setIsSecret(true);
        page->addChild(*pupilDecenterX);
        auto* pupilDecenterY = defineDoubleParam(desc, "pupilDecenterY", "Pupil Decenter Y", "Vertical pupil offset in normalized pupil space.", phaseAdvanced, lensdiffDefaultValues.pupilDecenterY, -0.5, 0.5, 0.001);
        pupilDecenterY->setIsSecret(true);
        page->addChild(*pupilDecenterY);
        auto* phaseFieldDefocus = defineDoubleParam(desc, "phaseFieldDefocus", "Defocus", "Edge delta applied to defocus.", phaseFieldPrimary, lensdiffDefaultValues.phaseFieldDefocus, -2.0, 2.0, 0.01);
        phaseFieldDefocus->setIsSecret(true);
        page->addChild(*phaseFieldDefocus);
        auto* phaseFieldAstigRadial = defineDoubleParam(desc, "phaseFieldAstigRadial", "Astig Radial", "Edge delta for radial astigmatism.", phaseFieldPrimary, lensdiffDefaultValues.phaseFieldAstigRadial, -2.0, 2.0, 0.01);
        phaseFieldAstigRadial->setIsSecret(true);
        page->addChild(*phaseFieldAstigRadial);
        auto* phaseFieldAstigTangential = defineDoubleParam(desc, "phaseFieldAstigTangential", "Astig Tangential", "Edge delta for tangential astigmatism.", phaseFieldPrimary, lensdiffDefaultValues.phaseFieldAstigTangential, -2.0, 2.0, 0.01);
        phaseFieldAstigTangential->setIsSecret(true);
        page->addChild(*phaseFieldAstigTangential);
        auto* phaseFieldComaRadial = defineDoubleParam(desc, "phaseFieldComaRadial", "Coma Radial", "Edge delta for radial coma.", phaseFieldPrimary, lensdiffDefaultValues.phaseFieldComaRadial, -2.0, 2.0, 0.01);
        phaseFieldComaRadial->setIsSecret(true);
        page->addChild(*phaseFieldComaRadial);
        auto* phaseFieldComaTangential = defineDoubleParam(desc, "phaseFieldComaTangential", "Coma Tangential", "Edge delta for tangential coma.", phaseFieldPrimary, lensdiffDefaultValues.phaseFieldComaTangential, -2.0, 2.0, 0.01);
        phaseFieldComaTangential->setIsSecret(true);
        page->addChild(*phaseFieldComaTangential);
        auto* phaseFieldSpherical = defineDoubleParam(desc, "phaseFieldSpherical", "Spherical", "Edge delta for spherical aberration.", phaseFieldPrimary, lensdiffDefaultValues.phaseFieldSpherical, -2.0, 2.0, 0.01);
        phaseFieldSpherical->setIsSecret(true);
        page->addChild(*phaseFieldSpherical);
        auto* phaseFieldTrefoilRadial = defineDoubleParam(desc, "phaseFieldTrefoilRadial", "Trefoil Radial", "Edge delta for radial trefoil.", phaseFieldHigherOrder, lensdiffDefaultValues.phaseFieldTrefoilRadial, -2.0, 2.0, 0.01);
        phaseFieldTrefoilRadial->setIsSecret(true);
        page->addChild(*phaseFieldTrefoilRadial);
        auto* phaseFieldTrefoilTangential = defineDoubleParam(desc, "phaseFieldTrefoilTangential", "Trefoil Tangential", "Edge delta for tangential trefoil.", phaseFieldHigherOrder, lensdiffDefaultValues.phaseFieldTrefoilTangential, -2.0, 2.0, 0.01);
        phaseFieldTrefoilTangential->setIsSecret(true);
        page->addChild(*phaseFieldTrefoilTangential);
        auto* phaseFieldSecondaryAstigRadial = defineDoubleParam(desc, "phaseFieldSecondaryAstigRadial", "Secondary Astig Radial", "Edge delta for radial secondary astigmatism.", phaseFieldHigherOrder, lensdiffDefaultValues.phaseFieldSecondaryAstigRadial, -2.0, 2.0, 0.01);
        phaseFieldSecondaryAstigRadial->setIsSecret(true);
        page->addChild(*phaseFieldSecondaryAstigRadial);
        auto* phaseFieldSecondaryAstigTangential = defineDoubleParam(desc, "phaseFieldSecondaryAstigTangential", "Secondary Astig Tangential", "Edge delta for tangential secondary astigmatism.", phaseFieldHigherOrder, lensdiffDefaultValues.phaseFieldSecondaryAstigTangential, -2.0, 2.0, 0.01);
        phaseFieldSecondaryAstigTangential->setIsSecret(true);
        page->addChild(*phaseFieldSecondaryAstigTangential);
        auto* phaseFieldQuadrafoilRadial = defineDoubleParam(desc, "phaseFieldQuadrafoilRadial", "Quadrafoil Radial", "Edge delta for radial quadrafoil.", phaseFieldHigherOrder, lensdiffDefaultValues.phaseFieldQuadrafoilRadial, -2.0, 2.0, 0.01);
        phaseFieldQuadrafoilRadial->setIsSecret(true);
        page->addChild(*phaseFieldQuadrafoilRadial);
        auto* phaseFieldQuadrafoilTangential = defineDoubleParam(desc, "phaseFieldQuadrafoilTangential", "Quadrafoil Tangential", "Edge delta for tangential quadrafoil.", phaseFieldHigherOrder, lensdiffDefaultValues.phaseFieldQuadrafoilTangential, -2.0, 2.0, 0.01);
        phaseFieldQuadrafoilTangential->setIsSecret(true);
        page->addChild(*phaseFieldQuadrafoilTangential);
        auto* phaseFieldSecondaryComaRadial = defineDoubleParam(desc, "phaseFieldSecondaryComaRadial", "Secondary Coma Radial", "Edge delta for radial secondary coma.", phaseFieldHigherOrder, lensdiffDefaultValues.phaseFieldSecondaryComaRadial, -2.0, 2.0, 0.01);
        phaseFieldSecondaryComaRadial->setIsSecret(true);
        page->addChild(*phaseFieldSecondaryComaRadial);
        auto* phaseFieldSecondaryComaTangential = defineDoubleParam(desc, "phaseFieldSecondaryComaTangential", "Secondary Coma Tangential", "Edge delta for tangential secondary coma.", phaseFieldHigherOrder, lensdiffDefaultValues.phaseFieldSecondaryComaTangential, -2.0, 2.0, 0.01);
        phaseFieldSecondaryComaTangential->setIsSecret(true);
        page->addChild(*phaseFieldSecondaryComaTangential);

        page->addChild(*defineChoiceParam(desc, "spectralMode", "Spectral Mode", "Active spectral bank mode.", spectrum, {"Mono", "Tristimulus", "Spectral 5", "Spectral 9"}, lensdiffDefaultValues.spectralMode));
        page->addChild(*defineChoiceParam(desc, "spectrumStyle", "Spectrum Style", "Wavelength-to-RGB mapping style.", spectrum, {"Natural", "Cyan-Magenta", "Warm-Cool"}, lensdiffDefaultValues.spectrumStyle));
        page->addChild(*defineDoubleParam(desc, "spectrumForce", "Spectrum Force", "Blend between Natural and the selected style.", spectrum, lensdiffDefaultValues.spectrumForce, 0.0, 1.0, 0.01));
        page->addChild(*defineDoubleParam(desc, "spectrumSaturation", "Spectrum Saturation", "Additional saturation on mapped spectral output.", spectrum, lensdiffDefaultValues.spectrumSaturation, 0.0, 2.0, 0.01));
        page->addChild(*defineBooleanParam(desc, "chromaticAffectsLuma", "Chromatic Affects Luma", "Allow style mapping to change output luminance.", spectrum, lensdiffDefaultValues.chromaticAffectsLuma));

        page->addChild(*defineChoiceParam(desc, "lookMode", "Look Mode", "Physical or split core/structure application.", composite, {"Physical", "Split"}, lensdiffDefaultValues.lookMode));
        page->addChild(*defineChoiceParam(desc, "energyMode", "Energy Mode", "Preserve or augment selected energy.", composite, {"Preserve", "Augment"}, lensdiffDefaultValues.energyMode));
        page->addChild(*defineDoubleParam(desc, "effectGain", "Effect Gain", "Overall effect gain.", composite, lensdiffDefaultValues.effectGain, 0.0, 4.0, 0.01));
        page->addChild(*defineDoubleParam(desc, "coreCompensation", "Core Compensation", "Amount removed from the selected source before adding the effect.", composite, lensdiffDefaultValues.coreCompensation, 0.0, 4.0, 0.01));
        page->addChild(*defineDoubleParam(desc, "anisotropyEmphasis", "Anisotropy Emphasis", "Boost directional PSF structure while preserving unit kernel energy.", composite, lensdiffDefaultValues.anisotropyEmphasis, 0.0, 1.0, 0.01));
        auto* compositeSplit = desc.defineGroupParam("CompositeSplitGroup");
        compositeSplit->setLabels("Split Look", "Split Look", "Split Look");
        compositeSplit->setOpen(false);
        compositeSplit->setParent(*composite);
        page->addChild(*compositeSplit);
        page->addChild(*defineDoubleParam(desc, "coreGain", "Core Gain", "Primary redistribution gain for isotropic energy.", compositeSplit, lensdiffDefaultValues.coreGain, 0.0, 4.0, 0.01));
        page->addChild(*defineDoubleParam(desc, "structureGain", "Structure Gain", "Primary redistribution gain for directional energy.", compositeSplit, lensdiffDefaultValues.structureGain, 0.0, 4.0, 0.01));
        page->addChild(*defineDoubleParam(desc, "coreShoulder", "Core Shoulder", "Soft compression applied to the veil component.", compositeSplit, lensdiffDefaultValues.coreShoulder, 0.0, 8.0, 0.05));
        page->addChild(*defineDoubleParam(desc, "structureShoulder", "Structure Shoulder", "Soft compression applied to directional structure.", compositeSplit, lensdiffDefaultValues.structureShoulder, 0.0, 8.0, 0.05));
        auto* backendPreference = desc.defineChoiceParam("backendPreference");
        backendPreference->setLabel("Backend Preference");
        backendPreference->setHint("Requested execution backend.");
        backendPreference->appendOption("Auto");
        backendPreference->appendOption("CPU");
        backendPreference->appendOption("CUDA");
        backendPreference->appendOption("Metal");
        backendPreference->setDefault(lensdiffDefaultValues.backendPreference);
        backendPreference->setParent(*composite);
        backendPreference->setIsSecret(true);

        page->addChild(*presets);
        page->addChild(*support);
        page->addChild(*definePushButtonParam(desc, "lensdiffWebsite", "Website", "Open moazelgabry.com in your default browser.", support));
        page->addChild(*defineLabelParam(desc, "supportVersion", "Version", "LensDiff build version.", support, kPluginDisplayVersion));
    }

    OFX::ImageEffect* createInstance(OfxImageEffectHandle handle, OFX::ContextEnum) override {
        return new LensDiffEffect(handle);
    }
};

} // namespace

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& ids) {
    static LensDiffFactory factory;
    ids.push_back(&factory);
}

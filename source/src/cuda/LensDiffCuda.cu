#include "LensDiffCuda.h"
#include "LensDiffCudaVkFFT.h"

#include "../core/LensDiffApertureImage.h"
#include "../core/LensDiffCpuReference.h"
#include "../core/LensDiffDiagnostics.h"
#include "../core/LensDiffPhase.h"
#include "../core/LensDiffSpectrum.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kGray18 = 0.18f;
constexpr float kMinimumSelectedCoreFloor = 0.2f;
constexpr float kZernikeDefocusNorm = 1.7320508075688772f;
constexpr float kZernikeAstigNorm = 2.4494897427831781f;
constexpr float kZernikeComaNorm = 2.8284271247461901f;
constexpr float kZernikeSphericalNorm = 2.2360679774997898f;
constexpr float kZernikeTrefoilNorm = 2.8284271247461901f;
constexpr float kZernikeSecondaryAstigNorm = 3.1622776601683795f;
constexpr float kZernikeQuadrafoilNorm = 3.1622776601683795f;
constexpr float kZernikeSecondaryComaNorm = 3.4641016151377544f;

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    std::size_t count = 0;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr(other.ptr), count(other.count) {
        other.ptr = nullptr;
        other.count = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr = other.ptr;
            count = other.count;
            other.ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }

    ~DeviceBuffer() { release(); }

    bool allocate(std::size_t newCount) {
        if (newCount == 0) {
            release();
            return true;
        }
        if (ptr != nullptr && count == newCount) {
            return true;
        }
        release();
        if (cudaMalloc(&ptr, newCount * sizeof(T)) != cudaSuccess) {
            ptr = nullptr;
            count = 0;
            return false;
        }
        count = newCount;
        return true;
    }

    void release() {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
            count = 0;
        }
    }
};

struct PackedHostImage {
    int width = 0;
    int height = 0;
    std::vector<float> pixels;
};

struct PlaneSet {
    DeviceBuffer<float> r;
    DeviceBuffer<float> g;
    DeviceBuffer<float> b;
};

struct SpectralPlaneSet {
    std::array<DeviceBuffer<float>, kLensDiffMaxSpectralBins> bins;
};

struct SpectralMapConfigGpu {
    int binCount = 0;
    float naturalMatrix[kLensDiffMaxSpectralBins * 3] = {0.0f};
    float styleMatrix[kLensDiffMaxSpectralBins * 3] = {0.0f};
};

struct KernelStatsHost {
    std::vector<float> sums;
    std::vector<int> counts;
    std::vector<float> ringEnergy;
    std::vector<float> ringPeak;
};

enum class CudaPlanLayoutKind : int {
    Single2D = 0,
    Batched2D = 1,
};

enum class CudaFftBackend : int {
    CuFFT = 0,
    VkFFT = 1,
};

enum class CudaFftSubsystem : int {
    GlobalRender = 0,
    PsfBank = 1,
    FieldRender = 2,
};

enum class CudaFftPolicy : int {
    Auto = 0,
    Legacy = 1,
    VkFFTFirst = 2,
};

enum class CudaFftBackendOverride : int {
    Default = 0,
    CuFFT = 1,
    VkFFT = 2,
};

struct CudaSubsystemFftStats {
    std::string requested = "default";
    std::string effective = "none";
    std::string fallback;
    int cufftPlanCacheHits = 0;
    int cufftPlanCacheMisses = 0;
    int vkfftPlanCacheHits = 0;
    int vkfftPlanCacheMisses = 0;
    double vkfftCompileInitMs = 0.0;
    std::size_t maxPlanWorkBytes = 0;
};

struct CudaPlanKey {
    int width = 0;
    int height = 0;
    int batchCount = 0;
    int layoutKind = 0;

    bool operator==(const CudaPlanKey& other) const {
        return width == other.width &&
               height == other.height &&
               batchCount == other.batchCount &&
               layoutKind == other.layoutKind;
    }
};

struct CudaPlanKeyHasher {
    std::size_t operator()(const CudaPlanKey& key) const noexcept {
        std::size_t hash = static_cast<std::size_t>(key.width);
        hash = hash * 1315423911u + static_cast<std::size_t>(key.height);
        hash = hash * 2654435761u + static_cast<std::size_t>(key.batchCount);
        hash = hash * 2246822519u + static_cast<std::size_t>(key.layoutKind);
        return hash;
    }
};

struct CudaPsfBuildContext {
    DeviceBuffer<float> rawPsf;
    DeviceBuffer<float> shiftedIntensity;
    DeviceBuffer<cufftComplex> rawSpectrum;
    DeviceBuffer<float> baseKernel;
    DeviceBuffer<float> meanKernel;
    DeviceBuffer<float> shapedKernel;
    DeviceBuffer<float> structureKernel;
    DeviceBuffer<float> ringSums;
    DeviceBuffer<int> ringCounts;
    DeviceBuffer<float> ringEnergy;
    DeviceBuffer<float> ringPeak;
    DeviceBuffer<float> cropCore;
    DeviceBuffer<float> cropFull;
    DeviceBuffer<float> cropStructure;
    DeviceBuffer<float> reductionScalarA;
    DeviceBuffer<float> reductionScalarB;
    DeviceBuffer<float> scalarScale;
    DeviceBuffer<float> totalEnergy;
    DeviceBuffer<float> globalPeak;
};

struct PersistentCudaPlanRepository {
    struct PlanEntry {
        cufftHandle handle = 0;
        DeviceBuffer<unsigned char> workspace;
        std::size_t workBytes = 0;
        bool inUse = false;
        std::uint64_t stamp = 0;
        cudaEvent_t readyEvent = nullptr;

        ~PlanEntry() {
            if (handle != 0) {
                cufftDestroy(handle);
                handle = 0;
            }
            if (readyEvent != nullptr) {
                cudaEventDestroy(readyEvent);
                readyEvent = nullptr;
            }
        }
    };

    std::mutex mutex;
    std::unordered_map<CudaPlanKey, std::vector<std::unique_ptr<PlanEntry>>, CudaPlanKeyHasher> entries;
    std::uint64_t nextStamp = 0;
};

struct CudaPlanLease {
    PersistentCudaPlanRepository* repository = nullptr;
    PersistentCudaPlanRepository::PlanEntry* entry = nullptr;
    cufftHandle standaloneHandle = 0;
    cudaStream_t releaseStream = nullptr;

    CudaPlanLease() = default;
    CudaPlanLease(const CudaPlanLease&) = delete;
    CudaPlanLease& operator=(const CudaPlanLease&) = delete;

    CudaPlanLease(CudaPlanLease&& other) noexcept
        : repository(other.repository),
          entry(other.entry),
          standaloneHandle(other.standaloneHandle),
          releaseStream(other.releaseStream) {
        other.repository = nullptr;
        other.entry = nullptr;
        other.standaloneHandle = 0;
        other.releaseStream = nullptr;
    }

    CudaPlanLease& operator=(CudaPlanLease&& other) noexcept {
        if (this != &other) {
            release();
            repository = other.repository;
            entry = other.entry;
            standaloneHandle = other.standaloneHandle;
            releaseStream = other.releaseStream;
            other.repository = nullptr;
            other.entry = nullptr;
            other.standaloneHandle = 0;
            other.releaseStream = nullptr;
        }
        return *this;
    }

    ~CudaPlanLease() { release(); }

    cufftHandle handle() const { return entry != nullptr ? entry->handle : standaloneHandle; }
    std::size_t workBytes() const { return entry != nullptr ? entry->workBytes : 0; }

    void release() {
        if (standaloneHandle != 0) {
            cufftDestroy(standaloneHandle);
            standaloneHandle = 0;
        }
        if (repository == nullptr || entry == nullptr) {
            releaseStream = nullptr;
            return;
        }
        if (entry->readyEvent != nullptr && releaseStream != nullptr) {
            cudaEventRecord(entry->readyEvent, releaseStream);
        }
        std::lock_guard<std::mutex> lock(repository->mutex);
        entry->inUse = false;
        repository = nullptr;
        entry = nullptr;
        releaseStream = nullptr;
    }
};

struct CudaFftPlanLease {
    CudaFftBackend backend = CudaFftBackend::CuFFT;
    CudaPlanLease cufftLease;
    LensDiffCudaVkFFTPlanLease vkfftLease;

    CudaFftPlanLease() = default;
    CudaFftPlanLease(const CudaFftPlanLease&) = delete;
    CudaFftPlanLease& operator=(const CudaFftPlanLease&) = delete;

    CudaFftPlanLease(CudaFftPlanLease&&) noexcept = default;
    CudaFftPlanLease& operator=(CudaFftPlanLease&&) noexcept = default;

    void release() {
        cufftLease.release();
        vkfftLease.release();
    }

    std::size_t workBytes() const {
        return backend == CudaFftBackend::VkFFT ? vkfftLease.workBytes() : cufftLease.workBytes();
    }
};

struct CudaRenderTimingBreakdown {
    double psfBankMs = 0.0;
    double sourceFftMs = 0.0;
    double kernelFftMs = 0.0;
    double convolutionMs = 0.0;
    double fieldZonesMs = 0.0;
    double scatterMs = 0.0;
    double creativeFringeMs = 0.0;
    double nativeResampleMs = 0.0;
    double compositeMs = 0.0;
    double outputCopyMs = 0.0;
    int rgbSourceCacheHits = 0;
    int rgbSourceCacheMisses = 0;
    int scalarSourceCacheHits = 0;
    int scalarSourceCacheMisses = 0;
    int kernelCacheHits = 0;
    int kernelCacheMisses = 0;
    int cufftPlanCacheHits = 0;
    int cufftPlanCacheMisses = 0;
    int vkfftPlanCacheHits = 0;
    int vkfftPlanCacheMisses = 0;
    int fieldZoneBatchDepth = 0;
    int hostSyncCount = 0;
    std::size_t maxPlanWorkBytes = 0;
    double vkfftCompileInitMs = 0.0;
    std::uint64_t fieldScratchEstimateBytes = 0;
    std::uint64_t fieldKeyHash = 0;
    std::uint64_t psfKeyHash = 0;
    double validationMs = 0.0;
    float validationEffectMaxAbs = 0.0f;
    float validationCoreMaxAbs = 0.0f;
    float validationStructureMaxAbs = 0.0f;
    bool validationEnabled = false;
    bool validationRan = false;
    bool validationReferenceLegacy = false;
    std::string validationNote;
    std::string fieldBranch;
    std::string fieldWeightSpace;
    std::string fftPolicy = "vkfft-first";
    std::string fftRequested = "vkfft";
    std::string fftEffective = "vkfft";
    std::string fftFallbackNote;
    CudaSubsystemFftStats globalFft;
    CudaSubsystemFftStats psfFft;
    CudaSubsystemFftStats fieldFft;
};

struct CudaRenderContext {
    cudaStream_t stream = nullptr;
    PersistentCudaPlanRepository* planRepository = nullptr;
    CudaRenderTimingBreakdown* timing = nullptr;
    std::string* error = nullptr;
};

std::uint64_t hashBytesFnv1a64(const void* data, std::size_t byteCount) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash = 1469598103934665603ull;
    for (std::size_t index = 0; index < byteCount; ++index) {
        hash ^= static_cast<std::uint64_t>(bytes[index]);
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string persistentKernelSpectrumCacheKey(const std::string& cacheNamespace,
                                             int deviceId,
                                             const LensDiffKernel& kernel,
                                             int paddedWidth,
                                             int paddedHeight) {
    const std::uint64_t valueHash =
        hashBytesFnv1a64(kernel.values.data(), kernel.values.size() * sizeof(float));
    return cacheNamespace + ":" +
           std::to_string(deviceId) + ":" +
           std::to_string(kernel.size) + ":" +
           std::to_string(kernel.values.size()) + ":" +
           std::to_string(paddedWidth) + "x" + std::to_string(paddedHeight) + ":" +
           std::to_string(static_cast<unsigned long long>(valueHash));
}

struct PersistentCudaKernelSpectrumCache {
    struct Entry {
        std::shared_ptr<DeviceBuffer<cufftComplex>> spectrum;
        std::size_t bytes = 0;
        std::uint64_t stamp = 0;
        cudaEvent_t readyEvent = nullptr;

        Entry() = default;
        Entry(const Entry&) = delete;
        Entry& operator=(const Entry&) = delete;

        Entry(Entry&& other) noexcept
            : spectrum(std::move(other.spectrum)),
              bytes(other.bytes),
              stamp(other.stamp),
              readyEvent(other.readyEvent) {
            other.bytes = 0;
            other.stamp = 0;
            other.readyEvent = nullptr;
        }

        Entry& operator=(Entry&& other) noexcept {
            if (this != &other) {
                release();
                spectrum = std::move(other.spectrum);
                bytes = other.bytes;
                stamp = other.stamp;
                readyEvent = other.readyEvent;
                other.bytes = 0;
                other.stamp = 0;
                other.readyEvent = nullptr;
            }
            return *this;
        }

        ~Entry() { release(); }

        void release() {
            if (readyEvent != nullptr) {
                cudaEventDestroy(readyEvent);
                readyEvent = nullptr;
            }
        }
    };

    std::mutex mutex;
    std::unordered_map<std::string, Entry> entries;
    std::size_t totalBytes = 0;
    std::uint64_t nextStamp = 0;
};

PersistentCudaKernelSpectrumCache& persistentCudaKernelSpectrumCache() {
    static PersistentCudaKernelSpectrumCache cache;
    return cache;
}

PersistentCudaPlanRepository& persistentCudaPlanRepository() {
    static PersistentCudaPlanRepository repository;
    return repository;
}

bool LensDiffCudaFieldValidateEnabled() {
    const char* value = std::getenv("LENSDIFF_CUDA_FIELD_VALIDATE");
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" && text != "OFF";
}

bool LensDiffCudaUseFrameWeightSpace() {
    const char* value = std::getenv("LENSDIFF_CUDA_FIELD_WEIGHT_SPACE");
    if (value == nullptr || *value == '\0') {
        return true;
    }
    const std::string text(value);
    if (text == "working" || text == "WORKING" || text == "Working") {
        return false;
    }
    return true;
}

bool LensDiffCudaExperimentalTiledFieldEnabled() {
    const char* value = std::getenv("LENSDIFF_CUDA_FIELD_TILED");
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" && text != "OFF";
}

bool LensDiffCudaExperimentalStackedFieldEnabled() {
    const char* value = std::getenv("LENSDIFF_CUDA_FIELD_STACKED");
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" && text != "OFF";
}

bool LensDiffCudaPersistentKernelCacheEnabled() {
    const char* value = std::getenv("LENSDIFF_CUDA_PERSISTENT_KERNEL_CACHE");
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" && text != "OFF";
}

bool LensDiffCudaBatchedFieldCacheEnabled() {
    const char* value = std::getenv("LENSDIFF_CUDA_BATCHED_FIELD_CACHE");
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" && text != "OFF";
}

std::uint64_t hashStringFnv1a64(const std::string& text) {
    return hashBytesFnv1a64(text.data(), text.size());
}

template <typename T>
void hashAppendValue(std::ostringstream& os, const T& value) {
    if constexpr (std::is_enum_v<T>) {
        os << static_cast<int>(value) << '|';
    } else {
        os << value << '|';
    }
}

std::uint64_t diagnosticFieldKeyHash(const LensDiffFieldKey& key) {
    std::ostringstream os;
    hashAppendValue(os, key.phaseFieldStrength);
    hashAppendValue(os, key.phaseFieldEdgeBias);
    hashAppendValue(os, key.phaseFieldDefocus);
    hashAppendValue(os, key.phaseFieldAstigRadial);
    hashAppendValue(os, key.phaseFieldAstigTangential);
    hashAppendValue(os, key.phaseFieldComaRadial);
    hashAppendValue(os, key.phaseFieldComaTangential);
    hashAppendValue(os, key.phaseFieldSpherical);
    hashAppendValue(os, key.phaseFieldTrefoilRadial);
    hashAppendValue(os, key.phaseFieldTrefoilTangential);
    hashAppendValue(os, key.phaseFieldSecondaryAstigRadial);
    hashAppendValue(os, key.phaseFieldSecondaryAstigTangential);
    hashAppendValue(os, key.phaseFieldQuadrafoilRadial);
    hashAppendValue(os, key.phaseFieldQuadrafoilTangential);
    hashAppendValue(os, key.phaseFieldSecondaryComaRadial);
    hashAppendValue(os, key.phaseFieldSecondaryComaTangential);
    return hashStringFnv1a64(os.str());
}

std::uint64_t diagnosticPsfKeyHash(const LensDiffPsfBankKey& key) {
    std::ostringstream os;
    hashAppendValue(os, key.apertureMode);
    hashAppendValue(os, key.apodizationMode);
    hashAppendValue(os, key.spectralMode);
    hashAppendValue(os, key.bladeCount);
    hashAppendValue(os, key.vaneCount);
    hashAppendValue(os, key.pupilResolution);
    hashAppendValue(os, key.frameShortSidePx);
    hashAppendValue(os, key.maxKernelRadiusPx);
    hashAppendValue(os, key.customAperturePath);
    hashAppendValue(os, key.customApertureNormalize);
    hashAppendValue(os, key.customApertureInvert);
    hashAppendValue(os, key.roundness);
    hashAppendValue(os, key.rotationDeg);
    hashAppendValue(os, key.centralObstruction);
    hashAppendValue(os, key.vaneThickness);
    hashAppendValue(os, key.diffractionScalePx);
    hashAppendValue(os, key.anisotropyEmphasis);
    hashAppendValue(os, key.phaseDefocus);
    hashAppendValue(os, key.phaseAstigmatism0);
    hashAppendValue(os, key.phaseAstigmatism45);
    hashAppendValue(os, key.phaseComaX);
    hashAppendValue(os, key.phaseComaY);
    hashAppendValue(os, key.phaseSpherical);
    hashAppendValue(os, key.phaseTrefoilX);
    hashAppendValue(os, key.phaseTrefoilY);
    hashAppendValue(os, key.phaseSecondaryAstigmatism0);
    hashAppendValue(os, key.phaseSecondaryAstigmatism45);
    hashAppendValue(os, key.phaseQuadrafoil0);
    hashAppendValue(os, key.phaseQuadrafoil45);
    hashAppendValue(os, key.phaseSecondaryComaX);
    hashAppendValue(os, key.phaseSecondaryComaY);
    hashAppendValue(os, key.pupilDecenterX);
    hashAppendValue(os, key.pupilDecenterY);
    hashAppendValue(os, key.chromaticFocus);
    hashAppendValue(os, key.chromaticSpread);
    return hashStringFnv1a64(os.str());
}

enum class FieldEffectKind : int {
    Full = 0,
    Core = 1,
    Structure = 2,
};

struct FieldZoneBatchPlan {
    std::vector<const LensDiffFieldZoneCache*> zones;
    LensDiffFieldKey fieldKey {};
    bool canonical3x3 = false;
};

struct FieldZoneSpectrumStacks {
    int paddedWidth = 0;
    int paddedHeight = 0;
    int zoneCount = 0;
    int spectralBinCount = 0;
    FieldEffectKind effectKind = FieldEffectKind::Full;
    DeviceBuffer<cufftComplex> spectrumStack;
};

struct StackImageParamsCuda {
    int pixelCount = 0;
    int stackDepth = 0;
};

struct ZonePlaneStackParamsCuda {
    int pixelCount = 0;
    int zoneCount = 0;
    int binCount = 0;
};

struct ZonePlaneAccumulateParamsCuda {
    int pixelCount = 0;
    int zoneCount = 0;
    int binCount = 0;
    int binIndex = 0;
};

struct PupilRasterParamsCuda {
    int size = 0;
    int apertureMode = 0;
    int apodizationMode = 0;
    int bladeCount = 0;
    int vaneCount = 0;
    int customWidth = 0;
    int customHeight = 0;
    float roundness = 0.0f;
    float rotationRad = 0.0f;
    float outerRadius = 0.86f;
    float centralObstruction = 0.0f;
    float vaneThickness = 0.0f;
    float pupilDecenterX = 0.0f;
    float pupilDecenterY = 0.0f;
    float fitHalfWidth = 1.0f;
    float fitHalfHeight = 1.0f;
    float starInnerRadiusRatio = 0.5f;
};

struct PhaseRasterParamsCuda {
    int size = 0;
    int hasPhase = 0;
    float rotationRad = 0.0f;
    float outerRadius = 0.86f;
    float pupilDecenterX = 0.0f;
    float pupilDecenterY = 0.0f;
    float phaseDefocus = 0.0f;
    float phaseAstigmatism0 = 0.0f;
    float phaseAstigmatism45 = 0.0f;
    float phaseComaX = 0.0f;
    float phaseComaY = 0.0f;
    float phaseSpherical = 0.0f;
    float phaseTrefoilX = 0.0f;
    float phaseTrefoilY = 0.0f;
    float phaseSecondaryAstigmatism0 = 0.0f;
    float phaseSecondaryAstigmatism45 = 0.0f;
    float phaseQuadrafoil0 = 0.0f;
    float phaseQuadrafoil45 = 0.0f;
    float phaseSecondaryComaX = 0.0f;
    float phaseSecondaryComaY = 0.0f;
};

LensDiffKernel buildGaussianKernelHost(float radiusPx) {
    const int radius = std::max(1, static_cast<int>(std::ceil(std::max(0.5f, radiusPx))));
    LensDiffKernel kernel {};
    kernel.size = radius * 2 + 1;
    kernel.values.assign(static_cast<std::size_t>(kernel.size) * kernel.size, 0.0f);
    const float sigma = std::max(0.5f, radiusPx * 0.5f);
    float sum = 0.0f;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            const float value = std::exp(-(static_cast<float>(x * x + y * y) / (2.0f * sigma * sigma)));
            kernel.values[static_cast<std::size_t>(y + radius) * kernel.size + static_cast<std::size_t>(x + radius)] = value;
            sum += value;
        }
    }
    if (sum > 0.0f) {
        const float invSum = 1.0f / sum;
        for (float& value : kernel.values) {
            value *= invSum;
        }
    }
    return kernel;
}

int paddedAdaptiveSupportRadiusHost(int estimatedRadius, int maxRadius) {
    const int padding = std::max(6, std::min(24, static_cast<int>(std::ceil(std::max(1, estimatedRadius) * 0.05f))));
    return std::max(4, std::min(maxRadius, estimatedRadius + padding));
}

int nextPowerOfTwo(int value) {
    int n = 1;
    while (n < value) {
        n <<= 1;
    }
    return n;
}

bool checkCuda(cudaError_t status, const char* stage, std::string* error) {
    if (status == cudaSuccess) {
        return true;
    }
    if (error != nullptr) {
        *error = std::string(stage) + ": " + cudaGetErrorString(status);
    }
    return false;
}

bool checkCufft(cufftResult status, const char* stage, std::string* error) {
    if (status == CUFFT_SUCCESS) {
        return true;
    }
    if (error != nullptr) {
        *error = std::string(stage) + ": cufft-error-" + std::to_string(static_cast<int>(status));
    }
    return false;
}

__device__ float saturateDevice(float v) {
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

__device__ float safeLumaDevice(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

__device__ float softShoulderDevice(float v, float shoulder) {
    if (shoulder <= 0.0f) {
        return fmaxf(0.0f, v);
    }
    const float x = fmaxf(0.0f, v);
    return shoulder * (1.0f - expf(-x / shoulder));
}

__device__ float decodeTransferDevice(float x, int transfer) {
    switch (transfer) {
        case 1:
            return x <= 0.02740668f ? x / 10.44426855f : exp2f(x / 0.07329248f - 7.0f) - 0.0075f;
        case 0:
        default:
            return x;
    }
}

__device__ float encodeTransferDevice(float x, int transfer) {
    switch (transfer) {
        case 1:
            return x <= 0.00262409f ? (x * 10.44426855f)
                                    : ((log2f(fmaxf(x, 0.0f) + 0.0075f) + 7.0f) * 0.07329248f);
        case 0:
        default:
            return x;
    }
}

__device__ float atomicMaxFloatDevice(float* address, float value) {
    int* addressAsInt = reinterpret_cast<int*>(address);
    int old = *addressAsInt;
    int assumed = 0;
    while (__int_as_float(old) < value) {
        assumed = old;
        old = atomicCAS(addressAsInt, assumed, __float_as_int(value));
        if (assumed == old) {
            break;
        }
    }
    return __int_as_float(old);
}

__device__ void mulSpectralMatrixDevice(const float* matrix,
                                        int binCount,
                                        const float* bins,
                                        float* outR,
                                        float* outG,
                                        float* outB) {
    if (binCount <= 1) {
        *outR = bins[0];
        *outG = bins[0];
        *outB = bins[0];
        return;
    }
    *outR = 0.0f;
    *outG = 0.0f;
    *outB = 0.0f;
    const int count = min(binCount, kLensDiffMaxSpectralBins);
    for (int i = 0; i < count; ++i) {
        *outR += matrix[i] * bins[i];
        *outG += matrix[kLensDiffMaxSpectralBins + i] * bins[i];
        *outB += matrix[kLensDiffMaxSpectralBins * 2 + i] * bins[i];
    }
}

__global__ void packDecodeKernel(const float* src,
                                 std::size_t srcRowFloats,
                                 int width,
                                 int height,
                                 int transfer,
                                 float* r,
                                 float* g,
                                 float* b,
                                 float* a) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const std::size_t srcIndex = static_cast<std::size_t>(y) * srcRowFloats + static_cast<std::size_t>(x) * 4U;
    const std::size_t index = static_cast<std::size_t>(y) * width + x;
    r[index] = decodeTransferDevice(src[srcIndex + 0], transfer);
    g[index] = decodeTransferDevice(src[srcIndex + 1], transfer);
    b[index] = decodeTransferDevice(src[srcIndex + 2], transfer);
    a[index] = src[srcIndex + 3];
}

__global__ void buildSelectionKernel(const float* r,
                                     const float* g,
                                     const float* b,
                                     int width,
                                     int height,
                                     int extractionMode,
                                     float thresholdStops,
                                     float softnessStops,
                                     float pointEmphasis,
                                     float* mask,
                                     float* driver) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const std::size_t index = static_cast<std::size_t>(y) * width + x;
    const float rv = r[index];
    const float gv = g[index];
    const float bv = b[index];
    const float maxRgb = fmaxf(rv, fmaxf(gv, bv));
    const float signal = extractionMode == 1 ? safeLumaDevice(rv, gv, bv) : maxRgb;

    float gate = 0.0f;
    if (signal > 0.0f) {
        const float stops = log2f(fmaxf(signal, 1e-6f) / kGray18);
        const float edge0 = thresholdStops - softnessStops * 0.5f;
        const float edge1 = thresholdStops + softnessStops * 0.5f;
        const float t = saturateDevice((stops - edge0) / fmaxf(edge1 - edge0, 1e-4f));
        gate = t * t * (3.0f - 2.0f * t);
    }

    const float thresholdLinear = kGray18 * exp2f(thresholdStops);
    const float pointBoost = 1.0f + pointEmphasis * fmaxf(0.0f, maxRgb / fmaxf(thresholdLinear, 1e-4f) - 1.0f);
    mask[index] = saturateDevice(gate * pointBoost);
    driver[index] = signal;
}

__global__ void applyMaskRgbKernel(const float* srcR,
                                   const float* srcG,
                                   const float* srcB,
                                   const float* mask,
                                   int width,
                                   int height,
                                   float scale,
                                   float* outR,
                                   float* outG,
                                   float* outB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const std::size_t index = static_cast<std::size_t>(y) * width + x;
    const float m = mask[index] * scale;
    outR[index] = srcR[index] * m;
    outG[index] = srcG[index] * m;
    outB[index] = srcB[index] * m;
}

__global__ void applyMaskScalarKernel(const float* driver,
                                      const float* mask,
                                      int width,
                                      int height,
                                      float scale,
                                      float* out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const std::size_t index = static_cast<std::size_t>(y) * width + x;
    out[index] = driver[index] * mask[index] * scale;
}

__global__ void padRealToComplexKernel(const float* src,
                                       int width,
                                       int height,
                                       int paddedWidth,
                                       int paddedHeight,
                                       cufftComplex* dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= paddedWidth || y >= paddedHeight) {
        return;
    }
    const std::size_t index = static_cast<std::size_t>(y) * paddedWidth + x;
    float value = 0.0f;
    if (x < width && y < height) {
        value = src[static_cast<std::size_t>(y) * width + x];
    }
    dst[index].x = value;
    dst[index].y = 0.0f;
}

__global__ void padRealWindowToComplexKernel(const float* src,
                                             int srcWidth,
                                             int srcHeight,
                                             int windowX,
                                             int windowY,
                                             int windowWidth,
                                             int windowHeight,
                                             int paddedWidth,
                                             int paddedHeight,
                                             cufftComplex* dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= paddedWidth || y >= paddedHeight) {
        return;
    }
    const std::size_t index = static_cast<std::size_t>(y) * paddedWidth + x;
    float value = 0.0f;
    if (x < windowWidth && y < windowHeight) {
        const int srcX = windowX + x;
        const int srcY = windowY + y;
        if (srcX >= 0 && srcX < srcWidth && srcY >= 0 && srcY < srcHeight) {
            value = src[static_cast<std::size_t>(srcY) * srcWidth + srcX];
        }
    }
    dst[index].x = value;
    dst[index].y = 0.0f;
}

__global__ void scatterKernelToComplexKernel(const float* kernelValues,
                                             int kernelSize,
                                             int paddedWidth,
                                             int paddedHeight,
                                             cufftComplex* dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kernelSize || y >= kernelSize) {
        return;
    }

    const int center = kernelSize / 2;
    const int px = (x - center + paddedWidth) % paddedWidth;
    const int py = (y - center + paddedHeight) % paddedHeight;
    const std::size_t dstIndex = static_cast<std::size_t>(py) * paddedWidth + px;
    const std::size_t srcIndex = static_cast<std::size_t>(y) * kernelSize + x;
    dst[dstIndex].x = kernelValues[srcIndex];
    dst[dstIndex].y = 0.0f;
}

__global__ void embedCenteredComplexPupilKernel(const float* amplitude,
                                                const float* phaseWaves,
                                                int srcSize,
                                                int dstSize,
                                                int offset,
                                                cufftComplex* dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= srcSize || y >= srcSize) {
        return;
    }

    const std::size_t srcIndex = static_cast<std::size_t>(y) * srcSize + x;
    const std::size_t dstIndex = static_cast<std::size_t>(y + offset) * dstSize + (x + offset);
    const float pupilAmplitude = amplitude[srcIndex];
    const float phaseRadians = phaseWaves != nullptr ? phaseWaves[srcIndex] * (2.0f * kPi) : 0.0f;
    dst[dstIndex].x = pupilAmplitude * cosf(phaseRadians);
    dst[dstIndex].y = pupilAmplitude * sinf(phaseRadians);
}

__device__ float sampleSquareBilinearDevice(const float* image, int size, float x, float y) {
    if (x < 0.0f || y < 0.0f || x > static_cast<float>(size - 1) || y > static_cast<float>(size - 1)) {
        return 0.0f;
    }
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = min(x0 + 1, size - 1);
    const int y1 = min(y0 + 1, size - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float v00 = image[static_cast<std::size_t>(y0) * size + x0];
    const float v10 = image[static_cast<std::size_t>(y0) * size + x1];
    const float v01 = image[static_cast<std::size_t>(y1) * size + x0];
    const float v11 = image[static_cast<std::size_t>(y1) * size + x1];
    const float vx0 = v00 + (v10 - v00) * tx;
    const float vx1 = v01 + (v11 - v01) * tx;
    return vx0 + (vx1 - vx0) * ty;
}

__device__ float samplePlaneBilinearDevice(const float* image, int width, int height, float x, float y) {
    if (x < 0.0f || y < 0.0f || x > static_cast<float>(width - 1) || y > static_cast<float>(height - 1)) {
        return 0.0f;
    }
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float v00 = image[static_cast<std::size_t>(y0) * width + x0];
    const float v10 = image[static_cast<std::size_t>(y0) * width + x1];
    const float v01 = image[static_cast<std::size_t>(y1) * width + x0];
    const float v11 = image[static_cast<std::size_t>(y1) * width + x1];
    const float vx0 = v00 + (v10 - v00) * tx;
    const float vx1 = v01 + (v11 - v01) * tx;
    return vx0 + (vx1 - vx0) * ty;
}

__device__ float sampleSquareFilteredDevice(const float* image, int size, float x, float y, float footprint) {
    if (footprint <= 1.0f) {
        return sampleSquareBilinearDevice(image, size, x, y);
    }
    const int taps = max(2, min(6, static_cast<int>(ceilf(footprint))));
    const float step = footprint / static_cast<float>(taps);
    float sum = 0.0f;
    for (int ty = 0; ty < taps; ++ty) {
        const float oy = (static_cast<float>(ty) + 0.5f) * step - footprint * 0.5f;
        for (int tx = 0; tx < taps; ++tx) {
            const float ox = (static_cast<float>(tx) + 0.5f) * step - footprint * 0.5f;
            sum += sampleSquareBilinearDevice(image, size, x + ox, y + oy);
        }
    }
    return sum / static_cast<float>(taps * taps);
}

__global__ void resamplePlaneKernel(const float* src,
                                    int srcWidth,
                                    int srcHeight,
                                    float* dst,
                                    int dstWidth,
                                    int dstHeight) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstWidth || y >= dstHeight) {
        return;
    }
    const float scaleX = static_cast<float>(srcWidth) / static_cast<float>(max(1, dstWidth));
    const float scaleY = static_cast<float>(srcHeight) / static_cast<float>(max(1, dstHeight));
    const float sx = (static_cast<float>(x) + 0.5f) * scaleX - 0.5f;
    const float sy = (static_cast<float>(y) + 0.5f) * scaleY - 0.5f;
    dst[static_cast<std::size_t>(y) * dstWidth + x] = samplePlaneBilinearDevice(src, srcWidth, srcHeight, sx, sy);
}

__device__ float clampUnitDevice(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ float polygonMetricCuda(float nx,
                                   float ny,
                                   int bladeCount,
                                   float roundness,
                                   float rotationRad,
                                   float outerRadius) {
    const float radius = sqrtf(nx * nx + ny * ny);
    const float circleMetric = radius / fmaxf(outerRadius, 1e-5f);
    if (bladeCount < 3) {
        return circleMetric;
    }

    const float angle = atan2f(ny, nx) - rotationRad;
    const float sector = 2.0f * kPi / static_cast<float>(bladeCount);
    float local = fmodf(angle, sector);
    if (local < 0.0f) {
        local += sector;
    }
    local -= sector * 0.5f;
    const float polygonRadius = outerRadius * cosf(kPi / static_cast<float>(bladeCount)) /
                                fmaxf(cosf(local), 1e-4f);
    const float polygonMetricValue = radius / fmaxf(polygonRadius, 1e-5f);
    return circleMetric * roundness + polygonMetricValue * (1.0f - roundness);
}

__device__ bool blockedByVanesCuda(float nx,
                                   float ny,
                                   int vaneCount,
                                   float thickness,
                                   float rotationRad,
                                   float outerRadius) {
    if (vaneCount <= 0 || thickness <= 0.0f) {
        return false;
    }
    const float scaledThickness = thickness * outerRadius;
    const int lineCount = max(1, vaneCount);
    for (int i = 0; i < lineCount; ++i) {
        const float angle = rotationRad + static_cast<float>(i) * kPi / static_cast<float>(lineCount);
        const float cs = cosf(angle);
        const float sn = sinf(angle);
        const float xr = nx * cs + ny * sn;
        const float yr = -nx * sn + ny * cs;
        if (fabsf(yr) <= scaledThickness && fabsf(xr) <= outerRadius) {
            return true;
        }
    }
    return false;
}

__device__ float starMetricCuda(float nx,
                                float ny,
                                int points,
                                float innerRadiusRatio,
                                float rotationRad,
                                float outerRadius) {
    const float radius = sqrtf(nx * nx + ny * ny);
    if (radius <= 1e-6f) {
        return 0.0f;
    }

    const int pointCount = max(3, points);
    const float innerRadius = outerRadius * fminf(fmaxf(innerRadiusRatio, 0.1f), 0.95f);
    const float angle = atan2f(ny, nx) - rotationRad + kPi * 0.5f;
    const float sector = 2.0f * kPi / static_cast<float>(pointCount);
    float local = fmodf(angle, sector);
    if (local < 0.0f) {
        local += sector;
    }
    const float halfSector = sector * 0.5f;
    const float t = local <= halfSector
        ? local / fmaxf(halfSector, 1e-6f)
        : (sector - local) / fmaxf(halfSector, 1e-6f);
    const float boundaryRadius = innerRadius + (outerRadius - innerRadius) * clampUnitDevice(t);
    return radius / fmaxf(boundaryRadius, 1e-5f);
}

__device__ float distancePointToSegmentCuda(float px, float py, float ax, float ay, float bx, float by) {
    const float vx = bx - ax;
    const float vy = by - ay;
    const float len2 = vx * vx + vy * vy;
    if (len2 <= 1e-10f) {
        const float dx = px - ax;
        const float dy = py - ay;
        return sqrtf(dx * dx + dy * dy);
    }

    const float t = fminf(fmaxf(((px - ax) * vx + (py - ay) * vy) / len2, 0.0f), 1.0f);
    const float cx = ax + t * vx;
    const float cy = ay + t * vy;
    const float dx = px - cx;
    const float dy = py - cy;
    return sqrtf(dx * dx + dy * dy);
}

__device__ bool squareGridMaskOpenCuda(float nx,
                                       float ny,
                                       int bladeCount,
                                       float roundness,
                                       float rotationRad,
                                       float outerRadius) {
    const float cs = cosf(-rotationRad);
    const float sn = sinf(-rotationRad);
    const float rx = nx * cs - ny * sn;
    const float ry = nx * sn + ny * cs;
    const float squareHalf = outerRadius * 0.82f;
    if (fabsf(rx) > squareHalf || fabsf(ry) > squareHalf) {
        return false;
    }

    const int divisions = max(3, bladeCount);
    const float pitch = (2.0f * squareHalf) / static_cast<float>(divisions);
    const float clampedRoundness = clampUnitDevice(roundness);
    const float barHalf = pitch * (0.06f + (1.0f - clampedRoundness) * 0.12f);
    const float wrappedX = fmodf(rx + squareHalf, pitch);
    const float wrappedY = fmodf(ry + squareHalf, pitch);
    const float positiveX = wrappedX < 0.0f ? wrappedX + pitch : wrappedX;
    const float positiveY = wrappedY < 0.0f ? wrappedY + pitch : wrappedY;
    const float distX = fminf(positiveX, pitch - positiveX);
    const float distY = fminf(positiveY, pitch - positiveY);
    const float edgeDistX = fabsf(fabsf(rx) - squareHalf);
    const float edgeDistY = fabsf(fabsf(ry) - squareHalf);
    return distX <= barHalf || distY <= barHalf || edgeDistX <= barHalf || edgeDistY <= barHalf;
}

__device__ bool snowflakeMaskOpenCuda(float nx,
                                      float ny,
                                      int bladeCount,
                                      float roundness,
                                      float rotationRad,
                                      float outerRadius) {
    const float radius = sqrtf(nx * nx + ny * ny);
    if (radius > outerRadius) {
        return false;
    }

    const int branchLevels = max(2, min(5, max(2, bladeCount / 2)));
    const float mainLength = outerRadius * 0.92f;
    const float branchAngle = kPi / 5.5f;
    const float baseThickness = outerRadius * (0.045f + (1.0f - clampUnitDevice(roundness)) * 0.05f);
    float minDistance = 1e20f;
    for (int arm = 0; arm < 6; ++arm) {
        const float angle = rotationRad - kPi * 0.5f + static_cast<float>(arm) * (kPi / 3.0f);
        const float cs = cosf(angle);
        const float sn = sinf(angle);
        const float ex = cs * mainLength;
        const float ey = sn * mainLength;
        minDistance = fminf(minDistance, distancePointToSegmentCuda(nx, ny, 0.0f, 0.0f, ex, ey));

        for (int level = 0; level < branchLevels; ++level) {
            const float t = 0.34f + static_cast<float>(level) * (0.44f / max(1, branchLevels - 1));
            const float mx = cs * mainLength * t;
            const float my = sn * mainLength * t;
            const float branchLength = outerRadius * (0.16f + 0.03f * static_cast<float>(level));
            for (int side = -1; side <= 1; side += 2) {
                const float branchTheta = angle + static_cast<float>(side) * branchAngle;
                const float bx = mx + cosf(branchTheta) * branchLength;
                const float by = my + sinf(branchTheta) * branchLength;
                minDistance = fminf(minDistance, distancePointToSegmentCuda(nx, ny, mx, my, bx, by));
            }
        }
    }
    return minDistance <= baseThickness;
}

__device__ bool spiralMaskOpenCuda(float nx,
                                   float ny,
                                   int bladeCount,
                                   float roundness,
                                   float rotationRad,
                                   float outerRadius) {
    const float radius = sqrtf(nx * nx + ny * ny);
    if (radius > outerRadius) {
        return false;
    }
    const float radialNorm = radius / fmaxf(outerRadius, 1e-6f);
    const float angle = atan2f(ny, nx) - rotationRad;
    const float clampedRoundness = clampUnitDevice(roundness);
    const float twist = 4.0f + (1.0f - clampedRoundness) * 8.0f;
    const float opening = 0.18f + clampedRoundness * 0.26f;
    const float phase = static_cast<float>(max(3, bladeCount)) * angle + twist * radialNorm * (2.0f * kPi);
    const float band = 0.5f + 0.5f * cosf(phase);
    return band >= (1.0f - opening);
}

__device__ float apodizationWeightCuda(int mode, float radialNorm) {
    switch (mode) {
        case 1:
            return cosf(fminf(radialNorm, 1.0f) * (kPi * 0.5f));
        case 2:
            return expf(-4.0f * radialNorm * radialNorm);
        case 0:
        default:
            return 1.0f;
    }
}

__device__ float evaluatePhaseWavesCuda(const PhaseRasterParamsCuda& params, float px, float py) {
    const float sx = px - params.pupilDecenterX;
    const float sy = py - params.pupilDecenterY;
    const float r2 = sx * sx + sy * sy;
    if (r2 > 1.0f) {
        return 0.0f;
    }

    const float defocus = kZernikeDefocusNorm * (2.0f * r2 - 1.0f);
    const float astig0 = kZernikeAstigNorm * (sx * sx - sy * sy);
    const float astig45 = kZernikeAstigNorm * (2.0f * sx * sy);
    const float comaX = kZernikeComaNorm * ((3.0f * r2 - 2.0f) * sx);
    const float comaY = kZernikeComaNorm * ((3.0f * r2 - 2.0f) * sy);
    const float spherical = kZernikeSphericalNorm * (6.0f * r2 * r2 - 6.0f * r2 + 1.0f);
    const float trefoilX = kZernikeTrefoilNorm * (sx * sx * sx - 3.0f * sx * sy * sy);
    const float trefoilY = kZernikeTrefoilNorm * (3.0f * sx * sx * sy - sy * sy * sy);
    const float secondaryAstigRadial = 4.0f * r2 - 3.0f;
    const float secondaryAstig0 = kZernikeSecondaryAstigNorm * secondaryAstigRadial * (sx * sx - sy * sy);
    const float secondaryAstig45 = kZernikeSecondaryAstigNorm * secondaryAstigRadial * (2.0f * sx * sy);
    const float quadrafoil0 = kZernikeQuadrafoilNorm * (sx * sx * sx * sx - 6.0f * sx * sx * sy * sy + sy * sy * sy * sy);
    const float quadrafoil45 = kZernikeQuadrafoilNorm * (4.0f * sx * sy * (sx * sx - sy * sy));
    const float secondaryComaRadial = 10.0f * r2 * r2 - 12.0f * r2 + 3.0f;
    const float secondaryComaX = kZernikeSecondaryComaNorm * secondaryComaRadial * sx;
    const float secondaryComaY = kZernikeSecondaryComaNorm * secondaryComaRadial * sy;

    return params.phaseDefocus * defocus +
           params.phaseAstigmatism0 * astig0 +
           params.phaseAstigmatism45 * astig45 +
           params.phaseComaX * comaX +
           params.phaseComaY * comaY +
           params.phaseSpherical * spherical +
           params.phaseTrefoilX * trefoilX +
           params.phaseTrefoilY * trefoilY +
           params.phaseSecondaryAstigmatism0 * secondaryAstig0 +
           params.phaseSecondaryAstigmatism45 * secondaryAstig45 +
           params.phaseQuadrafoil0 * quadrafoil0 +
           params.phaseQuadrafoil45 * quadrafoil45 +
           params.phaseSecondaryComaX * secondaryComaX +
           params.phaseSecondaryComaY * secondaryComaY;
}

__global__ void buildPupilAmplitudeKernel(const float* customImage,
                                          const PupilRasterParamsCuda params,
                                          float* outPupil) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.size || y >= params.size) {
        return;
    }

    const std::size_t index = static_cast<std::size_t>(y) * params.size + x;
    outPupil[index] = 0.0f;

    const float nx = ((static_cast<float>(x) + 0.5f) / static_cast<float>(params.size) - 0.5f) * 2.0f;
    const float ny = ((static_cast<float>(y) + 0.5f) / static_cast<float>(params.size) - 0.5f) * 2.0f;
    const float dx = nx - params.pupilDecenterX;
    const float dy = ny - params.pupilDecenterY;
    const float radius = sqrtf(dx * dx + dy * dy);
    if (radius > params.outerRadius) {
        return;
    }

    if (params.apertureMode == static_cast<int>(LensDiffApertureMode::Custom)) {
        if (customImage == nullptr || params.customWidth <= 0 || params.customHeight <= 0) {
            return;
        }
        const float cs = cosf(-params.rotationRad);
        const float sn = sinf(-params.rotationRad);
        const float rx = dx * cs - dy * sn;
        const float ry = dx * sn + dy * cs;
        const float ux = rx / params.outerRadius;
        const float uy = ry / params.outerRadius;
        if (fabsf(ux) > params.fitHalfWidth || fabsf(uy) > params.fitHalfHeight) {
            return;
        }
        const float sx = ((ux / params.fitHalfWidth) * 0.5f + 0.5f) * static_cast<float>(params.customWidth - 1);
        const float sy = ((uy / params.fitHalfHeight) * 0.5f + 0.5f) * static_cast<float>(params.customHeight - 1);
        const float radialNorm = radius / params.outerRadius;
        const float sample = samplePlaneBilinearDevice(customImage, params.customWidth, params.customHeight, sx, sy);
        outPupil[index] = sample * apodizationWeightCuda(params.apodizationMode, radialNorm);
        return;
    }

    bool insideShape = false;
    const bool useHexagon = params.apertureMode == static_cast<int>(LensDiffApertureMode::Hexagon);
    const bool useStar = params.apertureMode == static_cast<int>(LensDiffApertureMode::Star);
    const bool useSpiral = params.apertureMode == static_cast<int>(LensDiffApertureMode::Spiral);
    const bool useSquareGrid = params.apertureMode == static_cast<int>(LensDiffApertureMode::SquareGrid);
    const bool useSnowflake = params.apertureMode == static_cast<int>(LensDiffApertureMode::Snowflake);

    if (params.apertureMode == static_cast<int>(LensDiffApertureMode::Polygon) || useHexagon) {
        const int sides = useHexagon ? 6 : max(3, params.bladeCount);
        insideShape = polygonMetricCuda(dx, dy, sides, params.roundness, params.rotationRad, params.outerRadius) <= 1.0f;
    } else if (useSquareGrid) {
        insideShape = squareGridMaskOpenCuda(dx, dy, max(3, params.bladeCount), params.roundness, params.rotationRad, params.outerRadius);
    } else if (useSnowflake) {
        insideShape = snowflakeMaskOpenCuda(dx, dy, max(3, params.bladeCount), params.roundness, params.rotationRad, params.outerRadius);
    } else if (useStar) {
        insideShape = starMetricCuda(dx, dy, max(3, params.bladeCount), params.starInnerRadiusRatio, params.rotationRad, params.outerRadius) <= 1.0f;
    } else if (useSpiral) {
        insideShape = spiralMaskOpenCuda(dx, dy, max(3, params.bladeCount), params.roundness, params.rotationRad, params.outerRadius);
    } else {
        const float metric = polygonMetricCuda(dx, dy, max(3, params.bladeCount), params.roundness, params.rotationRad, params.outerRadius);
        insideShape = params.apertureMode == static_cast<int>(LensDiffApertureMode::Circle) ? radius <= params.outerRadius : metric <= 1.0f;
    }

    if (!insideShape || radius < params.centralObstruction) {
        return;
    }
    if (blockedByVanesCuda(dx, dy, params.vaneCount, params.vaneThickness, params.rotationRad, params.outerRadius)) {
        return;
    }

    outPupil[index] = apodizationWeightCuda(params.apodizationMode, radius / params.outerRadius);
}

__global__ void buildPhaseWavesKernel(const PhaseRasterParamsCuda params, float* outPhase) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.size || y >= params.size) {
        return;
    }

    const std::size_t index = static_cast<std::size_t>(y) * params.size + x;
    outPhase[index] = 0.0f;
    if (params.hasPhase == 0) {
        return;
    }

    const float nx = ((static_cast<float>(x) + 0.5f) / static_cast<float>(params.size) - 0.5f) * 2.0f;
    const float ny = ((static_cast<float>(y) + 0.5f) / static_cast<float>(params.size) - 0.5f) * 2.0f;
    const float radius = sqrtf(nx * nx + ny * ny);
    if (radius > params.outerRadius) {
        return;
    }

    const float cs = cosf(-params.rotationRad);
    const float sn = sinf(-params.rotationRad);
    const float rx = nx * cs - ny * sn;
    const float ry = nx * sn + ny * cs;
    const float px = rx / params.outerRadius;
    const float py = ry / params.outerRadius;
    outPhase[index] = evaluatePhaseWavesCuda(params, px, py);
}

__global__ void resampleRawPsfKernel(const float* rawPsf,
                                     int rawSize,
                                     float invScale,
                                     int supportRadius,
                                     float* outKernel) {
    const int kernelSize = supportRadius * 2 + 1;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kernelSize || y >= kernelSize) {
        return;
    }

    const float rawCenter = (rawSize & 1) == 0 ? static_cast<float>(rawSize) * 0.5f
                                                : static_cast<float>(rawSize - 1) * 0.5f;
    const float kernelCenter = static_cast<float>(kernelSize - 1) * 0.5f;
    const float dx = static_cast<float>(x) - kernelCenter;
    const float dy = static_cast<float>(y) - kernelCenter;
    const float sx = rawCenter + dx * invScale;
    const float sy = rawCenter + dy * invScale;

    outKernel[static_cast<std::size_t>(y) * kernelSize + x] = sampleSquareFilteredDevice(rawPsf, rawSize, sx, sy, invScale);
}

__global__ void accumulateRadialProfileKernel(const float* kernel,
                                              int kernelSize,
                                              int radiusMax,
                                              float center,
                                              float* sums,
                                              int* counts) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kernelSize || y >= kernelSize) {
        return;
    }
    const float dx = static_cast<float>(x) - center;
    const float dy = static_cast<float>(y) - center;
    const int r = min(radiusMax, static_cast<int>(roundf(sqrtf(dx * dx + dy * dy))));
    const float value = kernel[static_cast<std::size_t>(y) * kernelSize + x];
    atomicAdd(sums + r, value);
    atomicAdd(counts + r, 1);
}

__global__ void expandRadialMeanKernel(const float* sums,
                                       const int* counts,
                                       int kernelSize,
                                       int radiusMax,
                                       float center,
                                       float* meanKernel) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kernelSize || y >= kernelSize) {
        return;
    }
    const float dx = static_cast<float>(x) - center;
    const float dy = static_cast<float>(y) - center;
    const int r = min(radiusMax, static_cast<int>(roundf(sqrtf(dx * dx + dy * dy))));
    const int count = max(1, counts[r]);
    meanKernel[static_cast<std::size_t>(y) * kernelSize + x] = sums[r] / static_cast<float>(count);
}

__global__ void scaleBufferKernel(float* values, std::size_t count, float scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    values[index] *= scale;
}

__global__ void reshapeKernel(const float* original,
                              const float* meanKernel,
                              std::size_t count,
                              float gain,
                              float* shapedKernel) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float meanValue = meanKernel[index];
    const float residual = original[index] - meanValue;
    shapedKernel[index] = fmaxf(0.0f, meanValue + residual * gain);
}

__global__ void positiveResidualKernelCuda(const float* fullKernel,
                                           const float* meanKernel,
                                           std::size_t count,
                                           float* residualKernel) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    residualKernel[index] = fmaxf(0.0f, fullKernel[index] - meanKernel[index]);
}

__global__ void accumulateSupportStatsKernel(const float* kernel,
                                             int kernelSize,
                                             int radiusMax,
                                             float center,
                                             float* ringEnergy,
                                             float* ringPeak,
                                             float* totalEnergy,
                                             float* globalPeak) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kernelSize || y >= kernelSize) {
        return;
    }
    const float value = kernel[static_cast<std::size_t>(y) * kernelSize + x];
    const float dx = static_cast<float>(x) - center;
    const float dy = static_cast<float>(y) - center;
    const int r = min(radiusMax, static_cast<int>(ceilf(sqrtf(dx * dx + dy * dy))));
    atomicAdd(ringEnergy + r, value);
    atomicMaxFloatDevice(ringPeak + r, value);
    atomicAdd(totalEnergy, value);
    atomicMaxFloatDevice(globalPeak, value);
}

__global__ void cropKernelToRadiusCuda(const float* src,
                                       int srcSize,
                                       int radius,
                                       int dstSize,
                                       float* dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstSize || y >= dstSize) {
        return;
    }
    const int srcCenter = srcSize / 2;
    const int sx = srcCenter - radius + x;
    const int sy = srcCenter - radius + y;
    dst[static_cast<std::size_t>(y) * dstSize + x] = src[static_cast<std::size_t>(sy) * srcSize + sx];
}

__global__ void applySupportBoundaryTaperCuda(float* kernel,
                                              int size,
                                              int supportRadius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= size || y >= size) {
        return;
    }
    const float center = static_cast<float>(size - 1) * 0.5f;
    const float extent = fmaxf(1.0f, static_cast<float>(supportRadius));
    const float fadeWidth = fmaxf(6.0f, fminf(24.0f, extent * 0.04f));
    const float fadeStart = fmaxf(0.0f, extent - fadeWidth);
    const float dx = static_cast<float>(x) - center;
    const float dy = static_cast<float>(y) - center;
    const float radius = sqrtf(dx * dx + dy * dy);
    float weight = 1.0f;
    if (radius >= extent) {
        weight = 0.0f;
    } else {
        const float t = fminf(fmaxf((radius - fadeStart) / fmaxf(fadeWidth, 1e-6f), 0.0f), 1.0f);
        const float s = t * t * (3.0f - 2.0f * t);
        weight = fmaxf(0.0f, cosf(s * (kPi * 0.5f)));
    }
    kernel[static_cast<std::size_t>(y) * size + x] *= weight;
}

__global__ void reduceSumKernel(const float* values,
                                std::size_t count,
                                float* outSum) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    atomicAdd(outSum, values[index]);
}

__global__ void reduceScalarSumKernel(const float* values,
                                      std::size_t count,
                                      float* outSum) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    atomicAdd(outSum, values[index]);
}

__global__ void extractShiftedIntensityKernel(const cufftComplex* spectrum,
                                              int size,
                                              float* shiftedIntensity) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= size || y >= size) {
        return;
    }

    const int sx = (x + size / 2) % size;
    const int sy = (y + size / 2) % size;
    const cufftComplex value = spectrum[static_cast<std::size_t>(sy) * size + sx];
    shiftedIntensity[static_cast<std::size_t>(y) * size + x] = value.x * value.x + value.y * value.y;
}

__global__ void multiplySpectraKernel(const cufftComplex* a,
                                      const cufftComplex* b,
                                      cufftComplex* out,
                                      std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const cufftComplex av = a[index];
    const cufftComplex bv = b[index];
    out[index].x = av.x * bv.x - av.y * bv.y;
    out[index].y = av.x * bv.y + av.y * bv.x;
}

__global__ void multiplyComplexPairsStackKernel(const cufftComplex* a,
                                                const cufftComplex* b,
                                                cufftComplex* out,
                                                std::size_t countPerSlice,
                                                int sliceCount) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t totalCount = countPerSlice * static_cast<std::size_t>(max(0, sliceCount));
    if (index >= totalCount) {
        return;
    }
    const cufftComplex av = a[index];
    const cufftComplex bv = b[index];
    out[index].x = av.x * bv.x - av.y * bv.y;
    out[index].y = av.x * bv.y + av.y * bv.x;
}

__global__ void replicateComplexStackKernel(const cufftComplex* src,
                                            int srcSliceCount,
                                            int dstSliceCount,
                                            std::size_t countPerSlice,
                                            cufftComplex* dst) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t totalCount = countPerSlice * static_cast<std::size_t>(max(0, dstSliceCount));
    if (index >= totalCount) {
        return;
    }
    const int sliceIndex = static_cast<int>(index / countPerSlice);
    const std::size_t sliceOffset = index % countPerSlice;
    const int srcSlice = srcSliceCount > 0 ? sliceIndex % srcSliceCount : 0;
    dst[index] = src[static_cast<std::size_t>(srcSlice) * countPerSlice + sliceOffset];
}

__global__ void extractRealKernel(const cufftComplex* spectrum,
                                  int width,
                                  int height,
                                  int paddedWidth,
                                  float scale,
                                  float* out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const std::size_t outIndex = static_cast<std::size_t>(y) * width + x;
    const std::size_t srcIndex = static_cast<std::size_t>(y) * paddedWidth + x;
    out[outIndex] = fmaxf(0.0f, spectrum[srcIndex].x * scale);
}

__global__ void extractRealStackKernel(const cufftComplex* spectrum,
                                       int width,
                                       int height,
                                       int paddedWidth,
                                       std::size_t countPerSlice,
                                       int sliceCount,
                                       float scale,
                                       float* out) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t pixelCount = static_cast<std::size_t>(width) * height;
    const std::size_t totalCount = pixelCount * static_cast<std::size_t>(max(0, sliceCount));
    if (index >= totalCount) {
        return;
    }
    const int sliceIndex = static_cast<int>(index / pixelCount);
    const std::size_t pixelOffset = index % pixelCount;
    const int x = static_cast<int>(pixelOffset % static_cast<std::size_t>(width));
    const int y = static_cast<int>(pixelOffset / static_cast<std::size_t>(width));
    const std::size_t srcIndex = static_cast<std::size_t>(sliceIndex) * countPerSlice +
                                 static_cast<std::size_t>(y) * paddedWidth + static_cast<std::size_t>(x);
    out[index] = fmaxf(0.0f, spectrum[srcIndex].x * scale);
}

__global__ void packPlaneTripletsToRgbStackKernel(const float* planeStack,
                                                  std::size_t pixelCount,
                                                  int zoneCount,
                                                  float* outR,
                                                  float* outG,
                                                  float* outB) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t totalCount = pixelCount * static_cast<std::size_t>(max(0, zoneCount));
    if (index >= totalCount) {
        return;
    }
    const int zoneIndex = static_cast<int>(index / pixelCount);
    const std::size_t pixelOffset = index % pixelCount;
    const std::size_t stackBase = static_cast<std::size_t>(zoneIndex) * pixelCount * 3U;
    outR[index] = planeStack[stackBase + pixelOffset];
    outG[index] = planeStack[stackBase + pixelCount + pixelOffset];
    outB[index] = planeStack[stackBase + pixelCount * 2U + pixelOffset];
}

__global__ void applyShoulderKernel(float* r,
                                    float* g,
                                    float* b,
                                    std::size_t count,
                                    float shoulder) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    r[index] = softShoulderDevice(r[index], shoulder);
    g[index] = softShoulderDevice(g[index], shoulder);
    b[index] = softShoulderDevice(b[index], shoulder);
}

__global__ void applyShoulderStackKernel(float* r,
                                         float* g,
                                         float* b,
                                         std::size_t pixelCount,
                                         int stackDepth,
                                         float shoulder) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t totalCount = pixelCount * static_cast<std::size_t>(max(0, stackDepth));
    if (index >= totalCount) {
        return;
    }
    r[index] = softShoulderDevice(r[index], shoulder);
    g[index] = softShoulderDevice(g[index], shoulder);
    b[index] = softShoulderDevice(b[index], shoulder);
}

__global__ void combineRgbKernel(const float* aR,
                                 const float* aG,
                                 const float* aB,
                                 const float* bR,
                                 const float* bG,
                                 const float* bB,
                                 std::size_t count,
                                 float aGain,
                                 float bGain,
                                 float* outR,
                                 float* outG,
                                 float* outB) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    outR[index] = aR[index] * aGain + bR[index] * bGain;
    outG[index] = aG[index] * aGain + bG[index] * bGain;
    outB[index] = aB[index] * aGain + bB[index] * bGain;
}

__global__ void combineRgbStackKernel(const float* aR,
                                      const float* aG,
                                      const float* aB,
                                      const float* bR,
                                      const float* bG,
                                      const float* bB,
                                      std::size_t pixelCount,
                                      int stackDepth,
                                      float aGain,
                                      float bGain,
                                      float* outR,
                                      float* outG,
                                      float* outB) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t totalCount = pixelCount * static_cast<std::size_t>(max(0, stackDepth));
    if (index >= totalCount) {
        return;
    }
    outR[index] = aR[index] * aGain + bR[index] * bGain;
    outG[index] = aG[index] * aGain + bG[index] * bGain;
    outB[index] = aB[index] * aGain + bB[index] * bGain;
}

__device__ inline float fieldZoneWeightAtPixel(int localX,
                                               int localY,
                                               int tileOriginX,
                                               int tileOriginY,
                                               int frameX1,
                                               int frameY1,
                                               int frameWidth,
                                               int frameHeight,
                                               int zoneX,
                                               int zoneY,
                                               float gain) {
    const float denomX = static_cast<float>(max(1, frameWidth - 1));
    const float denomY = static_cast<float>(max(1, frameHeight - 1));
    const float px = (static_cast<float>(tileOriginX + localX - frameX1) / denomX) * 2.0f;
    const float py = (static_cast<float>(tileOriginY + localY - frameY1) / denomY) * 2.0f;
    const float wx = fmaxf(0.0f, 1.0f - fabsf(px - static_cast<float>(zoneX)));
    const float wy = fmaxf(0.0f, 1.0f - fabsf(py - static_cast<float>(zoneY)));
    return wx * wy * gain;
}

__global__ void accumulateWeightedRgbKernel(const float* srcR,
                                            const float* srcG,
                                            const float* srcB,
                                            int width,
                                            int height,
                                            int tileOriginX,
                                            int tileOriginY,
                                            int frameX1,
                                            int frameY1,
                                            int frameWidth,
                                            int frameHeight,
                                            int zoneX,
                                            int zoneY,
                                            float gain,
                                            float* dstR,
                                            float* dstG,
                                            float* dstB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float weight = fieldZoneWeightAtPixel(
        x, y, tileOriginX, tileOriginY, frameX1, frameY1, frameWidth, frameHeight, zoneX, zoneY, gain);
    if (weight <= 0.0f) {
        return;
    }

    const std::size_t index = static_cast<std::size_t>(y) * width + x;
    dstR[index] += srcR[index] * weight;
    dstG[index] += srcG[index] * weight;
    dstB[index] += srcB[index] * weight;
}

__global__ void accumulateWeightedRgbLegacyKernel(const float* srcR,
                                                  const float* srcG,
                                                  const float* srcB,
                                                  int width,
                                                  int height,
                                                  int zoneX,
                                                  int zoneY,
                                                  float gain,
                                                  float* dstR,
                                                  float* dstG,
                                                  float* dstB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float denomX = static_cast<float>(max(1, width - 1));
    const float denomY = static_cast<float>(max(1, height - 1));
    const float px = (static_cast<float>(x) / denomX) * 2.0f;
    const float py = (static_cast<float>(y) / denomY) * 2.0f;
    const float wx = fmaxf(0.0f, 1.0f - fabsf(px - static_cast<float>(zoneX)));
    const float wy = fmaxf(0.0f, 1.0f - fabsf(py - static_cast<float>(zoneY)));
    const float weight = wx * wy * gain;
    if (weight <= 0.0f) {
        return;
    }

    const std::size_t index = static_cast<std::size_t>(y) * width + x;
    dstR[index] += srcR[index] * weight;
    dstG[index] += srcG[index] * weight;
    dstB[index] += srcB[index] * weight;
}

__global__ void accumulateWeightedPlaneKernel(const float* srcPlane,
                                              int width,
                                              int height,
                                              int tileOriginX,
                                              int tileOriginY,
                                              int frameX1,
                                              int frameY1,
                                              int frameWidth,
                                              int frameHeight,
                                              int zoneX,
                                              int zoneY,
                                              float gain,
                                              float* dstPlane) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float weight = fieldZoneWeightAtPixel(
        x, y, tileOriginX, tileOriginY, frameX1, frameY1, frameWidth, frameHeight, zoneX, zoneY, gain);
    if (weight <= 0.0f) {
        return;
    }

    const std::size_t index = static_cast<std::size_t>(y) * width + x;
    dstPlane[index] += srcPlane[index] * weight;
}

__global__ void accumulateWeightedPlaneTileKernel(const float* srcPlane,
                                                  int srcWidth,
                                                  int srcHeight,
                                                  int srcOffsetX,
                                                  int srcOffsetY,
                                                  int tileWidth,
                                                  int tileHeight,
                                                  int dstTileX,
                                                  int dstTileY,
                                                  int dstWidth,
                                                  int dstHeight,
                                                  int frameX1,
                                                  int frameY1,
                                                  int frameWidth,
                                                  int frameHeight,
                                                  int zoneX,
                                                  int zoneY,
                                                  float gain,
                                                  float* dstPlane) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= tileWidth || y >= tileHeight) {
        return;
    }

    const int srcX = srcOffsetX + x;
    const int srcY = srcOffsetY + y;
    const int dstX = dstTileX + x;
    const int dstY = dstTileY + y;
    if (srcX < 0 || srcX >= srcWidth || srcY < 0 || srcY >= srcHeight ||
        dstX < 0 || dstX >= dstWidth || dstY < 0 || dstY >= dstHeight) {
        return;
    }

    const float weight = fieldZoneWeightAtPixel(
        x, y, dstTileX, dstTileY, frameX1, frameY1, frameWidth, frameHeight, zoneX, zoneY, gain);
    if (weight <= 0.0f) {
        return;
    }

    const std::size_t srcIndex = static_cast<std::size_t>(srcY) * srcWidth + srcX;
    const std::size_t dstIndex = static_cast<std::size_t>(dstY) * dstWidth + dstX;
    dstPlane[dstIndex] += srcPlane[srcIndex] * weight;
}

__global__ void accumulateWeightedRgbStackKernel(const float* srcR,
                                                 const float* srcG,
                                                 const float* srcB,
                                                 int width,
                                                 int height,
                                                 int tileOriginX,
                                                 int tileOriginY,
                                                 int frameX1,
                                                 int frameY1,
                                                 int frameWidth,
                                                 int frameHeight,
                                                 std::size_t pixelCount,
                                                 int zoneCount,
                                                 float* dstR,
                                                 float* dstG,
                                                 float* dstB) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pixelCount) {
        return;
    }
    const int x = static_cast<int>(index % static_cast<std::size_t>(width));
    const int y = static_cast<int>(index / static_cast<std::size_t>(width));
    float outRv = 0.0f;
    float outGv = 0.0f;
    float outBv = 0.0f;
    for (int zoneIndex = 0; zoneIndex < zoneCount; ++zoneIndex) {
        const int zoneX = zoneIndex % 3;
        const int zoneY = zoneIndex / 3;
        const float weight = fieldZoneWeightAtPixel(
            x, y, tileOriginX, tileOriginY, frameX1, frameY1, frameWidth, frameHeight, zoneX, zoneY, 1.0f);
        if (weight <= 0.0f) {
            continue;
        }
        const std::size_t srcIndex = static_cast<std::size_t>(zoneIndex) * pixelCount + index;
        outRv += srcR[srcIndex] * weight;
        outGv += srcG[srcIndex] * weight;
        outBv += srcB[srcIndex] * weight;
    }
    dstR[index] = outRv;
    dstG[index] = outGv;
    dstB[index] = outBv;
}

__global__ void accumulateWeightedPlanesStackKernel(const float* planeStack,
                                                    std::size_t pixelCount,
                                                    int width,
                                                    int height,
                                                    int tileOriginX,
                                                    int tileOriginY,
                                                    int frameX1,
                                                    int frameY1,
                                                    int frameWidth,
                                                    int frameHeight,
                                                    int zoneCount,
                                                    int binCount,
                                                    int binIndex,
                                                    float* dstPlane) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pixelCount) {
        return;
    }
    const int x = static_cast<int>(index % static_cast<std::size_t>(width));
    const int y = static_cast<int>(index / static_cast<std::size_t>(width));
    float outValue = 0.0f;
    for (int zoneIndex = 0; zoneIndex < zoneCount; ++zoneIndex) {
        const int zoneX = zoneIndex % 3;
        const int zoneY = zoneIndex / 3;
        const float weight = fieldZoneWeightAtPixel(
            x, y, tileOriginX, tileOriginY, frameX1, frameY1, frameWidth, frameHeight, zoneX, zoneY, 1.0f);
        if (weight <= 0.0f) {
            continue;
        }
        const std::size_t srcIndex = (static_cast<std::size_t>(zoneIndex) * binCount + static_cast<std::size_t>(binIndex)) * pixelCount + index;
        outValue += planeStack[srcIndex] * weight;
    }
    dstPlane[index] = outValue;
}

__global__ void applyCreativeFringeKernel(const float* srcR,
                                          const float* srcG,
                                          const float* srcB,
                                          int width,
                                          int height,
                                          float fringeAmount,
                                          float* outR,
                                          float* outG,
                                          float* outB,
                                          float* previewR,
                                          float* previewG,
                                          float* previewB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float cx = static_cast<float>(max(0, width - 1)) * 0.5f;
    const float cy = static_cast<float>(max(0, height - 1)) * 0.5f;
    const float dx = static_cast<float>(x) - cx;
    const float dy = static_cast<float>(y) - cy;
    const float length = sqrtf(dx * dx + dy * dy);
    const float invLength = length > 1e-6f ? 1.0f / length : 0.0f;
    const float shiftX = dx * invLength * fringeAmount;
    const float shiftY = dy * invLength * fringeAmount;

    const std::size_t index = static_cast<std::size_t>(y) * width + x;
    const float red = samplePlaneBilinearDevice(srcR, width, height, static_cast<float>(x) + shiftX, static_cast<float>(y) + shiftY);
    const float green = samplePlaneBilinearDevice(srcG, width, height, static_cast<float>(x), static_cast<float>(y));
    const float blue = samplePlaneBilinearDevice(srcB, width, height, static_cast<float>(x) - shiftX, static_cast<float>(y) - shiftY);

    outR[index] = red;
    outG[index] = green;
    outB[index] = blue;
    if (previewR != nullptr && previewG != nullptr && previewB != nullptr) {
        previewR[index] = fabsf(red - srcR[index]);
        previewG[index] = fabsf(green - srcG[index]);
        previewB[index] = fabsf(blue - srcB[index]);
    }
}

__global__ void lumaReduceKernel(const float* r,
                                 const float* g,
                                 const float* b,
                                 std::size_t count,
                                 float* sum) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    atomicAdd(sum, safeLumaDevice(r[index], g[index], b[index]));
}

__global__ void scaleRgbKernel(float* r,
                               float* g,
                               float* b,
                               std::size_t count,
                               float scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    r[index] *= scale;
    g[index] *= scale;
    b[index] *= scale;
}

__global__ void scaleRgbByScalarKernel(float* r,
                                       float* g,
                                       float* b,
                                       std::size_t count,
                                       const float* scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float value = scale != nullptr ? scale[0] : 1.0f;
    r[index] *= value;
    g[index] *= value;
    b[index] *= value;
}

__global__ void computeScalarScaleKernel(const float* sum,
                                         float epsilon,
                                         float* outScale) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || outScale == nullptr) {
        return;
    }
    const float value = (sum != nullptr) ? sum[0] : 0.0f;
    outScale[0] = value > epsilon ? 1.0f / value : 1.0f;
}

__global__ void computePreserveScaleKernelCuda(const float* inputEnergy,
                                               const float* effectEnergy,
                                               float epsilon,
                                               float* outScale) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || outScale == nullptr) {
        return;
    }
    const float inputValue = (inputEnergy != nullptr) ? inputEnergy[0] : 0.0f;
    const float effectValue = (effectEnergy != nullptr) ? effectEnergy[0] : 0.0f;
    outScale[0] = effectValue > epsilon ? (inputValue / effectValue) : 1.0f;
}

__global__ void scaleBufferByScalarKernel(float* values,
                                          std::size_t count,
                                          const float* scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    values[index] *= (scale != nullptr ? scale[0] : 1.0f);
}

__global__ void mapSpectralKernel(const float* bin0,
                                  const float* bin1,
                                  const float* bin2,
                                  const float* bin3,
                                  const float* bin4,
                                  const float* bin5,
                                  const float* bin6,
                                  const float* bin7,
                                  const float* bin8,
                                  std::size_t count,
                                  SpectralMapConfigGpu config,
                                  float force,
                                  float saturation,
                                  int chromaticAffectsLuma,
                                  float* outR,
                                  float* outG,
                                  float* outB) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    float spectralBins[kLensDiffMaxSpectralBins] = {
        bin0[index],
        bin1[index],
        bin2[index],
        bin3[index],
        bin4[index],
        bin5[index],
        bin6[index],
        bin7[index],
        bin8[index],
    };

    float naturalR = 0.0f;
    float naturalG = 0.0f;
    float naturalB = 0.0f;
    float styleR = 0.0f;
    float styleG = 0.0f;
    float styleB = 0.0f;
    mulSpectralMatrixDevice(config.naturalMatrix, config.binCount, spectralBins, &naturalR, &naturalG, &naturalB);
    mulSpectralMatrixDevice(config.styleMatrix, config.binCount, spectralBins, &styleR, &styleG, &styleB);

    float r = naturalR * (1.0f - force) + styleR * force;
    float g = naturalG * (1.0f - force) + styleG * force;
    float b = naturalB * (1.0f - force) + styleB * force;

    r = fmaxf(0.0f, r);
    g = fmaxf(0.0f, g);
    b = fmaxf(0.0f, b);

    const float gray = safeLumaDevice(r, g, b);
    r = gray + (r - gray) * saturation;
    g = gray + (g - gray) * saturation;
    b = gray + (b - gray) * saturation;

    if (chromaticAffectsLuma == 0) {
        const float targetLuma = safeLumaDevice(naturalR, naturalG, naturalB);
        const float currentLuma = safeLumaDevice(r, g, b);
        if (currentLuma > 1e-6f) {
            const float scale = targetLuma / currentLuma;
            r *= scale;
            g *= scale;
            b *= scale;
        }
    }

    outR[index] = r;
    outG[index] = g;
    outB[index] = b;
}

__global__ void mapSpectralStackKernel(const float* planeStack,
                                       std::size_t pixelCount,
                                       int zoneCount,
                                       int binCount,
                                       SpectralMapConfigGpu config,
                                       float force,
                                       float saturation,
                                       int chromaticAffectsLuma,
                                       float* outR,
                                       float* outG,
                                       float* outB) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t totalCount = pixelCount * static_cast<std::size_t>(max(0, zoneCount));
    if (index >= totalCount) {
        return;
    }

    const int zoneIndex = static_cast<int>(index / pixelCount);
    const std::size_t pixelOffset = index % pixelCount;
    float spectralBins[kLensDiffMaxSpectralBins] = {0.0f};
    const int localBinCount = min(binCount, kLensDiffMaxSpectralBins);
    for (int i = 0; i < localBinCount; ++i) {
        const std::size_t srcIndex = (static_cast<std::size_t>(zoneIndex) * binCount + static_cast<std::size_t>(i)) * pixelCount + pixelOffset;
        spectralBins[i] = planeStack[srcIndex];
    }

    float naturalR = 0.0f;
    float naturalG = 0.0f;
    float naturalB = 0.0f;
    float styleR = 0.0f;
    float styleG = 0.0f;
    float styleB = 0.0f;
    mulSpectralMatrixDevice(config.naturalMatrix, config.binCount, spectralBins, &naturalR, &naturalG, &naturalB);
    mulSpectralMatrixDevice(config.styleMatrix, config.binCount, spectralBins, &styleR, &styleG, &styleB);

    float r = naturalR * (1.0f - force) + styleR * force;
    float g = naturalG * (1.0f - force) + styleG * force;
    float b = naturalB * (1.0f - force) + styleB * force;

    r = fmaxf(0.0f, r);
    g = fmaxf(0.0f, g);
    b = fmaxf(0.0f, b);

    const float gray = safeLumaDevice(r, g, b);
    r = gray + (r - gray) * saturation;
    g = gray + (g - gray) * saturation;
    b = gray + (b - gray) * saturation;

    if (chromaticAffectsLuma == 0) {
        const float targetLuma = safeLumaDevice(naturalR, naturalG, naturalB);
        const float currentLuma = safeLumaDevice(r, g, b);
        if (currentLuma > 1e-6f) {
            const float scale = targetLuma / currentLuma;
            r *= scale;
            g *= scale;
            b *= scale;
        }
    }

    outR[index] = r;
    outG[index] = g;
    outB[index] = b;
}

__global__ void packGrayDebugKernel(const float* gray,
                                    std::size_t count,
                                    float* packed) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float v = gray[index];
    packed[index * 4U + 0U] = v;
    packed[index * 4U + 1U] = v;
    packed[index * 4U + 2U] = v;
    packed[index * 4U + 3U] = 1.0f;
}

__global__ void packRgbDebugKernel(const float* r,
                                   const float* g,
                                   const float* b,
                                   const float* a,
                                   std::size_t count,
                                   float alphaValue,
                                   float* packed) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    packed[index * 4U + 0U] = r[index];
    packed[index * 4U + 1U] = g[index];
    packed[index * 4U + 2U] = b[index];
    packed[index * 4U + 3U] = a != nullptr ? a[index] : alphaValue;
}

__global__ void compositeFinalKernel(const float* srcR,
                                     const float* srcG,
                                     const float* srcB,
                                     const float* srcA,
                                     const float* redistR,
                                     const float* redistG,
                                     const float* redistB,
                                     const float* effectR,
                                     const float* effectG,
                                     const float* effectB,
                                     std::size_t count,
                                     float coreCompensation,
                                     float effectGain,
                                     float maxRedistributedSubtractScale,
                                     int transfer,
                                     float* packed) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const float rawR = fmaxf(0.0f, srcR[index] - coreCompensation * redistR[index] + effectGain * effectR[index]);
    const float rawG = fmaxf(0.0f, srcG[index] - coreCompensation * redistG[index] + effectGain * effectG[index]);
    const float rawB = fmaxf(0.0f, srcB[index] - coreCompensation * redistB[index] + effectGain * effectB[index]);

    const float floorR = fmaxf(0.0f, srcR[index] - redistR[index] * maxRedistributedSubtractScale);
    const float floorG = fmaxf(0.0f, srcG[index] - redistG[index] * maxRedistributedSubtractScale);
    const float floorB = fmaxf(0.0f, srcB[index] - redistB[index] * maxRedistributedSubtractScale);

    const float linearR = fmaxf(rawR, floorR);
    const float linearG = fmaxf(rawG, floorG);
    const float linearB = fmaxf(rawB, floorB);

    packed[index * 4U + 0U] = encodeTransferDevice(linearR, transfer);
    packed[index * 4U + 1U] = encodeTransferDevice(linearG, transfer);
    packed[index * 4U + 2U] = encodeTransferDevice(linearB, transfer);
    packed[index * 4U + 3U] = srcA[index];
}

bool createPlan(cufftHandle* plan,
                int width,
                int height,
                cudaStream_t stream,
                std::string* error) {
    if (!checkCufft(cufftPlan2d(plan, height, width, CUFFT_C2C), "cufftPlan2d", error)) {
        return false;
    }
    if (!checkCufft(cufftSetStream(*plan, stream), "cufftSetStream", error)) {
        cufftDestroy(*plan);
        *plan = 0;
        return false;
    }
    return true;
}

bool acquirePlan(PersistentCudaPlanRepository* repository,
                 int width,
                 int height,
                 int batchCount,
                 cudaStream_t stream,
                 CudaRenderTimingBreakdown* timing,
                 CudaPlanLease* outPlan,
                 std::string* error) {
    if (outPlan == nullptr || batchCount <= 0) {
        if (error) *error = "cuda-invalid-plan-cache-request";
        return false;
    }
    outPlan->release();
    if (repository == nullptr) {
        if (timing != nullptr) {
            ++timing->cufftPlanCacheMisses;
        }
        if (batchCount == 1) {
            return createPlan(&outPlan->standaloneHandle, width, height, stream, error);
        }
        const int n[2] = {height, width};
        if (!checkCufft(cufftPlanMany(&outPlan->standaloneHandle,
                                      2,
                                      const_cast<int*>(n),
                                      nullptr,
                                      1,
                                      width * height,
                                      nullptr,
                                      1,
                                      width * height,
                                      CUFFT_C2C,
                                      batchCount),
                        "cufftPlanMany-standalone",
                        error)) {
            return false;
        }
        if (!checkCufft(cufftSetStream(outPlan->standaloneHandle, stream), "cufftSetStream-standalone-plan-many", error)) {
            cufftDestroy(outPlan->standaloneHandle);
            outPlan->standaloneHandle = 0;
            return false;
        }
        return true;
    }
    const CudaPlanKey key {width, height, batchCount, static_cast<int>(batchCount > 1 ? CudaPlanLayoutKind::Batched2D
                                                                                       : CudaPlanLayoutKind::Single2D)};
    std::lock_guard<std::mutex> lock(repository->mutex);
    auto& pool = repository->entries[key];
    for (auto& entryPtr : pool) {
        if (entryPtr != nullptr && !entryPtr->inUse) {
            entryPtr->inUse = true;
            entryPtr->stamp = ++repository->nextStamp;
            if (timing != nullptr) {
                ++timing->cufftPlanCacheHits;
                timing->maxPlanWorkBytes = std::max(timing->maxPlanWorkBytes, entryPtr->workBytes);
            }
            if (entryPtr->readyEvent != nullptr &&
                !checkCuda(cudaStreamWaitEvent(stream, entryPtr->readyEvent, 0),
                           "cudaStreamWaitEvent-plan-cache",
                           error)) {
                entryPtr->inUse = false;
                return false;
            }
            if (!checkCufft(cufftSetStream(entryPtr->handle, stream), "cufftSetStream-cached", error) ||
                !checkCufft(cufftSetWorkArea(entryPtr->handle, entryPtr->workspace.ptr), "cufftSetWorkArea-cached", error)) {
                entryPtr->inUse = false;
                return false;
            }
            outPlan->repository = repository;
            outPlan->entry = entryPtr.get();
            outPlan->releaseStream = stream;
            return true;
        }
    }

    if (timing != nullptr) {
        ++timing->cufftPlanCacheMisses;
    }

    const int n[2] = {height, width};
    auto entry = std::make_unique<PersistentCudaPlanRepository::PlanEntry>();
    std::size_t workBytes = 0;
    if (!checkCufft(cufftCreate(&entry->handle), "cufftCreate", error) ||
        !checkCufft(cufftSetAutoAllocation(entry->handle, 0), "cufftSetAutoAllocation", error) ||
        !checkCufft(cufftMakePlanMany(entry->handle,
                                      2,
                                      const_cast<int*>(n),
                                      nullptr,
                                      1,
                                      width * height,
                                      nullptr,
                                      1,
                                      width * height,
                                      CUFFT_C2C,
                                      batchCount,
                                      &workBytes),
                    "cufftMakePlanMany",
                    error)) {
        return false;
    }
    if (!entry->workspace.allocate(std::max<std::size_t>(workBytes, 1U))) {
        if (error) *error = "cuda-alloc-cufft-workspace";
        return false;
    }
    if (!checkCuda(cudaEventCreateWithFlags(&entry->readyEvent, cudaEventDisableTiming),
                   "cudaEventCreateWithFlags-plan-cache",
                   error)) {
        return false;
    }
    entry->workBytes = workBytes;
    entry->inUse = true;
    entry->stamp = ++repository->nextStamp;
    if (!checkCufft(cufftSetWorkArea(entry->handle, entry->workspace.ptr), "cufftSetWorkArea", error) ||
        !checkCufft(cufftSetStream(entry->handle, stream), "cufftSetStream-plan-many", error)) {
        return false;
    }
    if (timing != nullptr) {
        timing->maxPlanWorkBytes = std::max(timing->maxPlanWorkBytes, entry->workBytes);
    }
    auto* entryRaw = entry.get();
    pool.push_back(std::move(entry));
    outPlan->repository = repository;
    outPlan->entry = entryRaw;
    outPlan->releaseStream = stream;
    return true;
}

bool LensDiffCudaVkFFTStrictEnabled();
CudaFftPolicy LensDiffCudaFftPolicy();
CudaFftBackendOverride LensDiffCudaSubsystemFftOverride(CudaFftSubsystem subsystem);
const char* cudaFftBackendName(CudaFftBackend backend);
const char* cudaFftSubsystemName(CudaFftSubsystem subsystem);
const char* cudaFftPolicyName(CudaFftPolicy policy);
int cudaVkfftRepositoryTag(CudaFftSubsystem subsystem);
CudaSubsystemFftStats* cudaSubsystemFftStats(CudaRenderTimingBreakdown* timing, CudaFftSubsystem subsystem);
void recordCudaFftBackendUse(CudaRenderTimingBreakdown* timing, CudaFftSubsystem subsystem, CudaFftBackend backend);
void recordCudaFftFallback(CudaRenderTimingBreakdown* timing,
                           CudaFftSubsystem subsystem,
                           CudaFftBackend fromBackend,
                           CudaFftBackend toBackend,
                           const std::string& reason);
void recordCudaFftPolicySelection(CudaRenderTimingBreakdown* timing,
                                  CudaFftSubsystem subsystem,
                                  CudaFftBackend requestedBackend);
CudaFftBackend resolveCudaFftBackendForSubsystem(CudaFftSubsystem subsystem,
                                                 CudaFftBackend globalRequestedBackend,
                                                 CudaFftPolicy policy,
                                                 CudaRenderTimingBreakdown* timing);
bool usePersistentVkfftRepositoryForSubsystem(CudaFftSubsystem subsystem,
                                              CudaFftPolicy policy,
                                              bool persistentPlanRepositoryEnabled);

bool acquireFftPlan(CudaFftSubsystem subsystem,
                    CudaFftBackend requestedBackend,
                    bool vkfftStrict,
                    PersistentCudaPlanRepository* cufftRepository,
                    bool vkfftPersistentRepository,
                    int width,
                    int height,
                    int batchCount,
                    cudaStream_t stream,
                    CudaRenderTimingBreakdown* timing,
                    CudaFftPlanLease* outPlan,
                    std::string* error) {
    if (outPlan == nullptr) {
        if (error) {
            *error = "cuda-null-fft-plan-request";
        }
        return false;
    }
    outPlan->release();
    outPlan->backend = requestedBackend;
    if (requestedBackend == CudaFftBackend::VkFFT) {
        double initMs = 0.0;
        bool cacheHit = false;
        std::string vkfftError;
        if (lensDiffCudaVkFFTAcquirePlan(vkfftPersistentRepository,
                                         cudaVkfftRepositoryTag(subsystem),
                                         width,
                                         height,
                                         batchCount,
                                         stream,
                                         &initMs,
                                         &cacheHit,
                                         &outPlan->vkfftLease,
                                         &vkfftError)) {
            outPlan->backend = CudaFftBackend::VkFFT;
            if (timing != nullptr) {
                timing->vkfftCompileInitMs += initMs;
                if (cacheHit) {
                    ++timing->vkfftPlanCacheHits;
                } else {
                    ++timing->vkfftPlanCacheMisses;
                }
                timing->maxPlanWorkBytes = std::max(timing->maxPlanWorkBytes, outPlan->workBytes());
                if (CudaSubsystemFftStats* stats = cudaSubsystemFftStats(timing, subsystem)) {
                    stats->vkfftCompileInitMs += initMs;
                    if (cacheHit) {
                        ++stats->vkfftPlanCacheHits;
                    } else {
                        ++stats->vkfftPlanCacheMisses;
                    }
                    stats->maxPlanWorkBytes = std::max(stats->maxPlanWorkBytes, outPlan->workBytes());
                }
            }
            recordCudaFftBackendUse(timing, subsystem, CudaFftBackend::VkFFT);
            return true;
        }
        if (vkfftStrict) {
            if (error != nullptr) {
                *error = vkfftError;
            }
            return false;
        }
        recordCudaFftFallback(timing, subsystem, CudaFftBackend::VkFFT, CudaFftBackend::CuFFT, vkfftError);
    }

    if (!acquirePlan(cufftRepository, width, height, batchCount, stream, timing, &outPlan->cufftLease, error)) {
        return false;
    }
    if (timing != nullptr) {
        if (CudaSubsystemFftStats* stats = cudaSubsystemFftStats(timing, subsystem)) {
            if (outPlan->cufftLease.entry != nullptr) {
                ++stats->cufftPlanCacheHits;
                stats->maxPlanWorkBytes = std::max(stats->maxPlanWorkBytes, outPlan->workBytes());
            } else {
                ++stats->cufftPlanCacheMisses;
            }
        }
    }
    outPlan->backend = CudaFftBackend::CuFFT;
    recordCudaFftBackendUse(timing, subsystem, CudaFftBackend::CuFFT);
    return true;
}

bool execFftC2C(CudaFftPlanLease* plan,
                CudaFftSubsystem subsystem,
                DeviceBuffer<cufftComplex>& buffer,
                bool inverse,
                int width,
                int height,
                cudaStream_t stream,
                CudaRenderTimingBreakdown* timing,
                std::string* error) {
    if (plan == nullptr) {
        if (error) {
            *error = "cuda-null-fft-plan";
        }
        return false;
    }
    if (plan->backend == CudaFftBackend::VkFFT) {
        if (lensDiffCudaVkFFTExecC2C(&plan->vkfftLease, buffer.ptr, inverse, error)) {
            return true;
        }
        if (LensDiffCudaVkFFTStrictEnabled()) {
            return false;
        }
        std::string fallbackReason = error != nullptr ? *error : "cuda-vkfft-exec-failed";
        recordCudaFftFallback(timing, subsystem, CudaFftBackend::VkFFT, CudaFftBackend::CuFFT, fallbackReason);
        plan->release();
        if (!createPlan(&plan->cufftLease.standaloneHandle, width, height, stream, error)) {
            return false;
        }
        plan->backend = CudaFftBackend::CuFFT;
        if (error != nullptr) {
            *error = fallbackReason;
        }
    }
    return checkCufft(cufftExecC2C(plan->cufftLease.handle(),
                                   buffer.ptr,
                                   buffer.ptr,
                                   inverse ? CUFFT_INVERSE : CUFFT_FORWARD),
                      inverse ? "cufftExecC2C-inverse-fallback" : "cufftExecC2C-forward-fallback",
                      error);
}

bool LensDiffCudaLegacyPipelineEnabled() {
    const char* value = std::getenv("LENSDIFF_CUDA_LEGACY_PIPELINE");
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" && text != "OFF";
}

bool LensDiffCudaPersistentPlanRepositoryEnabled() {
    const char* value = std::getenv("LENSDIFF_CUDA_PERSISTENT_PLANS");
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" && text != "OFF";
}

CudaFftPolicy LensDiffCudaFftPolicy() {
    const char* value = std::getenv("LENSDIFF_CUDA_FFT_POLICY");
    if (value == nullptr || *value == '\0') {
        return CudaFftPolicy::VkFFTFirst;
    }
    const std::string text(value);
    if (text == "legacy" || text == "LEGACY") {
        return CudaFftPolicy::Legacy;
    }
    if (text == "auto" || text == "AUTO") {
        return CudaFftPolicy::Auto;
    }
    return CudaFftPolicy::VkFFTFirst;
}

CudaFftBackendOverride parseCudaFftBackendOverrideEnv(const char* envName) {
    const char* value = std::getenv(envName);
    if (value == nullptr || *value == '\0') {
        return CudaFftBackendOverride::Default;
    }
    const std::string text(value);
    if (text == "cufft" || text == "CUFFT" || text == "cuFFT" || text == "CuFFT") {
        return CudaFftBackendOverride::CuFFT;
    }
    if (text == "vkfft" || text == "VKFFT" || text == "VkFFT") {
        return CudaFftBackendOverride::VkFFT;
    }
    return CudaFftBackendOverride::Default;
}

CudaFftBackendOverride LensDiffCudaSubsystemFftOverride(CudaFftSubsystem subsystem) {
    switch (subsystem) {
        case CudaFftSubsystem::PsfBank:
            return parseCudaFftBackendOverrideEnv("LENSDIFF_CUDA_PSF_FFT_BACKEND");
        case CudaFftSubsystem::FieldRender:
            return parseCudaFftBackendOverrideEnv("LENSDIFF_CUDA_FIELD_FFT_BACKEND");
        case CudaFftSubsystem::GlobalRender:
        default:
            return parseCudaFftBackendOverrideEnv("LENSDIFF_CUDA_GLOBAL_FFT_BACKEND");
    }
}

CudaFftBackend LensDiffCudaRequestedFftBackend() {
    const char* value = std::getenv("LENSDIFF_CUDA_FFT_BACKEND");
    if (value == nullptr || *value == '\0') {
        return CudaFftBackend::VkFFT;
    }
    const std::string text(value);
    if (text == "cufft" || text == "CUFFT" || text == "cuFFT" || text == "CuFFT") {
        return CudaFftBackend::CuFFT;
    }
    return CudaFftBackend::VkFFT;
}

bool LensDiffCudaVkFFTStrictEnabled() {
    const char* value = std::getenv("LENSDIFF_CUDA_VKFFT_STRICT");
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" && text != "OFF";
}

const char* cudaFftBackendName(CudaFftBackend backend) {
    return backend == CudaFftBackend::VkFFT ? "vkfft" : "cufft";
}

const char* cudaFftSubsystemName(CudaFftSubsystem subsystem) {
    switch (subsystem) {
        case CudaFftSubsystem::PsfBank: return "psf";
        case CudaFftSubsystem::FieldRender: return "field";
        case CudaFftSubsystem::GlobalRender:
        default: return "global";
    }
}

const char* cudaFftPolicyName(CudaFftPolicy policy) {
    switch (policy) {
        case CudaFftPolicy::Legacy: return "legacy";
        case CudaFftPolicy::Auto: return "auto";
        case CudaFftPolicy::VkFFTFirst:
        default: return "vkfft-first";
    }
}

int cudaVkfftRepositoryTag(CudaFftSubsystem subsystem) {
    switch (subsystem) {
        case CudaFftSubsystem::PsfBank: return 1;
        case CudaFftSubsystem::FieldRender: return 2;
        case CudaFftSubsystem::GlobalRender:
        default: return 0;
    }
}

std::string cudaFftCacheNamespace(CudaFftSubsystem subsystem, CudaFftBackend backend) {
    return std::string(cudaFftSubsystemName(subsystem)) + ":" + cudaFftBackendName(backend);
}

CudaSubsystemFftStats* cudaSubsystemFftStats(CudaRenderTimingBreakdown* timing, CudaFftSubsystem subsystem) {
    if (timing == nullptr) {
        return nullptr;
    }
    switch (subsystem) {
        case CudaFftSubsystem::PsfBank: return &timing->psfFft;
        case CudaFftSubsystem::FieldRender: return &timing->fieldFft;
        case CudaFftSubsystem::GlobalRender:
        default: return &timing->globalFft;
    }
}

void recordCudaFftPolicySelection(CudaRenderTimingBreakdown* timing,
                                  CudaFftSubsystem subsystem,
                                  CudaFftBackend requestedBackend) {
    if (CudaSubsystemFftStats* stats = cudaSubsystemFftStats(timing, subsystem)) {
        stats->requested = cudaFftBackendName(requestedBackend);
    }
}

CudaFftBackend resolveCudaFftBackendForSubsystem(CudaFftSubsystem subsystem,
                                                 CudaFftBackend globalRequestedBackend,
                                                 CudaFftPolicy policy,
                                                 CudaRenderTimingBreakdown* timing) {
    const CudaFftBackendOverride backendOverride = LensDiffCudaSubsystemFftOverride(subsystem);
    if (backendOverride == CudaFftBackendOverride::CuFFT) {
        recordCudaFftPolicySelection(timing, subsystem, CudaFftBackend::CuFFT);
        return CudaFftBackend::CuFFT;
    }
    if (backendOverride == CudaFftBackendOverride::VkFFT) {
        recordCudaFftPolicySelection(timing, subsystem, CudaFftBackend::VkFFT);
        return CudaFftBackend::VkFFT;
    }

    CudaFftBackend resolved = globalRequestedBackend;
    if (policy == CudaFftPolicy::Legacy) {
        resolved = CudaFftBackend::CuFFT;
    }
    recordCudaFftPolicySelection(timing, subsystem, resolved);
    return resolved;
}

bool usePersistentVkfftRepositoryForSubsystem(CudaFftSubsystem subsystem,
                                              CudaFftPolicy policy,
                                              bool persistentPlanRepositoryEnabled) {
    if (policy == CudaFftPolicy::Legacy) {
        return persistentPlanRepositoryEnabled;
    }
    if (subsystem == CudaFftSubsystem::PsfBank || subsystem == CudaFftSubsystem::FieldRender) {
        return true;
    }
    return persistentPlanRepositoryEnabled || policy == CudaFftPolicy::VkFFTFirst || policy == CudaFftPolicy::Auto;
}

void recordCudaFftBackendUse(CudaRenderTimingBreakdown* timing, CudaFftSubsystem subsystem, CudaFftBackend backend) {
    if (timing == nullptr) {
        return;
    }
    const std::string name(cudaFftBackendName(backend));
    if (timing->fftEffective.empty()) {
        timing->fftEffective = name;
    } else if (timing->fftEffective != name && timing->fftEffective != "mixed") {
        timing->fftEffective = "mixed";
    }
    if (CudaSubsystemFftStats* stats = cudaSubsystemFftStats(timing, subsystem)) {
        if (stats->effective == "none") {
            stats->effective = name;
        } else if (stats->effective != name && stats->effective != "mixed") {
            stats->effective = "mixed";
        }
    }
}

void recordCudaFftFallback(CudaRenderTimingBreakdown* timing,
                           CudaFftSubsystem subsystem,
                           CudaFftBackend fromBackend,
                           CudaFftBackend toBackend,
                           const std::string& reason) {
    if (timing == nullptr) {
        return;
    }
    recordCudaFftBackendUse(timing, subsystem, toBackend);
    if (!timing->fftFallbackNote.empty()) {
        timing->fftFallbackNote += ";";
    }
    timing->fftFallbackNote += std::string(cudaFftBackendName(fromBackend)) +
                               "->" + cudaFftBackendName(toBackend) +
                               ":" + reason;
    if (CudaSubsystemFftStats* stats = cudaSubsystemFftStats(timing, subsystem)) {
        if (!stats->fallback.empty()) {
            stats->fallback += ";";
        }
        stats->fallback += std::string(cudaFftBackendName(fromBackend)) + "->" +
                           cudaFftBackendName(toBackend) + ":" + reason;
    }
}

FieldZoneBatchPlan buildFieldZoneBatchPlan(const LensDiffPsfBankCache& cache) {
    FieldZoneBatchPlan plan {};
    plan.fieldKey = cache.fieldKey;
    plan.canonical3x3 = cache.fieldGridSize == 3 && cache.fieldZones.size() == 9U;
    if (!plan.canonical3x3) {
        return plan;
    }
    plan.zones.reserve(cache.fieldZones.size());
    for (int zoneY = 0; zoneY < 3; ++zoneY) {
        for (int zoneX = 0; zoneX < 3; ++zoneX) {
            auto it = std::find_if(cache.fieldZones.begin(),
                                   cache.fieldZones.end(),
                                   [&](const LensDiffFieldZoneCache& zone) {
                                       return zone.zoneX == zoneX && zone.zoneY == zoneY;
                                   });
            if (it == cache.fieldZones.end()) {
                plan.zones.clear();
                plan.canonical3x3 = false;
                return plan;
            }
            plan.zones.push_back(&(*it));
        }
    }
    return plan;
}

bool reduceFloatToScalarGpu(const DeviceBuffer<float>& buffer,
                            std::size_t count,
                            DeviceBuffer<float>* scratch,
                            cudaStream_t stream,
                            std::string* error) {
    if (scratch == nullptr) {
        if (error) *error = "cuda-null-reduce-scratch";
        return false;
    }
    if (!scratch->allocate(1)) {
        if (error) *error = "cuda-alloc-reduce-scratch";
        return false;
    }
    if (!checkCuda(cudaMemsetAsync(scratch->ptr, 0, sizeof(float), stream), "cudaMemsetAsync-reduce-scalar", error)) {
        return false;
    }
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    reduceScalarSumKernel<<<grid, block, 0, stream>>>(buffer.ptr, count, scratch->ptr);
    return checkCuda(cudaGetLastError(), "reduceScalarSumKernel", error);
}

bool normalizeDeviceBufferGpu(DeviceBuffer<float>& buffer,
                              std::size_t count,
                              DeviceBuffer<float>* sumScratch,
                              DeviceBuffer<float>* scaleScratch,
                              cudaStream_t stream,
                              std::string* error) {
    if (count == 0) {
        return true;
    }
    if (sumScratch == nullptr || scaleScratch == nullptr) {
        if (error) *error = "cuda-null-normalize-scratch";
        return false;
    }
    if (!reduceFloatToScalarGpu(buffer, count, sumScratch, stream, error)) {
        return false;
    }
    if (!scaleScratch->allocate(1)) {
        if (error) *error = "cuda-alloc-normalize-scale";
        return false;
    }
    computeScalarScaleKernel<<<1, 1, 0, stream>>>(sumScratch->ptr, 1e-6f, scaleScratch->ptr);
    if (!checkCuda(cudaGetLastError(), "computeScalarScaleKernel", error)) {
        return false;
    }
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    scaleBufferByScalarKernel<<<grid, block, 0, stream>>>(buffer.ptr, count, scaleScratch->ptr);
    return checkCuda(cudaGetLastError(), "scaleBufferByScalarKernel", error);
}

bool makeImageSpectrum(const float* src,
                       CudaFftSubsystem subsystem,
                       int width,
                       int height,
                       int paddedWidth,
                       int paddedHeight,
                       CudaFftPlanLease* plan,
                       cudaStream_t stream,
                       CudaRenderTimingBreakdown* timing,
                       DeviceBuffer<cufftComplex>& out,
                       std::string* error) {
    const std::size_t count = static_cast<std::size_t>(paddedWidth) * paddedHeight;
    if (!out.allocate(count)) {
        if (error) *error = "cuda-alloc-image-spectrum";
        return false;
    }

    dim3 block(16, 16);
    dim3 grid((paddedWidth + block.x - 1) / block.x, (paddedHeight + block.y - 1) / block.y);
    padRealToComplexKernel<<<grid, block, 0, stream>>>(src, width, height, paddedWidth, paddedHeight, out.ptr);
    if (!checkCuda(cudaGetLastError(), "padRealToComplexKernel", error)) {
        return false;
    }

    return execFftC2C(plan, subsystem, out, false, paddedWidth, paddedHeight, stream, timing, error);
}

bool makeImageSpectrumWindow(const float* src,
                             int srcWidth,
                             int srcHeight,
                             int windowX,
                             int windowY,
                             int windowWidth,
                             int windowHeight,
                             int paddedWidth,
                             int paddedHeight,
                             CudaFftPlanLease* plan,
                             cudaStream_t stream,
                             CudaRenderTimingBreakdown* timing,
                             DeviceBuffer<cufftComplex>& out,
                             std::string* error) {
    const std::size_t count = static_cast<std::size_t>(paddedWidth) * paddedHeight;
    if (!out.allocate(count)) {
        if (error) *error = "cuda-alloc-image-spectrum-window";
        return false;
    }

    dim3 block(16, 16);
    dim3 grid((paddedWidth + block.x - 1) / block.x, (paddedHeight + block.y - 1) / block.y);
    padRealWindowToComplexKernel<<<grid, block, 0, stream>>>(
        src,
        srcWidth,
        srcHeight,
        windowX,
        windowY,
        windowWidth,
        windowHeight,
        paddedWidth,
        paddedHeight,
        out.ptr);
    if (!checkCuda(cudaGetLastError(), "padRealWindowToComplexKernel", error)) {
        return false;
    }

    return execFftC2C(plan, CudaFftSubsystem::GlobalRender, out, false, paddedWidth, paddedHeight, stream, timing, error);
}

bool makeKernelSpectrum(const LensDiffKernel& kernel,
                        CudaFftSubsystem subsystem,
                        int paddedWidth,
                        int paddedHeight,
                        CudaFftPlanLease* plan,
                        cudaStream_t stream,
                        CudaRenderTimingBreakdown* timing,
                        DeviceBuffer<float>& deviceKernel,
                        DeviceBuffer<cufftComplex>& out,
                        std::string* error) {
    const std::size_t kernelCount = kernel.values.size();
    const std::size_t paddedCount = static_cast<std::size_t>(paddedWidth) * paddedHeight;
    if (!deviceKernel.allocate(kernelCount) || !out.allocate(paddedCount)) {
        if (error) *error = "cuda-alloc-kernel-spectrum";
        return false;
    }

    if (!checkCuda(cudaMemcpyAsync(deviceKernel.ptr,
                                   kernel.values.data(),
                                   kernelCount * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "cudaMemcpyAsync-kernel-values",
                   error)) {
        return false;
    }

    if (!checkCuda(cudaMemsetAsync(out.ptr, 0, paddedCount * sizeof(cufftComplex), stream),
                   "cudaMemsetAsync-kernel-spectrum",
                   error)) {
        return false;
    }

    dim3 block(16, 16);
    dim3 grid((kernel.size + block.x - 1) / block.x, (kernel.size + block.y - 1) / block.y);
    scatterKernelToComplexKernel<<<grid, block, 0, stream>>>(deviceKernel.ptr, kernel.size, paddedWidth, paddedHeight, out.ptr);
    if (!checkCuda(cudaGetLastError(), "scatterKernelToComplexKernel", error)) {
        return false;
    }

    return execFftC2C(plan, subsystem, out, false, paddedWidth, paddedHeight, stream, timing, error);
}

bool convolveSpectrumToPlane(const DeviceBuffer<cufftComplex>& imageSpectrum,
                             const DeviceBuffer<cufftComplex>& kernelSpectrum,
                             CudaFftSubsystem subsystem,
                             int width,
                             int height,
                             int paddedWidth,
                             int paddedHeight,
                             CudaFftPlanLease* plan,
                             cudaStream_t stream,
                             CudaRenderTimingBreakdown* timing,
                             DeviceBuffer<cufftComplex>& tempSpectrum,
                             DeviceBuffer<float>& outPlane,
                             std::string* error) {
    const std::size_t paddedCount = static_cast<std::size_t>(paddedWidth) * paddedHeight;
    const std::size_t planeCount = static_cast<std::size_t>(width) * height;
    if (!tempSpectrum.allocate(paddedCount) || !outPlane.allocate(planeCount)) {
        if (error) *error = "cuda-alloc-convolution-temp";
        return false;
    }

    const int multiplyBlock = 256;
    const int multiplyGrid = static_cast<int>((paddedCount + multiplyBlock - 1) / multiplyBlock);
    multiplySpectraKernel<<<multiplyGrid, multiplyBlock, 0, stream>>>(imageSpectrum.ptr, kernelSpectrum.ptr, tempSpectrum.ptr, paddedCount);
    if (!checkCuda(cudaGetLastError(), "multiplySpectraKernel", error)) {
        return false;
    }

    if (!execFftC2C(plan, subsystem, tempSpectrum, true, paddedWidth, paddedHeight, stream, timing, error)) {
        return false;
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    const float scale = 1.0f / static_cast<float>(paddedWidth * paddedHeight);
    extractRealKernel<<<grid, block, 0, stream>>>(tempSpectrum.ptr, width, height, paddedWidth, scale, outPlane.ptr);
    return checkCuda(cudaGetLastError(), "extractRealKernel", error);
}

bool convolveSpectrumStackToPlaneStack(const DeviceBuffer<cufftComplex>& imageSpectrumStack,
                                       const DeviceBuffer<cufftComplex>& kernelSpectrumStack,
                                       CudaFftSubsystem subsystem,
                                       int width,
                                       int height,
                                       int paddedWidth,
                                       int paddedHeight,
                                       int batchCount,
                                       CudaFftPlanLease* plan,
                                       cudaStream_t stream,
                                       DeviceBuffer<cufftComplex>& tempSpectrum,
                                       DeviceBuffer<float>& outPlaneStack,
                                       CudaRenderTimingBreakdown* timing,
                                       std::string* error) {
    const std::size_t paddedCount = static_cast<std::size_t>(paddedWidth) * paddedHeight;
    const std::size_t planeCount = static_cast<std::size_t>(width) * height;
    const std::size_t stackSpectrumCount = paddedCount * static_cast<std::size_t>(batchCount);
    const std::size_t stackPlaneCount = planeCount * static_cast<std::size_t>(batchCount);
    if (!tempSpectrum.allocate(stackSpectrumCount) || !outPlaneStack.allocate(stackPlaneCount)) {
        if (error) *error = "cuda-alloc-convolution-stack-temp";
        return false;
    }

    const int multiplyBlock = 256;
    const int multiplyGrid = static_cast<int>((stackSpectrumCount + multiplyBlock - 1) / multiplyBlock);
    multiplyComplexPairsStackKernel<<<multiplyGrid, multiplyBlock, 0, stream>>>(imageSpectrumStack.ptr,
                                                                                kernelSpectrumStack.ptr,
                                                                                tempSpectrum.ptr,
                                                                                paddedCount,
                                                                                batchCount);
    if (!checkCuda(cudaGetLastError(), "multiplyComplexPairsStackKernel", error) ||
        !execFftC2C(plan, subsystem, tempSpectrum, true, paddedWidth, paddedHeight, stream, timing, error)) {
        return false;
    }

    const int block = 256;
    const int grid = static_cast<int>((stackPlaneCount + block - 1) / block);
    const float scale = 1.0f / static_cast<float>(paddedWidth * paddedHeight);
    extractRealStackKernel<<<grid, block, 0, stream>>>(tempSpectrum.ptr,
                                                       width,
                                                       height,
                                                       paddedWidth,
                                                       paddedCount,
                                                       batchCount,
                                                       scale,
                                                       outPlaneStack.ptr);
    return checkCuda(cudaGetLastError(), "extractRealStackKernel", error);
}

bool downloadFloatBuffer(const DeviceBuffer<float>& buffer,
                         std::size_t count,
                         cudaStream_t stream,
                         std::vector<float>* out,
                         const char* stage,
                         std::string* error) {
    if (out == nullptr) {
        if (error) *error = "cuda-null-download-target";
        return false;
    }
    out->assign(count, 0.0f);
    if (count == 0) {
        return true;
    }
    if (!checkCuda(cudaMemcpyAsync(out->data(), buffer.ptr, count * sizeof(float), cudaMemcpyDeviceToHost, stream), stage, error)) {
        return false;
    }
    return checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize-download-float-buffer", error);
}

bool buildPupilAmplitudeOnCuda(const LensDiffParams& params,
                               int pupilSize,
                               cudaStream_t stream,
                               DeviceBuffer<float>* devicePupil,
                               std::string* error) {
    if (devicePupil == nullptr) {
        if (error) *error = "cuda-null-device-pupil";
        return false;
    }
    const std::size_t pupilCount = static_cast<std::size_t>(pupilSize) * pupilSize;
    if (!devicePupil->allocate(pupilCount)) {
        if (error) *error = "cuda-alloc-device-pupil";
        return false;
    }

    DeviceBuffer<float> deviceCustomImage;
    const float* customImagePtr = nullptr;
    PupilRasterParamsCuda rasterParams {};
    rasterParams.size = pupilSize;
    rasterParams.apertureMode = static_cast<int>(params.apertureMode);
    rasterParams.apodizationMode = static_cast<int>(params.apodizationMode);
    rasterParams.bladeCount = params.bladeCount;
    rasterParams.vaneCount = params.vaneCount;
    rasterParams.roundness = static_cast<float>(params.roundness);
    rasterParams.rotationRad = static_cast<float>(params.rotationDeg * kPi / 180.0);
    rasterParams.outerRadius = 0.86f;
    rasterParams.centralObstruction = static_cast<float>(fmax(0.0, fmin(0.95, params.centralObstruction))) * rasterParams.outerRadius;
    rasterParams.vaneThickness = static_cast<float>(fmax(0.0, params.vaneThickness));
    rasterParams.pupilDecenterX = static_cast<float>(params.pupilDecenterX);
    rasterParams.pupilDecenterY = static_cast<float>(params.pupilDecenterY);
    rasterParams.starInnerRadiusRatio = 0.18f + rasterParams.roundness * 0.62f;

    if (params.apertureMode == LensDiffApertureMode::Custom) {
        LensDiffApertureImage image;
        std::string imageError;
        if (LoadLensDiffPreparedApertureImage(params.customAperturePath,
                                              params.customApertureNormalize,
                                              params.customApertureInvert,
                                              &image,
                                              &imageError) &&
            image.width > 0 &&
            image.height > 0 &&
            !image.values.empty()) {
            rasterParams.customWidth = image.width;
            rasterParams.customHeight = image.height;
            const float imageAspect = static_cast<float>(image.width) / static_cast<float>(max(1, image.height));
            rasterParams.fitHalfWidth = imageAspect >= 1.0f ? 1.0f : imageAspect;
            rasterParams.fitHalfHeight = imageAspect >= 1.0f ? 1.0f / imageAspect : 1.0f;
            if (!deviceCustomImage.allocate(image.values.size())) {
                if (error) *error = "cuda-alloc-custom-aperture-image";
                return false;
            }
            if (!checkCuda(cudaMemcpyAsync(deviceCustomImage.ptr,
                                           image.values.data(),
                                           image.values.size() * sizeof(float),
                                           cudaMemcpyHostToDevice,
                                           stream),
                           "cudaMemcpyAsync-custom-aperture-image",
                           error)) {
                return false;
            }
            customImagePtr = deviceCustomImage.ptr;
        } else {
            rasterParams.customWidth = 0;
            rasterParams.customHeight = 0;
            rasterParams.fitHalfWidth = 1.0f;
            rasterParams.fitHalfHeight = 1.0f;
        }
    }

    dim3 block(16, 16);
    dim3 grid((pupilSize + block.x - 1) / block.x, (pupilSize + block.y - 1) / block.y);
    buildPupilAmplitudeKernel<<<grid, block, 0, stream>>>(customImagePtr, rasterParams, devicePupil->ptr);
    return checkCuda(cudaGetLastError(), "buildPupilAmplitudeKernel", error);
}

bool buildPhaseWavesOnCuda(const LensDiffParams& params,
                           int pupilSize,
                           cudaStream_t stream,
                           DeviceBuffer<float>* devicePhase,
                           bool* hasPhase,
                           std::string* error) {
    if (devicePhase == nullptr) {
        if (error) *error = "cuda-null-device-phase";
        return false;
    }
    const std::size_t pupilCount = static_cast<std::size_t>(pupilSize) * pupilSize;
    const bool usePhase = HasLensDiffNonFlatPhase(params);
    if (hasPhase != nullptr) {
        *hasPhase = usePhase;
    }
    if (!devicePhase->allocate(pupilCount)) {
        if (error) *error = "cuda-alloc-device-phase";
        return false;
    }

    PhaseRasterParamsCuda rasterParams {};
    rasterParams.size = pupilSize;
    rasterParams.hasPhase = usePhase ? 1 : 0;
    rasterParams.rotationRad = static_cast<float>(params.rotationDeg * kPi / 180.0);
    rasterParams.outerRadius = 0.86f;
    rasterParams.pupilDecenterX = static_cast<float>(params.pupilDecenterX);
    rasterParams.pupilDecenterY = static_cast<float>(params.pupilDecenterY);
    rasterParams.phaseDefocus = static_cast<float>(params.phaseDefocus);
    rasterParams.phaseAstigmatism0 = static_cast<float>(params.phaseAstigmatism0);
    rasterParams.phaseAstigmatism45 = static_cast<float>(params.phaseAstigmatism45);
    rasterParams.phaseComaX = static_cast<float>(params.phaseComaX);
    rasterParams.phaseComaY = static_cast<float>(params.phaseComaY);
    rasterParams.phaseSpherical = static_cast<float>(params.phaseSpherical);
    rasterParams.phaseTrefoilX = static_cast<float>(params.phaseTrefoilX);
    rasterParams.phaseTrefoilY = static_cast<float>(params.phaseTrefoilY);
    rasterParams.phaseSecondaryAstigmatism0 = static_cast<float>(params.phaseSecondaryAstigmatism0);
    rasterParams.phaseSecondaryAstigmatism45 = static_cast<float>(params.phaseSecondaryAstigmatism45);
    rasterParams.phaseQuadrafoil0 = static_cast<float>(params.phaseQuadrafoil0);
    rasterParams.phaseQuadrafoil45 = static_cast<float>(params.phaseQuadrafoil45);
    rasterParams.phaseSecondaryComaX = static_cast<float>(params.phaseSecondaryComaX);
    rasterParams.phaseSecondaryComaY = static_cast<float>(params.phaseSecondaryComaY);

    dim3 block(16, 16);
    dim3 grid((pupilSize + block.x - 1) / block.x, (pupilSize + block.y - 1) / block.y);
    buildPhaseWavesKernel<<<grid, block, 0, stream>>>(rasterParams, devicePhase->ptr);
    return checkCuda(cudaGetLastError(), "buildPhaseWavesKernel", error);
}

bool buildShiftedRawPsfOnCuda(const DeviceBuffer<float>& devicePupil,
                              const DeviceBuffer<float>* devicePhase,
                              int pupilSize,
                              int rawPsfSize,
                              CudaFftPolicy fftPolicy,
                              CudaFftBackend requestedFftBackend,
                              bool vkfftStrict,
                              cudaStream_t stream,
                              CudaPsfBuildContext* buildContext,
                              DeviceBuffer<float>* deviceShiftedRawPsf,
                              std::vector<float>* shiftedRawPsfDisplay,
                              CudaRenderTimingBreakdown* timing,
                              std::string* error) {
    if (deviceShiftedRawPsf == nullptr || buildContext == nullptr) {
        if (error) *error = "missing-shifted-raw-psf-device-output";
        return false;
    }

    const std::size_t rawCount = static_cast<std::size_t>(rawPsfSize) * rawPsfSize;
    const bool usePhase = devicePhase != nullptr && devicePhase->ptr != nullptr;

    if (!buildContext->shiftedIntensity.allocate(rawCount) ||
        !deviceShiftedRawPsf->allocate(rawCount) ||
        !buildContext->rawSpectrum.allocate(rawCount)) {
        if (error) *error = "cuda-alloc-raw-psf-build";
        return false;
    }

    if (!checkCuda(cudaMemsetAsync(buildContext->rawSpectrum.ptr, 0, rawCount * sizeof(cufftComplex), stream),
                   "cudaMemsetAsync-raw-psf-spectrum",
                   error)) {
        return false;
    }

    const int offset = std::max(0, (rawPsfSize - pupilSize) / 2);
    dim3 block(16, 16);
    dim3 pupilGrid((pupilSize + block.x - 1) / block.x, (pupilSize + block.y - 1) / block.y);
    embedCenteredComplexPupilKernel<<<pupilGrid, block, 0, stream>>>(
        devicePupil.ptr,
        usePhase ? devicePhase->ptr : nullptr,
        pupilSize,
        rawPsfSize,
        offset,
        buildContext->rawSpectrum.ptr);
    if (!checkCuda(cudaGetLastError(), "embedCenteredComplexPupilKernel", error)) {
        return false;
    }

    CudaFftPlanLease fftPlan;
    const CudaFftBackend psfFftBackend =
        resolveCudaFftBackendForSubsystem(CudaFftSubsystem::PsfBank, requestedFftBackend, fftPolicy, timing);
    if (!acquireFftPlan(CudaFftSubsystem::PsfBank,
                        psfFftBackend,
                        vkfftStrict,
                        nullptr,
                        usePersistentVkfftRepositoryForSubsystem(CudaFftSubsystem::PsfBank, fftPolicy, false),
                        rawPsfSize,
                        rawPsfSize,
                        1,
                        stream,
                        timing,
                        &fftPlan,
                        error)) {
        return false;
    }

    const bool fftOk = execFftC2C(&fftPlan, CudaFftSubsystem::PsfBank, buildContext->rawSpectrum, false, rawPsfSize, rawPsfSize, stream, timing, error);
    if (!fftOk) {
        return false;
    }

    dim3 rawGrid((rawPsfSize + block.x - 1) / block.x, (rawPsfSize + block.y - 1) / block.y);
    extractShiftedIntensityKernel<<<rawGrid, block, 0, stream>>>(buildContext->rawSpectrum.ptr,
                                                                 rawPsfSize,
                                                                 buildContext->shiftedIntensity.ptr);
    if (!checkCuda(cudaGetLastError(), "extractShiftedIntensityKernel", error)) {
        return false;
    }
    if (!normalizeDeviceBufferGpu(buildContext->shiftedIntensity,
                                  rawCount,
                                  &buildContext->reductionScalarA,
                                  &buildContext->reductionScalarB,
                                  stream,
                                  error)) {
        return false;
    }
    if (!checkCuda(cudaMemcpyAsync(deviceShiftedRawPsf->ptr,
                                   buildContext->shiftedIntensity.ptr,
                                   rawCount * sizeof(float),
                                   cudaMemcpyDeviceToDevice,
                                   stream),
                   "cudaMemcpyAsync-raw-psf-device-copy",
                   error)) {
        return false;
    }
    if (shiftedRawPsfDisplay != nullptr) {
        shiftedRawPsfDisplay->assign(rawCount, 0.0f);
        if (!checkCuda(cudaMemcpyAsync(shiftedRawPsfDisplay->data(),
                                       deviceShiftedRawPsf->ptr,
                                       rawCount * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       stream),
                       "cudaMemcpyAsync-raw-psf-readback",
                       error) ||
            !checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize-raw-psf-readback", error)) {
            return false;
        }
    }
    return true;
}

std::vector<float> radialMeanProfileCuda(const std::vector<float>& image, int size) {
    const int radiusMax = size / 2;
    std::vector<float> sums(static_cast<std::size_t>(radiusMax + 1), 0.0f);
    std::vector<int> counts(static_cast<std::size_t>(radiusMax + 1), 0);
    const float center = (size & 1) == 0 ? static_cast<float>(size) * 0.5f
                                         : static_cast<float>(size - 1) * 0.5f;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const float dx = static_cast<float>(x) - center;
            const float dy = static_cast<float>(y) - center;
            const int r = std::min(radiusMax, static_cast<int>(std::round(std::sqrt(dx * dx + dy * dy))));
            sums[static_cast<std::size_t>(r)] += image[static_cast<std::size_t>(y) * size + x];
            counts[static_cast<std::size_t>(r)] += 1;
        }
    }
    std::vector<float> profile(static_cast<std::size_t>(radiusMax + 1), 0.0f);
    for (int r = 0; r <= radiusMax; ++r) {
        profile[static_cast<std::size_t>(r)] = sums[static_cast<std::size_t>(r)] / static_cast<float>(std::max(1, counts[static_cast<std::size_t>(r)]));
    }
    return profile;
}

float refineParabolicMinimumCuda(float left, float center, float right) {
    const float denom = left - 2.0f * center + right;
    if (std::abs(denom) < 1e-6f) {
        return 0.0f;
    }
    return std::max(-0.5f, std::min(0.5f, 0.5f * (left - right) / denom));
}

float estimateFirstMinimumRadiusCuda(const std::vector<float>& shiftedRawPsf, int size) {
    const std::vector<float> profile = radialMeanProfileCuda(shiftedRawPsf, size);
    if (profile.size() < 4U) {
        return 1.0f;
    }
    std::vector<float> smooth(profile.size(), 0.0f);
    for (std::size_t i = 0; i < profile.size(); ++i) {
        const float left = profile[i == 0 ? 0 : i - 1];
        const float center = profile[i];
        const float right = profile[i + 1 < profile.size() ? i + 1 : i];
        smooth[i] = (left + 2.0f * center + right) * 0.25f;
    }
    int bestIndex = 1;
    float bestValue = smooth[1];
    int descentCount = smooth[1] < smooth[0] ? 1 : 0;
    for (int i = 2; i + 1 < static_cast<int>(smooth.size()); ++i) {
        if (smooth[static_cast<std::size_t>(i)] <= smooth[static_cast<std::size_t>(i - 1)]) {
            descentCount += 1;
            if (smooth[static_cast<std::size_t>(i)] < bestValue) {
                bestValue = smooth[static_cast<std::size_t>(i)];
                bestIndex = i;
            }
            continue;
        }
        if (descentCount >= 2 && bestIndex > 0 && bestIndex + 1 < static_cast<int>(smooth.size())) {
            const float offset = refineParabolicMinimumCuda(smooth[static_cast<std::size_t>(bestIndex - 1)],
                                                            smooth[static_cast<std::size_t>(bestIndex)],
                                                            smooth[static_cast<std::size_t>(bestIndex + 1)]);
            return std::max(1.0f, static_cast<float>(bestIndex) + offset);
        }
        bestIndex = i;
        bestValue = smooth[static_cast<std::size_t>(i)];
        descentCount = 0;
    }
    const int fallbackLimit = std::max(2, std::min<int>(static_cast<int>(smooth.size()) - 1, std::max(4, size / 8)));
    bestIndex = 1;
    bestValue = smooth[1];
    for (int i = 2; i <= fallbackLimit; ++i) {
        if (smooth[static_cast<std::size_t>(i)] < bestValue) {
            bestValue = smooth[static_cast<std::size_t>(i)];
            bestIndex = i;
        }
    }
    if (bestIndex > 0 && bestIndex + 1 < static_cast<int>(smooth.size())) {
        const float offset = refineParabolicMinimumCuda(smooth[static_cast<std::size_t>(bestIndex - 1)],
                                                        smooth[static_cast<std::size_t>(bestIndex)],
                                                        smooth[static_cast<std::size_t>(bestIndex + 1)]);
        return std::max(1.0f, static_cast<float>(bestIndex) + offset);
    }
    return static_cast<float>(bestIndex);
}

std::vector<float> wavelengthsForModeCuda(LensDiffSpectralMode mode) {
    switch (mode) {
        case LensDiffSpectralMode::Spectral9:
            return {420.0f, 450.0f, 480.0f, 510.0f, 540.0f, 570.0f, 600.0f, 630.0f, 660.0f};
        case LensDiffSpectralMode::Spectral5:
            return {440.0f, 490.0f, 540.0f, 590.0f, 640.0f};
        case LensDiffSpectralMode::Tristimulus:
            return {460.0f, 550.0f, 610.0f};
        case LensDiffSpectralMode::Mono:
        default:
            return {550.0f};
    }
}

bool buildBaseKernelsFromRawPsfOnCuda(const std::vector<float>& rawPsf,
                                      int rawPsfSize,
                                      const std::vector<float>& wavelengths,
                                      float scaleBase,
                                      int supportRadius,
                                      cudaStream_t stream,
                                      std::vector<LensDiffKernel>* outKernels,
                                      std::string* error) {
    if (outKernels == nullptr) {
        if (error) *error = "missing-base-kernel-output";
        return false;
    }

    const int kernelSize = supportRadius * 2 + 1;
    const std::size_t rawCount = static_cast<std::size_t>(rawPsfSize) * rawPsfSize;
    const std::size_t kernelCount = static_cast<std::size_t>(kernelSize) * kernelSize;
    DeviceBuffer<float> deviceRawPsf;
    DeviceBuffer<float> deviceKernel;
    if (!deviceRawPsf.allocate(rawCount) || !deviceKernel.allocate(kernelCount)) {
        if (error) *error = "cuda-alloc-base-kernels";
        return false;
    }
    if (!checkCuda(cudaMemcpyAsync(deviceRawPsf.ptr,
                                   rawPsf.data(),
                                   rawCount * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "cudaMemcpyAsync-upload-raw-psf",
                   error)) {
        return false;
    }

    dim3 block(16, 16);
    dim3 grid((kernelSize + block.x - 1) / block.x, (kernelSize + block.y - 1) / block.y);
    std::vector<float> hostKernel(kernelCount, 0.0f);
    outKernels->clear();
    outKernels->reserve(wavelengths.size());
    for (float wavelength : wavelengths) {
        const float scaleFactor = scaleBase * (wavelength / 550.0f);
        const float invScale = 1.0f / std::max(scaleFactor, 0.05f);
        resampleRawPsfKernel<<<grid, block, 0, stream>>>(deviceRawPsf.ptr, rawPsfSize, invScale, supportRadius, deviceKernel.ptr);
        if (!checkCuda(cudaGetLastError(), "resampleRawPsfKernel", error)) {
            return false;
        }
        if (!checkCuda(cudaMemcpyAsync(hostKernel.data(),
                                       deviceKernel.ptr,
                                       kernelCount * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       stream),
                       "cudaMemcpyAsync-download-base-kernel",
                       error)) {
            return false;
        }
        if (!checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize-base-kernel", error)) {
            return false;
        }
        float sum = 0.0f;
        for (float value : hostKernel) {
            sum += value;
        }
        if (sum > 0.0f) {
            const float invSum = 1.0f / sum;
            for (float& value : hostKernel) {
                value *= invSum;
            }
        }
        LensDiffKernel kernel {};
        kernel.size = kernelSize;
        kernel.values = hostKernel;
        outKernels->push_back(std::move(kernel));
    }
    return true;
}

bool normalizeDeviceBuffer(DeviceBuffer<float>& buffer,
                           std::size_t count,
                           cudaStream_t stream,
                           std::string* error) {
    if (count == 0) {
        return true;
    }
    float hostSum = 0.0f;
    DeviceBuffer<float> deviceSum;
    if (!deviceSum.allocate(1)) {
        if (error) *error = "cuda-alloc-normalize-sum";
        return false;
    }
    if (!checkCuda(cudaMemsetAsync(deviceSum.ptr, 0, sizeof(float), stream), "cudaMemsetAsync-normalize-sum", error)) {
        return false;
    }
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    lumaReduceKernel<<<grid, block, 0, stream>>>(buffer.ptr, buffer.ptr, buffer.ptr, count, deviceSum.ptr);
    if (!checkCuda(cudaGetLastError(), "lumaReduceKernel-normalize", error)) {
        return false;
    }
    if (!checkCuda(cudaMemcpyAsync(&hostSum, deviceSum.ptr, sizeof(float), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync-normalize-sum", error)) {
        return false;
    }
    if (!checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize-normalize-sum", error)) {
        return false;
    }
    if (hostSum <= 0.0f) {
        return true;
    }
    scaleBufferKernel<<<grid, block, 0, stream>>>(buffer.ptr, count, 1.0f / hostSum);
    return checkCuda(cudaGetLastError(), "scaleBufferKernel-normalize", error);
}

bool buildAzimuthalMeanKernelOnCuda(const LensDiffKernel& baseKernel,
                                    cudaStream_t stream,
                                    DeviceBuffer<float>* meanKernel,
                                    KernelStatsHost* outStats,
                                    std::string* error) {
    if (meanKernel == nullptr || outStats == nullptr) {
        if (error) *error = "cuda-null-mean-kernel-target";
        return false;
    }
    const int kernelSize = baseKernel.size;
    const std::size_t kernelCount = baseKernel.values.size();
    const int radiusMax = kernelSize / 2;
    DeviceBuffer<float> deviceBaseKernel;
    DeviceBuffer<float> deviceSums;
    DeviceBuffer<int> deviceCounts;
    if (!deviceBaseKernel.allocate(kernelCount) || !deviceSums.allocate(static_cast<std::size_t>(radiusMax + 1)) ||
        !deviceCounts.allocate(static_cast<std::size_t>(radiusMax + 1)) || !meanKernel->allocate(kernelCount)) {
        if (error) *error = "cuda-alloc-mean-kernel";
        return false;
    }
    if (!checkCuda(cudaMemcpyAsync(deviceBaseKernel.ptr,
                                   baseKernel.values.data(),
                                   kernelCount * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "cudaMemcpyAsync-upload-base-kernel-mean",
                   error)) {
        return false;
    }
    if (!checkCuda(cudaMemsetAsync(deviceSums.ptr, 0, static_cast<std::size_t>(radiusMax + 1) * sizeof(float), stream),
                   "cudaMemsetAsync-radial-sums",
                   error) ||
        !checkCuda(cudaMemsetAsync(deviceCounts.ptr, 0, static_cast<std::size_t>(radiusMax + 1) * sizeof(int), stream),
                   "cudaMemsetAsync-radial-counts",
                   error)) {
        return false;
    }

    dim3 block2d(16, 16);
    dim3 grid2d((kernelSize + block2d.x - 1) / block2d.x, (kernelSize + block2d.y - 1) / block2d.y);
    const float center = static_cast<float>(kernelSize - 1) * 0.5f;
    accumulateRadialProfileKernel<<<grid2d, block2d, 0, stream>>>(deviceBaseKernel.ptr, kernelSize, radiusMax, center, deviceSums.ptr, deviceCounts.ptr);
    if (!checkCuda(cudaGetLastError(), "accumulateRadialProfileKernel", error)) {
        return false;
    }
    expandRadialMeanKernel<<<grid2d, block2d, 0, stream>>>(deviceSums.ptr, deviceCounts.ptr, kernelSize, radiusMax, center, meanKernel->ptr);
    if (!checkCuda(cudaGetLastError(), "expandRadialMeanKernel", error)) {
        return false;
    }
    if (!normalizeDeviceBuffer(*meanKernel, kernelCount, stream, error)) {
        return false;
    }

    outStats->sums.assign(static_cast<std::size_t>(radiusMax + 1), 0.0f);
    outStats->counts.assign(static_cast<std::size_t>(radiusMax + 1), 0);
    if (!checkCuda(cudaMemcpyAsync(outStats->sums.data(),
                                   deviceSums.ptr,
                                   static_cast<std::size_t>(radiusMax + 1) * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   stream),
                   "cudaMemcpyAsync-download-radial-sums",
                   error) ||
        !checkCuda(cudaMemcpyAsync(outStats->counts.data(),
                                   deviceCounts.ptr,
                                   static_cast<std::size_t>(radiusMax + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   stream),
                   "cudaMemcpyAsync-download-radial-counts",
                   error)) {
        return false;
    }
    return checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize-mean-kernel", error);
}

bool buildShapedKernelOnCuda(const LensDiffKernel& baseKernel,
                             const DeviceBuffer<float>& meanKernel,
                             float anisotropyEmphasis,
                             cudaStream_t stream,
                             DeviceBuffer<float>* shapedKernel,
                             std::string* error) {
    if (shapedKernel == nullptr) {
        if (error) *error = "cuda-null-shaped-kernel-target";
        return false;
    }
    const std::size_t kernelCount = baseKernel.values.size();
    DeviceBuffer<float> deviceBaseKernel;
    if (!deviceBaseKernel.allocate(kernelCount) || !shapedKernel->allocate(kernelCount)) {
        if (error) *error = "cuda-alloc-shaped-kernel";
        return false;
    }
    if (!checkCuda(cudaMemcpyAsync(deviceBaseKernel.ptr,
                                   baseKernel.values.data(),
                                   kernelCount * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "cudaMemcpyAsync-upload-base-kernel-shaped",
                   error)) {
        return false;
    }
    const float gain = 1.0f + std::max(0.0f, anisotropyEmphasis) * 4.0f;
    const int block = 256;
    const int grid = static_cast<int>((kernelCount + block - 1) / block);
    reshapeKernel<<<grid, block, 0, stream>>>(deviceBaseKernel.ptr, meanKernel.ptr, kernelCount, gain, shapedKernel->ptr);
    if (!checkCuda(cudaGetLastError(), "reshapeKernel", error)) {
        return false;
    }
    return normalizeDeviceBuffer(*shapedKernel, kernelCount, stream, error);
}

bool buildStructureKernelOnCuda(const DeviceBuffer<float>& shapedKernel,
                                const DeviceBuffer<float>& meanKernel,
                                std::size_t kernelCount,
                                cudaStream_t stream,
                                DeviceBuffer<float>* structureKernel,
                                std::string* error) {
    if (structureKernel == nullptr) {
        if (error) *error = "cuda-null-structure-kernel-target";
        return false;
    }
    if (!structureKernel->allocate(kernelCount)) {
        if (error) *error = "cuda-alloc-structure-kernel";
        return false;
    }
    const int block = 256;
    const int grid = static_cast<int>((kernelCount + block - 1) / block);
    positiveResidualKernelCuda<<<grid, block, 0, stream>>>(shapedKernel.ptr, meanKernel.ptr, kernelCount, structureKernel->ptr);
    if (!checkCuda(cudaGetLastError(), "positiveResidualKernelCuda", error)) {
        return false;
    }
    return normalizeDeviceBuffer(*structureKernel, kernelCount, stream, error);
}

bool computeAdaptiveSupportStatsOnCuda(const DeviceBuffer<float>& shapedKernel,
                                       int kernelSize,
                                       int maxRadius,
                                       cudaStream_t stream,
                                       KernelStatsHost* outStats,
                                       float* outTotalEnergy,
                                       float* outGlobalPeak,
                                       std::string* error) {
    if (outStats == nullptr || outTotalEnergy == nullptr || outGlobalPeak == nullptr) {
        if (error) *error = "cuda-null-support-stats-target";
        return false;
    }
    const int radiusMax = std::min(maxRadius, kernelSize / 2);
    DeviceBuffer<float> deviceRingEnergy;
    DeviceBuffer<float> deviceRingPeak;
    DeviceBuffer<float> deviceTotalEnergy;
    DeviceBuffer<float> deviceGlobalPeak;
    if (!deviceRingEnergy.allocate(static_cast<std::size_t>(radiusMax + 1)) ||
        !deviceRingPeak.allocate(static_cast<std::size_t>(radiusMax + 1)) ||
        !deviceTotalEnergy.allocate(1) ||
        !deviceGlobalPeak.allocate(1)) {
        if (error) *error = "cuda-alloc-support-stats";
        return false;
    }
    if (!checkCuda(cudaMemsetAsync(deviceRingEnergy.ptr, 0, static_cast<std::size_t>(radiusMax + 1) * sizeof(float), stream),
                   "cudaMemsetAsync-ring-energy",
                   error) ||
        !checkCuda(cudaMemsetAsync(deviceRingPeak.ptr, 0, static_cast<std::size_t>(radiusMax + 1) * sizeof(float), stream),
                   "cudaMemsetAsync-ring-peak",
                   error) ||
        !checkCuda(cudaMemsetAsync(deviceTotalEnergy.ptr, 0, sizeof(float), stream), "cudaMemsetAsync-total-energy", error) ||
        !checkCuda(cudaMemsetAsync(deviceGlobalPeak.ptr, 0, sizeof(float), stream), "cudaMemsetAsync-global-peak", error)) {
        return false;
    }
    dim3 block2d(16, 16);
    dim3 grid2d((kernelSize + block2d.x - 1) / block2d.x, (kernelSize + block2d.y - 1) / block2d.y);
    const float center = static_cast<float>(kernelSize - 1) * 0.5f;
    accumulateSupportStatsKernel<<<grid2d, block2d, 0, stream>>>(
        shapedKernel.ptr, kernelSize, radiusMax, center, deviceRingEnergy.ptr, deviceRingPeak.ptr, deviceTotalEnergy.ptr, deviceGlobalPeak.ptr);
    if (!checkCuda(cudaGetLastError(), "accumulateSupportStatsKernel", error)) {
        return false;
    }

    outStats->ringEnergy.assign(static_cast<std::size_t>(radiusMax + 1), 0.0f);
    outStats->ringPeak.assign(static_cast<std::size_t>(radiusMax + 1), 0.0f);
    if (!checkCuda(cudaMemcpyAsync(outStats->ringEnergy.data(),
                                   deviceRingEnergy.ptr,
                                   static_cast<std::size_t>(radiusMax + 1) * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   stream),
                   "cudaMemcpyAsync-download-ring-energy",
                   error) ||
        !checkCuda(cudaMemcpyAsync(outStats->ringPeak.data(),
                                   deviceRingPeak.ptr,
                                   static_cast<std::size_t>(radiusMax + 1) * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   stream),
                   "cudaMemcpyAsync-download-ring-peak",
                   error) ||
        !checkCuda(cudaMemcpyAsync(outTotalEnergy, deviceTotalEnergy.ptr, sizeof(float), cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync-download-total-energy",
                   error) ||
        !checkCuda(cudaMemcpyAsync(outGlobalPeak, deviceGlobalPeak.ptr, sizeof(float), cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync-download-global-peak",
                   error)) {
        return false;
    }
    return checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize-support-stats", error);
}

int estimateAdaptiveSupportRadiusFromStats(const KernelStatsHost& stats, int maxRadius, float totalEnergy, float globalPeak) {
    const int radiusMax = std::min(maxRadius, static_cast<int>(stats.ringEnergy.size()) - 1);
    if (radiusMax < 0 || totalEnergy <= 1e-6f || globalPeak <= 1e-6f) {
        return std::max(4, maxRadius);
    }
    std::vector<float> outsidePeak(static_cast<std::size_t>(radiusMax + 2), 0.0f);
    for (int r = radiusMax; r >= 0; --r) {
        outsidePeak[static_cast<std::size_t>(r)] = std::max(stats.ringPeak[static_cast<std::size_t>(r)], outsidePeak[static_cast<std::size_t>(r + 1)]);
    }
    float cumulativeEnergy = 0.0f;
    for (int r = 0; r <= radiusMax; ++r) {
        cumulativeEnergy += stats.ringEnergy[static_cast<std::size_t>(r)];
        const float captured = cumulativeEnergy / totalEnergy;
        const float remainingPeak = outsidePeak[static_cast<std::size_t>(std::min(radiusMax + 1, r + 1))];
        if (captured >= 0.9999f && remainingPeak <= globalPeak * 2e-4f) {
            return std::max(4, r + 1);
        }
    }
    return std::max(4, radiusMax);
}

bool downloadCroppedKernel(const DeviceBuffer<float>& srcKernel,
                           int srcSize,
                           int radius,
                           cudaStream_t stream,
                           LensDiffKernel* outKernel,
                           std::string* error) {
    if (outKernel == nullptr) {
        if (error) *error = "cuda-null-download-kernel-target";
        return false;
    }
    const int clampedRadius = std::max(1, std::min(radius, srcSize / 2));
    const int dstSize = clampedRadius * 2 + 1;
    const std::size_t dstCount = static_cast<std::size_t>(dstSize) * dstSize;
    DeviceBuffer<float> deviceCropped;
    DeviceBuffer<float> deviceSum;
    if (!deviceCropped.allocate(dstCount)) {
        if (error) *error = "cuda-alloc-cropped-kernel";
        return false;
    }
    if (!deviceSum.allocate(1)) {
        if (error) *error = "cuda-alloc-cropped-kernel-sum";
        return false;
    }
    dim3 block2d(16, 16);
    dim3 grid2d((dstSize + block2d.x - 1) / block2d.x, (dstSize + block2d.y - 1) / block2d.y);
    cropKernelToRadiusCuda<<<grid2d, block2d, 0, stream>>>(srcKernel.ptr, srcSize, clampedRadius, dstSize, deviceCropped.ptr);
    if (!checkCuda(cudaGetLastError(), "cropKernelToRadiusCuda", error)) {
        return false;
    }
    applySupportBoundaryTaperCuda<<<grid2d, block2d, 0, stream>>>(deviceCropped.ptr, dstSize, clampedRadius);
    if (!checkCuda(cudaGetLastError(), "applySupportBoundaryTaperCuda", error)) {
        return false;
    }
    if (!checkCuda(cudaMemsetAsync(deviceSum.ptr, 0, sizeof(float), stream), "cudaMemsetAsync-cropped-kernel-sum", error)) {
        return false;
    }
    const int sumBlock = 256;
    const int sumGrid = static_cast<int>((dstCount + sumBlock - 1) / sumBlock);
    reduceSumKernel<<<sumGrid, sumBlock, 0, stream>>>(deviceCropped.ptr, dstCount, deviceSum.ptr);
    if (!checkCuda(cudaGetLastError(), "reduceSumKernel-cropped-kernel", error)) {
        return false;
    }
    float sum = 0.0f;
    if (!checkCuda(cudaMemcpyAsync(&sum, deviceSum.ptr, sizeof(float), cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync-cropped-kernel-sum",
                   error)) {
        return false;
    }
    if (!checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize-cropped-kernel-sum", error)) {
        return false;
    }
    if (sum > 0.0f) {
        scaleBufferKernel<<<sumGrid, sumBlock, 0, stream>>>(deviceCropped.ptr, dstCount, 1.0f / sum);
        if (!checkCuda(cudaGetLastError(), "scaleBufferKernel-cropped-kernel", error)) {
            return false;
        }
    }
    outKernel->size = dstSize;
    outKernel->values.assign(dstCount, 0.0f);
    if (!checkCuda(cudaMemcpyAsync(outKernel->values.data(),
                                   deviceCropped.ptr,
                                   dstCount * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   stream),
                   "cudaMemcpyAsync-download-cropped-kernel",
                   error)) {
        return false;
    }
    if (!checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize-download-cropped-kernel", error)) {
        return false;
    }
    return true;
}

bool finalizePsfBankOnCuda(const LensDiffParams& params,
                           const LensDiffPsfBankKey& key,
                           const std::vector<float>& pupil,
                           const std::vector<float>& phaseWaves,
                           int pupilSize,
                           const DeviceBuffer<float>& rawPsf,
                           int rawPsfSize,
                           const std::vector<float>& wavelengths,
                           float scaleBase,
                           cudaStream_t stream,
                           LensDiffPsfBankCache* cache,
                           CudaPsfBuildContext* buildContext,
                           std::string* error) {
    if (cache == nullptr) {
        if (error) *error = "cuda-null-psf-cache";
        return false;
    }
    CudaPsfBuildContext localContext {};
    CudaPsfBuildContext* psfContext = buildContext != nullptr ? buildContext : &localContext;
    *cache = {};
    cache->valid = true;
    cache->key = key;
    cache->supportRadiusPx = 4;
    cache->pupilDisplay = pupil;
    cache->pupilDisplaySize = pupilSize;
    cache->phaseDisplay = phaseWaves;
    cache->phaseDisplaySize = pupilSize;
    cache->bins.clear();
    cache->bins.reserve(wavelengths.size());

    const int maxKernelRadiusPx = std::max(4, ResolveLensDiffMaxKernelRadiusPx(params));
    const int kernelSize = maxKernelRadiusPx * 2 + 1;
    const int radiusMax = kernelSize / 2;
    const std::size_t kernelCount = static_cast<std::size_t>(kernelSize) * kernelSize;
    if (!psfContext->baseKernel.allocate(kernelCount) ||
        !psfContext->meanKernel.allocate(kernelCount) ||
        !psfContext->shapedKernel.allocate(kernelCount) ||
        !psfContext->structureKernel.allocate(kernelCount) ||
        !psfContext->ringSums.allocate(static_cast<std::size_t>(radiusMax + 1)) ||
        !psfContext->ringCounts.allocate(static_cast<std::size_t>(radiusMax + 1)) ||
        !psfContext->ringEnergy.allocate(static_cast<std::size_t>(radiusMax + 1)) ||
        !psfContext->ringPeak.allocate(static_cast<std::size_t>(radiusMax + 1)) ||
        !psfContext->totalEnergy.allocate(1) ||
        !psfContext->globalPeak.allocate(1)) {
        if (error) *error = "cuda-alloc-psf-build-context";
        return false;
    }

    dim3 block2d(16, 16);
    dim3 kernelGrid((kernelSize + block2d.x - 1) / block2d.x, (kernelSize + block2d.y - 1) / block2d.y);

    for (std::size_t i = 0; i < wavelengths.size(); ++i) {
        const auto wavelengthStart = std::chrono::steady_clock::now();
        const float wavelength = wavelengths[i];
        const float scaleFactor = scaleBase * (wavelength / 550.0f);
        const float invScale = 1.0f / std::max(scaleFactor, 0.05f);
        const auto finalizeStart = std::chrono::steady_clock::now();

        if (!checkCuda(cudaMemsetAsync(psfContext->ringSums.ptr, 0, static_cast<std::size_t>(radiusMax + 1) * sizeof(float), stream),
                       "cudaMemsetAsync-psf-ring-sums",
                       error) ||
            !checkCuda(cudaMemsetAsync(psfContext->ringCounts.ptr, 0, static_cast<std::size_t>(radiusMax + 1) * sizeof(int), stream),
                       "cudaMemsetAsync-psf-ring-counts",
                       error) ||
            !checkCuda(cudaMemsetAsync(psfContext->ringEnergy.ptr, 0, static_cast<std::size_t>(radiusMax + 1) * sizeof(float), stream),
                       "cudaMemsetAsync-psf-ring-energy",
                       error) ||
            !checkCuda(cudaMemsetAsync(psfContext->ringPeak.ptr, 0, static_cast<std::size_t>(radiusMax + 1) * sizeof(float), stream),
                       "cudaMemsetAsync-psf-ring-peak",
                       error) ||
            !checkCuda(cudaMemsetAsync(psfContext->totalEnergy.ptr, 0, sizeof(float), stream), "cudaMemsetAsync-psf-total-energy", error) ||
            !checkCuda(cudaMemsetAsync(psfContext->globalPeak.ptr, 0, sizeof(float), stream), "cudaMemsetAsync-psf-global-peak", error)) {
            return false;
        }

        resampleRawPsfKernel<<<kernelGrid, block2d, 0, stream>>>(rawPsf.ptr, rawPsfSize, invScale, maxKernelRadiusPx, psfContext->baseKernel.ptr);
        if (!checkCuda(cudaGetLastError(), "resampleRawPsfKernel-psf-finalize", error) ||
            !normalizeDeviceBufferGpu(psfContext->baseKernel, kernelCount, &psfContext->reductionScalarA, &psfContext->scalarScale, stream, error)) {
            return false;
        }

        const float center = static_cast<float>(kernelSize - 1) * 0.5f;
        accumulateRadialProfileKernel<<<kernelGrid, block2d, 0, stream>>>(psfContext->baseKernel.ptr,
                                                                           kernelSize,
                                                                           radiusMax,
                                                                           center,
                                                                           psfContext->ringSums.ptr,
                                                                           psfContext->ringCounts.ptr);
        if (!checkCuda(cudaGetLastError(), "accumulateRadialProfileKernel-psf-finalize", error)) {
            return false;
        }
        expandRadialMeanKernel<<<kernelGrid, block2d, 0, stream>>>(psfContext->ringSums.ptr,
                                                                   psfContext->ringCounts.ptr,
                                                                   kernelSize,
                                                                   radiusMax,
                                                                   center,
                                                                   psfContext->meanKernel.ptr);
        if (!checkCuda(cudaGetLastError(), "expandRadialMeanKernel-psf-finalize", error) ||
            !normalizeDeviceBufferGpu(psfContext->meanKernel, kernelCount, &psfContext->reductionScalarA, &psfContext->scalarScale, stream, error)) {
            return false;
        }

        const float gain = 1.0f + std::max(0.0f, static_cast<float>(params.anisotropyEmphasis)) * 4.0f;
        const int flatBlock = 256;
        const int flatGrid = static_cast<int>((kernelCount + flatBlock - 1) / flatBlock);
        reshapeKernel<<<flatGrid, flatBlock, 0, stream>>>(psfContext->baseKernel.ptr,
                                                          psfContext->meanKernel.ptr,
                                                          kernelCount,
                                                          gain,
                                                          psfContext->shapedKernel.ptr);
        if (!checkCuda(cudaGetLastError(), "reshapeKernel-psf-finalize", error) ||
            !normalizeDeviceBufferGpu(psfContext->shapedKernel, kernelCount, &psfContext->reductionScalarA, &psfContext->scalarScale, stream, error)) {
            return false;
        }

        positiveResidualKernelCuda<<<flatGrid, flatBlock, 0, stream>>>(psfContext->shapedKernel.ptr,
                                                                       psfContext->meanKernel.ptr,
                                                                       kernelCount,
                                                                       psfContext->structureKernel.ptr);
        if (!checkCuda(cudaGetLastError(), "positiveResidualKernelCuda-psf-finalize", error) ||
            !normalizeDeviceBufferGpu(psfContext->structureKernel, kernelCount, &psfContext->reductionScalarA, &psfContext->scalarScale, stream, error)) {
            return false;
        }

        accumulateSupportStatsKernel<<<kernelGrid, block2d, 0, stream>>>(psfContext->shapedKernel.ptr,
                                                                          kernelSize,
                                                                          radiusMax,
                                                                          center,
                                                                          psfContext->ringEnergy.ptr,
                                                                          psfContext->ringPeak.ptr,
                                                                          psfContext->totalEnergy.ptr,
                                                                          psfContext->globalPeak.ptr);
        if (!checkCuda(cudaGetLastError(), "accumulateSupportStatsKernel-psf-finalize", error)) {
            return false;
        }

        KernelStatsHost stats;
        stats.ringEnergy.assign(static_cast<std::size_t>(radiusMax + 1), 0.0f);
        stats.ringPeak.assign(static_cast<std::size_t>(radiusMax + 1), 0.0f);
        float totalEnergy = 0.0f;
        float globalPeak = 0.0f;
        if (!checkCuda(cudaMemcpyAsync(stats.ringEnergy.data(),
                                       psfContext->ringEnergy.ptr,
                                       static_cast<std::size_t>(radiusMax + 1) * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       stream),
                       "cudaMemcpyAsync-psf-ring-energy-readback",
                       error) ||
            !checkCuda(cudaMemcpyAsync(stats.ringPeak.data(),
                                       psfContext->ringPeak.ptr,
                                       static_cast<std::size_t>(radiusMax + 1) * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       stream),
                       "cudaMemcpyAsync-psf-ring-peak-readback",
                       error) ||
            !checkCuda(cudaMemcpyAsync(&totalEnergy, psfContext->totalEnergy.ptr, sizeof(float), cudaMemcpyDeviceToHost, stream),
                       "cudaMemcpyAsync-psf-total-energy-readback",
                       error) ||
            !checkCuda(cudaMemcpyAsync(&globalPeak, psfContext->globalPeak.ptr, sizeof(float), cudaMemcpyDeviceToHost, stream),
                       "cudaMemcpyAsync-psf-global-peak-readback",
                       error) ||
            !checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize-psf-support-readback", error)) {
            return false;
        }
        if (LensDiffTimingEnabled()) {
            LogLensDiffTimingStage(
                "cuda-psf-wavelength-finalize",
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - finalizeStart)
                    .count(),
                "nm=" + std::to_string(static_cast<int>(std::lround(wavelength))));
        }

        const auto readbackStart = std::chrono::steady_clock::now();
        const int effectiveRadius = paddedAdaptiveSupportRadiusHost(
            estimateAdaptiveSupportRadiusFromStats(stats, maxKernelRadiusPx, totalEnergy, globalPeak),
            maxKernelRadiusPx);
        if (LensDiffTimingEnabled()) {
            LogLensDiffTimingStage(
                "cuda-psf-support-radius-readback",
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - readbackStart)
                    .count(),
                "nm=" + std::to_string(static_cast<int>(std::lround(wavelength))) +
                    ",radius=" + std::to_string(effectiveRadius));
        }

        const int croppedSize = effectiveRadius * 2 + 1;
        const std::size_t croppedCount = static_cast<std::size_t>(croppedSize) * croppedSize;
        if (!psfContext->cropCore.allocate(croppedCount) ||
            !psfContext->cropFull.allocate(croppedCount) ||
            !psfContext->cropStructure.allocate(croppedCount)) {
            if (error) *error = "cuda-alloc-psf-crop";
            return false;
        }

        const auto cropStart = std::chrono::steady_clock::now();
        dim3 cropGrid((croppedSize + block2d.x - 1) / block2d.x, (croppedSize + block2d.y - 1) / block2d.y);
        cropKernelToRadiusCuda<<<cropGrid, block2d, 0, stream>>>(psfContext->meanKernel.ptr,
                                                                 kernelSize,
                                                                 effectiveRadius,
                                                                 croppedSize,
                                                                 psfContext->cropCore.ptr);
        cropKernelToRadiusCuda<<<cropGrid, block2d, 0, stream>>>(psfContext->shapedKernel.ptr,
                                                                 kernelSize,
                                                                 effectiveRadius,
                                                                 croppedSize,
                                                                 psfContext->cropFull.ptr);
        cropKernelToRadiusCuda<<<cropGrid, block2d, 0, stream>>>(psfContext->structureKernel.ptr,
                                                                 kernelSize,
                                                                 effectiveRadius,
                                                                 croppedSize,
                                                                 psfContext->cropStructure.ptr);
        if (!checkCuda(cudaGetLastError(), "cropKernelToRadiusCuda-psf-finalize", error)) {
            return false;
        }
        applySupportBoundaryTaperCuda<<<cropGrid, block2d, 0, stream>>>(psfContext->cropCore.ptr, croppedSize, effectiveRadius);
        applySupportBoundaryTaperCuda<<<cropGrid, block2d, 0, stream>>>(psfContext->cropFull.ptr, croppedSize, effectiveRadius);
        applySupportBoundaryTaperCuda<<<cropGrid, block2d, 0, stream>>>(psfContext->cropStructure.ptr, croppedSize, effectiveRadius);
        if (!checkCuda(cudaGetLastError(), "applySupportBoundaryTaperCuda-psf-finalize", error) ||
            !normalizeDeviceBufferGpu(psfContext->cropCore, croppedCount, &psfContext->reductionScalarA, &psfContext->scalarScale, stream, error) ||
            !normalizeDeviceBufferGpu(psfContext->cropFull, croppedCount, &psfContext->reductionScalarA, &psfContext->scalarScale, stream, error) ||
            !normalizeDeviceBufferGpu(psfContext->cropStructure, croppedCount, &psfContext->reductionScalarA, &psfContext->scalarScale, stream, error)) {
            return false;
        }

        LensDiffPsfBin bin {};
        bin.wavelengthNm = wavelength;
        bin.core.size = croppedSize;
        bin.full.size = croppedSize;
        bin.structure.size = croppedSize;
        bin.core.values.assign(croppedCount, 0.0f);
        bin.full.values.assign(croppedCount, 0.0f);
        bin.structure.values.assign(croppedCount, 0.0f);
        if (!checkCuda(cudaMemcpyAsync(bin.core.values.data(),
                                       psfContext->cropCore.ptr,
                                       croppedCount * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       stream),
                       "cudaMemcpyAsync-crop-core-readback",
                       error) ||
            !checkCuda(cudaMemcpyAsync(bin.full.values.data(),
                                       psfContext->cropFull.ptr,
                                       croppedCount * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       stream),
                       "cudaMemcpyAsync-crop-full-readback",
                       error) ||
            !checkCuda(cudaMemcpyAsync(bin.structure.values.data(),
                                       psfContext->cropStructure.ptr,
                                       croppedCount * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       stream),
                       "cudaMemcpyAsync-crop-structure-readback",
                       error) ||
            !checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize-psf-crop-readback", error)) {
            return false;
        }
        if (LensDiffTimingEnabled()) {
            LogLensDiffTimingStage(
                "cuda-psf-crop-normalize",
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - cropStart)
                    .count(),
                "nm=" + std::to_string(static_cast<int>(std::lround(wavelength))) +
                    ",cropped=" + std::to_string(croppedSize));
        }

        cache->supportRadiusPx = std::max(cache->supportRadiusPx, effectiveRadius);
        cache->bins.push_back(std::move(bin));
        if (LensDiffTimingEnabled()) {
            LogLensDiffTimingStage(
                "cuda-psf-wavelength-total",
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - wavelengthStart)
                    .count(),
                "nm=" + std::to_string(static_cast<int>(std::lround(wavelength))));
        }
    }
    return true;
}

bool computeEnergySum(const DeviceBuffer<float>& r,
                      const DeviceBuffer<float>& g,
                      const DeviceBuffer<float>& b,
                      std::size_t count,
                      cudaStream_t stream,
                      float* outSum,
                      std::string* error) {
    DeviceBuffer<float> deviceSum;
    if (!deviceSum.allocate(1)) {
        if (error) *error = "cuda-alloc-energy-sum";
        return false;
    }
    if (!checkCuda(cudaMemsetAsync(deviceSum.ptr, 0, sizeof(float), stream), "cudaMemsetAsync-energy-sum", error)) {
        return false;
    }

    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    lumaReduceKernel<<<grid, block, 0, stream>>>(r.ptr, g.ptr, b.ptr, count, deviceSum.ptr);
    if (!checkCuda(cudaGetLastError(), "lumaReduceKernel", error)) {
        return false;
    }
    if (!checkCuda(cudaMemcpyAsync(outSum, deviceSum.ptr, sizeof(float), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync-energy-readback", error)) {
        return false;
    }
    if (!checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize-energy-readback", error)) {
        return false;
    }
    return true;
}

bool copyPackedToDestination(const LensDiffRenderRequest& request,
                             const float* packedSrc,
                             int packedWidth,
                             int packedHeight,
                             cudaStream_t stream,
                             cudaMemcpyKind copyKind,
                             std::string* error) {
    const LensDiffImageRect outputRect {
        std::max(request.renderWindow.x1, request.dst.bounds.x1),
        std::max(request.renderWindow.y1, request.dst.bounds.y1),
        std::min(request.renderWindow.x2, request.dst.bounds.x2),
        std::min(request.renderWindow.y2, request.dst.bounds.y2),
    };
    if (outputRect.width() <= 0 || outputRect.height() <= 0) {
        return true;
    }

    const int srcOffsetX = outputRect.x1 - request.src.bounds.x1;
    const int srcOffsetY = outputRect.y1 - request.src.bounds.y1;
    if (srcOffsetX < 0 || srcOffsetY < 0 || srcOffsetX + outputRect.width() > packedWidth || srcOffsetY + outputRect.height() > packedHeight) {
        if (error) *error = "packed-output-copy-out-of-range";
        return false;
    }

    auto* dstBytes = static_cast<std::uint8_t*>(request.dst.data);
    auto* srcBytes = reinterpret_cast<const std::uint8_t*>(packedSrc);
    const std::size_t dstOffset = static_cast<std::size_t>(outputRect.y1 - request.dst.bounds.y1) * static_cast<std::size_t>(request.dst.rowBytes) +
                                  static_cast<std::size_t>(outputRect.x1 - request.dst.bounds.x1) * 4U * sizeof(float);
    const std::size_t srcOffset = (static_cast<std::size_t>(srcOffsetY) * packedWidth + static_cast<std::size_t>(srcOffsetX)) * 4U * sizeof(float);
    const std::size_t copyBytesPerRow = static_cast<std::size_t>(outputRect.width()) * 4U * sizeof(float);
    return checkCuda(cudaMemcpy2DAsync(dstBytes + dstOffset,
                                       static_cast<std::size_t>(request.dst.rowBytes),
                                       srcBytes + srcOffset,
                                       static_cast<std::size_t>(packedWidth) * 4U * sizeof(float),
                                       copyBytesPerRow,
                                       static_cast<std::size_t>(outputRect.height()),
                                       copyKind,
                                       stream),
                     "cudaMemcpy2DAsync-packed-output",
                     error);
}

bool createStaticDebugImage(const LensDiffParams& params,
                            const LensDiffPsfBankCache& cache,
                            int outWidth,
                            int outHeight,
                            PackedHostImage* outImage) {
    if (outImage == nullptr) return false;
    outImage->width = outWidth;
    outImage->height = outHeight;
    LensDiffPsfBankCache* mutableCache = const_cast<LensDiffPsfBankCache*>(&cache);
    outImage->pixels = GetLensDiffStaticDebugRgbaCached(params, mutableCache, outWidth, outHeight);
    return true;
}

bool buildPsfBankGlobalOnlyCuda(const LensDiffParams& params,
                                LensDiffPsfBankCache& cache,
                                CudaPsfBuildContext* buildContext,
                                CudaFftPolicy fftPolicy,
                                CudaFftBackend requestedFftBackend,
                                bool vkfftStrict,
                                CudaRenderTimingBreakdown* timing,
                                cudaStream_t stream,
                                std::string* error) {
    const LensDiffPsfBankKey key = MakeLensDiffPsfBankKey(params);
    if (cache.valid && cache.key == key) {
        return true;
    }
    LensDiffScopedTimer timer("cuda-psf-bank-global");

    const int pupilSize = GetLensDiffEffectivePupilResolution(params.pupilResolution);
    const int maxKernelRadiusPx = std::max(4, ResolveLensDiffMaxKernelRadiusPx(params));
    const int rawPsfSize = ChooseLensDiffRawPsfSize(pupilSize, maxKernelRadiusPx);
    DeviceBuffer<float> devicePupil;
    DeviceBuffer<float> devicePhase;
    bool hasPhase = false;
    if (!buildPupilAmplitudeOnCuda(params, pupilSize, stream, &devicePupil, error) ||
        !buildPhaseWavesOnCuda(params, pupilSize, stream, &devicePhase, &hasPhase, error)) {
        return false;
    }

    CudaPsfBuildContext localBuildContext {};
    CudaPsfBuildContext* psfContext = buildContext != nullptr ? buildContext : &localBuildContext;
    if (!buildShiftedRawPsfOnCuda(devicePupil,
                                  hasPhase ? &devicePhase : nullptr,
                                  pupilSize,
                                  rawPsfSize,
                                  fftPolicy,
                                  requestedFftBackend,
                                  vkfftStrict,
                                  stream,
                                  psfContext,
                                  &psfContext->rawPsf,
                                  nullptr,
                                  timing,
                                  error)) {
        return false;
    }

    std::vector<float> pupil;
    if (!downloadFloatBuffer(devicePupil,
                             static_cast<std::size_t>(pupilSize) * pupilSize,
                             stream,
                             &pupil,
                             "cudaMemcpyAsync-download-pupil-display",
                             error)) {
        return false;
    }
    std::vector<float> phaseWaves;
    if (hasPhase) {
        if (!downloadFloatBuffer(devicePhase,
                                 static_cast<std::size_t>(pupilSize) * pupilSize,
                                 stream,
                                 &phaseWaves,
                                 "cudaMemcpyAsync-download-phase-display",
                                 error)) {
            return false;
        }
    } else {
        phaseWaves.assign(static_cast<std::size_t>(pupilSize) * pupilSize, 0.0f);
    }

    if (phaseWaves.size() != pupil.size()) {
        phaseWaves.assign(pupil.size(), 0.0f);
    }

    const std::shared_ptr<const std::vector<float>> referenceRawPsf = GetLensDiffReferenceRawPsfCached(pupilSize, rawPsfSize);
    const float referenceFirstZeroRadius = std::max(1.0f, estimateFirstMinimumRadiusCuda(*referenceRawPsf, rawPsfSize));
    const float scaleBase = static_cast<float>(std::max(1.0, ResolveLensDiffDiffractionScalePx(params))) / referenceFirstZeroRadius;
    const std::vector<float> wavelengths = wavelengthsForModeCuda(params.spectralMode);

    return finalizePsfBankOnCuda(params,
                                 key,
                                 pupil,
                                 phaseWaves,
                                 pupilSize,
                                 psfContext->rawPsf,
                                 rawPsfSize,
                                 wavelengths,
                                 scaleBase,
                                 stream,
                                 &cache,
                                 psfContext,
                                  error);
}

bool buildFieldZoneCachesBatchedCuda(const LensDiffParams& params,
                                     LensDiffPsfBankCache* cache,
                                     CudaPsfBuildContext* buildContext,
                                     CudaFftPolicy fftPolicy,
                                     CudaFftBackend requestedFftBackend,
                                     bool vkfftStrict,
                                     CudaRenderTimingBreakdown* timing,
                                     cudaStream_t stream,
                                     std::string* error) {
    if (cache == nullptr || buildContext == nullptr) {
        if (error) *error = "cuda-null-field-zone-batch-cache";
        return false;
    }
    const int pupilSize = GetLensDiffEffectivePupilResolution(params.pupilResolution);
    const int maxKernelRadiusPx = std::max(4, ResolveLensDiffMaxKernelRadiusPx(params));
    const int rawPsfSize = ChooseLensDiffRawPsfSize(pupilSize, maxKernelRadiusPx);
    const std::size_t rawCount = static_cast<std::size_t>(rawPsfSize) * rawPsfSize;
    constexpr int zoneCount = 9;

    struct ZonePrep {
        LensDiffFieldZoneCache zone;
        std::vector<float> pupil;
        std::vector<float> phase;
    };
    std::array<ZonePrep, zoneCount> zonePrep {};
    DeviceBuffer<cufftComplex> batchedSpectrum;
    DeviceBuffer<float> batchedShiftedIntensity;
    if (!batchedSpectrum.allocate(rawCount * zoneCount) ||
        !batchedShiftedIntensity.allocate(rawCount * zoneCount)) {
        if (error) *error = "cuda-alloc-field-zone-batch";
        return false;
    }
    if (!checkCuda(cudaMemsetAsync(batchedSpectrum.ptr,
                                   0,
                                   rawCount * zoneCount * sizeof(cufftComplex),
                                   stream),
                   "cudaMemsetAsync-field-zone-batch-spectrum",
                   error)) {
        return false;
    }

    const int offset = std::max(0, (rawPsfSize - pupilSize) / 2);
    dim3 block2d(16, 16);
    dim3 pupilGrid((pupilSize + block2d.x - 1) / block2d.x, (pupilSize + block2d.y - 1) / block2d.y);
    for (int zoneY = 0; zoneY < 3; ++zoneY) {
        for (int zoneX = 0; zoneX < 3; ++zoneX) {
            const int zoneIndex = zoneY * 3 + zoneX;
            const float normalizedX = static_cast<float>(zoneX - 1);
            const float normalizedY = static_cast<float>(zoneY - 1);
            ZonePrep& prep = zonePrep[static_cast<std::size_t>(zoneIndex)];
            prep.zone.zoneX = zoneX;
            prep.zone.zoneY = zoneY;
            prep.zone.normalizedX = normalizedX;
            prep.zone.normalizedY = normalizedY;
            prep.zone.radialNorm = std::min(1.0f, std::sqrt(normalizedX * normalizedX + normalizedY * normalizedY) / std::sqrt(2.0f));
            prep.zone.resolvedParams = ResolveLensDiffFieldZoneParams(params, normalizedX, normalizedY);

            DeviceBuffer<float> devicePupil;
            DeviceBuffer<float> devicePhase;
            bool hasPhase = false;
            if (!buildPupilAmplitudeOnCuda(prep.zone.resolvedParams, pupilSize, stream, &devicePupil, error) ||
                !buildPhaseWavesOnCuda(prep.zone.resolvedParams, pupilSize, stream, &devicePhase, &hasPhase, error) ||
                !downloadFloatBuffer(devicePupil,
                                     static_cast<std::size_t>(pupilSize) * pupilSize,
                                     stream,
                                     &prep.pupil,
                                     "cudaMemcpyAsync-download-field-zone-pupil",
                                     error)) {
                return false;
            }
            if (hasPhase) {
                if (!downloadFloatBuffer(devicePhase,
                                         static_cast<std::size_t>(pupilSize) * pupilSize,
                                         stream,
                                         &prep.phase,
                                         "cudaMemcpyAsync-download-field-zone-phase",
                                         error)) {
                    return false;
                }
            } else {
                prep.phase.assign(static_cast<std::size_t>(pupilSize) * pupilSize, 0.0f);
            }

            embedCenteredComplexPupilKernel<<<pupilGrid, block2d, 0, stream>>>(devicePupil.ptr,
                                                                               hasPhase ? devicePhase.ptr : nullptr,
                                                                               pupilSize,
                                                                               rawPsfSize,
                                                                               offset,
                                                                               batchedSpectrum.ptr + rawCount * static_cast<std::size_t>(zoneIndex));
            if (!checkCuda(cudaGetLastError(), "embedCenteredComplexPupilKernel-field-zone-batch", error)) {
                return false;
            }
        }
    }

    CudaFftPlanLease batchPlan;
    const CudaFftBackend psfFftBackend =
        resolveCudaFftBackendForSubsystem(CudaFftSubsystem::PsfBank, requestedFftBackend, fftPolicy, timing);
    if (!acquireFftPlan(CudaFftSubsystem::PsfBank,
                        psfFftBackend,
                        vkfftStrict,
                        nullptr,
                        usePersistentVkfftRepositoryForSubsystem(CudaFftSubsystem::PsfBank, fftPolicy, false),
                        rawPsfSize,
                        rawPsfSize,
                        zoneCount,
                        stream,
                        timing,
                        &batchPlan,
                        error)) {
        return false;
    }
    const bool fftOk = execFftC2C(&batchPlan, CudaFftSubsystem::PsfBank, batchedSpectrum, false, rawPsfSize, rawPsfSize, stream, timing, error);
    if (!fftOk) {
        return false;
    }

    dim3 rawGrid((rawPsfSize + block2d.x - 1) / block2d.x, (rawPsfSize + block2d.y - 1) / block2d.y);
    for (int zoneIndex = 0; zoneIndex < zoneCount; ++zoneIndex) {
        extractShiftedIntensityKernel<<<rawGrid, block2d, 0, stream>>>(
            batchedSpectrum.ptr + rawCount * static_cast<std::size_t>(zoneIndex),
            rawPsfSize,
            batchedShiftedIntensity.ptr + rawCount * static_cast<std::size_t>(zoneIndex));
        if (!checkCuda(cudaGetLastError(), "extractShiftedIntensityKernel-field-zone-batch", error)) {
            return false;
        }
    }

    const std::shared_ptr<const std::vector<float>> referenceRawPsf = GetLensDiffReferenceRawPsfCached(pupilSize, rawPsfSize);
    const float referenceFirstZeroRadius = std::max(1.0f, estimateFirstMinimumRadiusCuda(*referenceRawPsf, rawPsfSize));
    cache->fieldGridSize = 3;
    cache->fieldKey = MakeLensDiffFieldKey(params);
    cache->fieldZones.clear();
    cache->fieldZones.reserve(zoneCount);
    for (int zoneIndex = 0; zoneIndex < zoneCount; ++zoneIndex) {
        ZonePrep& prep = zonePrep[static_cast<std::size_t>(zoneIndex)];
        if (!buildContext->rawPsf.allocate(rawCount) ||
            !checkCuda(cudaMemcpyAsync(buildContext->rawPsf.ptr,
                                       batchedShiftedIntensity.ptr + rawCount * static_cast<std::size_t>(zoneIndex),
                                       rawCount * sizeof(float),
                                       cudaMemcpyDeviceToDevice,
                                       stream),
                       "cudaMemcpyAsync-field-zone-raw-psf-slice",
                       error)) {
            return false;
        }
        LensDiffPsfBankCache zoneCache {};
        const float scaleBase = static_cast<float>(std::max(1.0, ResolveLensDiffDiffractionScalePx(prep.zone.resolvedParams))) / referenceFirstZeroRadius;
        const std::vector<float> wavelengths = wavelengthsForModeCuda(prep.zone.resolvedParams.spectralMode);
        const LensDiffPsfBankKey key = MakeLensDiffPsfBankKey(prep.zone.resolvedParams);
        if (!finalizePsfBankOnCuda(prep.zone.resolvedParams,
                                   key,
                                   prep.pupil,
                                   prep.phase,
                                   pupilSize,
                                   buildContext->rawPsf,
                                   rawPsfSize,
                                   wavelengths,
                                   scaleBase,
                                   stream,
                                   &zoneCache,
                                   buildContext,
                                   error)) {
            return false;
        }
        prep.zone.key = zoneCache.key;
        prep.zone.bins = std::move(zoneCache.bins);
        prep.zone.supportRadiusPx = zoneCache.supportRadiusPx;
        prep.zone.pupilDisplaySize = zoneCache.pupilDisplaySize;
        prep.zone.pupilDisplay = std::move(zoneCache.pupilDisplay);
        prep.zone.phaseDisplaySize = zoneCache.phaseDisplaySize;
        prep.zone.phaseDisplay = std::move(zoneCache.phaseDisplay);
        cache->fieldZones.push_back(std::move(prep.zone));
    }
    return true;
}

bool buildFieldZoneCachesLegacyCuda(const LensDiffParams& params,
                                    LensDiffPsfBankCache* cache,
                                    CudaFftPolicy fftPolicy,
                                    CudaFftBackend requestedFftBackend,
                                    bool vkfftStrict,
                                    CudaRenderTimingBreakdown* timing,
                                    cudaStream_t stream,
                                    std::string* error) {
    if (cache == nullptr) {
        if (error) *error = "cuda-null-field-zone-legacy-cache";
        return false;
    }
    const LensDiffFieldKey fieldKey = MakeLensDiffFieldKey(params);
    cache->fieldGridSize = 3;
    cache->fieldKey = fieldKey;
    cache->fieldZones.clear();
    cache->fieldZones.reserve(9);
    for (int zoneY = 0; zoneY < 3; ++zoneY) {
        for (int zoneX = 0; zoneX < 3; ++zoneX) {
            const float normalizedX = static_cast<float>(zoneX - 1);
            const float normalizedY = static_cast<float>(zoneY - 1);
            LensDiffFieldZoneCache zone {};
            zone.zoneX = zoneX;
            zone.zoneY = zoneY;
            zone.normalizedX = normalizedX;
            zone.normalizedY = normalizedY;
            zone.radialNorm = std::min(1.0f,
                                       std::sqrt(normalizedX * normalizedX + normalizedY * normalizedY) /
                                           std::sqrt(2.0f));
            zone.resolvedParams = ResolveLensDiffFieldZoneParams(params, normalizedX, normalizedY);

            LensDiffPsfBankCache zoneCache {};
            CudaPsfBuildContext zoneBuildContext {};
            if (!buildPsfBankGlobalOnlyCuda(zone.resolvedParams,
                                           zoneCache,
                                           &zoneBuildContext,
                                           fftPolicy,
                                           requestedFftBackend,
                                           vkfftStrict,
                                           timing,
                                           stream,
                                           error)) {
                return false;
            }
            zone.key = zoneCache.key;
            zone.bins = std::move(zoneCache.bins);
            zone.supportRadiusPx = zoneCache.supportRadiusPx;
            zone.pupilDisplaySize = zoneCache.pupilDisplaySize;
            zone.pupilDisplay = std::move(zoneCache.pupilDisplay);
            zone.phaseDisplaySize = zoneCache.phaseDisplaySize;
            zone.phaseDisplay = std::move(zoneCache.phaseDisplay);
            cache->fieldZones.push_back(std::move(zone));
        }
    }
    return true;
}

bool ensurePsfBankCuda(const LensDiffParams& params,
                       LensDiffPsfBankCache& cache,
                       CudaFftPolicy fftPolicy,
                       CudaFftBackend requestedFftBackend,
                       bool vkfftStrict,
                       CudaRenderTimingBreakdown* timing,
                       cudaStream_t stream,
                       std::string* error) {
    const LensDiffPsfBankKey key = MakeLensDiffPsfBankKey(params);
    const bool needFieldZones = HasLensDiffFieldPhase(params);
    const LensDiffFieldKey fieldKey = MakeLensDiffFieldKey(params);
    const bool canReuseFieldZones = needFieldZones &&
                                    cache.fieldGridSize == 3 &&
                                    cache.fieldZones.size() == 9U &&
                                    cache.fieldKey == fieldKey;
    if (cache.valid && cache.key == key &&
        ((!needFieldZones && cache.fieldZones.empty()) || canReuseFieldZones)) {
        return true;
    }

    CudaPsfBuildContext buildContext {};
    if (!buildPsfBankGlobalOnlyCuda(params,
                                    cache,
                                    &buildContext,
                                    fftPolicy,
                                    requestedFftBackend,
                                    vkfftStrict,
                                    timing,
                                    stream,
                                    error)) {
        return false;
    }
    if (!needFieldZones) {
        cache.fieldGridSize = 0;
        cache.fieldKey = {};
        cache.fieldZones.clear();
        return true;
    }

    LensDiffScopedTimer timer("cuda-field-zones");
    if (LensDiffCudaBatchedFieldCacheEnabled()) {
        if (!buildFieldZoneCachesBatchedCuda(params,
                                             &cache,
                                             &buildContext,
                                             fftPolicy,
                                             requestedFftBackend,
                                             vkfftStrict,
                                             timing,
                                             stream,
                                             error)) {
            return false;
        }
    } else if (!buildFieldZoneCachesLegacyCuda(params,
                                               &cache,
                                               fftPolicy,
                                               requestedFftBackend,
                                               vkfftStrict,
                                               timing,
                                               stream,
                                               error)) {
        return false;
    }
    return true;
}

} // namespace

bool RunLensDiffCuda(const LensDiffRenderRequest& request,
                     const LensDiffParams& params,
                     LensDiffPsfBankCache& cache,
                     std::string* error) {
    if (!request.hostEnabledCudaRender || request.cudaStream == nullptr) {
        if (error) {
            *error = "cuda-render-not-enabled-by-host";
        }
        return false;
    }
    if (request.src.data == nullptr || request.dst.data == nullptr) {
        if (error) {
            *error = "missing-source-or-destination-cuda-buffer";
        }
        return false;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(request.cudaStream);
    const bool legacyPipeline = LensDiffCudaLegacyPipelineEnabled();
    const bool validateField = LensDiffCudaFieldValidateEnabled();
    const bool useFrameWeightSpace = LensDiffCudaUseFrameWeightSpace();
    const bool stackedFieldEnabled = LensDiffCudaExperimentalStackedFieldEnabled();
    const CudaFftPolicy fftPolicy = LensDiffCudaFftPolicy();
    const CudaFftBackend requestedFftBackend = LensDiffCudaRequestedFftBackend();
    const bool vkfftStrict = LensDiffCudaVkFFTStrictEnabled();
    const CudaFftBackend globalRequestedFftBackend =
        resolveCudaFftBackendForSubsystem(CudaFftSubsystem::GlobalRender, requestedFftBackend, fftPolicy, nullptr);
    const CudaFftBackend psfRequestedFftBackend =
        resolveCudaFftBackendForSubsystem(CudaFftSubsystem::PsfBank, requestedFftBackend, fftPolicy, nullptr);
    const CudaFftBackend fieldRequestedFftBackend =
        resolveCudaFftBackendForSubsystem(CudaFftSubsystem::FieldRender, requestedFftBackend, fftPolicy, nullptr);
    const bool nonFieldExecution = !HasLensDiffFieldPhase(params);
    const bool persistentKernelCacheEnabled = nonFieldExecution || LensDiffCudaPersistentKernelCacheEnabled();
    const bool persistentPlanRepositoryEnabled = nonFieldExecution || LensDiffCudaPersistentPlanRepositoryEnabled();
    CudaRenderTimingBreakdown timing {};
    PersistentCudaPlanRepository* planRepository = persistentPlanRepositoryEnabled ? &persistentCudaPlanRepository() : nullptr;
    timing.validationEnabled = validateField;
    timing.fieldWeightSpace = useFrameWeightSpace ? "frame" : "working";
    timing.fftPolicy = cudaFftPolicyName(fftPolicy);
    timing.fftRequested = cudaFftBackendName(requestedFftBackend);
    timing.fftEffective.clear();
    timing.globalFft.requested = cudaFftBackendName(globalRequestedFftBackend);
    timing.psfFft.requested = cudaFftBackendName(psfRequestedFftBackend);
    timing.fieldFft.requested = cudaFftBackendName(fieldRequestedFftBackend);
    auto timeCall = [&](double& accumulator, auto&& fn) {
        const auto start = std::chrono::steady_clock::now();
        auto result = fn();
        accumulator += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                           std::chrono::steady_clock::now() - start)
                           .count();
        return result;
    };
    auto logTimingBreakdown = [&]() {
        if (!LensDiffTimingEnabled()) {
            return;
        }
        LogLensDiffTimingStage(
            "cuda-stage-psf-bank",
            timing.psfBankMs,
            "legacy=" + std::to_string(legacyPipeline ? 1 : 0) +
                ",policy=" + timing.fftPolicy +
                ",requested=" + timing.psfFft.requested +
                ",effective=" + timing.psfFft.effective +
                ",cufftPlanHits=" + std::to_string(timing.psfFft.cufftPlanCacheHits) +
                ",cufftPlanMisses=" + std::to_string(timing.psfFft.cufftPlanCacheMisses) +
                ",vkfftPlanHits=" + std::to_string(timing.psfFft.vkfftPlanCacheHits) +
                ",vkfftPlanMisses=" + std::to_string(timing.psfFft.vkfftPlanCacheMisses) +
                ",vkfftInitMs=" + std::to_string(timing.psfFft.vkfftCompileInitMs) +
                ",maxWorkBytes=" + std::to_string(static_cast<unsigned long long>(timing.psfFft.maxPlanWorkBytes)) +
                (timing.psfFft.fallback.empty() ? std::string() : ",fallback=" + timing.psfFft.fallback));
        LogLensDiffTimingStage(
            "cuda-stage-source-fft",
            timing.sourceFftMs,
            "rgbHits=" + std::to_string(timing.rgbSourceCacheHits) +
                ",rgbMisses=" + std::to_string(timing.rgbSourceCacheMisses) +
                ",scalarHits=" + std::to_string(timing.scalarSourceCacheHits) +
                ",scalarMisses=" + std::to_string(timing.scalarSourceCacheMisses));
        LogLensDiffTimingStage(
            "cuda-stage-kernel-fft",
            timing.kernelFftMs,
            "hits=" + std::to_string(timing.kernelCacheHits) +
                ",misses=" + std::to_string(timing.kernelCacheMisses) +
                ",cufftPlanHits=" + std::to_string(timing.cufftPlanCacheHits) +
                ",cufftPlanMisses=" + std::to_string(timing.cufftPlanCacheMisses) +
                ",vkfftPlanHits=" + std::to_string(timing.vkfftPlanCacheHits) +
                ",vkfftPlanMisses=" + std::to_string(timing.vkfftPlanCacheMisses) +
                ",vkfftInitMs=" + std::to_string(timing.vkfftCompileInitMs) +
                ",maxWorkBytes=" + std::to_string(static_cast<unsigned long long>(timing.maxPlanWorkBytes)) +
                ",policy=" + timing.fftPolicy +
                ",requested=" + timing.globalFft.requested +
                ",effective=" + timing.globalFft.effective +
                (timing.globalFft.fallback.empty() ? std::string() : ",fallback=" + timing.globalFft.fallback));
        LogLensDiffTimingStage("cuda-stage-convolution", timing.convolutionMs);
        LogLensDiffTimingStage(
            "cuda-stage-field-zones",
            timing.fieldZonesMs,
            "legacy=" + std::to_string(legacyPipeline ? 1 : 0) +
                ",batchDepth=" + std::to_string(timing.fieldZoneBatchDepth) +
                ",hostSyncs=" + std::to_string(timing.hostSyncCount) +
                ",branch=" + timing.fieldBranch +
                ",weightSpace=" + timing.fieldWeightSpace +
                ",policy=" + timing.fftPolicy +
                ",requested=" + timing.fieldFft.requested +
                ",effective=" + timing.fieldFft.effective +
                ",cufftPlanHits=" + std::to_string(timing.fieldFft.cufftPlanCacheHits) +
                ",cufftPlanMisses=" + std::to_string(timing.fieldFft.cufftPlanCacheMisses) +
                ",vkfftPlanHits=" + std::to_string(timing.fieldFft.vkfftPlanCacheHits) +
                ",vkfftPlanMisses=" + std::to_string(timing.fieldFft.vkfftPlanCacheMisses) +
                ",vkfftInitMs=" + std::to_string(timing.fieldFft.vkfftCompileInitMs) +
                ",maxWorkBytes=" + std::to_string(static_cast<unsigned long long>(timing.fieldFft.maxPlanWorkBytes)) +
                (timing.fieldFft.fallback.empty() ? std::string() : ",fallback=" + timing.fieldFft.fallback) +
                ",fieldKeyHash=" + std::to_string(static_cast<unsigned long long>(timing.fieldKeyHash)) +
                ",psfKeyHash=" + std::to_string(static_cast<unsigned long long>(timing.psfKeyHash)) +
                ",scratchEstimate=" + std::to_string(static_cast<unsigned long long>(timing.fieldScratchEstimateBytes)) +
                ",frameBounds=(" + std::to_string(request.frameBounds.x1) + "," + std::to_string(request.frameBounds.y1) +
                    ")-(" + std::to_string(request.frameBounds.x2) + "," + std::to_string(request.frameBounds.y2) + ")");
        LogLensDiffTimingStage("cuda-stage-scatter", timing.scatterMs);
        LogLensDiffTimingStage("cuda-stage-creative-fringe", timing.creativeFringeMs);
        LogLensDiffTimingStage("cuda-stage-native-resample", timing.nativeResampleMs);
        LogLensDiffTimingStage("cuda-stage-composite", timing.compositeMs);
        LogLensDiffTimingStage("cuda-stage-output-copy", timing.outputCopyMs);
        if (timing.validationEnabled && timing.validationRan) {
            LogLensDiffTimingStage(
                "cuda-stage-field-validation",
                timing.validationMs,
                "referenceLegacy=" + std::to_string(timing.validationReferenceLegacy ? 1 : 0) +
                    ",effectMaxAbs=" + std::to_string(timing.validationEffectMaxAbs) +
                    ",coreMaxAbs=" + std::to_string(timing.validationCoreMaxAbs) +
                    ",structureMaxAbs=" + std::to_string(timing.validationStructureMaxAbs) +
                    (timing.validationNote.empty() ? std::string() : ("," + timing.validationNote)));
        }
    };
    if (!timeCall(timing.psfBankMs, [&] {
            return ensurePsfBankCuda(params, cache, fftPolicy, requestedFftBackend, vkfftStrict, &timing, stream, error);
        })) {
        return false;
    }
    const LensDiffSpectrumConfig spectrumConfig = BuildLensDiffSpectrumConfig(params, cache.bins);

    const int nativeWidth = request.src.bounds.width();
    const int nativeHeight = request.src.bounds.height();
    const std::size_t nativePixelCount = static_cast<std::size_t>(nativeWidth) * nativeHeight;
    const double workingScale = ResolveLensDiffEffectWorkingScale(params);
    const bool resolutionAwareActive = params.resolutionAware && std::abs(workingScale - 1.0) > 1e-6;
    const int width = resolutionAwareActive ? std::max(1, static_cast<int>(std::lround(nativeWidth * workingScale))) : nativeWidth;
    const int height = resolutionAwareActive ? std::max(1, static_cast<int>(std::lround(nativeHeight * workingScale))) : nativeHeight;
    const std::size_t pixelCount = static_cast<std::size_t>(width) * height;
    const std::size_t srcRowFloats = static_cast<std::size_t>(request.src.rowBytes) / sizeof(float);
    const int inputTransfer = static_cast<int>(params.inputTransfer);
    const LensDiffImageRect fieldFrameBounds =
        useFrameWeightSpace
            ? ((request.frameBounds.width() > 0 && request.frameBounds.height() > 0) ? request.frameBounds : request.src.bounds)
            : LensDiffImageRect {0, 0, width, height};
    const int fieldTileOriginX = useFrameWeightSpace ? request.src.bounds.x1 : 0;
    const int fieldTileOriginY = useFrameWeightSpace ? request.src.bounds.y1 : 0;
    const int fieldFrameX1 = fieldFrameBounds.x1;
    const int fieldFrameY1 = fieldFrameBounds.y1;
    const int fieldFrameWidth = std::max(1, fieldFrameBounds.width());
    const int fieldFrameHeight = std::max(1, fieldFrameBounds.height());
    timing.fieldKeyHash = diagnosticFieldKeyHash(cache.fieldKey);
    timing.psfKeyHash = diagnosticPsfKeyHash(cache.key);

    const bool staticDebug = params.debugView == LensDiffDebugView::Pupil ||
                             params.debugView == LensDiffDebugView::Psf ||
                             params.debugView == LensDiffDebugView::Otf ||
                             params.debugView == LensDiffDebugView::Phase ||
                             params.debugView == LensDiffDebugView::PhaseEdge ||
                             params.debugView == LensDiffDebugView::FieldPsf ||
                             params.debugView == LensDiffDebugView::ChromaticSplit;
    if (staticDebug) {
        PackedHostImage debugImage;
        if (!createStaticDebugImage(params, cache, nativeWidth, nativeHeight, &debugImage)) {
            if (error) *error = "failed-to-build-static-debug-image";
            return false;
        }
        return copyPackedToDestination(request, debugImage.pixels.data(), debugImage.width, debugImage.height, stream, cudaMemcpyHostToDevice, error);
    }

    DeviceBuffer<float> nativeSrcR;
    DeviceBuffer<float> nativeSrcG;
    DeviceBuffer<float> nativeSrcB;
    DeviceBuffer<float> nativeSrcA;
    DeviceBuffer<float> srcR;
    DeviceBuffer<float> srcG;
    DeviceBuffer<float> srcB;
    DeviceBuffer<float> selectionMask;
    DeviceBuffer<float> driver;
    DeviceBuffer<float> redistributedR;
    DeviceBuffer<float> redistributedG;
    DeviceBuffer<float> redistributedB;
    DeviceBuffer<float> redistributedDriver;
    DeviceBuffer<float> packedOutput;

    if (!nativeSrcR.allocate(nativePixelCount) || !nativeSrcG.allocate(nativePixelCount) ||
        !nativeSrcB.allocate(nativePixelCount) || !nativeSrcA.allocate(nativePixelCount) ||
        !packedOutput.allocate(nativePixelCount * 4U) ||
        (resolutionAwareActive && (!srcR.allocate(pixelCount) || !srcG.allocate(pixelCount) || !srcB.allocate(pixelCount))) ||
        !selectionMask.allocate(pixelCount) || !driver.allocate(pixelCount) ||
        !redistributedR.allocate(pixelCount) || !redistributedG.allocate(pixelCount) || !redistributedB.allocate(pixelCount) ||
        !redistributedDriver.allocate(pixelCount)) {
        if (error) *error = "cuda-alloc-base-images";
        return false;
    }

    dim3 block2d(16, 16);
    dim3 grid2d((width + block2d.x - 1) / block2d.x, (height + block2d.y - 1) / block2d.y);
    dim3 nativeGrid2d((nativeWidth + block2d.x - 1) / block2d.x, (nativeHeight + block2d.y - 1) / block2d.y);
    const float* srcDevice = static_cast<const float*>(request.src.data);

    packDecodeKernel<<<nativeGrid2d, block2d, 0, stream>>>(srcDevice,
                                                            srcRowFloats,
                                                            nativeWidth,
                                                            nativeHeight,
                                                            inputTransfer,
                                                            nativeSrcR.ptr,
                                                            nativeSrcG.ptr,
                                                            nativeSrcB.ptr,
                                                            nativeSrcA.ptr);
    if (!checkCuda(cudaGetLastError(), "packDecodeKernel", error)) {
        return false;
    }

    const DeviceBuffer<float>* effectSrcR = &nativeSrcR;
    const DeviceBuffer<float>* effectSrcG = &nativeSrcG;
    const DeviceBuffer<float>* effectSrcB = &nativeSrcB;
    if (resolutionAwareActive) {
        resamplePlaneKernel<<<grid2d, block2d, 0, stream>>>(nativeSrcR.ptr, nativeWidth, nativeHeight, srcR.ptr, width, height);
        resamplePlaneKernel<<<grid2d, block2d, 0, stream>>>(nativeSrcG.ptr, nativeWidth, nativeHeight, srcG.ptr, width, height);
        resamplePlaneKernel<<<grid2d, block2d, 0, stream>>>(nativeSrcB.ptr, nativeWidth, nativeHeight, srcB.ptr, width, height);
        if (!checkCuda(cudaGetLastError(), "resamplePlaneKernel-src", error)) {
            return false;
        }
        effectSrcR = &srcR;
        effectSrcG = &srcG;
        effectSrcB = &srcB;
    }

    buildSelectionKernel<<<grid2d, block2d, 0, stream>>>(effectSrcR->ptr,
                                                          effectSrcG->ptr,
                                                          effectSrcB->ptr,
                                                          width,
                                                          height,
                                                          params.extractionMode == LensDiffExtractionMode::Luma ? 1 : 0,
                                                          static_cast<float>(params.threshold),
                                                          static_cast<float>(std::max(0.01, params.softnessStops)),
                                                          static_cast<float>(std::max(0.0, params.pointEmphasis)),
                                                          selectionMask.ptr,
                                                          driver.ptr);
    if (!checkCuda(cudaGetLastError(), "buildSelectionKernel", error)) {
        return false;
    }

    const int flatBlock = 256;
    const int flatGrid = static_cast<int>((pixelCount + flatBlock - 1) / flatBlock);
    const int nativeFlatGrid = static_cast<int>((nativePixelCount + flatBlock - 1) / flatBlock);
    if (params.debugView == LensDiffDebugView::Selection) {
        if (resolutionAwareActive) {
            DeviceBuffer<float> selectionMaskDisplay;
            if (!selectionMaskDisplay.allocate(nativePixelCount)) {
                if (error) *error = "cuda-alloc-selection-display";
                return false;
            }
            resamplePlaneKernel<<<nativeGrid2d, block2d, 0, stream>>>(selectionMask.ptr,
                                                                       width,
                                                                       height,
                                                                       selectionMaskDisplay.ptr,
                                                                       nativeWidth,
                                                                       nativeHeight);
            if (!checkCuda(cudaGetLastError(), "resamplePlaneKernel-selection-display", error)) {
                return false;
            }
            packGrayDebugKernel<<<nativeFlatGrid, flatBlock, 0, stream>>>(selectionMaskDisplay.ptr, nativePixelCount, packedOutput.ptr);
        } else {
            packGrayDebugKernel<<<flatGrid, flatBlock, 0, stream>>>(selectionMask.ptr, pixelCount, packedOutput.ptr);
        }
        if (!checkCuda(cudaGetLastError(), "packGrayDebugKernel-selection", error)) {
            return false;
        }
        return copyPackedToDestination(request, packedOutput.ptr, nativeWidth, nativeHeight, stream, cudaMemcpyDeviceToDevice, error);
    }

    const float redistributionScale = 1.0f - static_cast<float>(std::clamp(params.corePreserve, 0.0, 1.0));
    applyMaskRgbKernel<<<grid2d, block2d, 0, stream>>>(effectSrcR->ptr, effectSrcG->ptr, effectSrcB->ptr, selectionMask.ptr, width, height, redistributionScale,
                                                        redistributedR.ptr, redistributedG.ptr, redistributedB.ptr);
    if (!checkCuda(cudaGetLastError(), "applyMaskRgbKernel", error)) {
        return false;
    }
    applyMaskScalarKernel<<<grid2d, block2d, 0, stream>>>(driver.ptr, selectionMask.ptr, width, height, redistributionScale, redistributedDriver.ptr);
    if (!checkCuda(cudaGetLastError(), "applyMaskScalarKernel", error)) {
        return false;
    }

    const bool splitMode = params.lookMode == LensDiffLookMode::Split;
    const bool needCore = splitMode || params.debugView == LensDiffDebugView::Core;
    const bool needStructure = splitMode || params.debugView == LensDiffDebugView::Structure;
    auto allocatePlaneSetCount = [&](PlaneSet& set, std::size_t count) -> bool {
        return set.r.allocate(count) && set.g.allocate(count) && set.b.allocate(count);
    };
    auto allocatePlaneSet = [&](PlaneSet& set) -> bool {
        return allocatePlaneSetCount(set, pixelCount);
    };
    auto clearPlaneSet = [&](PlaneSet& set, const char* stage) -> bool {
        return checkCuda(cudaMemsetAsync(set.r.ptr, 0, pixelCount * sizeof(float), stream), stage, error) &&
               checkCuda(cudaMemsetAsync(set.g.ptr, 0, pixelCount * sizeof(float), stream), stage, error) &&
               checkCuda(cudaMemsetAsync(set.b.ptr, 0, pixelCount * sizeof(float), stream), stage, error);
    };
    auto copyPlaneSet = [&](const PlaneSet& src, PlaneSet& dst, const char* stage) -> bool {
        return checkCuda(cudaMemcpyAsync(dst.r.ptr, src.r.ptr, pixelCount * sizeof(float), cudaMemcpyDeviceToDevice, stream), stage, error) &&
               checkCuda(cudaMemcpyAsync(dst.g.ptr, src.g.ptr, pixelCount * sizeof(float), cudaMemcpyDeviceToDevice, stream), stage, error) &&
               checkCuda(cudaMemcpyAsync(dst.b.ptr, src.b.ptr, pixelCount * sizeof(float), cudaMemcpyDeviceToDevice, stream), stage, error);
    };
    auto allocateNativePlaneSet = [&](PlaneSet& set) -> bool {
        return set.r.allocate(nativePixelCount) && set.g.allocate(nativePixelCount) && set.b.allocate(nativePixelCount);
    };
    auto resamplePlaneToNative = [&](const DeviceBuffer<float>& srcPlane,
                                     DeviceBuffer<float>& dstPlane,
                                     const char* stage) -> bool {
        if (!dstPlane.allocate(nativePixelCount)) {
            if (error) *error = stage;
            return false;
        }
        resamplePlaneKernel<<<nativeGrid2d, block2d, 0, stream>>>(srcPlane.ptr, width, height, dstPlane.ptr, nativeWidth, nativeHeight);
        return checkCuda(cudaGetLastError(), stage, error);
    };
    auto resamplePlaneSetToNative = [&](const PlaneSet& srcSet,
                                        PlaneSet& dstSet,
                                        const char* stage) -> bool {
        if (!allocateNativePlaneSet(dstSet)) {
            if (error) *error = stage;
            return false;
        }
        return resamplePlaneToNative(srcSet.r, dstSet.r, stage) &&
               resamplePlaneToNative(srcSet.g, dstSet.g, stage) &&
               resamplePlaneToNative(srcSet.b, dstSet.b, stage);
    };
    std::unordered_map<std::string, DeviceBuffer<cufftComplex>> rgbSourceSpectrumCache;
    std::unordered_map<std::string, DeviceBuffer<cufftComplex>> scalarSourceSpectrumCache;
    std::unordered_map<std::string, std::shared_ptr<DeviceBuffer<cufftComplex>>> kernelSpectrumCache;
    std::unordered_map<std::string, FieldZoneSpectrumStacks> fieldKernelStackCache;
    std::unordered_map<std::string, DeviceBuffer<cufftComplex>> fieldReplicatedSpectrumCache;
    std::unordered_map<std::string, DeviceBuffer<cufftComplex>> complexScratchCache;
    std::unordered_map<std::string, DeviceBuffer<float>> floatScratchCache;
    std::unordered_map<std::string, PlaneSet> planeSetScratchCache;
    std::unordered_map<std::string, SpectralPlaneSet> spectralPlaneScratchCache;
    std::unordered_map<std::string, DeviceBuffer<cufftComplex>> fieldRgbTripletSpectrumCache;
    const FieldZoneBatchPlan fieldPlan = buildFieldZoneBatchPlan(cache);
    const std::string globalCacheNamespace = cudaFftCacheNamespace(CudaFftSubsystem::GlobalRender, globalRequestedFftBackend);
    const std::string fieldCacheNamespace = cudaFftCacheNamespace(CudaFftSubsystem::FieldRender, fieldRequestedFftBackend);
    auto sourceSpectrumCacheKey = [&](const std::string& cacheNamespace,
                                      const void* ptr,
                                      int paddedWidth,
                                      int paddedHeight,
                                      int channels) {
        return cacheNamespace + ":" +
               std::to_string(reinterpret_cast<std::uintptr_t>(ptr)) + ":" +
               std::to_string(paddedWidth) + "x" + std::to_string(paddedHeight) +
               ":" + std::to_string(channels);
    };
    auto windowSourceSpectrumCacheKey = [&](const std::string& cacheNamespace,
                                            const void* ptr,
                                            int windowX,
                                            int windowY,
                                            int windowWidth,
                                            int windowHeight,
                                            int paddedWidth,
                                            int paddedHeight,
                                            int channels) {
        return cacheNamespace + ":" +
               std::to_string(reinterpret_cast<std::uintptr_t>(ptr)) + ":" +
               std::to_string(windowX) + "," + std::to_string(windowY) + "," +
               std::to_string(windowWidth) + "x" + std::to_string(windowHeight) + ":" +
               std::to_string(paddedWidth) + "x" + std::to_string(paddedHeight) + ":" +
               std::to_string(channels);
    };
    auto kernelSpectrumCacheKey = [&](const std::string& cacheNamespace,
                                      const LensDiffKernel& kernel,
                                      int paddedWidth,
                                      int paddedHeight) {
        return cacheNamespace + ":" +
               std::to_string(reinterpret_cast<std::uintptr_t>(kernel.values.data())) + ":" +
               std::to_string(kernel.size) + ":" +
               std::to_string(paddedWidth) + "x" + std::to_string(paddedHeight);
    };
    auto fieldStackCacheKey = [&](int paddedWidth,
                                  int paddedHeight,
                                  FieldEffectKind effectKind,
                                  int binCount,
                                  int repeatPerKernel,
                                  int zoneStart,
                                  int zoneCount) {
        return fieldCacheNamespace + ":" +
                std::to_string(static_cast<unsigned long long>(timing.fieldKeyHash)) + ":" +
                std::to_string(paddedWidth) + "x" + std::to_string(paddedHeight) + ":" +
                std::to_string(static_cast<int>(effectKind)) + ":" +
                std::to_string(binCount) + ":" + std::to_string(repeatPerKernel) + ":" +
                std::to_string(zoneStart) + ":" + std::to_string(zoneCount);
    };
    auto estimateStackWorkingBytes = [&](int paddedWidth, int paddedHeight, int sliceCount) -> std::uint64_t {
        const std::uint64_t paddedCount = static_cast<std::uint64_t>(paddedWidth) * static_cast<std::uint64_t>(paddedHeight);
        const std::uint64_t planeCount = static_cast<std::uint64_t>(pixelCount);
        const std::uint64_t slices = static_cast<std::uint64_t>(std::max(0, sliceCount));
        return slices * (paddedCount * static_cast<std::uint64_t>(sizeof(cufftComplex)) * 3ULL +
                         planeCount * static_cast<std::uint64_t>(sizeof(float)));
    };
    constexpr std::uint64_t kFieldStackBudgetBytes = 1536ULL * 1024ULL * 1024ULL;
    auto getPlanForDimsFor = [&](CudaFftSubsystem subsystem,
                                 int paddedWidth,
                                 int paddedHeight,
                                 int batchCount,
                                 CudaFftPlanLease* outPlan) -> bool {
        const CudaFftBackend subsystemBackend =
            subsystem == CudaFftSubsystem::PsfBank ? psfRequestedFftBackend :
            (subsystem == CudaFftSubsystem::FieldRender ? fieldRequestedFftBackend : globalRequestedFftBackend);
        return acquireFftPlan(subsystem,
                              subsystemBackend,
                              vkfftStrict,
                              planRepository,
                              usePersistentVkfftRepositoryForSubsystem(subsystem, fftPolicy, persistentPlanRepositoryEnabled),
                              paddedWidth,
                              paddedHeight,
                              batchCount,
                              stream,
                              &timing,
                              outPlan,
                              error);
    };
    auto getPlanForDims = [&](int paddedWidth, int paddedHeight, int batchCount, CudaFftPlanLease* outPlan) -> bool {
        return getPlanForDimsFor(CudaFftSubsystem::GlobalRender, paddedWidth, paddedHeight, batchCount, outPlan);
    };
    auto acquireDirectPlanForSubsystem = [&](CudaFftSubsystem subsystem,
                                             int paddedWidth,
                                             int paddedHeight,
                                             int batchCount,
                                             CudaFftPlanLease* outPlan) -> bool {
        const CudaFftBackend subsystemBackend =
            subsystem == CudaFftSubsystem::PsfBank ? psfRequestedFftBackend :
            (subsystem == CudaFftSubsystem::FieldRender ? fieldRequestedFftBackend : globalRequestedFftBackend);
        return acquireFftPlan(subsystem,
                              subsystemBackend,
                              vkfftStrict,
                              nullptr,
                              usePersistentVkfftRepositoryForSubsystem(subsystem, fftPolicy, persistentPlanRepositoryEnabled),
                              paddedWidth,
                              paddedHeight,
                              batchCount,
                              stream,
                              &timing,
                              outPlan,
                              error);
    };
    auto getComplexScratch = [&](const std::string& key,
                                 std::size_t elementCount,
                                 DeviceBuffer<cufftComplex>** outBuffer) -> bool {
        if (outBuffer == nullptr) {
            if (error) *error = "cuda-null-complex-scratch";
            return false;
        }
        auto it = complexScratchCache.find(key);
        if (it == complexScratchCache.end()) {
            it = complexScratchCache.emplace(key, DeviceBuffer<cufftComplex> {}).first;
        }
        if (!it->second.allocate(elementCount)) {
            if (error) *error = "cuda-alloc-complex-scratch";
            return false;
        }
        *outBuffer = &it->second;
        return true;
    };
    auto getFloatScratch = [&](const std::string& key,
                               std::size_t elementCount,
                               DeviceBuffer<float>** outBuffer) -> bool {
        if (outBuffer == nullptr) {
            if (error) *error = "cuda-null-float-scratch";
            return false;
        }
        auto it = floatScratchCache.find(key);
        if (it == floatScratchCache.end()) {
            it = floatScratchCache.emplace(key, DeviceBuffer<float> {}).first;
        }
        if (!it->second.allocate(elementCount)) {
            if (error) *error = "cuda-alloc-float-scratch";
            return false;
        }
        *outBuffer = &it->second;
        return true;
    };
    auto getPlaneSetScratch = [&](const std::string& key,
                                  std::size_t elementCount,
                                  PlaneSet** outSet) -> bool {
        if (outSet == nullptr) {
            if (error) *error = "cuda-null-plane-scratch";
            return false;
        }
        auto it = planeSetScratchCache.find(key);
        if (it == planeSetScratchCache.end()) {
            it = planeSetScratchCache.emplace(key, PlaneSet {}).first;
        }
        if (!allocatePlaneSetCount(it->second, elementCount)) {
            if (error) *error = "cuda-alloc-plane-scratch";
            return false;
        }
        *outSet = &it->second;
        return true;
    };
    auto getSpectralPlaneScratch = [&](const std::string& key,
                                       SpectralPlaneSet** outSet) -> bool {
        if (outSet == nullptr) {
            if (error) *error = "cuda-null-spectral-scratch";
            return false;
        }
        auto it = spectralPlaneScratchCache.find(key);
        if (it == spectralPlaneScratchCache.end()) {
            it = spectralPlaneScratchCache.emplace(key, SpectralPlaneSet {}).first;
        }
        for (auto& plane : it->second.bins) {
            if (!plane.allocate(pixelCount)) {
                if (error) *error = "cuda-alloc-spectral-scratch";
                return false;
            }
        }
        *outSet = &it->second;
        return true;
    };
    auto getScalarSourceSpectrumFor = [&](CudaFftSubsystem subsystem,
                                          const std::string& cacheNamespace,
                                          const DeviceBuffer<float>& source,
                                          int paddedWidth,
                                          int paddedHeight,
                                          DeviceBuffer<cufftComplex>** outSpectrum) -> bool {
        if (outSpectrum == nullptr) {
            return false;
        }
        const std::string key = sourceSpectrumCacheKey(cacheNamespace, source.ptr, paddedWidth, paddedHeight, 1);
        auto it = scalarSourceSpectrumCache.find(key);
        if (it != scalarSourceSpectrumCache.end()) {
            ++timing.scalarSourceCacheHits;
            *outSpectrum = &it->second;
            return true;
        }
        ++timing.scalarSourceCacheMisses;
        CudaFftPlanLease plan;
        DeviceBuffer<cufftComplex> spectrum;
        if (!getPlanForDimsFor(subsystem, paddedWidth, paddedHeight, 1, &plan) ||
            !timeCall(timing.sourceFftMs, [&] {
                return makeImageSpectrum(source.ptr, subsystem, width, height, paddedWidth, paddedHeight, &plan, stream, &timing, spectrum, error);
            })) {
            return false;
        }
        auto insert = scalarSourceSpectrumCache.emplace(key, std::move(spectrum));
        *outSpectrum = &insert.first->second;
        return true;
    };
    auto getScalarSourceSpectrum = [&](const DeviceBuffer<float>& source,
                                       int paddedWidth,
                                       int paddedHeight,
                                       DeviceBuffer<cufftComplex>** outSpectrum) -> bool {
        return getScalarSourceSpectrumFor(CudaFftSubsystem::GlobalRender,
                                          globalCacheNamespace,
                                          source,
                                          paddedWidth,
                                          paddedHeight,
                                          outSpectrum);
    };
    auto getFieldScalarSourceSpectrum = [&](const DeviceBuffer<float>& source,
                                            int paddedWidth,
                                            int paddedHeight,
                                            DeviceBuffer<cufftComplex>** outSpectrum) -> bool {
        return getScalarSourceSpectrumFor(CudaFftSubsystem::FieldRender,
                                          fieldCacheNamespace,
                                          source,
                                          paddedWidth,
                                          paddedHeight,
                                          outSpectrum);
    };
    auto getScalarSourceSpectrumWindowFor = [&](CudaFftSubsystem subsystem,
                                                const std::string& cacheNamespace,
                                                const DeviceBuffer<float>& source,
                                                int windowX,
                                                int windowY,
                                                int windowWidth,
                                                int windowHeight,
                                                int paddedWidth,
                                                int paddedHeight,
                                                DeviceBuffer<cufftComplex>** outSpectrum) -> bool {
        if (outSpectrum == nullptr) {
            return false;
        }
        const std::string key = windowSourceSpectrumCacheKey(cacheNamespace,
                                                             source.ptr,
                                                             windowX,
                                                             windowY,
                                                             windowWidth,
                                                             windowHeight,
                                                             paddedWidth,
                                                             paddedHeight,
                                                             1);
        auto it = scalarSourceSpectrumCache.find(key);
        if (it != scalarSourceSpectrumCache.end()) {
            ++timing.scalarSourceCacheHits;
            *outSpectrum = &it->second;
            return true;
        }
        ++timing.scalarSourceCacheMisses;
        CudaFftPlanLease plan;
        DeviceBuffer<cufftComplex> spectrum;
        if (!getPlanForDimsFor(subsystem, paddedWidth, paddedHeight, 1, &plan) ||
            !timeCall(timing.sourceFftMs, [&] {
                return makeImageSpectrumWindow(source.ptr,
                                               width,
                                               height,
                                               windowX,
                                               windowY,
                                               windowWidth,
                                               windowHeight,
                                               paddedWidth,
                                               paddedHeight,
                                               &plan,
                                               stream,
                                               &timing,
                                               spectrum,
                                               error);
            })) {
            return false;
        }
        auto insert = scalarSourceSpectrumCache.emplace(key, std::move(spectrum));
        *outSpectrum = &insert.first->second;
        return true;
    };
    auto getScalarSourceSpectrumWindow = [&](const DeviceBuffer<float>& source,
                                             int windowX,
                                             int windowY,
                                             int windowWidth,
                                             int windowHeight,
                                             int paddedWidth,
                                             int paddedHeight,
                                             DeviceBuffer<cufftComplex>** outSpectrum) -> bool {
        return getScalarSourceSpectrumWindowFor(CudaFftSubsystem::GlobalRender,
                                                globalCacheNamespace,
                                                source,
                                                windowX,
                                                windowY,
                                                windowWidth,
                                                windowHeight,
                                                paddedWidth,
                                                paddedHeight,
                                                outSpectrum);
    };
    auto getFieldScalarSourceSpectrumWindow = [&](const DeviceBuffer<float>& source,
                                                  int windowX,
                                                  int windowY,
                                                  int windowWidth,
                                                  int windowHeight,
                                                  int paddedWidth,
                                                  int paddedHeight,
                                                  DeviceBuffer<cufftComplex>** outSpectrum) -> bool {
        return getScalarSourceSpectrumWindowFor(CudaFftSubsystem::FieldRender,
                                                fieldCacheNamespace,
                                                source,
                                                windowX,
                                                windowY,
                                                windowWidth,
                                                windowHeight,
                                                paddedWidth,
                                                paddedHeight,
                                                outSpectrum);
    };
    auto getKernelSpectrumFor = [&](CudaFftSubsystem subsystem,
                                    const std::string& cacheNamespace,
                                    const LensDiffKernel& kernel,
                                    int paddedWidth,
                                    int paddedHeight,
                                    DeviceBuffer<cufftComplex>** outSpectrum) -> bool {
        if (outSpectrum == nullptr) {
            return false;
        }
        const std::string key = kernelSpectrumCacheKey(cacheNamespace, kernel, paddedWidth, paddedHeight);
        auto it = kernelSpectrumCache.find(key);
        if (it != kernelSpectrumCache.end()) {
            ++timing.kernelCacheHits;
            *outSpectrum = it->second.get();
            return true;
        }
        int currentDevice = 0;
        if (!checkCuda(cudaGetDevice(&currentDevice), "cudaGetDevice-kernel-spectrum-cache", error)) {
            return false;
        }
        const std::string persistentKey =
            persistentKernelSpectrumCacheKey(cacheNamespace, currentDevice, kernel, paddedWidth, paddedHeight);
        if (persistentKernelCacheEnabled) {
            auto& persistentCache = persistentCudaKernelSpectrumCache();
            std::lock_guard<std::mutex> lock(persistentCache.mutex);
            auto persistentIt = persistentCache.entries.find(persistentKey);
            if (persistentIt != persistentCache.entries.end() &&
                persistentIt->second.spectrum != nullptr &&
                persistentIt->second.spectrum->ptr != nullptr) {
                if (persistentIt->second.readyEvent != nullptr &&
                    !checkCuda(cudaStreamWaitEvent(stream, persistentIt->second.readyEvent, 0),
                               "cudaStreamWaitEvent-kernel-spectrum-cache",
                               error)) {
                    return false;
                }
                ++timing.kernelCacheHits;
                persistentIt->second.stamp = ++persistentCache.nextStamp;
                auto insert = kernelSpectrumCache.emplace(key, persistentIt->second.spectrum);
                *outSpectrum = insert.first->second.get();
                return true;
            }
        }
        ++timing.kernelCacheMisses;
        CudaFftPlanLease plan;
        DeviceBuffer<float> deviceKernel;
        DeviceBuffer<cufftComplex> spectrum;
        std::shared_ptr<DeviceBuffer<cufftComplex>> sharedSpectrum;
        if (!getPlanForDimsFor(subsystem, paddedWidth, paddedHeight, 1, &plan) ||
            !timeCall(timing.kernelFftMs, [&] {
                return makeKernelSpectrum(kernel, subsystem, paddedWidth, paddedHeight, &plan, stream, &timing, deviceKernel, spectrum, error);
            })) {
            return false;
        }
        sharedSpectrum = std::make_shared<DeviceBuffer<cufftComplex>>();
        *sharedSpectrum = std::move(spectrum);
        if (persistentKernelCacheEnabled) {
            cudaEvent_t readyEvent = nullptr;
            if (!checkCuda(cudaEventCreateWithFlags(&readyEvent, cudaEventDisableTiming),
                           "cudaEventCreateWithFlags-kernel-spectrum-cache",
                           error) ||
                !checkCuda(cudaEventRecord(readyEvent, stream),
                           "cudaEventRecord-kernel-spectrum-cache",
                           error)) {
                if (readyEvent != nullptr) {
                    cudaEventDestroy(readyEvent);
                }
                return false;
            }
            auto& persistentCache = persistentCudaKernelSpectrumCache();
            std::lock_guard<std::mutex> lock(persistentCache.mutex);
            constexpr std::size_t kMaxPersistentKernelSpectrumBytes = 2048U * 1024U * 1024U;
            const std::size_t spectrumBytes = sharedSpectrum->count * sizeof(cufftComplex);
            while (persistentCache.totalBytes + spectrumBytes > kMaxPersistentKernelSpectrumBytes &&
                   !persistentCache.entries.empty()) {
                auto oldestIt = persistentCache.entries.begin();
                for (auto entryIt = std::next(persistentCache.entries.begin());
                     entryIt != persistentCache.entries.end();
                     ++entryIt) {
                    if (entryIt->second.stamp < oldestIt->second.stamp) {
                        oldestIt = entryIt;
                    }
                }
                persistentCache.totalBytes -= oldestIt->second.bytes;
                persistentCache.entries.erase(oldestIt);
            }
            if (spectrumBytes <= kMaxPersistentKernelSpectrumBytes) {
                PersistentCudaKernelSpectrumCache::Entry entry {};
                entry.spectrum = sharedSpectrum;
                entry.bytes = spectrumBytes;
                entry.stamp = ++persistentCache.nextStamp;
                entry.readyEvent = readyEvent;
                persistentCache.totalBytes += entry.bytes;
                persistentCache.entries[persistentKey] = std::move(entry);
            } else {
                cudaEventDestroy(readyEvent);
            }
        }
        auto insert = kernelSpectrumCache.emplace(key, std::move(sharedSpectrum));
        *outSpectrum = insert.first->second.get();
        return true;
    };
    auto getKernelSpectrum = [&](const LensDiffKernel& kernel,
                                 int paddedWidth,
                                 int paddedHeight,
                                 DeviceBuffer<cufftComplex>** outSpectrum) -> bool {
        return getKernelSpectrumFor(CudaFftSubsystem::GlobalRender,
                                    globalCacheNamespace,
                                    kernel,
                                    paddedWidth,
                                    paddedHeight,
                                    outSpectrum);
    };
    auto getFieldKernelSpectrum = [&](const LensDiffKernel& kernel,
                                      int paddedWidth,
                                      int paddedHeight,
                                      DeviceBuffer<cufftComplex>** outSpectrum) -> bool {
        return getKernelSpectrumFor(CudaFftSubsystem::FieldRender,
                                    fieldCacheNamespace,
                                    kernel,
                                    paddedWidth,
                                    paddedHeight,
                                    outSpectrum);
    };
    auto getRgbPlaneSpectrumFor = [&](CudaFftSubsystem subsystem,
                                      const std::string& cacheNamespace,
                                      const DeviceBuffer<float>& source,
                                      int paddedWidth,
                                      int paddedHeight,
                                      DeviceBuffer<cufftComplex>** outSpectrum) -> bool {
        if (outSpectrum == nullptr) {
            return false;
        }
        const std::string key = sourceSpectrumCacheKey(cacheNamespace, source.ptr, paddedWidth, paddedHeight, 3);
        auto it = rgbSourceSpectrumCache.find(key);
        if (it != rgbSourceSpectrumCache.end()) {
            ++timing.rgbSourceCacheHits;
            *outSpectrum = &it->second;
            return true;
        }
        ++timing.rgbSourceCacheMisses;
        CudaFftPlanLease plan;
        DeviceBuffer<cufftComplex> spectrum;
        if (!getPlanForDimsFor(subsystem, paddedWidth, paddedHeight, 1, &plan) ||
            !timeCall(timing.sourceFftMs, [&] {
                return makeImageSpectrum(source.ptr, subsystem, width, height, paddedWidth, paddedHeight, &plan, stream, &timing, spectrum, error);
            })) {
            return false;
        }
        auto insert = rgbSourceSpectrumCache.emplace(key, std::move(spectrum));
        *outSpectrum = &insert.first->second;
        return true;
    };
    auto getRgbPlaneSpectrum = [&](const DeviceBuffer<float>& source,
                                   int paddedWidth,
                                   int paddedHeight,
                                   DeviceBuffer<cufftComplex>** outSpectrum) -> bool {
        return getRgbPlaneSpectrumFor(CudaFftSubsystem::GlobalRender,
                                      globalCacheNamespace,
                                      source,
                                      paddedWidth,
                                      paddedHeight,
                                      outSpectrum);
    };
    auto getFieldRgbPlaneSpectrum = [&](const DeviceBuffer<float>& source,
                                        int paddedWidth,
                                        int paddedHeight,
                                        DeviceBuffer<cufftComplex>** outSpectrum) -> bool {
        return getRgbPlaneSpectrumFor(CudaFftSubsystem::FieldRender,
                                      fieldCacheNamespace,
                                      source,
                                      paddedWidth,
                                      paddedHeight,
                                      outSpectrum);
    };

    auto convolvePlaneSet = [&](const DeviceBuffer<float>& srcRPlane,
                                const DeviceBuffer<float>& srcGPlane,
                                const DeviceBuffer<float>& srcBPlane,
                                const LensDiffKernel& kernel,
                                PlaneSet& dst,
                                const char* stagePrefix) -> bool {
        const int paddedWidth = nextPowerOfTwo(width + kernel.size - 1);
        const int paddedHeight = nextPowerOfTwo(height + kernel.size - 1);
        CudaFftPlanLease plan;
        DeviceBuffer<cufftComplex> tempSpectrum;
        DeviceBuffer<cufftComplex>* imageSpecR = nullptr;
        DeviceBuffer<cufftComplex>* imageSpecG = nullptr;
        DeviceBuffer<cufftComplex>* imageSpecB = nullptr;
        DeviceBuffer<cufftComplex>* kernelSpec = nullptr;
        const bool ok = getPlanForDims(paddedWidth, paddedHeight, 1, &plan) &&
                        getRgbPlaneSpectrum(srcRPlane, paddedWidth, paddedHeight, &imageSpecR) &&
                        getRgbPlaneSpectrum(srcGPlane, paddedWidth, paddedHeight, &imageSpecG) &&
                        getRgbPlaneSpectrum(srcBPlane, paddedWidth, paddedHeight, &imageSpecB) &&
                        getKernelSpectrum(kernel, paddedWidth, paddedHeight, &kernelSpec) &&
                        convolveSpectrumToPlane(*imageSpecR, *kernelSpec, CudaFftSubsystem::GlobalRender, width, height, paddedWidth, paddedHeight, &plan, stream, &timing, tempSpectrum, dst.r, error) &&
                        convolveSpectrumToPlane(*imageSpecG, *kernelSpec, CudaFftSubsystem::GlobalRender, width, height, paddedWidth, paddedHeight, &plan, stream, &timing, tempSpectrum, dst.g, error) &&
                        convolveSpectrumToPlane(*imageSpecB, *kernelSpec, CudaFftSubsystem::GlobalRender, width, height, paddedWidth, paddedHeight, &plan, stream, &timing, tempSpectrum, dst.b, error);
        if (!ok && error && error->empty()) {
            *error = stagePrefix;
        }
        return ok;
    };

    auto renderFromBins = [&](const std::vector<LensDiffPsfBin>& bins,
                              PlaneSet& outEffect,
                              PlaneSet* outCore,
                              PlaneSet* outStructure) -> bool {
        if (!allocatePlaneSet(outEffect)) {
            if (error) *error = "cuda-alloc-zone-effect";
            return false;
        }
        const bool localNeedCore = splitMode || outCore != nullptr;
        const bool localNeedStructure = splitMode || outStructure != nullptr;
        PlaneSet localCorePlanes;
        PlaneSet localStructurePlanes;
        PlaneSet* corePlanes = outCore != nullptr ? outCore : &localCorePlanes;
        PlaneSet* structurePlanes = outStructure != nullptr ? outStructure : &localStructurePlanes;
        if ((localNeedCore && !allocatePlaneSet(*corePlanes)) ||
            (localNeedStructure && !allocatePlaneSet(*structurePlanes))) {
            if (error) *error = "cuda-alloc-zone-split";
            return false;
        }

        if (params.spectralMode == LensDiffSpectralMode::Mono) {
            PlaneSet fullEffect;
            if (!splitMode && !allocatePlaneSet(fullEffect)) {
                if (error) *error = "cuda-alloc-zone-full";
                return false;
            }

            if (!splitMode && !convolvePlaneSet(redistributedR, redistributedG, redistributedB, bins.front().full, fullEffect, "cuda-convolve-zone-full")) {
                return false;
            }
            if (localNeedCore && !convolvePlaneSet(redistributedR, redistributedG, redistributedB, bins.front().core, *corePlanes, "cuda-convolve-zone-core")) {
                return false;
            }
            if (localNeedStructure && !convolvePlaneSet(redistributedR, redistributedG, redistributedB, bins.front().structure, *structurePlanes, "cuda-convolve-zone-structure")) {
                return false;
            }

            if (splitMode) {
                if (params.coreShoulder > 0.0) {
                    applyShoulderKernel<<<flatGrid, flatBlock, 0, stream>>>(
                        corePlanes->r.ptr, corePlanes->g.ptr, corePlanes->b.ptr, pixelCount, static_cast<float>(params.coreShoulder));
                    if (!checkCuda(cudaGetLastError(), "applyShoulderKernel-zone-core", error)) {
                        return false;
                    }
                }
                if (params.structureShoulder > 0.0) {
                    applyShoulderKernel<<<flatGrid, flatBlock, 0, stream>>>(
                        structurePlanes->r.ptr, structurePlanes->g.ptr, structurePlanes->b.ptr, pixelCount, static_cast<float>(params.structureShoulder));
                    if (!checkCuda(cudaGetLastError(), "applyShoulderKernel-zone-structure", error)) {
                        return false;
                    }
                }
                combineRgbKernel<<<flatGrid, flatBlock, 0, stream>>>(
                    corePlanes->r.ptr, corePlanes->g.ptr, corePlanes->b.ptr,
                    structurePlanes->r.ptr, structurePlanes->g.ptr, structurePlanes->b.ptr,
                    pixelCount,
                    static_cast<float>(std::max(0.0, params.coreGain)),
                    static_cast<float>(std::max(0.0, params.structureGain)),
                    outEffect.r.ptr, outEffect.g.ptr, outEffect.b.ptr);
                return checkCuda(cudaGetLastError(), "combineRgbKernel-zone-mono", error);
            }
            return copyPlaneSet(fullEffect, outEffect, "cudaMemcpyAsync-zone-effect");
        }

        const LensDiffSpectrumConfig zoneSpectrumConfig = BuildLensDiffSpectrumConfig(params, bins);
        const int paddedWidth = nextPowerOfTwo(width + bins.front().full.size - 1);
        const int paddedHeight = nextPowerOfTwo(height + bins.front().full.size - 1);
        CudaFftPlanLease plan;
        DeviceBuffer<cufftComplex> tempSpectrum;
        DeviceBuffer<cufftComplex>* driverSpectrum = nullptr;
        SpectralPlaneSet fullBins;
        SpectralPlaneSet coreBins;
        SpectralPlaneSet structureBins;

        auto allocateSpectral = [&](SpectralPlaneSet& set) -> bool {
            for (auto& plane : set.bins) {
                if (!plane.allocate(pixelCount)) {
                    return false;
                }
            }
            return true;
        };
        auto convolveBins = [&](const std::vector<LensDiffKernel>& kernels, SpectralPlaneSet& dst) -> bool {
            const int activeBins = std::min<int>(static_cast<int>(kernels.size()), kLensDiffMaxSpectralBins);
            for (int i = 0; i < activeBins; ++i) {
                DeviceBuffer<cufftComplex>* kernelSpectrum = nullptr;
                if (!getKernelSpectrum(kernels[static_cast<std::size_t>(i)],
                                       paddedWidth,
                                       paddedHeight,
                                       &kernelSpectrum) ||
                    !timeCall(timing.convolutionMs, [&] {
                        return convolveSpectrumToPlane(*driverSpectrum,
                                             *kernelSpectrum,
                                             CudaFftSubsystem::GlobalRender,
                                             width,
                                             height,
                                             paddedWidth,
                                             paddedHeight,
                                              &plan,
                                              stream,
                                              &timing,
                                              tempSpectrum,
                                              dst.bins[static_cast<std::size_t>(i)],
                                              error);
                    })) {
                    return false;
                }
            }
            for (int i = activeBins; i < kLensDiffMaxSpectralBins; ++i) {
                if (!checkCuda(cudaMemsetAsync(dst.bins[static_cast<std::size_t>(i)].ptr,
                                               0,
                                               pixelCount * sizeof(float),
                                               stream),
                               "cudaMemsetAsync-zone-spectral-zero",
                               error)) {
                    return false;
                }
            }
            return true;
        };
        auto mapBins = [&](const SpectralPlaneSet& srcBins, PlaneSet& dst) -> bool {
            SpectralMapConfigGpu mapConfig {};
            mapConfig.binCount = zoneSpectrumConfig.binCount;
            for (int i = 0; i < kLensDiffMaxSpectralBins * 3; ++i) {
                mapConfig.naturalMatrix[static_cast<std::size_t>(i)] = zoneSpectrumConfig.naturalMatrix[static_cast<std::size_t>(i)];
                mapConfig.styleMatrix[static_cast<std::size_t>(i)] = zoneSpectrumConfig.styleMatrix[static_cast<std::size_t>(i)];
            }
            mapSpectralKernel<<<flatGrid, flatBlock, 0, stream>>>(
                srcBins.bins[0].ptr, srcBins.bins[1].ptr, srcBins.bins[2].ptr,
                srcBins.bins[3].ptr, srcBins.bins[4].ptr, srcBins.bins[5].ptr,
                srcBins.bins[6].ptr, srcBins.bins[7].ptr, srcBins.bins[8].ptr,
                pixelCount,
                mapConfig,
                static_cast<float>(std::clamp(params.spectrumForce, 0.0, 1.0)),
                static_cast<float>(std::max(0.0, params.spectrumSaturation)),
                params.chromaticAffectsLuma ? 1 : 0,
                dst.r.ptr, dst.g.ptr, dst.b.ptr);
            return checkCuda(cudaGetLastError(), "mapSpectralKernel-zone", error);
        };

        if (!getPlanForDims(paddedWidth, paddedHeight, 1, &plan)) {
            return false;
        }
        bool ok = getScalarSourceSpectrum(redistributedDriver, paddedWidth, paddedHeight, &driverSpectrum);
        if (ok && !splitMode) {
            ok = allocateSpectral(fullBins);
        }
        if (ok && localNeedCore) {
            ok = allocateSpectral(coreBins);
        }
        if (ok && localNeedStructure) {
            ok = allocateSpectral(structureBins);
        }
        if (!ok) {
            if (error && error->empty()) *error = "cuda-alloc-zone-spectral";
            return false;
        }

        if (ok && !splitMode) {
            std::vector<LensDiffKernel> kernels;
            kernels.reserve(bins.size());
            for (const auto& bin : bins) kernels.push_back(bin.full);
            ok = convolveBins(kernels, fullBins);
        }
        if (ok && localNeedCore) {
            std::vector<LensDiffKernel> kernels;
            kernels.reserve(bins.size());
            for (const auto& bin : bins) kernels.push_back(bin.core);
            ok = convolveBins(kernels, coreBins);
        }
        if (ok && localNeedStructure) {
            std::vector<LensDiffKernel> kernels;
            kernels.reserve(bins.size());
            for (const auto& bin : bins) kernels.push_back(bin.structure);
            ok = convolveBins(kernels, structureBins);
        }
        if (ok && !splitMode) {
            ok = mapBins(fullBins, outEffect);
        }
        if (ok && localNeedCore) {
            ok = mapBins(coreBins, *corePlanes);
        }
        if (ok && localNeedStructure) {
            ok = mapBins(structureBins, *structurePlanes);
        }
        if (!ok) {
            return false;
        }

        if (splitMode) {
            if (params.coreShoulder > 0.0) {
                applyShoulderKernel<<<flatGrid, flatBlock, 0, stream>>>(
                    corePlanes->r.ptr, corePlanes->g.ptr, corePlanes->b.ptr, pixelCount, static_cast<float>(params.coreShoulder));
                if (!checkCuda(cudaGetLastError(), "applyShoulderKernel-zone-core-spectral", error)) {
                    return false;
                }
            }
            if (params.structureShoulder > 0.0) {
                applyShoulderKernel<<<flatGrid, flatBlock, 0, stream>>>(
                    structurePlanes->r.ptr, structurePlanes->g.ptr, structurePlanes->b.ptr, pixelCount, static_cast<float>(params.structureShoulder));
                if (!checkCuda(cudaGetLastError(), "applyShoulderKernel-zone-structure-spectral", error)) {
                    return false;
                }
            }
            combineRgbKernel<<<flatGrid, flatBlock, 0, stream>>>(
                corePlanes->r.ptr, corePlanes->g.ptr, corePlanes->b.ptr,
                structurePlanes->r.ptr, structurePlanes->g.ptr, structurePlanes->b.ptr,
                pixelCount,
                static_cast<float>(std::max(0.0, params.coreGain)),
                static_cast<float>(std::max(0.0, params.structureGain)),
                outEffect.r.ptr, outEffect.g.ptr, outEffect.b.ptr);
            ok = checkCuda(cudaGetLastError(), "combineRgbKernel-zone-spectral", error);
        }
        return ok;
    };
    auto kernelForEffect = [&](const LensDiffPsfBin& bin, FieldEffectKind effectKind) -> const LensDiffKernel& {
        switch (effectKind) {
            case FieldEffectKind::Core: return bin.core;
            case FieldEffectKind::Structure: return bin.structure;
            case FieldEffectKind::Full:
            default: return bin.full;
        }
    };
    auto buildFieldKernelSpectrumStack = [&](FieldEffectKind effectKind,
                                             int paddedWidth,
                                             int paddedHeight,
                                             int binCount,
                                             int repeatPerKernel,
                                             int zoneStart,
                                             int zoneCount,
                                             DeviceBuffer<cufftComplex>** outStack) -> bool {
        if (outStack == nullptr || !fieldPlan.canonical3x3 || zoneStart < 0 || zoneCount <= 0 ||
            zoneStart + zoneCount > static_cast<int>(fieldPlan.zones.size())) {
            return false;
        }
        const std::string key = fieldStackCacheKey(
            paddedWidth, paddedHeight, effectKind, binCount, repeatPerKernel, zoneStart, zoneCount);
        auto it = fieldKernelStackCache.find(key);
        if (it != fieldKernelStackCache.end()) {
            *outStack = &it->second.spectrumStack;
            return true;
        }
        FieldZoneSpectrumStacks stacks {};
        stacks.paddedWidth = paddedWidth;
        stacks.paddedHeight = paddedHeight;
        stacks.zoneCount = zoneCount;
        stacks.spectralBinCount = binCount;
        stacks.effectKind = effectKind;
        const std::size_t paddedCount = static_cast<std::size_t>(paddedWidth) * paddedHeight;
        const int sliceCount = stacks.zoneCount * binCount * repeatPerKernel;
        if (!stacks.spectrumStack.allocate(paddedCount * static_cast<std::size_t>(sliceCount))) {
            if (error) *error = "cuda-alloc-field-kernel-spectrum-stack";
            return false;
        }
        const auto buildStart = std::chrono::steady_clock::now();
        for (int localZoneIndex = 0; localZoneIndex < stacks.zoneCount; ++localZoneIndex) {
            const int zoneIndex = zoneStart + localZoneIndex;
            const LensDiffFieldZoneCache* zone = fieldPlan.zones[static_cast<std::size_t>(zoneIndex)];
            for (int binIndex = 0; binIndex < binCount; ++binIndex) {
                DeviceBuffer<cufftComplex>* kernelSpectrum = nullptr;
                if (zone == nullptr ||
                    !getFieldKernelSpectrum(kernelForEffect(zone->bins[static_cast<std::size_t>(binIndex)], effectKind),
                                       paddedWidth,
                                       paddedHeight,
                                       &kernelSpectrum)) {
                    return false;
                }
                for (int repeat = 0; repeat < repeatPerKernel; ++repeat) {
                    const int dstSlice = (localZoneIndex * binCount + binIndex) * repeatPerKernel + repeat;
                    if (!checkCuda(cudaMemcpyAsync(stacks.spectrumStack.ptr + paddedCount * static_cast<std::size_t>(dstSlice),
                                                   kernelSpectrum->ptr,
                                                   paddedCount * sizeof(cufftComplex),
                                                   cudaMemcpyDeviceToDevice,
                                                   stream),
                                   "cudaMemcpyAsync-field-kernel-spectrum-stack",
                                   error)) {
                        return false;
                    }
                }
            }
        }
        if (LensDiffTimingEnabled()) {
            LogLensDiffTimingStage(
                "cuda-stage-zone-kernel-stack-build",
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - buildStart)
                    .count(),
                "zones=" + std::to_string(stacks.zoneCount) +
                    ",bins=" + std::to_string(binCount));
        }
        auto insert = fieldKernelStackCache.emplace(key, std::move(stacks));
        *outStack = &insert.first->second.spectrumStack;
        return true;
    };
    auto getReplicatedSpectrumStack = [&](const DeviceBuffer<cufftComplex>& srcSpectrum,
                                          const std::string& key,
                                          int srcSliceCount,
                                          int dstSliceCount,
                                          int paddedWidth,
                                          int paddedHeight,
                                          DeviceBuffer<cufftComplex>** outStack) -> bool {
        if (outStack == nullptr) {
            return false;
        }
        auto it = fieldReplicatedSpectrumCache.find(key);
        if (it != fieldReplicatedSpectrumCache.end()) {
            *outStack = &it->second;
            return true;
        }
        const std::size_t paddedCount = static_cast<std::size_t>(paddedWidth) * paddedHeight;
        DeviceBuffer<cufftComplex> stack;
        if (!stack.allocate(paddedCount * static_cast<std::size_t>(dstSliceCount))) {
            if (error) *error = "cuda-alloc-field-replicated-spectrum-stack";
            return false;
        }
        const int block = 256;
        const int grid = static_cast<int>(((paddedCount * static_cast<std::size_t>(dstSliceCount)) + block - 1) / block);
        replicateComplexStackKernel<<<grid, block, 0, stream>>>(srcSpectrum.ptr,
                                                                srcSliceCount,
                                                                dstSliceCount,
                                                                paddedCount,
                                                                stack.ptr);
        if (!checkCuda(cudaGetLastError(), "replicateComplexStackKernel", error)) {
            return false;
        }
        auto insert = fieldReplicatedSpectrumCache.emplace(key, std::move(stack));
        *outStack = &insert.first->second;
        return true;
    };
    auto getRgbTripletSpectrumStack = [&](int paddedWidth,
                                          int paddedHeight,
                                          DeviceBuffer<cufftComplex>** outStack) -> bool {
        if (outStack == nullptr) {
            if (error) *error = "cuda-null-rgb-triplet-stack";
            return false;
        }
        const std::string key = fieldCacheNamespace + ":field-rgb-triplet:" + std::to_string(paddedWidth) + "x" + std::to_string(paddedHeight);
        auto it = fieldRgbTripletSpectrumCache.find(key);
        if (it != fieldRgbTripletSpectrumCache.end()) {
            *outStack = &it->second;
            return true;
        }
        DeviceBuffer<cufftComplex>* srcR = nullptr;
        DeviceBuffer<cufftComplex>* srcG = nullptr;
        DeviceBuffer<cufftComplex>* srcB = nullptr;
        if (!getFieldRgbPlaneSpectrum(redistributedR, paddedWidth, paddedHeight, &srcR) ||
            !getFieldRgbPlaneSpectrum(redistributedG, paddedWidth, paddedHeight, &srcG) ||
            !getFieldRgbPlaneSpectrum(redistributedB, paddedWidth, paddedHeight, &srcB)) {
            return false;
        }
        const std::size_t paddedCount = static_cast<std::size_t>(paddedWidth) * paddedHeight;
        DeviceBuffer<cufftComplex> stack;
        if (!stack.allocate(paddedCount * 3U)) {
            if (error) *error = "cuda-alloc-rgb-triplet-spectrum-stack";
            return false;
        }
        if (!checkCuda(cudaMemcpyAsync(stack.ptr,
                                       srcR->ptr,
                                       paddedCount * sizeof(cufftComplex),
                                       cudaMemcpyDeviceToDevice,
                                       stream),
                       "cudaMemcpyAsync-rgb-triplet-spectrum-r",
                       error) ||
            !checkCuda(cudaMemcpyAsync(stack.ptr + paddedCount,
                                       srcG->ptr,
                                       paddedCount * sizeof(cufftComplex),
                                       cudaMemcpyDeviceToDevice,
                                       stream),
                       "cudaMemcpyAsync-rgb-triplet-spectrum-g",
                       error) ||
            !checkCuda(cudaMemcpyAsync(stack.ptr + paddedCount * 2U,
                                       srcB->ptr,
                                       paddedCount * sizeof(cufftComplex),
                                       cudaMemcpyDeviceToDevice,
                                       stream),
                       "cudaMemcpyAsync-rgb-triplet-spectrum-b",
                       error)) {
            return false;
        }
    auto insert = fieldRgbTripletSpectrumCache.emplace(key, std::move(stack));
        *outStack = &insert.first->second;
        return true;
    };
    int tiledFieldTileWidth = 0;
    int tiledFieldTileHeight = 0;
    int tiledFieldChunkZones = 0;
    std::uint64_t tiledFieldWorkingBytes = 0;
    auto estimateTiledSpectralWorkingBytes = [&](int maxKernelSize,
                                                 int tileWidth,
                                                 int tileHeight,
                                                 int chunkZoneCount,
                                                 int binCount) -> std::uint64_t {
        const int patchWidth = std::min(width, tileWidth + maxKernelSize - 1);
        const int patchHeight = std::min(height, tileHeight + maxKernelSize - 1);
        const int paddedWidth = nextPowerOfTwo(patchWidth + maxKernelSize - 1);
        const int paddedHeight = nextPowerOfTwo(patchHeight + maxKernelSize - 1);
        return estimateStackWorkingBytes(paddedWidth, paddedHeight, chunkZoneCount * binCount);
    };
    auto chooseTiledSpectralFieldLayout = [&](FieldEffectKind effectKind,
                                              int* outMaxKernelSize,
                                              int* outTileWidth,
                                              int* outTileHeight,
                                              int* outChunkZones,
                                              std::uint64_t* outWorkingBytes) -> bool {
        if (outMaxKernelSize == nullptr || outTileWidth == nullptr || outTileHeight == nullptr ||
            outChunkZones == nullptr || outWorkingBytes == nullptr ||
            !fieldPlan.canonical3x3 || params.spectralMode == LensDiffSpectralMode::Mono) {
            return false;
        }
        const int zoneCount = static_cast<int>(fieldPlan.zones.size());
        const int binCount = std::min<int>(static_cast<int>(fieldPlan.zones.front()->bins.size()), kLensDiffMaxSpectralBins);
        int maxKernelSize = 1;
        for (const LensDiffFieldZoneCache* zone : fieldPlan.zones) {
            for (int binIndex = 0; binIndex < binCount; ++binIndex) {
                maxKernelSize = std::max(maxKernelSize,
                                         kernelForEffect(zone->bins[static_cast<std::size_t>(binIndex)], effectKind).size);
            }
        }
        int tileWidth = std::min(width, 1024);
        int tileHeight = std::min(height, 1024);
        tileWidth = std::max(64, tileWidth);
        tileHeight = std::max(64, tileHeight);
        const std::uint64_t targetBudgetBytes = kFieldStackBudgetBytes * 3ULL / 4ULL;
        while (true) {
            const std::uint64_t oneZoneBytes = std::max<std::uint64_t>(
                1ULL,
                estimateTiledSpectralWorkingBytes(maxKernelSize, tileWidth, tileHeight, 1, binCount));
            int chunkZones = static_cast<int>(targetBudgetBytes / oneZoneBytes);
            chunkZones = std::max(1, std::min(zoneCount, chunkZones));
            std::uint64_t workingBytes =
                estimateTiledSpectralWorkingBytes(maxKernelSize, tileWidth, tileHeight, chunkZones, binCount);
            while (workingBytes > targetBudgetBytes && chunkZones > 1) {
                --chunkZones;
                workingBytes =
                    estimateTiledSpectralWorkingBytes(maxKernelSize, tileWidth, tileHeight, chunkZones, binCount);
            }
            if (workingBytes <= targetBudgetBytes || (tileWidth <= 64 && tileHeight <= 64 && chunkZones == 1)) {
                *outMaxKernelSize = maxKernelSize;
                *outTileWidth = tileWidth;
                *outTileHeight = tileHeight;
                *outChunkZones = chunkZones;
                *outWorkingBytes = workingBytes;
                return true;
            }
            if (tileWidth >= tileHeight && tileWidth > 64) {
                tileWidth = std::max(64, (tileWidth + 1) / 2);
            } else if (tileHeight > 64) {
                tileHeight = std::max(64, (tileHeight + 1) / 2);
            } else {
                *outMaxKernelSize = maxKernelSize;
                *outTileWidth = tileWidth;
                *outTileHeight = tileHeight;
                *outChunkZones = 1;
                *outWorkingBytes = workingBytes;
                return true;
            }
        }
    };
    auto renderFieldZonesSpectralTiled = [&](FieldEffectKind effectKind, PlaneSet* outPlaneSet) -> bool {
        if (outPlaneSet == nullptr || !fieldPlan.canonical3x3 || params.spectralMode == LensDiffSpectralMode::Mono) {
            return false;
        }
        const int zoneCount = static_cast<int>(fieldPlan.zones.size());
        const int binCount = std::min<int>(static_cast<int>(fieldPlan.zones.front()->bins.size()), kLensDiffMaxSpectralBins);
        int maxKernelSize = 1;
        int tileWidth = std::min(width, 512);
        int tileHeight = std::min(height, 512);
        int chunkZoneCount = 1;
        std::uint64_t workingBytes = 0;
        if (!chooseTiledSpectralFieldLayout(effectKind,
                                            &maxKernelSize,
                                            &tileWidth,
                                            &tileHeight,
                                            &chunkZoneCount,
                                            &workingBytes)) {
            return false;
        }
        tiledFieldTileWidth = tileWidth;
        tiledFieldTileHeight = tileHeight;
        tiledFieldChunkZones = chunkZoneCount;
        tiledFieldWorkingBytes = workingBytes;
        timing.fieldZoneBatchDepth = std::max(timing.fieldZoneBatchDepth, chunkZoneCount);

        SpectralPlaneSet* weightedPlanes = nullptr;
        if (!getSpectralPlaneScratch("field-weighted-planes:" + std::to_string(binCount), &weightedPlanes)) {
            return false;
        }
        for (int binIndex = 0; binIndex < kLensDiffMaxSpectralBins; ++binIndex) {
            if (!checkCuda(cudaMemsetAsync(weightedPlanes->bins[static_cast<std::size_t>(binIndex)].ptr,
                                           0,
                                           pixelCount * sizeof(float),
                                           stream),
                           "cudaMemsetAsync-field-weighted-plane-reset",
                           error)) {
                return false;
            }
        }

        const auto convolutionStart = std::chrono::steady_clock::now();
        std::size_t tileCount = 0;
        for (int tileY = 0; tileY < height; tileY += tileHeight) {
            const int activeTileHeight = std::min(tileHeight, height - tileY);
            for (int tileX = 0; tileX < width; tileX += tileWidth) {
                const int activeTileWidth = std::min(tileWidth, width - tileX);
                const int patchX = std::max(0, tileX - (maxKernelSize - 1) / 2);
                const int patchY = std::max(0, tileY - (maxKernelSize - 1) / 2);
                const int patchWidth = std::min(width, tileX + activeTileWidth + maxKernelSize / 2) - patchX;
                const int patchHeight = std::min(height, tileY + activeTileHeight + maxKernelSize / 2) - patchY;
                const int patchOffsetX = tileX - patchX;
                const int patchOffsetY = tileY - patchY;
                const int paddedWidth = nextPowerOfTwo(patchWidth + maxKernelSize - 1);
                const int paddedHeight = nextPowerOfTwo(patchHeight + maxKernelSize - 1);
                const std::size_t patchPixelCount = static_cast<std::size_t>(patchWidth) * patchHeight;
                const std::size_t paddedCount = static_cast<std::size_t>(paddedWidth) * paddedHeight;
                DeviceBuffer<cufftComplex>* driverSpectrum = nullptr;
                if (!getFieldScalarSourceSpectrumWindow(redistributedDriver,
                                                  patchX,
                                                  patchY,
                                                  patchWidth,
                                                  patchHeight,
                                                  paddedWidth,
                                                  paddedHeight,
                                                  &driverSpectrum)) {
                    return false;
                }
                for (int zoneBase = 0; zoneBase < zoneCount; zoneBase += chunkZoneCount) {
                    const int activeZoneCount = std::min(chunkZoneCount, zoneCount - zoneBase);
                    DeviceBuffer<cufftComplex>* sourceStack = nullptr;
                    DeviceBuffer<cufftComplex>* kernelStack = nullptr;
                    if (!getReplicatedSpectrumStack(*driverSpectrum,
                                        fieldCacheNamespace + ":field-driver-tile:" + std::to_string(patchX) + "," + std::to_string(patchY) +
                                                        ":" + std::to_string(patchWidth) + "x" + std::to_string(patchHeight) + ":" +
                                                        std::to_string(paddedWidth) + "x" + std::to_string(paddedHeight) + ":" +
                                                        std::to_string(activeZoneCount * binCount),
                                                    1,
                                                    activeZoneCount * binCount,
                                                    paddedWidth,
                                                    paddedHeight,
                                                    &sourceStack) ||
                        !buildFieldKernelSpectrumStack(effectKind,
                                                       paddedWidth,
                                                       paddedHeight,
                                                       binCount,
                                                       1,
                                                       zoneBase,
                                                       activeZoneCount,
                                                       &kernelStack)) {
                        return false;
                    }
                    CudaFftPlanLease plan;
                    if (!getPlanForDimsFor(CudaFftSubsystem::FieldRender,
                                           paddedWidth,
                                           paddedHeight,
                                           activeZoneCount * binCount,
                                           &plan)) {
                        return false;
                    }
                    DeviceBuffer<cufftComplex>* tempSpectrum = nullptr;
                    DeviceBuffer<float>* planeStack = nullptr;
                    if (!getComplexScratch("field-tile-temp-spectrum:" + std::to_string(paddedWidth) + "x" +
                                               std::to_string(paddedHeight) + ":" + std::to_string(activeZoneCount * binCount),
                                           paddedCount * static_cast<std::size_t>(activeZoneCount * binCount),
                                           &tempSpectrum) ||
                        !getFloatScratch("field-tile-plane-stack:" + std::to_string(patchWidth) + "x" +
                                             std::to_string(patchHeight) + ":" + std::to_string(activeZoneCount) + ":" +
                                             std::to_string(binCount),
                                         patchPixelCount * static_cast<std::size_t>(activeZoneCount * binCount),
                                         &planeStack)) {
                        return false;
                    }
                    if (!timeCall(timing.convolutionMs, [&] {
                            return convolveSpectrumStackToPlaneStack(*sourceStack,
                                                                     *kernelStack,
                                                                     CudaFftSubsystem::FieldRender,
                                                                     patchWidth,
                                                                     patchHeight,
                                                                     paddedWidth,
                                                                     paddedHeight,
                                                                     activeZoneCount * binCount,
                                                                     &plan,
                                                                     stream,
                                                                     *tempSpectrum,
                                                                     *planeStack,
                                                                     &timing,
                                                                     error);
                        })) {
                        return false;
                    }
                    dim3 tileGrid((activeTileWidth + block2d.x - 1) / block2d.x,
                                  (activeTileHeight + block2d.y - 1) / block2d.y);
                    for (int localZoneIndex = 0; localZoneIndex < activeZoneCount; ++localZoneIndex) {
                        const LensDiffFieldZoneCache* zone =
                            fieldPlan.zones[static_cast<std::size_t>(zoneBase + localZoneIndex)];
                        if (zone == nullptr) {
                            continue;
                        }
                        for (int binIndex = 0; binIndex < binCount; ++binIndex) {
                            float* srcPlane = planeStack->ptr +
                                patchPixelCount * (static_cast<std::size_t>(localZoneIndex) * static_cast<std::size_t>(binCount) +
                                                   static_cast<std::size_t>(binIndex));
                            accumulateWeightedPlaneTileKernel<<<tileGrid, block2d, 0, stream>>>(
                                srcPlane,
                                patchWidth,
                                patchHeight,
                                patchOffsetX,
                                patchOffsetY,
                                activeTileWidth,
                                activeTileHeight,
                                tileX,
                                tileY,
                                width,
                                height,
                                fieldFrameX1,
                                fieldFrameY1,
                                fieldFrameWidth,
                                fieldFrameHeight,
                                zone->zoneX,
                                zone->zoneY,
                                1.0f,
                                weightedPlanes->bins[static_cast<std::size_t>(binIndex)].ptr);
                            if (!checkCuda(cudaGetLastError(), "accumulateWeightedPlaneTileKernel", error)) {
                                return false;
                            }
                        }
                    }
                }
                ++tileCount;
            }
        }
        if (LensDiffTimingEnabled()) {
            LogLensDiffTimingStage(
                "cuda-stage-zone-convolution-stack",
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - convolutionStart)
                    .count(),
                "zones=" + std::to_string(zoneCount) +
                    ",bins=" + std::to_string(binCount) +
                    ",mode=tiled" +
                    ",chunkZones=" + std::to_string(chunkZoneCount) +
                    ",tile=" + std::to_string(tileWidth) + "x" + std::to_string(tileHeight) +
                    ",tiles=" + std::to_string(tileCount));
        }

        if (!allocatePlaneSet(*outPlaneSet)) {
            if (error) *error = "cuda-alloc-field-tiled-effect";
            return false;
        }
        SpectralMapConfigGpu mapConfig {};
        mapConfig.binCount = spectrumConfig.binCount;
        for (int matrixIndex = 0; matrixIndex < kLensDiffMaxSpectralBins * 3; ++matrixIndex) {
            mapConfig.naturalMatrix[static_cast<std::size_t>(matrixIndex)] = spectrumConfig.naturalMatrix[static_cast<std::size_t>(matrixIndex)];
            mapConfig.styleMatrix[static_cast<std::size_t>(matrixIndex)] = spectrumConfig.styleMatrix[static_cast<std::size_t>(matrixIndex)];
        }
        mapSpectralKernel<<<flatGrid, flatBlock, 0, stream>>>(
            weightedPlanes->bins[0].ptr, weightedPlanes->bins[1].ptr, weightedPlanes->bins[2].ptr,
            weightedPlanes->bins[3].ptr, weightedPlanes->bins[4].ptr, weightedPlanes->bins[5].ptr,
            weightedPlanes->bins[6].ptr, weightedPlanes->bins[7].ptr, weightedPlanes->bins[8].ptr,
            pixelCount,
            mapConfig,
            static_cast<float>(std::clamp(params.spectrumForce, 0.0, 1.0)),
            static_cast<float>(std::max(0.0, params.spectrumSaturation)),
            params.chromaticAffectsLuma ? 1 : 0,
            outPlaneSet->r.ptr, outPlaneSet->g.ptr, outPlaneSet->b.ptr);
        return checkCuda(cudaGetLastError(), "mapSpectralKernel-field-tiled", error);
    };
    auto renderFieldZonesStackedSpectral = [&](FieldEffectKind effectKind, PlaneSet* outPlaneSet) -> bool {
        if (outPlaneSet == nullptr || !fieldPlan.canonical3x3 || params.spectralMode == LensDiffSpectralMode::Mono) {
            return false;
        }
        const int zoneCount = static_cast<int>(fieldPlan.zones.size());
        const int binCount = std::min<int>(static_cast<int>(fieldPlan.zones.front()->bins.size()), kLensDiffMaxSpectralBins);
        int maxKernelSize = 1;
        for (const LensDiffFieldZoneCache* zone : fieldPlan.zones) {
            for (int binIndex = 0; binIndex < binCount; ++binIndex) {
                maxKernelSize = std::max(maxKernelSize,
                                         kernelForEffect(zone->bins[static_cast<std::size_t>(binIndex)], effectKind).size);
            }
        }
        const int paddedWidth = nextPowerOfTwo(width + maxKernelSize - 1);
        const int paddedHeight = nextPowerOfTwo(height + maxKernelSize - 1);
        DeviceBuffer<cufftComplex>* driverSpectrum = nullptr;
        DeviceBuffer<cufftComplex>* sourceStack = nullptr;
        DeviceBuffer<cufftComplex>* kernelStack = nullptr;
        if (!getFieldScalarSourceSpectrum(redistributedDriver, paddedWidth, paddedHeight, &driverSpectrum) ||
            !getReplicatedSpectrumStack(*driverSpectrum,
                                        fieldCacheNamespace + ":field-driver:" + std::to_string(paddedWidth) + "x" + std::to_string(paddedHeight) +
                                            ":" + std::to_string(zoneCount * binCount),
                                        1,
                                        zoneCount * binCount,
                                        paddedWidth,
                                        paddedHeight,
                                        &sourceStack) ||
            !buildFieldKernelSpectrumStack(effectKind, paddedWidth, paddedHeight, binCount, 1, 0, zoneCount, &kernelStack)) {
            return false;
        }

        CudaFftPlanLease plan;
        if (!getPlanForDimsFor(CudaFftSubsystem::FieldRender, paddedWidth, paddedHeight, zoneCount * binCount, &plan)) {
            return false;
        }
        DeviceBuffer<cufftComplex>* tempSpectrum = nullptr;
        DeviceBuffer<float>* planeStack = nullptr;
        const std::size_t paddedCount = static_cast<std::size_t>(paddedWidth) * paddedHeight;
        if (!getComplexScratch("field-temp-spectrum:" + std::to_string(paddedWidth) + "x" + std::to_string(paddedHeight) +
                                   ":" + std::to_string(zoneCount * binCount),
                               paddedCount * static_cast<std::size_t>(zoneCount * binCount),
                               &tempSpectrum) ||
            !getFloatScratch("field-plane-stack:" + std::to_string(paddedWidth) + "x" + std::to_string(paddedHeight) +
                                 ":" + std::to_string(zoneCount) + ":" + std::to_string(binCount),
                             pixelCount * static_cast<std::size_t>(zoneCount * binCount),
                             &planeStack)) {
            return false;
        }
        const auto convolutionStart = std::chrono::steady_clock::now();
        if (!timeCall(timing.convolutionMs, [&] {
                return convolveSpectrumStackToPlaneStack(*sourceStack,
                                                         *kernelStack,
                                                         CudaFftSubsystem::FieldRender,
                                                         width,
                                                         height,
                                                         paddedWidth,
                                                         paddedHeight,
                                                         zoneCount * binCount,
                                                         &plan,
                                                         stream,
                                                         *tempSpectrum,
                                                         *planeStack,
                                                         &timing,
                                                         error);
            })) {
            return false;
        }
        if (LensDiffTimingEnabled()) {
            LogLensDiffTimingStage(
                "cuda-stage-zone-convolution-stack",
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - convolutionStart)
                    .count(),
                "zones=" + std::to_string(zoneCount) +
                    ",bins=" + std::to_string(binCount));
        }

        SpectralPlaneSet* weightedPlanes = nullptr;
        if (!getSpectralPlaneScratch("field-weighted-planes:" + std::to_string(binCount), &weightedPlanes)) {
            return false;
        }
        const auto accumulateStart = std::chrono::steady_clock::now();
        for (int binIndex = 0; binIndex < binCount; ++binIndex) {
            accumulateWeightedPlanesStackKernel<<<flatGrid, flatBlock, 0, stream>>>(planeStack->ptr,
                                                                                     pixelCount,
                                                                                     width,
                                                                                     height,
                                                                                     fieldTileOriginX,
                                                                                     fieldTileOriginY,
                                                                                     fieldFrameX1,
                                                                                     fieldFrameY1,
                                                                                     fieldFrameWidth,
                                                                                     fieldFrameHeight,
                                                                                     zoneCount,
                                                                                     binCount,
                                                                                     binIndex,
                                                                                     weightedPlanes->bins[static_cast<std::size_t>(binIndex)].ptr);
            if (!checkCuda(cudaGetLastError(), "accumulateWeightedPlanesStackKernel", error)) {
                return false;
            }
        }
        for (int binIndex = binCount; binIndex < kLensDiffMaxSpectralBins; ++binIndex) {
            if (!checkCuda(cudaMemsetAsync(weightedPlanes->bins[static_cast<std::size_t>(binIndex)].ptr,
                                           0,
                                           pixelCount * sizeof(float),
                                           stream),
                           "cudaMemsetAsync-field-weighted-plane-zero",
                           error)) {
                return false;
            }
        }
        if (LensDiffTimingEnabled()) {
            LogLensDiffTimingStage(
                "cuda-stage-zone-weighted-accumulate",
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - accumulateStart)
                    .count(),
                "zones=" + std::to_string(zoneCount) +
                    ",bins=" + std::to_string(binCount));
        }

        if (!allocatePlaneSet(*outPlaneSet)) {
            if (error) *error = "cuda-alloc-field-stacked-effect";
            return false;
        }
        SpectralMapConfigGpu mapConfig {};
        mapConfig.binCount = spectrumConfig.binCount;
        for (int matrixIndex = 0; matrixIndex < kLensDiffMaxSpectralBins * 3; ++matrixIndex) {
            mapConfig.naturalMatrix[static_cast<std::size_t>(matrixIndex)] = spectrumConfig.naturalMatrix[static_cast<std::size_t>(matrixIndex)];
            mapConfig.styleMatrix[static_cast<std::size_t>(matrixIndex)] = spectrumConfig.styleMatrix[static_cast<std::size_t>(matrixIndex)];
        }
        mapSpectralKernel<<<flatGrid, flatBlock, 0, stream>>>(
            weightedPlanes->bins[0].ptr, weightedPlanes->bins[1].ptr, weightedPlanes->bins[2].ptr,
            weightedPlanes->bins[3].ptr, weightedPlanes->bins[4].ptr, weightedPlanes->bins[5].ptr,
            weightedPlanes->bins[6].ptr, weightedPlanes->bins[7].ptr, weightedPlanes->bins[8].ptr,
            pixelCount,
            mapConfig,
            static_cast<float>(std::clamp(params.spectrumForce, 0.0, 1.0)),
            static_cast<float>(std::max(0.0, params.spectrumSaturation)),
            params.chromaticAffectsLuma ? 1 : 0,
            outPlaneSet->r.ptr, outPlaneSet->g.ptr, outPlaneSet->b.ptr);
        return checkCuda(cudaGetLastError(), "mapSpectralKernel-field-stacked", error);
    };
    auto renderFieldZonesSpectralMicroBatched = [&](FieldEffectKind effectKind, PlaneSet* outPlaneSet) -> bool {
        if (outPlaneSet == nullptr || !fieldPlan.canonical3x3 || params.spectralMode == LensDiffSpectralMode::Mono) {
            return false;
        }
        const int zoneCount = static_cast<int>(fieldPlan.zones.size());
        const int binCount = std::min<int>(static_cast<int>(fieldPlan.zones.front()->bins.size()), kLensDiffMaxSpectralBins);
        int maxKernelSize = 1;
        for (const LensDiffFieldZoneCache* zone : fieldPlan.zones) {
            for (int binIndex = 0; binIndex < binCount; ++binIndex) {
                maxKernelSize = std::max(maxKernelSize,
                                         kernelForEffect(zone->bins[static_cast<std::size_t>(binIndex)], effectKind).size);
            }
        }
        const int paddedWidth = nextPowerOfTwo(width + maxKernelSize - 1);
        const int paddedHeight = nextPowerOfTwo(height + maxKernelSize - 1);
        DeviceBuffer<cufftComplex>* driverSpectrum = nullptr;
        if (!getFieldScalarSourceSpectrum(redistributedDriver, paddedWidth, paddedHeight, &driverSpectrum)) {
            return false;
        }
        SpectralPlaneSet* weightedPlanes = nullptr;
        if (!getSpectralPlaneScratch("field-weighted-planes:" + std::to_string(binCount), &weightedPlanes)) {
            return false;
        }
        for (int binIndex = 0; binIndex < kLensDiffMaxSpectralBins; ++binIndex) {
            if (!checkCuda(cudaMemsetAsync(weightedPlanes->bins[static_cast<std::size_t>(binIndex)].ptr,
                                           0,
                                           pixelCount * sizeof(float),
                                           stream),
                           "cudaMemsetAsync-field-weighted-plane-reset",
                           error)) {
                return false;
            }
        }

        const std::uint64_t perZoneStackBytes = std::max<std::uint64_t>(
            1ULL,
            estimateStackWorkingBytes(paddedWidth, paddedHeight, binCount));
        const std::uint64_t targetBudgetBytes = kFieldStackBudgetBytes * 3ULL / 4ULL;
        int chunkZoneCount = static_cast<int>(targetBudgetBytes / perZoneStackBytes);
        chunkZoneCount = std::max(1, std::min(zoneCount, chunkZoneCount));
        timing.fieldZoneBatchDepth = std::max(timing.fieldZoneBatchDepth, chunkZoneCount);

        const auto convolutionStart = std::chrono::steady_clock::now();
        const std::size_t paddedCount = static_cast<std::size_t>(paddedWidth) * paddedHeight;
        for (int zoneBase = 0; zoneBase < zoneCount; zoneBase += chunkZoneCount) {
            const int activeZoneCount = std::min(chunkZoneCount, zoneCount - zoneBase);
            DeviceBuffer<cufftComplex>* sourceStack = nullptr;
            DeviceBuffer<cufftComplex>* kernelStack = nullptr;
            if (!getReplicatedSpectrumStack(*driverSpectrum,
                                            "field-driver-chunk:" + std::to_string(paddedWidth) + "x" +
                                                std::to_string(paddedHeight) + ":" +
                                                std::to_string(activeZoneCount * binCount),
                                            1,
                                            activeZoneCount * binCount,
                                            paddedWidth,
                                            paddedHeight,
                                            &sourceStack) ||
                !buildFieldKernelSpectrumStack(effectKind,
                                               paddedWidth,
                                               paddedHeight,
                                               binCount,
                                               1,
                                               zoneBase,
                                               activeZoneCount,
                                               &kernelStack)) {
                return false;
            }
            CudaFftPlanLease plan;
            if (!getPlanForDimsFor(CudaFftSubsystem::FieldRender,
                                   paddedWidth,
                                   paddedHeight,
                                   activeZoneCount * binCount,
                                   &plan)) {
                return false;
            }
            DeviceBuffer<cufftComplex>* tempSpectrum = nullptr;
            DeviceBuffer<float>* planeStack = nullptr;
            if (!getComplexScratch("field-micro-temp-spectrum:" + std::to_string(paddedWidth) + "x" +
                                       std::to_string(paddedHeight) + ":" + std::to_string(activeZoneCount * binCount),
                                   paddedCount * static_cast<std::size_t>(activeZoneCount * binCount),
                                   &tempSpectrum) ||
                !getFloatScratch("field-micro-plane-stack:" + std::to_string(paddedWidth) + "x" +
                                     std::to_string(paddedHeight) + ":" + std::to_string(activeZoneCount) + ":" +
                                     std::to_string(binCount),
                                 pixelCount * static_cast<std::size_t>(activeZoneCount * binCount),
                                 &planeStack)) {
                return false;
            }
            if (!timeCall(timing.convolutionMs, [&] {
                    return convolveSpectrumStackToPlaneStack(*sourceStack,
                                                             *kernelStack,
                                                             CudaFftSubsystem::FieldRender,
                                                             width,
                                                             height,
                                                             paddedWidth,
                                                             paddedHeight,
                                                             activeZoneCount * binCount,
                                                             &plan,
                                                             stream,
                                                             *tempSpectrum,
                                                             *planeStack,
                                                             &timing,
                                                             error);
                })) {
                return false;
            }
            for (int localZoneIndex = 0; localZoneIndex < activeZoneCount; ++localZoneIndex) {
                const LensDiffFieldZoneCache* zone = fieldPlan.zones[static_cast<std::size_t>(zoneBase + localZoneIndex)];
                if (zone == nullptr) {
                    continue;
                }
                for (int binIndex = 0; binIndex < binCount; ++binIndex) {
                    float* srcPlane = planeStack->ptr +
                        pixelCount * (static_cast<std::size_t>(localZoneIndex) * static_cast<std::size_t>(binCount) +
                                      static_cast<std::size_t>(binIndex));
                    accumulateWeightedPlaneKernel<<<grid2d, block2d, 0, stream>>>(srcPlane,
                                                                                   width,
                                                                                   height,
                                                                                   fieldTileOriginX,
                                                                                   fieldTileOriginY,
                                                                                   fieldFrameX1,
                                                                                   fieldFrameY1,
                                                                                   fieldFrameWidth,
                                                                                   fieldFrameHeight,
                                                                                   zone->zoneX,
                                                                                   zone->zoneY,
                                                                                   1.0f,
                                                                                   weightedPlanes->bins[static_cast<std::size_t>(binIndex)].ptr);
                    if (!checkCuda(cudaGetLastError(), "accumulateWeightedPlaneKernel", error)) {
                        return false;
                    }
                }
            }
        }
        if (LensDiffTimingEnabled()) {
            LogLensDiffTimingStage(
                "cuda-stage-zone-convolution-stack",
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - convolutionStart)
                    .count(),
                "zones=" + std::to_string(zoneCount) +
                    ",bins=" + std::to_string(binCount) +
                    ",mode=" + std::string(chunkZoneCount > 1 ? "chunked" : "micro") +
                    ",chunkZones=" + std::to_string(chunkZoneCount));
        }

        if (!allocatePlaneSet(*outPlaneSet)) {
            if (error) *error = "cuda-alloc-field-micro-effect";
            return false;
        }
        SpectralMapConfigGpu mapConfig {};
        mapConfig.binCount = spectrumConfig.binCount;
        for (int matrixIndex = 0; matrixIndex < kLensDiffMaxSpectralBins * 3; ++matrixIndex) {
            mapConfig.naturalMatrix[static_cast<std::size_t>(matrixIndex)] = spectrumConfig.naturalMatrix[static_cast<std::size_t>(matrixIndex)];
            mapConfig.styleMatrix[static_cast<std::size_t>(matrixIndex)] = spectrumConfig.styleMatrix[static_cast<std::size_t>(matrixIndex)];
        }
        mapSpectralKernel<<<flatGrid, flatBlock, 0, stream>>>(
            weightedPlanes->bins[0].ptr, weightedPlanes->bins[1].ptr, weightedPlanes->bins[2].ptr,
            weightedPlanes->bins[3].ptr, weightedPlanes->bins[4].ptr, weightedPlanes->bins[5].ptr,
            weightedPlanes->bins[6].ptr, weightedPlanes->bins[7].ptr, weightedPlanes->bins[8].ptr,
            pixelCount,
            mapConfig,
            static_cast<float>(std::clamp(params.spectrumForce, 0.0, 1.0)),
            static_cast<float>(std::max(0.0, params.spectrumSaturation)),
            params.chromaticAffectsLuma ? 1 : 0,
            outPlaneSet->r.ptr, outPlaneSet->g.ptr, outPlaneSet->b.ptr);
        return checkCuda(cudaGetLastError(), "mapSpectralKernel-field-micro", error);
    };
    auto allocatePlaneSetStack = [&](PlaneSet& set, int stackDepth) -> bool {
        const std::size_t stackCount = pixelCount * static_cast<std::size_t>(stackDepth);
        return set.r.allocate(stackCount) && set.g.allocate(stackCount) && set.b.allocate(stackCount);
    };
    auto packPlaneTripletsToRgbStack = [&](const DeviceBuffer<float>& planeStack,
                                           int zoneCount,
                                           PlaneSet* outStack) -> bool {
        if (outStack == nullptr || !allocatePlaneSetStack(*outStack, zoneCount)) {
            if (error) *error = "cuda-alloc-pack-rgb-stack";
            return false;
        }
        const int block = 256;
        const int grid = static_cast<int>(((pixelCount * static_cast<std::size_t>(zoneCount)) + block - 1) / block);
        packPlaneTripletsToRgbStackKernel<<<grid, block, 0, stream>>>(planeStack.ptr,
                                                                      pixelCount,
                                                                      zoneCount,
                                                                      outStack->r.ptr,
                                                                      outStack->g.ptr,
                                                                      outStack->b.ptr);
        return checkCuda(cudaGetLastError(), "packPlaneTripletsToRgbStackKernel", error);
    };
    auto applyShoulderStack = [&](PlaneSet& stack, int stackDepth, float shoulder) -> bool {
        if (shoulder <= 0.0f) {
            return true;
        }
        const std::size_t stackCount = pixelCount * static_cast<std::size_t>(stackDepth);
        const int block = 256;
        const int grid = static_cast<int>((stackCount + block - 1) / block);
        applyShoulderStackKernel<<<grid, block, 0, stream>>>(stack.r.ptr,
                                                             stack.g.ptr,
                                                             stack.b.ptr,
                                                             pixelCount,
                                                             stackDepth,
                                                             shoulder);
        return checkCuda(cudaGetLastError(), "applyShoulderStackKernel", error);
    };
    auto combineRgbStack = [&](const PlaneSet& a,
                               const PlaneSet& b,
                               int stackDepth,
                               PlaneSet* outStack) -> bool {
        if (outStack == nullptr || !allocatePlaneSetStack(*outStack, stackDepth)) {
            if (error) *error = "cuda-alloc-combine-rgb-stack";
            return false;
        }
        const std::size_t stackCount = pixelCount * static_cast<std::size_t>(stackDepth);
        const int block = 256;
        const int grid = static_cast<int>((stackCount + block - 1) / block);
        combineRgbStackKernel<<<grid, block, 0, stream>>>(a.r.ptr,
                                                          a.g.ptr,
                                                          a.b.ptr,
                                                          b.r.ptr,
                                                          b.g.ptr,
                                                          b.b.ptr,
                                                          pixelCount,
                                                          stackDepth,
                                                          static_cast<float>(std::max(0.0, params.coreGain)),
                                                          static_cast<float>(std::max(0.0, params.structureGain)),
                                                          outStack->r.ptr,
                                                          outStack->g.ptr,
                                                          outStack->b.ptr);
        return checkCuda(cudaGetLastError(), "combineRgbStackKernel", error);
    };
    auto accumulateWeightedRgbStack = [&](const PlaneSet& stack,
                                          int zoneCount,
                                          PlaneSet* outSet) -> bool {
        if (outSet == nullptr || !allocatePlaneSet(*outSet)) {
            if (error) *error = "cuda-alloc-accumulate-rgb-stack";
            return false;
        }
        const int block = 256;
        const int grid = static_cast<int>((pixelCount + block - 1) / block);
        accumulateWeightedRgbStackKernel<<<grid, block, 0, stream>>>(stack.r.ptr,
                                                                      stack.g.ptr,
                                                                      stack.b.ptr,
                                                                      width,
                                                                      height,
                                                                      fieldTileOriginX,
                                                                      fieldTileOriginY,
                                                                      fieldFrameX1,
                                                                      fieldFrameY1,
                                                                      fieldFrameWidth,
                                                                      fieldFrameHeight,
                                                                      pixelCount,
                                                                      zoneCount,
                                                                      outSet->r.ptr,
                                                                      outSet->g.ptr,
                                                                     outSet->b.ptr);
        return checkCuda(cudaGetLastError(), "accumulateWeightedRgbStackKernel", error);
    };
    auto mapSpectralPlaneStackToRgbStack = [&](const DeviceBuffer<float>& planeStack,
                                               int zoneCount,
                                               int binCount,
                                               PlaneSet* outStack) -> bool {
        if (outStack == nullptr || !allocatePlaneSetStack(*outStack, zoneCount)) {
            if (error) *error = "cuda-alloc-map-spectral-stack";
            return false;
        }
        SpectralMapConfigGpu mapConfig {};
        mapConfig.binCount = spectrumConfig.binCount;
        for (int matrixIndex = 0; matrixIndex < kLensDiffMaxSpectralBins * 3; ++matrixIndex) {
            mapConfig.naturalMatrix[static_cast<std::size_t>(matrixIndex)] = spectrumConfig.naturalMatrix[static_cast<std::size_t>(matrixIndex)];
            mapConfig.styleMatrix[static_cast<std::size_t>(matrixIndex)] = spectrumConfig.styleMatrix[static_cast<std::size_t>(matrixIndex)];
        }
        const int block = 256;
        const int grid = static_cast<int>(((pixelCount * static_cast<std::size_t>(zoneCount)) + block - 1) / block);
        mapSpectralStackKernel<<<grid, block, 0, stream>>>(planeStack.ptr,
                                                           pixelCount,
                                                           zoneCount,
                                                           binCount,
                                                           mapConfig,
                                                           static_cast<float>(std::clamp(params.spectrumForce, 0.0, 1.0)),
                                                           static_cast<float>(std::max(0.0, params.spectrumSaturation)),
                                                           params.chromaticAffectsLuma ? 1 : 0,
                                                           outStack->r.ptr,
                                                           outStack->g.ptr,
                                                           outStack->b.ptr);
        return checkCuda(cudaGetLastError(), "mapSpectralStackKernel", error);
    };
    auto convolveMonoFieldStack = [&](FieldEffectKind effectKind, PlaneSet* outStack) -> bool {
        if (outStack == nullptr || !fieldPlan.canonical3x3) {
            return false;
        }
        const int zoneCount = static_cast<int>(fieldPlan.zones.size());
        int maxKernelSize = 1;
        for (const LensDiffFieldZoneCache* zone : fieldPlan.zones) {
            maxKernelSize = std::max(maxKernelSize, kernelForEffect(zone->bins.front(), effectKind).size);
        }
        const int paddedWidth = nextPowerOfTwo(width + maxKernelSize - 1);
        const int paddedHeight = nextPowerOfTwo(height + maxKernelSize - 1);
        DeviceBuffer<cufftComplex>* rgbTripletSpectrum = nullptr;
        DeviceBuffer<cufftComplex>* sourceStack = nullptr;
        DeviceBuffer<cufftComplex>* kernelStack = nullptr;
        if (!getRgbTripletSpectrumStack(paddedWidth, paddedHeight, &rgbTripletSpectrum) ||
            !getReplicatedSpectrumStack(*rgbTripletSpectrum,
                                            fieldCacheNamespace + ":field-rgb-triplet-replicated:" + std::to_string(paddedWidth) + "x" +
                                            std::to_string(paddedHeight) + ":" + std::to_string(zoneCount * 3),
                                        3,
                                        zoneCount * 3,
                                        paddedWidth,
                                        paddedHeight,
                                        &sourceStack)) {
            return false;
        }
        if (!buildFieldKernelSpectrumStack(effectKind, paddedWidth, paddedHeight, 1, 3, 0, zoneCount, &kernelStack)) {
            return false;
        }
        CudaFftPlanLease plan;
        if (!getPlanForDimsFor(CudaFftSubsystem::FieldRender, paddedWidth, paddedHeight, zoneCount * 3, &plan)) {
            return false;
        }
        const std::size_t paddedCount = static_cast<std::size_t>(paddedWidth) * paddedHeight;
        DeviceBuffer<cufftComplex>* tempSpectrum = nullptr;
        DeviceBuffer<float>* planeStack = nullptr;
        if (!getComplexScratch("field-mono-temp-spectrum:" + std::to_string(paddedWidth) + "x" +
                                   std::to_string(paddedHeight) + ":" + std::to_string(zoneCount * 3),
                               paddedCount * static_cast<std::size_t>(zoneCount * 3),
                               &tempSpectrum) ||
            !getFloatScratch("field-mono-plane-stack:" + std::to_string(paddedWidth) + "x" +
                                 std::to_string(paddedHeight) + ":" + std::to_string(zoneCount),
                             pixelCount * static_cast<std::size_t>(zoneCount * 3),
                             &planeStack)) {
            return false;
        }
        if (!timeCall(timing.convolutionMs, [&] {
                return convolveSpectrumStackToPlaneStack(*sourceStack,
                                                         *kernelStack,
                                                         CudaFftSubsystem::FieldRender,
                                                         width,
                                                         height,
                                                         paddedWidth,
                                                         paddedHeight,
                                                         zoneCount * 3,
                                                          &plan,
                                                          stream,
                                                          *tempSpectrum,
                                                          *planeStack,
                                                          &timing,
                                                          error);
            })) {
            return false;
        }
        if (!packPlaneTripletsToRgbStack(*planeStack, zoneCount, outStack)) {
            return false;
        }
        return true;
    };
    auto convolvePlaneSetStable = [&](const DeviceBuffer<float>& srcRPlane,
                                      const DeviceBuffer<float>& srcGPlane,
                                      const DeviceBuffer<float>& srcBPlane,
                                      const LensDiffKernel& kernel,
                                      CudaFftSubsystem subsystem,
                                      PlaneSet& dst,
                                      const char* stagePrefix) -> bool {
        const int paddedWidth = nextPowerOfTwo(width + kernel.size - 1);
        const int paddedHeight = nextPowerOfTwo(height + kernel.size - 1);
        CudaFftPlanLease plan;
        DeviceBuffer<cufftComplex> tempSpectrum;
        DeviceBuffer<cufftComplex> imageSpecR;
        DeviceBuffer<cufftComplex> imageSpecG;
        DeviceBuffer<cufftComplex> imageSpecB;
        DeviceBuffer<cufftComplex> kernelSpec;
        DeviceBuffer<float> deviceKernel;
        const bool ok =
            acquireDirectPlanForSubsystem(subsystem, paddedWidth, paddedHeight, 1, &plan) &&
            makeImageSpectrum(srcRPlane.ptr, subsystem, width, height, paddedWidth, paddedHeight, &plan, stream, &timing, imageSpecR, error) &&
            makeImageSpectrum(srcGPlane.ptr, subsystem, width, height, paddedWidth, paddedHeight, &plan, stream, &timing, imageSpecG, error) &&
            makeImageSpectrum(srcBPlane.ptr, subsystem, width, height, paddedWidth, paddedHeight, &plan, stream, &timing, imageSpecB, error) &&
            makeKernelSpectrum(kernel, subsystem, paddedWidth, paddedHeight, &plan, stream, &timing, deviceKernel, kernelSpec, error) &&
            convolveSpectrumToPlane(imageSpecR, kernelSpec, subsystem, width, height, paddedWidth, paddedHeight, &plan, stream, &timing, tempSpectrum, dst.r, error) &&
            convolveSpectrumToPlane(imageSpecG, kernelSpec, subsystem, width, height, paddedWidth, paddedHeight, &plan, stream, &timing, tempSpectrum, dst.g, error) &&
            convolveSpectrumToPlane(imageSpecB, kernelSpec, subsystem, width, height, paddedWidth, paddedHeight, &plan, stream, &timing, tempSpectrum, dst.b, error);
        if (!ok && error && error->empty()) {
            *error = stagePrefix;
        }
        return ok;
    };
    auto renderFromBinsStable = [&](CudaFftSubsystem subsystem,
                                    const std::vector<LensDiffPsfBin>& bins,
                                    PlaneSet& outEffect,
                                    PlaneSet* outCore,
                                    PlaneSet* outStructure) -> bool {
        if (!allocatePlaneSet(outEffect)) {
            if (error) *error = "cuda-alloc-zone-effect";
            return false;
        }
        const bool localNeedCore = outCore != nullptr;
        const bool localNeedStructure = outStructure != nullptr;
        if ((localNeedCore && !allocatePlaneSet(*outCore)) ||
            (localNeedStructure && !allocatePlaneSet(*outStructure))) {
            if (error) *error = "cuda-alloc-zone-split";
            return false;
        }

        if (params.spectralMode == LensDiffSpectralMode::Mono) {
            PlaneSet fullEffect;
            if (!splitMode && !allocatePlaneSet(fullEffect)) {
                if (error) *error = "cuda-alloc-zone-full";
                return false;
            }

            if (!splitMode &&
                !convolvePlaneSetStable(redistributedR, redistributedG, redistributedB, bins.front().full, subsystem, fullEffect, "cuda-convolve-zone-full")) {
                return false;
            }
            if (localNeedCore &&
                !convolvePlaneSetStable(redistributedR, redistributedG, redistributedB, bins.front().core, subsystem, *outCore, "cuda-convolve-zone-core")) {
                return false;
            }
            if (localNeedStructure &&
                !convolvePlaneSetStable(redistributedR, redistributedG, redistributedB, bins.front().structure, subsystem, *outStructure, "cuda-convolve-zone-structure")) {
                return false;
            }

            if (splitMode) {
                if (params.coreShoulder > 0.0) {
                    applyShoulderKernel<<<flatGrid, flatBlock, 0, stream>>>(
                        outCore->r.ptr, outCore->g.ptr, outCore->b.ptr, pixelCount, static_cast<float>(params.coreShoulder));
                    if (!checkCuda(cudaGetLastError(), "applyShoulderKernel-zone-core", error)) {
                        return false;
                    }
                }
                if (params.structureShoulder > 0.0) {
                    applyShoulderKernel<<<flatGrid, flatBlock, 0, stream>>>(
                        outStructure->r.ptr, outStructure->g.ptr, outStructure->b.ptr, pixelCount, static_cast<float>(params.structureShoulder));
                    if (!checkCuda(cudaGetLastError(), "applyShoulderKernel-zone-structure", error)) {
                        return false;
                    }
                }
                combineRgbKernel<<<flatGrid, flatBlock, 0, stream>>>(
                    outCore->r.ptr, outCore->g.ptr, outCore->b.ptr,
                    outStructure->r.ptr, outStructure->g.ptr, outStructure->b.ptr,
                    pixelCount,
                    static_cast<float>(std::max(0.0, params.coreGain)),
                    static_cast<float>(std::max(0.0, params.structureGain)),
                    outEffect.r.ptr, outEffect.g.ptr, outEffect.b.ptr);
                return checkCuda(cudaGetLastError(), "combineRgbKernel-zone-mono", error);
            }
            return copyPlaneSet(fullEffect, outEffect, "cudaMemcpyAsync-zone-effect");
        }

        const LensDiffSpectrumConfig zoneSpectrumConfig = BuildLensDiffSpectrumConfig(params, bins);
        const int paddedWidth = nextPowerOfTwo(width + bins.front().full.size - 1);
        const int paddedHeight = nextPowerOfTwo(height + bins.front().full.size - 1);
        CudaFftPlanLease plan;
        DeviceBuffer<cufftComplex> tempSpectrum;
        DeviceBuffer<cufftComplex> driverSpectrum;
        DeviceBuffer<float> kernelBuffer;
        SpectralPlaneSet fullBins;
        SpectralPlaneSet coreBins;
        SpectralPlaneSet structureBins;

        auto allocateSpectral = [&](SpectralPlaneSet& set) -> bool {
            for (auto& plane : set.bins) {
                if (!plane.allocate(pixelCount)) {
                    return false;
                }
            }
            return true;
        };
        auto convolveBinsStable = [&](const std::vector<LensDiffKernel>& kernels, SpectralPlaneSet& dst) -> bool {
            const int activeBins = std::min<int>(static_cast<int>(kernels.size()), kLensDiffMaxSpectralBins);
            for (int i = 0; i < activeBins; ++i) {
                DeviceBuffer<cufftComplex> tempKernelSpectrum;
                if (!makeKernelSpectrum(kernels[static_cast<std::size_t>(i)],
                                        subsystem,
                                        paddedWidth,
                                        paddedHeight,
                                        &plan,
                                        stream,
                                        &timing,
                                        kernelBuffer,
                                        tempKernelSpectrum,
                                        error) ||
                    !convolveSpectrumToPlane(driverSpectrum,
                                             tempKernelSpectrum,
                                             subsystem,
                                             width,
                                             height,
                                             paddedWidth,
                                             paddedHeight,
                                             &plan,
                                             stream,
                                             &timing,
                                             tempSpectrum,
                                             dst.bins[static_cast<std::size_t>(i)],
                                             error)) {
                    return false;
                }
            }
            for (int i = activeBins; i < kLensDiffMaxSpectralBins; ++i) {
                if (!checkCuda(cudaMemsetAsync(dst.bins[static_cast<std::size_t>(i)].ptr,
                                               0,
                                               pixelCount * sizeof(float),
                                               stream),
                               "cudaMemsetAsync-zone-spectral-zero",
                               error)) {
                    return false;
                }
            }
            return true;
        };
        auto mapBinsStable = [&](const SpectralPlaneSet& srcBins, PlaneSet& dst) -> bool {
            SpectralMapConfigGpu mapConfig {};
            mapConfig.binCount = zoneSpectrumConfig.binCount;
            for (int i = 0; i < kLensDiffMaxSpectralBins * 3; ++i) {
                mapConfig.naturalMatrix[static_cast<std::size_t>(i)] = zoneSpectrumConfig.naturalMatrix[static_cast<std::size_t>(i)];
                mapConfig.styleMatrix[static_cast<std::size_t>(i)] = zoneSpectrumConfig.styleMatrix[static_cast<std::size_t>(i)];
            }
            mapSpectralKernel<<<flatGrid, flatBlock, 0, stream>>>(
                srcBins.bins[0].ptr, srcBins.bins[1].ptr, srcBins.bins[2].ptr,
                srcBins.bins[3].ptr, srcBins.bins[4].ptr, srcBins.bins[5].ptr,
                srcBins.bins[6].ptr, srcBins.bins[7].ptr, srcBins.bins[8].ptr,
                pixelCount,
                mapConfig,
                static_cast<float>(std::clamp(params.spectrumForce, 0.0, 1.0)),
                static_cast<float>(std::max(0.0, params.spectrumSaturation)),
                params.chromaticAffectsLuma ? 1 : 0,
                dst.r.ptr, dst.g.ptr, dst.b.ptr);
            return checkCuda(cudaGetLastError(), "mapSpectralKernel-zone", error);
        };

        if (!acquireDirectPlanForSubsystem(subsystem, paddedWidth, paddedHeight, 1, &plan)) {
            return false;
        }
        bool ok = makeImageSpectrum(redistributedDriver.ptr,
                                    subsystem,
                                    width,
                                    height,
                                    paddedWidth,
                                    paddedHeight,
                                    &plan,
                                    stream,
                                    &timing,
                                    driverSpectrum,
                                    error);
        if (ok && !splitMode) {
            ok = allocateSpectral(fullBins);
        }
        if (ok && localNeedCore) {
            ok = allocateSpectral(coreBins);
        }
        if (ok && localNeedStructure) {
            ok = allocateSpectral(structureBins);
        }
        if (!ok) {
            if (error && error->empty()) *error = "cuda-alloc-zone-spectral";
            return false;
        }

        if (ok && !splitMode) {
            std::vector<LensDiffKernel> kernels;
            kernels.reserve(bins.size());
            for (const auto& bin : bins) kernels.push_back(bin.full);
            ok = convolveBinsStable(kernels, fullBins);
        }
        if (ok && localNeedCore) {
            std::vector<LensDiffKernel> kernels;
            kernels.reserve(bins.size());
            for (const auto& bin : bins) kernels.push_back(bin.core);
            ok = convolveBinsStable(kernels, coreBins);
        }
        if (ok && localNeedStructure) {
            std::vector<LensDiffKernel> kernels;
            kernels.reserve(bins.size());
            for (const auto& bin : bins) kernels.push_back(bin.structure);
            ok = convolveBinsStable(kernels, structureBins);
        }
        if (ok && !splitMode) {
            ok = mapBinsStable(fullBins, outEffect);
        }
        if (ok && localNeedCore) {
            ok = mapBinsStable(coreBins, *outCore);
        }
        if (ok && localNeedStructure) {
            ok = mapBinsStable(structureBins, *outStructure);
        }
        if (!ok) {
            return false;
        }

        if (splitMode) {
            if (params.coreShoulder > 0.0) {
                applyShoulderKernel<<<flatGrid, flatBlock, 0, stream>>>(
                    outCore->r.ptr, outCore->g.ptr, outCore->b.ptr, pixelCount, static_cast<float>(params.coreShoulder));
                if (!checkCuda(cudaGetLastError(), "applyShoulderKernel-zone-core-spectral", error)) {
                    return false;
                }
            }
            if (params.structureShoulder > 0.0) {
                applyShoulderKernel<<<flatGrid, flatBlock, 0, stream>>>(
                    outStructure->r.ptr, outStructure->g.ptr, outStructure->b.ptr, pixelCount, static_cast<float>(params.structureShoulder));
                if (!checkCuda(cudaGetLastError(), "applyShoulderKernel-zone-structure-spectral", error)) {
                    return false;
                }
            }
            combineRgbKernel<<<flatGrid, flatBlock, 0, stream>>>(
                outCore->r.ptr, outCore->g.ptr, outCore->b.ptr,
                outStructure->r.ptr, outStructure->g.ptr, outStructure->b.ptr,
                pixelCount,
                static_cast<float>(std::max(0.0, params.coreGain)),
                static_cast<float>(std::max(0.0, params.structureGain)),
                outEffect.r.ptr, outEffect.g.ptr, outEffect.b.ptr);
            ok = checkCuda(cudaGetLastError(), "combineRgbKernel-zone-spectral", error);
        }
        return ok;
    };
    auto renderFieldZonesStackedMono = [&](FieldEffectKind effectKind, PlaneSet* outPlaneSet) -> bool {
        PlaneSet* rgbStack = nullptr;
        const int zoneCount = static_cast<int>(fieldPlan.zones.size());
        if (!getPlaneSetScratch("field-mono-rgb-stack:" + std::to_string(static_cast<int>(effectKind)),
                                pixelCount * static_cast<std::size_t>(zoneCount),
                                &rgbStack)) {
            return false;
        }
        return convolveMonoFieldStack(effectKind, rgbStack) &&
               accumulateWeightedRgbStack(*rgbStack, zoneCount, outPlaneSet);
    };
    auto renderFieldZonesStackedSplit = [&](PlaneSet* outEffect, PlaneSet* outCore, PlaneSet* outStructure) -> bool {
        if (!fieldPlan.canonical3x3 || outEffect == nullptr) {
            return false;
        }
        const int zoneCount = static_cast<int>(fieldPlan.zones.size());
        PlaneSet* coreStack = nullptr;
        PlaneSet* structureStack = nullptr;
        if (!getPlaneSetScratch("field-split-core-stack", pixelCount * static_cast<std::size_t>(zoneCount), &coreStack) ||
            !getPlaneSetScratch("field-split-structure-stack", pixelCount * static_cast<std::size_t>(zoneCount), &structureStack)) {
            return false;
        }
        if (params.spectralMode == LensDiffSpectralMode::Mono) {
            if (!convolveMonoFieldStack(FieldEffectKind::Core, coreStack) ||
                !convolveMonoFieldStack(FieldEffectKind::Structure, structureStack) ||
                !applyShoulderStack(*coreStack, zoneCount, static_cast<float>(params.coreShoulder)) ||
                !applyShoulderStack(*structureStack, zoneCount, static_cast<float>(params.structureShoulder))) {
                return false;
            }
            PlaneSet* effectStack = nullptr;
            if (!getPlaneSetScratch("field-split-effect-stack", pixelCount * static_cast<std::size_t>(zoneCount), &effectStack) ||
                !combineRgbStack(*coreStack, *structureStack, zoneCount, effectStack) ||
                !accumulateWeightedRgbStack(*effectStack, zoneCount, outEffect)) {
                return false;
            }
            if (outCore != nullptr && !accumulateWeightedRgbStack(*coreStack, zoneCount, outCore)) {
                return false;
            }
            if (outStructure != nullptr && !accumulateWeightedRgbStack(*structureStack, zoneCount, outStructure)) {
                return false;
            }
            return true;
        }
        DeviceBuffer<float>* corePlaneStack = nullptr;
        DeviceBuffer<float>* structurePlaneStack = nullptr;
        int binCount = std::min<int>(static_cast<int>(fieldPlan.zones.front()->bins.size()), kLensDiffMaxSpectralBins);
        auto convolveSpectralStack = [&](FieldEffectKind effectKind, DeviceBuffer<float>* outPlaneStack) -> bool {
            int maxKernelSize = 1;
            for (const LensDiffFieldZoneCache* zone : fieldPlan.zones) {
                for (int binIndex = 0; binIndex < binCount; ++binIndex) {
                    maxKernelSize = std::max(maxKernelSize, kernelForEffect(zone->bins[static_cast<std::size_t>(binIndex)], effectKind).size);
                }
            }
            const int paddedWidth = nextPowerOfTwo(width + maxKernelSize - 1);
            const int paddedHeight = nextPowerOfTwo(height + maxKernelSize - 1);
            DeviceBuffer<cufftComplex>* driverSpectrum = nullptr;
            DeviceBuffer<cufftComplex>* sourceStack = nullptr;
            DeviceBuffer<cufftComplex>* kernelStack = nullptr;
            if (!getFieldScalarSourceSpectrum(redistributedDriver, paddedWidth, paddedHeight, &driverSpectrum) ||
                !getReplicatedSpectrumStack(*driverSpectrum,
                                            fieldCacheNamespace + ":field-split-driver:" + std::to_string(paddedWidth) + "x" + std::to_string(paddedHeight) +
                                                ":" + std::to_string(zoneCount * binCount),
                                            1,
                                            zoneCount * binCount,
                                            paddedWidth,
                                            paddedHeight,
                                            &sourceStack) ||
                !buildFieldKernelSpectrumStack(effectKind, paddedWidth, paddedHeight, binCount, 1, 0, zoneCount, &kernelStack)) {
                return false;
            }
            CudaFftPlanLease plan;
            if (!getPlanForDimsFor(CudaFftSubsystem::FieldRender, paddedWidth, paddedHeight, zoneCount * binCount, &plan)) {
                return false;
            }
            const std::size_t paddedCount = static_cast<std::size_t>(paddedWidth) * paddedHeight;
            DeviceBuffer<cufftComplex>* tempSpectrum = nullptr;
            if (!getComplexScratch("field-split-temp-spectrum:" + std::to_string(static_cast<int>(effectKind)) + ":" +
                                       std::to_string(paddedWidth) + "x" + std::to_string(paddedHeight) + ":" +
                                       std::to_string(zoneCount * binCount),
                                   paddedCount * static_cast<std::size_t>(zoneCount * binCount),
                                   &tempSpectrum)) {
                return false;
            }
            return timeCall(timing.convolutionMs, [&] {
                return convolveSpectrumStackToPlaneStack(*sourceStack,
                                                         *kernelStack,
                                                         CudaFftSubsystem::FieldRender,
                                                         width,
                                                         height,
                                                         paddedWidth,
                                                         paddedHeight,
                                                         zoneCount * binCount,
                                                         &plan,
                                                         stream,
                                                         *tempSpectrum,
                                                         *outPlaneStack,
                                                         &timing,
                                                         error);
            });
        };
        if (!getFloatScratch("field-split-plane-stack-core:" + std::to_string(binCount),
                             pixelCount * static_cast<std::size_t>(zoneCount * binCount),
                             &corePlaneStack) ||
            !getFloatScratch("field-split-plane-stack-structure:" + std::to_string(binCount),
                             pixelCount * static_cast<std::size_t>(zoneCount * binCount),
                             &structurePlaneStack) ||
            !convolveSpectralStack(FieldEffectKind::Core, corePlaneStack) ||
            !convolveSpectralStack(FieldEffectKind::Structure, structurePlaneStack) ||
            !mapSpectralPlaneStackToRgbStack(*corePlaneStack, zoneCount, binCount, coreStack) ||
            !mapSpectralPlaneStackToRgbStack(*structurePlaneStack, zoneCount, binCount, structureStack) ||
            !applyShoulderStack(*coreStack, zoneCount, static_cast<float>(params.coreShoulder)) ||
            !applyShoulderStack(*structureStack, zoneCount, static_cast<float>(params.structureShoulder))) {
            return false;
        }
        PlaneSet* effectStack = nullptr;
        if (!getPlaneSetScratch("field-split-effect-stack", pixelCount * static_cast<std::size_t>(zoneCount), &effectStack) ||
            !combineRgbStack(*coreStack, *structureStack, zoneCount, effectStack) ||
            !accumulateWeightedRgbStack(*effectStack, zoneCount, outEffect)) {
            return false;
        }
        if (outCore != nullptr && !accumulateWeightedRgbStack(*coreStack, zoneCount, outCore)) {
            return false;
        }
        if (outStructure != nullptr && !accumulateWeightedRgbStack(*structureStack, zoneCount, outStructure)) {
            return false;
        }
        return true;
    };
    auto accumulateFieldZone = [&](const PlaneSet& src, int zoneX, int zoneY, float gain, PlaneSet& dst) -> bool {
        accumulateWeightedRgbLegacyKernel<<<grid2d, block2d, 0, stream>>>(
            src.r.ptr, src.g.ptr, src.b.ptr,
            width, height,
            zoneX, zoneY, gain,
            dst.r.ptr, dst.g.ptr, dst.b.ptr);
        return checkCuda(cudaGetLastError(), "accumulateWeightedRgbLegacyKernel-field-zone", error);
    };
    auto renderFieldZonesLegacy = [&](PlaneSet* outEffect, PlaneSet* outCore, PlaneSet* outStructure) -> bool {
        if (outEffect == nullptr) {
            return false;
        }
        if (!allocatePlaneSet(*outEffect) ||
            (outCore != nullptr && !allocatePlaneSet(*outCore)) ||
            (outStructure != nullptr && !allocatePlaneSet(*outStructure))) {
            if (error) *error = "cuda-alloc-field-accum";
            return false;
        }
        if (!clearPlaneSet(*outEffect, "cudaMemsetAsync-field-effect") ||
            (outCore != nullptr && !clearPlaneSet(*outCore, "cudaMemsetAsync-field-core")) ||
            (outStructure != nullptr && !clearPlaneSet(*outStructure, "cudaMemsetAsync-field-structure"))) {
            return false;
        }
        for (const auto& zone : cache.fieldZones) {
            PlaneSet zoneEffect;
            PlaneSet zoneCore;
            PlaneSet zoneStructure;
            if (!renderFromBinsStable(CudaFftSubsystem::FieldRender,
                                      zone.bins,
                                      zoneEffect,
                                      outCore != nullptr ? &zoneCore : nullptr,
                                      outStructure != nullptr ? &zoneStructure : nullptr)) {
                return false;
            }
            if (!accumulateFieldZone(zoneEffect, zone.zoneX, zone.zoneY, 1.0f, *outEffect) ||
                (outCore != nullptr && !accumulateFieldZone(zoneCore, zone.zoneX, zone.zoneY, 1.0f, *outCore)) ||
                (outStructure != nullptr && !accumulateFieldZone(zoneStructure, zone.zoneX, zone.zoneY, 1.0f, *outStructure))) {
                return false;
            }
        }
        return true;
    };
    auto downloadDeviceBuffer = [&](const DeviceBuffer<float>& buffer, std::vector<float>* host) -> bool {
        if (host == nullptr) {
            return false;
        }
        host->assign(buffer.count, 0.0f);
        if (buffer.count == 0) {
            return true;
        }
        if (!checkCuda(cudaMemcpyAsync(host->data(),
                                       buffer.ptr,
                                       buffer.count * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       stream),
                       "cudaMemcpyAsync-download-validation",
                       error)) {
            return false;
        }
        ++timing.hostSyncCount;
        return checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize-validation", error);
    };
    auto comparePlaneBuffers = [&](const DeviceBuffer<float>& fast,
                                   const DeviceBuffer<float>& reference,
                                   float* outMaxAbs,
                                   std::uint64_t* outFastHash,
                                   std::uint64_t* outReferenceHash) -> bool {
        std::vector<float> fastHost;
        std::vector<float> referenceHost;
        if (!downloadDeviceBuffer(fast, &fastHost) || !downloadDeviceBuffer(reference, &referenceHost)) {
            return false;
        }
        float maxAbs = 0.0f;
        const std::size_t count = std::min(fastHost.size(), referenceHost.size());
        for (std::size_t index = 0; index < count; ++index) {
            maxAbs = std::max(maxAbs, std::abs(fastHost[index] - referenceHost[index]));
        }
        if (outMaxAbs != nullptr) {
            *outMaxAbs = maxAbs;
        }
        if (outFastHash != nullptr) {
            *outFastHash = hashBytesFnv1a64(fastHost.data(), fastHost.size() * sizeof(float));
        }
        if (outReferenceHash != nullptr) {
            *outReferenceHash = hashBytesFnv1a64(referenceHost.data(), referenceHost.size() * sizeof(float));
        }
        return true;
    };
    auto comparePlaneSets = [&](const PlaneSet& fast,
                                const PlaneSet& reference,
                                float* outMaxAbs,
                                const char* label) -> bool {
        float maxAbs = 0.0f;
        std::uint64_t fastHashR = 0;
        std::uint64_t fastHashG = 0;
        std::uint64_t fastHashB = 0;
        std::uint64_t referenceHashR = 0;
        std::uint64_t referenceHashG = 0;
        std::uint64_t referenceHashB = 0;
        float planeMaxR = 0.0f;
        float planeMaxG = 0.0f;
        float planeMaxB = 0.0f;
        if (!comparePlaneBuffers(fast.r, reference.r, &planeMaxR, &fastHashR, &referenceHashR) ||
            !comparePlaneBuffers(fast.g, reference.g, &planeMaxG, &fastHashG, &referenceHashG) ||
            !comparePlaneBuffers(fast.b, reference.b, &planeMaxB, &fastHashB, &referenceHashB)) {
            return false;
        }
        maxAbs = std::max(planeMaxR, std::max(planeMaxG, planeMaxB));
        if (outMaxAbs != nullptr) {
            *outMaxAbs = std::max(*outMaxAbs, maxAbs);
        }
        if (!timing.validationNote.empty()) {
            timing.validationNote += ",";
        }
        timing.validationNote += std::string(label) + "HashFast=" + std::to_string(static_cast<unsigned long long>(fastHashR ^ fastHashG ^ fastHashB));
        timing.validationNote += "," + std::string(label) + "HashLegacy=" + std::to_string(static_cast<unsigned long long>(referenceHashR ^ referenceHashG ^ referenceHashB));
        return true;
    };
    auto estimateStackedFieldBytes = [&]() -> std::uint64_t {
        if (!fieldPlan.canonical3x3) {
            return 0;
        }
        const int zoneCount = static_cast<int>(fieldPlan.zones.size());
        if (splitMode) {
            if (params.spectralMode == LensDiffSpectralMode::Mono) {
                int maxCoreKernel = 1;
                int maxStructureKernel = 1;
                for (const LensDiffFieldZoneCache* zone : fieldPlan.zones) {
                    maxCoreKernel = std::max(maxCoreKernel, kernelForEffect(zone->bins.front(), FieldEffectKind::Core).size);
                    maxStructureKernel = std::max(maxStructureKernel, kernelForEffect(zone->bins.front(), FieldEffectKind::Structure).size);
                }
                const int paddedWidthCore = nextPowerOfTwo(width + maxCoreKernel - 1);
                const int paddedHeightCore = nextPowerOfTwo(height + maxCoreKernel - 1);
                const int paddedWidthStructure = nextPowerOfTwo(width + maxStructureKernel - 1);
                const int paddedHeightStructure = nextPowerOfTwo(height + maxStructureKernel - 1);
                return estimateStackWorkingBytes(paddedWidthCore, paddedHeightCore, zoneCount * 3) +
                       estimateStackWorkingBytes(paddedWidthStructure, paddedHeightStructure, zoneCount * 3) +
                       static_cast<std::uint64_t>(pixelCount) * static_cast<std::uint64_t>(zoneCount) * sizeof(float) * 9ULL;
            }
            const int binCount = std::min<int>(static_cast<int>(fieldPlan.zones.front()->bins.size()), kLensDiffMaxSpectralBins);
            int maxCoreKernel = 1;
            int maxStructureKernel = 1;
            for (const LensDiffFieldZoneCache* zone : fieldPlan.zones) {
                for (int binIndex = 0; binIndex < binCount; ++binIndex) {
                    maxCoreKernel = std::max(maxCoreKernel, kernelForEffect(zone->bins[static_cast<std::size_t>(binIndex)], FieldEffectKind::Core).size);
                    maxStructureKernel = std::max(maxStructureKernel, kernelForEffect(zone->bins[static_cast<std::size_t>(binIndex)], FieldEffectKind::Structure).size);
                }
            }
            const int paddedWidthCore = nextPowerOfTwo(width + maxCoreKernel - 1);
            const int paddedHeightCore = nextPowerOfTwo(height + maxCoreKernel - 1);
            const int paddedWidthStructure = nextPowerOfTwo(width + maxStructureKernel - 1);
            const int paddedHeightStructure = nextPowerOfTwo(height + maxStructureKernel - 1);
            return estimateStackWorkingBytes(paddedWidthCore, paddedHeightCore, zoneCount * binCount) +
                   estimateStackWorkingBytes(paddedWidthStructure, paddedHeightStructure, zoneCount * binCount) +
                   static_cast<std::uint64_t>(pixelCount) * static_cast<std::uint64_t>(zoneCount) * sizeof(float) * 9ULL;
        }
        if (params.spectralMode == LensDiffSpectralMode::Mono) {
            int maxKernelSize = 1;
            for (const LensDiffFieldZoneCache* zone : fieldPlan.zones) {
                maxKernelSize = std::max(maxKernelSize, kernelForEffect(zone->bins.front(), FieldEffectKind::Full).size);
            }
            const int paddedWidth = nextPowerOfTwo(width + maxKernelSize - 1);
            const int paddedHeight = nextPowerOfTwo(height + maxKernelSize - 1);
            return estimateStackWorkingBytes(paddedWidth, paddedHeight, zoneCount * 3) +
                   static_cast<std::uint64_t>(pixelCount) * static_cast<std::uint64_t>(zoneCount) * sizeof(float) * 3ULL;
        }
        const int binCount = std::min<int>(static_cast<int>(fieldPlan.zones.front()->bins.size()), kLensDiffMaxSpectralBins);
        int maxKernelSize = 1;
        for (const LensDiffFieldZoneCache* zone : fieldPlan.zones) {
            for (int binIndex = 0; binIndex < binCount; ++binIndex) {
                maxKernelSize = std::max(maxKernelSize,
                                         kernelForEffect(zone->bins[static_cast<std::size_t>(binIndex)], FieldEffectKind::Full).size);
            }
        }
        const int paddedWidth = nextPowerOfTwo(width + maxKernelSize - 1);
        const int paddedHeight = nextPowerOfTwo(height + maxKernelSize - 1);
        return estimateStackWorkingBytes(paddedWidth, paddedHeight, zoneCount * binCount);
    };
    const std::uint64_t estimatedFieldStackBytes = estimateStackedFieldBytes();
    const bool preferTiledSpectralField =
        stackedFieldEnabled &&
        LensDiffCudaExperimentalTiledFieldEnabled() &&
        fieldPlan.canonical3x3 &&
        !splitMode &&
        params.spectralMode != LensDiffSpectralMode::Mono &&
        estimatedFieldStackBytes > kFieldStackBudgetBytes;
    const bool preferLegacySizedFieldExecution =
        stackedFieldEnabled &&
        fieldPlan.canonical3x3 &&
        !preferTiledSpectralField &&
        estimatedFieldStackBytes > kFieldStackBudgetBytes;
    timing.fieldScratchEstimateBytes = estimatedFieldStackBytes;

    PlaneSet effect;
    PlaneSet coreEffect;
    PlaneSet structureEffect;
    const bool shouldRunFieldValidation =
        validateField && !cache.fieldZones.empty() && !legacyPipeline;
    if (cache.fieldZones.empty()) {
        timing.fieldBranch = "global";
        const bool useStableGlobalSplitPath = splitMode;
        if (!(useStableGlobalSplitPath
                  ? renderFromBinsStable(CudaFftSubsystem::GlobalRender,
                                         cache.bins,
                                         effect,
                                         needCore ? &coreEffect : nullptr,
                                         needStructure ? &structureEffect : nullptr)
                  : renderFromBins(cache.bins,
                                   effect,
                                   needCore ? &coreEffect : nullptr,
                                   needStructure ? &structureEffect : nullptr))) {
            return false;
        }
    } else if (stackedFieldEnabled && fieldPlan.canonical3x3 && splitMode && !preferLegacySizedFieldExecution) {
        timing.fieldBranch = "stacked-split";
        const auto fieldZonesStart = std::chrono::steady_clock::now();
        timing.fieldZoneBatchDepth = std::max(timing.fieldZoneBatchDepth, static_cast<int>(fieldPlan.zones.size()));
        if (!renderFieldZonesStackedSplit(&effect, needCore ? &coreEffect : nullptr, needStructure ? &structureEffect : nullptr)) {
            return false;
        }
        timing.fieldZonesMs += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                                    std::chrono::steady_clock::now() - fieldZonesStart)
                                    .count();
    } else if (preferTiledSpectralField) {
        timing.fieldBranch = "tiled-spectral";
        const auto fieldZonesStart = std::chrono::steady_clock::now();
        if (!renderFieldZonesSpectralTiled(FieldEffectKind::Full, &effect) ||
            (needCore && !renderFieldZonesSpectralTiled(FieldEffectKind::Core, &coreEffect)) ||
            (needStructure && !renderFieldZonesSpectralTiled(FieldEffectKind::Structure, &structureEffect))) {
            return false;
        }
        if (tiledFieldWorkingBytes > 0) {
            timing.fieldScratchEstimateBytes = tiledFieldWorkingBytes;
        }
        timing.fieldZonesMs += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                                    std::chrono::steady_clock::now() - fieldZonesStart)
                                    .count();
    } else if (stackedFieldEnabled && fieldPlan.canonical3x3 && params.spectralMode == LensDiffSpectralMode::Mono && !preferLegacySizedFieldExecution) {
        timing.fieldBranch = "stacked-mono";
        const auto fieldZonesStart = std::chrono::steady_clock::now();
        timing.fieldZoneBatchDepth = std::max(timing.fieldZoneBatchDepth, static_cast<int>(fieldPlan.zones.size()));
        if (!renderFieldZonesStackedMono(FieldEffectKind::Full, &effect) ||
            (needCore && !renderFieldZonesStackedMono(FieldEffectKind::Core, &coreEffect)) ||
            (needStructure && !renderFieldZonesStackedMono(FieldEffectKind::Structure, &structureEffect))) {
            return false;
        }
        timing.fieldZonesMs += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                                    std::chrono::steady_clock::now() - fieldZonesStart)
                                    .count();
    } else if (stackedFieldEnabled && fieldPlan.canonical3x3 && !preferLegacySizedFieldExecution) {
        timing.fieldBranch = "stacked-spectral";
        const auto fieldZonesStart = std::chrono::steady_clock::now();
        timing.fieldZoneBatchDepth = std::max(timing.fieldZoneBatchDepth, static_cast<int>(fieldPlan.zones.size()));
        if (!renderFieldZonesStackedSpectral(FieldEffectKind::Full, &effect) ||
            (needCore && !renderFieldZonesStackedSpectral(FieldEffectKind::Core, &coreEffect)) ||
            (needStructure && !renderFieldZonesStackedSpectral(FieldEffectKind::Structure, &structureEffect))) {
            return false;
        }
        timing.fieldZonesMs += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                                    std::chrono::steady_clock::now() - fieldZonesStart)
                                    .count();
    } else {
        timing.fieldBranch = "legacy";
        const auto fieldZonesStart = std::chrono::steady_clock::now();
        timing.fieldZoneBatchDepth = std::max(timing.fieldZoneBatchDepth, static_cast<int>(cache.fieldZones.size()));
        if (!renderFieldZonesLegacy(&effect, needCore ? &coreEffect : nullptr, needStructure ? &structureEffect : nullptr)) {
            return false;
        }
        timing.fieldZonesMs += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                                   std::chrono::steady_clock::now() - fieldZonesStart)
                                   .count();
    }
    if (shouldRunFieldValidation && timing.fieldBranch != "legacy" && timing.fieldBranch != "global") {
        PlaneSet legacyEffect;
        PlaneSet legacyCore;
        PlaneSet legacyStructure;
        const auto validationStart = std::chrono::steady_clock::now();
        timing.validationReferenceLegacy = true;
        if (!renderFieldZonesLegacy(&legacyEffect, needCore ? &legacyCore : nullptr, needStructure ? &legacyStructure : nullptr) ||
            !comparePlaneSets(effect, legacyEffect, &timing.validationEffectMaxAbs, "effect") ||
            (needCore && !comparePlaneSets(coreEffect, legacyCore, &timing.validationCoreMaxAbs, "core")) ||
            (needStructure && !comparePlaneSets(structureEffect, legacyStructure, &timing.validationStructureMaxAbs, "structure"))) {
            return false;
        }
        timing.validationRan = true;
        timing.validationMs += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                                   std::chrono::steady_clock::now() - validationStart)
                                   .count();
    }

    if (params.energyMode == LensDiffEnergyMode::Preserve) {
        DeviceBuffer<float> inputEnergy;
        DeviceBuffer<float> effectEnergy;
        DeviceBuffer<float> preserveScale;
        if (!inputEnergy.allocate(1) || !effectEnergy.allocate(1) || !preserveScale.allocate(1)) {
            if (error) *error = "cuda-alloc-preserve-energy";
            return false;
        }
        if (!checkCuda(cudaMemsetAsync(inputEnergy.ptr, 0, sizeof(float), stream), "cudaMemsetAsync-input-energy", error) ||
            !checkCuda(cudaMemsetAsync(effectEnergy.ptr, 0, sizeof(float), stream), "cudaMemsetAsync-effect-energy", error)) {
            return false;
        }
        lumaReduceKernel<<<flatGrid, flatBlock, 0, stream>>>(redistributedR.ptr, redistributedG.ptr, redistributedB.ptr, pixelCount, inputEnergy.ptr);
        lumaReduceKernel<<<flatGrid, flatBlock, 0, stream>>>(effect.r.ptr, effect.g.ptr, effect.b.ptr, pixelCount, effectEnergy.ptr);
        if (!checkCuda(cudaGetLastError(), "lumaReduceKernel-preserve", error)) {
            return false;
        }
        computePreserveScaleKernelCuda<<<1, 1, 0, stream>>>(inputEnergy.ptr, effectEnergy.ptr, 1e-6f, preserveScale.ptr);
        if (!checkCuda(cudaGetLastError(), "computePreserveScaleKernelCuda", error)) {
            return false;
        }
        scaleRgbByScalarKernel<<<flatGrid, flatBlock, 0, stream>>>(effect.r.ptr, effect.g.ptr, effect.b.ptr, pixelCount, preserveScale.ptr);
        if (!checkCuda(cudaGetLastError(), "scaleRgbByScalarKernel-preserve", error)) {
            return false;
        }
    }

    PlaneSet scatterPreview;
    const double scatterRadiusPx = ResolveLensDiffScatterRadiusPx(params);
    const bool scatterActive = params.scatterAmount > 1e-6 && scatterRadiusPx > 0.25;
    if (scatterActive || params.debugView == LensDiffDebugView::Scatter) {
        if (!timeCall(timing.scatterMs, [&] {
                if (!allocatePlaneSet(scatterPreview)) {
                    if (error) *error = "cuda-alloc-scatter-preview";
                    return false;
                }
                if (scatterActive) {
                    const LensDiffKernel scatterKernel = buildGaussianKernelHost(static_cast<float>(scatterRadiusPx));
                    if (!convolvePlaneSet(effect.r, effect.g, effect.b, scatterKernel, scatterPreview, "cuda-convolve-scatter")) {
                        return false;
                    }
                    combineRgbKernel<<<flatGrid, flatBlock, 0, stream>>>(
                        effect.r.ptr, effect.g.ptr, effect.b.ptr,
                        scatterPreview.r.ptr, scatterPreview.g.ptr, scatterPreview.b.ptr,
                        pixelCount,
                        1.0f,
                        static_cast<float>(std::max(0.0, params.scatterAmount)),
                        effect.r.ptr, effect.g.ptr, effect.b.ptr);
                    if (!checkCuda(cudaGetLastError(), "combineRgbKernel-scatter", error)) {
                        return false;
                    }
                } else if (!clearPlaneSet(scatterPreview, "cudaMemsetAsync-scatter-preview")) {
                    return false;
                }
                return true;
            })) {
            return false;
        }
    }

    PlaneSet creativeFringePreview;
    const double creativeFringePx = ResolveLensDiffCreativeFringePx(params);
    const bool creativeFringeActive = creativeFringePx > 1e-6;
    if (creativeFringeActive || params.debugView == LensDiffDebugView::CreativeFringe) {
        if (!timeCall(timing.creativeFringeMs, [&] {
                PlaneSet fringedEffect;
                if (!allocatePlaneSet(fringedEffect) ||
                    !allocatePlaneSet(creativeFringePreview)) {
                    if (error) *error = "cuda-alloc-creative-fringe";
                    return false;
                }
                if (creativeFringeActive) {
                    applyCreativeFringeKernel<<<grid2d, block2d, 0, stream>>>(
                        effect.r.ptr, effect.g.ptr, effect.b.ptr,
                        width, height,
                        static_cast<float>(std::max(0.0, creativeFringePx)),
                        fringedEffect.r.ptr, fringedEffect.g.ptr, fringedEffect.b.ptr,
                        creativeFringePreview.r.ptr, creativeFringePreview.g.ptr, creativeFringePreview.b.ptr);
                    if (!checkCuda(cudaGetLastError(), "applyCreativeFringeKernel", error) ||
                        !copyPlaneSet(fringedEffect, effect, "cudaMemcpyAsync-fringed-effect")) {
                        return false;
                    }
                } else if (!clearPlaneSet(creativeFringePreview, "cudaMemsetAsync-creative-preview")) {
                    return false;
                }
                return true;
            })) {
            return false;
        }
    }

    const float effectGain = params.energyMode == LensDiffEnergyMode::Preserve
        ? static_cast<float>(std::clamp(params.effectGain, 0.0, 1.0))
        : static_cast<float>(std::max(0.0, params.effectGain));
    const float coreCompensation = params.energyMode == LensDiffEnergyMode::Preserve
        ? effectGain
        : static_cast<float>(std::max(0.0, params.coreCompensation));
    const float protectedCoreFraction = std::max(
        kMinimumSelectedCoreFloor,
        static_cast<float>(std::clamp(params.corePreserve, 0.0, 1.0)));
    const float maxRedistributedSubtractScale = redistributionScale > 1e-6f
        ? (1.0f - protectedCoreFraction) / redistributionScale
        : 0.0f;

    if (!timeCall(timing.compositeMs, [&] {
            if (params.debugView == LensDiffDebugView::Core) {
                if (resolutionAwareActive) {
                    PlaneSet display;
                    if (!timeCall(timing.nativeResampleMs, [&] {
                            return resamplePlaneSetToNative(coreEffect, display, "cuda-resample-core-display");
                        })) {
                        return false;
                    }
                    packRgbDebugKernel<<<nativeFlatGrid, flatBlock, 0, stream>>>(display.r.ptr, display.g.ptr, display.b.ptr, nullptr, nativePixelCount, 1.0f, packedOutput.ptr);
                } else {
                    packRgbDebugKernel<<<flatGrid, flatBlock, 0, stream>>>(coreEffect.r.ptr, coreEffect.g.ptr, coreEffect.b.ptr, nullptr, pixelCount, 1.0f, packedOutput.ptr);
                }
                return checkCuda(cudaGetLastError(), "packRgbDebugKernel-core", error);
            } else if (params.debugView == LensDiffDebugView::Structure) {
                if (resolutionAwareActive) {
                    PlaneSet display;
                    if (!timeCall(timing.nativeResampleMs, [&] {
                            return resamplePlaneSetToNative(structureEffect, display, "cuda-resample-structure-display");
                        })) {
                        return false;
                    }
                    packRgbDebugKernel<<<nativeFlatGrid, flatBlock, 0, stream>>>(display.r.ptr, display.g.ptr, display.b.ptr, nullptr, nativePixelCount, 1.0f, packedOutput.ptr);
                } else {
                    packRgbDebugKernel<<<flatGrid, flatBlock, 0, stream>>>(structureEffect.r.ptr, structureEffect.g.ptr, structureEffect.b.ptr, nullptr, pixelCount, 1.0f, packedOutput.ptr);
                }
                return checkCuda(cudaGetLastError(), "packRgbDebugKernel-structure", error);
            } else if (params.debugView == LensDiffDebugView::Effect) {
                if (resolutionAwareActive) {
                    PlaneSet display;
                    if (!timeCall(timing.nativeResampleMs, [&] {
                            return resamplePlaneSetToNative(effect, display, "cuda-resample-effect-display");
                        })) {
                        return false;
                    }
                    packRgbDebugKernel<<<nativeFlatGrid, flatBlock, 0, stream>>>(display.r.ptr, display.g.ptr, display.b.ptr, nullptr, nativePixelCount, 1.0f, packedOutput.ptr);
                } else {
                    packRgbDebugKernel<<<flatGrid, flatBlock, 0, stream>>>(effect.r.ptr, effect.g.ptr, effect.b.ptr, nullptr, pixelCount, 1.0f, packedOutput.ptr);
                }
                return checkCuda(cudaGetLastError(), "packRgbDebugKernel-effect", error);
            } else if (params.debugView == LensDiffDebugView::Scatter) {
                if (resolutionAwareActive) {
                    PlaneSet display;
                    if (!timeCall(timing.nativeResampleMs, [&] {
                            return resamplePlaneSetToNative(scatterPreview, display, "cuda-resample-scatter-display");
                        })) {
                        return false;
                    }
                    packRgbDebugKernel<<<nativeFlatGrid, flatBlock, 0, stream>>>(display.r.ptr, display.g.ptr, display.b.ptr, nullptr, nativePixelCount, 1.0f, packedOutput.ptr);
                } else {
                    packRgbDebugKernel<<<flatGrid, flatBlock, 0, stream>>>(scatterPreview.r.ptr, scatterPreview.g.ptr, scatterPreview.b.ptr, nullptr, pixelCount, 1.0f, packedOutput.ptr);
                }
                return checkCuda(cudaGetLastError(), "packRgbDebugKernel-scatter", error);
            } else if (params.debugView == LensDiffDebugView::CreativeFringe) {
                if (resolutionAwareActive) {
                    PlaneSet display;
                    if (!timeCall(timing.nativeResampleMs, [&] {
                            return resamplePlaneSetToNative(creativeFringePreview, display, "cuda-resample-creative-display");
                        })) {
                        return false;
                    }
                    packRgbDebugKernel<<<nativeFlatGrid, flatBlock, 0, stream>>>(display.r.ptr, display.g.ptr, display.b.ptr, nullptr, nativePixelCount, 1.0f, packedOutput.ptr);
                } else {
                    packRgbDebugKernel<<<flatGrid, flatBlock, 0, stream>>>(creativeFringePreview.r.ptr, creativeFringePreview.g.ptr, creativeFringePreview.b.ptr, nullptr, pixelCount, 1.0f, packedOutput.ptr);
                }
                return checkCuda(cudaGetLastError(), "packRgbDebugKernel-creative-fringe", error);
            }

            DeviceBuffer<float> redistributedRDisplay;
            DeviceBuffer<float> redistributedGDisplay;
            DeviceBuffer<float> redistributedBDisplay;
            DeviceBuffer<float> effectRDisplay;
            DeviceBuffer<float> effectGDisplay;
            DeviceBuffer<float> effectBDisplay;
            const DeviceBuffer<float>* compositeRedistR = &redistributedR;
            const DeviceBuffer<float>* compositeRedistG = &redistributedG;
            const DeviceBuffer<float>* compositeRedistB = &redistributedB;
            const DeviceBuffer<float>* compositeEffectR = &effect.r;
            const DeviceBuffer<float>* compositeEffectG = &effect.g;
            const DeviceBuffer<float>* compositeEffectB = &effect.b;
            std::size_t compositeCount = pixelCount;
            int compositeFlatGrid = flatGrid;
            if (resolutionAwareActive) {
                if (!timeCall(timing.nativeResampleMs, [&] {
                        return resamplePlaneToNative(redistributedR, redistributedRDisplay, "cuda-resample-redistributed-r") &&
                               resamplePlaneToNative(redistributedG, redistributedGDisplay, "cuda-resample-redistributed-g") &&
                               resamplePlaneToNative(redistributedB, redistributedBDisplay, "cuda-resample-redistributed-b") &&
                               resamplePlaneToNative(effect.r, effectRDisplay, "cuda-resample-effect-r") &&
                               resamplePlaneToNative(effect.g, effectGDisplay, "cuda-resample-effect-g") &&
                               resamplePlaneToNative(effect.b, effectBDisplay, "cuda-resample-effect-b");
                    })) {
                    return false;
                }
                compositeRedistR = &redistributedRDisplay;
                compositeRedistG = &redistributedGDisplay;
                compositeRedistB = &redistributedBDisplay;
                compositeEffectR = &effectRDisplay;
                compositeEffectG = &effectGDisplay;
                compositeEffectB = &effectBDisplay;
                compositeCount = nativePixelCount;
                compositeFlatGrid = nativeFlatGrid;
            }
            compositeFinalKernel<<<compositeFlatGrid, flatBlock, 0, stream>>>(nativeSrcR.ptr,
                                                                               nativeSrcG.ptr,
                                                                               nativeSrcB.ptr,
                                                                               nativeSrcA.ptr,
                                                                               compositeRedistR->ptr,
                                                                               compositeRedistG->ptr,
                                                                               compositeRedistB->ptr,
                                                                               compositeEffectR->ptr,
                                                                               compositeEffectG->ptr,
                                                                               compositeEffectB->ptr,
                                                                               compositeCount,
                                                                               coreCompensation,
                                                                               effectGain,
                                                                               maxRedistributedSubtractScale,
                                                                               inputTransfer,
                                                                               packedOutput.ptr);
            return checkCuda(cudaGetLastError(), "compositeFinalKernel", error);
        })) {
        return false;
    }

    bool copyOk = false;
    if (!timeCall(timing.outputCopyMs, [&] {
            copyOk = copyPackedToDestination(request, packedOutput.ptr, nativeWidth, nativeHeight, stream, cudaMemcpyDeviceToDevice, error);
            return copyOk;
        })) {
        return false;
    }
    logTimingBreakdown();
    return copyOk;
}

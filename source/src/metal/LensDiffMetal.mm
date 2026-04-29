#include "LensDiffMetal.h"
#include "LensDiffMetalVkFFT.h"

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dlfcn.h>

#include "../core/LensDiffApertureImage.h"
#include "../core/LensDiffCpuReference.h"
#include "../core/LensDiffDiagnostics.h"
#include "../core/LensDiffPhase.h"
#include "../core/LensDiffSpectrum.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace {

using Complex = std::complex<float>;

constexpr float kPi = 3.14159265358979323846f;
constexpr float kMinimumSelectedCoreFloor = 0.2f;

std::uint64_t hashKernelValues(const LensDiffKernel& kernel) {
    const std::uint64_t offset = 1469598103934665603ull;
    const std::uint64_t prime = 1099511628211ull;
    std::uint64_t hash = offset;
    for (float value : kernel.values) {
        std::uint32_t bits = 0u;
        static_assert(sizeof(bits) == sizeof(value), "float hash size mismatch");
        std::memcpy(&bits, &value, sizeof(bits));
        hash ^= static_cast<std::uint64_t>(bits);
        hash *= prime;
    }
    hash ^= static_cast<std::uint64_t>(kernel.size);
    hash *= prime;
    return hash;
}

template <typename T>
T clampValue(T value, T lo, T hi) {
    return std::max(lo, std::min(value, hi));
}

float saturate(float value) {
    return clampValue(value, 0.0f, 1.0f);
}

float safeLuma(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

float softShoulder(float value, float shoulder) {
    if (shoulder <= 0.0f) {
        return std::max(0.0f, value);
    }
    const float x = std::max(0.0f, value);
    return shoulder * (1.0f - std::exp(-x / shoulder));
}

int nextPowerOfTwo(int value) {
    int out = 1;
    while (out < value) {
        out <<= 1;
    }
    return out;
}

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

void applySupportBoundaryTaperHost(LensDiffKernel& kernel, int supportRadius) {
    if (kernel.size <= 0 || kernel.values.empty() || supportRadius <= 0) {
        return;
    }

    const float center = static_cast<float>(kernel.size - 1) * 0.5f;
    const float extent = std::max(1.0f, static_cast<float>(supportRadius));
    const float fadeWidth = std::max(6.0f, std::min(24.0f, extent * 0.04f));
    const float fadeStart = std::max(0.0f, extent - fadeWidth);

    for (int y = 0; y < kernel.size; ++y) {
        for (int x = 0; x < kernel.size; ++x) {
            const float dx = static_cast<float>(x) - center;
            const float dy = static_cast<float>(y) - center;
            const float radius = std::sqrt(dx * dx + dy * dy);
            float weight = 1.0f;
            if (radius >= extent) {
                weight = 0.0f;
            } else {
                const float t = clampValue((radius - fadeStart) / std::max(fadeWidth, 1e-6f), 0.0f, 1.0f);
                const float s = t * t * (3.0f - 2.0f * t);
                weight = std::max(0.0f, std::cos(s * (kPi * 0.5f)));
            }
            kernel.values[static_cast<std::size_t>(y) * kernel.size + x] *= weight;
        }
    }
}

void normalizeKernelHost(LensDiffKernel& kernel) {
    float sum = 0.0f;
    for (float value : kernel.values) {
        sum += value;
    }
    if (sum > 0.0f) {
        const float invSum = 1.0f / sum;
        for (float& value : kernel.values) {
            value *= invSum;
        }
    }
}

int paddedAdaptiveSupportRadiusHost(int estimatedRadius, int maxRadius) {
    const int padding = std::max(6, std::min(24, static_cast<int>(std::ceil(std::max(1, estimatedRadius) * 0.05f))));
    return std::max(4, std::min(maxRadius, estimatedRadius + padding));
}

bool isPowerOfTwo(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

void radix2Fft1d(std::vector<Complex>& data, bool inverse) {
    const int count = static_cast<int>(data.size());
    if (!isPowerOfTwo(count)) {
        return;
    }

    for (int i = 1, j = 0; i < count; ++i) {
        int bit = count >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }

    for (int len = 2; len <= count; len <<= 1) {
        const float angle = 2.0f * kPi / static_cast<float>(len) * (inverse ? 1.0f : -1.0f);
        const Complex wlen(std::cos(angle), std::sin(angle));
        for (int i = 0; i < count; i += len) {
            Complex w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; ++j) {
                const Complex u = data[i + j];
                const Complex v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        const float invCount = 1.0f / static_cast<float>(count);
        for (Complex& value : data) {
            value *= invCount;
        }
    }
}

void fft1d(std::vector<Complex>& data, bool inverse) {
    const int count = static_cast<int>(data.size());
    if (count <= 1) {
        return;
    }
    if (isPowerOfTwo(count)) {
        radix2Fft1d(data, inverse);
        return;
    }

    const int workspaceSize = nextPowerOfTwo(count * 2 - 1);
    std::vector<Complex> a(static_cast<std::size_t>(workspaceSize), Complex(0.0f, 0.0f));
    std::vector<Complex> b(static_cast<std::size_t>(workspaceSize), Complex(0.0f, 0.0f));
    const double sign = inverse ? 1.0 : -1.0;

    for (int i = 0; i < count; ++i) {
        const double index = static_cast<double>(i);
        const double phase = sign * static_cast<double>(kPi) * index * index / static_cast<double>(count);
        const Complex chirp(static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase)));
        const Complex chirpConjugate(static_cast<float>(std::cos(-phase)), static_cast<float>(std::sin(-phase)));
        a[static_cast<std::size_t>(i)] = data[static_cast<std::size_t>(i)] * chirp;
        b[static_cast<std::size_t>(i)] = chirpConjugate;
        if (i != 0) {
            b[static_cast<std::size_t>(workspaceSize - i)] = chirpConjugate;
        }
    }

    radix2Fft1d(a, false);
    radix2Fft1d(b, false);
    for (int i = 0; i < workspaceSize; ++i) {
        a[static_cast<std::size_t>(i)] *= b[static_cast<std::size_t>(i)];
    }
    radix2Fft1d(a, true);

    const float inverseScale = inverse ? (1.0f / static_cast<float>(count)) : 1.0f;
    for (int i = 0; i < count; ++i) {
        const double index = static_cast<double>(i);
        const double phase = sign * static_cast<double>(kPi) * index * index / static_cast<double>(count);
        const Complex chirp(static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase)));
        data[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i)] * chirp * inverseScale;
    }
}

void fft2d(std::vector<Complex>& data, int width, int height, bool inverse) {
    for (int y = 0; y < height; ++y) {
        std::vector<Complex> row(static_cast<std::size_t>(width));
        for (int x = 0; x < width; ++x) {
            row[x] = data[static_cast<std::size_t>(y) * width + x];
        }
        fft1d(row, inverse);
        for (int x = 0; x < width; ++x) {
            data[static_cast<std::size_t>(y) * width + x] = row[x];
        }
    }

    for (int x = 0; x < width; ++x) {
        std::vector<Complex> column(static_cast<std::size_t>(height));
        for (int y = 0; y < height; ++y) {
            column[y] = data[static_cast<std::size_t>(y) * width + x];
        }
        fft1d(column, inverse);
        for (int y = 0; y < height; ++y) {
            data[static_cast<std::size_t>(y) * width + x] = column[y];
        }
    }
}

std::vector<float> fftShiftSquare(const std::vector<float>& src, int size) {
    std::vector<float> out(src.size(), 0.0f);
    const int half = size / 2;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const int sx = (x + half) % size;
            const int sy = (y + half) % size;
            out[static_cast<std::size_t>(y) * size + x] = src[static_cast<std::size_t>(sy) * size + sx];
        }
    }
    return out;
}

float sampleSquareBilinear(const std::vector<float>& image, int size, float x, float y) {
    if (x < 0.0f || y < 0.0f || x > static_cast<float>(size - 1) || y > static_cast<float>(size - 1)) {
        return 0.0f;
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, size - 1);
    const int y1 = std::min(y0 + 1, size - 1);
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

std::vector<float> computeOtfMagnitudeImage(const LensDiffKernel& kernel) {
    const int size = std::max(1, kernel.size);
    std::vector<Complex> spectrum(static_cast<std::size_t>(size) * size, Complex(0.0f, 0.0f));
    const int center = kernel.size / 2;
    for (int y = 0; y < kernel.size; ++y) {
        for (int x = 0; x < kernel.size; ++x) {
            const int px = (x - center + size) % size;
            const int py = (y - center + size) % size;
            spectrum[static_cast<std::size_t>(py) * size + px] =
                Complex(kernel.values[static_cast<std::size_t>(y) * kernel.size + x], 0.0f);
        }
    }
    fft2d(spectrum, size, size, false);

    std::vector<float> out(static_cast<std::size_t>(size) * size, 0.0f);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            out[static_cast<std::size_t>(y) * size + x] =
                std::abs(spectrum[static_cast<std::size_t>(y) * size + x]);
        }
    }
    return fftShiftSquare(out, size);
}

std::vector<float> makeCenteredSquareImage(const LensDiffParams& params,
                                           const LensDiffPsfBankCache& cache,
                                           int* outSquareSize) {
    switch (params.debugView) {
        case LensDiffDebugView::Pupil:
            if (outSquareSize) {
                *outSquareSize = cache.pupilDisplaySize;
            }
            return cache.pupilDisplay;
        case LensDiffDebugView::Psf:
            if (outSquareSize) {
                *outSquareSize = cache.bins.empty() ? 0 : cache.bins.front().full.size;
            }
            return cache.bins.empty() ? std::vector<float>() : cache.bins.front().full.values;
        case LensDiffDebugView::Otf:
            if (cache.bins.empty()) {
                if (outSquareSize) {
                    *outSquareSize = 0;
                }
                return {};
            }
            if (outSquareSize) {
                *outSquareSize = std::max(1, cache.bins.front().full.size);
            }
            return computeOtfMagnitudeImage(cache.bins.front().full);
        default:
            if (outSquareSize) {
                *outSquareSize = 0;
            }
            return {};
    }
}

std::vector<float> buildStaticDebugRgba(const LensDiffParams& params,
                                        const LensDiffPsfBankCache& cache,
                                        int outWidth,
                                        int outHeight) {
    LensDiffPsfBankCache* mutableCache = const_cast<LensDiffPsfBankCache*>(&cache);
    return GetLensDiffStaticDebugRgbaCached(params, mutableCache, outWidth, outHeight);
}

std::string nsErrorString(NSError* error) {
    if (!error) {
        return {};
    }
    NSString* description = error.localizedDescription ?: @"unknown metal error";
    return std::string(description.UTF8String ? description.UTF8String : "unknown metal error");
}

constexpr const char* kLensDiffMetalLibraryFilename = "LensDiff.metallib";
extern std::mutex gWorkQueueMutex;
extern id<MTLDevice> gWorkQueueDevice;
extern id<MTLCommandQueue> gWorkQueue;

std::string lensDiffBundledMetalLibraryPath() {
    Dl_info info {};
    if (dladdr(reinterpret_cast<const void*>(&RunLensDiffMetal), &info) == 0 || info.dli_fname == nullptr) {
        return {};
    }
    const std::filesystem::path binaryPath(info.dli_fname);
    const std::filesystem::path resourcePath = binaryPath.parent_path().parent_path() / "Resources" / kLensDiffMetalLibraryFilename;
    if (!std::filesystem::exists(resourcePath)) {
        return {};
    }
    return resourcePath.string();
}

id<MTLCommandQueue> ensureWorkQueue(id<MTLDevice> device, std::string* error) {
    if (device == nil) {
        if (error) *error = "missing-metal-device";
        return nil;
    }

    std::lock_guard<std::mutex> lock(gWorkQueueMutex);
    if (gWorkQueueDevice == device && gWorkQueue != nil) {
        return gWorkQueue;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        if (error) *error = "failed-metal-work-queue";
        return nil;
    }

    gWorkQueueDevice = device;
    gWorkQueue = queue;
    return gWorkQueue;
}

bool validateRowBytes(const LensDiffImageView& view) {
    const std::ptrdiff_t minRowBytes = static_cast<std::ptrdiff_t>(view.bounds.width()) * 4 * static_cast<std::ptrdiff_t>(sizeof(float));
    return view.data != nullptr &&
           view.bounds.width() > 0 &&
           view.bounds.height() > 0 &&
           view.rowBytes >= minRowBytes &&
           (view.rowBytes % static_cast<std::ptrdiff_t>(sizeof(float))) == 0;
}

uint32_t rowFloats(const LensDiffImageView& view) {
    return static_cast<uint32_t>(view.rowBytes / static_cast<std::ptrdiff_t>(sizeof(float)));
}

LensDiffImageRect intersectRect(const LensDiffImageRect& a, const LensDiffImageRect& b) {
    LensDiffImageRect out {};
    out.x1 = std::max(a.x1, b.x1);
    out.y1 = std::max(a.y1, b.y1);
    out.x2 = std::min(a.x2, b.x2);
    out.y2 = std::min(a.y2, b.y2);
    if (out.x2 < out.x1) out.x2 = out.x1;
    if (out.y2 < out.y1) out.y2 = out.y1;
    return out;
}

NSUInteger ceilDiv(NSUInteger value, NSUInteger divisor) {
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

struct DecodeParamsGpu {
    int srcWidth = 0;
    int srcHeight = 0;
    uint32_t srcRowFloats = 0;
    int inputTransfer = 0;
};

struct PrepareParamsGpu {
    int width = 0;
    int height = 0;
    int extractionMode = 0;
    float threshold = 0.0f;
    float softnessStops = 0.0f;
    float pointEmphasis = 0.0f;
    float corePreserve = 0.0f;
};

struct ResampleParamsGpu {
    int srcWidth = 0;
    int srcHeight = 0;
    int dstWidth = 0;
    int dstHeight = 0;
};

struct FftImageParamsGpu {
    int width = 0;
    int height = 0;
    int paddedSize = 0;
    int batchStride = 0;
    int batchCount = 0;
    int channelIndex = 0;
    int kernelSize = 0;
};

struct ConvolutionParamsGpu {
    int width = 0;
    int height = 0;
    int kernelSize = 0;
    int kernelRadius = 0;
};

struct ShoulderParamsGpu {
    int width = 0;
    int height = 0;
    float shoulder = 0.0f;
};

struct CombineParamsGpu {
    int width = 0;
    int height = 0;
    float coreGain = 1.0f;
    float structureGain = 1.0f;
};

struct SpectralMapParamsGpu {
    int width = 0;
    int height = 0;
    int binCount = 1;
    int chromaticAffectsLuma = 0;
    float spectrumForce = 0.0f;
    float spectrumSaturation = 1.0f;
    float naturalMatrix[kLensDiffMaxSpectralBins * 3] = {0.0f};
    float styleMatrix[kLensDiffMaxSpectralBins * 3] = {0.0f};
};

struct CompositeParamsGpu {
    int width = 0;
    int height = 0;
    float effectGain = 1.0f;
    float coreCompensation = 1.0f;
    float maxRedistributedSubtractScale = 0.0f;
};

struct ScaleParamsGpu {
    int width = 0;
    int height = 0;
    float scale = 1.0f;
};

struct FieldBlendParamsGpu {
    int width = 0;
    int height = 0;
    int zoneX = 0;
    int zoneY = 0;
    float gain = 1.0f;
};

struct FringeParamsGpu {
    int width = 0;
    int height = 0;
    float fringeAmount = 0.0f;
};

struct OutputParamsGpu {
    int srcWidth = 0;
    int srcHeight = 0;
    int srcOriginX = 0;
    int srcOriginY = 0;
    int dstOriginX = 0;
    int dstOriginY = 0;
    int dstX2 = 0;
    int dstY2 = 0;
    int renderX1 = 0;
    int renderY1 = 0;
    int renderX2 = 0;
    int renderY2 = 0;
    uint32_t dstRowFloats = 0;
    int inputTransfer = 0;
    int encodeOutput = 0;
    int writeAlphaFromLinear = 0;
};

struct ReductionParamsGpu {
    int width = 0;
    int height = 0;
};

struct ScalarReduceParamsGpu {
    int count = 0;
};

struct DynamicScaleParamsGpu {
    int width = 0;
    int height = 0;
};

struct ScalarScaleParamsGpu {
    int count = 0;
    float epsilon = 1e-6f;
};

struct PreserveScaleParamsGpu {
    float epsilon = 1e-6f;
};

struct KernelShapeParamsGpu {
    int kernelSize = 0;
    int radiusMax = 0;
};

struct PupilRasterParamsGpu {
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

struct PhaseRasterParamsGpu {
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

struct EmbedComplexParamsGpu {
    int pupilSize = 0;
    int rawPsfSize = 0;
    int offset = 0;
};

struct FftParamsGpu {
    int size = 0;
    int log2Size = 0;
};

struct FftStageParamsGpu {
    int size = 0;
    int halfSize = 0;
    int stageSize = 0;
};

struct BatchFftParamsGpu {
    int length = 0;
    int log2Length = 0;
    int batchStride = 0;
    int batchCount = 0;
};

struct BatchFftStageParamsGpu {
    int length = 0;
    int halfSize = 0;
    int stageSize = 0;
    int batchStride = 0;
    int batchCount = 0;
    int inverse = 0;
};

struct BluesteinParamsGpu {
    int signalLength = 0;
    int convolutionLength = 0;
    int batchCount = 0;
};

struct ReplicateComplexParamsGpu {
    int length = 0;
    int srcBatchCount = 0;
    int dstBatchCount = 0;
    int srcStride = 0;
    int dstStride = 0;
};

struct StackImageParamsGpu {
    int width = 0;
    int height = 0;
    int stackDepth = 0;
    int planeStride = 0;
};

struct ZonePlaneStackParamsGpu {
    int width = 0;
    int height = 0;
    int zoneCount = 0;
    int binCount = 0;
    int planeStride = 0;
};

struct ZonePlaneAccumulateParamsGpu {
    int width = 0;
    int height = 0;
    int zoneCount = 0;
    int binCount = 0;
    int planeStride = 0;
    int binIndex = 0;
};

struct PipelineBundle {
    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> buildPupil = nil;
    id<MTLComputePipelineState> buildPhase = nil;
    id<MTLComputePipelineState> embedComplexPupil = nil;
    id<MTLComputePipelineState> bitReverseRows = nil;
    id<MTLComputePipelineState> fftRowsStage = nil;
    id<MTLComputePipelineState> bitReverseColumns = nil;
    id<MTLComputePipelineState> fftColumnsStage = nil;
    id<MTLComputePipelineState> extractShiftedIntensity = nil;
    id<MTLComputePipelineState> transposeComplex = nil;
    id<MTLComputePipelineState> transposeComplexStack = nil;
    id<MTLComputePipelineState> bitReverseBatched = nil;
    id<MTLComputePipelineState> fftBatchedStage = nil;
    id<MTLComputePipelineState> scaleComplexBatched = nil;
    id<MTLComputePipelineState> copyComplexBatched = nil;
    id<MTLComputePipelineState> buildBluesteinChirp = nil;
    id<MTLComputePipelineState> buildBluesteinInput = nil;
    id<MTLComputePipelineState> multiplyBluesteinSpectra = nil;
    id<MTLComputePipelineState> extractBluesteinOutput = nil;
    id<MTLComputePipelineState> decodeSource = nil;
    id<MTLComputePipelineState> prepare = nil;
    id<MTLComputePipelineState> prepareFromLinear = nil;
    id<MTLComputePipelineState> convolveRgb = nil;
    id<MTLComputePipelineState> convolveScalar = nil;
    id<MTLComputePipelineState> padPlaneToComplex = nil;
    id<MTLComputePipelineState> padRgbChannelToComplex = nil;
    id<MTLComputePipelineState> padRgbToComplexStack = nil;
    id<MTLComputePipelineState> scatterKernelToComplex = nil;
    id<MTLComputePipelineState> multiplyComplex = nil;
    id<MTLComputePipelineState> multiplyComplexBroadcast = nil;
    id<MTLComputePipelineState> multiplyComplexPairsStack = nil;
    id<MTLComputePipelineState> replicateComplexStack = nil;
    id<MTLComputePipelineState> extractRealPlane = nil;
    id<MTLComputePipelineState> packPlanesToRgba = nil;
    id<MTLComputePipelineState> packPlaneTripletsToRgbaStack = nil;
    id<MTLComputePipelineState> applyShoulder = nil;
    id<MTLComputePipelineState> applyShoulderStack = nil;
    id<MTLComputePipelineState> combine = nil;
    id<MTLComputePipelineState> combineStack = nil;
    id<MTLComputePipelineState> mapSpectral = nil;
    id<MTLComputePipelineState> mapSpectralStack = nil;
    id<MTLComputePipelineState> composite = nil;
    id<MTLComputePipelineState> scaleRgb = nil;
    id<MTLComputePipelineState> scaleScalar = nil;
    id<MTLComputePipelineState> accumulateWeighted = nil;
    id<MTLComputePipelineState> accumulateWeightedRgbStack = nil;
    id<MTLComputePipelineState> accumulateWeightedPlanesStack = nil;
    id<MTLComputePipelineState> creativeFringe = nil;
    id<MTLComputePipelineState> packGray = nil;
    id<MTLComputePipelineState> packRgb = nil;
    id<MTLComputePipelineState> resampleRgba = nil;
    id<MTLComputePipelineState> resampleGray = nil;
    id<MTLComputePipelineState> reduceLuma = nil;
    id<MTLComputePipelineState> reduceFloat = nil;
    id<MTLComputePipelineState> computeScalarScale = nil;
    id<MTLComputePipelineState> computePreserveScale = nil;
    id<MTLComputePipelineState> scaleRgbDynamic = nil;
    id<MTLComputePipelineState> resampleRawPsf = nil;
    id<MTLComputePipelineState> ringSumCount = nil;
    id<MTLComputePipelineState> expandMean = nil;
    id<MTLComputePipelineState> reshapeKernel = nil;
    id<MTLComputePipelineState> positiveResidual = nil;
    id<MTLComputePipelineState> ringEnergyPeak = nil;
    id<MTLComputePipelineState> cropKernel = nil;
};

bool encodeSquareForwardFft(id<MTLCommandBuffer> commandBuffer,
                            id<MTLComputeCommandEncoder> encoder,
                            PipelineBundle* pipelines,
                            id<MTLBuffer> spectrum,
                            id<MTLBuffer> scratch,
                            id<MTLBuffer> transpose,
                            int size,
                            std::string* error);
bool encodeSquareInverseFft(id<MTLCommandBuffer> commandBuffer,
                            id<MTLComputeCommandEncoder> encoder,
                            PipelineBundle* pipelines,
                            id<MTLBuffer> spectrum,
                            id<MTLBuffer> scratch,
                            id<MTLBuffer> transpose,
                            int size,
                            std::string* error);
bool encodeSquareForwardFftStack(id<MTLCommandBuffer> commandBuffer,
                                 id<MTLComputeCommandEncoder> encoder,
                                 PipelineBundle* pipelines,
                                 id<MTLBuffer> spectrum,
                                 id<MTLBuffer> scratch,
                                 id<MTLBuffer> transpose,
                                 int size,
                                 int imageCount,
                                 std::string* error);
bool encodeSquareInverseFftStack(id<MTLCommandBuffer> commandBuffer,
                                 id<MTLComputeCommandEncoder> encoder,
                                 PipelineBundle* pipelines,
                                 id<MTLBuffer> spectrum,
                                 id<MTLBuffer> scratch,
                                 id<MTLBuffer> transpose,
                                 int size,
                                 int imageCount,
                                 std::string* error);

std::mutex gPipelineMutex;
PipelineBundle gPipelines;
std::mutex gWorkQueueMutex;
id<MTLDevice> gWorkQueueDevice = nil;
id<MTLCommandQueue> gWorkQueue = nil;
std::mutex gHeapMutex;

struct MetalHeapRecord {
    id<MTLHeap> heap = nil;
    NSUInteger size = 0;
};

std::unordered_map<std::uintptr_t, std::vector<MetalHeapRecord>> gHeapPools;

const char* kLensDiffMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float kGray18 = 0.18f;
constant float kLensDiffPi = 3.14159265358979323846f;
#define LENSDIFF_MAX_SPECTRAL_BINS 9

struct DecodeParamsGpu {
    int srcWidth;
    int srcHeight;
    uint srcRowFloats;
    int inputTransfer;
};

struct PrepareParamsGpu {
    int width;
    int height;
    int extractionMode;
    float threshold;
    float softnessStops;
    float pointEmphasis;
    float corePreserve;
};

struct ResampleParamsGpu {
    int srcWidth;
    int srcHeight;
    int dstWidth;
    int dstHeight;
};

struct FftImageParamsGpu {
    int width;
    int height;
    int paddedSize;
    int batchStride;
    int batchCount;
    int channelIndex;
    int kernelSize;
};

struct ConvolutionParamsGpu {
    int width;
    int height;
    int kernelSize;
    int kernelRadius;
};

struct ShoulderParamsGpu {
    int width;
    int height;
    float shoulder;
};

struct CombineParamsGpu {
    int width;
    int height;
    float coreGain;
    float structureGain;
};

struct SpectralMapParamsGpu {
    int width;
    int height;
    int binCount;
    int chromaticAffectsLuma;
    float spectrumForce;
    float spectrumSaturation;
    float naturalMatrix[15];
    float styleMatrix[15];
};

struct CompositeParamsGpu {
    int width;
    int height;
    float effectGain;
    float coreCompensation;
    float maxRedistributedSubtractScale;
};

struct ScaleParamsGpu {
    int width;
    int height;
    float scale;
};

struct FieldBlendParamsGpu {
    int width;
    int height;
    int zoneX;
    int zoneY;
    float gain;
};

struct FringeParamsGpu {
    int width;
    int height;
    float fringeAmount;
};

struct OutputParamsGpu {
    int srcWidth;
    int srcHeight;
    int srcOriginX;
    int srcOriginY;
    int dstOriginX;
    int dstOriginY;
    int dstX2;
    int dstY2;
    int renderX1;
    int renderY1;
    int renderX2;
    int renderY2;
    uint dstRowFloats;
    int inputTransfer;
    int encodeOutput;
    int writeAlphaFromLinear;
};

struct ReductionParamsGpu {
    int width;
    int height;
};

struct ScalarReduceParamsGpu {
    int count;
};

struct DynamicScaleParamsGpu {
    int width;
    int height;
};

struct ScalarScaleParamsGpu {
    int count;
    float epsilon;
};

struct PreserveScaleParamsGpu {
    float epsilon;
};

struct KernelShapeParamsGpu {
    int kernelSize;
    int radiusMax;
};

struct PupilRasterParamsGpu {
    int size;
    int apertureMode;
    int apodizationMode;
    int bladeCount;
    int vaneCount;
    int customWidth;
    int customHeight;
    float roundness;
    float rotationRad;
    float outerRadius;
    float centralObstruction;
    float vaneThickness;
    float pupilDecenterX;
    float pupilDecenterY;
    float fitHalfWidth;
    float fitHalfHeight;
    float starInnerRadiusRatio;
};

struct PhaseRasterParamsGpu {
    int size;
    int hasPhase;
    float rotationRad;
    float outerRadius;
    float pupilDecenterX;
    float pupilDecenterY;
    float phaseDefocus;
    float phaseAstigmatism0;
    float phaseAstigmatism45;
    float phaseComaX;
    float phaseComaY;
    float phaseSpherical;
    float phaseTrefoilX;
    float phaseTrefoilY;
    float phaseSecondaryAstigmatism0;
    float phaseSecondaryAstigmatism45;
    float phaseQuadrafoil0;
    float phaseQuadrafoil45;
    float phaseSecondaryComaX;
    float phaseSecondaryComaY;
};

struct EmbedComplexParamsGpu {
    int pupilSize;
    int rawPsfSize;
    int offset;
};

struct FftParamsGpu {
    int size;
    int log2Size;
};

struct FftStageParamsGpu {
    int size;
    int halfSize;
    int stageSize;
};

struct BatchFftParamsGpu {
    int length;
    int log2Length;
    int batchStride;
    int batchCount;
};

struct BatchFftStageParamsGpu {
    int length;
    int halfSize;
    int stageSize;
    int batchStride;
    int batchCount;
    int inverse;
};

struct BluesteinParamsGpu {
    int signalLength;
    int convolutionLength;
    int batchCount;
};

struct ReplicateComplexParamsGpu {
    int length;
    int srcBatchCount;
    int dstBatchCount;
    int srcStride;
    int dstStride;
};

struct StackImageParamsGpu {
    int width;
    int height;
    int stackDepth;
    int planeStride;
};

struct ZonePlaneStackParamsGpu {
    int width;
    int height;
    int zoneCount;
    int binCount;
    int planeStride;
};

struct ZonePlaneAccumulateParamsGpu {
    int width;
    int height;
    int zoneCount;
    int binCount;
    int planeStride;
    int binIndex;
};

inline float saturateSafe(float value) {
    return clamp(value, 0.0f, 1.0f);
}

inline float safeLuma(float3 rgb) {
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

inline float softShoulderSafe(float value, float shoulder) {
    if (shoulder <= 0.0f) {
        return max(0.0f, value);
    }
    const float x = max(0.0f, value);
    return shoulder * (1.0f - exp(-x / shoulder));
}

inline float decodeChannel(float x, int tf) {
    if (tf == 1) {
        return x <= 0.02740668f ? x / 10.44426855f : exp2(x / 0.07329248f - 7.0f) - 0.0075f;
    }
    return x;
}

inline float encodeChannel(float x, int tf) {
    if (tf == 1) {
        constexpr float kA = 0.0075f;
        constexpr float kB = 7.0f;
        constexpr float kC = 0.07329248f;
        constexpr float kM = 10.44426855f;
        constexpr float kLinCut = 0.00262409f;
        return x <= kLinCut ? (x * kM) : ((log2(max(x, 0.0f) + kA) + kB) * kC);
    }
    return x;
}

inline float4 loadSourcePixel(device const float* src,
                              uint rowFloats,
                              int width,
                              int height,
                              int x,
                              int y) {
    if (x < 0 || y < 0 || x >= width || y >= height) {
        return float4(0.0f);
    }
    const uint index = uint(y) * rowFloats + uint(x) * 4u;
    return float4(src[index + 0u], src[index + 1u], src[index + 2u], src[index + 3u]);
}

inline float3 mulSpectralMatrix(const constant float* matrix, thread const float* bins, int binCount) {
    if (binCount <= 1) {
        return float3(bins[0]);
    }
    float3 out = float3(0.0f);
    const int count = min(binCount, LENSDIFF_MAX_SPECTRAL_BINS);
    for (int i = 0; i < count; ++i) {
        out.x += matrix[i] * bins[i];
        out.y += matrix[LENSDIFF_MAX_SPECTRAL_BINS + i] * bins[i];
        out.z += matrix[LENSDIFF_MAX_SPECTRAL_BINS * 2 + i] * bins[i];
    }
    return out;
}

inline float3 mapSpectral(thread const float* bins, constant SpectralMapParamsGpu& params) {
    const float3 natural = mulSpectralMatrix(params.naturalMatrix, bins, params.binCount);
    const float3 styled = mulSpectralMatrix(params.styleMatrix, bins, params.binCount);
    float3 rgb = mix(natural, styled, saturateSafe(params.spectrumForce));
    rgb = max(rgb, float3(0.0f));

    const float gray = safeLuma(rgb);
    rgb = float3(gray) + (rgb - float3(gray)) * max(0.0f, params.spectrumSaturation);

    if (params.chromaticAffectsLuma == 0) {
        const float targetLuma = safeLuma(natural);
        const float currentLuma = safeLuma(rgb);
        if (currentLuma > 1e-6f) {
            rgb *= targetLuma / currentLuma;
        }
    }
    return rgb;
}

inline float sampleSquareBilinear(device const float* image, int size, float x, float y) {
    if (x < 0.0f || y < 0.0f || x > float(size - 1) || y > float(size - 1)) {
        return 0.0f;
    }
    const int x0 = int(floor(x));
    const int y0 = int(floor(y));
    const int x1 = min(x0 + 1, size - 1);
    const int y1 = min(y0 + 1, size - 1);
    const float tx = x - float(x0);
    const float ty = y - float(y0);
    const float v00 = image[uint(y0 * size + x0)];
    const float v10 = image[uint(y0 * size + x1)];
    const float v01 = image[uint(y1 * size + x0)];
    const float v11 = image[uint(y1 * size + x1)];
    const float vx0 = mix(v00, v10, tx);
    const float vx1 = mix(v01, v11, tx);
    return mix(vx0, vx1, ty);
}

inline float4 sampleRgbaBilinear(device const float4* image, int width, int height, float x, float y) {
    const float clampedX = clamp(x, 0.0f, float(max(0, width - 1)));
    const float clampedY = clamp(y, 0.0f, float(max(0, height - 1)));
    const int x0 = int(floor(clampedX));
    const int y0 = int(floor(clampedY));
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float tx = clampedX - float(x0);
    const float ty = clampedY - float(y0);
    const float4 p00 = image[uint(y0 * width + x0)];
    const float4 p10 = image[uint(y0 * width + x1)];
    const float4 p01 = image[uint(y1 * width + x0)];
    const float4 p11 = image[uint(y1 * width + x1)];
    const float4 a = mix(p00, p10, tx);
    const float4 b = mix(p01, p11, tx);
    return mix(a, b, ty);
}

inline float samplePlaneBilinear(device const float* image, int width, int height, float x, float y) {
    if (x < 0.0f || y < 0.0f || x > float(width - 1) || y > float(height - 1)) {
        return 0.0f;
    }
    const int x0 = int(floor(x));
    const int y0 = int(floor(y));
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float tx = x - float(x0);
    const float ty = y - float(y0);
    const float v00 = image[uint(y0 * width + x0)];
    const float v10 = image[uint(y0 * width + x1)];
    const float v01 = image[uint(y1 * width + x0)];
    const float v11 = image[uint(y1 * width + x1)];
    const float vx0 = mix(v00, v10, tx);
    const float vx1 = mix(v01, v11, tx);
    return mix(vx0, vx1, ty);
}

inline float sampleSquareFiltered(device const float* image, int size, float x, float y, float footprint) {
    if (footprint <= 1.0f) {
        return sampleSquareBilinear(image, size, x, y);
    }
    const int taps = clamp(int(ceil(footprint)), 2, 6);
    const float step = footprint / float(taps);
    float sum = 0.0f;
    for (int ty = 0; ty < taps; ++ty) {
        const float oy = (float(ty) + 0.5f) * step - footprint * 0.5f;
        for (int tx = 0; tx < taps; ++tx) {
            const float ox = (float(tx) + 0.5f) * step - footprint * 0.5f;
            sum += sampleSquareBilinear(image, size, x + ox, y + oy);
        }
    }
    return sum / float(taps * taps);
}

inline float clampUnit(float value) {
    return clamp(value, 0.0f, 1.0f);
}

inline float polygonMetricGpu(float nx,
                              float ny,
                              int bladeCount,
                              float roundness,
                              float rotationRad,
                              float outerRadius) {
    const float radius = length(float2(nx, ny));
    const float circleMetric = radius / max(outerRadius, 1e-5f);
    if (bladeCount < 3) {
        return circleMetric;
    }

    const float angle = atan2(ny, nx) - rotationRad;
    const float sector = 2.0f * kLensDiffPi / float(bladeCount);
    float local = fmod(angle, sector);
    if (local < 0.0f) {
        local += sector;
    }
    local -= sector * 0.5f;
    const float polygonRadius = outerRadius * cos(kLensDiffPi / float(bladeCount)) / max(cos(local), 1e-4f);
    const float polygonMetricValue = radius / max(polygonRadius, 1e-5f);
    return circleMetric * roundness + polygonMetricValue * (1.0f - roundness);
}

inline bool blockedByVanesGpu(float nx,
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
        const float angle = rotationRad + float(i) * kLensDiffPi / float(lineCount);
        const float cs = cos(angle);
        const float sn = sin(angle);
        const float xr = nx * cs + ny * sn;
        const float yr = -nx * sn + ny * cs;
        if (abs(yr) <= scaledThickness && abs(xr) <= outerRadius) {
            return true;
        }
    }
    return false;
}

inline float starMetricGpu(float nx,
                           float ny,
                           int points,
                           float innerRadiusRatio,
                           float rotationRad,
                           float outerRadius) {
    const float radius = length(float2(nx, ny));
    if (radius <= 1e-6f) {
        return 0.0f;
    }

    const int pointCount = max(3, points);
    const float innerRadius = outerRadius * clamp(innerRadiusRatio, 0.1f, 0.95f);
    const float angle = atan2(ny, nx) - rotationRad + kLensDiffPi * 0.5f;
    const float sector = 2.0f * kLensDiffPi / float(pointCount);
    float local = fmod(angle, sector);
    if (local < 0.0f) {
        local += sector;
    }
    const float halfSector = sector * 0.5f;
    const float t = local <= halfSector
        ? local / max(halfSector, 1e-6f)
        : (sector - local) / max(halfSector, 1e-6f);
    const float boundaryRadius = innerRadius + (outerRadius - innerRadius) * clampUnit(t);
    return radius / max(boundaryRadius, 1e-5f);
}

inline float distancePointToSegmentGpu(float px, float py, float ax, float ay, float bx, float by) {
    const float vx = bx - ax;
    const float vy = by - ay;
    const float len2 = vx * vx + vy * vy;
    if (len2 <= 1e-10f) {
        return length(float2(px - ax, py - ay));
    }
    const float t = clamp(((px - ax) * vx + (py - ay) * vy) / len2, 0.0f, 1.0f);
    const float cx = ax + t * vx;
    const float cy = ay + t * vy;
    return length(float2(px - cx, py - cy));
}

inline bool squareGridMaskOpenGpu(float nx,
                                  float ny,
                                  int bladeCount,
                                  float roundness,
                                  float rotationRad,
                                  float outerRadius) {
    const float cs = cos(-rotationRad);
    const float sn = sin(-rotationRad);
    const float rx = nx * cs - ny * sn;
    const float ry = nx * sn + ny * cs;
    const float squareHalf = outerRadius * 0.82f;
    if (abs(rx) > squareHalf || abs(ry) > squareHalf) {
        return false;
    }

    const int divisions = max(3, bladeCount);
    const float pitch = (2.0f * squareHalf) / float(divisions);
    const float barHalf = pitch * (0.06f + (1.0f - clampUnit(roundness)) * 0.12f);
    float wrappedX = fmod(rx + squareHalf, pitch);
    float wrappedY = fmod(ry + squareHalf, pitch);
    if (wrappedX < 0.0f) wrappedX += pitch;
    if (wrappedY < 0.0f) wrappedY += pitch;
    const float distX = min(wrappedX, pitch - wrappedX);
    const float distY = min(wrappedY, pitch - wrappedY);
    const float edgeDistX = abs(abs(rx) - squareHalf);
    const float edgeDistY = abs(abs(ry) - squareHalf);
    return distX <= barHalf || distY <= barHalf || edgeDistX <= barHalf || edgeDistY <= barHalf;
}

inline bool snowflakeMaskOpenGpu(float nx,
                                 float ny,
                                 int bladeCount,
                                 float roundness,
                                 float rotationRad,
                                 float outerRadius) {
    const float radius = length(float2(nx, ny));
    if (radius > outerRadius) {
        return false;
    }

    const int branchLevels = max(2, min(5, max(2, bladeCount / 2)));
    const float mainLength = outerRadius * 0.92f;
    const float branchAngle = kLensDiffPi / 5.5f;
    const float baseThickness = outerRadius * (0.045f + (1.0f - clampUnit(roundness)) * 0.05f);
    float minDistance = 1e20f;
    for (int arm = 0; arm < 6; ++arm) {
        const float angle = rotationRad - kLensDiffPi * 0.5f + float(arm) * (kLensDiffPi / 3.0f);
        const float cs = cos(angle);
        const float sn = sin(angle);
        const float ex = cs * mainLength;
        const float ey = sn * mainLength;
        minDistance = min(minDistance, distancePointToSegmentGpu(nx, ny, 0.0f, 0.0f, ex, ey));
        for (int level = 0; level < branchLevels; ++level) {
            const float t = 0.34f + float(level) * (0.44f / float(max(1, branchLevels - 1)));
            const float mx = cs * mainLength * t;
            const float my = sn * mainLength * t;
            const float branchLength = outerRadius * (0.16f + 0.03f * float(level));
            for (int side = -1; side <= 1; side += 2) {
                const float branchTheta = angle + float(side) * branchAngle;
                const float bx = mx + cos(branchTheta) * branchLength;
                const float by = my + sin(branchTheta) * branchLength;
                minDistance = min(minDistance, distancePointToSegmentGpu(nx, ny, mx, my, bx, by));
            }
        }
    }
    return minDistance <= baseThickness;
}

inline bool spiralMaskOpenGpu(float nx,
                              float ny,
                              int bladeCount,
                              float roundness,
                              float rotationRad,
                              float outerRadius) {
    const float radius = length(float2(nx, ny));
    if (radius > outerRadius) {
        return false;
    }
    const float radialNorm = radius / max(outerRadius, 1e-6f);
    const float angle = atan2(ny, nx) - rotationRad;
    const float clampedRoundness = clampUnit(roundness);
    const float twist = 4.0f + (1.0f - clampedRoundness) * 8.0f;
    const float opening = 0.18f + clampedRoundness * 0.26f;
    const float phase = float(max(3, bladeCount)) * angle + twist * radialNorm * (2.0f * kLensDiffPi);
    const float band = 0.5f + 0.5f * cos(phase);
    return band >= (1.0f - opening);
}

inline float apodizationWeightGpu(int mode, float radialNorm) {
    switch (mode) {
        case 1:
            return cos(min(radialNorm, 1.0f) * (kLensDiffPi * 0.5f));
        case 2:
            return exp(-4.0f * radialNorm * radialNorm);
        case 0:
        default:
            return 1.0f;
    }
}

inline float evaluatePhaseWavesGpu(constant PhaseRasterParamsGpu& params, float px, float py) {
    constexpr float kZernikeDefocusNorm = 1.7320508075688772f;
    constexpr float kZernikeAstigNorm = 2.4494897427831781f;
    constexpr float kZernikeComaNorm = 2.8284271247461901f;
    constexpr float kZernikeSphericalNorm = 2.2360679774997898f;
    constexpr float kZernikeTrefoilNorm = 2.8284271247461901f;
    constexpr float kZernikeSecondaryAstigNorm = 3.1622776601683795f;
    constexpr float kZernikeQuadrafoilNorm = 3.1622776601683795f;
    constexpr float kZernikeSecondaryComaNorm = 3.4641016151377544f;

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

inline uint bitReverseGpu(uint value, uint bits) {
    uint out = 0u;
    for (uint i = 0u; i < bits; ++i) {
        out = (out << 1u) | (value & 1u);
        value >>= 1u;
    }
    return out;
}

inline float2 complexMul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

kernel void lensDiffBuildPupilKernel(device const float* customImage [[buffer(0)]],
                                     device float* outPupil [[buffer(1)]],
                                     constant PupilRasterParamsGpu& params [[buffer(2)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.size || int(gid.y) >= params.size) {
        return;
    }
    const uint index = gid.y * uint(params.size) + gid.x;
    outPupil[index] = 0.0f;

    const float nx = ((float(gid.x) + 0.5f) / float(params.size) - 0.5f) * 2.0f;
    const float ny = ((float(gid.y) + 0.5f) / float(params.size) - 0.5f) * 2.0f;
    const float dx = nx - params.pupilDecenterX;
    const float dy = ny - params.pupilDecenterY;
    const float radius = length(float2(dx, dy));
    if (radius > params.outerRadius) {
        return;
    }

    if (params.apertureMode == 4) {
        if (customImage == nullptr || params.customWidth <= 0 || params.customHeight <= 0) {
            return;
        }
        const float cs = cos(-params.rotationRad);
        const float sn = sin(-params.rotationRad);
        const float rx = dx * cs - dy * sn;
        const float ry = dx * sn + dy * cs;
        const float ux = rx / params.outerRadius;
        const float uy = ry / params.outerRadius;
        if (abs(ux) > params.fitHalfWidth || abs(uy) > params.fitHalfHeight) {
            return;
        }
        const float sx = ((ux / params.fitHalfWidth) * 0.5f + 0.5f) * float(params.customWidth - 1);
        const float sy = ((uy / params.fitHalfHeight) * 0.5f + 0.5f) * float(params.customHeight - 1);
        const float radialNorm = radius / params.outerRadius;
        outPupil[index] = samplePlaneBilinear(customImage, params.customWidth, params.customHeight, sx, sy) *
                          apodizationWeightGpu(params.apodizationMode, radialNorm);
        return;
    }

    bool insideShape = false;
    const bool useHexagon = params.apertureMode == 5;
    const bool useStar = params.apertureMode == 2;
    const bool useSpiral = params.apertureMode == 3;
    const bool useSquareGrid = params.apertureMode == 6;
    const bool useSnowflake = params.apertureMode == 7;
    if (params.apertureMode == 1 || useHexagon) {
        const int sides = useHexagon ? 6 : max(3, params.bladeCount);
        insideShape = polygonMetricGpu(dx, dy, sides, params.roundness, params.rotationRad, params.outerRadius) <= 1.0f;
    } else if (useSquareGrid) {
        insideShape = squareGridMaskOpenGpu(dx, dy, max(3, params.bladeCount), params.roundness, params.rotationRad, params.outerRadius);
    } else if (useSnowflake) {
        insideShape = snowflakeMaskOpenGpu(dx, dy, max(3, params.bladeCount), params.roundness, params.rotationRad, params.outerRadius);
    } else if (useStar) {
        insideShape = starMetricGpu(dx, dy, max(3, params.bladeCount), params.starInnerRadiusRatio, params.rotationRad, params.outerRadius) <= 1.0f;
    } else if (useSpiral) {
        insideShape = spiralMaskOpenGpu(dx, dy, max(3, params.bladeCount), params.roundness, params.rotationRad, params.outerRadius);
    } else {
        const float metric = polygonMetricGpu(dx, dy, max(3, params.bladeCount), params.roundness, params.rotationRad, params.outerRadius);
        insideShape = params.apertureMode == 0 ? radius <= params.outerRadius : metric <= 1.0f;
    }

    if (!insideShape || radius < params.centralObstruction) {
        return;
    }
    if (blockedByVanesGpu(dx, dy, params.vaneCount, params.vaneThickness, params.rotationRad, params.outerRadius)) {
        return;
    }
    outPupil[index] = apodizationWeightGpu(params.apodizationMode, radius / params.outerRadius);
}

kernel void lensDiffBuildPhaseKernel(device float* outPhase [[buffer(0)]],
                                     constant PhaseRasterParamsGpu& params [[buffer(1)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.size || int(gid.y) >= params.size) {
        return;
    }
    const uint index = gid.y * uint(params.size) + gid.x;
    outPhase[index] = 0.0f;
    if (params.hasPhase == 0) {
        return;
    }

    const float nx = ((float(gid.x) + 0.5f) / float(params.size) - 0.5f) * 2.0f;
    const float ny = ((float(gid.y) + 0.5f) / float(params.size) - 0.5f) * 2.0f;
    const float radius = length(float2(nx, ny));
    if (radius > params.outerRadius) {
        return;
    }
    const float cs = cos(-params.rotationRad);
    const float sn = sin(-params.rotationRad);
    const float rx = nx * cs - ny * sn;
    const float ry = nx * sn + ny * cs;
    outPhase[index] = evaluatePhaseWavesGpu(params, rx / params.outerRadius, ry / params.outerRadius);
}

kernel void lensDiffEmbedComplexPupilKernel(device const float* amplitude [[buffer(0)]],
                                            device const float* phaseWaves [[buffer(1)]],
                                            device float2* dst [[buffer(2)]],
                                            constant EmbedComplexParamsGpu& params [[buffer(3)]],
                                            uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.pupilSize || int(gid.y) >= params.pupilSize) {
        return;
    }
    const uint srcIndex = gid.y * uint(params.pupilSize) + gid.x;
    const uint dstIndex = uint(int(gid.y) + params.offset) * uint(params.rawPsfSize) + uint(int(gid.x) + params.offset);
    const float amplitudeValue = amplitude[srcIndex];
    const float phaseRadians = phaseWaves != nullptr ? phaseWaves[srcIndex] * (2.0f * kLensDiffPi) : 0.0f;
    dst[dstIndex] = float2(amplitudeValue * cos(phaseRadians), amplitudeValue * sin(phaseRadians));
}

kernel void lensDiffBitReverseRowsKernel(device const float2* src [[buffer(0)]],
                                         device float2* dst [[buffer(1)]],
                                         constant FftParamsGpu& params [[buffer(2)]],
                                         uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.size || int(gid.y) >= params.size) {
        return;
    }
    const uint reversed = bitReverseGpu(gid.x, uint(params.log2Size));
    dst[gid.y * uint(params.size) + reversed] = src[gid.y * uint(params.size) + gid.x];
}

kernel void lensDiffFftRowsStageKernel(device const float2* src [[buffer(0)]],
                                       device float2* dst [[buffer(1)]],
                                       constant FftStageParamsGpu& params [[buffer(2)]],
                                       uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.size / 2 || int(gid.y) >= params.size) {
        return;
    }
    const int butterfly = int(gid.x);
    const int group = butterfly / params.halfSize;
    const int k = butterfly % params.halfSize;
    const int base = group * params.stageSize;
    const int evenIndex = base + k;
    const int oddIndex = evenIndex + params.halfSize;
    const float angle = -2.0f * kLensDiffPi * float(k) / float(params.stageSize);
    const float2 twiddle = float2(cos(angle), sin(angle));
    const uint rowBase = gid.y * uint(params.size);
    const float2 u = src[rowBase + uint(evenIndex)];
    const float2 v = complexMul(src[rowBase + uint(oddIndex)], twiddle);
    dst[rowBase + uint(evenIndex)] = u + v;
    dst[rowBase + uint(oddIndex)] = u - v;
}

kernel void lensDiffBitReverseColumnsKernel(device const float2* src [[buffer(0)]],
                                            device float2* dst [[buffer(1)]],
                                            constant FftParamsGpu& params [[buffer(2)]],
                                            uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.size || int(gid.y) >= params.size) {
        return;
    }
    const uint reversed = bitReverseGpu(gid.y, uint(params.log2Size));
    dst[reversed * uint(params.size) + gid.x] = src[gid.y * uint(params.size) + gid.x];
}

kernel void lensDiffFftColumnsStageKernel(device const float2* src [[buffer(0)]],
                                          device float2* dst [[buffer(1)]],
                                          constant FftStageParamsGpu& params [[buffer(2)]],
                                          uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.size || int(gid.y) >= params.size / 2) {
        return;
    }
    const int butterfly = int(gid.y);
    const int group = butterfly / params.halfSize;
    const int k = butterfly % params.halfSize;
    const int base = group * params.stageSize;
    const int evenIndex = base + k;
    const int oddIndex = evenIndex + params.halfSize;
    const float angle = -2.0f * kLensDiffPi * float(k) / float(params.stageSize);
    const float2 twiddle = float2(cos(angle), sin(angle));
    const uint x = gid.x;
    const uint evenOffset = uint(evenIndex) * uint(params.size) + x;
    const uint oddOffset = uint(oddIndex) * uint(params.size) + x;
    const float2 u = src[evenOffset];
    const float2 v = complexMul(src[oddOffset], twiddle);
    dst[evenOffset] = u + v;
    dst[oddOffset] = u - v;
}

kernel void lensDiffExtractShiftedIntensityKernel(device const float2* spectrum [[buffer(0)]],
                                                  device float* out [[buffer(1)]],
                                                  constant FftParamsGpu& params [[buffer(2)]],
                                                  uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.size || int(gid.y) >= params.size) {
        return;
    }
    const uint halfOffset = uint(params.size / 2);
    const uint sx = (gid.x + halfOffset) % uint(params.size);
    const uint sy = (gid.y + halfOffset) % uint(params.size);
    const float2 value = spectrum[sy * uint(params.size) + sx];
    out[gid.y * uint(params.size) + gid.x] = dot(value, value);
}

kernel void lensDiffTransposeComplexKernel(device const float2* src [[buffer(0)]],
                                           device float2* dst [[buffer(1)]],
                                           constant FftParamsGpu& params [[buffer(2)]],
                                           uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.size || int(gid.y) >= params.size) {
        return;
    }
    dst[gid.x * uint(params.size) + gid.y] = src[gid.y * uint(params.size) + gid.x];
}

kernel void lensDiffTransposeComplexStackKernel(device const float2* src [[buffer(0)]],
                                                device float2* dst [[buffer(1)]],
                                                constant FftImageParamsGpu& params [[buffer(2)]],
                                                uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.paddedSize || int(gid.y) >= params.paddedSize * params.batchCount) {
        return;
    }
    const int batchIndex = int(gid.y) / params.paddedSize;
    const int localY = int(gid.y) - batchIndex * params.paddedSize;
    const uint srcIndex = uint(batchIndex * params.batchStride + localY * params.paddedSize + int(gid.x));
    const uint dstIndex = uint(batchIndex * params.batchStride + int(gid.x) * params.paddedSize + localY);
    dst[dstIndex] = src[srcIndex];
}

kernel void lensDiffBitReverseBatchedKernel(device const float2* src [[buffer(0)]],
                                            device float2* dst [[buffer(1)]],
                                            constant BatchFftParamsGpu& params [[buffer(2)]],
                                            uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.length || int(gid.y) >= params.batchCount) {
        return;
    }
    const uint reversed = bitReverseGpu(gid.x, uint(params.log2Length));
    dst[gid.y * uint(params.batchStride) + reversed] = src[gid.y * uint(params.batchStride) + gid.x];
}

kernel void lensDiffFftBatchedStageKernel(device const float2* src [[buffer(0)]],
                                          device float2* dst [[buffer(1)]],
                                          constant BatchFftStageParamsGpu& params [[buffer(2)]],
                                          uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.length / 2 || int(gid.y) >= params.batchCount) {
        return;
    }
    const int butterfly = int(gid.x);
    const int group = butterfly / params.halfSize;
    const int k = butterfly % params.halfSize;
    const int base = group * params.stageSize;
    const int evenIndex = base + k;
    const int oddIndex = evenIndex + params.halfSize;
    const float angle = (params.inverse != 0 ? 2.0f : -2.0f) * kLensDiffPi * float(k) / float(params.stageSize);
    const float2 twiddle = float2(cos(angle), sin(angle));
    const uint batchBase = gid.y * uint(params.batchStride);
    const float2 u = src[batchBase + uint(evenIndex)];
    const float2 v = complexMul(src[batchBase + uint(oddIndex)], twiddle);
    dst[batchBase + uint(evenIndex)] = u + v;
    dst[batchBase + uint(oddIndex)] = u - v;
}

kernel void lensDiffScaleComplexBatchedKernel(device float2* values [[buffer(0)]],
                                              constant BatchFftParamsGpu& params [[buffer(1)]],
                                              uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.length || int(gid.y) >= params.batchCount) {
        return;
    }
    const float scale = 1.0f / max(float(params.length), 1.0f);
    values[gid.y * uint(params.batchStride) + gid.x] *= scale;
}

kernel void lensDiffCopyComplexBatchedKernel(device const float2* src [[buffer(0)]],
                                             device float2* dst [[buffer(1)]],
                                             constant BatchFftParamsGpu& params [[buffer(2)]],
                                             uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.length || int(gid.y) >= params.batchCount) {
        return;
    }
    const uint index = gid.y * uint(params.batchStride) + gid.x;
    dst[index] = src[index];
}

kernel void lensDiffBuildBluesteinChirpKernel(device float2* out [[buffer(0)]],
                                              constant BluesteinParamsGpu& params [[buffer(1)]],
                                              uint gid [[thread_position_in_grid]]) {
    if (int(gid) >= params.convolutionLength) {
        return;
    }
    out[gid] = float2(0.0f, 0.0f);
    if (int(gid) < params.signalLength) {
        const float k = float(gid);
        const float angle = kLensDiffPi * (k * k) / float(params.signalLength);
        const float2 chirp = float2(cos(angle), sin(angle));
        out[gid] = chirp;
        if (gid != 0u) {
            out[uint(params.convolutionLength) - gid] = chirp;
        }
    }
}

kernel void lensDiffBuildBluesteinInputKernel(device const float2* src [[buffer(0)]],
                                              device float2* dst [[buffer(1)]],
                                              constant BluesteinParamsGpu& params [[buffer(2)]],
                                              uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.convolutionLength || int(gid.y) >= params.batchCount) {
        return;
    }
    const uint index = gid.y * uint(params.convolutionLength) + gid.x;
    dst[index] = float2(0.0f, 0.0f);
    if (int(gid.x) >= params.signalLength) {
        return;
    }
    const float k = float(gid.x);
    const float angle = -kLensDiffPi * (k * k) / float(params.signalLength);
    const float2 chirp = float2(cos(angle), sin(angle));
    dst[index] = complexMul(src[gid.y * uint(params.signalLength) + gid.x], chirp);
}

kernel void lensDiffMultiplyBluesteinSpectraKernel(device const float2* lhs [[buffer(0)]],
                                                   device const float2* rhs [[buffer(1)]],
                                                   device float2* dst [[buffer(2)]],
                                                   constant BluesteinParamsGpu& params [[buffer(3)]],
                                                   uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.convolutionLength || int(gid.y) >= params.batchCount) {
        return;
    }
    const uint index = gid.y * uint(params.convolutionLength) + gid.x;
    dst[index] = complexMul(lhs[index], rhs[gid.x]);
}

kernel void lensDiffExtractBluesteinOutputKernel(device const float2* src [[buffer(0)]],
                                                 device float2* dst [[buffer(1)]],
                                                 constant BluesteinParamsGpu& params [[buffer(2)]],
                                                 uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.signalLength || int(gid.y) >= params.batchCount) {
        return;
    }
    const float k = float(gid.x);
    const float angle = -kLensDiffPi * (k * k) / float(params.signalLength);
    const float2 chirp = float2(cos(angle), sin(angle));
    const uint srcIndex = gid.y * uint(params.convolutionLength) + gid.x;
    const uint dstIndex = gid.y * uint(params.signalLength) + gid.x;
    dst[dstIndex] = complexMul(src[srcIndex], chirp);
}

kernel void lensDiffDecodeSourceKernel(device const float* src [[buffer(0)]],
                                       device float4* linearSrc [[buffer(1)]],
                                       constant DecodeParamsGpu& params [[buffer(2)]],
                                       uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.srcWidth || int(gid.y) >= params.srcHeight) {
        return;
    }
    const uint index = gid.y * uint(params.srcWidth) + gid.x;
    const float4 raw = loadSourcePixel(src, params.srcRowFloats, params.srcWidth, params.srcHeight, int(gid.x), int(gid.y));
    linearSrc[index] = float4(
        decodeChannel(raw.x, params.inputTransfer),
        decodeChannel(raw.y, params.inputTransfer),
        decodeChannel(raw.z, params.inputTransfer),
        raw.w);
}

kernel void lensDiffResampleRgbaKernel(device const float4* src [[buffer(0)]],
                                       device float4* dst [[buffer(1)]],
                                       constant ResampleParamsGpu& params [[buffer(2)]],
                                       uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.dstWidth || int(gid.y) >= params.dstHeight) {
        return;
    }
    const float scaleX = float(params.srcWidth) / float(max(1, params.dstWidth));
    const float scaleY = float(params.srcHeight) / float(max(1, params.dstHeight));
    const float sx = (float(gid.x) + 0.5f) * scaleX - 0.5f;
    const float sy = (float(gid.y) + 0.5f) * scaleY - 0.5f;
    dst[gid.y * uint(params.dstWidth) + gid.x] = sampleRgbaBilinear(src, params.srcWidth, params.srcHeight, sx, sy);
}

kernel void lensDiffResampleGrayKernel(device const float* src [[buffer(0)]],
                                       device float* dst [[buffer(1)]],
                                       constant ResampleParamsGpu& params [[buffer(2)]],
                                       uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.dstWidth || int(gid.y) >= params.dstHeight) {
        return;
    }
    const float scaleX = float(params.srcWidth) / float(max(1, params.dstWidth));
    const float scaleY = float(params.srcHeight) / float(max(1, params.dstHeight));
    const float sx = (float(gid.x) + 0.5f) * scaleX - 0.5f;
    const float sy = (float(gid.y) + 0.5f) * scaleY - 0.5f;
    dst[gid.y * uint(params.dstWidth) + gid.x] = samplePlaneBilinear(src, params.srcWidth, params.srcHeight, sx, sy);
}

kernel void lensDiffPrepareFromLinearKernel(device const float4* linearSrc [[buffer(0)]],
                                            device float4* redistributed [[buffer(1)]],
                                            device float* driver [[buffer(2)]],
                                            device float* mask [[buffer(3)]],
                                            constant PrepareParamsGpu& params [[buffer(4)]],
                                            uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }

    const uint index = gid.y * uint(params.width) + gid.x;
    const float4 linearSrcPixel = linearSrc[index];
    const float3 linear = linearSrcPixel.xyz;
    const float signal = params.extractionMode == 1 ? safeLuma(linear) : max(max(linear.x, linear.y), linear.z);
    float gate = 0.0f;
    if (signal > 0.0f) {
        const float stops = log2(max(signal, 1e-6f) / kGray18);
        const float edge0 = params.threshold - params.softnessStops * 0.5f;
        const float edge1 = params.threshold + params.softnessStops * 0.5f;
        const float t = saturateSafe((stops - edge0) / max(edge1 - edge0, 1e-4f));
        gate = t * t * (3.0f - 2.0f * t);
    }

    const float thresholdLinear = kGray18 * exp2(params.threshold);
    const float maxRgb = max(max(linear.x, linear.y), linear.z);
    const float pointBoost = 1.0f + params.pointEmphasis * max(0.0f, maxRgb / max(thresholdLinear, 1e-4f) - 1.0f);
    const float finalMask = saturateSafe(gate * pointBoost);
    const float redistributionScale = 1.0f - clamp(params.corePreserve, 0.0f, 1.0f);
    const float m = finalMask * redistributionScale;

    mask[index] = finalMask;
    redistributed[index] = float4(linear * m, linearSrcPixel.w);
    driver[index] = signal * m;
}

kernel void lensDiffPrepareKernel(device const float* src [[buffer(0)]],
                                  device float4* linearSrc [[buffer(1)]],
                                  device float4* redistributed [[buffer(2)]],
                                  device float* driver [[buffer(3)]],
                                  device float* mask [[buffer(4)]],
                                  constant PrepareParamsGpu& params [[buffer(5)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }

    const uint index = gid.y * uint(params.width) + gid.x;
    const float4 raw = loadSourcePixel(src, 0u, params.width, params.height, int(gid.x), int(gid.y));
    const float3 linear = raw.xyz;
    linearSrc[index] = raw;

    const float signal = params.extractionMode == 1 ? safeLuma(linear) : max(max(linear.x, linear.y), linear.z);
    float gate = 0.0f;
    if (signal > 0.0f) {
        const float stops = log2(max(signal, 1e-6f) / kGray18);
        const float edge0 = params.threshold - params.softnessStops * 0.5f;
        const float edge1 = params.threshold + params.softnessStops * 0.5f;
        const float t = saturateSafe((stops - edge0) / max(edge1 - edge0, 1e-4f));
        gate = t * t * (3.0f - 2.0f * t);
    }

    const float thresholdLinear = kGray18 * exp2(params.threshold);
    const float maxRgb = max(max(linear.x, linear.y), linear.z);
    const float pointBoost = 1.0f + params.pointEmphasis * max(0.0f, maxRgb / max(thresholdLinear, 1e-4f) - 1.0f);
    const float finalMask = saturateSafe(gate * pointBoost);
    const float redistributionScale = 1.0f - clamp(params.corePreserve, 0.0f, 1.0f);
    const float m = finalMask * redistributionScale;

    mask[index] = finalMask;
    redistributed[index] = float4(linear * m, raw.w);
    driver[index] = signal * m;
}

kernel void lensDiffPadPlaneToComplexKernel(device const float* src [[buffer(0)]],
                                            device float2* dst [[buffer(1)]],
                                            constant FftImageParamsGpu& params [[buffer(2)]],
                                            uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.paddedSize || int(gid.y) >= params.paddedSize) {
        return;
    }
    const uint index = gid.y * uint(params.paddedSize) + gid.x;
    float value = 0.0f;
    if (int(gid.x) < params.width && int(gid.y) < params.height) {
        value = src[gid.y * uint(params.width) + gid.x];
    }
    dst[index] = float2(value, 0.0f);
}

kernel void lensDiffPadRgbChannelToComplexKernel(device const float4* src [[buffer(0)]],
                                                 device float2* dst [[buffer(1)]],
                                                 constant FftImageParamsGpu& params [[buffer(2)]],
                                                 uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.paddedSize || int(gid.y) >= params.paddedSize) {
        return;
    }
    const uint index = gid.y * uint(params.paddedSize) + gid.x;
    float value = 0.0f;
    if (int(gid.x) < params.width && int(gid.y) < params.height) {
        const float4 pixel = src[gid.y * uint(params.width) + gid.x];
        switch (params.channelIndex) {
            case 1: value = pixel.y; break;
            case 2: value = pixel.z; break;
            default: value = pixel.x; break;
        }
    }
    dst[index] = float2(value, 0.0f);
}

kernel void lensDiffPadRgbToComplexStackKernel(device const float4* src [[buffer(0)]],
                                               device float2* dst [[buffer(1)]],
                                               constant FftImageParamsGpu& params [[buffer(2)]],
                                               uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.paddedSize || int(gid.y) >= params.paddedSize * params.batchCount) {
        return;
    }
    const int batchIndex = int(gid.y) / params.paddedSize;
    const int localY = int(gid.y) - batchIndex * params.paddedSize;
    const uint index = uint(batchIndex * params.batchStride + localY * params.paddedSize + int(gid.x));
    float value = 0.0f;
    if (int(gid.x) < params.width && localY < params.height) {
        const float4 pixel = src[uint(localY) * uint(params.width) + gid.x];
        switch (batchIndex) {
            case 1: value = pixel.y; break;
            case 2: value = pixel.z; break;
            default: value = pixel.x; break;
        }
    }
    dst[index] = float2(value, 0.0f);
}

kernel void lensDiffScatterKernelToComplexKernel(device const float* kernelValues [[buffer(0)]],
                                                 device float2* dst [[buffer(1)]],
                                                 constant FftImageParamsGpu& params [[buffer(2)]],
                                                 uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.kernelSize || int(gid.y) >= params.kernelSize) {
        return;
    }
    const int radius = params.kernelSize / 2;
    const int dx = (int(gid.x) - radius + params.paddedSize) % params.paddedSize;
    const int dy = (int(gid.y) - radius + params.paddedSize) % params.paddedSize;
    dst[uint(dy) * uint(params.paddedSize) + uint(dx)] =
        float2(kernelValues[gid.y * uint(params.kernelSize) + gid.x], 0.0f);
}

kernel void lensDiffMultiplyComplexKernel(device const float2* lhs [[buffer(0)]],
                                          device const float2* rhs [[buffer(1)]],
                                          device float2* dst [[buffer(2)]],
                                          uint gid [[thread_position_in_grid]]) {
    dst[gid] = complexMul(lhs[gid], rhs[gid]);
}

kernel void lensDiffMultiplyComplexBroadcastKernel(device const float2* lhs [[buffer(0)]],
                                                   device const float2* rhs [[buffer(1)]],
                                                   device float2* dst [[buffer(2)]],
                                                   constant BatchFftParamsGpu& params [[buffer(3)]],
                                                   uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.length || int(gid.y) >= params.batchCount) {
        return;
    }
    const uint batchBase = gid.y * uint(params.batchStride);
    const uint index = batchBase + gid.x;
    dst[index] = complexMul(lhs[index], rhs[gid.x]);
}

kernel void lensDiffMultiplyComplexPairsStackKernel(device const float2* lhs [[buffer(0)]],
                                                    device const float2* rhs [[buffer(1)]],
                                                    device float2* dst [[buffer(2)]],
                                                    constant BatchFftParamsGpu& params [[buffer(3)]],
                                                    uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.length || int(gid.y) >= params.batchCount) {
        return;
    }
    const uint batchBase = gid.y * uint(params.batchStride);
    const uint index = batchBase + gid.x;
    dst[index] = complexMul(lhs[index], rhs[index]);
}

kernel void lensDiffReplicateComplexStackKernel(device const float2* src [[buffer(0)]],
                                                device float2* dst [[buffer(1)]],
                                                constant ReplicateComplexParamsGpu& params [[buffer(2)]],
                                                uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.length || int(gid.y) >= params.dstBatchCount) {
        return;
    }
    const int srcBatch = params.srcBatchCount > 0 ? (int(gid.y) % params.srcBatchCount) : 0;
    const uint srcIndex = uint(srcBatch * params.srcStride) + gid.x;
    const uint dstIndex = uint(int(gid.y) * params.dstStride) + gid.x;
    dst[dstIndex] = src[srcIndex];
}

kernel void lensDiffExtractRealPlaneKernel(device const float2* spectrum [[buffer(0)]],
                                           device float* dst [[buffer(1)]],
                                           constant FftImageParamsGpu& params [[buffer(2)]],
                                           uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }
    const uint srcIndex = gid.y * uint(params.paddedSize) + gid.x;
    const uint dstIndex = gid.y * uint(params.width) + gid.x;
    const float scale = 1.0f / max(float(params.batchStride), 1.0f);
    dst[dstIndex] = max(0.0f, spectrum[srcIndex].x * scale);
}

kernel void lensDiffPackPlanesToRgbaKernel(device const float* rPlane [[buffer(0)]],
                                           device const float* gPlane [[buffer(1)]],
                                           device const float* bPlane [[buffer(2)]],
                                           device float4* dst [[buffer(3)]],
                                           constant FftImageParamsGpu& params [[buffer(4)]],
                                           uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }
    const uint index = gid.y * uint(params.width) + gid.x;
    dst[index] = float4(max(rPlane[index], 0.0f), max(gPlane[index], 0.0f), max(bPlane[index], 0.0f), 1.0f);
}

kernel void lensDiffPackPlaneTripletsToRgbaStackKernel(device const float* planeStack [[buffer(0)]],
                                                       device float4* dst [[buffer(1)]],
                                                       constant StackImageParamsGpu& params [[buffer(2)]],
                                                       uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height * params.stackDepth) {
        return;
    }
    const int stackIndex = int(gid.y) / params.height;
    const int localY = int(gid.y) - stackIndex * params.height;
    const uint pixelIndex = uint(localY * params.width + int(gid.x));
    const uint sliceBase = uint(stackIndex * 3 * params.planeStride);
    const float r = planeStack[sliceBase + pixelIndex];
    const float g = planeStack[sliceBase + uint(params.planeStride) + pixelIndex];
    const float b = planeStack[sliceBase + uint(params.planeStride * 2) + pixelIndex];
    dst[uint(stackIndex * params.planeStride) + pixelIndex] = float4(max(r, 0.0f), max(g, 0.0f), max(b, 0.0f), 1.0f);
}

kernel void lensDiffConvolveRgbKernel(device const float4* src [[buffer(0)]],
                                      device const float* kernelValues [[buffer(1)]],
                                      device float4* dst [[buffer(2)]],
                                      constant ConvolutionParamsGpu& params [[buffer(3)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }

    float3 accum = float3(0.0f);
    for (int ky = 0; ky < params.kernelSize; ++ky) {
        const int sy = int(gid.y) + ky - params.kernelRadius;
        if (sy < 0 || sy >= params.height) {
            continue;
        }
        for (int kx = 0; kx < params.kernelSize; ++kx) {
            const int sx = int(gid.x) + kx - params.kernelRadius;
            if (sx < 0 || sx >= params.width) {
                continue;
            }
            const float weight = kernelValues[uint(ky * params.kernelSize + kx)];
            const float4 sample = src[uint(sy * params.width + sx)];
            accum += sample.xyz * weight;
        }
    }
    dst[gid.y * uint(params.width) + gid.x] = float4(max(accum, float3(0.0f)), 1.0f);
}

kernel void lensDiffConvolveScalarKernel(device const float* src [[buffer(0)]],
                                         device const float* kernelValues [[buffer(1)]],
                                         device float* dst [[buffer(2)]],
                                         constant ConvolutionParamsGpu& params [[buffer(3)]],
                                         uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }

    float accum = 0.0f;
    for (int ky = 0; ky < params.kernelSize; ++ky) {
        const int sy = int(gid.y) + ky - params.kernelRadius;
        if (sy < 0 || sy >= params.height) {
            continue;
        }
        for (int kx = 0; kx < params.kernelSize; ++kx) {
            const int sx = int(gid.x) + kx - params.kernelRadius;
            if (sx < 0 || sx >= params.width) {
                continue;
            }
            const float weight = kernelValues[uint(ky * params.kernelSize + kx)];
            accum += src[uint(sy * params.width + sx)] * weight;
        }
    }
    dst[gid.y * uint(params.width) + gid.x] = max(accum, 0.0f);
}

kernel void lensDiffApplyShoulderKernel(device float4* image [[buffer(0)]],
                                        constant ShoulderParamsGpu& params [[buffer(1)]],
                                        uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }
    const uint index = gid.y * uint(params.width) + gid.x;
    float4 pixel = image[index];
    pixel.x = softShoulderSafe(pixel.x, params.shoulder);
    pixel.y = softShoulderSafe(pixel.y, params.shoulder);
    pixel.z = softShoulderSafe(pixel.z, params.shoulder);
    image[index] = pixel;
}

kernel void lensDiffApplyShoulderStackKernel(device float4* image [[buffer(0)]],
                                             constant StackImageParamsGpu& stackParams [[buffer(1)]],
                                             constant ShoulderParamsGpu& params [[buffer(2)]],
                                             uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= stackParams.width || int(gid.y) >= stackParams.height * stackParams.stackDepth) {
        return;
    }
    const int stackIndex = int(gid.y) / stackParams.height;
    const int localY = int(gid.y) - stackIndex * stackParams.height;
    const uint index = uint(stackIndex * stackParams.planeStride + localY * stackParams.width + int(gid.x));
    float4 pixel = image[index];
    pixel.x = softShoulderSafe(pixel.x, params.shoulder);
    pixel.y = softShoulderSafe(pixel.y, params.shoulder);
    pixel.z = softShoulderSafe(pixel.z, params.shoulder);
    image[index] = pixel;
}

kernel void lensDiffCombineKernel(device const float4* core [[buffer(0)]],
                                  device const float4* structure [[buffer(1)]],
                                  device float4* effect [[buffer(2)]],
                                  constant CombineParamsGpu& params [[buffer(3)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }
    const uint index = gid.y * uint(params.width) + gid.x;
    const float3 rgb = core[index].xyz * max(0.0f, params.coreGain) +
                       structure[index].xyz * max(0.0f, params.structureGain);
    effect[index] = float4(max(rgb, float3(0.0f)), 1.0f);
}

kernel void lensDiffCombineStackKernel(device const float4* core [[buffer(0)]],
                                       device const float4* structure [[buffer(1)]],
                                       device float4* effect [[buffer(2)]],
                                       constant StackImageParamsGpu& stackParams [[buffer(3)]],
                                       constant CombineParamsGpu& params [[buffer(4)]],
                                       uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= stackParams.width || int(gid.y) >= stackParams.height * stackParams.stackDepth) {
        return;
    }
    const int stackIndex = int(gid.y) / stackParams.height;
    const int localY = int(gid.y) - stackIndex * stackParams.height;
    const uint index = uint(stackIndex * stackParams.planeStride + localY * stackParams.width + int(gid.x));
    const float3 rgb = core[index].xyz * max(0.0f, params.coreGain) +
                       structure[index].xyz * max(0.0f, params.structureGain);
    effect[index] = float4(max(rgb, float3(0.0f)), 1.0f);
}

kernel void lensDiffMapSpectralKernel(device const float* bin0 [[buffer(0)]],
                                      device const float* bin1 [[buffer(1)]],
                                      device const float* bin2 [[buffer(2)]],
                                      device const float* bin3 [[buffer(3)]],
                                      device const float* bin4 [[buffer(4)]],
                                      device const float* bin5 [[buffer(5)]],
                                      device const float* bin6 [[buffer(6)]],
                                      device const float* bin7 [[buffer(7)]],
                                      device const float* bin8 [[buffer(8)]],
                                      device float4* dst [[buffer(9)]],
                                      constant SpectralMapParamsGpu& params [[buffer(10)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }
    const uint index = gid.y * uint(params.width) + gid.x;
    float bins[LENSDIFF_MAX_SPECTRAL_BINS] = {
        bin0[index], bin1[index], bin2[index], bin3[index], bin4[index],
        bin5[index], bin6[index], bin7[index], bin8[index]
    };
    const float3 rgb = mapSpectral(bins, params);
    dst[index] = float4(max(rgb, float3(0.0f)), 1.0f);
}

kernel void lensDiffMapSpectralStackKernel(device const float* planeStack [[buffer(0)]],
                                           device float4* dst [[buffer(1)]],
                                           constant ZonePlaneStackParamsGpu& stackParams [[buffer(2)]],
                                           constant SpectralMapParamsGpu& params [[buffer(3)]],
                                           uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= stackParams.width || int(gid.y) >= stackParams.height * stackParams.zoneCount) {
        return;
    }
    const int zoneIndex = int(gid.y) / stackParams.height;
    const int localY = int(gid.y) - zoneIndex * stackParams.height;
    const uint pixelIndex = uint(localY * stackParams.width + int(gid.x));
    const uint zoneBase = uint(zoneIndex * stackParams.binCount * stackParams.planeStride);
    float bins[LENSDIFF_MAX_SPECTRAL_BINS] = {0.0f};
    const int activeBins = min(params.binCount, stackParams.binCount);
    for (int i = 0; i < activeBins; ++i) {
        bins[i] = planeStack[zoneBase + uint(i * stackParams.planeStride) + pixelIndex];
    }
    const float3 rgb = mapSpectral(bins, params);
    dst[uint(zoneIndex * stackParams.planeStride) + pixelIndex] = float4(max(rgb, float3(0.0f)), 1.0f);
}

kernel void lensDiffCompositeKernel(device const float4* linearSrc [[buffer(0)]],
                                    device const float4* redistributed [[buffer(1)]],
                                    device const float4* effect [[buffer(2)]],
                                    device float4* dst [[buffer(3)]],
                                    constant CompositeParamsGpu& params [[buffer(4)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }
    const uint index = gid.y * uint(params.width) + gid.x;
    const float3 raw = max(float3(0.0f),
                           linearSrc[index].xyz - redistributed[index].xyz * params.coreCompensation +
                               effect[index].xyz * params.effectGain);
    const float3 floorRgb = max(float3(0.0f),
                                linearSrc[index].xyz - redistributed[index].xyz * params.maxRedistributedSubtractScale);
    const float3 rgb = max(raw, floorRgb);
    dst[index] = float4(rgb, linearSrc[index].w);
}

kernel void lensDiffScaleRgbKernel(device float4* image [[buffer(0)]],
                                   constant ScaleParamsGpu& params [[buffer(1)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }
    const uint index = gid.y * uint(params.width) + gid.x;
    image[index] = float4(image[index].xyz * params.scale, image[index].w);
}

kernel void lensDiffScaleScalarKernel(device float* values [[buffer(0)]],
                                      device const float* scale [[buffer(1)]],
                                      constant ScalarScaleParamsGpu& params [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]]) {
    if (int(gid) >= params.count) {
        return;
    }
    values[gid] *= scale[0];
}

kernel void lensDiffComputePreserveScaleKernel(device const float* inputEnergy [[buffer(0)]],
                                               device const float* effectEnergy [[buffer(1)]],
                                               device float* outScale [[buffer(2)]],
                                               constant PreserveScaleParamsGpu& params [[buffer(3)]],
                                               uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) {
        return;
    }
    const float effect = effectEnergy[0];
    const float input = inputEnergy[0];
    outScale[0] = effect > params.epsilon ? input / effect : 1.0f;
}

kernel void lensDiffScaleRgbDynamicKernel(device float4* image [[buffer(0)]],
                                          device const float* scale [[buffer(1)]],
                                          constant DynamicScaleParamsGpu& params [[buffer(2)]],
                                          uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }
    const uint index = gid.y * uint(params.width) + gid.x;
    image[index] = float4(image[index].xyz * scale[0], image[index].w);
}

kernel void lensDiffAccumulateWeightedKernel(device const float4* src [[buffer(0)]],
                                             device float4* dst [[buffer(1)]],
                                             constant FieldBlendParamsGpu& params [[buffer(2)]],
                                             uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }
    const float denomX = float(max(1, params.width - 1));
    const float denomY = float(max(1, params.height - 1));
    const float px = (float(gid.x) / denomX) * 2.0f;
    const float py = (float(gid.y) / denomY) * 2.0f;
    const float wx = max(0.0f, 1.0f - abs(px - float(params.zoneX)));
    const float wy = max(0.0f, 1.0f - abs(py - float(params.zoneY)));
    const float weight = wx * wy * params.gain;
    if (weight <= 0.0f) {
        return;
    }
    const uint index = gid.y * uint(params.width) + gid.x;
    const float4 srcPixel = src[index];
    dst[index] += float4(srcPixel.xyz * weight, 0.0f);
}

kernel void lensDiffAccumulateWeightedRgbStackKernel(device const float4* srcStack [[buffer(0)]],
                                                     device float4* dst [[buffer(1)]],
                                                     constant StackImageParamsGpu& params [[buffer(2)]],
                                                     uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }
    const float denomX = float(max(1, params.width - 1));
    const float denomY = float(max(1, params.height - 1));
    const float px = (float(gid.x) / denomX) * 2.0f;
    const float py = (float(gid.y) / denomY) * 2.0f;
    const uint pixelIndex = gid.y * uint(params.width) + gid.x;
    float3 accum = float3(0.0f);
    for (int zoneIndex = 0; zoneIndex < params.stackDepth; ++zoneIndex) {
        const int zoneX = zoneIndex % 3;
        const int zoneY = zoneIndex / 3;
        const float wx = max(0.0f, 1.0f - abs(px - float(zoneX)));
        const float wy = max(0.0f, 1.0f - abs(py - float(zoneY)));
        const float weight = wx * wy;
        if (weight <= 0.0f) {
            continue;
        }
        const uint srcIndex = uint(zoneIndex * params.planeStride) + pixelIndex;
        accum += srcStack[srcIndex].xyz * weight;
    }
    dst[pixelIndex] += float4(accum, 0.0f);
}

kernel void lensDiffAccumulateWeightedPlanesStackKernel(device const float* planeStack [[buffer(0)]],
                                                        device float* dst [[buffer(1)]],
                                                        constant ZonePlaneAccumulateParamsGpu& params [[buffer(2)]],
                                                        uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }
    const float denomX = float(max(1, params.width - 1));
    const float denomY = float(max(1, params.height - 1));
    const float px = (float(gid.x) / denomX) * 2.0f;
    const float py = (float(gid.y) / denomY) * 2.0f;
    const uint pixelIndex = gid.y * uint(params.width) + gid.x;
    float accum = 0.0f;
    for (int zoneIndex = 0; zoneIndex < params.zoneCount; ++zoneIndex) {
        const int zoneX = zoneIndex % 3;
        const int zoneY = zoneIndex / 3;
        const float wx = max(0.0f, 1.0f - abs(px - float(zoneX)));
        const float wy = max(0.0f, 1.0f - abs(py - float(zoneY)));
        const float weight = wx * wy;
        if (weight <= 0.0f) {
            continue;
        }
        const uint srcIndex =
            uint(zoneIndex * params.binCount * params.planeStride + params.binIndex * params.planeStride) + pixelIndex;
        accum += planeStack[srcIndex] * weight;
    }
    dst[pixelIndex] = max(accum, 0.0f);
}

kernel void lensDiffCreativeFringeKernel(device const float4* src [[buffer(0)]],
                                         device float4* dst [[buffer(1)]],
                                         device float4* preview [[buffer(2)]],
                                         constant FringeParamsGpu& params [[buffer(3)]],
                                         uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) {
        return;
    }
    const float cx = float(max(0, params.width - 1)) * 0.5f;
    const float cy = float(max(0, params.height - 1)) * 0.5f;
    const float2 delta = float2(float(gid.x) - cx, float(gid.y) - cy);
    const float lengthValue = length(delta);
    const float2 direction = lengthValue > 1e-6f ? delta / lengthValue : float2(0.0f);
    const float2 shift = direction * params.fringeAmount;
    const float4 red = sampleRgbaBilinear(src, params.width, params.height, float(gid.x) + shift.x, float(gid.y) + shift.y);
    const float4 green = sampleRgbaBilinear(src, params.width, params.height, float(gid.x), float(gid.y));
    const float4 blue = sampleRgbaBilinear(src, params.width, params.height, float(gid.x) - shift.x, float(gid.y) - shift.y);
    const uint index = gid.y * uint(params.width) + gid.x;
    const float4 sourcePixel = src[index];
    const float4 outPixel = float4(red.x, green.y, blue.z, sourcePixel.w);
    dst[index] = outPixel;
    preview[index] = float4(abs(outPixel.x - sourcePixel.x),
                            abs(outPixel.y - sourcePixel.y),
                            abs(outPixel.z - sourcePixel.z),
                            1.0f);
}

kernel void lensDiffPackGrayKernel(device const float* gray [[buffer(0)]],
                                   device float* dst [[buffer(1)]],
                                   constant OutputParamsGpu& params [[buffer(2)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    const int outWidth = params.renderX2 - params.renderX1;
    const int outHeight = params.renderY2 - params.renderY1;
    if (int(gid.x) >= outWidth || int(gid.y) >= outHeight) {
        return;
    }

    const int absoluteX = params.renderX1 + int(gid.x);
    const int absoluteY = params.renderY1 + int(gid.y);
    if (absoluteX < params.dstOriginX || absoluteY < params.dstOriginY ||
        absoluteX >= params.dstX2 || absoluteY >= params.dstY2) {
        return;
    }
    const int sx = absoluteX - params.srcOriginX;
    const int sy = absoluteY - params.srcOriginY;
    if (sx < 0 || sy < 0 || sx >= params.srcWidth || sy >= params.srcHeight) {
        return;
    }

    const float value = gray[uint(sy * params.srcWidth + sx)];
    const uint dstIndex = uint(absoluteY - params.dstOriginY) * params.dstRowFloats +
                          uint(absoluteX - params.dstOriginX) * 4u;
    dst[dstIndex + 0u] = value;
    dst[dstIndex + 1u] = value;
    dst[dstIndex + 2u] = value;
    dst[dstIndex + 3u] = 1.0f;
}

kernel void lensDiffPackRgbKernel(device const float4* src [[buffer(0)]],
                                  device const float4* linearSrc [[buffer(1)]],
                                  device float* dst [[buffer(2)]],
                                  constant OutputParamsGpu& params [[buffer(3)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    const int outWidth = params.renderX2 - params.renderX1;
    const int outHeight = params.renderY2 - params.renderY1;
    if (int(gid.x) >= outWidth || int(gid.y) >= outHeight) {
        return;
    }

    const int absoluteX = params.renderX1 + int(gid.x);
    const int absoluteY = params.renderY1 + int(gid.y);
    if (absoluteX < params.dstOriginX || absoluteY < params.dstOriginY ||
        absoluteX >= params.dstX2 || absoluteY >= params.dstY2) {
        return;
    }
    const int sx = absoluteX - params.srcOriginX;
    const int sy = absoluteY - params.srcOriginY;
    if (sx < 0 || sy < 0 || sx >= params.srcWidth || sy >= params.srcHeight) {
        return;
    }

    const uint srcIndex = uint(sy * params.srcWidth + sx);
    float4 pixel = src[srcIndex];
    if (params.encodeOutput != 0) {
        pixel.x = encodeChannel(pixel.x, params.inputTransfer);
        pixel.y = encodeChannel(pixel.y, params.inputTransfer);
        pixel.z = encodeChannel(pixel.z, params.inputTransfer);
    }
    pixel.w = params.writeAlphaFromLinear != 0 ? linearSrc[srcIndex].w : 1.0f;

    const uint dstIndex = uint(absoluteY - params.dstOriginY) * params.dstRowFloats +
                          uint(absoluteX - params.dstOriginX) * 4u;
    dst[dstIndex + 0u] = pixel.x;
    dst[dstIndex + 1u] = pixel.y;
    dst[dstIndex + 2u] = pixel.z;
    dst[dstIndex + 3u] = pixel.w;
}

kernel void lensDiffReduceLumaKernel(device const float4* src [[buffer(0)]],
                                     device float* partials [[buffer(1)]],
                                     constant ReductionParamsGpu& params [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint tgIndex [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[256];
    const uint total = uint(params.width * params.height);
    float value = 0.0f;
    if (gid < total) {
        value = safeLuma(src[gid].xyz);
    }
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (tid < offset) {
            scratch[tid] += scratch[tid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        partials[tgIndex] = scratch[0];
    }
}

kernel void lensDiffReduceFloatKernel(device const float* src [[buffer(0)]],
                                      device float* partials [[buffer(1)]],
                                      constant ScalarReduceParamsGpu& params [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint tgIndex [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[256];
    float value = 0.0f;
    if (int(gid) < params.count) {
        value = src[gid];
    }
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (tid < offset) {
            scratch[tid] += scratch[tid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        partials[tgIndex] = scratch[0];
    }
}

kernel void lensDiffComputeScalarScaleKernel(device const float* sum [[buffer(0)]],
                                             device float* outScale [[buffer(1)]],
                                             constant ScalarScaleParamsGpu& params [[buffer(2)]],
                                             uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) {
        return;
    }
    outScale[0] = sum[0] > params.epsilon ? (1.0f / sum[0]) : 1.0f;
}

kernel void lensDiffResampleRawPsfKernel(device const float* rawPsf [[buffer(0)]],
                                         device float* outKernel [[buffer(1)]],
                                         constant float& invScale [[buffer(2)]],
                                         constant ConvolutionParamsGpu& params [[buffer(3)]],
                                         uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.kernelSize || int(gid.y) >= params.kernelSize) {
        return;
    }
    const float rawCenter = (params.height & 1) == 0 ? float(params.height) * 0.5f
                                                      : float(params.height - 1) * 0.5f;
    const float kernelCenter = float(params.kernelSize - 1) * 0.5f;
    const float dx = float(gid.x) - kernelCenter;
    const float dy = float(gid.y) - kernelCenter;
    const float sx = rawCenter + dx * invScale;
    const float sy = rawCenter + dy * invScale;
    outKernel[uint(gid.y) * uint(params.kernelSize) + gid.x] =
        sampleSquareFiltered(rawPsf, params.height, sx, sy, invScale);
}

kernel void lensDiffRingSumCountKernel(device const float* kernelValues [[buffer(0)]],
                                       device float* ringSums [[buffer(1)]],
                                       device uint* ringCounts [[buffer(2)]],
                                       constant KernelShapeParamsGpu& params [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]]) {
    if (int(gid) > params.radiusMax) {
        return;
    }
    const float center = float(params.kernelSize - 1) * 0.5f;
    float sum = 0.0f;
    uint count = 0u;
    for (int y = 0; y < params.kernelSize; ++y) {
        for (int x = 0; x < params.kernelSize; ++x) {
            const float dx = float(x) - center;
            const float dy = float(y) - center;
            const int r = min(params.radiusMax, int(round(length(float2(dx, dy)))));
            if (r == int(gid)) {
                sum += kernelValues[uint(y * params.kernelSize + x)];
                count += 1u;
            }
        }
    }
    ringSums[gid] = sum;
    ringCounts[gid] = count;
}

kernel void lensDiffExpandMeanKernel(device const float* ringSums [[buffer(0)]],
                                     device const uint* ringCounts [[buffer(1)]],
                                     device float* meanKernel [[buffer(2)]],
                                     constant KernelShapeParamsGpu& params [[buffer(3)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.kernelSize || int(gid.y) >= params.kernelSize) {
        return;
    }
    const float center = float(params.kernelSize - 1) * 0.5f;
    const float dx = float(gid.x) - center;
    const float dy = float(gid.y) - center;
    const int r = min(params.radiusMax, int(round(length(float2(dx, dy)))));
    const uint count = max(ringCounts[uint(r)], 1u);
    meanKernel[uint(gid.y) * uint(params.kernelSize) + gid.x] = ringSums[uint(r)] / float(count);
}

kernel void lensDiffReshapeKernel(device const float* original [[buffer(0)]],
                                  device const float* meanKernel [[buffer(1)]],
                                  device float* shapedKernel [[buffer(2)]],
                                  constant float& gain [[buffer(3)]],
                                  constant KernelShapeParamsGpu& params [[buffer(4)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.kernelSize || int(gid.y) >= params.kernelSize) {
        return;
    }
    const uint index = uint(gid.y) * uint(params.kernelSize) + gid.x;
    const float meanValue = meanKernel[index];
    const float residual = original[index] - meanValue;
    shapedKernel[index] = max(0.0f, meanValue + residual * gain);
}

kernel void lensDiffPositiveResidualKernel(device const float* fullKernel [[buffer(0)]],
                                           device const float* meanKernel [[buffer(1)]],
                                           device float* residualKernel [[buffer(2)]],
                                           constant KernelShapeParamsGpu& params [[buffer(3)]],
                                           uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.kernelSize || int(gid.y) >= params.kernelSize) {
        return;
    }
    const uint index = uint(gid.y) * uint(params.kernelSize) + gid.x;
    residualKernel[index] = max(0.0f, fullKernel[index] - meanKernel[index]);
}

kernel void lensDiffRingEnergyPeakKernel(device const float* kernelValues [[buffer(0)]],
                                         device float* ringEnergy [[buffer(1)]],
                                         device float* ringPeak [[buffer(2)]],
                                         constant KernelShapeParamsGpu& params [[buffer(3)]],
                                         uint gid [[thread_position_in_grid]]) {
    if (int(gid) > params.radiusMax) {
        return;
    }
    const float center = float(params.kernelSize - 1) * 0.5f;
    float energy = 0.0f;
    float peak = 0.0f;
    for (int y = 0; y < params.kernelSize; ++y) {
        for (int x = 0; x < params.kernelSize; ++x) {
            const float dx = float(x) - center;
            const float dy = float(y) - center;
            const int r = min(params.radiusMax, int(ceil(length(float2(dx, dy)))));
            if (r == int(gid)) {
                const float value = kernelValues[uint(y * params.kernelSize + x)];
                energy += value;
                peak = max(peak, value);
            }
        }
    }
    ringEnergy[gid] = energy;
    ringPeak[gid] = peak;
}

kernel void lensDiffCropKernel(device const float* src [[buffer(0)]],
                               device float* dst [[buffer(1)]],
                               constant ConvolutionParamsGpu& params [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= params.kernelSize || int(gid.y) >= params.kernelSize) {
        return;
    }
    const int srcCenter = params.width / 2;
    const int sx = srcCenter - params.kernelRadius + int(gid.x);
    const int sy = srcCenter - params.kernelRadius + int(gid.y);
    dst[uint(gid.y) * uint(params.kernelSize) + gid.x] = src[uint(sy * params.width + sx)];
}
)METAL";

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,
                                         id<MTLLibrary> library,
                                         NSString* name,
                                         std::string* error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        if (error) {
            *error = "missing-metal-function:" + std::string(name.UTF8String ? name.UTF8String : "unknown");
        }
        return nil;
    }

    NSError* pipelineError = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&pipelineError];
    if (pipeline == nil && error) {
        *error = "failed-metal-pipeline:" + std::string(name.UTF8String ? name.UTF8String : "unknown") + ":" +
                 nsErrorString(pipelineError);
    }
    return pipeline;
}

PipelineBundle* ensurePipelines(id<MTLDevice> device, std::string* error) {
    std::lock_guard<std::mutex> lock(gPipelineMutex);
    if (gPipelines.device == device && gPipelines.prepare != nil) {
        return &gPipelines;
    }

    const auto libraryLoadStart = std::chrono::steady_clock::now();
    std::string libraryMode = "runtime-source";
    NSError* libraryError = nil;
    id<MTLLibrary> library = nil;
    const std::string bundledLibraryPath = lensDiffBundledMetalLibraryPath();
    if (!bundledLibraryPath.empty()) {
        libraryMode = "precompiled";
        NSString* libraryPath = [NSString stringWithUTF8String:bundledLibraryPath.c_str()];
        NSURL* libraryURL = [NSURL fileURLWithPath:libraryPath];
        library = [device newLibraryWithURL:libraryURL error:&libraryError];
    }
    if (library == nil) {
        if (!bundledLibraryPath.empty()) {
            libraryMode = "runtime-source-fallback";
        }
        libraryError = nil;
        NSString* source = [NSString stringWithUTF8String:kLensDiffMetalSource];
        library = [device newLibraryWithSource:source options:nil error:&libraryError];
    }
    const double libraryLoadMs =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::steady_clock::now() - libraryLoadStart).count();
    LogLensDiffTimingStage("metal-library-load", libraryLoadMs, "mode=" + libraryMode);
    if (library == nil) {
        if (error) {
            *error = "failed-metal-library:" + nsErrorString(libraryError);
        }
        return nullptr;
    }

    PipelineBundle next {};
    next.device = device;
    next.library = library;
    next.buildPupil = makePipeline(device, library, @"lensDiffBuildPupilKernel", error);
    next.buildPhase = makePipeline(device, library, @"lensDiffBuildPhaseKernel", error);
    next.embedComplexPupil = makePipeline(device, library, @"lensDiffEmbedComplexPupilKernel", error);
    next.bitReverseRows = makePipeline(device, library, @"lensDiffBitReverseRowsKernel", error);
    next.fftRowsStage = makePipeline(device, library, @"lensDiffFftRowsStageKernel", error);
    next.bitReverseColumns = makePipeline(device, library, @"lensDiffBitReverseColumnsKernel", error);
    next.fftColumnsStage = makePipeline(device, library, @"lensDiffFftColumnsStageKernel", error);
    next.extractShiftedIntensity = makePipeline(device, library, @"lensDiffExtractShiftedIntensityKernel", error);
    next.transposeComplex = makePipeline(device, library, @"lensDiffTransposeComplexKernel", error);
    next.transposeComplexStack = makePipeline(device, library, @"lensDiffTransposeComplexStackKernel", error);
    next.bitReverseBatched = makePipeline(device, library, @"lensDiffBitReverseBatchedKernel", error);
    next.fftBatchedStage = makePipeline(device, library, @"lensDiffFftBatchedStageKernel", error);
    next.scaleComplexBatched = makePipeline(device, library, @"lensDiffScaleComplexBatchedKernel", error);
    next.copyComplexBatched = makePipeline(device, library, @"lensDiffCopyComplexBatchedKernel", error);
    next.buildBluesteinChirp = makePipeline(device, library, @"lensDiffBuildBluesteinChirpKernel", error);
    next.buildBluesteinInput = makePipeline(device, library, @"lensDiffBuildBluesteinInputKernel", error);
    next.multiplyBluesteinSpectra = makePipeline(device, library, @"lensDiffMultiplyBluesteinSpectraKernel", error);
    next.extractBluesteinOutput = makePipeline(device, library, @"lensDiffExtractBluesteinOutputKernel", error);
    next.decodeSource = makePipeline(device, library, @"lensDiffDecodeSourceKernel", error);
    next.prepare = makePipeline(device, library, @"lensDiffPrepareKernel", error);
    next.prepareFromLinear = makePipeline(device, library, @"lensDiffPrepareFromLinearKernel", error);
    next.convolveRgb = makePipeline(device, library, @"lensDiffConvolveRgbKernel", error);
    next.convolveScalar = makePipeline(device, library, @"lensDiffConvolveScalarKernel", error);
    next.padPlaneToComplex = makePipeline(device, library, @"lensDiffPadPlaneToComplexKernel", error);
    next.padRgbChannelToComplex = makePipeline(device, library, @"lensDiffPadRgbChannelToComplexKernel", error);
    next.padRgbToComplexStack = makePipeline(device, library, @"lensDiffPadRgbToComplexStackKernel", error);
    next.scatterKernelToComplex = makePipeline(device, library, @"lensDiffScatterKernelToComplexKernel", error);
    next.multiplyComplex = makePipeline(device, library, @"lensDiffMultiplyComplexKernel", error);
    next.multiplyComplexBroadcast = makePipeline(device, library, @"lensDiffMultiplyComplexBroadcastKernel", error);
    next.multiplyComplexPairsStack = makePipeline(device, library, @"lensDiffMultiplyComplexPairsStackKernel", error);
    next.replicateComplexStack = makePipeline(device, library, @"lensDiffReplicateComplexStackKernel", error);
    next.extractRealPlane = makePipeline(device, library, @"lensDiffExtractRealPlaneKernel", error);
    next.packPlanesToRgba = makePipeline(device, library, @"lensDiffPackPlanesToRgbaKernel", error);
    next.packPlaneTripletsToRgbaStack = makePipeline(device, library, @"lensDiffPackPlaneTripletsToRgbaStackKernel", error);
    next.applyShoulder = makePipeline(device, library, @"lensDiffApplyShoulderKernel", error);
    next.applyShoulderStack = makePipeline(device, library, @"lensDiffApplyShoulderStackKernel", error);
    next.combine = makePipeline(device, library, @"lensDiffCombineKernel", error);
    next.combineStack = makePipeline(device, library, @"lensDiffCombineStackKernel", error);
    next.mapSpectral = makePipeline(device, library, @"lensDiffMapSpectralKernel", error);
    next.mapSpectralStack = makePipeline(device, library, @"lensDiffMapSpectralStackKernel", error);
    next.composite = makePipeline(device, library, @"lensDiffCompositeKernel", error);
    next.scaleRgb = makePipeline(device, library, @"lensDiffScaleRgbKernel", error);
    next.scaleScalar = makePipeline(device, library, @"lensDiffScaleScalarKernel", error);
    next.accumulateWeighted = makePipeline(device, library, @"lensDiffAccumulateWeightedKernel", error);
    next.accumulateWeightedRgbStack = makePipeline(device, library, @"lensDiffAccumulateWeightedRgbStackKernel", error);
    next.accumulateWeightedPlanesStack = makePipeline(device, library, @"lensDiffAccumulateWeightedPlanesStackKernel", error);
    next.creativeFringe = makePipeline(device, library, @"lensDiffCreativeFringeKernel", error);
    next.packGray = makePipeline(device, library, @"lensDiffPackGrayKernel", error);
    next.packRgb = makePipeline(device, library, @"lensDiffPackRgbKernel", error);
    next.resampleRgba = makePipeline(device, library, @"lensDiffResampleRgbaKernel", error);
    next.resampleGray = makePipeline(device, library, @"lensDiffResampleGrayKernel", error);
    next.reduceLuma = makePipeline(device, library, @"lensDiffReduceLumaKernel", error);
    next.reduceFloat = makePipeline(device, library, @"lensDiffReduceFloatKernel", error);
    next.computeScalarScale = makePipeline(device, library, @"lensDiffComputeScalarScaleKernel", error);
    next.computePreserveScale = makePipeline(device, library, @"lensDiffComputePreserveScaleKernel", error);
    next.scaleRgbDynamic = makePipeline(device, library, @"lensDiffScaleRgbDynamicKernel", error);
    next.resampleRawPsf = makePipeline(device, library, @"lensDiffResampleRawPsfKernel", error);
    next.ringSumCount = makePipeline(device, library, @"lensDiffRingSumCountKernel", error);
    next.expandMean = makePipeline(device, library, @"lensDiffExpandMeanKernel", error);
    next.reshapeKernel = makePipeline(device, library, @"lensDiffReshapeKernel", error);
    next.positiveResidual = makePipeline(device, library, @"lensDiffPositiveResidualKernel", error);
    next.ringEnergyPeak = makePipeline(device, library, @"lensDiffRingEnergyPeakKernel", error);
    next.cropKernel = makePipeline(device, library, @"lensDiffCropKernel", error);
    if (next.buildPupil == nil || next.buildPhase == nil || next.embedComplexPupil == nil ||
        next.bitReverseRows == nil || next.fftRowsStage == nil ||
        next.bitReverseColumns == nil || next.fftColumnsStage == nil ||
        next.extractShiftedIntensity == nil || next.transposeComplex == nil || next.transposeComplexStack == nil ||
        next.bitReverseBatched == nil || next.fftBatchedStage == nil || next.scaleComplexBatched == nil || next.copyComplexBatched == nil ||
        next.buildBluesteinChirp == nil || next.buildBluesteinInput == nil ||
        next.multiplyBluesteinSpectra == nil || next.extractBluesteinOutput == nil ||
        next.decodeSource == nil || next.prepare == nil || next.prepareFromLinear == nil ||
        next.convolveRgb == nil || next.convolveScalar == nil ||
        next.padPlaneToComplex == nil || next.padRgbChannelToComplex == nil || next.padRgbToComplexStack == nil ||
        next.scatterKernelToComplex == nil || next.multiplyComplex == nil || next.multiplyComplexBroadcast == nil ||
        next.multiplyComplexPairsStack == nil || next.replicateComplexStack == nil ||
        next.extractRealPlane == nil || next.packPlanesToRgba == nil || next.packPlaneTripletsToRgbaStack == nil ||
        next.applyShoulder == nil || next.applyShoulderStack == nil ||
        next.combine == nil || next.combineStack == nil ||
        next.mapSpectral == nil || next.mapSpectralStack == nil || next.composite == nil || next.scaleRgb == nil || next.scaleScalar == nil ||
        next.accumulateWeighted == nil || next.accumulateWeightedRgbStack == nil || next.accumulateWeightedPlanesStack == nil || next.creativeFringe == nil ||
        next.packGray == nil || next.packRgb == nil || next.resampleRgba == nil || next.resampleGray == nil ||
        next.reduceLuma == nil || next.reduceFloat == nil || next.computeScalarScale == nil || next.computePreserveScale == nil ||
        next.scaleRgbDynamic == nil || next.resampleRawPsf == nil ||
        next.ringSumCount == nil || next.expandMean == nil || next.reshapeKernel == nil ||
        next.positiveResidual == nil || next.ringEnergyPeak == nil || next.cropKernel == nil) {
        return nullptr;
    }

    gPipelines = next;
    return &gPipelines;
}

bool LensDiffMetalHeapsEnabled();

id<MTLBuffer> makeSharedBuffer(id<MTLDevice> device, NSUInteger length, std::string* error) {
    auto tryHeapBuffer = [&](NSUInteger byteCount) -> id<MTLBuffer> {
        if (device == nil || byteCount == 0) {
            return nil;
        }
        constexpr NSUInteger kHeapThresholdBytes = 1u << 20;
        constexpr NSUInteger kMinimumHeapBytes = 64u << 20;
        if (byteCount < kHeapThresholdBytes) {
            return nil;
        }
        if (!LensDiffMetalHeapsEnabled()) {
            return nil;
        }
        if (![device respondsToSelector:@selector(newHeapWithDescriptor:)]) {
            return nil;
        }

        const MTLResourceOptions options = MTLResourceStorageModeShared;
        const MTLSizeAndAlign sizeAndAlign = [device heapBufferSizeAndAlignWithLength:byteCount options:options];
        if (sizeAndAlign.size == 0) {
            return nil;
        }
        auto alignUp = [](NSUInteger value, NSUInteger alignment) {
            if (alignment == 0) {
                return value;
            }
            const NSUInteger mask = alignment - 1u;
            return (value + mask) & ~mask;
        };
        const NSUInteger allocationSize =
            alignUp(sizeAndAlign.size, std::max<NSUInteger>(sizeAndAlign.align, static_cast<NSUInteger>(4096u)));
        const std::uintptr_t deviceKey = reinterpret_cast<std::uintptr_t>(device);

        std::lock_guard<std::mutex> lock(gHeapMutex);
        auto& heaps = gHeapPools[deviceKey];
        for (const MetalHeapRecord& record : heaps) {
            if (record.heap == nil) {
                continue;
            }
            id<MTLBuffer> heapBuffer = [record.heap newBufferWithLength:byteCount options:options];
            if (heapBuffer != nil) {
                return heapBuffer;
            }
        }

        const NSUInteger heapSize = std::max(kMinimumHeapBytes, allocationSize * static_cast<NSUInteger>(4u));
        MTLHeapDescriptor* descriptor = [[MTLHeapDescriptor alloc] init];
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
        descriptor.size = heapSize;
        id<MTLHeap> heap = [device newHeapWithDescriptor:descriptor];
        if (heap == nil) {
            return nil;
        }
        heaps.push_back(MetalHeapRecord {heap, heapSize});
        return [heap newBufferWithLength:byteCount options:options];
    };

    id<MTLBuffer> buffer = tryHeapBuffer(length);
    if (buffer == nil) {
        buffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
    }
    if (buffer == nil && error) {
        *error = "failed-metal-buffer-allocation";
    }
    return buffer;
}

id<MTLBuffer> makeSharedBufferWithBytes(id<MTLDevice> device,
                                        const void* bytes,
                                        NSUInteger length,
                                        std::string* error) {
    id<MTLBuffer> buffer = makeSharedBuffer(device, length, error);
    if (buffer == nil) {
        if (error != nullptr && error->empty()) {
            *error = "failed-metal-buffer-upload";
        }
        return nil;
    }
    if (bytes != nullptr && length > 0) {
        std::memcpy(buffer.contents, bytes, length);
    }
    return buffer;
}

template <typename ParamsT>
id<MTLBuffer> makeParamBuffer(id<MTLDevice> device, const ParamsT& params, std::string* error) {
    return makeSharedBufferWithBytes(device, &params, sizeof(ParamsT), error);
}

MTLSize threadgroup2d(id<MTLComputePipelineState> pipeline) {
    const NSUInteger width = std::max<NSUInteger>(1, std::min<NSUInteger>(16, pipeline.threadExecutionWidth));
    const NSUInteger height = std::max<NSUInteger>(1, std::min<NSUInteger>(16, pipeline.maxTotalThreadsPerThreadgroup / width));
    return MTLSizeMake(width, height, 1);
}

void dispatch2d(id<MTLComputeCommandEncoder> encoder,
                id<MTLComputePipelineState> pipeline,
                int width,
                int height) {
    const MTLSize threads = threadgroup2d(pipeline);
    [encoder setComputePipelineState:pipeline];
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(std::max(0, width)),
                                         static_cast<NSUInteger>(std::max(0, height)),
                                         1)
       threadsPerThreadgroup:threads];
}

void dispatch1d256(id<MTLComputeCommandEncoder> encoder,
                   id<MTLComputePipelineState> pipeline,
                   NSUInteger count) {
    [encoder setComputePipelineState:pipeline];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

bool commitAndWait(id<MTLCommandBuffer> commandBuffer, std::string* error) {
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.error != nil) {
        if (error) {
            *error = "metal-command-buffer-failed:" + nsErrorString(commandBuffer.error);
        }
        std::ostringstream note;
        note << "status=" << static_cast<int>(commandBuffer.status)
             << " error=" << nsErrorString(commandBuffer.error);
        LogLensDiffDiagnosticEvent("metal-command-buffer-failed", note.str());
        return false;
    }
    return true;
}

bool lensDiffMetalEnvFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" && text != "OFF";
}

thread_local int gLensDiffMetalHeapsOverride = -1;
thread_local int gLensDiffMetalVkFFTOverride = -1;

bool LensDiffMetalLegacySyncEnabled() {
    return lensDiffMetalEnvFlagEnabled("LENSDIFF_METAL_LEGACY_SYNC");
}

bool LensDiffMetalFastFieldEnabled() {
    return lensDiffMetalEnvFlagEnabled("LENSDIFF_METAL_FAST_FIELD");
}

bool LensDiffMetalFastSplitEnabled() {
    return lensDiffMetalEnvFlagEnabled("LENSDIFF_METAL_FAST_SPLIT");
}

bool LensDiffMetalFastResolutionAwareEnabled() {
    return !lensDiffMetalEnvFlagEnabled("LENSDIFF_METAL_DISABLE_FAST_RESOLUTION_AWARE");
}

bool LensDiffMetalHeapsRequested() {
    return !lensDiffMetalEnvFlagEnabled("LENSDIFF_METAL_DISABLE_HEAPS");
}

bool LensDiffMetalHeapsForceEnabled() {
    return lensDiffMetalEnvFlagEnabled("LENSDIFF_METAL_FORCE_HEAPS");
}

bool LensDiffMetalHeapsEnabled() {
    if (gLensDiffMetalHeapsOverride >= 0) {
        return gLensDiffMetalHeapsOverride != 0;
    }
    return LensDiffMetalHeapsRequested();
}

bool LensDiffMetalVkFFTRequested() {
    return !lensDiffMetalEnvFlagEnabled("LENSDIFF_METAL_DISABLE_VKFFT");
}

bool LensDiffMetalVkFFTEnabled() {
    if (gLensDiffMetalVkFFTOverride >= 0) {
        return gLensDiffMetalVkFFTOverride != 0;
    }
    return LensDiffMetalVkFFTRequested();
}

struct LensDiffMetalRuntimeOverrideScope {
    int previousHeaps = -1;
    int previousVkFFT = -1;

    LensDiffMetalRuntimeOverrideScope(bool heapsEnabled, bool vkfftEnabled)
        : previousHeaps(gLensDiffMetalHeapsOverride)
        , previousVkFFT(gLensDiffMetalVkFFTOverride) {
        gLensDiffMetalHeapsOverride = heapsEnabled ? 1 : 0;
        gLensDiffMetalVkFFTOverride = vkfftEnabled ? 1 : 0;
    }

    ~LensDiffMetalRuntimeOverrideScope() {
        gLensDiffMetalHeapsOverride = previousHeaps;
        gLensDiffMetalVkFFTOverride = previousVkFFT;
    }
};

struct MetalRenderTimingCounters {
    int commandBufferCount = 0;
    int waitCount = 0;
    int fieldZoneBatchDepth = 0;
};

enum class MetalScratchFamily : int {
    FftScratch = 0,
    FftTranspose = 1,
    TempSpectrum = 2,
    KernelSpectrumStack = 3,
};

struct MetalScratchKey {
    int paddedSize = 0;
    int stackDepth = 0;
    int family = 0;

    bool operator==(const MetalScratchKey& other) const {
        return paddedSize == other.paddedSize &&
               stackDepth == other.stackDepth &&
               family == other.family;
    }
};

struct MetalScratchKeyHasher {
    std::size_t operator()(const MetalScratchKey& key) const noexcept {
        std::size_t hash = static_cast<std::size_t>(key.paddedSize);
        hash = hash * 1315423911u + static_cast<std::size_t>(key.stackDepth);
        hash = hash * 2654435761u + static_cast<std::size_t>(key.family);
        return hash;
    }
};

struct MetalScratchCache {
    std::unordered_map<MetalScratchKey, id<MTLBuffer>, MetalScratchKeyHasher> buffers;

    id<MTLBuffer> acquire(id<MTLDevice> device,
                          MetalScratchFamily family,
                          int paddedSize,
                          int stackDepth,
                          NSUInteger byteCount,
                          std::string* error) {
        const MetalScratchKey key {paddedSize, stackDepth, static_cast<int>(family)};
        auto it = buffers.find(key);
        if (it != buffers.end() && it->second != nil && it->second.length >= byteCount) {
            return it->second;
        }
        id<MTLBuffer> buffer = makeSharedBuffer(device, byteCount, error);
        if (buffer == nil) {
            return nil;
        }
        buffers[key] = buffer;
        return buffer;
    }
};

struct MetalRenderContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    PipelineBundle* pipelines = nullptr;
    MetalScratchCache* scratchCache = nullptr;
    MetalRenderTimingCounters* counters = nullptr;
    std::string* error = nullptr;
    id<MTLCommandBuffer> commandBuffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
};

id<MTLBuffer> makeSharedBuffer(id<MTLDevice> device, NSUInteger length, std::string* error);
template <typename ParamsT>
id<MTLBuffer> makeParamBuffer(id<MTLDevice> device, const ParamsT& params, std::string* error);

enum class PsfBufferSlot : int {
    BaseKernel = 0,
    MeanKernel = 1,
    ShapedKernel = 2,
    StructureKernel = 3,
    RingSums = 4,
    RingCounts = 5,
    RingEnergy = 6,
    RingPeak = 7,
    CropCore = 8,
    CropFull = 9,
    CropStructure = 10,
    ReductionA = 11,
    ReductionB = 12,
    ScalarScale = 13,
};

struct PsfBufferKey {
    int slot = 0;
    NSUInteger byteCount = 0;

    bool operator==(const PsfBufferKey& other) const {
        return slot == other.slot && byteCount == other.byteCount;
    }
};

struct PsfBufferKeyHasher {
    std::size_t operator()(const PsfBufferKey& key) const noexcept {
        return (static_cast<std::size_t>(key.slot) * 2654435761u) ^ static_cast<std::size_t>(key.byteCount);
    }
};

struct MetalPsfBuildContext {
    id<MTLDevice> device = nil;
    std::unordered_map<PsfBufferKey, id<MTLBuffer>, PsfBufferKeyHasher> buffers;

    id<MTLBuffer> acquire(PsfBufferSlot slot, NSUInteger byteCount, std::string* error) {
        const PsfBufferKey key {static_cast<int>(slot), byteCount};
        auto it = buffers.find(key);
        if (it != buffers.end() && it->second != nil && it->second.length >= byteCount) {
            return it->second;
        }
        id<MTLBuffer> buffer = makeSharedBuffer(device, byteCount, error);
        if (buffer == nil) {
            return nil;
        }
        buffers[key] = buffer;
        return buffer;
    }
};

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

struct FieldZoneKernelStacks {
    int paddedSize = 0;
    int zoneCount = 0;
    int spectralBinCount = 0;
    FieldEffectKind effectKind = FieldEffectKind::Full;
    id<MTLBuffer> spectrumStack = nil;
};

bool commitAndWaitCounted(MetalRenderTimingCounters* counters,
                          id<MTLCommandBuffer> commandBuffer,
                          std::string* error) {
    if (counters != nullptr) {
        ++counters->waitCount;
    }
    return commitAndWait(commandBuffer, error);
}

bool beginMetalStage(MetalRenderContext* context) {
    if (context == nullptr || context->queue == nil) {
        if (context != nullptr && context->error != nullptr) {
            *context->error = "metal-invalid-stage-context";
        }
        return false;
    }
    if (context->commandBuffer != nil || context->encoder != nil) {
        if (context->error != nullptr) {
            *context->error = "metal-stage-already-open";
        }
        return false;
    }
    context->commandBuffer = [context->queue commandBuffer];
    context->encoder = [context->commandBuffer computeCommandEncoder];
    if (context->commandBuffer == nil || context->encoder == nil) {
        if (context->error != nullptr) {
            *context->error = "metal-stage-create-failed";
        }
        context->commandBuffer = nil;
        context->encoder = nil;
        return false;
    }
    if (context->counters != nullptr) {
        ++context->counters->commandBufferCount;
    }
    return true;
}

bool endMetalStage(MetalRenderContext* context) {
    if (context == nullptr || context->commandBuffer == nil || context->encoder == nil) {
        if (context != nullptr && context->error != nullptr) {
            *context->error = "metal-stage-not-open";
        }
        return false;
    }
    [context->encoder endEncoding];
    id<MTLCommandBuffer> commandBuffer = context->commandBuffer;
    context->encoder = nil;
    context->commandBuffer = nil;
    return commitAndWaitCounted(context->counters, commandBuffer, context->error);
}

bool encodeReduceFloatToScalar(id<MTLComputeCommandEncoder> encoder,
                               id<MTLDevice> device,
                               PipelineBundle* pipelines,
                               MetalPsfBuildContext* psfContext,
                               id<MTLBuffer> srcBuffer,
                               NSUInteger count,
                               id<MTLBuffer>* outScalar,
                               std::string* error) {
    if (encoder == nil || device == nil || pipelines == nullptr || psfContext == nullptr ||
        srcBuffer == nil || outScalar == nullptr || count == 0) {
        if (error) *error = "metal-invalid-reduce-float-to-scalar";
        return false;
    }
    const NSUInteger groups = ceilDiv(count, static_cast<NSUInteger>(256));
    id<MTLBuffer> current = psfContext->acquire(PsfBufferSlot::ReductionA, groups * sizeof(float), error);
    const ScalarReduceParamsGpu paramsGpu {static_cast<int>(count)};
    id<MTLBuffer> paramsBuffer = makeParamBuffer(device, paramsGpu, error);
    if (current == nil || paramsBuffer == nil) {
        return false;
    }
    [encoder setBuffer:srcBuffer offset:0 atIndex:0];
    [encoder setBuffer:current offset:0 atIndex:1];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
    dispatch1d256(encoder, pipelines->reduceFloat, count);

    NSUInteger currentCount = groups;
    bool useA = false;
    while (currentCount > 1u) {
        const NSUInteger nextCount = ceilDiv(currentCount, static_cast<NSUInteger>(256));
        id<MTLBuffer> next = psfContext->acquire(useA ? PsfBufferSlot::ReductionA : PsfBufferSlot::ReductionB,
                                                 nextCount * sizeof(float),
                                                 error);
        const ScalarReduceParamsGpu reduceParams {static_cast<int>(currentCount)};
        id<MTLBuffer> reduceParamsBuffer = makeParamBuffer(device, reduceParams, error);
        if (next == nil || reduceParamsBuffer == nil) {
            return false;
        }
        [encoder setBuffer:current offset:0 atIndex:0];
        [encoder setBuffer:next offset:0 atIndex:1];
        [encoder setBuffer:reduceParamsBuffer offset:0 atIndex:2];
        dispatch1d256(encoder, pipelines->reduceFloat, currentCount);
        current = next;
        currentCount = nextCount;
        useA = !useA;
    }
    *outScalar = current;
    return true;
}

bool encodeNormalizeScalarBufferMetal(id<MTLComputeCommandEncoder> encoder,
                                      id<MTLDevice> device,
                                      PipelineBundle* pipelines,
                                      MetalPsfBuildContext* psfContext,
                                      id<MTLBuffer> buffer,
                                      NSUInteger count,
                                      std::string* error) {
    if (encoder == nil || device == nil || pipelines == nullptr || psfContext == nullptr || buffer == nil || count == 0) {
        if (error) *error = "metal-invalid-normalize-scalar";
        return false;
    }
    id<MTLBuffer> sumBuffer = nil;
    if (!encodeReduceFloatToScalar(encoder, device, pipelines, psfContext, buffer, count, &sumBuffer, error)) {
        return false;
    }
    id<MTLBuffer> scaleBuffer = psfContext->acquire(PsfBufferSlot::ScalarScale, sizeof(float), error);
    const ScalarScaleParamsGpu scaleParams {static_cast<int>(count), 1e-6f};
    id<MTLBuffer> scaleParamsBuffer = makeParamBuffer(device, scaleParams, error);
    if (scaleBuffer == nil || scaleParamsBuffer == nil) {
        return false;
    }
    [encoder setBuffer:sumBuffer offset:0 atIndex:0];
    [encoder setBuffer:scaleBuffer offset:0 atIndex:1];
    [encoder setBuffer:scaleParamsBuffer offset:0 atIndex:2];
    dispatch1d256(encoder, pipelines->computeScalarScale, 1);
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBuffer:scaleBuffer offset:0 atIndex:1];
    [encoder setBuffer:scaleParamsBuffer offset:0 atIndex:2];
    dispatch1d256(encoder, pipelines->scaleScalar, count);
    return true;
}

bool encodeReplicateComplexStackMetal(MetalRenderContext* context,
                                      id<MTLBuffer> srcSpectrum,
                                      int srcBatchCount,
                                      int dstBatchCount,
                                      int paddedSize,
                                      id<MTLBuffer>* outSpectrumStack) {
    if (context == nullptr || context->encoder == nil || srcSpectrum == nil || outSpectrumStack == nullptr ||
        srcBatchCount <= 0 || dstBatchCount <= 0 || paddedSize <= 0) {
        if (context != nullptr && context->error != nullptr) {
            *context->error = "metal-invalid-replicate-complex-stack";
        }
        return false;
    }
    const NSUInteger paddedCount = static_cast<NSUInteger>(paddedSize) * static_cast<NSUInteger>(paddedSize);
    const NSUInteger sliceBytes = paddedCount * sizeof(float) * 2u;
    const NSUInteger stackBytes = sliceBytes * static_cast<NSUInteger>(dstBatchCount);
    id<MTLBuffer> dstSpectrum = makeSharedBuffer(context->device, stackBytes, context->error);
    const ReplicateComplexParamsGpu params {static_cast<int>(paddedCount), srcBatchCount, dstBatchCount,
                                            static_cast<int>(paddedCount), static_cast<int>(paddedCount)};
    id<MTLBuffer> paramsBuffer = makeParamBuffer(context->device, params, context->error);
    if (dstSpectrum == nil || paramsBuffer == nil) {
        return false;
    }
    [context->encoder setBuffer:srcSpectrum offset:0 atIndex:0];
    [context->encoder setBuffer:dstSpectrum offset:0 atIndex:1];
    [context->encoder setBuffer:paramsBuffer offset:0 atIndex:2];
    dispatch2d(context->encoder, context->pipelines->replicateComplexStack, static_cast<int>(paddedCount), dstBatchCount);
    *outSpectrumStack = dstSpectrum;
    return true;
}

bool encodeBuildKernelSpectrumStackMetal(MetalRenderContext* context,
                                         const std::vector<const LensDiffKernel*>& kernels,
                                         int repeatPerKernel,
                                         int paddedSize,
                                         id<MTLBuffer>* outSpectrumStack) {
    if (context == nullptr || context->encoder == nil || outSpectrumStack == nullptr ||
        kernels.empty() || repeatPerKernel <= 0 || paddedSize <= 0) {
        if (context != nullptr && context->error != nullptr) {
            *context->error = "metal-invalid-build-kernel-spectrum-stack";
        }
        return false;
    }
    const NSUInteger paddedCount = static_cast<NSUInteger>(paddedSize) * static_cast<NSUInteger>(paddedSize);
    const int sliceCount = static_cast<int>(kernels.size()) * repeatPerKernel;
    const NSUInteger sliceBytes = paddedCount * sizeof(float) * 2u;
    id<MTLBuffer> spectrumStack = makeSharedBuffer(context->device,
                                                   sliceBytes * static_cast<NSUInteger>(sliceCount),
                                                   context->error);
    id<MTLBuffer> scratch = context->scratchCache->acquire(context->device,
                                                           MetalScratchFamily::FftScratch,
                                                           paddedSize,
                                                           sliceCount,
                                                           sliceBytes * static_cast<NSUInteger>(sliceCount),
                                                           context->error);
    id<MTLBuffer> transpose = context->scratchCache->acquire(context->device,
                                                             MetalScratchFamily::FftTranspose,
                                                             paddedSize,
                                                             sliceCount,
                                                             sliceBytes * static_cast<NSUInteger>(sliceCount),
                                                             context->error);
    if (spectrumStack == nil || scratch == nil || transpose == nil) {
        return false;
    }
    std::memset(spectrumStack.contents, 0, spectrumStack.length);
    auto* values = static_cast<float*>(spectrumStack.contents);
    for (std::size_t kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex) {
        const LensDiffKernel* kernel = kernels[kernelIndex];
        if (kernel == nullptr || kernel->size <= 0 || kernel->values.empty()) {
            if (context->error) *context->error = "metal-null-kernel-in-stack";
            return false;
        }
        const int radius = kernel->size / 2;
        for (int repeat = 0; repeat < repeatPerKernel; ++repeat) {
            const int sliceIndex = static_cast<int>(kernelIndex) * repeatPerKernel + repeat;
            const NSUInteger sliceOffset = static_cast<NSUInteger>(sliceIndex) * paddedCount * 2u;
            for (int y = 0; y < kernel->size; ++y) {
                for (int x = 0; x < kernel->size; ++x) {
                    const int dx = (x - radius + paddedSize) % paddedSize;
                    const int dy = (y - radius + paddedSize) % paddedSize;
                    const NSUInteger complexIndex = sliceOffset + (static_cast<NSUInteger>(dy) * paddedSize + static_cast<NSUInteger>(dx)) * 2u;
                    values[complexIndex] = kernel->values[static_cast<std::size_t>(y) * kernel->size + static_cast<std::size_t>(x)];
                    values[complexIndex + 1u] = 0.0f;
                }
            }
        }
    }
    if (!encodeSquareForwardFftStack(context->commandBuffer, context->encoder, context->pipelines, spectrumStack, scratch, transpose, paddedSize, sliceCount, context->error)) {
        return false;
    }
    *outSpectrumStack = spectrumStack;
    return true;
}

bool encodeConvolvePairwiseStackToPlaneStackMetal(MetalRenderContext* context,
                                                  id<MTLBuffer> sourceSpectrumStack,
                                                  id<MTLBuffer> kernelSpectrumStack,
                                                  int width,
                                                  int height,
                                                  int paddedSize,
                                                  int batchCount,
                                                  id<MTLBuffer>* outPlaneStack) {
    if (context == nullptr || context->encoder == nil || sourceSpectrumStack == nil || kernelSpectrumStack == nil ||
        outPlaneStack == nullptr || batchCount <= 0 || paddedSize <= 0) {
        if (context != nullptr && context->error != nullptr) {
            *context->error = "metal-invalid-pairwise-stack-convolution";
        }
        return false;
    }
    const NSUInteger paddedCount = static_cast<NSUInteger>(paddedSize) * static_cast<NSUInteger>(paddedSize);
    const NSUInteger sliceBytes = paddedCount * sizeof(float) * 2u;
    const NSUInteger stackBytes = sliceBytes * static_cast<NSUInteger>(batchCount);
    const NSUInteger planeCount = static_cast<NSUInteger>(width) * static_cast<NSUInteger>(height);
    const NSUInteger planeStackBytes = planeCount * sizeof(float) * static_cast<NSUInteger>(batchCount);
    id<MTLBuffer> tempSpectrum = context->scratchCache->acquire(context->device,
                                                                MetalScratchFamily::TempSpectrum,
                                                                paddedSize,
                                                                batchCount,
                                                                stackBytes,
                                                                context->error);
    id<MTLBuffer> scratch = context->scratchCache->acquire(context->device,
                                                           MetalScratchFamily::FftScratch,
                                                           paddedSize,
                                                           batchCount,
                                                           stackBytes,
                                                           context->error);
    id<MTLBuffer> transpose = context->scratchCache->acquire(context->device,
                                                             MetalScratchFamily::FftTranspose,
                                                             paddedSize,
                                                             batchCount,
                                                             stackBytes,
                                                             context->error);
    id<MTLBuffer> planeStack = makeSharedBuffer(context->device, planeStackBytes, context->error);
    const BatchFftParamsGpu batchParams {static_cast<int>(paddedCount), 0, static_cast<int>(paddedCount), batchCount};
    id<MTLBuffer> batchParamsBuffer = makeParamBuffer(context->device, batchParams, context->error);
    const FftImageParamsGpu planeParams {width, height, paddedSize, paddedSize * paddedSize, 1, 0, 0};
    id<MTLBuffer> planeParamsBuffer = makeParamBuffer(context->device, planeParams, context->error);
    if (tempSpectrum == nil || scratch == nil || transpose == nil || planeStack == nil ||
        batchParamsBuffer == nil || planeParamsBuffer == nil) {
        return false;
    }
    [context->encoder setBuffer:sourceSpectrumStack offset:0 atIndex:0];
    [context->encoder setBuffer:kernelSpectrumStack offset:0 atIndex:1];
    [context->encoder setBuffer:tempSpectrum offset:0 atIndex:2];
    [context->encoder setBuffer:batchParamsBuffer offset:0 atIndex:3];
    dispatch2d(context->encoder, context->pipelines->multiplyComplexPairsStack, static_cast<int>(paddedCount), batchCount);
    if (!encodeSquareInverseFftStack(context->commandBuffer, context->encoder, context->pipelines, tempSpectrum, scratch, transpose, paddedSize, batchCount, context->error)) {
        return false;
    }
    for (int sliceIndex = 0; sliceIndex < batchCount; ++sliceIndex) {
        [context->encoder setBuffer:tempSpectrum offset:sliceBytes * static_cast<NSUInteger>(sliceIndex) atIndex:0];
        [context->encoder setBuffer:planeStack offset:planeCount * sizeof(float) * static_cast<NSUInteger>(sliceIndex) atIndex:1];
        [context->encoder setBuffer:planeParamsBuffer offset:0 atIndex:2];
        dispatch2d(context->encoder, context->pipelines->extractRealPlane, width, height);
    }
    *outPlaneStack = planeStack;
    return true;
}

bool encodePackPlaneTripletsToRgbaStackMetal(MetalRenderContext* context,
                                             id<MTLBuffer> planeStack,
                                             int width,
                                             int height,
                                             int stackDepth,
                                             id<MTLBuffer>* outImageStack) {
    if (context == nullptr || context->encoder == nil || planeStack == nil || outImageStack == nullptr ||
        width <= 0 || height <= 0 || stackDepth <= 0) {
        if (context != nullptr && context->error != nullptr) {
            *context->error = "metal-invalid-pack-plane-triplets";
        }
        return false;
    }
    const NSUInteger planeCount = static_cast<NSUInteger>(width) * static_cast<NSUInteger>(height);
    id<MTLBuffer> imageStack = makeSharedBuffer(context->device,
                                                planeCount * static_cast<NSUInteger>(stackDepth) * 4u * sizeof(float),
                                                context->error);
    const StackImageParamsGpu params {width, height, stackDepth, static_cast<int>(planeCount)};
    id<MTLBuffer> paramsBuffer = makeParamBuffer(context->device, params, context->error);
    if (imageStack == nil || paramsBuffer == nil) {
        return false;
    }
    [context->encoder setBuffer:planeStack offset:0 atIndex:0];
    [context->encoder setBuffer:imageStack offset:0 atIndex:1];
    [context->encoder setBuffer:paramsBuffer offset:0 atIndex:2];
    dispatch2d(context->encoder, context->pipelines->packPlaneTripletsToRgbaStack, width, height * stackDepth);
    *outImageStack = imageStack;
    return true;
}

FieldZoneBatchPlan buildFieldZoneBatchPlan(const LensDiffPsfBankCache& cache) {
    FieldZoneBatchPlan plan {};
    plan.fieldKey = cache.fieldKey;
    plan.zones.reserve(cache.fieldZones.size());
    for (const auto& zone : cache.fieldZones) {
        plan.zones.push_back(&zone);
    }
    std::sort(plan.zones.begin(),
              plan.zones.end(),
              [](const LensDiffFieldZoneCache* a, const LensDiffFieldZoneCache* b) {
                  if (a->zoneY != b->zoneY) return a->zoneY < b->zoneY;
                  return a->zoneX < b->zoneX;
              });
    plan.canonical3x3 = plan.zones.size() == 9u;
    if (plan.canonical3x3) {
        for (std::size_t i = 0; i < plan.zones.size(); ++i) {
            const int expectedX = static_cast<int>(i % 3u);
            const int expectedY = static_cast<int>(i / 3u);
            if (plan.zones[i]->zoneX != expectedX || plan.zones[i]->zoneY != expectedY) {
                plan.canonical3x3 = false;
                break;
            }
        }
    }
    return plan;
}

bool encodeScalarSpectrumMetal(MetalRenderContext* context,
                               id<MTLBuffer> srcPlane,
                               int width,
                               int height,
                               int paddedSize,
                               id<MTLBuffer>* outSpectrum) {
    if (context == nullptr || context->encoder == nil || outSpectrum == nullptr) {
        if (context != nullptr && context->error != nullptr) {
            *context->error = "metal-invalid-encode-scalar-spectrum";
        }
        return false;
    }
    const NSUInteger paddedCount = static_cast<NSUInteger>(paddedSize) * static_cast<NSUInteger>(paddedSize);
    id<MTLBuffer> spectrum = makeSharedBuffer(context->device, paddedCount * sizeof(float) * 2u, context->error);
    id<MTLBuffer> scratch = context->scratchCache->acquire(context->device,
                                                           MetalScratchFamily::FftScratch,
                                                           paddedSize,
                                                           1,
                                                           paddedCount * sizeof(float) * 2u,
                                                           context->error);
    id<MTLBuffer> transpose = context->scratchCache->acquire(context->device,
                                                             MetalScratchFamily::FftTranspose,
                                                             paddedSize,
                                                             1,
                                                             paddedCount * sizeof(float) * 2u,
                                                             context->error);
    const FftImageParamsGpu params {width, height, paddedSize, paddedSize * paddedSize, 1, 0, 0};
    id<MTLBuffer> paramsBuffer = makeParamBuffer(context->device, params, context->error);
    if (spectrum == nil || scratch == nil || transpose == nil || paramsBuffer == nil) {
        return false;
    }

    [context->encoder setBuffer:srcPlane offset:0 atIndex:0];
    [context->encoder setBuffer:spectrum offset:0 atIndex:1];
    [context->encoder setBuffer:paramsBuffer offset:0 atIndex:2];
    dispatch2d(context->encoder, context->pipelines->padPlaneToComplex, paddedSize, paddedSize);
    if (!encodeSquareForwardFft(context->commandBuffer, context->encoder, context->pipelines, spectrum, scratch, transpose, paddedSize, context->error)) {
        return false;
    }
    *outSpectrum = spectrum;
    return true;
}

bool encodeRgbSpectraStackMetal(MetalRenderContext* context,
                                id<MTLBuffer> srcImage,
                                int width,
                                int height,
                                int paddedSize,
                                id<MTLBuffer>* outSpectrum) {
    if (context == nullptr || context->encoder == nil || outSpectrum == nullptr) {
        if (context != nullptr && context->error != nullptr) {
            *context->error = "metal-invalid-encode-rgb-spectrum-stack";
        }
        return false;
    }
    const NSUInteger paddedCount = static_cast<NSUInteger>(paddedSize) * static_cast<NSUInteger>(paddedSize);
    const NSUInteger stackBytes = paddedCount * 3u * sizeof(float) * 2u;
    id<MTLBuffer> spectrum = makeSharedBuffer(context->device, stackBytes, context->error);
    id<MTLBuffer> scratch = context->scratchCache->acquire(context->device,
                                                           MetalScratchFamily::FftScratch,
                                                           paddedSize,
                                                           3,
                                                           stackBytes,
                                                           context->error);
    id<MTLBuffer> transpose = context->scratchCache->acquire(context->device,
                                                             MetalScratchFamily::FftTranspose,
                                                             paddedSize,
                                                             3,
                                                             stackBytes,
                                                             context->error);
    const FftImageParamsGpu params {width, height, paddedSize, paddedSize * paddedSize, 3, 0, 0};
    id<MTLBuffer> paramsBuffer = makeParamBuffer(context->device, params, context->error);
    if (spectrum == nil || scratch == nil || transpose == nil || paramsBuffer == nil) {
        return false;
    }

    [context->encoder setBuffer:srcImage offset:0 atIndex:0];
    [context->encoder setBuffer:spectrum offset:0 atIndex:1];
    [context->encoder setBuffer:paramsBuffer offset:0 atIndex:2];
    dispatch2d(context->encoder, context->pipelines->padRgbToComplexStack, paddedSize, paddedSize * 3);
    if (!encodeSquareForwardFftStack(context->commandBuffer, context->encoder, context->pipelines, spectrum, scratch, transpose, paddedSize, 3, context->error)) {
        return false;
    }
    *outSpectrum = spectrum;
    return true;
}

bool encodeKernelSpectrumMetal(MetalRenderContext* context,
                               const LensDiffKernel& kernel,
                               int paddedSize,
                               id<MTLBuffer>* outSpectrum) {
    if (context == nullptr || context->encoder == nil || outSpectrum == nullptr) {
        if (context != nullptr && context->error != nullptr) {
            *context->error = "metal-invalid-encode-kernel-spectrum";
        }
        return false;
    }
    const NSUInteger paddedCount = static_cast<NSUInteger>(paddedSize) * static_cast<NSUInteger>(paddedSize);
    id<MTLBuffer> kernelValues = makeSharedBufferWithBytes(context->device,
                                                           kernel.values.data(),
                                                           kernel.values.size() * sizeof(float),
                                                           context->error);
    id<MTLBuffer> spectrum = makeSharedBuffer(context->device, paddedCount * sizeof(float) * 2u, context->error);
    id<MTLBuffer> scratch = context->scratchCache->acquire(context->device,
                                                           MetalScratchFamily::FftScratch,
                                                           paddedSize,
                                                           1,
                                                           paddedCount * sizeof(float) * 2u,
                                                           context->error);
    id<MTLBuffer> transpose = context->scratchCache->acquire(context->device,
                                                             MetalScratchFamily::FftTranspose,
                                                             paddedSize,
                                                             1,
                                                             paddedCount * sizeof(float) * 2u,
                                                             context->error);
    const FftImageParamsGpu params {0, 0, paddedSize, paddedSize * paddedSize, 1, 0, kernel.size};
    id<MTLBuffer> paramsBuffer = makeParamBuffer(context->device, params, context->error);
    if (kernelValues == nil || spectrum == nil || scratch == nil || transpose == nil || paramsBuffer == nil) {
        return false;
    }
    std::memset(spectrum.contents, 0, paddedCount * sizeof(float) * 2u);

    [context->encoder setBuffer:kernelValues offset:0 atIndex:0];
    [context->encoder setBuffer:spectrum offset:0 atIndex:1];
    [context->encoder setBuffer:paramsBuffer offset:0 atIndex:2];
    dispatch2d(context->encoder, context->pipelines->scatterKernelToComplex, kernel.size, kernel.size);
    if (!encodeSquareForwardFft(context->commandBuffer, context->encoder, context->pipelines, spectrum, scratch, transpose, paddedSize, context->error)) {
        return false;
    }
    *outSpectrum = spectrum;
    return true;
}

bool encodePackPlanesToRgbaMetal(MetalRenderContext* context,
                                 id<MTLBuffer> rPlane,
                                 id<MTLBuffer> gPlane,
                                 id<MTLBuffer> bPlane,
                                 int width,
                                 int height,
                                 id<MTLBuffer>* outImage) {
    if (context == nullptr || context->encoder == nil || outImage == nullptr) {
        if (context != nullptr && context->error != nullptr) {
            *context->error = "metal-invalid-pack-planes";
        }
        return false;
    }
    const NSUInteger pixelCount = static_cast<NSUInteger>(width) * static_cast<NSUInteger>(height);
    id<MTLBuffer> image = makeSharedBuffer(context->device, pixelCount * 4u * sizeof(float), context->error);
    const FftImageParamsGpu params {width, height, 0, 0, 1, 0, 0};
    id<MTLBuffer> paramsBuffer = makeParamBuffer(context->device, params, context->error);
    if (image == nil || paramsBuffer == nil) {
        return false;
    }

    [context->encoder setBuffer:rPlane offset:0 atIndex:0];
    [context->encoder setBuffer:gPlane offset:0 atIndex:1];
    [context->encoder setBuffer:bPlane offset:0 atIndex:2];
    [context->encoder setBuffer:image offset:0 atIndex:3];
    [context->encoder setBuffer:paramsBuffer offset:0 atIndex:4];
    dispatch2d(context->encoder, context->pipelines->packPlanesToRgba, width, height);
    *outImage = image;
    return true;
}

bool encodeConvolveRgbSpectrumStackToImageMetal(MetalRenderContext* context,
                                                id<MTLBuffer> imageSpectrumStack,
                                                id<MTLBuffer> kernelSpectrum,
                                                int width,
                                                int height,
                                                int paddedSize,
                                                id<MTLBuffer>* outImage) {
    if (context == nullptr || context->encoder == nil || outImage == nullptr) {
        if (context != nullptr && context->error != nullptr) {
            *context->error = "metal-invalid-convolve-rgb-stack";
        }
        return false;
    }
    const NSUInteger paddedCount = static_cast<NSUInteger>(paddedSize) * static_cast<NSUInteger>(paddedSize);
    const NSUInteger stackBytes = paddedCount * 3u * sizeof(float) * 2u;
    const NSUInteger planeCount = static_cast<NSUInteger>(width) * static_cast<NSUInteger>(height);
    id<MTLBuffer> tempSpectrum = context->scratchCache->acquire(context->device,
                                                                MetalScratchFamily::TempSpectrum,
                                                                paddedSize,
                                                                3,
                                                                stackBytes,
                                                                context->error);
    id<MTLBuffer> scratch = context->scratchCache->acquire(context->device,
                                                           MetalScratchFamily::FftScratch,
                                                           paddedSize,
                                                           3,
                                                           stackBytes,
                                                           context->error);
    id<MTLBuffer> transpose = context->scratchCache->acquire(context->device,
                                                             MetalScratchFamily::FftTranspose,
                                                             paddedSize,
                                                             3,
                                                             stackBytes,
                                                             context->error);
    id<MTLBuffer> rPlane = makeSharedBuffer(context->device, planeCount * sizeof(float), context->error);
    id<MTLBuffer> gPlane = makeSharedBuffer(context->device, planeCount * sizeof(float), context->error);
    id<MTLBuffer> bPlane = makeSharedBuffer(context->device, planeCount * sizeof(float), context->error);
    const BatchFftParamsGpu batchParams {static_cast<int>(paddedCount), 0, static_cast<int>(paddedCount), 3};
    id<MTLBuffer> batchParamsBuffer = makeParamBuffer(context->device, batchParams, context->error);
    const FftImageParamsGpu planeParams {width, height, paddedSize, paddedSize * paddedSize, 3, 0, 0};
    id<MTLBuffer> planeParamsBuffer = makeParamBuffer(context->device, planeParams, context->error);
    if (tempSpectrum == nil || scratch == nil || transpose == nil ||
        rPlane == nil || gPlane == nil || bPlane == nil ||
        batchParamsBuffer == nil || planeParamsBuffer == nil) {
        return false;
    }

    [context->encoder setBuffer:imageSpectrumStack offset:0 atIndex:0];
    [context->encoder setBuffer:kernelSpectrum offset:0 atIndex:1];
    [context->encoder setBuffer:tempSpectrum offset:0 atIndex:2];
    [context->encoder setBuffer:batchParamsBuffer offset:0 atIndex:3];
    dispatch2d(context->encoder, context->pipelines->multiplyComplexBroadcast, static_cast<int>(paddedCount), 3);
    if (!encodeSquareInverseFftStack(context->commandBuffer, context->encoder, context->pipelines, tempSpectrum, scratch, transpose, paddedSize, 3, context->error)) {
        return false;
    }
    [context->encoder setBuffer:tempSpectrum offset:0 atIndex:0];
    [context->encoder setBuffer:rPlane offset:0 atIndex:1];
    [context->encoder setBuffer:planeParamsBuffer offset:0 atIndex:2];
    dispatch2d(context->encoder, context->pipelines->extractRealPlane, width, height);
    [context->encoder setBuffer:tempSpectrum offset:static_cast<NSUInteger>(paddedCount * sizeof(float) * 2u) atIndex:0];
    [context->encoder setBuffer:gPlane offset:0 atIndex:1];
    [context->encoder setBuffer:planeParamsBuffer offset:0 atIndex:2];
    dispatch2d(context->encoder, context->pipelines->extractRealPlane, width, height);
    [context->encoder setBuffer:tempSpectrum offset:static_cast<NSUInteger>(paddedCount * 2u * sizeof(float) * 2u) atIndex:0];
    [context->encoder setBuffer:bPlane offset:0 atIndex:1];
    [context->encoder setBuffer:planeParamsBuffer offset:0 atIndex:2];
    dispatch2d(context->encoder, context->pipelines->extractRealPlane, width, height);
    return encodePackPlanesToRgbaMetal(context, rPlane, gPlane, bPlane, width, height, outImage);
}

bool encodeConvolveScalarSpectrumToPlanesStackMetal(MetalRenderContext* context,
                                                    id<MTLBuffer> imageSpectrum,
                                                    const std::vector<id<MTLBuffer>>& kernelSpectra,
                                                    int width,
                                                    int height,
                                                    int paddedSize,
                                                    std::array<id<MTLBuffer>, kLensDiffMaxSpectralBins>* outPlanes) {
    if (context == nullptr || context->encoder == nil || outPlanes == nullptr) {
        if (context != nullptr && context->error != nullptr) {
            *context->error = "metal-invalid-convolve-scalar-stack";
        }
        return false;
    }
    const int activeBins = std::min<int>(static_cast<int>(kernelSpectra.size()), kLensDiffMaxSpectralBins);
    if (activeBins <= 0) {
        if (context->error != nullptr) {
            *context->error = "metal-empty-spectral-stack";
        }
        return false;
    }

    const NSUInteger paddedCount = static_cast<NSUInteger>(paddedSize) * static_cast<NSUInteger>(paddedSize);
    const NSUInteger planeCount = static_cast<NSUInteger>(width) * static_cast<NSUInteger>(height);
    const NSUInteger spectrumSliceBytes = paddedCount * sizeof(float) * 2u;
    const NSUInteger stackBytes = spectrumSliceBytes * static_cast<NSUInteger>(activeBins);
    id<MTLBuffer> kernelSpectrumStack = context->scratchCache->acquire(context->device,
                                                                       MetalScratchFamily::KernelSpectrumStack,
                                                                       paddedSize,
                                                                       activeBins,
                                                                       stackBytes,
                                                                       context->error);
    id<MTLBuffer> tempSpectrum = context->scratchCache->acquire(context->device,
                                                                MetalScratchFamily::TempSpectrum,
                                                                paddedSize,
                                                                activeBins,
                                                                stackBytes,
                                                                context->error);
    id<MTLBuffer> scratch = context->scratchCache->acquire(context->device,
                                                           MetalScratchFamily::FftScratch,
                                                           paddedSize,
                                                           activeBins,
                                                           stackBytes,
                                                           context->error);
    id<MTLBuffer> transpose = context->scratchCache->acquire(context->device,
                                                             MetalScratchFamily::FftTranspose,
                                                             paddedSize,
                                                             activeBins,
                                                             stackBytes,
                                                             context->error);
    const BatchFftParamsGpu batchParams {static_cast<int>(paddedCount), 0, static_cast<int>(paddedCount), activeBins};
    id<MTLBuffer> batchParamsBuffer = makeParamBuffer(context->device, batchParams, context->error);
    const FftImageParamsGpu planeParams {width, height, paddedSize, paddedSize * paddedSize, 1, 0, 0};
    id<MTLBuffer> planeParamsBuffer = makeParamBuffer(context->device, planeParams, context->error);
    if (kernelSpectrumStack == nil || tempSpectrum == nil || scratch == nil || transpose == nil ||
        batchParamsBuffer == nil || planeParamsBuffer == nil) {
        return false;
    }

    for (int i = 0; i < activeBins; ++i) {
        if (kernelSpectra[static_cast<std::size_t>(i)] == nil) {
            if (context->error != nullptr) {
                *context->error = "metal-null-kernel-spectrum";
            }
            return false;
        }
        std::memcpy(static_cast<char*>(kernelSpectrumStack.contents) + spectrumSliceBytes * static_cast<NSUInteger>(i),
                    kernelSpectra[static_cast<std::size_t>(i)].contents,
                    spectrumSliceBytes);
        (*outPlanes)[static_cast<std::size_t>(i)] = makeSharedBuffer(context->device, planeCount * sizeof(float), context->error);
        if ((*outPlanes)[static_cast<std::size_t>(i)] == nil) {
            return false;
        }
    }

    [context->encoder setBuffer:kernelSpectrumStack offset:0 atIndex:0];
    [context->encoder setBuffer:imageSpectrum offset:0 atIndex:1];
    [context->encoder setBuffer:tempSpectrum offset:0 atIndex:2];
    [context->encoder setBuffer:batchParamsBuffer offset:0 atIndex:3];
    dispatch2d(context->encoder, context->pipelines->multiplyComplexBroadcast, static_cast<int>(paddedCount), activeBins);
    if (!encodeSquareInverseFftStack(context->commandBuffer, context->encoder, context->pipelines, tempSpectrum, scratch, transpose, paddedSize, activeBins, context->error)) {
        return false;
    }
    for (int i = 0; i < activeBins; ++i) {
        [context->encoder setBuffer:tempSpectrum offset:spectrumSliceBytes * static_cast<NSUInteger>(i) atIndex:0];
        [context->encoder setBuffer:(*outPlanes)[static_cast<std::size_t>(i)] offset:0 atIndex:1];
        [context->encoder setBuffer:planeParamsBuffer offset:0 atIndex:2];
        dispatch2d(context->encoder, context->pipelines->extractRealPlane, width, height);
    }
    for (int i = activeBins; i < kLensDiffMaxSpectralBins; ++i) {
        (*outPlanes)[static_cast<std::size_t>(i)] = nil;
    }
    return true;
}

bool normalizeScalarBuffer(id<MTLDevice> device,
                           id<MTLBuffer> buffer,
                           NSUInteger count,
                           std::string* error) {
    if (buffer == nil || count == 0) {
        return false;
    }
    float* values = static_cast<float*>(buffer.contents);
    float sum = 0.0f;
    for (NSUInteger i = 0; i < count; ++i) {
        sum += values[i];
    }
    if (sum <= 0.0f) {
        return true;
    }
    const float invSum = 1.0f / sum;
    for (NSUInteger i = 0; i < count; ++i) {
        values[i] *= invSum;
    }
    return true;
}

int estimateAdaptiveSupportRadiusMetal(const float* ringEnergy,
                                       const float* ringPeak,
                                       int radiusMax,
                                       float totalEnergy,
                                       float globalPeak) {
    if (radiusMax < 0 || totalEnergy <= 1e-6f || globalPeak <= 1e-6f) {
        return std::max(4, radiusMax);
    }
    std::vector<float> outsidePeak(static_cast<std::size_t>(radiusMax + 2), 0.0f);
    for (int r = radiusMax; r >= 0; --r) {
        outsidePeak[static_cast<std::size_t>(r)] = std::max(ringPeak[static_cast<std::size_t>(r)],
                                                            outsidePeak[static_cast<std::size_t>(r + 1)]);
    }
    float cumulativeEnergy = 0.0f;
    for (int r = 0; r <= radiusMax; ++r) {
        cumulativeEnergy += ringEnergy[static_cast<std::size_t>(r)];
        const float captured = cumulativeEnergy / totalEnergy;
        const float remainingPeak = outsidePeak[static_cast<std::size_t>(std::min(radiusMax + 1, r + 1))];
        if (captured >= 0.9999f && remainingPeak <= globalPeak * 2e-4f) {
            return std::max(4, r + 1);
        }
    }
    return std::max(4, radiusMax);
}

int log2Int(int value) {
    int result = 0;
    int current = std::max(1, value);
    while (current > 1) {
        current >>= 1;
        ++result;
    }
    return result;
}

void normalizeScalarImageHost(std::vector<float>* image) {
    if (image == nullptr) {
        return;
    }
    float sum = 0.0f;
    for (float value : *image) {
        sum += value;
    }
    if (sum <= 0.0f) {
        return;
    }
    const float invSum = 1.0f / sum;
    for (float& value : *image) {
        value *= invSum;
    }
}

bool encodeBatchedForwardFft(id<MTLComputeCommandEncoder> encoder,
                             PipelineBundle* pipelines,
                             id<MTLBuffer> src,
                             id<MTLBuffer> scratch,
                             int length,
                             int batchCount,
                             std::string* error) {
    if (encoder == nil || pipelines == nullptr || src == nil || scratch == nil || length <= 0 || batchCount <= 0 || !isPowerOfTwo(length)) {
        if (error) *error = "metal-invalid-batched-fft";
        return false;
    }
    const BatchFftParamsGpu fftParams {length, log2Int(length), length, batchCount};
    id<MTLBuffer> fftParamsBuffer = makeParamBuffer(pipelines->device, fftParams, error);
    if (fftParamsBuffer == nil) {
        return false;
    }
    [encoder setBuffer:src offset:0 atIndex:0];
    [encoder setBuffer:scratch offset:0 atIndex:1];
    [encoder setBuffer:fftParamsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->bitReverseBatched, length, batchCount);

    id<MTLBuffer> stageSrc = scratch;
    id<MTLBuffer> stageDst = src;
    for (int stageSize = 2; stageSize <= length; stageSize <<= 1) {
        const BatchFftStageParamsGpu stageParams {length, stageSize / 2, stageSize, length, batchCount, 0};
        id<MTLBuffer> stageParamsBuffer = makeParamBuffer(pipelines->device, stageParams, error);
        if (stageParamsBuffer == nil) {
            return false;
        }
        [encoder setBuffer:stageSrc offset:0 atIndex:0];
        [encoder setBuffer:stageDst offset:0 atIndex:1];
        [encoder setBuffer:stageParamsBuffer offset:0 atIndex:2];
        dispatch2d(encoder, pipelines->fftBatchedStage, length / 2, batchCount);
        std::swap(stageSrc, stageDst);
    }
    if (stageSrc != src) {
        [encoder setBuffer:stageSrc offset:0 atIndex:0];
        [encoder setBuffer:src offset:0 atIndex:1];
        [encoder setBuffer:fftParamsBuffer offset:0 atIndex:2];
        dispatch2d(encoder, pipelines->copyComplexBatched, length, batchCount);
    }
    return true;
}

bool encodeBatchedInverseFft(id<MTLComputeCommandEncoder> encoder,
                             PipelineBundle* pipelines,
                             id<MTLBuffer> src,
                             id<MTLBuffer> scratch,
                             int length,
                             int batchCount,
                             std::string* error) {
    if (encoder == nil || pipelines == nullptr || src == nil || scratch == nil || length <= 0 || batchCount <= 0 || !isPowerOfTwo(length)) {
        if (error) *error = "metal-invalid-batched-ifft";
        return false;
    }
    const BatchFftParamsGpu fftParams {length, log2Int(length), length, batchCount};
    id<MTLBuffer> fftParamsBuffer = makeParamBuffer(pipelines->device, fftParams, error);
    if (fftParamsBuffer == nil) {
        return false;
    }
    [encoder setBuffer:src offset:0 atIndex:0];
    [encoder setBuffer:scratch offset:0 atIndex:1];
    [encoder setBuffer:fftParamsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->bitReverseBatched, length, batchCount);

    id<MTLBuffer> stageSrc = scratch;
    id<MTLBuffer> stageDst = src;
    for (int stageSize = 2; stageSize <= length; stageSize <<= 1) {
        const BatchFftStageParamsGpu stageParams {length, stageSize / 2, stageSize, length, batchCount, 1};
        id<MTLBuffer> stageParamsBuffer = makeParamBuffer(pipelines->device, stageParams, error);
        if (stageParamsBuffer == nil) {
            return false;
        }
        [encoder setBuffer:stageSrc offset:0 atIndex:0];
        [encoder setBuffer:stageDst offset:0 atIndex:1];
        [encoder setBuffer:stageParamsBuffer offset:0 atIndex:2];
        dispatch2d(encoder, pipelines->fftBatchedStage, length / 2, batchCount);
        std::swap(stageSrc, stageDst);
    }

    if (stageSrc != src) {
        [encoder setBuffer:stageSrc offset:0 atIndex:0];
        [encoder setBuffer:src offset:0 atIndex:1];
        [encoder setBuffer:fftParamsBuffer offset:0 atIndex:2];
        dispatch2d(encoder, pipelines->copyComplexBatched, length, batchCount);
    }
    return true;
}

bool encodeTransposeComplex(id<MTLComputeCommandEncoder> encoder,
                            PipelineBundle* pipelines,
                            id<MTLBuffer> src,
                            id<MTLBuffer> dst,
                            int size,
                            std::string* error) {
    if (encoder == nil || pipelines == nullptr || src == nil || dst == nil || size <= 0) {
        if (error) *error = "metal-invalid-complex-transpose";
        return false;
    }
    const FftParamsGpu params {size, log2Int(std::max(1, size))};
    id<MTLBuffer> paramsBuffer = makeParamBuffer(pipelines->device, params, error);
    if (paramsBuffer == nil) {
        return false;
    }
    [encoder setBuffer:src offset:0 atIndex:0];
    [encoder setBuffer:dst offset:0 atIndex:1];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->transposeComplex, size, size);
    return true;
}

bool encodeBluestein2dFft(id<MTLComputeCommandEncoder> encoder,
                          PipelineBundle* pipelines,
                          id<MTLBuffer> srcSpectrum,
                          id<MTLBuffer> tempSpectrum,
                          id<MTLBuffer> tempSpectrum2,
                          int size,
                          std::string* error) {
    if (encoder == nil || pipelines == nullptr || srcSpectrum == nil || tempSpectrum == nil || tempSpectrum2 == nil || size <= 0) {
        if (error) *error = "metal-invalid-bluestein-fft";
        return false;
    }
    const int convolutionLength = nextPowerOfTwo(size * 2 - 1);
    const std::size_t convolutionCount = static_cast<std::size_t>(convolutionLength) * size;
    const std::size_t chirpCount = static_cast<std::size_t>(convolutionLength);
    const std::size_t outputCount = static_cast<std::size_t>(size) * size;

    id<MTLDevice> device = pipelines->device;
    id<MTLBuffer> chirpSpatial = makeSharedBuffer(device, chirpCount * sizeof(float) * 2U, error);
    id<MTLBuffer> chirpScratch = makeSharedBuffer(device, chirpCount * sizeof(float) * 2U, error);
    id<MTLBuffer> bluesteinInput = makeSharedBuffer(device, convolutionCount * sizeof(float) * 2U, error);
    id<MTLBuffer> bluesteinScratch = makeSharedBuffer(device, convolutionCount * sizeof(float) * 2U, error);
    id<MTLBuffer> bluesteinOutput = makeSharedBuffer(device, outputCount * sizeof(float) * 2U, error);
    if (chirpSpatial == nil || chirpScratch == nil || bluesteinInput == nil || bluesteinScratch == nil || bluesteinOutput == nil) {
        return false;
    }
    std::memset(chirpSpatial.contents, 0, chirpCount * sizeof(float) * 2U);
    std::memset(chirpScratch.contents, 0, chirpCount * sizeof(float) * 2U);
    std::memset(bluesteinInput.contents, 0, convolutionCount * sizeof(float) * 2U);
    std::memset(bluesteinScratch.contents, 0, convolutionCount * sizeof(float) * 2U);
    std::memset(bluesteinOutput.contents, 0, outputCount * sizeof(float) * 2U);

    const BluesteinParamsGpu bluesteinParams {size, convolutionLength, size};
    const BluesteinParamsGpu chirpParams {size, convolutionLength, 1};
    id<MTLBuffer> chirpParamsBuffer = makeParamBuffer(device, chirpParams, error);
    id<MTLBuffer> bluesteinParamsBuffer = makeParamBuffer(device, bluesteinParams, error);
    if (chirpParamsBuffer == nil || bluesteinParamsBuffer == nil) {
        return false;
    }

    [encoder setBuffer:chirpSpatial offset:0 atIndex:0];
    [encoder setBuffer:chirpParamsBuffer offset:0 atIndex:1];
    dispatch1d256(encoder, pipelines->buildBluesteinChirp, static_cast<NSUInteger>(convolutionLength));
    if (!encodeBatchedForwardFft(encoder, pipelines, chirpSpatial, chirpScratch, convolutionLength, 1, error)) {
        return false;
    }

    [encoder setBuffer:srcSpectrum offset:0 atIndex:0];
    [encoder setBuffer:bluesteinInput offset:0 atIndex:1];
    [encoder setBuffer:bluesteinParamsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->buildBluesteinInput, convolutionLength, size);
    if (!encodeBatchedForwardFft(encoder, pipelines, bluesteinInput, bluesteinScratch, convolutionLength, size, error)) {
        return false;
    }
    [encoder setBuffer:bluesteinInput offset:0 atIndex:0];
    [encoder setBuffer:chirpSpatial offset:0 atIndex:1];
    [encoder setBuffer:tempSpectrum offset:0 atIndex:2];
    [encoder setBuffer:bluesteinParamsBuffer offset:0 atIndex:3];
    dispatch2d(encoder, pipelines->multiplyBluesteinSpectra, convolutionLength, size);
    if (!encodeBatchedInverseFft(encoder, pipelines, tempSpectrum, bluesteinScratch, convolutionLength, size, error)) {
        return false;
    }
    [encoder setBuffer:tempSpectrum offset:0 atIndex:0];
    [encoder setBuffer:bluesteinOutput offset:0 atIndex:1];
    [encoder setBuffer:bluesteinParamsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->extractBluesteinOutput, size, size);

    if (!encodeTransposeComplex(encoder, pipelines, bluesteinOutput, tempSpectrum2, size, error)) {
        return false;
    }
    [encoder setBuffer:tempSpectrum2 offset:0 atIndex:0];
    [encoder setBuffer:bluesteinInput offset:0 atIndex:1];
    [encoder setBuffer:bluesteinParamsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->buildBluesteinInput, convolutionLength, size);
    if (!encodeBatchedForwardFft(encoder, pipelines, bluesteinInput, bluesteinScratch, convolutionLength, size, error)) {
        return false;
    }
    [encoder setBuffer:bluesteinInput offset:0 atIndex:0];
    [encoder setBuffer:chirpSpatial offset:0 atIndex:1];
    [encoder setBuffer:tempSpectrum offset:0 atIndex:2];
    [encoder setBuffer:bluesteinParamsBuffer offset:0 atIndex:3];
    dispatch2d(encoder, pipelines->multiplyBluesteinSpectra, convolutionLength, size);
    if (!encodeBatchedInverseFft(encoder, pipelines, tempSpectrum, bluesteinScratch, convolutionLength, size, error)) {
        return false;
    }
    [encoder setBuffer:tempSpectrum offset:0 atIndex:0];
    [encoder setBuffer:bluesteinOutput offset:0 atIndex:1];
    [encoder setBuffer:bluesteinParamsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->extractBluesteinOutput, size, size);
    return encodeTransposeComplex(encoder, pipelines, bluesteinOutput, srcSpectrum, size, error);
}

bool encodeSquareForwardFft(id<MTLCommandBuffer> commandBuffer,
                            id<MTLComputeCommandEncoder> encoder,
                            PipelineBundle* pipelines,
                            id<MTLBuffer> spectrum,
                            id<MTLBuffer> scratch,
                            id<MTLBuffer> transpose,
                            int size,
                            std::string* error) {
    if (encoder == nil || pipelines == nullptr || spectrum == nil || scratch == nil || transpose == nil || size <= 0 || !isPowerOfTwo(size)) {
        if (error) *error = "metal-invalid-square-forward-fft";
        return false;
    }
    if (LensDiffMetalVkFFTEnabled() &&
        lensDiffMetalVkFFTEncodeSquare(commandBuffer, encoder, spectrum, size, 1, false, nullptr)) {
        return true;
    }
    return encodeBatchedForwardFft(encoder, pipelines, spectrum, scratch, size, size, error) &&
           encodeTransposeComplex(encoder, pipelines, spectrum, transpose, size, error) &&
           encodeBatchedForwardFft(encoder, pipelines, transpose, scratch, size, size, error) &&
           encodeTransposeComplex(encoder, pipelines, transpose, spectrum, size, error);
}

bool encodeSquareInverseFft(id<MTLCommandBuffer> commandBuffer,
                            id<MTLComputeCommandEncoder> encoder,
                            PipelineBundle* pipelines,
                            id<MTLBuffer> spectrum,
                            id<MTLBuffer> scratch,
                            id<MTLBuffer> transpose,
                            int size,
                            std::string* error) {
    if (encoder == nil || pipelines == nullptr || spectrum == nil || scratch == nil || transpose == nil || size <= 0 || !isPowerOfTwo(size)) {
        if (error) *error = "metal-invalid-square-inverse-fft";
        return false;
    }
    if (LensDiffMetalVkFFTEnabled() &&
        lensDiffMetalVkFFTEncodeSquare(commandBuffer, encoder, spectrum, size, 1, true, nullptr)) {
        return true;
    }
    return encodeBatchedInverseFft(encoder, pipelines, spectrum, scratch, size, size, error) &&
           encodeTransposeComplex(encoder, pipelines, spectrum, transpose, size, error) &&
           encodeBatchedInverseFft(encoder, pipelines, transpose, scratch, size, size, error) &&
           encodeTransposeComplex(encoder, pipelines, transpose, spectrum, size, error);
}

bool encodeTransposeComplexStack(id<MTLComputeCommandEncoder> encoder,
                                 PipelineBundle* pipelines,
                                 id<MTLBuffer> src,
                                 id<MTLBuffer> dst,
                                 int size,
                                 int imageCount,
                                 std::string* error) {
    if (encoder == nil || pipelines == nullptr || src == nil || dst == nil || size <= 0 || imageCount <= 0) {
        if (error) *error = "metal-invalid-complex-transpose-stack";
        return false;
    }
    const FftImageParamsGpu params {size, size, size, size * size, imageCount, 0, 0};
    id<MTLBuffer> paramsBuffer = makeParamBuffer(pipelines->device, params, error);
    if (paramsBuffer == nil) {
        return false;
    }
    [encoder setBuffer:src offset:0 atIndex:0];
    [encoder setBuffer:dst offset:0 atIndex:1];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->transposeComplexStack, size, size * imageCount);
    return true;
}

bool encodeSquareForwardFftStack(id<MTLCommandBuffer> commandBuffer,
                                 id<MTLComputeCommandEncoder> encoder,
                                 PipelineBundle* pipelines,
                                 id<MTLBuffer> spectrum,
                                 id<MTLBuffer> scratch,
                                 id<MTLBuffer> transpose,
                                 int size,
                                 int imageCount,
                                 std::string* error) {
    if (encoder == nil || pipelines == nullptr || spectrum == nil || scratch == nil || transpose == nil ||
        size <= 0 || imageCount <= 0 || !isPowerOfTwo(size)) {
        if (error) *error = "metal-invalid-square-forward-fft-stack";
        return false;
    }
    if (LensDiffMetalVkFFTEnabled() &&
        lensDiffMetalVkFFTEncodeSquare(commandBuffer, encoder, spectrum, size, imageCount, false, nullptr)) {
        return true;
    }
    const int batchCount = size * imageCount;
    return encodeBatchedForwardFft(encoder, pipelines, spectrum, scratch, size, batchCount, error) &&
           encodeTransposeComplexStack(encoder, pipelines, spectrum, transpose, size, imageCount, error) &&
           encodeBatchedForwardFft(encoder, pipelines, transpose, scratch, size, batchCount, error) &&
           encodeTransposeComplexStack(encoder, pipelines, transpose, spectrum, size, imageCount, error);
}

bool encodeSquareInverseFftStack(id<MTLCommandBuffer> commandBuffer,
                                 id<MTLComputeCommandEncoder> encoder,
                                 PipelineBundle* pipelines,
                                 id<MTLBuffer> spectrum,
                                 id<MTLBuffer> scratch,
                                 id<MTLBuffer> transpose,
                                 int size,
                                 int imageCount,
                                 std::string* error) {
    if (encoder == nil || pipelines == nullptr || spectrum == nil || scratch == nil || transpose == nil ||
        size <= 0 || imageCount <= 0 || !isPowerOfTwo(size)) {
        if (error) *error = "metal-invalid-square-inverse-fft-stack";
        return false;
    }
    if (LensDiffMetalVkFFTEnabled() &&
        lensDiffMetalVkFFTEncodeSquare(commandBuffer, encoder, spectrum, size, imageCount, true, nullptr)) {
        return true;
    }
    const int batchCount = size * imageCount;
    return encodeBatchedInverseFft(encoder, pipelines, spectrum, scratch, size, batchCount, error) &&
           encodeTransposeComplexStack(encoder, pipelines, spectrum, transpose, size, imageCount, error) &&
           encodeBatchedInverseFft(encoder, pipelines, transpose, scratch, size, batchCount, error) &&
           encodeTransposeComplexStack(encoder, pipelines, transpose, spectrum, size, imageCount, error);
}

bool packPlanesToRgbaMetal(id<MTLDevice> device,
                           id<MTLCommandQueue> queue,
                           PipelineBundle* pipelines,
                           id<MTLBuffer> rPlane,
                           id<MTLBuffer> gPlane,
                           id<MTLBuffer> bPlane,
                           int width,
                           int height,
                           id<MTLBuffer>* outImage,
                           std::string* error);

bool makeScalarSpectrumMetal(id<MTLDevice> device,
                             id<MTLCommandQueue> queue,
                             PipelineBundle* pipelines,
                             id<MTLBuffer> srcPlane,
                             int width,
                             int height,
                             int paddedSize,
                             id<MTLBuffer>* outSpectrum,
                             std::string* error) {
    const NSUInteger paddedCount = static_cast<NSUInteger>(paddedSize) * static_cast<NSUInteger>(paddedSize);
    id<MTLBuffer> spectrum = makeSharedBuffer(device, paddedCount * sizeof(float) * 2u, error);
    id<MTLBuffer> scratch = makeSharedBuffer(device, paddedCount * sizeof(float) * 2u, error);
    id<MTLBuffer> transpose = makeSharedBuffer(device, paddedCount * sizeof(float) * 2u, error);
    const FftImageParamsGpu params {width, height, paddedSize, paddedSize * paddedSize, 1, 0, 0};
    id<MTLBuffer> paramsBuffer = makeParamBuffer(device, params, error);
    if (spectrum == nil || scratch == nil || transpose == nil || paramsBuffer == nil) {
        return false;
    }

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setBuffer:srcPlane offset:0 atIndex:0];
    [encoder setBuffer:spectrum offset:0 atIndex:1];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->padPlaneToComplex, paddedSize, paddedSize);
    const bool ok = encodeSquareForwardFft(commandBuffer, encoder, pipelines, spectrum, scratch, transpose, paddedSize, error);
    [encoder endEncoding];
    if (!ok || !commitAndWait(commandBuffer, error)) {
        return false;
    }
    *outSpectrum = spectrum;
    return true;
}

bool makeRgbChannelSpectrumMetal(id<MTLDevice> device,
                                 id<MTLCommandQueue> queue,
                                 PipelineBundle* pipelines,
                                 id<MTLBuffer> srcImage,
                                 int channelIndex,
                                 int width,
                                 int height,
                                 int paddedSize,
                                 id<MTLBuffer>* outSpectrum,
                                 std::string* error) {
    const NSUInteger paddedCount = static_cast<NSUInteger>(paddedSize) * static_cast<NSUInteger>(paddedSize);
    id<MTLBuffer> spectrum = makeSharedBuffer(device, paddedCount * sizeof(float) * 2u, error);
    id<MTLBuffer> scratch = makeSharedBuffer(device, paddedCount * sizeof(float) * 2u, error);
    id<MTLBuffer> transpose = makeSharedBuffer(device, paddedCount * sizeof(float) * 2u, error);
    const FftImageParamsGpu params {width, height, paddedSize, paddedSize * paddedSize, 1, channelIndex, 0};
    id<MTLBuffer> paramsBuffer = makeParamBuffer(device, params, error);
    if (spectrum == nil || scratch == nil || transpose == nil || paramsBuffer == nil) {
        return false;
    }

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setBuffer:srcImage offset:0 atIndex:0];
    [encoder setBuffer:spectrum offset:0 atIndex:1];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->padRgbChannelToComplex, paddedSize, paddedSize);
    const bool ok = encodeSquareForwardFft(commandBuffer, encoder, pipelines, spectrum, scratch, transpose, paddedSize, error);
    [encoder endEncoding];
    if (!ok || !commitAndWait(commandBuffer, error)) {
        return false;
    }
    *outSpectrum = spectrum;
    return true;
}

bool makeRgbSpectraStackMetal(id<MTLDevice> device,
                              id<MTLCommandQueue> queue,
                              PipelineBundle* pipelines,
                              id<MTLBuffer> srcImage,
                              int width,
                              int height,
                              int paddedSize,
                              id<MTLBuffer>* outSpectrum,
                              std::string* error) {
    const NSUInteger paddedCount = static_cast<NSUInteger>(paddedSize) * static_cast<NSUInteger>(paddedSize);
    const NSUInteger stackBytes = paddedCount * 3u * sizeof(float) * 2u;
    id<MTLBuffer> spectrum = makeSharedBuffer(device, stackBytes, error);
    id<MTLBuffer> scratch = makeSharedBuffer(device, stackBytes, error);
    id<MTLBuffer> transpose = makeSharedBuffer(device, stackBytes, error);
    const FftImageParamsGpu params {width, height, paddedSize, paddedSize * paddedSize, 3, 0, 0};
    id<MTLBuffer> paramsBuffer = makeParamBuffer(device, params, error);
    if (spectrum == nil || scratch == nil || transpose == nil || paramsBuffer == nil) {
        return false;
    }

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setBuffer:srcImage offset:0 atIndex:0];
    [encoder setBuffer:spectrum offset:0 atIndex:1];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->padRgbToComplexStack, paddedSize, paddedSize * 3);
    const bool ok = encodeSquareForwardFftStack(commandBuffer, encoder, pipelines, spectrum, scratch, transpose, paddedSize, 3, error);
    [encoder endEncoding];
    if (!ok || !commitAndWait(commandBuffer, error)) {
        return false;
    }
    *outSpectrum = spectrum;
    return true;
}

bool makeKernelSpectrumMetal(id<MTLDevice> device,
                             id<MTLCommandQueue> queue,
                             PipelineBundle* pipelines,
                             const LensDiffKernel& kernel,
                             int paddedSize,
                             id<MTLBuffer>* outSpectrum,
                             std::string* error) {
    const NSUInteger paddedCount = static_cast<NSUInteger>(paddedSize) * static_cast<NSUInteger>(paddedSize);
    id<MTLBuffer> kernelValues = makeSharedBufferWithBytes(device,
                                                           kernel.values.data(),
                                                           kernel.values.size() * sizeof(float),
                                                           error);
    id<MTLBuffer> spectrum = makeSharedBuffer(device, paddedCount * sizeof(float) * 2u, error);
    id<MTLBuffer> scratch = makeSharedBuffer(device, paddedCount * sizeof(float) * 2u, error);
    id<MTLBuffer> transpose = makeSharedBuffer(device, paddedCount * sizeof(float) * 2u, error);
    const FftImageParamsGpu params {0, 0, paddedSize, paddedSize * paddedSize, 1, 0, kernel.size};
    id<MTLBuffer> paramsBuffer = makeParamBuffer(device, params, error);
    if (kernelValues == nil || spectrum == nil || scratch == nil || transpose == nil || paramsBuffer == nil) {
        return false;
    }
    std::memset(spectrum.contents, 0, paddedCount * sizeof(float) * 2u);

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setBuffer:kernelValues offset:0 atIndex:0];
    [encoder setBuffer:spectrum offset:0 atIndex:1];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->scatterKernelToComplex, kernel.size, kernel.size);
    const bool ok = encodeSquareForwardFft(commandBuffer, encoder, pipelines, spectrum, scratch, transpose, paddedSize, error);
    [encoder endEncoding];
    if (!ok || !commitAndWait(commandBuffer, error)) {
        return false;
    }
    *outSpectrum = spectrum;
    return true;
}

bool convolveSpectrumToPlaneMetal(id<MTLDevice> device,
                                  id<MTLCommandQueue> queue,
                                  PipelineBundle* pipelines,
                                  id<MTLBuffer> imageSpectrum,
                                  id<MTLBuffer> kernelSpectrum,
                                  int width,
                                  int height,
                                  int paddedSize,
                                  id<MTLBuffer>* outPlane,
                                  std::string* error) {
    const NSUInteger paddedCount = static_cast<NSUInteger>(paddedSize) * static_cast<NSUInteger>(paddedSize);
    const NSUInteger planeCount = static_cast<NSUInteger>(width) * static_cast<NSUInteger>(height);
    id<MTLBuffer> tempSpectrum = makeSharedBuffer(device, paddedCount * sizeof(float) * 2u, error);
    id<MTLBuffer> scratch = makeSharedBuffer(device, paddedCount * sizeof(float) * 2u, error);
    id<MTLBuffer> transpose = makeSharedBuffer(device, paddedCount * sizeof(float) * 2u, error);
    id<MTLBuffer> plane = makeSharedBuffer(device, planeCount * sizeof(float), error);
    const FftImageParamsGpu params {width, height, paddedSize, paddedSize * paddedSize, 1, 0, 0};
    id<MTLBuffer> paramsBuffer = makeParamBuffer(device, params, error);
    if (tempSpectrum == nil || scratch == nil || transpose == nil || plane == nil || paramsBuffer == nil) {
        return false;
    }

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setBuffer:imageSpectrum offset:0 atIndex:0];
    [encoder setBuffer:kernelSpectrum offset:0 atIndex:1];
    [encoder setBuffer:tempSpectrum offset:0 atIndex:2];
    dispatch1d256(encoder, pipelines->multiplyComplex, paddedCount);
    const bool ok = encodeSquareInverseFft(commandBuffer, encoder, pipelines, tempSpectrum, scratch, transpose, paddedSize, error);
    [encoder setBuffer:tempSpectrum offset:0 atIndex:0];
    [encoder setBuffer:plane offset:0 atIndex:1];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->extractRealPlane, width, height);
    [encoder endEncoding];
    if (!ok || !commitAndWait(commandBuffer, error)) {
        return false;
    }
    *outPlane = plane;
    return true;
}

bool convolveRgbSpectrumStackToImageMetal(id<MTLDevice> device,
                                          id<MTLCommandQueue> queue,
                                          PipelineBundle* pipelines,
                                          id<MTLBuffer> imageSpectrumStack,
                                          id<MTLBuffer> kernelSpectrum,
                                          int width,
                                          int height,
                                          int paddedSize,
                                          id<MTLBuffer>* outImage,
                                          std::string* error) {
    const NSUInteger paddedCount = static_cast<NSUInteger>(paddedSize) * static_cast<NSUInteger>(paddedSize);
    const NSUInteger stackBytes = paddedCount * 3u * sizeof(float) * 2u;
    const NSUInteger planeCount = static_cast<NSUInteger>(width) * static_cast<NSUInteger>(height);
    id<MTLBuffer> tempSpectrum = makeSharedBuffer(device, stackBytes, error);
    id<MTLBuffer> scratch = makeSharedBuffer(device, stackBytes, error);
    id<MTLBuffer> transpose = makeSharedBuffer(device, stackBytes, error);
    id<MTLBuffer> rPlane = makeSharedBuffer(device, planeCount * sizeof(float), error);
    id<MTLBuffer> gPlane = makeSharedBuffer(device, planeCount * sizeof(float), error);
    id<MTLBuffer> bPlane = makeSharedBuffer(device, planeCount * sizeof(float), error);
    const BatchFftParamsGpu batchParams {static_cast<int>(paddedCount), 0, static_cast<int>(paddedCount), 3};
    id<MTLBuffer> batchParamsBuffer = makeParamBuffer(device, batchParams, error);
    const FftImageParamsGpu planeParams {width, height, paddedSize, paddedSize * paddedSize, 3, 0, 0};
    id<MTLBuffer> planeParamsBuffer = makeParamBuffer(device, planeParams, error);
    if (tempSpectrum == nil || scratch == nil || transpose == nil ||
        rPlane == nil || gPlane == nil || bPlane == nil ||
        batchParamsBuffer == nil || planeParamsBuffer == nil) {
        return false;
    }

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setBuffer:imageSpectrumStack offset:0 atIndex:0];
    [encoder setBuffer:kernelSpectrum offset:0 atIndex:1];
    [encoder setBuffer:tempSpectrum offset:0 atIndex:2];
    [encoder setBuffer:batchParamsBuffer offset:0 atIndex:3];
    dispatch2d(encoder, pipelines->multiplyComplexBroadcast, static_cast<int>(paddedCount), 3);
    const bool ok = encodeSquareInverseFftStack(commandBuffer, encoder, pipelines, tempSpectrum, scratch, transpose, paddedSize, 3, error);
    [encoder setBuffer:tempSpectrum offset:0 atIndex:0];
    [encoder setBuffer:rPlane offset:0 atIndex:1];
    [encoder setBuffer:planeParamsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->extractRealPlane, width, height);
    [encoder setBuffer:tempSpectrum offset:static_cast<NSUInteger>(paddedCount * sizeof(float) * 2u) atIndex:0];
    [encoder setBuffer:gPlane offset:0 atIndex:1];
    [encoder setBuffer:planeParamsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->extractRealPlane, width, height);
    [encoder setBuffer:tempSpectrum offset:static_cast<NSUInteger>(paddedCount * 2u * sizeof(float) * 2u) atIndex:0];
    [encoder setBuffer:bPlane offset:0 atIndex:1];
    [encoder setBuffer:planeParamsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->extractRealPlane, width, height);
    [encoder endEncoding];
    if (!ok || !commitAndWait(commandBuffer, error)) {
        return false;
    }
    return packPlanesToRgbaMetal(device, queue, pipelines, rPlane, gPlane, bPlane, width, height, outImage, error);
}

bool convolveScalarSpectrumToPlanesStackMetal(id<MTLDevice> device,
                                              id<MTLCommandQueue> queue,
                                              PipelineBundle* pipelines,
                                              id<MTLBuffer> imageSpectrum,
                                              const std::vector<id<MTLBuffer>>& kernelSpectra,
                                              int width,
                                              int height,
                                              int paddedSize,
                                              std::array<id<MTLBuffer>, kLensDiffMaxSpectralBins>* outPlanes,
                                              std::string* error) {
    if (outPlanes == nullptr) {
        return false;
    }
    const int activeBins = std::min<int>(static_cast<int>(kernelSpectra.size()), kLensDiffMaxSpectralBins);
    if (activeBins <= 0) {
        return false;
    }
    const NSUInteger paddedCount = static_cast<NSUInteger>(paddedSize) * static_cast<NSUInteger>(paddedSize);
    const NSUInteger planeCount = static_cast<NSUInteger>(width) * static_cast<NSUInteger>(height);
    const NSUInteger spectrumSliceBytes = paddedCount * sizeof(float) * 2u;
    const NSUInteger stackBytes = spectrumSliceBytes * static_cast<NSUInteger>(activeBins);
    id<MTLBuffer> kernelSpectrumStack = makeSharedBuffer(device, stackBytes, error);
    id<MTLBuffer> tempSpectrum = makeSharedBuffer(device, stackBytes, error);
    id<MTLBuffer> scratch = makeSharedBuffer(device, stackBytes, error);
    id<MTLBuffer> transpose = makeSharedBuffer(device, stackBytes, error);
    const BatchFftParamsGpu batchParams {static_cast<int>(paddedCount), 0, static_cast<int>(paddedCount), activeBins};
    id<MTLBuffer> batchParamsBuffer = makeParamBuffer(device, batchParams, error);
    const FftImageParamsGpu planeParams {width, height, paddedSize, paddedSize * paddedSize, 1, 0, 0};
    id<MTLBuffer> planeParamsBuffer = makeParamBuffer(device, planeParams, error);
    if (kernelSpectrumStack == nil || tempSpectrum == nil || scratch == nil || transpose == nil ||
        batchParamsBuffer == nil || planeParamsBuffer == nil) {
        return false;
    }
    for (int i = 0; i < activeBins; ++i) {
        if (kernelSpectra[static_cast<std::size_t>(i)] == nil) {
            return false;
        }
        std::memcpy(static_cast<char*>(kernelSpectrumStack.contents) + spectrumSliceBytes * static_cast<NSUInteger>(i),
                    kernelSpectra[static_cast<std::size_t>(i)].contents,
                    spectrumSliceBytes);
        (*outPlanes)[static_cast<std::size_t>(i)] = makeSharedBuffer(device, planeCount * sizeof(float), error);
        if ((*outPlanes)[static_cast<std::size_t>(i)] == nil) {
            return false;
        }
    }

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setBuffer:kernelSpectrumStack offset:0 atIndex:0];
    [encoder setBuffer:imageSpectrum offset:0 atIndex:1];
    [encoder setBuffer:tempSpectrum offset:0 atIndex:2];
    [encoder setBuffer:batchParamsBuffer offset:0 atIndex:3];
    dispatch2d(encoder, pipelines->multiplyComplexBroadcast, static_cast<int>(paddedCount), activeBins);
    const bool ok = encodeSquareInverseFftStack(commandBuffer, encoder, pipelines, tempSpectrum, scratch, transpose, paddedSize, activeBins, error);
    for (int i = 0; i < activeBins; ++i) {
        [encoder setBuffer:tempSpectrum offset:spectrumSliceBytes * static_cast<NSUInteger>(i) atIndex:0];
        [encoder setBuffer:(*outPlanes)[static_cast<std::size_t>(i)] offset:0 atIndex:1];
        [encoder setBuffer:planeParamsBuffer offset:0 atIndex:2];
        dispatch2d(encoder, pipelines->extractRealPlane, width, height);
    }
    [encoder endEncoding];
    if (!ok || !commitAndWait(commandBuffer, error)) {
        return false;
    }
    for (int i = activeBins; i < kLensDiffMaxSpectralBins; ++i) {
        (*outPlanes)[static_cast<std::size_t>(i)] = nil;
    }
    return true;
}

bool packPlanesToRgbaMetal(id<MTLDevice> device,
                           id<MTLCommandQueue> queue,
                           PipelineBundle* pipelines,
                           id<MTLBuffer> rPlane,
                           id<MTLBuffer> gPlane,
                           id<MTLBuffer> bPlane,
                           int width,
                           int height,
                           id<MTLBuffer>* outImage,
                           std::string* error) {
    const NSUInteger pixelCount = static_cast<NSUInteger>(width) * static_cast<NSUInteger>(height);
    id<MTLBuffer> image = makeSharedBuffer(device, pixelCount * 4u * sizeof(float), error);
    const FftImageParamsGpu params {width, height, 0, 0, 1, 0, 0};
    id<MTLBuffer> paramsBuffer = makeParamBuffer(device, params, error);
    if (image == nil || paramsBuffer == nil) {
        return false;
    }

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setBuffer:rPlane offset:0 atIndex:0];
    [encoder setBuffer:gPlane offset:0 atIndex:1];
    [encoder setBuffer:bPlane offset:0 atIndex:2];
    [encoder setBuffer:image offset:0 atIndex:3];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:4];
    dispatch2d(encoder, pipelines->packPlanesToRgba, width, height);
    [encoder endEncoding];
    if (!commitAndWait(commandBuffer, error)) {
        return false;
    }
    *outImage = image;
    return true;
}

bool buildOpticalPrepAndRawPsfMetal(const LensDiffParams& params,
                                    int pupilSize,
                                    int rawPsfSize,
                                    id<MTLDevice> device,
                                    id<MTLCommandQueue> queue,
                                    PipelineBundle* pipelines,
                                    std::vector<float>* outPupil,
                                    std::vector<float>* outPhase,
                                    std::vector<float>* outRawPsf,
                                    std::string* error) {
    if (device == nil || queue == nil || pipelines == nullptr || outPupil == nullptr || outPhase == nullptr || outRawPsf == nullptr) {
        if (error) *error = "metal-optical-prep-null-target";
        return false;
    }

    const std::size_t pupilCount = static_cast<std::size_t>(pupilSize) * pupilSize;
    const std::size_t rawCount = static_cast<std::size_t>(rawPsfSize) * rawPsfSize;
    id<MTLBuffer> pupilBuffer = makeSharedBuffer(device, pupilCount * sizeof(float), error);
    id<MTLBuffer> phaseBuffer = makeSharedBuffer(device, pupilCount * sizeof(float), error);
    id<MTLBuffer> spectrumA = makeSharedBuffer(device, rawCount * sizeof(float) * 2U, error);
    id<MTLBuffer> spectrumB = makeSharedBuffer(device, rawCount * sizeof(float) * 2U, error);
    id<MTLBuffer> spectrumC = makeSharedBuffer(device, rawCount * sizeof(float) * 2U, error);
    id<MTLBuffer> rawPsfBuffer = makeSharedBuffer(device, rawCount * sizeof(float), error);
    if (pupilBuffer == nil || phaseBuffer == nil || spectrumA == nil || spectrumB == nil || spectrumC == nil || rawPsfBuffer == nil) {
        return false;
    }

    std::memset(pupilBuffer.contents, 0, pupilCount * sizeof(float));
    std::memset(phaseBuffer.contents, 0, pupilCount * sizeof(float));
    std::memset(spectrumA.contents, 0, rawCount * sizeof(float) * 2U);
    std::memset(spectrumB.contents, 0, rawCount * sizeof(float) * 2U);
    std::memset(spectrumC.contents, 0, rawCount * sizeof(float) * 2U);
    std::memset(rawPsfBuffer.contents, 0, rawCount * sizeof(float));

    id<MTLBuffer> customImageBuffer = nil;
    PupilRasterParamsGpu pupilParams {};
    pupilParams.size = pupilSize;
    pupilParams.apertureMode = static_cast<int>(params.apertureMode);
    pupilParams.apodizationMode = static_cast<int>(params.apodizationMode);
    pupilParams.bladeCount = params.bladeCount;
    pupilParams.vaneCount = params.vaneCount;
    pupilParams.roundness = static_cast<float>(params.roundness);
    pupilParams.rotationRad = static_cast<float>(params.rotationDeg * kPi / 180.0);
    pupilParams.outerRadius = 0.86f;
    pupilParams.centralObstruction = static_cast<float>(std::clamp(params.centralObstruction, 0.0, 0.95)) * pupilParams.outerRadius;
    pupilParams.vaneThickness = static_cast<float>(std::max(0.0, params.vaneThickness));
    pupilParams.pupilDecenterX = static_cast<float>(params.pupilDecenterX);
    pupilParams.pupilDecenterY = static_cast<float>(params.pupilDecenterY);
    pupilParams.starInnerRadiusRatio = 0.18f + pupilParams.roundness * 0.62f;
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
            customImageBuffer = makeSharedBufferWithBytes(device, image.values.data(), image.values.size() * sizeof(float), error);
            if (customImageBuffer == nil) {
                return false;
            }
            pupilParams.customWidth = image.width;
            pupilParams.customHeight = image.height;
            const float imageAspect = static_cast<float>(image.width) / static_cast<float>(std::max(1, image.height));
            pupilParams.fitHalfWidth = imageAspect >= 1.0f ? 1.0f : imageAspect;
            pupilParams.fitHalfHeight = imageAspect >= 1.0f ? 1.0f / imageAspect : 1.0f;
        }
    }

    const bool hasPhase = HasLensDiffNonFlatPhase(params);
    PhaseRasterParamsGpu phaseParams {};
    phaseParams.size = pupilSize;
    phaseParams.hasPhase = hasPhase ? 1 : 0;
    phaseParams.rotationRad = static_cast<float>(params.rotationDeg * kPi / 180.0);
    phaseParams.outerRadius = 0.86f;
    phaseParams.pupilDecenterX = static_cast<float>(params.pupilDecenterX);
    phaseParams.pupilDecenterY = static_cast<float>(params.pupilDecenterY);
    phaseParams.phaseDefocus = static_cast<float>(params.phaseDefocus);
    phaseParams.phaseAstigmatism0 = static_cast<float>(params.phaseAstigmatism0);
    phaseParams.phaseAstigmatism45 = static_cast<float>(params.phaseAstigmatism45);
    phaseParams.phaseComaX = static_cast<float>(params.phaseComaX);
    phaseParams.phaseComaY = static_cast<float>(params.phaseComaY);
    phaseParams.phaseSpherical = static_cast<float>(params.phaseSpherical);
    phaseParams.phaseTrefoilX = static_cast<float>(params.phaseTrefoilX);
    phaseParams.phaseTrefoilY = static_cast<float>(params.phaseTrefoilY);
    phaseParams.phaseSecondaryAstigmatism0 = static_cast<float>(params.phaseSecondaryAstigmatism0);
    phaseParams.phaseSecondaryAstigmatism45 = static_cast<float>(params.phaseSecondaryAstigmatism45);
    phaseParams.phaseQuadrafoil0 = static_cast<float>(params.phaseQuadrafoil0);
    phaseParams.phaseQuadrafoil45 = static_cast<float>(params.phaseQuadrafoil45);
    phaseParams.phaseSecondaryComaX = static_cast<float>(params.phaseSecondaryComaX);
    phaseParams.phaseSecondaryComaY = static_cast<float>(params.phaseSecondaryComaY);

    id<MTLBuffer> pupilParamsBuffer = makeParamBuffer(device, pupilParams, error);
    id<MTLBuffer> phaseParamsBuffer = makeParamBuffer(device, phaseParams, error);
    const EmbedComplexParamsGpu embedParams {pupilSize, rawPsfSize, std::max(0, (rawPsfSize - pupilSize) / 2)};
    id<MTLBuffer> embedParamsBuffer = makeParamBuffer(device, embedParams, error);
    if (pupilParamsBuffer == nil || phaseParamsBuffer == nil || embedParamsBuffer == nil) {
        return false;
    }

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (commandBuffer == nil) {
        if (error) *error = "metal-command-buffer-create-failed";
        return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        if (error) *error = "metal-compute-encoder-create-failed";
        return false;
    }

    [encoder setBuffer:customImageBuffer offset:0 atIndex:0];
    [encoder setBuffer:pupilBuffer offset:0 atIndex:1];
    [encoder setBuffer:pupilParamsBuffer offset:0 atIndex:2];
    dispatch2d(encoder, pipelines->buildPupil, pupilSize, pupilSize);

    [encoder setBuffer:phaseBuffer offset:0 atIndex:0];
    [encoder setBuffer:phaseParamsBuffer offset:0 atIndex:1];
    dispatch2d(encoder, pipelines->buildPhase, pupilSize, pupilSize);

    [encoder setBuffer:pupilBuffer offset:0 atIndex:0];
    [encoder setBuffer:phaseBuffer offset:0 atIndex:1];
    [encoder setBuffer:spectrumA offset:0 atIndex:2];
    [encoder setBuffer:embedParamsBuffer offset:0 atIndex:3];
    dispatch2d(encoder, pipelines->embedComplexPupil, pupilSize, pupilSize);

    bool gpuRawPsfBuilt = false;
    if (isPowerOfTwo(rawPsfSize)) {
        const FftParamsGpu fftParams {rawPsfSize, log2Int(rawPsfSize)};
        id<MTLBuffer> fftParamsBuffer = makeParamBuffer(device, fftParams, error);
        if (fftParamsBuffer == nil) {
            [encoder endEncoding];
            return false;
        }
        [encoder setBuffer:spectrumA offset:0 atIndex:0];
        [encoder setBuffer:spectrumB offset:0 atIndex:1];
        [encoder setBuffer:fftParamsBuffer offset:0 atIndex:2];
        dispatch2d(encoder, pipelines->bitReverseRows, rawPsfSize, rawPsfSize);

        id<MTLBuffer> rowSrc = spectrumB;
        id<MTLBuffer> rowDst = spectrumA;
        for (int stageSize = 2; stageSize <= rawPsfSize; stageSize <<= 1) {
            const FftStageParamsGpu stageParams {rawPsfSize, stageSize / 2, stageSize};
            id<MTLBuffer> stageParamsBuffer = makeParamBuffer(device, stageParams, error);
            if (stageParamsBuffer == nil) {
                [encoder endEncoding];
                return false;
            }
            [encoder setBuffer:rowSrc offset:0 atIndex:0];
            [encoder setBuffer:rowDst offset:0 atIndex:1];
            [encoder setBuffer:stageParamsBuffer offset:0 atIndex:2];
            dispatch2d(encoder, pipelines->fftRowsStage, rawPsfSize / 2, rawPsfSize);
            std::swap(rowSrc, rowDst);
        }

        [encoder setBuffer:rowSrc offset:0 atIndex:0];
        [encoder setBuffer:rowDst offset:0 atIndex:1];
        [encoder setBuffer:fftParamsBuffer offset:0 atIndex:2];
        dispatch2d(encoder, pipelines->bitReverseColumns, rawPsfSize, rawPsfSize);

        id<MTLBuffer> colSrc = rowDst;
        id<MTLBuffer> colDst = rowSrc;
        for (int stageSize = 2; stageSize <= rawPsfSize; stageSize <<= 1) {
            const FftStageParamsGpu stageParams {rawPsfSize, stageSize / 2, stageSize};
            id<MTLBuffer> stageParamsBuffer = makeParamBuffer(device, stageParams, error);
            if (stageParamsBuffer == nil) {
                [encoder endEncoding];
                return false;
            }
            [encoder setBuffer:colSrc offset:0 atIndex:0];
            [encoder setBuffer:colDst offset:0 atIndex:1];
            [encoder setBuffer:stageParamsBuffer offset:0 atIndex:2];
            dispatch2d(encoder, pipelines->fftColumnsStage, rawPsfSize, rawPsfSize / 2);
            std::swap(colSrc, colDst);
        }

        [encoder setBuffer:colSrc offset:0 atIndex:0];
        [encoder setBuffer:rawPsfBuffer offset:0 atIndex:1];
        [encoder setBuffer:fftParamsBuffer offset:0 atIndex:2];
        dispatch2d(encoder, pipelines->extractShiftedIntensity, rawPsfSize, rawPsfSize);
        gpuRawPsfBuilt = true;
    } else {
        if (!encodeBluestein2dFft(encoder, pipelines, spectrumA, spectrumB, spectrumC, rawPsfSize, error)) {
            [encoder endEncoding];
            return false;
        }
        const FftParamsGpu fftParams {rawPsfSize, log2Int(std::max(1, rawPsfSize))};
        id<MTLBuffer> fftParamsBuffer = makeParamBuffer(device, fftParams, error);
        if (fftParamsBuffer == nil) {
            [encoder endEncoding];
            return false;
        }
        [encoder setBuffer:spectrumA offset:0 atIndex:0];
        [encoder setBuffer:rawPsfBuffer offset:0 atIndex:1];
        [encoder setBuffer:fftParamsBuffer offset:0 atIndex:2];
        dispatch2d(encoder, pipelines->extractShiftedIntensity, rawPsfSize, rawPsfSize);
        gpuRawPsfBuilt = true;
    }

    [encoder endEncoding];
    if (!commitAndWait(commandBuffer, error)) {
        return false;
    }

    const float* pupilValues = static_cast<const float*>(pupilBuffer.contents);
    const float* phaseValues = static_cast<const float*>(phaseBuffer.contents);
    outPupil->assign(pupilValues, pupilValues + pupilCount);
    outPhase->assign(phaseValues, phaseValues + pupilCount);

    if (gpuRawPsfBuilt) {
        const float* rawValues = static_cast<const float*>(rawPsfBuffer.contents);
        outRawPsf->assign(rawValues, rawValues + rawCount);
        normalizeScalarImageHost(outRawPsf);
    }
    return true;
}

bool finalizePsfBankMetal(id<MTLDevice> device,
                          id<MTLCommandQueue> queue,
                          PipelineBundle* pipelines,
                          const LensDiffParams& params,
                          const LensDiffPsfBankKey& key,
                          const std::vector<float>& pupil,
                          const std::vector<float>& phaseWaves,
                          int pupilSize,
                          const std::vector<float>& rawPsf,
                          int rawPsfSize,
                          const std::vector<float>& wavelengths,
                          float scaleBase,
                          LensDiffPsfBankCache* cache,
                          MetalPsfBuildContext* buildContext,
                          std::string* error) {
    if (cache == nullptr || pipelines == nullptr) {
        if (error) *error = "metal-null-psf-cache";
        return false;
    }
    MetalPsfBuildContext localBuildContext {};
    localBuildContext.device = device;
    MetalPsfBuildContext* psfContext = buildContext != nullptr ? buildContext : &localBuildContext;
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

    const NSUInteger rawCount = static_cast<NSUInteger>(rawPsf.size());
    id<MTLBuffer> rawPsfBuffer = makeSharedBufferWithBytes(device, rawPsf.data(), rawCount * sizeof(float), error);
    if (rawPsfBuffer == nil) {
        return false;
    }
    const int supportRadius = std::max(4, ResolveLensDiffMaxKernelRadiusPx(params));
    const int kernelSize = supportRadius * 2 + 1;
    const NSUInteger kernelCount = static_cast<NSUInteger>(kernelSize) * static_cast<NSUInteger>(kernelSize);
    const KernelShapeParamsGpu shapeParams {kernelSize, kernelSize / 2};
    id<MTLBuffer> shapeParamsBuffer = makeParamBuffer(device, shapeParams, error);
    const ConvolutionParamsGpu resampleParams {0, rawPsfSize, kernelSize, supportRadius};
    id<MTLBuffer> resampleParamsBuffer = makeParamBuffer(device, resampleParams, error);
    if (shapeParamsBuffer == nil || resampleParamsBuffer == nil) {
        return false;
    }

    auto acquireKernelBuffer = [&](PsfBufferSlot slot) {
        return psfContext->acquire(slot, kernelCount * sizeof(float), error);
    };
    auto acquireRingFloatBuffer = [&](PsfBufferSlot slot) {
        return psfContext->acquire(slot, static_cast<NSUInteger>(shapeParams.radiusMax + 1) * sizeof(float), error);
    };
    auto acquireRingCountBuffer = [&]() {
        return psfContext->acquire(PsfBufferSlot::RingCounts,
                                   static_cast<NSUInteger>(shapeParams.radiusMax + 1) * sizeof(uint32_t),
                                   error);
    };

    for (float wavelength : wavelengths) {
        const auto wavelengthStart = std::chrono::steady_clock::now();
        const float scaleFactor = scaleBase * (wavelength / 550.0f);
        const float invScale = 1.0f / std::max(scaleFactor, 0.05f);
        id<MTLBuffer> invScaleBuffer = makeSharedBufferWithBytes(device, &invScale, sizeof(float), error);
        id<MTLBuffer> baseKernel = acquireKernelBuffer(PsfBufferSlot::BaseKernel);
        id<MTLBuffer> ringSums = acquireRingFloatBuffer(PsfBufferSlot::RingSums);
        id<MTLBuffer> ringCounts = acquireRingCountBuffer();
        id<MTLBuffer> meanKernel = acquireKernelBuffer(PsfBufferSlot::MeanKernel);
        id<MTLBuffer> gainBuffer = nil;
        id<MTLBuffer> shapedKernel = acquireKernelBuffer(PsfBufferSlot::ShapedKernel);
        id<MTLBuffer> structureKernel = acquireKernelBuffer(PsfBufferSlot::StructureKernel);
        id<MTLBuffer> ringEnergy = acquireRingFloatBuffer(PsfBufferSlot::RingEnergy);
        id<MTLBuffer> ringPeak = acquireRingFloatBuffer(PsfBufferSlot::RingPeak);
        if (invScaleBuffer == nil || baseKernel == nil || ringSums == nil || ringCounts == nil || meanKernel == nil ||
            shapedKernel == nil || structureKernel == nil || ringEnergy == nil || ringPeak == nil) {
            return false;
        }
        const float gain = 1.0f + std::max(0.0, params.anisotropyEmphasis) * 4.0f;
        gainBuffer = makeSharedBufferWithBytes(device, &gain, sizeof(float), error);
        if (gainBuffer == nil) {
            return false;
        }

        std::memset(ringSums.contents, 0, static_cast<NSUInteger>(shapeParams.radiusMax + 1) * sizeof(float));
        std::memset(ringCounts.contents, 0, static_cast<NSUInteger>(shapeParams.radiusMax + 1) * sizeof(uint32_t));
        std::memset(ringEnergy.contents, 0, static_cast<NSUInteger>(shapeParams.radiusMax + 1) * sizeof(float));
        std::memset(ringPeak.contents, 0, static_cast<NSUInteger>(shapeParams.radiusMax + 1) * sizeof(float));

        const auto finalizeStart = std::chrono::steady_clock::now();
        {
            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setBuffer:rawPsfBuffer offset:0 atIndex:0];
            [encoder setBuffer:baseKernel offset:0 atIndex:1];
            [encoder setBuffer:invScaleBuffer offset:0 atIndex:2];
            [encoder setBuffer:resampleParamsBuffer offset:0 atIndex:3];
            dispatch2d(encoder, pipelines->resampleRawPsf, kernelSize, kernelSize);
            if (!encodeNormalizeScalarBufferMetal(encoder, device, pipelines, psfContext, baseKernel, kernelCount, error)) {
                [encoder endEncoding];
                return false;
            }
            [encoder setBuffer:baseKernel offset:0 atIndex:0];
            [encoder setBuffer:ringSums offset:0 atIndex:1];
            [encoder setBuffer:ringCounts offset:0 atIndex:2];
            [encoder setBuffer:shapeParamsBuffer offset:0 atIndex:3];
            dispatch1d256(encoder, pipelines->ringSumCount, static_cast<NSUInteger>(shapeParams.radiusMax + 1));
            [encoder setBuffer:ringSums offset:0 atIndex:0];
            [encoder setBuffer:ringCounts offset:0 atIndex:1];
            [encoder setBuffer:meanKernel offset:0 atIndex:2];
            [encoder setBuffer:shapeParamsBuffer offset:0 atIndex:3];
            dispatch2d(encoder, pipelines->expandMean, kernelSize, kernelSize);
            if (!encodeNormalizeScalarBufferMetal(encoder, device, pipelines, psfContext, meanKernel, kernelCount, error)) {
                [encoder endEncoding];
                return false;
            }
            [encoder setBuffer:baseKernel offset:0 atIndex:0];
            [encoder setBuffer:meanKernel offset:0 atIndex:1];
            [encoder setBuffer:shapedKernel offset:0 atIndex:2];
            [encoder setBuffer:gainBuffer offset:0 atIndex:3];
            [encoder setBuffer:shapeParamsBuffer offset:0 atIndex:4];
            dispatch2d(encoder, pipelines->reshapeKernel, kernelSize, kernelSize);
            if (!encodeNormalizeScalarBufferMetal(encoder, device, pipelines, psfContext, shapedKernel, kernelCount, error)) {
                [encoder endEncoding];
                return false;
            }
            [encoder setBuffer:shapedKernel offset:0 atIndex:0];
            [encoder setBuffer:meanKernel offset:0 atIndex:1];
            [encoder setBuffer:structureKernel offset:0 atIndex:2];
            [encoder setBuffer:shapeParamsBuffer offset:0 atIndex:3];
            dispatch2d(encoder, pipelines->positiveResidual, kernelSize, kernelSize);
            if (!encodeNormalizeScalarBufferMetal(encoder, device, pipelines, psfContext, structureKernel, kernelCount, error)) {
                [encoder endEncoding];
                return false;
            }
            [encoder setBuffer:shapedKernel offset:0 atIndex:0];
            [encoder setBuffer:ringEnergy offset:0 atIndex:1];
            [encoder setBuffer:ringPeak offset:0 atIndex:2];
            [encoder setBuffer:shapeParamsBuffer offset:0 atIndex:3];
            dispatch1d256(encoder, pipelines->ringEnergyPeak, static_cast<NSUInteger>(shapeParams.radiusMax + 1));
            [encoder endEncoding];

            if (!commitAndWait(commandBuffer, error)) return false;
        }
        if (LensDiffTimingEnabled()) {
            LogLensDiffTimingStage(
                "metal-psf-wavelength-finalize",
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - finalizeStart)
                    .count(),
                "nm=" + std::to_string(static_cast<int>(std::lround(wavelength))));
        }
        const auto readbackStart = std::chrono::steady_clock::now();
        const float* ringEnergyValues = static_cast<const float*>(ringEnergy.contents);
        const float* ringPeakValues = static_cast<const float*>(ringPeak.contents);
        float totalEnergy = 0.0f;
        float globalPeak = 0.0f;
        for (int r = 0; r <= shapeParams.radiusMax; ++r) {
            totalEnergy += ringEnergyValues[static_cast<std::size_t>(r)];
            globalPeak = std::max(globalPeak, ringPeakValues[static_cast<std::size_t>(r)]);
        }
        const int effectiveRadius =
            paddedAdaptiveSupportRadiusHost(
                estimateAdaptiveSupportRadiusMetal(ringEnergyValues, ringPeakValues, std::min(shapeParams.radiusMax, std::max(4, ResolveLensDiffMaxKernelRadiusPx(params))),
                                                   totalEnergy, globalPeak),
                std::max(4, ResolveLensDiffMaxKernelRadiusPx(params)));
        if (LensDiffTimingEnabled()) {
            LogLensDiffTimingStage(
                "metal-psf-support-radius-readback",
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - readbackStart)
                    .count(),
                "nm=" + std::to_string(static_cast<int>(std::lround(wavelength))) +
                    ",radius=" + std::to_string(effectiveRadius));
        }
        const int croppedSize = effectiveRadius * 2 + 1;
        const NSUInteger croppedCount = static_cast<NSUInteger>(croppedSize) * static_cast<NSUInteger>(croppedSize);
        const ConvolutionParamsGpu cropParams {kernelSize, 0, croppedSize, effectiveRadius};
        id<MTLBuffer> cropParamsBuffer = makeParamBuffer(device, cropParams, error);
        id<MTLBuffer> coreCropped = psfContext->acquire(PsfBufferSlot::CropCore, croppedCount * sizeof(float), error);
        id<MTLBuffer> fullCropped = psfContext->acquire(PsfBufferSlot::CropFull, croppedCount * sizeof(float), error);
        id<MTLBuffer> structureCropped = psfContext->acquire(PsfBufferSlot::CropStructure, croppedCount * sizeof(float), error);
        if (cropParamsBuffer == nil || coreCropped == nil || fullCropped == nil || structureCropped == nil) {
            return false;
        }
        const auto cropStart = std::chrono::steady_clock::now();
        {
            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            auto encodeCrop = [&](id<MTLBuffer> src, id<MTLBuffer> dst) {
                [encoder setBuffer:src offset:0 atIndex:0];
                [encoder setBuffer:dst offset:0 atIndex:1];
                [encoder setBuffer:cropParamsBuffer offset:0 atIndex:2];
                dispatch2d(encoder, pipelines->cropKernel, croppedSize, croppedSize);
            };
            encodeCrop(meanKernel, coreCropped);
            encodeCrop(shapedKernel, fullCropped);
            encodeCrop(structureKernel, structureCropped);
            if (!encodeNormalizeScalarBufferMetal(encoder, device, pipelines, psfContext, coreCropped, croppedCount, error) ||
                !encodeNormalizeScalarBufferMetal(encoder, device, pipelines, psfContext, fullCropped, croppedCount, error) ||
                !encodeNormalizeScalarBufferMetal(encoder, device, pipelines, psfContext, structureCropped, croppedCount, error)) {
                [encoder endEncoding];
                return false;
            }
            [encoder endEncoding];
            if (!commitAndWait(commandBuffer, error)) {
                return false;
            }
        }
        if (LensDiffTimingEnabled()) {
            LogLensDiffTimingStage(
                "metal-psf-crop-normalize",
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - cropStart)
                    .count(),
                "nm=" + std::to_string(static_cast<int>(std::lround(wavelength))) +
                    ",cropped=" + std::to_string(croppedSize));
        }

        LensDiffPsfBin bin {};
        bin.wavelengthNm = wavelength;
        bin.core.size = croppedSize;
        bin.core.values.assign(static_cast<float*>(coreCropped.contents), static_cast<float*>(coreCropped.contents) + croppedCount);
        bin.full.size = croppedSize;
        bin.full.values.assign(static_cast<float*>(fullCropped.contents), static_cast<float*>(fullCropped.contents) + croppedCount);
        bin.structure.size = croppedSize;
        bin.structure.values.assign(static_cast<float*>(structureCropped.contents), static_cast<float*>(structureCropped.contents) + croppedCount);
        applySupportBoundaryTaperHost(bin.core, effectiveRadius);
        normalizeKernelHost(bin.core);
        applySupportBoundaryTaperHost(bin.full, effectiveRadius);
        normalizeKernelHost(bin.full);
        applySupportBoundaryTaperHost(bin.structure, effectiveRadius);
        normalizeKernelHost(bin.structure);
        cache->supportRadiusPx = std::max(cache->supportRadiusPx, effectiveRadius);
        cache->bins.push_back(std::move(bin));
        if (LensDiffTimingEnabled()) {
            LogLensDiffTimingStage(
                "metal-psf-wavelength-total",
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - wavelengthStart)
                    .count(),
                "nm=" + std::to_string(static_cast<int>(std::lround(wavelength))));
        }
    }

    return true;
}

bool buildPsfBankGlobalOnlyMetal(const LensDiffParams& params,
                                 LensDiffPsfBankCache& cache,
                                 id<MTLDevice> device,
                                 id<MTLCommandQueue> queue,
                                 PipelineBundle* pipelines,
                                 MetalPsfBuildContext* buildContext,
                                 std::string* error) {
    const LensDiffPsfBankKey key = MakeLensDiffPsfBankKey(params);
    if (cache.valid && cache.key == key) {
        return true;
    }
    LensDiffScopedTimer timer("metal-psf-bank-global");
    if (device == nil || queue == nil || pipelines == nullptr) {
        if (error) *error = "metal-psf-cache-missing-device-or-queue";
        return false;
    }

    const int pupilSize = GetLensDiffEffectivePupilResolution(params.pupilResolution);
    const int maxKernelRadiusPx = std::max(4, ResolveLensDiffMaxKernelRadiusPx(params));
    const int rawPsfSize = ChooseLensDiffRawPsfSize(pupilSize, maxKernelRadiusPx);
    std::vector<float> pupil;
    std::vector<float> phaseWaves;
    std::vector<float> rawPsf;
    if (!buildOpticalPrepAndRawPsfMetal(params, pupilSize, rawPsfSize, device, queue, pipelines, &pupil, &phaseWaves, &rawPsf, error)) {
        return false;
    }
    const std::shared_ptr<const std::vector<float>> referenceRawPsf = GetLensDiffReferenceRawPsfCached(pupilSize, rawPsfSize);
    const float referenceFirstZeroRadius = std::max(1.0f, EstimateLensDiffFirstMinimumRadius(*referenceRawPsf, rawPsfSize));
    const float scaleBase = static_cast<float>(std::max(1.0, ResolveLensDiffDiffractionScalePx(params))) / referenceFirstZeroRadius;
    const std::vector<float> wavelengths = GetLensDiffSpectralWavelengths(params.spectralMode);

    if (!finalizePsfBankMetal(device, queue, pipelines, params, key, pupil, phaseWaves, pupilSize, rawPsf, rawPsfSize, wavelengths, scaleBase, &cache, buildContext, error)) {
        return false;
    }
    if (!cache.valid || !(cache.key == key)) {
        if (error) *error = "metal-psf-cache-build-failed";
        return false;
    }
    return true;
}

bool ensurePsfBankMetal(const LensDiffParams& params,
                        LensDiffPsfBankCache& cache,
                        id<MTLDevice> device,
                        id<MTLCommandQueue> queue,
                        PipelineBundle* pipelines,
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

    MetalPsfBuildContext buildContext {};
    buildContext.device = device;
    if (!buildPsfBankGlobalOnlyMetal(params, cache, device, queue, pipelines, &buildContext, error)) {
        return false;
    }
    if (!needFieldZones) {
        cache.fieldGridSize = 0;
        cache.fieldKey = {};
        cache.fieldZones.clear();
        return true;
    }

    LensDiffScopedTimer timer("metal-field-zones");
    cache.fieldGridSize = 3;
    cache.fieldKey = fieldKey;
    cache.fieldZones.clear();
    cache.fieldZones.reserve(9);
    for (int zoneY = 0; zoneY < 3; ++zoneY) {
        for (int zoneX = 0; zoneX < 3; ++zoneX) {
            const float normalizedX = static_cast<float>(zoneX - 1);
            const float normalizedY = static_cast<float>(zoneY - 1);
            LensDiffFieldZoneCache zone {};
            zone.zoneX = zoneX;
            zone.zoneY = zoneY;
            zone.normalizedX = normalizedX;
            zone.normalizedY = normalizedY;
            zone.radialNorm = std::min(1.0f, std::sqrt(normalizedX * normalizedX + normalizedY * normalizedY) / std::sqrt(2.0f));
            zone.resolvedParams = ResolveLensDiffFieldZoneParams(params, normalizedX, normalizedY);

            LensDiffPsfBankCache zoneCache {};
            if (!buildPsfBankGlobalOnlyMetal(zone.resolvedParams, zoneCache, device, queue, pipelines, &buildContext, error)) {
                return false;
            }
            zone.key = zoneCache.key;
            zone.bins = std::move(zoneCache.bins);
            zone.supportRadiusPx = zoneCache.supportRadiusPx;
            zone.pupilDisplaySize = zoneCache.pupilDisplaySize;
            zone.pupilDisplay = std::move(zoneCache.pupilDisplay);
            zone.phaseDisplaySize = zoneCache.phaseDisplaySize;
            zone.phaseDisplay = std::move(zoneCache.phaseDisplay);
            cache.fieldZones.push_back(std::move(zone));
        }
    }
    return true;
}

} // namespace

bool RunLensDiffMetal(const LensDiffRenderRequest& request,
                      const LensDiffParams& params,
                      LensDiffPsfBankCache& cache,
                      std::string* error) {
    if (!request.hostEnabledMetalRender || request.metalCommandQueue == nullptr) {
        if (error) *error = "metal-render-not-enabled-by-host";
        return false;
    }
    if (!validateRowBytes(request.src) || !validateRowBytes(request.dst)) {
        if (error) *error = "invalid-metal-image-layout";
        return false;
    }

    bool renderScopeResult = false;
    {
    bool renderSucceeded = false;
    struct MetalRenderScopeLogger {
        bool* success = nullptr;
        std::string* error = nullptr;

        ~MetalRenderScopeLogger() {
            std::ostringstream note;
            note << "success=" << ((success != nullptr && *success) ? "true" : "false");
            if (error != nullptr && !error->empty()) {
                note << " error=" << *error;
            }
            LogLensDiffDiagnosticEvent("metal-render-exit", note.str());
        }
    } renderScope {&renderSucceeded, error};

    LogLensDiffDiagnosticEvent("metal-render-enter");
    id<MTLCommandQueue> hostQueue = (__bridge id<MTLCommandQueue>)request.metalCommandQueue;
    id<MTLBuffer> srcBuffer = (__bridge id<MTLBuffer>)request.src.data;
    id<MTLBuffer> dstBuffer = (__bridge id<MTLBuffer>)request.dst.data;
    if (hostQueue == nil || srcBuffer == nil || dstBuffer == nil) {
        if (error) *error = "missing-metal-queue-or-buffer";
        return false;
    }

    id<MTLDevice> device = hostQueue.device;
    // Resolve provides the device-facing queue, but LensDiff submits many small synchronous command
    // buffers during PSF prep and debug/composite staging. Keep that work on our own queue so we do
    // not block host-managed queue scheduling while still using the same device and shared buffers.
    id<MTLCommandQueue> queue = ensureWorkQueue(device, error);
    if (queue == nil) {
        return false;
    }
    {
        std::ostringstream queueNote;
        queueNote << "hostQueue=" << (__bridge const void*)hostQueue
                  << " workQueue=" << (__bridge const void*)queue;
        LogLensDiffDiagnosticEvent("metal-queues-ready", queueNote.str());
    }
    PipelineBundle* pipelines = ensurePipelines(device, error);
    if (pipelines == nullptr) {
        return false;
    }
    LogLensDiffDiagnosticEvent("metal-pipelines-ready");

    struct MetalRenderTimingBreakdown {
        double psfBankMs = 0.0;
        double sourceFftMs = 0.0;
        double kernelFftMs = 0.0;
        double convolutionMs = 0.0;
        double fieldZonesMs = 0.0;
        double compositeOutputMs = 0.0;
        int commandBufferCount = 0;
        int waitCount = 0;
        int fieldZoneBatchDepth = 0;
        int rgbSourceCacheHits = 0;
        int rgbSourceCacheMisses = 0;
        int scalarSourceCacheHits = 0;
        int scalarSourceCacheMisses = 0;
        int kernelCacheHits = 0;
        int kernelCacheMisses = 0;
    } timing {};
    std::string executionModeNote = "mode=unknown";

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
            "metal-stage-psf-bank",
            timing.psfBankMs,
            executionModeNote);
        LogLensDiffTimingStage(
            "metal-stage-source-fft",
            timing.sourceFftMs,
            "rgbHits=" + std::to_string(timing.rgbSourceCacheHits) +
                ",rgbMisses=" + std::to_string(timing.rgbSourceCacheMisses) +
                ",scalarHits=" + std::to_string(timing.scalarSourceCacheHits) +
                ",scalarMisses=" + std::to_string(timing.scalarSourceCacheMisses));
        LogLensDiffTimingStage(
            "metal-stage-kernel-fft",
            timing.kernelFftMs,
            "hits=" + std::to_string(timing.kernelCacheHits) +
                ",misses=" + std::to_string(timing.kernelCacheMisses));
        LogLensDiffTimingStage("metal-stage-convolution", timing.convolutionMs);
        LogLensDiffTimingStage(
            "metal-stage-field-zones",
            timing.fieldZonesMs,
            "zones=" + std::to_string(static_cast<int>(cache.fieldZones.size())) +
                ",batchDepth=" + std::to_string(timing.fieldZoneBatchDepth));
        LogLensDiffTimingStage(
            "metal-stage-composite-output",
            timing.compositeOutputMs,
            "commandBuffers=" + std::to_string(timing.commandBufferCount) +
                ",waits=" + std::to_string(timing.waitCount));
    };

    const int nativeWidth = request.src.bounds.width();
    const int nativeHeight = request.src.bounds.height();
    const NSUInteger nativePixelCount = static_cast<NSUInteger>(nativeWidth) * static_cast<NSUInteger>(nativeHeight);
    const NSUInteger nativeRgbaBytes = nativePixelCount * 4u * sizeof(float);
    const NSUInteger nativeScalarBytes = nativePixelCount * sizeof(float);
    const double workingScale = ResolveLensDiffEffectWorkingScale(params);
    const bool resolutionAwareActive = params.resolutionAware && std::abs(workingScale - 1.0) > 1e-6;
    const int width = resolutionAwareActive ? std::max(1, static_cast<int>(std::lround(nativeWidth * workingScale))) : nativeWidth;
    const int height = resolutionAwareActive ? std::max(1, static_cast<int>(std::lround(nativeHeight * workingScale))) : nativeHeight;
    const NSUInteger pixelCount = static_cast<NSUInteger>(width) * static_cast<NSUInteger>(height);
    const NSUInteger rgbaBytes = pixelCount * 4u * sizeof(float);
    const NSUInteger scalarBytes = pixelCount * sizeof(float);
    const bool splitMode = params.lookMode == LensDiffLookMode::Split;
    const bool needCore = splitMode || params.debugView == LensDiffDebugView::Core;
    const bool needStructure = splitMode || params.debugView == LensDiffDebugView::Structure;
    const bool requestedLegacySync = LensDiffMetalLegacySyncEnabled();
    const bool allowFastField = LensDiffMetalFastFieldEnabled();
    const bool allowFastSplit = LensDiffMetalFastSplitEnabled();
    const bool allowFastResolutionAware = LensDiffMetalFastResolutionAwareEnabled();
    const bool fieldRequested = HasLensDiffFieldPhase(params);
    // Mirror the stabilized CUDA rollout: keep the optimized GPU path as the default only for the
    // ordinary global physical render, and require explicit opt-in before using it for field,
    // split, or resolution-aware requests that are more sensitive to memory churn and parity drift.
    const bool fastPathAllowed =
        !requestedLegacySync &&
        (!resolutionAwareActive || allowFastResolutionAware) &&
        ((!fieldRequested && !splitMode) ||
         (!fieldRequested && splitMode && allowFastSplit) ||
         (fieldRequested && allowFastField));
    const bool legacySync = !fastPathAllowed;
    const bool heapsRequested = LensDiffMetalHeapsRequested();
    const bool heapsForceEnabled = LensDiffMetalHeapsForceEnabled();
    const bool vkfftRequested = LensDiffMetalVkFFTRequested();
    const bool heapsSafeForMode = !resolutionAwareActive || heapsForceEnabled;
    const bool heapsEnabled = heapsRequested && !legacySync && heapsSafeForMode;
    const bool vkfftEnabled = vkfftRequested && !legacySync;
    LensDiffMetalRuntimeOverrideScope runtimeOverride(heapsEnabled, vkfftEnabled);
    if (!timeCall(timing.psfBankMs, [&] { return ensurePsfBankMetal(params, cache, device, queue, pipelines, error); })) {
        return false;
    }
    const FieldZoneBatchPlan fieldPlan = buildFieldZoneBatchPlan(cache);
    executionModeNote =
        "mode=" + std::string(legacySync ? "stable" : "fast") +
        ",vkfftRequested=" + std::to_string(vkfftRequested ? 1 : 0) +
        ",vkfftEffective=" + std::to_string(vkfftEnabled ? 1 : 0) +
        ",heapsRequested=" + std::to_string(heapsRequested ? 1 : 0) +
        ",heapsEffective=" + std::to_string(heapsEnabled ? 1 : 0) +
        ",heapsForce=" + std::to_string(heapsForceEnabled ? 1 : 0) +
        ",requestedLegacy=" + std::to_string(requestedLegacySync ? 1 : 0) +
        ",field=" + std::to_string(cache.fieldZones.empty() ? 0 : 1) +
        ",canonical3x3=" + std::to_string(fieldPlan.canonical3x3 ? 1 : 0) +
        ",split=" + std::to_string(splitMode ? 1 : 0) +
        ",resolutionAware=" + std::to_string(resolutionAwareActive ? 1 : 0) +
        ",fastField=" + std::to_string(allowFastField ? 1 : 0) +
        ",fastSplit=" + std::to_string(allowFastSplit ? 1 : 0) +
        ",fastResolutionAware=" + std::to_string(allowFastResolutionAware ? 1 : 0);
    LogLensDiffDiagnosticEvent("metal-render-mode", executionModeNote);
    MetalScratchCache scratchCache {};
    MetalRenderTimingCounters renderCounters {};
    MetalRenderContext renderContext {device, queue, pipelines, &scratchCache, &renderCounters, error};

    auto makeSizedBuffer = [&](NSUInteger byteCount) { return makeSharedBuffer(device, byteCount, error); };
    auto makeRgbaBuffer = [&]() { return makeSizedBuffer(rgbaBytes); };
    auto makeScalarBuffer = [&]() { return makeSizedBuffer(scalarBytes); };
    auto makeNativeRgbaBuffer = [&]() { return makeSizedBuffer(nativeRgbaBytes); };
    auto makeNativeScalarBuffer = [&]() { return makeSizedBuffer(nativeScalarBytes); };
    auto zeroBuffer = [&](id<MTLBuffer> buffer, NSUInteger byteCount) -> bool {
        if (buffer == nil) {
            return false;
        }
        std::memset(buffer.contents, 0, byteCount);
        return true;
    };
    auto zeroRgbaBuffer = [&](id<MTLBuffer> buffer) -> bool { return zeroBuffer(buffer, rgbaBytes); };
    auto zeroNativeRgbaBuffer = [&](id<MTLBuffer> buffer) -> bool { return zeroBuffer(buffer, nativeRgbaBytes); };
    auto encode2d = [&](id<MTLComputePipelineState> pipeline,
                        NSArray<id<MTLBuffer>>* buffers,
                        int dispatchWidth,
                        int dispatchHeight) -> bool {
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        ++renderCounters.commandBufferCount;
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        for (NSUInteger i = 0; i < buffers.count; ++i) {
            [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        }
        dispatch2d(encoder, pipeline, dispatchWidth, dispatchHeight);
        [encoder endEncoding];
        return commitAndWaitCounted(&renderCounters, commandBuffer, error);
    };
    auto encode2dDispatch = [&](id<MTLComputeCommandEncoder> encoder,
                                id<MTLComputePipelineState> pipeline,
                                NSArray<id<MTLBuffer>>* buffers,
                                int dispatchWidth,
                                int dispatchHeight) -> bool {
        if (encoder == nil || pipeline == nil || buffers == nil) {
            if (error) *error = "metal-invalid-2d-dispatch";
            return false;
        }
        for (NSUInteger i = 0; i < buffers.count; ++i) {
            [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        }
        dispatch2d(encoder, pipeline, dispatchWidth, dispatchHeight);
        return true;
    };
    auto encode1dDispatch = [&](id<MTLComputeCommandEncoder> encoder,
                                id<MTLComputePipelineState> pipeline,
                                NSArray<id<MTLBuffer>>* buffers,
                                NSUInteger count) -> bool {
        if (encoder == nil || pipeline == nil || buffers == nil) {
            if (error) *error = "metal-invalid-1d-dispatch";
            return false;
        }
        for (NSUInteger i = 0; i < buffers.count; ++i) {
            [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        }
        dispatch1d256(encoder, pipeline, count);
        return true;
    };
    auto encodeReduceEnergyToScalar = [&](id<MTLComputeCommandEncoder> encoder,
                                          id<MTLBuffer> imageBuffer,
                                          id<MTLBuffer>* outScalar) -> bool {
        if (encoder == nil || imageBuffer == nil || outScalar == nullptr) {
            if (error) *error = "metal-invalid-reduce-energy";
            return false;
        }
        const NSUInteger groups = ceilDiv(pixelCount, static_cast<NSUInteger>(256));
        id<MTLBuffer> current = makeSharedBuffer(device, groups * sizeof(float), error);
        const ReductionParamsGpu paramsGpu {width, height};
        id<MTLBuffer> paramsBuffer = makeParamBuffer(device, paramsGpu, error);
        if (current == nil || paramsBuffer == nil) {
            return false;
        }
        [encoder setBuffer:imageBuffer offset:0 atIndex:0];
        [encoder setBuffer:current offset:0 atIndex:1];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
        dispatch1d256(encoder, pipelines->reduceLuma, pixelCount);

        NSUInteger currentCount = groups;
        while (currentCount > 1u) {
            const NSUInteger nextCount = ceilDiv(currentCount, static_cast<NSUInteger>(256));
            id<MTLBuffer> next = makeSharedBuffer(device, nextCount * sizeof(float), error);
            const ScalarReduceParamsGpu scalarParams {static_cast<int>(currentCount)};
            id<MTLBuffer> scalarParamsBuffer = makeParamBuffer(device, scalarParams, error);
            if (next == nil || scalarParamsBuffer == nil) {
                return false;
            }
            [encoder setBuffer:current offset:0 atIndex:0];
            [encoder setBuffer:next offset:0 atIndex:1];
            [encoder setBuffer:scalarParamsBuffer offset:0 atIndex:2];
            dispatch1d256(encoder, pipelines->reduceFloat, currentCount);
            current = next;
            currentCount = nextCount;
        }

        *outScalar = current;
        return true;
    };
    auto encodeResampleRgbaDispatch = [&](id<MTLComputeCommandEncoder> encoder,
                                          id<MTLBuffer> srcImage,
                                          id<MTLBuffer> dstImage,
                                          int srcWidth,
                                          int srcHeight,
                                          int dstWidth,
                                          int dstHeight) -> bool {
        const ResampleParamsGpu paramsGpu {srcWidth, srcHeight, dstWidth, dstHeight};
        id<MTLBuffer> paramsBuffer = makeParamBuffer(device, paramsGpu, error);
        return paramsBuffer != nil &&
               encode2dDispatch(encoder, pipelines->resampleRgba, @[srcImage, dstImage, paramsBuffer], dstWidth, dstHeight);
    };
    auto encodeResampleGrayDispatch = [&](id<MTLComputeCommandEncoder> encoder,
                                          id<MTLBuffer> srcImage,
                                          id<MTLBuffer> dstImage,
                                          int srcWidth,
                                          int srcHeight,
                                          int dstWidth,
                                          int dstHeight) -> bool {
        const ResampleParamsGpu paramsGpu {srcWidth, srcHeight, dstWidth, dstHeight};
        id<MTLBuffer> paramsBuffer = makeParamBuffer(device, paramsGpu, error);
        return paramsBuffer != nil &&
               encode2dDispatch(encoder, pipelines->resampleGray, @[srcImage, dstImage, paramsBuffer], dstWidth, dstHeight);
    };
    const bool staticDebug = params.debugView == LensDiffDebugView::Pupil ||
                             params.debugView == LensDiffDebugView::Psf ||
                             params.debugView == LensDiffDebugView::Otf ||
                             params.debugView == LensDiffDebugView::Phase ||
                             params.debugView == LensDiffDebugView::PhaseEdge ||
                             params.debugView == LensDiffDebugView::FieldPsf ||
                             params.debugView == LensDiffDebugView::ChromaticSplit;
    if (staticDebug) {
        const std::vector<float> debugRgba = buildStaticDebugRgba(params, cache, nativeWidth, nativeHeight);
        id<MTLBuffer> debugBuffer = makeSharedBufferWithBytes(device, debugRgba.data(), nativeRgbaBytes, error);
        const LensDiffImageRect outputRect = intersectRect(request.renderWindow, request.dst.bounds);
        if (outputRect.width() <= 0 || outputRect.height() <= 0) {
            renderSucceeded = true;
            LogLensDiffDiagnosticEvent("metal-output-empty", "static-debug-renderWindow-outside-dst");
            return true;
        }
        const OutputParamsGpu outParams {
            nativeWidth, nativeHeight, request.src.bounds.x1, request.src.bounds.y1,
            request.dst.bounds.x1, request.dst.bounds.y1,
            request.dst.bounds.x2, request.dst.bounds.y2,
            outputRect.x1, outputRect.y1,
            outputRect.x2, outputRect.y2,
            rowFloats(request.dst), 0, 0, 0};
        id<MTLBuffer> outParamsBuffer = makeParamBuffer(device, outParams, error);
        if (debugBuffer == nil || outParamsBuffer == nil) {
            return false;
        }
        return encode2d(pipelines->packRgb, @[debugBuffer, debugBuffer, dstBuffer, outParamsBuffer],
                        outputRect.width(), outputRect.height());
    }

    id<MTLBuffer> linearSrcNative = makeNativeRgbaBuffer();
    id<MTLBuffer> linearSrc = resolutionAwareActive ? makeRgbaBuffer() : linearSrcNative;
    id<MTLBuffer> redistributed = makeRgbaBuffer();
    id<MTLBuffer> driver = makeScalarBuffer();
    id<MTLBuffer> mask = makeScalarBuffer();
    if (linearSrcNative == nil || linearSrc == nil || redistributed == nil || driver == nil || mask == nil) {
        return false;
    }

    const DecodeParamsGpu decodeParams {
        nativeWidth, nativeHeight, rowFloats(request.src), static_cast<int>(params.inputTransfer)};
    id<MTLBuffer> decodeParamsBuffer = makeParamBuffer(device, decodeParams, error);
    const PrepareParamsGpu prepareParams {
        width, height,
        params.extractionMode == LensDiffExtractionMode::Luma ? 1 : 0,
        static_cast<float>(params.threshold), static_cast<float>(std::max(0.01, params.softnessStops)),
        static_cast<float>(params.pointEmphasis), static_cast<float>(params.corePreserve)};
    id<MTLBuffer> prepareParamsBuffer = makeParamBuffer(device, prepareParams, error);
    id<MTLBuffer> linearResampleParamsBuffer = nil;
    if (resolutionAwareActive) {
        const ResampleParamsGpu linearResampleParams {nativeWidth, nativeHeight, width, height};
        linearResampleParamsBuffer = makeParamBuffer(device, linearResampleParams, error);
    }
    if (decodeParamsBuffer == nil || prepareParamsBuffer == nil ||
        (resolutionAwareActive && linearResampleParamsBuffer == nil)) {
        return false;
    }
    {
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        ++renderCounters.commandBufferCount;
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        const bool ok = encode2dDispatch(encoder,
                                         pipelines->decodeSource,
                                         @[srcBuffer, linearSrcNative, decodeParamsBuffer],
                                         nativeWidth,
                                         nativeHeight) &&
                        (!resolutionAwareActive ||
                         encode2dDispatch(encoder,
                                          pipelines->resampleRgba,
                                          @[linearSrcNative, linearSrc, linearResampleParamsBuffer],
                                          width,
                                          height)) &&
                        encode2dDispatch(encoder,
                                         pipelines->prepareFromLinear,
                                         @[linearSrc, redistributed, driver, mask, prepareParamsBuffer],
                                         width,
                                         height);
        [encoder endEncoding];
        if (!ok || !commitAndWaitCounted(&renderCounters, commandBuffer, error)) {
            return false;
        }
    }
    const float redistributionScale = 1.0f - static_cast<float>(clampValue(params.corePreserve, 0.0, 1.0));
    const float protectedCoreFraction = std::max(
        kMinimumSelectedCoreFloor,
        static_cast<float>(clampValue(params.corePreserve, 0.0, 1.0)));
    const float maxRedistributedSubtractScale = redistributionScale > 1e-6f
        ? (1.0f - protectedCoreFraction) / redistributionScale
        : 0.0f;

    struct RgbSourceSpectraMetal {
        int paddedSize = 0;
        id<MTLBuffer> stack = nil;
    };

    std::unordered_map<std::string, id<MTLBuffer>> kernelSpectrumCache;
    std::unordered_map<std::string, id<MTLBuffer>> rgbSourceSpectrumCache;
    std::unordered_map<std::string, id<MTLBuffer>> scalarSourceSpectrumCache;
    std::unordered_map<std::string, FieldZoneKernelStacks> fieldKernelStackCache;
    std::unordered_map<std::string, id<MTLBuffer>> fieldReplicatedSpectrumCache;

    auto paddedFftSizeForKernel = [&](int kernelSize) {
        return nextPowerOfTwo(std::max(width + kernelSize - 1, height + kernelSize - 1));
    };
    auto sourceSpectrumCacheKey = [&](id<MTLBuffer> source, int paddedSize) {
        return std::to_string(reinterpret_cast<std::uintptr_t>(source)) + ":" + std::to_string(paddedSize);
    };
    auto kernelSpectrumCacheKey = [&](const LensDiffKernel& kernel, int paddedSize) {
        return std::to_string(paddedSize) + ":" + std::to_string(kernel.size) + ":" + std::to_string(hashKernelValues(kernel));
    };
    auto fieldStackCacheKey = [&](int paddedSize, FieldEffectKind effectKind, int binCount, int repeatPerKernel) {
        return std::to_string(static_cast<int>(effectKind)) + ":" + std::to_string(paddedSize) + ":" +
               std::to_string(binCount) + ":" + std::to_string(repeatPerKernel);
    };
    auto getKernelSpectrum = [&](const LensDiffKernel& kernel, int paddedSize, id<MTLBuffer>* outSpectrum) -> bool {
        if (outSpectrum == nullptr) {
            return false;
        }
        const std::string key = kernelSpectrumCacheKey(kernel, paddedSize);
        auto it = kernelSpectrumCache.find(key);
        if (it != kernelSpectrumCache.end()) {
            ++timing.kernelCacheHits;
            *outSpectrum = it->second;
            return true;
        }
        id<MTLBuffer> spectrum = nil;
        ++timing.kernelCacheMisses;
        if (!timeCall(timing.kernelFftMs, [&] {
                return legacySync
                    ? makeKernelSpectrumMetal(device, queue, pipelines, kernel, paddedSize, &spectrum, error)
                    : encodeKernelSpectrumMetal(&renderContext, kernel, paddedSize, &spectrum);
            })) {
            return false;
        }
        kernelSpectrumCache.emplace(key, spectrum);
        *outSpectrum = spectrum;
        return true;
    };
    auto buildRgbSourceSpectra = [&](id<MTLBuffer> source,
                                     int paddedSize,
                                     RgbSourceSpectraMetal* out) -> bool {
        if (out == nullptr) {
            return false;
        }
        out->paddedSize = paddedSize;
        const std::string key = sourceSpectrumCacheKey(source, paddedSize);
        auto it = rgbSourceSpectrumCache.find(key);
        if (it != rgbSourceSpectrumCache.end()) {
            ++timing.rgbSourceCacheHits;
            out->stack = it->second;
            return true;
        }
        id<MTLBuffer> spectrum = nil;
        ++timing.rgbSourceCacheMisses;
        if (!timeCall(timing.sourceFftMs, [&] {
                return legacySync
                    ? makeRgbSpectraStackMetal(device, queue, pipelines, source, width, height, paddedSize, &spectrum, error)
                    : encodeRgbSpectraStackMetal(&renderContext, source, width, height, paddedSize, &spectrum);
            })) {
            return false;
        }
        rgbSourceSpectrumCache.emplace(key, spectrum);
        out->stack = spectrum;
        return true;
    };
    auto getScalarSourceSpectrum = [&](id<MTLBuffer> source,
                                       int paddedSize,
                                       id<MTLBuffer>* outSpectrum) -> bool {
        if (outSpectrum == nullptr) {
            return false;
        }
        const std::string key = sourceSpectrumCacheKey(source, paddedSize);
        auto it = scalarSourceSpectrumCache.find(key);
        if (it != scalarSourceSpectrumCache.end()) {
            ++timing.scalarSourceCacheHits;
            *outSpectrum = it->second;
            return true;
        }
        id<MTLBuffer> spectrum = nil;
        ++timing.scalarSourceCacheMisses;
        if (!timeCall(timing.sourceFftMs, [&] {
                return legacySync
                    ? makeScalarSpectrumMetal(device, queue, pipelines, source, width, height, paddedSize, &spectrum, error)
                    : encodeScalarSpectrumMetal(&renderContext, source, width, height, paddedSize, &spectrum);
            })) {
            return false;
        }
        scalarSourceSpectrumCache.emplace(key, spectrum);
        *outSpectrum = spectrum;
        return true;
    };
    auto kernelForEffect = [&](const LensDiffPsfBin& bin, FieldEffectKind effectKind) -> const LensDiffKernel& {
        switch (effectKind) {
            case FieldEffectKind::Core: return bin.core;
            case FieldEffectKind::Structure: return bin.structure;
            case FieldEffectKind::Full:
            default: return bin.full;
        }
    };
    auto getFieldKernelSpectrumStack = [&](FieldEffectKind effectKind,
                                           int paddedSize,
                                           int binIndex,
                                           int repeatPerKernel,
                                           id<MTLBuffer>* outSpectrumStack) -> bool {
        if (outSpectrumStack == nullptr || !fieldPlan.canonical3x3) {
            return false;
        }
        const std::string key = fieldStackCacheKey(paddedSize, effectKind, binIndex, repeatPerKernel);
        auto it = fieldKernelStackCache.find(key);
        if (it != fieldKernelStackCache.end()) {
            *outSpectrumStack = it->second.spectrumStack;
            return true;
        }
        std::vector<const LensDiffKernel*> kernels;
        kernels.reserve(fieldPlan.zones.size());
        for (const LensDiffFieldZoneCache* zone : fieldPlan.zones) {
            if (zone == nullptr || binIndex >= static_cast<int>(zone->bins.size())) {
                return false;
            }
            kernels.push_back(&kernelForEffect(zone->bins[static_cast<std::size_t>(binIndex)], effectKind));
        }
        id<MTLBuffer> stack = nil;
        if (!encodeBuildKernelSpectrumStackMetal(&renderContext, kernels, repeatPerKernel, paddedSize, &stack)) {
            return false;
        }
        FieldZoneKernelStacks stacks {};
        stacks.paddedSize = paddedSize;
        stacks.zoneCount = static_cast<int>(fieldPlan.zones.size());
        stacks.spectralBinCount = binIndex;
        stacks.effectKind = effectKind;
        stacks.spectrumStack = stack;
        fieldKernelStackCache.emplace(key, stacks);
        *outSpectrumStack = stack;
        return true;
    };
    auto getReplicatedSpectrumStack = [&](id<MTLBuffer> sourceSpectrum,
                                          const std::string& key,
                                          int srcBatchCount,
                                          int dstBatchCount,
                                          int paddedSize,
                                          id<MTLBuffer>* outSpectrumStack) -> bool {
        if (outSpectrumStack == nullptr) {
            return false;
        }
        auto it = fieldReplicatedSpectrumCache.find(key);
        if (it != fieldReplicatedSpectrumCache.end()) {
            *outSpectrumStack = it->second;
            return true;
        }
        id<MTLBuffer> stack = nil;
        if (!encodeReplicateComplexStackMetal(&renderContext, sourceSpectrum, srcBatchCount, dstBatchCount, paddedSize, &stack)) {
            return false;
        }
        fieldReplicatedSpectrumCache.emplace(key, stack);
        *outSpectrumStack = stack;
        return true;
    };
    auto convolveRgbSpectraToImage = [&](const RgbSourceSpectraMetal& sourceSpectra,
                                         const LensDiffKernel& kernel,
                                         id<MTLBuffer>* outBuffer) -> bool {
        id<MTLBuffer> kernelSpectrum = nil;
        if (!getKernelSpectrum(kernel, sourceSpectra.paddedSize, &kernelSpectrum) ||
            !timeCall(timing.convolutionMs, [&] {
                return legacySync
                    ? convolveRgbSpectrumStackToImageMetal(device, queue, pipelines, sourceSpectra.stack, kernelSpectrum, width, height,
                                                           sourceSpectra.paddedSize, outBuffer, error)
                    : encodeConvolveRgbSpectrumStackToImageMetal(&renderContext, sourceSpectra.stack, kernelSpectrum, width, height,
                                                                 sourceSpectra.paddedSize, outBuffer);
            })) {
            return false;
        }
        return true;
    };
    auto runRgbConvolutionFromSource = [&](id<MTLBuffer> source, const LensDiffKernel& kernel, id<MTLBuffer>* outBuffer) -> bool {
        RgbSourceSpectraMetal sourceSpectra {};
        return buildRgbSourceSpectra(source, paddedFftSizeForKernel(kernel.size), &sourceSpectra) &&
               convolveRgbSpectraToImage(sourceSpectra, kernel, outBuffer);
    };
    auto runShoulder = [&](id<MTLBuffer> image, float shoulder) -> bool {
        if (shoulder <= 0.0f) return true;
        const ShoulderParamsGpu shoulderParams {width, height, shoulder};
        id<MTLBuffer> shoulderBuffer = makeParamBuffer(device, shoulderParams, error);
        return shoulderBuffer != nil &&
               (legacySync
                    ? encode2d(pipelines->applyShoulder, @[image, shoulderBuffer], width, height)
                    : encode2dDispatch(renderContext.encoder, pipelines->applyShoulder, @[image, shoulderBuffer], width, height));
    };
    auto mapSpectralImageFromDriver = [&](id<MTLBuffer> driverSource,
                                          const std::vector<LensDiffPsfBin>& bins,
                                          bool useCore,
                                          bool useStructure,
                                          id<MTLBuffer>* outBuffer) -> bool {
        const LensDiffSpectrumConfig zoneSpectrumConfig = BuildLensDiffSpectrumConfig(params, bins);
        std::array<id<MTLBuffer>, kLensDiffMaxSpectralBins> planes {};
        planes.fill(nil);
        const int activeBins = std::min<int>(static_cast<int>(bins.size()), kLensDiffMaxSpectralBins);
        int maxKernelSize = 1;
        for (int i = 0; i < activeBins; ++i) {
            const LensDiffKernel* kernel = &bins[static_cast<std::size_t>(i)].full;
            if (useCore && !useStructure) kernel = &bins[static_cast<std::size_t>(i)].core;
            else if (!useCore && useStructure) kernel = &bins[static_cast<std::size_t>(i)].structure;
            maxKernelSize = std::max(maxKernelSize, kernel->size);
        }
        const int paddedSize = paddedFftSizeForKernel(maxKernelSize);
        id<MTLBuffer> driverSpectrum = nil;
        if (!getScalarSourceSpectrum(driverSource, paddedSize, &driverSpectrum)) {
            return false;
        }
        std::vector<id<MTLBuffer>> kernelSpectra;
        kernelSpectra.reserve(static_cast<std::size_t>(activeBins));
        for (int i = 0; i < activeBins; ++i) {
            const LensDiffKernel* kernel = &bins[static_cast<std::size_t>(i)].full;
            if (useCore && !useStructure) kernel = &bins[static_cast<std::size_t>(i)].core;
            else if (!useCore && useStructure) kernel = &bins[static_cast<std::size_t>(i)].structure;
            id<MTLBuffer> kernelSpectrum = nil;
            if (!getKernelSpectrum(*kernel, paddedSize, &kernelSpectrum)) {
                return false;
            }
            kernelSpectra.push_back(kernelSpectrum);
        }
        if (!timeCall(timing.convolutionMs, [&] {
                return legacySync
                    ? convolveScalarSpectrumToPlanesStackMetal(device, queue, pipelines, driverSpectrum, kernelSpectra, width, height,
                                                               paddedSize, &planes, error)
                    : encodeConvolveScalarSpectrumToPlanesStackMetal(&renderContext, driverSpectrum, kernelSpectra, width, height,
                                                                     paddedSize, &planes);
            })) {
            return false;
        }
        *outBuffer = makeRgbaBuffer();
        SpectralMapParamsGpu spectralParams {};
        spectralParams.width = width;
        spectralParams.height = height;
        spectralParams.binCount = zoneSpectrumConfig.binCount;
        spectralParams.chromaticAffectsLuma = params.chromaticAffectsLuma ? 1 : 0;
        spectralParams.spectrumForce = static_cast<float>(params.spectrumForce);
        spectralParams.spectrumSaturation = static_cast<float>(std::max(0.0, params.spectrumSaturation));
        std::copy(zoneSpectrumConfig.naturalMatrix.begin(), zoneSpectrumConfig.naturalMatrix.end(), spectralParams.naturalMatrix);
        std::copy(zoneSpectrumConfig.styleMatrix.begin(), zoneSpectrumConfig.styleMatrix.end(), spectralParams.styleMatrix);
        id<MTLBuffer> spectralParamsBuffer = makeParamBuffer(device, spectralParams, error);
        for (int i = activeBins; i < kLensDiffMaxSpectralBins; ++i) {
            planes[static_cast<std::size_t>(i)] = planes[0];
        }
        return *outBuffer != nil && spectralParamsBuffer != nil &&
               (legacySync
                    ? encode2d(pipelines->mapSpectral,
                               @[planes[0], planes[1], planes[2], planes[3], planes[4],
                                 planes[5], planes[6], planes[7], planes[8],
                                 *outBuffer, spectralParamsBuffer],
                               width, height)
                    : encode2dDispatch(renderContext.encoder,
                                       pipelines->mapSpectral,
                                       @[planes[0], planes[1], planes[2], planes[3], planes[4],
                                         planes[5], planes[6], planes[7], planes[8],
                                         *outBuffer, spectralParamsBuffer],
                                       width,
                                       height));
    };
    auto renderFromBins = [&](const std::vector<LensDiffPsfBin>& bins,
                              id<MTLBuffer>* outEffect,
                              id<MTLBuffer>* outCore,
                              id<MTLBuffer>* outStructure) -> bool {
        if (params.spectralMode == LensDiffSpectralMode::Mono) {
            int monoMaxKernelSize = bins.front().full.size;
            if (splitMode || outCore != nullptr) monoMaxKernelSize = std::max(monoMaxKernelSize, bins.front().core.size);
            if (splitMode || outStructure != nullptr) monoMaxKernelSize = std::max(monoMaxKernelSize, bins.front().structure.size);
            RgbSourceSpectraMetal redistributedSpectra {};
            if (!buildRgbSourceSpectra(redistributed, paddedFftSizeForKernel(monoMaxKernelSize), &redistributedSpectra)) {
                return false;
            }
            auto convolveRedistributed = [&](const LensDiffKernel& kernel, id<MTLBuffer>* outBuffer) {
                return convolveRgbSpectraToImage(redistributedSpectra, kernel, outBuffer);
            };
            id<MTLBuffer> localCore = nil;
            id<MTLBuffer> localStructure = nil;
            id<MTLBuffer> localFull = nil;
            if (splitMode) {
                if (!convolveRedistributed(bins.front().core, &localCore) ||
                    !convolveRedistributed(bins.front().structure, &localStructure) ||
                    !runShoulder(localCore, static_cast<float>(params.coreShoulder)) ||
                    !runShoulder(localStructure, static_cast<float>(params.structureShoulder))) {
                    return false;
                }
                *outEffect = makeRgbaBuffer();
                const CombineParamsGpu combineParams {width, height, static_cast<float>(std::max(0.0, params.coreGain)),
                                                      static_cast<float>(std::max(0.0, params.structureGain))};
                id<MTLBuffer> combineParamsBuffer = makeParamBuffer(device, combineParams, error);
                if (*outEffect == nil || combineParamsBuffer == nil ||
                    !(legacySync
                        ? encode2d(pipelines->combine, @[localCore, localStructure, *outEffect, combineParamsBuffer], width, height)
                        : encode2dDispatch(renderContext.encoder, pipelines->combine, @[localCore, localStructure, *outEffect, combineParamsBuffer], width, height))) {
                    return false;
                }
                if (outCore != nullptr) *outCore = localCore;
                if (outStructure != nullptr) *outStructure = localStructure;
                return true;
            }
            if (!convolveRedistributed(bins.front().full, &localFull)) {
                return false;
            }
            *outEffect = localFull;
            if (outCore != nullptr && !convolveRedistributed(bins.front().core, outCore)) return false;
            if (outStructure != nullptr && !convolveRedistributed(bins.front().structure, outStructure)) return false;
            return true;
        }

        id<MTLBuffer> localCore = nil;
        id<MTLBuffer> localStructure = nil;
        if (splitMode) {
            if (!mapSpectralImageFromDriver(driver, bins, true, false, &localCore) ||
                !mapSpectralImageFromDriver(driver, bins, false, true, &localStructure) ||
                !runShoulder(localCore, static_cast<float>(params.coreShoulder)) ||
                !runShoulder(localStructure, static_cast<float>(params.structureShoulder))) {
                return false;
            }
            *outEffect = makeRgbaBuffer();
            const CombineParamsGpu combineParams {width, height, static_cast<float>(std::max(0.0, params.coreGain)),
                                                  static_cast<float>(std::max(0.0, params.structureGain))};
            id<MTLBuffer> combineParamsBuffer = makeParamBuffer(device, combineParams, error);
            if (*outEffect == nil || combineParamsBuffer == nil ||
                !(legacySync
                    ? encode2d(pipelines->combine, @[localCore, localStructure, *outEffect, combineParamsBuffer], width, height)
                    : encode2dDispatch(renderContext.encoder, pipelines->combine, @[localCore, localStructure, *outEffect, combineParamsBuffer], width, height))) {
                return false;
            }
            if (outCore != nullptr) *outCore = localCore;
            if (outStructure != nullptr) *outStructure = localStructure;
            return true;
        }
        if (!mapSpectralImageFromDriver(driver, bins, true, true, outEffect)) return false;
        if (outCore != nullptr && !mapSpectralImageFromDriver(driver, bins, true, false, outCore)) return false;
        if (outStructure != nullptr && !mapSpectralImageFromDriver(driver, bins, false, true, outStructure)) return false;
        return true;
    };
    auto encodeApplyShoulderStackBuffer = [&](id<MTLBuffer> imageStack, float shoulder, int stackDepth) -> bool {
        if (imageStack == nil || shoulder <= 0.0f) {
            return true;
        }
        const StackImageParamsGpu stackParams {width, height, stackDepth, static_cast<int>(pixelCount)};
        const ShoulderParamsGpu shoulderParams {width, height, shoulder};
        id<MTLBuffer> stackParamsBuffer = makeParamBuffer(device, stackParams, error);
        id<MTLBuffer> shoulderParamsBuffer = makeParamBuffer(device, shoulderParams, error);
        return stackParamsBuffer != nil && shoulderParamsBuffer != nil &&
               encode2dDispatch(renderContext.encoder,
                                pipelines->applyShoulderStack,
                                @[imageStack, stackParamsBuffer, shoulderParamsBuffer],
                                width,
                                height * stackDepth);
    };
    auto encodeCombineStackBuffers = [&](id<MTLBuffer> coreStack,
                                         id<MTLBuffer> structureStack,
                                         int stackDepth,
                                         id<MTLBuffer>* outEffectStack) -> bool {
        if (coreStack == nil || structureStack == nil || outEffectStack == nullptr) {
            return false;
        }
        id<MTLBuffer> effectStack = makeSharedBuffer(device, rgbaBytes * static_cast<NSUInteger>(stackDepth), error);
        const StackImageParamsGpu stackParams {width, height, stackDepth, static_cast<int>(pixelCount)};
        const CombineParamsGpu combineParams {width, height, static_cast<float>(std::max(0.0, params.coreGain)),
                                              static_cast<float>(std::max(0.0, params.structureGain))};
        id<MTLBuffer> stackParamsBuffer = makeParamBuffer(device, stackParams, error);
        id<MTLBuffer> combineParamsBuffer = makeParamBuffer(device, combineParams, error);
        if (effectStack == nil || stackParamsBuffer == nil || combineParamsBuffer == nil) {
            return false;
        }
        const bool ok = encode2dDispatch(renderContext.encoder,
                                         pipelines->combineStack,
                                         @[coreStack, structureStack, effectStack, stackParamsBuffer, combineParamsBuffer],
                                         width,
                                         height * stackDepth);
        if (ok) {
            *outEffectStack = effectStack;
        }
        return ok;
    };
    auto encodeAccumulateWeightedRgbStackBuffer = [&](id<MTLBuffer> imageStack,
                                                      int stackDepth,
                                                      id<MTLBuffer>* outBuffer) -> bool {
        if (imageStack == nil || outBuffer == nullptr) {
            return false;
        }
        id<MTLBuffer> output = makeRgbaBuffer();
        const StackImageParamsGpu stackParams {width, height, stackDepth, static_cast<int>(pixelCount)};
        id<MTLBuffer> stackParamsBuffer = makeParamBuffer(device, stackParams, error);
        if (output == nil || stackParamsBuffer == nil || !zeroRgbaBuffer(output)) {
            return false;
        }
        const bool ok = encode2dDispatch(renderContext.encoder,
                                         pipelines->accumulateWeightedRgbStack,
                                         @[imageStack, output, stackParamsBuffer],
                                         width,
                                         height);
        if (ok) {
            *outBuffer = output;
        }
        return ok;
    };
    auto mapSpectralPlaneStackToRgbStack = [&](id<MTLBuffer> planeStack,
                                               const LensDiffSpectrumConfig& spectrumConfig,
                                               int zoneCount,
                                               int binCount,
                                               id<MTLBuffer>* outRgbStack) -> bool {
        if (planeStack == nil || outRgbStack == nullptr) {
            return false;
        }
        id<MTLBuffer> rgbStack = makeSharedBuffer(device, rgbaBytes * static_cast<NSUInteger>(zoneCount), error);
        ZonePlaneStackParamsGpu stackParams {width, height, zoneCount, binCount, static_cast<int>(pixelCount)};
        SpectralMapParamsGpu spectralParams {};
        spectralParams.width = width;
        spectralParams.height = height;
        spectralParams.binCount = spectrumConfig.binCount;
        spectralParams.chromaticAffectsLuma = params.chromaticAffectsLuma ? 1 : 0;
        spectralParams.spectrumForce = static_cast<float>(params.spectrumForce);
        spectralParams.spectrumSaturation = static_cast<float>(std::max(0.0, params.spectrumSaturation));
        std::copy(spectrumConfig.naturalMatrix.begin(), spectrumConfig.naturalMatrix.end(), spectralParams.naturalMatrix);
        std::copy(spectrumConfig.styleMatrix.begin(), spectrumConfig.styleMatrix.end(), spectralParams.styleMatrix);
        id<MTLBuffer> stackParamsBuffer = makeParamBuffer(device, stackParams, error);
        id<MTLBuffer> spectralParamsBuffer = makeParamBuffer(device, spectralParams, error);
        if (rgbStack == nil || stackParamsBuffer == nil || spectralParamsBuffer == nil) {
            return false;
        }
        const bool ok = encode2dDispatch(renderContext.encoder,
                                         pipelines->mapSpectralStack,
                                         @[planeStack, rgbStack, stackParamsBuffer, spectralParamsBuffer],
                                         width,
                                         height * zoneCount);
        if (ok) {
            *outRgbStack = rgbStack;
        }
        return ok;
    };
    auto accumulateWeightedPlaneStack = [&](id<MTLBuffer> planeStack,
                                            int zoneCount,
                                            int binCount,
                                            std::array<id<MTLBuffer>, kLensDiffMaxSpectralBins>* outPlanes) -> bool {
        if (planeStack == nil || outPlanes == nullptr) {
            return false;
        }
        outPlanes->fill(nil);
        for (int binIndex = 0; binIndex < binCount; ++binIndex) {
            id<MTLBuffer> dstPlane = makeScalarBuffer();
            ZonePlaneAccumulateParamsGpu accumulateParams {width, height, zoneCount, binCount, static_cast<int>(pixelCount), binIndex};
            id<MTLBuffer> accumulateParamsBuffer = makeParamBuffer(device, accumulateParams, error);
            if (dstPlane == nil || accumulateParamsBuffer == nil) {
                return false;
            }
            if (!encode2dDispatch(renderContext.encoder,
                                  pipelines->accumulateWeightedPlanesStack,
                                  @[planeStack, dstPlane, accumulateParamsBuffer],
                                  width,
                                  height)) {
                return false;
            }
            (*outPlanes)[static_cast<std::size_t>(binIndex)] = dstPlane;
        }
        return true;
    };
    auto buildSpectralKernelStack = [&](FieldEffectKind effectKind,
                                        int paddedSize,
                                        int binCount,
                                        id<MTLBuffer>* outStack) -> bool {
        if (outStack == nullptr || !fieldPlan.canonical3x3) {
            return false;
        }
        const std::string key = fieldStackCacheKey(paddedSize, effectKind, binCount, 1);
        auto it = fieldKernelStackCache.find(key);
        if (it != fieldKernelStackCache.end()) {
            *outStack = it->second.spectrumStack;
            return true;
        }
        std::vector<const LensDiffKernel*> kernels;
        kernels.reserve(fieldPlan.zones.size() * static_cast<std::size_t>(binCount));
        for (const LensDiffFieldZoneCache* zone : fieldPlan.zones) {
            for (int binIndex = 0; binIndex < binCount; ++binIndex) {
                kernels.push_back(&kernelForEffect(zone->bins[static_cast<std::size_t>(binIndex)], effectKind));
            }
        }
        id<MTLBuffer> stack = nil;
        if (!encodeBuildKernelSpectrumStackMetal(&renderContext, kernels, 1, paddedSize, &stack)) {
            return false;
        }
        fieldKernelStackCache.emplace(key, FieldZoneKernelStacks {paddedSize, static_cast<int>(fieldPlan.zones.size()), binCount, effectKind, stack});
        *outStack = stack;
        return true;
    };
    auto convolveMonoFieldStack = [&](FieldEffectKind effectKind, id<MTLBuffer>* outRgbStack) -> bool {
        if (!fieldPlan.canonical3x3 || outRgbStack == nullptr) {
            return false;
        }
        int maxKernelSize = 1;
        for (const LensDiffFieldZoneCache* zone : fieldPlan.zones) {
            maxKernelSize = std::max(maxKernelSize, kernelForEffect(zone->bins.front(), effectKind).size);
        }
        const int paddedSize = paddedFftSizeForKernel(maxKernelSize);
        RgbSourceSpectraMetal redistributedSpectra {};
        if (!buildRgbSourceSpectra(redistributed, paddedSize, &redistributedSpectra)) {
            return false;
        }
        id<MTLBuffer> sourceSpectrumStack = nil;
        id<MTLBuffer> kernelSpectrumStack = nil;
        const int zoneCount = static_cast<int>(fieldPlan.zones.size());
        if (!getReplicatedSpectrumStack(redistributedSpectra.stack,
                                        "mono:" + std::to_string(static_cast<int>(effectKind)) + ":" + std::to_string(paddedSize),
                                        3,
                                        zoneCount * 3,
                                        paddedSize,
                                        &sourceSpectrumStack) ||
            !getFieldKernelSpectrumStack(effectKind, paddedSize, 0, 3, &kernelSpectrumStack)) {
            return false;
        }
        id<MTLBuffer> planeStack = nil;
        if (!timeCall(timing.convolutionMs, [&] {
                return encodeConvolvePairwiseStackToPlaneStackMetal(&renderContext,
                                                                    sourceSpectrumStack,
                                                                    kernelSpectrumStack,
                                                                    width,
                                                                    height,
                                                                    paddedSize,
                                                                    zoneCount * 3,
                                                                    &planeStack);
            })) {
            return false;
        }
        return encodePackPlaneTripletsToRgbaStackMetal(&renderContext, planeStack, width, height, zoneCount, outRgbStack);
    };
    auto convolveSpectralFieldStack = [&](FieldEffectKind effectKind,
                                          id<MTLBuffer>* outPlaneStack,
                                          int* outBinCount) -> bool {
        if (!fieldPlan.canonical3x3 || outPlaneStack == nullptr || outBinCount == nullptr) {
            return false;
        }
        const int binCount = std::min<int>(static_cast<int>(fieldPlan.zones.front()->bins.size()), kLensDiffMaxSpectralBins);
        int maxKernelSize = 1;
        for (const LensDiffFieldZoneCache* zone : fieldPlan.zones) {
            for (int binIndex = 0; binIndex < binCount; ++binIndex) {
                maxKernelSize = std::max(maxKernelSize, kernelForEffect(zone->bins[static_cast<std::size_t>(binIndex)], effectKind).size);
            }
        }
        const int paddedSize = paddedFftSizeForKernel(maxKernelSize);
        id<MTLBuffer> driverSpectrum = nil;
        if (!getScalarSourceSpectrum(driver, paddedSize, &driverSpectrum)) {
            return false;
        }
        const int zoneCount = static_cast<int>(fieldPlan.zones.size());
        id<MTLBuffer> sourceSpectrumStack = nil;
        id<MTLBuffer> kernelSpectrumStack = nil;
        if (!getReplicatedSpectrumStack(driverSpectrum,
                                        "spectral:" + std::to_string(static_cast<int>(effectKind)) + ":" + std::to_string(paddedSize) + ":" + std::to_string(binCount),
                                        1,
                                        zoneCount * binCount,
                                        paddedSize,
                                        &sourceSpectrumStack) ||
            !buildSpectralKernelStack(effectKind, paddedSize, binCount, &kernelSpectrumStack)) {
            return false;
        }
        if (!timeCall(timing.convolutionMs, [&] {
                return encodeConvolvePairwiseStackToPlaneStackMetal(&renderContext,
                                                                    sourceSpectrumStack,
                                                                    kernelSpectrumStack,
                                                                    width,
                                                                    height,
                                                                    paddedSize,
                                                                    zoneCount * binCount,
                                                                    outPlaneStack);
            })) {
            return false;
        }
        *outBinCount = binCount;
        return true;
    };
    auto renderFieldZonesStacked = [&](id<MTLBuffer>* outEffect,
                                       id<MTLBuffer>* outCore,
                                       id<MTLBuffer>* outStructure) -> bool {
        if (!fieldPlan.canonical3x3) {
            return false;
        }
        const int zoneCount = static_cast<int>(fieldPlan.zones.size());
        if (params.spectralMode == LensDiffSpectralMode::Mono) {
            id<MTLBuffer> fullStack = nil;
            if (splitMode) {
                id<MTLBuffer> coreStack = nil;
                id<MTLBuffer> structureStack = nil;
                if (!convolveMonoFieldStack(FieldEffectKind::Core, &coreStack) ||
                    !convolveMonoFieldStack(FieldEffectKind::Structure, &structureStack)) {
                    return false;
                }
                if (!encodeApplyShoulderStackBuffer(coreStack, static_cast<float>(params.coreShoulder), zoneCount) ||
                    !encodeApplyShoulderStackBuffer(structureStack, static_cast<float>(params.structureShoulder), zoneCount)) {
                    return false;
                }
                if (!encodeCombineStackBuffers(coreStack, structureStack, zoneCount, &fullStack)) {
                    return false;
                }
                return encodeAccumulateWeightedRgbStackBuffer(fullStack, zoneCount, outEffect) &&
                       (!outCore || encodeAccumulateWeightedRgbStackBuffer(coreStack, zoneCount, outCore)) &&
                       (!outStructure || encodeAccumulateWeightedRgbStackBuffer(structureStack, zoneCount, outStructure));
            }
            if (!convolveMonoFieldStack(FieldEffectKind::Full, &fullStack)) {
                return false;
            }
            return encodeAccumulateWeightedRgbStackBuffer(fullStack, zoneCount, outEffect) &&
                   (!outCore || ([&] {
                        id<MTLBuffer> coreStack = nil;
                        return convolveMonoFieldStack(FieldEffectKind::Core, &coreStack) &&
                               encodeAccumulateWeightedRgbStackBuffer(coreStack, zoneCount, outCore);
                   })()) &&
                   (!outStructure || ([&] {
                        id<MTLBuffer> structureStack = nil;
                        return convolveMonoFieldStack(FieldEffectKind::Structure, &structureStack) &&
                               encodeAccumulateWeightedRgbStackBuffer(structureStack, zoneCount, outStructure);
                   })());
        }

        const LensDiffSpectrumConfig zoneSpectrumConfig = BuildLensDiffSpectrumConfig(params, fieldPlan.zones.front()->bins);
        if (splitMode) {
            id<MTLBuffer> corePlaneStack = nil;
            id<MTLBuffer> structurePlaneStack = nil;
            int binCount = 0;
            if (!convolveSpectralFieldStack(FieldEffectKind::Core, &corePlaneStack, &binCount) ||
                !convolveSpectralFieldStack(FieldEffectKind::Structure, &structurePlaneStack, &binCount)) {
                return false;
            }
            id<MTLBuffer> coreStack = nil;
            id<MTLBuffer> structureStack = nil;
            if (!mapSpectralPlaneStackToRgbStack(corePlaneStack, zoneSpectrumConfig, zoneCount, binCount, &coreStack) ||
                !mapSpectralPlaneStackToRgbStack(structurePlaneStack, zoneSpectrumConfig, zoneCount, binCount, &structureStack) ||
                !encodeApplyShoulderStackBuffer(coreStack, static_cast<float>(params.coreShoulder), zoneCount) ||
                !encodeApplyShoulderStackBuffer(structureStack, static_cast<float>(params.structureShoulder), zoneCount)) {
                return false;
            }
            id<MTLBuffer> effectStack = nil;
            if (!encodeCombineStackBuffers(coreStack, structureStack, zoneCount, &effectStack)) {
                return false;
            }
            return encodeAccumulateWeightedRgbStackBuffer(effectStack, zoneCount, outEffect) &&
                   (!outCore || encodeAccumulateWeightedRgbStackBuffer(coreStack, zoneCount, outCore)) &&
                   (!outStructure || encodeAccumulateWeightedRgbStackBuffer(structureStack, zoneCount, outStructure));
        }

        id<MTLBuffer> fullPlaneStack = nil;
        int binCount = 0;
        if (!convolveSpectralFieldStack(FieldEffectKind::Full, &fullPlaneStack, &binCount)) {
            return false;
        }
        std::array<id<MTLBuffer>, kLensDiffMaxSpectralBins> weightedPlanes {};
        if (!accumulateWeightedPlaneStack(fullPlaneStack, zoneCount, binCount, &weightedPlanes)) {
            return false;
        }
        SpectralMapParamsGpu spectralParams {};
        spectralParams.width = width;
        spectralParams.height = height;
        spectralParams.binCount = zoneSpectrumConfig.binCount;
        spectralParams.chromaticAffectsLuma = params.chromaticAffectsLuma ? 1 : 0;
        spectralParams.spectrumForce = static_cast<float>(params.spectrumForce);
        spectralParams.spectrumSaturation = static_cast<float>(std::max(0.0, params.spectrumSaturation));
        std::copy(zoneSpectrumConfig.naturalMatrix.begin(), zoneSpectrumConfig.naturalMatrix.end(), spectralParams.naturalMatrix);
        std::copy(zoneSpectrumConfig.styleMatrix.begin(), zoneSpectrumConfig.styleMatrix.end(), spectralParams.styleMatrix);
        id<MTLBuffer> spectralParamsBuffer = makeParamBuffer(device, spectralParams, error);
        *outEffect = makeRgbaBuffer();
        for (int i = binCount; i < kLensDiffMaxSpectralBins; ++i) {
            weightedPlanes[static_cast<std::size_t>(i)] = weightedPlanes[0];
        }
        if (*outEffect == nil || spectralParamsBuffer == nil ||
            !encode2dDispatch(renderContext.encoder,
                              pipelines->mapSpectral,
                              @[weightedPlanes[0], weightedPlanes[1], weightedPlanes[2], weightedPlanes[3], weightedPlanes[4],
                                weightedPlanes[5], weightedPlanes[6], weightedPlanes[7], weightedPlanes[8],
                                *outEffect, spectralParamsBuffer],
                              width,
                              height)) {
            return false;
        }
        return (!outCore || ([&] {
                    id<MTLBuffer> corePlaneStack = nil;
                    int coreBinCount = 0;
                    std::array<id<MTLBuffer>, kLensDiffMaxSpectralBins> corePlanes {};
                    id<MTLBuffer> coreParamsBuffer = makeParamBuffer(device, spectralParams, error);
                    *outCore = makeRgbaBuffer();
                    return coreParamsBuffer != nil && *outCore != nil &&
                           convolveSpectralFieldStack(FieldEffectKind::Core, &corePlaneStack, &coreBinCount) &&
                           accumulateWeightedPlaneStack(corePlaneStack, zoneCount, coreBinCount, &corePlanes) &&
                           ([&] {
                               for (int i = coreBinCount; i < kLensDiffMaxSpectralBins; ++i) {
                                   corePlanes[static_cast<std::size_t>(i)] = corePlanes[0];
                               }
                               return encode2dDispatch(renderContext.encoder,
                                                       pipelines->mapSpectral,
                                                       @[corePlanes[0], corePlanes[1], corePlanes[2], corePlanes[3], corePlanes[4],
                                                         corePlanes[5], corePlanes[6], corePlanes[7], corePlanes[8],
                                                         *outCore, coreParamsBuffer],
                                                       width,
                                                       height);
                           })();
               })()) &&
               (!outStructure || ([&] {
                    id<MTLBuffer> structurePlaneStack = nil;
                    int structureBinCount = 0;
                    std::array<id<MTLBuffer>, kLensDiffMaxSpectralBins> structurePlanes {};
                    id<MTLBuffer> structureParamsBuffer = makeParamBuffer(device, spectralParams, error);
                    *outStructure = makeRgbaBuffer();
                    return structureParamsBuffer != nil && *outStructure != nil &&
                           convolveSpectralFieldStack(FieldEffectKind::Structure, &structurePlaneStack, &structureBinCount) &&
                           accumulateWeightedPlaneStack(structurePlaneStack, zoneCount, structureBinCount, &structurePlanes) &&
                           ([&] {
                               for (int i = structureBinCount; i < kLensDiffMaxSpectralBins; ++i) {
                                   structurePlanes[static_cast<std::size_t>(i)] = structurePlanes[0];
                               }
                               return encode2dDispatch(renderContext.encoder,
                                                       pipelines->mapSpectral,
                                                       @[structurePlanes[0], structurePlanes[1], structurePlanes[2], structurePlanes[3], structurePlanes[4],
                                                         structurePlanes[5], structurePlanes[6], structurePlanes[7], structurePlanes[8],
                                                         *outStructure, structureParamsBuffer],
                                                       width,
                                                       height);
                           })();
               })());
    };

    id<MTLBuffer> effectBuffer = nil;
    id<MTLBuffer> coreBuffer = nil;
    id<MTLBuffer> structureBuffer = nil;
    id<MTLBuffer> scatterPreview = nil;
    id<MTLBuffer> creativeFringePreview = nil;
    id<MTLBuffer> finalBuffer = nil;

    if (!legacySync && !beginMetalStage(&renderContext)) {
        return false;
    }

    if (cache.fieldZones.empty()) {
        if (!renderFromBins(cache.bins, &effectBuffer, needCore ? &coreBuffer : nullptr, needStructure ? &structureBuffer : nullptr)) {
            return false;
        }
    } else if (!legacySync && fieldPlan.canonical3x3) {
        const auto fieldZonesStart = std::chrono::steady_clock::now();
        timing.fieldZoneBatchDepth = std::max(timing.fieldZoneBatchDepth, static_cast<int>(fieldPlan.zones.size()));
        if (!renderFieldZonesStacked(&effectBuffer, needCore ? &coreBuffer : nullptr, needStructure ? &structureBuffer : nullptr)) {
            return false;
        }
        timing.fieldZonesMs += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                                   std::chrono::steady_clock::now() - fieldZonesStart)
                                   .count();
    } else {
        const auto fieldZonesStart = std::chrono::steady_clock::now();
        timing.fieldZoneBatchDepth = std::max(timing.fieldZoneBatchDepth, static_cast<int>(cache.fieldZones.size()));
        effectBuffer = makeRgbaBuffer();
        if (effectBuffer == nil || !zeroRgbaBuffer(effectBuffer) ||
            (needCore && ((coreBuffer = makeRgbaBuffer()) == nil || !zeroRgbaBuffer(coreBuffer))) ||
            (needStructure && ((structureBuffer = makeRgbaBuffer()) == nil || !zeroRgbaBuffer(structureBuffer)))) {
            return false;
        }
        id<MTLCommandBuffer> fieldBlendCommandBuffer = legacySync ? [queue commandBuffer] : nil;
        id<MTLComputeCommandEncoder> fieldBlendEncoder = legacySync ? [fieldBlendCommandBuffer computeCommandEncoder] : renderContext.encoder;
        if (fieldBlendEncoder == nil) {
            if (error) *error = "metal-field-zone-encoder-create-failed";
            return false;
        }
        if (legacySync) {
            ++renderCounters.commandBufferCount;
        }
        for (const auto& zone : cache.fieldZones) {
            id<MTLBuffer> zoneEffect = nil;
            id<MTLBuffer> zoneCore = nil;
            id<MTLBuffer> zoneStructure = nil;
            if (!renderFromBins(zone.bins, &zoneEffect, needCore ? &zoneCore : nullptr, needStructure ? &zoneStructure : nullptr)) {
                return false;
            }
            const FieldBlendParamsGpu blendParams {width, height, zone.zoneX, zone.zoneY, 1.0f};
            id<MTLBuffer> blendParamsBuffer = makeParamBuffer(device, blendParams, error);
            if (blendParamsBuffer == nil ||
                !encode2dDispatch(fieldBlendEncoder, pipelines->accumulateWeighted, @[zoneEffect, effectBuffer, blendParamsBuffer], width, height) ||
                (needCore && !encode2dDispatch(fieldBlendEncoder, pipelines->accumulateWeighted, @[zoneCore, coreBuffer, blendParamsBuffer], width, height)) ||
                (needStructure && !encode2dDispatch(fieldBlendEncoder, pipelines->accumulateWeighted, @[zoneStructure, structureBuffer, blendParamsBuffer], width, height))) {
                return false;
            }
        }
        if (legacySync) {
            [fieldBlendEncoder endEncoding];
            if (!commitAndWaitCounted(&renderCounters, fieldBlendCommandBuffer, error)) {
                return false;
            }
        }
        timing.fieldZonesMs += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                                   std::chrono::steady_clock::now() - fieldZonesStart)
                                   .count();
    }

    if (params.energyMode == LensDiffEnergyMode::Preserve && effectBuffer != nil) {
        id<MTLBuffer> inputEnergyBuffer = nil;
        id<MTLBuffer> effectEnergyBuffer = nil;
        id<MTLBuffer> preserveScaleBuffer = makeSizedBuffer(sizeof(float));
        const PreserveScaleParamsGpu preserveScaleParams {1e-6f};
        const DynamicScaleParamsGpu dynamicScaleParams {width, height};
        id<MTLBuffer> preserveScaleParamsBuffer = makeParamBuffer(device, preserveScaleParams, error);
        id<MTLBuffer> dynamicScaleParamsBuffer = makeParamBuffer(device, dynamicScaleParams, error);
        if (preserveScaleBuffer == nil || preserveScaleParamsBuffer == nil || dynamicScaleParamsBuffer == nil) {
            return false;
        }
        if (legacySync) {
            id<MTLCommandBuffer> preserveCommandBuffer = [queue commandBuffer];
            ++renderCounters.commandBufferCount;
            id<MTLComputeCommandEncoder> preserveEncoder = [preserveCommandBuffer computeCommandEncoder];
            const bool ok = encodeReduceEnergyToScalar(preserveEncoder, redistributed, &inputEnergyBuffer) &&
                            encodeReduceEnergyToScalar(preserveEncoder, effectBuffer, &effectEnergyBuffer) &&
                            encode1dDispatch(preserveEncoder,
                                             pipelines->computePreserveScale,
                                             @[inputEnergyBuffer, effectEnergyBuffer, preserveScaleBuffer, preserveScaleParamsBuffer],
                                             1) &&
                            encode2dDispatch(preserveEncoder,
                                             pipelines->scaleRgbDynamic,
                                             @[effectBuffer, preserveScaleBuffer, dynamicScaleParamsBuffer],
                                             width,
                                             height);
            [preserveEncoder endEncoding];
            if (!ok || !commitAndWaitCounted(&renderCounters, preserveCommandBuffer, error)) {
                return false;
            }
        } else {
            const bool ok = encodeReduceEnergyToScalar(renderContext.encoder, redistributed, &inputEnergyBuffer) &&
                            encodeReduceEnergyToScalar(renderContext.encoder, effectBuffer, &effectEnergyBuffer) &&
                            encode1dDispatch(renderContext.encoder,
                                             pipelines->computePreserveScale,
                                             @[inputEnergyBuffer, effectEnergyBuffer, preserveScaleBuffer, preserveScaleParamsBuffer],
                                             1) &&
                            encode2dDispatch(renderContext.encoder,
                                             pipelines->scaleRgbDynamic,
                                             @[effectBuffer, preserveScaleBuffer, dynamicScaleParamsBuffer],
                                             width,
                                             height);
            if (!ok) {
                return false;
            }
        }
    }

    const double scatterRadiusPx = ResolveLensDiffScatterRadiusPx(params);
    const bool scatterActive = params.scatterAmount > 1e-6 && scatterRadiusPx > 0.25;
    if (scatterActive || params.debugView == LensDiffDebugView::Scatter) {
        if (scatterActive) {
            const LensDiffKernel scatterKernel = buildGaussianKernelHost(static_cast<float>(scatterRadiusPx));
            if (!runRgbConvolutionFromSource(effectBuffer, scatterKernel, &scatterPreview)) {
                return false;
            }
            const CombineParamsGpu combineParams {width, height, 1.0f, static_cast<float>(std::max(0.0, params.scatterAmount))};
            id<MTLBuffer> combineParamsBuffer = makeParamBuffer(device, combineParams, error);
            id<MTLBuffer> combinedEffect = makeRgbaBuffer();
            if (combineParamsBuffer == nil || combinedEffect == nil ||
                !(legacySync
                    ? encode2d(pipelines->combine, @[effectBuffer, scatterPreview, combinedEffect, combineParamsBuffer], width, height)
                    : encode2dDispatch(renderContext.encoder, pipelines->combine, @[effectBuffer, scatterPreview, combinedEffect, combineParamsBuffer], width, height))) {
                return false;
            }
            effectBuffer = combinedEffect;
        } else {
            scatterPreview = makeRgbaBuffer();
            if (scatterPreview == nil || !zeroRgbaBuffer(scatterPreview)) {
                return false;
            }
        }
    }

    const double creativeFringePx = ResolveLensDiffCreativeFringePx(params);
    const bool creativeFringeActive = creativeFringePx > 1e-6;
    if (creativeFringeActive || params.debugView == LensDiffDebugView::CreativeFringe) {
        id<MTLBuffer> fringedEffect = makeRgbaBuffer();
        creativeFringePreview = makeRgbaBuffer();
        const FringeParamsGpu fringeParams {width, height, static_cast<float>(std::max(0.0, creativeFringePx))};
        id<MTLBuffer> fringeParamsBuffer = makeParamBuffer(device, fringeParams, error);
        if (fringedEffect == nil || creativeFringePreview == nil || fringeParamsBuffer == nil) {
            return false;
        }
        if (creativeFringeActive) {
            if (!(legacySync
                    ? encode2d(pipelines->creativeFringe, @[effectBuffer, fringedEffect, creativeFringePreview, fringeParamsBuffer], width, height)
                    : encode2dDispatch(renderContext.encoder, pipelines->creativeFringe, @[effectBuffer, fringedEffect, creativeFringePreview, fringeParamsBuffer], width, height))) {
                return false;
            }
            effectBuffer = fringedEffect;
        } else if (!zeroRgbaBuffer(creativeFringePreview)) {
            return false;
        }
    }

    const float effectGain = params.energyMode == LensDiffEnergyMode::Preserve
        ? static_cast<float>(clampValue(params.effectGain, 0.0, 1.0))
        : static_cast<float>(std::max(0.0, params.effectGain));
    const float coreCompensation = params.energyMode == LensDiffEnergyMode::Preserve
        ? effectGain
        : static_cast<float>(std::max(0.0, params.coreCompensation));

    if (params.debugView == LensDiffDebugView::Final) {
        const auto compositeStart = std::chrono::steady_clock::now();
        id<MTLBuffer> compositeRedistributed = redistributed;
        id<MTLBuffer> compositeEffect = effectBuffer;
        int compositeWidth = width;
        int compositeHeight = height;
        if (resolutionAwareActive) {
            compositeRedistributed = makeNativeRgbaBuffer();
            compositeEffect = makeNativeRgbaBuffer();
            compositeWidth = nativeWidth;
            compositeHeight = nativeHeight;
            if (compositeRedistributed == nil || compositeEffect == nil) {
                return false;
            }
        }
        finalBuffer = resolutionAwareActive ? makeNativeRgbaBuffer() : makeRgbaBuffer();
        const CompositeParamsGpu compositeParams {compositeWidth, compositeHeight, effectGain, coreCompensation, maxRedistributedSubtractScale};
        id<MTLBuffer> compositeParamsBuffer = makeParamBuffer(device, compositeParams, error);
        if (finalBuffer == nil || compositeRedistributed == nil || compositeEffect == nil || compositeParamsBuffer == nil) {
            return false;
        }
        if (resolutionAwareActive) {
            if (legacySync) {
                id<MTLCommandBuffer> finalCompositeCommandBuffer = [queue commandBuffer];
                ++renderCounters.commandBufferCount;
                id<MTLComputeCommandEncoder> finalCompositeEncoder = [finalCompositeCommandBuffer computeCommandEncoder];
                const bool ok = encodeResampleRgbaDispatch(finalCompositeEncoder, redistributed, compositeRedistributed, width, height, nativeWidth, nativeHeight) &&
                                encodeResampleRgbaDispatch(finalCompositeEncoder, effectBuffer, compositeEffect, width, height, nativeWidth, nativeHeight) &&
                                encode2dDispatch(finalCompositeEncoder,
                                                 pipelines->composite,
                                                 @[linearSrcNative, compositeRedistributed, compositeEffect, finalBuffer, compositeParamsBuffer],
                                                 compositeWidth,
                                                 compositeHeight);
                [finalCompositeEncoder endEncoding];
                if (!ok || !commitAndWaitCounted(&renderCounters, finalCompositeCommandBuffer, error)) {
                    return false;
                }
            } else {
                const bool ok = encodeResampleRgbaDispatch(renderContext.encoder, redistributed, compositeRedistributed, width, height, nativeWidth, nativeHeight) &&
                                encodeResampleRgbaDispatch(renderContext.encoder, effectBuffer, compositeEffect, width, height, nativeWidth, nativeHeight) &&
                                encode2dDispatch(renderContext.encoder,
                                                 pipelines->composite,
                                                 @[linearSrcNative, compositeRedistributed, compositeEffect, finalBuffer, compositeParamsBuffer],
                                                 compositeWidth,
                                                 compositeHeight);
                if (!ok) {
                    return false;
                }
            }
        } else if (!(legacySync
                         ? encode2d(pipelines->composite,
                                    @[linearSrc, compositeRedistributed, compositeEffect, finalBuffer, compositeParamsBuffer],
                                    compositeWidth,
                                    compositeHeight)
                         : encode2dDispatch(renderContext.encoder,
                                            pipelines->composite,
                                            @[linearSrc, compositeRedistributed, compositeEffect, finalBuffer, compositeParamsBuffer],
                                            compositeWidth,
                                            compositeHeight))) {
            return false;
        }
        timing.compositeOutputMs += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                                        std::chrono::steady_clock::now() - compositeStart)
                                        .count();
    }

    if (!legacySync && !endMetalStage(&renderContext)) {
        return false;
    }

    const LensDiffImageRect outputRect = intersectRect(request.renderWindow, request.dst.bounds);
    if (outputRect.width() <= 0 || outputRect.height() <= 0) {
        renderSucceeded = true;
        LogLensDiffDiagnosticEvent("metal-output-empty", "renderWindow-outside-dst");
        renderScopeResult = true;
        return true;
    }

    const OutputParamsGpu outParams {
        nativeWidth, nativeHeight, request.src.bounds.x1, request.src.bounds.y1,
        request.dst.bounds.x1, request.dst.bounds.y1,
        request.dst.bounds.x2, request.dst.bounds.y2,
        outputRect.x1, outputRect.y1,
        outputRect.x2, outputRect.y2,
        rowFloats(request.dst), static_cast<int>(params.inputTransfer),
        params.debugView == LensDiffDebugView::Final ? 1 : 0,
        params.debugView == LensDiffDebugView::Final ? 1 : 0};
    id<MTLBuffer> outParamsBuffer = makeParamBuffer(device, outParams, error);
    if (outParamsBuffer == nil) {
        return false;
    }

    const auto outputStart = std::chrono::steady_clock::now();
    if (!legacySync && !beginMetalStage(&renderContext)) {
        return false;
    }
    bool outputOk = false;
    switch (params.debugView) {
        case LensDiffDebugView::Selection:
            if (resolutionAwareActive) {
                id<MTLBuffer> nativeMask = makeNativeScalarBuffer();
                if (nativeMask == nil) {
                    return false;
                }
                if (legacySync) {
                    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                    ++renderCounters.commandBufferCount;
                    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                    const bool ok = encodeResampleGrayDispatch(encoder, mask, nativeMask, width, height, nativeWidth, nativeHeight) &&
                                    encode2dDispatch(encoder,
                                                     pipelines->packGray,
                                                     @[nativeMask, dstBuffer, outParamsBuffer],
                                                     outputRect.width(),
                                                     outputRect.height());
                    [encoder endEncoding];
                    outputOk = ok && commitAndWaitCounted(&renderCounters, commandBuffer, error);
                } else {
                    outputOk = encodeResampleGrayDispatch(renderContext.encoder, mask, nativeMask, width, height, nativeWidth, nativeHeight) &&
                               encode2dDispatch(renderContext.encoder,
                                                pipelines->packGray,
                                                @[nativeMask, dstBuffer, outParamsBuffer],
                                                outputRect.width(),
                                                outputRect.height());
                }
                break;
            }
            outputOk = legacySync
                           ? encode2d(pipelines->packGray,
                           @[mask, dstBuffer, outParamsBuffer],
                           outputRect.width(),
                           outputRect.height())
                : encode2dDispatch(renderContext.encoder,
                                   pipelines->packGray,
                                   @[mask, dstBuffer, outParamsBuffer],
                                   outputRect.width(),
                                   outputRect.height());
            break;
        case LensDiffDebugView::Core:
            if (coreBuffer == nil) return false;
            if (resolutionAwareActive) {
                id<MTLBuffer> nativeCore = makeNativeRgbaBuffer();
                if (nativeCore == nil) return false;
                if (legacySync) {
                    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                    ++renderCounters.commandBufferCount;
                    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                    const bool ok = encodeResampleRgbaDispatch(encoder, coreBuffer, nativeCore, width, height, nativeWidth, nativeHeight) &&
                                    encode2dDispatch(encoder,
                                                     pipelines->packRgb,
                                                     @[nativeCore, linearSrcNative, dstBuffer, outParamsBuffer],
                                                     outputRect.width(),
                                                     outputRect.height());
                    [encoder endEncoding];
                    outputOk = ok && commitAndWaitCounted(&renderCounters, commandBuffer, error);
                } else {
                    outputOk = encodeResampleRgbaDispatch(renderContext.encoder, coreBuffer, nativeCore, width, height, nativeWidth, nativeHeight) &&
                               encode2dDispatch(renderContext.encoder,
                                                pipelines->packRgb,
                                                @[nativeCore, linearSrcNative, dstBuffer, outParamsBuffer],
                                                outputRect.width(),
                                                outputRect.height());
                }
                break;
            }
            outputOk = legacySync
                           ? encode2d(pipelines->packRgb,
                           @[coreBuffer, linearSrcNative, dstBuffer, outParamsBuffer],
                           outputRect.width(),
                           outputRect.height())
                : encode2dDispatch(renderContext.encoder,
                                   pipelines->packRgb,
                                   @[coreBuffer, linearSrcNative, dstBuffer, outParamsBuffer],
                                   outputRect.width(),
                                   outputRect.height());
            break;
        case LensDiffDebugView::Structure:
            if (structureBuffer == nil) return false;
            if (resolutionAwareActive) {
                id<MTLBuffer> nativeStructure = makeNativeRgbaBuffer();
                if (nativeStructure == nil) return false;
                if (legacySync) {
                    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                    ++renderCounters.commandBufferCount;
                    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                    const bool ok = encodeResampleRgbaDispatch(encoder, structureBuffer, nativeStructure, width, height, nativeWidth, nativeHeight) &&
                                    encode2dDispatch(encoder,
                                                     pipelines->packRgb,
                                                     @[nativeStructure, linearSrcNative, dstBuffer, outParamsBuffer],
                                                     outputRect.width(),
                                                     outputRect.height());
                    [encoder endEncoding];
                    outputOk = ok && commitAndWaitCounted(&renderCounters, commandBuffer, error);
                } else {
                    outputOk = encodeResampleRgbaDispatch(renderContext.encoder, structureBuffer, nativeStructure, width, height, nativeWidth, nativeHeight) &&
                               encode2dDispatch(renderContext.encoder,
                                                pipelines->packRgb,
                                                @[nativeStructure, linearSrcNative, dstBuffer, outParamsBuffer],
                                                outputRect.width(),
                                                outputRect.height());
                }
                break;
            }
            outputOk = legacySync
                           ? encode2d(pipelines->packRgb,
                           @[structureBuffer, linearSrcNative, dstBuffer, outParamsBuffer],
                           outputRect.width(),
                           outputRect.height())
                : encode2dDispatch(renderContext.encoder,
                                   pipelines->packRgb,
                                   @[structureBuffer, linearSrcNative, dstBuffer, outParamsBuffer],
                                   outputRect.width(),
                                   outputRect.height());
            break;
        case LensDiffDebugView::Effect:
            if (effectBuffer == nil) return false;
            if (resolutionAwareActive) {
                id<MTLBuffer> nativeEffect = makeNativeRgbaBuffer();
                if (nativeEffect == nil) return false;
                if (legacySync) {
                    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                    ++renderCounters.commandBufferCount;
                    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                    const bool ok = encodeResampleRgbaDispatch(encoder, effectBuffer, nativeEffect, width, height, nativeWidth, nativeHeight) &&
                                    encode2dDispatch(encoder,
                                                     pipelines->packRgb,
                                                     @[nativeEffect, linearSrcNative, dstBuffer, outParamsBuffer],
                                                     outputRect.width(),
                                                     outputRect.height());
                    [encoder endEncoding];
                    outputOk = ok && commitAndWaitCounted(&renderCounters, commandBuffer, error);
                } else {
                    outputOk = encodeResampleRgbaDispatch(renderContext.encoder, effectBuffer, nativeEffect, width, height, nativeWidth, nativeHeight) &&
                               encode2dDispatch(renderContext.encoder,
                                                pipelines->packRgb,
                                                @[nativeEffect, linearSrcNative, dstBuffer, outParamsBuffer],
                                                outputRect.width(),
                                                outputRect.height());
                }
                break;
            }
            outputOk = legacySync
                           ? encode2d(pipelines->packRgb,
                           @[effectBuffer, linearSrcNative, dstBuffer, outParamsBuffer],
                           outputRect.width(),
                           outputRect.height())
                : encode2dDispatch(renderContext.encoder,
                                   pipelines->packRgb,
                                   @[effectBuffer, linearSrcNative, dstBuffer, outParamsBuffer],
                                   outputRect.width(),
                                   outputRect.height());
            break;
        case LensDiffDebugView::Scatter:
            if (scatterPreview == nil) return false;
            if (resolutionAwareActive) {
                id<MTLBuffer> nativeScatter = makeNativeRgbaBuffer();
                if (nativeScatter == nil) return false;
                if (legacySync) {
                    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                    ++renderCounters.commandBufferCount;
                    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                    const bool ok = encodeResampleRgbaDispatch(encoder, scatterPreview, nativeScatter, width, height, nativeWidth, nativeHeight) &&
                                    encode2dDispatch(encoder,
                                                     pipelines->packRgb,
                                                     @[nativeScatter, linearSrcNative, dstBuffer, outParamsBuffer],
                                                     outputRect.width(),
                                                     outputRect.height());
                    [encoder endEncoding];
                    outputOk = ok && commitAndWaitCounted(&renderCounters, commandBuffer, error);
                } else {
                    outputOk = encodeResampleRgbaDispatch(renderContext.encoder, scatterPreview, nativeScatter, width, height, nativeWidth, nativeHeight) &&
                               encode2dDispatch(renderContext.encoder,
                                                pipelines->packRgb,
                                                @[nativeScatter, linearSrcNative, dstBuffer, outParamsBuffer],
                                                outputRect.width(),
                                                outputRect.height());
                }
                break;
            }
            outputOk = legacySync
                           ? encode2d(pipelines->packRgb,
                           @[scatterPreview, linearSrcNative, dstBuffer, outParamsBuffer],
                           outputRect.width(),
                           outputRect.height())
                : encode2dDispatch(renderContext.encoder,
                                   pipelines->packRgb,
                                   @[scatterPreview, linearSrcNative, dstBuffer, outParamsBuffer],
                                   outputRect.width(),
                                   outputRect.height());
            break;
        case LensDiffDebugView::CreativeFringe:
            if (creativeFringePreview == nil) return false;
            if (resolutionAwareActive) {
                id<MTLBuffer> nativeFringe = makeNativeRgbaBuffer();
                if (nativeFringe == nil) return false;
                if (legacySync) {
                    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                    ++renderCounters.commandBufferCount;
                    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                    const bool ok = encodeResampleRgbaDispatch(encoder, creativeFringePreview, nativeFringe, width, height, nativeWidth, nativeHeight) &&
                                    encode2dDispatch(encoder,
                                                     pipelines->packRgb,
                                                     @[nativeFringe, linearSrcNative, dstBuffer, outParamsBuffer],
                                                     outputRect.width(),
                                                     outputRect.height());
                    [encoder endEncoding];
                    outputOk = ok && commitAndWaitCounted(&renderCounters, commandBuffer, error);
                } else {
                    outputOk = encodeResampleRgbaDispatch(renderContext.encoder, creativeFringePreview, nativeFringe, width, height, nativeWidth, nativeHeight) &&
                               encode2dDispatch(renderContext.encoder,
                                                pipelines->packRgb,
                                                @[nativeFringe, linearSrcNative, dstBuffer, outParamsBuffer],
                                                outputRect.width(),
                                                outputRect.height());
                }
                break;
            }
            outputOk = legacySync
                           ? encode2d(pipelines->packRgb,
                           @[creativeFringePreview, linearSrcNative, dstBuffer, outParamsBuffer],
                           outputRect.width(),
                           outputRect.height())
                : encode2dDispatch(renderContext.encoder,
                                   pipelines->packRgb,
                                   @[creativeFringePreview, linearSrcNative, dstBuffer, outParamsBuffer],
                                   outputRect.width(),
                                   outputRect.height());
            break;
        case LensDiffDebugView::Final:
        default:
            outputOk = finalBuffer != nil &&
                       (legacySync
                            ? encode2d(pipelines->packRgb, @[finalBuffer, linearSrcNative, dstBuffer, outParamsBuffer], outputRect.width(), outputRect.height())
                            : encode2dDispatch(renderContext.encoder,
                                               pipelines->packRgb,
                                               @[finalBuffer, linearSrcNative, dstBuffer, outParamsBuffer],
                                               outputRect.width(),
                                               outputRect.height()));
            break;
    }
    if (!legacySync) {
        outputOk = outputOk && endMetalStage(&renderContext);
    }
    timing.compositeOutputMs += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                                    std::chrono::steady_clock::now() - outputStart)
                                    .count();
    timing.commandBufferCount = renderCounters.commandBufferCount;
    timing.waitCount = renderCounters.waitCount;
    renderSucceeded = outputOk;
    if (outputOk) {
        std::ostringstream outputNote;
        outputNote << "commandBuffers=" << renderCounters.commandBufferCount
                   << " waits=" << renderCounters.waitCount
                   << " fieldBatchDepth=" << renderCounters.fieldZoneBatchDepth;
        LogLensDiffDiagnosticEvent("metal-output-ready", outputNote.str());
    }
    logTimingBreakdown();
    {
        std::ostringstream returnNote;
        returnNote << "outputOk=" << (outputOk ? "true" : "false")
                   << " commandBuffers=" << renderCounters.commandBufferCount
                   << " waits=" << renderCounters.waitCount
                   << " legacySync=" << (legacySync ? "true" : "false")
                   << " vkfftEffective=" << (vkfftEnabled ? "true" : "false")
                   << " heapsEffective=" << (heapsEnabled ? "true" : "false")
                   << " fastResolutionAware=" << (allowFastResolutionAware ? "true" : "false");
        LogLensDiffDiagnosticEvent("metal-render-return-ready", returnNote.str());
    }
    renderScopeResult = outputOk;
    }
    LogLensDiffDiagnosticEvent("metal-render-scope-exit",
                               renderScopeResult ? "outputOk=true" : "outputOk=false");
    return renderScopeResult;
}

#else

bool RunLensDiffMetal(const LensDiffRenderRequest&,
                      const LensDiffParams&,
                      LensDiffPsfBankCache&,
                      std::string* error) {
    if (error) {
        *error = "metal-backend-not-available";
    }
    return false;
}

#endif

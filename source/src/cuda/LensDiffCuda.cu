#include "LensDiffCuda.h"

#include "../core/LensDiffApertureImage.h"
#include "../core/LensDiffCpuReference.h"
#include "../core/LensDiffDiagnostics.h"
#include "../core/LensDiffPhase.h"
#include "../core/LensDiffSpectrum.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <string>
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

__global__ void accumulateWeightedRgbKernel(const float* srcR,
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

bool makeImageSpectrum(const float* src,
                       int width,
                       int height,
                       int paddedWidth,
                       int paddedHeight,
                       cufftHandle plan,
                       cudaStream_t stream,
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

    return checkCufft(cufftExecC2C(plan, out.ptr, out.ptr, CUFFT_FORWARD), "cufftExecC2C-forward-image", error);
}

bool makeKernelSpectrum(const LensDiffKernel& kernel,
                        int paddedWidth,
                        int paddedHeight,
                        cufftHandle plan,
                        cudaStream_t stream,
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

    return checkCufft(cufftExecC2C(plan, out.ptr, out.ptr, CUFFT_FORWARD), "cufftExecC2C-forward-kernel", error);
}

bool convolveSpectrumToPlane(const DeviceBuffer<cufftComplex>& imageSpectrum,
                             const DeviceBuffer<cufftComplex>& kernelSpectrum,
                             int width,
                             int height,
                             int paddedWidth,
                             int paddedHeight,
                             cufftHandle plan,
                             cudaStream_t stream,
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

    if (!checkCufft(cufftExecC2C(plan, tempSpectrum.ptr, tempSpectrum.ptr, CUFFT_INVERSE), "cufftExecC2C-inverse", error)) {
        return false;
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    const float scale = 1.0f / static_cast<float>(paddedWidth * paddedHeight);
    extractRealKernel<<<grid, block, 0, stream>>>(tempSpectrum.ptr, width, height, paddedWidth, scale, outPlane.ptr);
    return checkCuda(cudaGetLastError(), "extractRealKernel", error);
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
                              cudaStream_t stream,
                              std::vector<float>* shiftedRawPsf,
                              std::string* error) {
    if (shiftedRawPsf == nullptr) {
        if (error) *error = "missing-shifted-raw-psf-output";
        return false;
    }

    const std::size_t rawCount = static_cast<std::size_t>(rawPsfSize) * rawPsfSize;
    DeviceBuffer<float> deviceShiftedIntensity;
    DeviceBuffer<cufftComplex> deviceSpectrum;
    cufftHandle plan = 0;
    const bool usePhase = devicePhase != nullptr && devicePhase->ptr != nullptr;

    if (!deviceShiftedIntensity.allocate(rawCount) ||
        !deviceSpectrum.allocate(rawCount)) {
        if (error) *error = "cuda-alloc-raw-psf-build";
        return false;
    }

    if (!checkCuda(cudaMemsetAsync(deviceSpectrum.ptr, 0, rawCount * sizeof(cufftComplex), stream),
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
        deviceSpectrum.ptr);
    if (!checkCuda(cudaGetLastError(), "embedCenteredComplexPupilKernel", error)) {
        return false;
    }

    if (!createPlan(&plan, rawPsfSize, rawPsfSize, stream, error)) {
        return false;
    }

    const bool fftOk = checkCufft(cufftExecC2C(plan, deviceSpectrum.ptr, deviceSpectrum.ptr, CUFFT_FORWARD),
                                  "cufftExecC2C-forward-raw-psf",
                                  error);
    if (!fftOk) {
        cufftDestroy(plan);
        return false;
    }

    dim3 rawGrid((rawPsfSize + block.x - 1) / block.x, (rawPsfSize + block.y - 1) / block.y);
    extractShiftedIntensityKernel<<<rawGrid, block, 0, stream>>>(deviceSpectrum.ptr, rawPsfSize, deviceShiftedIntensity.ptr);
    if (!checkCuda(cudaGetLastError(), "extractShiftedIntensityKernel", error)) {
        cufftDestroy(plan);
        return false;
    }

    shiftedRawPsf->assign(rawCount, 0.0f);
    if (!checkCuda(cudaMemcpyAsync(shiftedRawPsf->data(),
                                   deviceShiftedIntensity.ptr,
                                   rawCount * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   stream),
                   "cudaMemcpyAsync-raw-psf-readback",
                   error)) {
        cufftDestroy(plan);
        return false;
    }

    if (!checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize-raw-psf-readback", error)) {
        cufftDestroy(plan);
        return false;
    }

    cufftDestroy(plan);

    float total = 0.0f;
    for (float value : *shiftedRawPsf) {
        total += value;
    }
    if (total > 0.0f) {
        const float invTotal = 1.0f / total;
        for (float& value : *shiftedRawPsf) {
            value *= invTotal;
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
                           const std::vector<float>& wavelengths,
                           const std::vector<LensDiffKernel>& baseKernels,
                           cudaStream_t stream,
                           LensDiffPsfBankCache* cache,
                           std::string* error) {
    if (cache == nullptr) {
        if (error) *error = "cuda-null-psf-cache";
        return false;
    }
    *cache = {};
    cache->valid = true;
    cache->key = key;
    cache->supportRadiusPx = 4;
    cache->pupilDisplay = pupil;
    cache->pupilDisplaySize = pupilSize;
    cache->phaseDisplay = phaseWaves;
    cache->phaseDisplaySize = pupilSize;
    cache->bins.clear();
    cache->bins.reserve(std::min(wavelengths.size(), baseKernels.size()));

    for (std::size_t i = 0; i < wavelengths.size() && i < baseKernels.size(); ++i) {
        const LensDiffKernel& baseKernel = baseKernels[i];
        const std::size_t kernelCount = baseKernel.values.size();
        DeviceBuffer<float> meanKernel;
        DeviceBuffer<float> shapedKernel;
        DeviceBuffer<float> structureKernel;
        KernelStatsHost stats;
        if (!buildAzimuthalMeanKernelOnCuda(baseKernel, stream, &meanKernel, &stats, error) ||
            !buildShapedKernelOnCuda(baseKernel, meanKernel, static_cast<float>(params.anisotropyEmphasis), stream, &shapedKernel, error) ||
            !buildStructureKernelOnCuda(shapedKernel, meanKernel, kernelCount, stream, &structureKernel, error)) {
            return false;
        }
        float totalEnergy = 0.0f;
        float globalPeak = 0.0f;
        const int maxKernelRadiusPx = std::max(4, ResolveLensDiffMaxKernelRadiusPx(params));
        if (!computeAdaptiveSupportStatsOnCuda(shapedKernel,
                                               baseKernel.size,
                                               maxKernelRadiusPx,
                                               stream,
                                               &stats,
                                               &totalEnergy,
                                               &globalPeak,
                                               error)) {
            return false;
        }
        const int effectiveRadius = paddedAdaptiveSupportRadiusHost(
            estimateAdaptiveSupportRadiusFromStats(stats, maxKernelRadiusPx, totalEnergy, globalPeak),
            maxKernelRadiusPx);

        LensDiffPsfBin bin {};
        bin.wavelengthNm = wavelengths[i];
        if (!downloadCroppedKernel(meanKernel, baseKernel.size, effectiveRadius, stream, &bin.core, error) ||
            !downloadCroppedKernel(shapedKernel, baseKernel.size, effectiveRadius, stream, &bin.full, error) ||
            !downloadCroppedKernel(structureKernel, baseKernel.size, effectiveRadius, stream, &bin.structure, error)) {
            return false;
        }
        cache->supportRadiusPx = std::max(cache->supportRadiusPx, effectiveRadius);
        cache->bins.push_back(std::move(bin));
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

    std::vector<float> rawPsf;
    if (!buildShiftedRawPsfOnCuda(devicePupil, hasPhase ? &devicePhase : nullptr, pupilSize, rawPsfSize, stream, &rawPsf, error)) {
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
    std::vector<LensDiffKernel> baseKernels;
    if (!buildBaseKernelsFromRawPsfOnCuda(rawPsf,
                                          rawPsfSize,
                                          wavelengths,
                                          scaleBase,
                                          maxKernelRadiusPx,
                                          stream,
                                          &baseKernels,
                                          error)) {
        return false;
    }

    return finalizePsfBankOnCuda(params,
                                 key,
                                 pupil,
                                 phaseWaves,
                                 pupilSize,
                                 wavelengths,
                                 baseKernels,
                                 stream,
                                 &cache,
                                 error);
}

bool ensurePsfBankCuda(const LensDiffParams& params,
                       LensDiffPsfBankCache& cache,
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

    if (!buildPsfBankGlobalOnlyCuda(params, cache, stream, error)) {
        return false;
    }
    if (!needFieldZones) {
        cache.fieldGridSize = 0;
        cache.fieldKey = {};
        cache.fieldZones.clear();
        return true;
    }

    LensDiffScopedTimer timer("cuda-field-zones");
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
            if (!buildPsfBankGlobalOnlyCuda(zone.resolvedParams, zoneCache, stream, error)) {
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
    if (!ensurePsfBankCuda(params, cache, stream, error)) {
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
    auto allocatePlaneSet = [&](PlaneSet& set) -> bool {
        return set.r.allocate(pixelCount) && set.g.allocate(pixelCount) && set.b.allocate(pixelCount);
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

    auto convolvePlaneSet = [&](const DeviceBuffer<float>& srcRPlane,
                                const DeviceBuffer<float>& srcGPlane,
                                const DeviceBuffer<float>& srcBPlane,
                                const LensDiffKernel& kernel,
                                PlaneSet& dst,
                                const char* stagePrefix) -> bool {
        const int paddedWidth = nextPowerOfTwo(width + kernel.size - 1);
        const int paddedHeight = nextPowerOfTwo(height + kernel.size - 1);
        cufftHandle plan = 0;
        DeviceBuffer<cufftComplex> tempSpectrum;
        DeviceBuffer<cufftComplex> imageSpecR;
        DeviceBuffer<cufftComplex> imageSpecG;
        DeviceBuffer<cufftComplex> imageSpecB;
        DeviceBuffer<cufftComplex> kernelSpec;
        DeviceBuffer<float> deviceKernel;
        if (!createPlan(&plan, paddedWidth, paddedHeight, stream, error)) {
            return false;
        }
        const bool ok = makeImageSpectrum(srcRPlane.ptr, width, height, paddedWidth, paddedHeight, plan, stream, imageSpecR, error) &&
                        makeImageSpectrum(srcGPlane.ptr, width, height, paddedWidth, paddedHeight, plan, stream, imageSpecG, error) &&
                        makeImageSpectrum(srcBPlane.ptr, width, height, paddedWidth, paddedHeight, plan, stream, imageSpecB, error) &&
                        makeKernelSpectrum(kernel, paddedWidth, paddedHeight, plan, stream, deviceKernel, kernelSpec, error) &&
                        convolveSpectrumToPlane(imageSpecR, kernelSpec, width, height, paddedWidth, paddedHeight, plan, stream, tempSpectrum, dst.r, error) &&
                        convolveSpectrumToPlane(imageSpecG, kernelSpec, width, height, paddedWidth, paddedHeight, plan, stream, tempSpectrum, dst.g, error) &&
                        convolveSpectrumToPlane(imageSpecB, kernelSpec, width, height, paddedWidth, paddedHeight, plan, stream, tempSpectrum, dst.b, error);
        cufftDestroy(plan);
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

            if (!splitMode && !convolvePlaneSet(redistributedR, redistributedG, redistributedB, bins.front().full, fullEffect, "cuda-convolve-zone-full")) {
                return false;
            }
            if (localNeedCore && !convolvePlaneSet(redistributedR, redistributedG, redistributedB, bins.front().core, *outCore, "cuda-convolve-zone-core")) {
                return false;
            }
            if (localNeedStructure && !convolvePlaneSet(redistributedR, redistributedG, redistributedB, bins.front().structure, *outStructure, "cuda-convolve-zone-structure")) {
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
        cufftHandle plan = 0;
        DeviceBuffer<cufftComplex> tempSpectrum;
        DeviceBuffer<cufftComplex> driverSpectrum;
        DeviceBuffer<cufftComplex> tempKernelSpectrum;
        DeviceBuffer<float> deviceKernel;
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
                if (!makeKernelSpectrum(kernels[static_cast<std::size_t>(i)],
                                        paddedWidth,
                                        paddedHeight,
                                        plan,
                                        stream,
                                        deviceKernel,
                                        tempKernelSpectrum,
                                        error) ||
                    !convolveSpectrumToPlane(driverSpectrum,
                                             tempKernelSpectrum,
                                             width,
                                             height,
                                             paddedWidth,
                                             paddedHeight,
                                             plan,
                                             stream,
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

        if (!createPlan(&plan, paddedWidth, paddedHeight, stream, error)) {
            return false;
        }
        bool ok = makeImageSpectrum(redistributedDriver.ptr, width, height, paddedWidth, paddedHeight, plan, stream, driverSpectrum, error);
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
            cufftDestroy(plan);
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
            ok = mapBins(coreBins, *outCore);
        }
        if (ok && localNeedStructure) {
            ok = mapBins(structureBins, *outStructure);
        }
        if (!ok) {
            cufftDestroy(plan);
            return false;
        }

        if (splitMode) {
            if (params.coreShoulder > 0.0) {
                applyShoulderKernel<<<flatGrid, flatBlock, 0, stream>>>(
                    outCore->r.ptr, outCore->g.ptr, outCore->b.ptr, pixelCount, static_cast<float>(params.coreShoulder));
                if (!checkCuda(cudaGetLastError(), "applyShoulderKernel-zone-core-spectral", error)) {
                    cufftDestroy(plan);
                    return false;
                }
            }
            if (params.structureShoulder > 0.0) {
                applyShoulderKernel<<<flatGrid, flatBlock, 0, stream>>>(
                    outStructure->r.ptr, outStructure->g.ptr, outStructure->b.ptr, pixelCount, static_cast<float>(params.structureShoulder));
                if (!checkCuda(cudaGetLastError(), "applyShoulderKernel-zone-structure-spectral", error)) {
                    cufftDestroy(plan);
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
        cufftDestroy(plan);
        return ok;
    };

    PlaneSet effect;
    PlaneSet coreEffect;
    PlaneSet structureEffect;
    if (cache.fieldZones.empty()) {
        if (!renderFromBins(cache.bins,
                            effect,
                            needCore ? &coreEffect : nullptr,
                            needStructure ? &structureEffect : nullptr)) {
            return false;
        }
    } else {
        if (!allocatePlaneSet(effect) ||
            (needCore && !allocatePlaneSet(coreEffect)) ||
            (needStructure && !allocatePlaneSet(structureEffect))) {
            if (error) *error = "cuda-alloc-field-accum";
            return false;
        }
        if (!clearPlaneSet(effect, "cudaMemsetAsync-field-effect") ||
            (needCore && !clearPlaneSet(coreEffect, "cudaMemsetAsync-field-core")) ||
            (needStructure && !clearPlaneSet(structureEffect, "cudaMemsetAsync-field-structure"))) {
            return false;
        }
        for (const auto& zone : cache.fieldZones) {
            PlaneSet zoneEffect;
            PlaneSet zoneCore;
            PlaneSet zoneStructure;
            if (!renderFromBins(zone.bins,
                                zoneEffect,
                                needCore ? &zoneCore : nullptr,
                                needStructure ? &zoneStructure : nullptr)) {
                return false;
            }
            accumulateWeightedRgbKernel<<<grid2d, block2d, 0, stream>>>(
                zoneEffect.r.ptr, zoneEffect.g.ptr, zoneEffect.b.ptr,
                width, height, zone.zoneX, zone.zoneY, 1.0f,
                effect.r.ptr, effect.g.ptr, effect.b.ptr);
            if (!checkCuda(cudaGetLastError(), "accumulateWeightedRgbKernel-effect", error)) {
                return false;
            }
            if (needCore) {
                accumulateWeightedRgbKernel<<<grid2d, block2d, 0, stream>>>(
                    zoneCore.r.ptr, zoneCore.g.ptr, zoneCore.b.ptr,
                    width, height, zone.zoneX, zone.zoneY, 1.0f,
                    coreEffect.r.ptr, coreEffect.g.ptr, coreEffect.b.ptr);
                if (!checkCuda(cudaGetLastError(), "accumulateWeightedRgbKernel-core", error)) {
                    return false;
                }
            }
            if (needStructure) {
                accumulateWeightedRgbKernel<<<grid2d, block2d, 0, stream>>>(
                    zoneStructure.r.ptr, zoneStructure.g.ptr, zoneStructure.b.ptr,
                    width, height, zone.zoneX, zone.zoneY, 1.0f,
                    structureEffect.r.ptr, structureEffect.g.ptr, structureEffect.b.ptr);
                if (!checkCuda(cudaGetLastError(), "accumulateWeightedRgbKernel-structure", error)) {
                    return false;
                }
            }
        }
    }

    if (params.energyMode == LensDiffEnergyMode::Preserve) {
        float inputEnergy = 0.0f;
        float effectEnergy = 0.0f;
        if (!computeEnergySum(redistributedR, redistributedG, redistributedB, pixelCount, stream, &inputEnergy, error) ||
            !computeEnergySum(effect.r, effect.g, effect.b, pixelCount, stream, &effectEnergy, error)) {
            return false;
        }
        if (effectEnergy > 1e-6f) {
            const float scale = inputEnergy / effectEnergy;
            scaleRgbKernel<<<flatGrid, flatBlock, 0, stream>>>(effect.r.ptr, effect.g.ptr, effect.b.ptr, pixelCount, scale);
            if (!checkCuda(cudaGetLastError(), "scaleRgbKernel-effect-preserve", error)) {
                return false;
            }
        }
    }

    PlaneSet scatterPreview;
    const double scatterRadiusPx = ResolveLensDiffScatterRadiusPx(params);
    const bool scatterActive = params.scatterAmount > 1e-6 && scatterRadiusPx > 0.25;
    if (scatterActive || params.debugView == LensDiffDebugView::Scatter) {
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
    }

    PlaneSet creativeFringePreview;
    const double creativeFringePx = ResolveLensDiffCreativeFringePx(params);
    const bool creativeFringeActive = creativeFringePx > 1e-6;
    if (creativeFringeActive || params.debugView == LensDiffDebugView::CreativeFringe) {
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

    if (params.debugView == LensDiffDebugView::Core) {
        if (resolutionAwareActive) {
            PlaneSet display;
            if (!resamplePlaneSetToNative(coreEffect, display, "cuda-resample-core-display")) {
                return false;
            }
            packRgbDebugKernel<<<nativeFlatGrid, flatBlock, 0, stream>>>(display.r.ptr, display.g.ptr, display.b.ptr, nullptr, nativePixelCount, 1.0f, packedOutput.ptr);
        } else {
            packRgbDebugKernel<<<flatGrid, flatBlock, 0, stream>>>(coreEffect.r.ptr, coreEffect.g.ptr, coreEffect.b.ptr, nullptr, pixelCount, 1.0f, packedOutput.ptr);
        }
        if (!checkCuda(cudaGetLastError(), "packRgbDebugKernel-core", error)) {
            return false;
        }
    } else if (params.debugView == LensDiffDebugView::Structure) {
        if (resolutionAwareActive) {
            PlaneSet display;
            if (!resamplePlaneSetToNative(structureEffect, display, "cuda-resample-structure-display")) {
                return false;
            }
            packRgbDebugKernel<<<nativeFlatGrid, flatBlock, 0, stream>>>(display.r.ptr, display.g.ptr, display.b.ptr, nullptr, nativePixelCount, 1.0f, packedOutput.ptr);
        } else {
            packRgbDebugKernel<<<flatGrid, flatBlock, 0, stream>>>(structureEffect.r.ptr, structureEffect.g.ptr, structureEffect.b.ptr, nullptr, pixelCount, 1.0f, packedOutput.ptr);
        }
        if (!checkCuda(cudaGetLastError(), "packRgbDebugKernel-structure", error)) {
            return false;
        }
    } else if (params.debugView == LensDiffDebugView::Effect) {
        if (resolutionAwareActive) {
            PlaneSet display;
            if (!resamplePlaneSetToNative(effect, display, "cuda-resample-effect-display")) {
                return false;
            }
            packRgbDebugKernel<<<nativeFlatGrid, flatBlock, 0, stream>>>(display.r.ptr, display.g.ptr, display.b.ptr, nullptr, nativePixelCount, 1.0f, packedOutput.ptr);
        } else {
            packRgbDebugKernel<<<flatGrid, flatBlock, 0, stream>>>(effect.r.ptr, effect.g.ptr, effect.b.ptr, nullptr, pixelCount, 1.0f, packedOutput.ptr);
        }
        if (!checkCuda(cudaGetLastError(), "packRgbDebugKernel-effect", error)) {
            return false;
        }
    } else if (params.debugView == LensDiffDebugView::Scatter) {
        if (resolutionAwareActive) {
            PlaneSet display;
            if (!resamplePlaneSetToNative(scatterPreview, display, "cuda-resample-scatter-display")) {
                return false;
            }
            packRgbDebugKernel<<<nativeFlatGrid, flatBlock, 0, stream>>>(display.r.ptr, display.g.ptr, display.b.ptr, nullptr, nativePixelCount, 1.0f, packedOutput.ptr);
        } else {
            packRgbDebugKernel<<<flatGrid, flatBlock, 0, stream>>>(scatterPreview.r.ptr, scatterPreview.g.ptr, scatterPreview.b.ptr, nullptr, pixelCount, 1.0f, packedOutput.ptr);
        }
        if (!checkCuda(cudaGetLastError(), "packRgbDebugKernel-scatter", error)) {
            return false;
        }
    } else if (params.debugView == LensDiffDebugView::CreativeFringe) {
        if (resolutionAwareActive) {
            PlaneSet display;
            if (!resamplePlaneSetToNative(creativeFringePreview, display, "cuda-resample-creative-display")) {
                return false;
            }
            packRgbDebugKernel<<<nativeFlatGrid, flatBlock, 0, stream>>>(display.r.ptr, display.g.ptr, display.b.ptr, nullptr, nativePixelCount, 1.0f, packedOutput.ptr);
        } else {
            packRgbDebugKernel<<<flatGrid, flatBlock, 0, stream>>>(creativeFringePreview.r.ptr, creativeFringePreview.g.ptr, creativeFringePreview.b.ptr, nullptr, pixelCount, 1.0f, packedOutput.ptr);
        }
        if (!checkCuda(cudaGetLastError(), "packRgbDebugKernel-creative-fringe", error)) {
            return false;
        }
    } else {
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
            if (!resamplePlaneToNative(redistributedR, redistributedRDisplay, "cuda-resample-redistributed-r") ||
                !resamplePlaneToNative(redistributedG, redistributedGDisplay, "cuda-resample-redistributed-g") ||
                !resamplePlaneToNative(redistributedB, redistributedBDisplay, "cuda-resample-redistributed-b") ||
                !resamplePlaneToNative(effect.r, effectRDisplay, "cuda-resample-effect-r") ||
                !resamplePlaneToNative(effect.g, effectGDisplay, "cuda-resample-effect-g") ||
                !resamplePlaneToNative(effect.b, effectBDisplay, "cuda-resample-effect-b")) {
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
        if (!checkCuda(cudaGetLastError(), "compositeFinalKernel", error)) {
            return false;
        }
    }

    return copyPackedToDestination(request, packedOutput.ptr, nativeWidth, nativeHeight, stream, cudaMemcpyDeviceToDevice, error);
}

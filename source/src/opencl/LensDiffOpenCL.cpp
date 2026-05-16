#include "LensDiffOpenCL.h"
#include "LensDiffOpenCLVkFFT.h"

#include "../core/LensDiffCpuReference.h"
#include "../core/LensDiffDiagnostics.h"
#include "../core/LensDiffSpectrum.h"
#include "../core/LensDiffTransfer.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kOpenCLWorkgroup1D = 256;
constexpr int kLensDiffDebugFinal = 0;
constexpr int kLensDiffDebugSelection = 1;
constexpr int kLensDiffDebugPupil = 2;
constexpr int kLensDiffDebugPsf = 3;
constexpr int kLensDiffDebugOtf = 4;
constexpr int kLensDiffDebugCore = 5;
constexpr int kLensDiffDebugStructure = 6;
constexpr int kLensDiffDebugEffect = 7;
constexpr int kLensDiffDebugPhase = 8;
constexpr int kLensDiffDebugPhaseEdge = 9;
constexpr int kLensDiffDebugFieldPsf = 10;
constexpr int kLensDiffDebugChromaticSplit = 11;
constexpr int kLensDiffDebugCreativeFringe = 12;
constexpr int kLensDiffDebugScatter = 13;

constexpr std::size_t kLensDiffOpenCLSourcePartCount = 3;
const char* const kLensDiffOpenCLSourceParts[kLensDiffOpenCLSourcePartCount] = {R"CLC(
inline float saturateSafe(float v) { return fmin(fmax(v, 0.0f), 1.0f); }
inline float safeLuma3(float3 rgb) { return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z; }
inline float softShoulderValue(float v, float shoulder) {
    if (shoulder <= 0.0f) return fmax(v, 0.0f);
    const float x = fmax(v, 0.0f);
    return shoulder * (1.0f - exp(-x / shoulder));
}
inline float decodeDavinciIntermediate(float x) {
    const float kA = 0.0075f;
    const float kB = 7.0f;
    const float kC = 0.07329248f;
    const float kM = 10.44426855f;
    const float kLogCut = 0.02740668f;
    return x <= kLogCut ? (x / kM) : (pow(2.0f, x / kC - kB) - kA);
}
inline float encodeDavinciIntermediate(float x) {
    const float kA = 0.0075f;
    const float kB = 7.0f;
    const float kC = 0.07329248f;
    const float kM = 10.44426855f;
    const float kLinCut = 0.00262409f;
    return x <= kLinCut ? (x * kM) : ((log2(fmax(x, 0.0f) + kA) + kB) * kC);
}
inline float4 decodeTransfer(float4 rgba, int transfer) {
    if (transfer == 1) {
        rgba.x = decodeDavinciIntermediate(rgba.x);
        rgba.y = decodeDavinciIntermediate(rgba.y);
        rgba.z = decodeDavinciIntermediate(rgba.z);
    }
    return rgba;
}
inline float4 encodeTransfer(float4 rgba, int transfer) {
    if (transfer == 1) {
        rgba.x = encodeDavinciIntermediate(rgba.x);
        rgba.y = encodeDavinciIntermediate(rgba.y);
        rgba.z = encodeDavinciIntermediate(rgba.z);
    }
    return rgba;
}
inline float4 sampleRgbaLinear(__global const float4* image, int width, int height, float fx, float fy) {
    const float x = clamp(fx, 0.0f, (float)max(0, width - 1));
    const float y = clamp(fy, 0.0f, (float)max(0, height - 1));
    const int x0 = (int)floor(x);
    const int y0 = (int)floor(y);
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float tx = x - (float)x0;
    const float ty = y - (float)y0;
    const float4 p00 = image[y0 * width + x0];
    const float4 p10 = image[y0 * width + x1];
    const float4 p01 = image[y1 * width + x0];
    const float4 p11 = image[y1 * width + x1];
    return mix(mix(p00, p10, tx), mix(p01, p11, tx), ty);
}

__kernel void lensDiffDecodeBuffer(__global const uchar* src,
                                   __global float4* dst,
                                   int width,
                                   int height,
                                   long rowBytes,
                                   int transfer) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    __global const float* pixel = (__global const float*)(src + (long)y * rowBytes + (long)x * 16L);
    dst[y * width + x] = decodeTransfer((float4)(pixel[0], pixel[1], pixel[2], pixel[3]), transfer);
}

__kernel void lensDiffDecodeImage(read_only image2d_t src,
                                  __global float4* dst,
                                  int width,
                                  int height,
                                  int transfer) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    const sampler_t s = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    dst[y * width + x] = decodeTransfer(read_imagef(src, s, (int2)(x, y)), transfer);
}

__kernel void lensDiffPrepare(__global const float4* nativeSrc,
                              __global float4* workingSrc,
                              __global float* mask,
                              __global float4* redistributed,
                              __global float* driver,
                              int nativeWidth,
                              int nativeHeight,
                              int width,
                              int height,
                              int extractionMode,
                              float thresholdStops,
                              float softnessStops,
                              float pointEmphasis,
                              float redistributionScale) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    const float sx = width == nativeWidth ? (float)x : ((float)x + 0.5f) * (float)nativeWidth / (float)width - 0.5f;
    const float sy = height == nativeHeight ? (float)y : ((float)y + 0.5f) * (float)nativeHeight / (float)height - 0.5f;
    const float4 src = sampleRgbaLinear(nativeSrc, nativeWidth, nativeHeight, sx, sy);
    const float maxRgb = fmax(src.x, fmax(src.y, src.z));
    const float signal = extractionMode == 1 ? safeLuma3(src.xyz) : maxRgb;
    const float softness = fmax(0.01f, softnessStops);
    float gate = 0.0f;
    if (signal > 0.0f) {
        const float stops = log2(fmax(signal, 1.0e-6f) / 0.18f);
        const float edge0 = thresholdStops - softness * 0.5f;
        const float edge1 = thresholdStops + softness * 0.5f;
        const float t = saturateSafe((stops - edge0) / fmax(edge1 - edge0, 1.0e-4f));
        gate = t * t * (3.0f - 2.0f * t);
    }
    const float thresholdLinear = 0.18f * exp2(thresholdStops);
    const float pointBoost = 1.0f + pointEmphasis * fmax(0.0f, maxRgb / fmax(thresholdLinear, 1.0e-4f) - 1.0f);
    const float m = saturateSafe(gate * pointBoost);
    const int index = y * width + x;
    workingSrc[index] = src;
    mask[index] = m;
    redistributed[index] = (float4)(src.x * m * redistributionScale,
                                    src.y * m * redistributionScale,
                                    src.z * m * redistributionScale,
                                    src.w);
    driver[index] = signal * m * redistributionScale;
}

__kernel void lensDiffClearRgba(__global float4* dst, int count) {
    const int i = (int)get_global_id(0);
    if (i < count) dst[i] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
}

__kernel void lensDiffConvolveRgba(__global const float4* src,
                                   __global const float* kernelValues,
                                   __global float4* dst,
                                   int width,
                                   int height,
                                   int kernelSize,
                                   float shoulder) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    const int center = kernelSize / 2;
    float3 sum = (float3)(0.0f, 0.0f, 0.0f);
    for (int ky = 0; ky < kernelSize; ++ky) {
        const int sy = y + center - ky;
        if (sy < 0 || sy >= height) continue;
        for (int kx = 0; kx < kernelSize; ++kx) {
            const int sx = x + center - kx;
            if (sx < 0 || sx >= width) continue;
            const float k = kernelValues[ky * kernelSize + kx];
            sum += src[sy * width + sx].xyz * k;
        }
    }
    if (shoulder > 0.0f) {
        sum.x = softShoulderValue(sum.x, shoulder);
        sum.y = softShoulderValue(sum.y, shoulder);
        sum.z = softShoulderValue(sum.z, shoulder);
    } else {
        sum = fmax(sum, (float3)(0.0f));
    }
    dst[y * width + x] = (float4)(sum.x, sum.y, sum.z, src[y * width + x].w);
}

__kernel void lensDiffConvolveScalarPlane(__global const float* src,
                                          __global const float* kernelValues,
                                          __global float* planes,
                                          int width,
                                          int height,
                                          int kernelSize,
                                          int planeIndex,
                                          int planeStride) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    const int center = kernelSize / 2;
    float sum = 0.0f;
    for (int ky = 0; ky < kernelSize; ++ky) {
        const int sy = y + center - ky;
        if (sy < 0 || sy >= height) continue;
        for (int kx = 0; kx < kernelSize; ++kx) {
            const int sx = x + center - kx;
            if (sx < 0 || sx >= width) continue;
            sum += src[sy * width + sx] * kernelValues[ky * kernelSize + kx];
        }
    }
    planes[planeIndex * planeStride + y * width + x] = fmax(sum, 0.0f);
}

)CLC",
R"CLC(
__kernel void lensDiffPadRgbaToComplexStack(__global const float4* src,
                                            __global float2* spectrum,
                                            int width,
                                            int height,
                                            int paddedSize) {
    const int x = (int)get_global_id(0);
    const int gy = (int)get_global_id(1);
    const int channel = gy / paddedSize;
    const int y = gy - channel * paddedSize;
    if (x >= paddedSize || y >= paddedSize || channel >= 3) return;
    float value = 0.0f;
    if (x < width && y < height) {
        const float4 rgba = src[y * width + x];
        value = channel == 0 ? rgba.x : (channel == 1 ? rgba.y : rgba.z);
    }
    const int slice = paddedSize * paddedSize;
    spectrum[channel * slice + y * paddedSize + x] = (float2)(value, 0.0f);
}

__kernel void lensDiffPadScalarToComplex(__global const float* src,
                                         __global float2* spectrum,
                                         int width,
                                         int height,
                                         int paddedSize) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= paddedSize || y >= paddedSize) return;
    const float value = (x < width && y < height) ? src[y * width + x] : 0.0f;
    spectrum[y * paddedSize + x] = (float2)(value, 0.0f);
}

__kernel void lensDiffScatterKernelToComplex(__global const float* kernelValues,
                                             __global float2* spectrum,
                                             int kernelSize,
                                             int paddedSize) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= kernelSize || y >= kernelSize) return;
    const int center = kernelSize / 2;
    const int px = (x - center + paddedSize) % paddedSize;
    const int py = (y - center + paddedSize) % paddedSize;
    spectrum[py * paddedSize + px] = (float2)(kernelValues[y * kernelSize + x], 0.0f);
}

__kernel void lensDiffMultiplyComplexBroadcast(__global const float2* imageSpectrum,
                                               __global const float2* kernelSpectrum,
                                               __global float2* dst,
                                               int paddedCount,
                                               int batchCount) {
    const int x = (int)get_global_id(0);
    const int batch = (int)get_global_id(1);
    if (x >= paddedCount || batch >= batchCount) return;
    const float2 a = imageSpectrum[batch * paddedCount + x];
    const float2 b = kernelSpectrum[x];
    dst[batch * paddedCount + x] = (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__kernel void lensDiffMultiplyComplex(__global const float2* imageSpectrum,
                                      __global const float2* kernelSpectrum,
                                      __global float2* dst,
                                      int paddedCount) {
    const int x = (int)get_global_id(0);
    if (x >= paddedCount) return;
    const float2 a = imageSpectrum[x];
    const float2 b = kernelSpectrum[x];
    dst[x] = (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__kernel void lensDiffExtractRgbaStack(__global const float2* spectrum,
                                       __global const float4* alphaSource,
                                       __global float4* dst,
                                       int width,
                                       int height,
                                       int paddedSize,
                                       float scale,
                                       float shoulder) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    const int slice = paddedSize * paddedSize;
    const int srcIndex = y * paddedSize + x;
    float3 rgb = (float3)(fmax(0.0f, spectrum[srcIndex].x * scale),
                          fmax(0.0f, spectrum[slice + srcIndex].x * scale),
                          fmax(0.0f, spectrum[slice * 2 + srcIndex].x * scale));
    if (shoulder > 0.0f) {
        rgb.x = softShoulderValue(rgb.x, shoulder);
        rgb.y = softShoulderValue(rgb.y, shoulder);
        rgb.z = softShoulderValue(rgb.z, shoulder);
    }
    dst[y * width + x] = (float4)(rgb.x, rgb.y, rgb.z, alphaSource[y * width + x].w);
}

__kernel void lensDiffExtractScalarPlane(__global const float2* spectrum,
                                         __global float* planes,
                                         int width,
                                         int height,
                                         int paddedSize,
                                         int planeIndex,
                                         int planeStride,
                                         float scale) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    planes[planeIndex * planeStride + y * width + x] =
        fmax(0.0f, spectrum[y * paddedSize + x].x * scale);
}

__kernel void lensDiffMapSpectral(__global const float* planes,
                                  __global float4* dst,
                                  __global const float* naturalMatrix,
                                  __global const float* styleMatrix,
                                  int width,
                                  int height,
                                  int planeStride,
                                  int binCount,
                                  float spectrumForce,
                                  float spectrumSaturation,
                                  int chromaticAffectsLuma) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    const int idx = y * width + x;
    float3 natural = (float3)(0.0f);
    float3 styled = (float3)(0.0f);
    for (int i = 0; i < binCount && i < 9; ++i) {
        const float v = planes[i * planeStride + idx];
        natural.x += naturalMatrix[i] * v;
        natural.y += naturalMatrix[9 + i] * v;
        natural.z += naturalMatrix[18 + i] * v;
        styled.x += styleMatrix[i] * v;
        styled.y += styleMatrix[9 + i] * v;
        styled.z += styleMatrix[18 + i] * v;
    }
    float3 rgb = mix(natural, styled, saturateSafe(spectrumForce));
    rgb = fmax(rgb, (float3)(0.0f));
    const float gray = safeLuma3(rgb);
    rgb = (float3)(gray) + (rgb - (float3)(gray)) * fmax(0.0f, spectrumSaturation);
    if (!chromaticAffectsLuma) {
        const float target = safeLuma3(natural);
        const float current = safeLuma3(rgb);
        if (current > 1.0e-6f) rgb *= target / current;
    }
    dst[idx] = (float4)(rgb.x, rgb.y, rgb.z, 1.0f);
}

__kernel void lensDiffCombineSplit(__global const float4* core,
                                   __global const float4* structure,
                                   __global float4* dst,
                                   int count,
                                   float coreGain,
                                   float structureGain) {
    const int i = (int)get_global_id(0);
    if (i >= count) return;
    const float3 rgb = core[i].xyz * fmax(coreGain, 0.0f) + structure[i].xyz * fmax(structureGain, 0.0f);
    dst[i] = (float4)(rgb.x, rgb.y, rgb.z, core[i].w);
}

__kernel void lensDiffApplyShoulder(__global float4* image,
                                    int count,
                                    float shoulder) {
    const int i = (int)get_global_id(0);
    if (i >= count || shoulder <= 0.0f) return;
    float4 rgba = image[i];
    rgba.x = softShoulderValue(rgba.x, shoulder);
    rgba.y = softShoulderValue(rgba.y, shoulder);
    rgba.z = softShoulderValue(rgba.z, shoulder);
    image[i] = rgba;
}

__kernel void lensDiffAccumulateWeighted(__global const float4* src,
                                         __global float4* dst,
                                         int width,
                                         int height,
                                         int zoneX,
                                         int zoneY) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    const float denomX = (float)max(1, width - 1);
    const float denomY = (float)max(1, height - 1);
    const float px = ((float)x / denomX) * 2.0f;
    const float py = ((float)y / denomY) * 2.0f;
    const float wx = fmax(0.0f, 1.0f - fabs(px - (float)zoneX));
    const float wy = fmax(0.0f, 1.0f - fabs(py - (float)zoneY));
    const float w = wx * wy;
    if (w <= 0.0f) return;
    const int idx = y * width + x;
    const float4 s = src[idx];
    const float4 d = dst[idx];
    dst[idx] = (float4)(d.x + s.x * w, d.y + s.y * w, d.z + s.z * w, s.w);
}

)CLC",
R"CLC(
__kernel void lensDiffReduceLuma(__global const float4* src,
                                 __global float* partial,
                                 int count) {
    __local float localSum[256];
    const int gid = (int)get_global_id(0);
    const int lid = (int)get_local_id(0);
    const int group = (int)get_group_id(0);
    float v = 0.0f;
    if (gid < count) v = safeLuma3(src[gid].xyz);
    localSum[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) localSum[lid] += localSum[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[group] = localSum[0];
}

__kernel void lensDiffReduceFloat(__global const float* src,
                                  __global float* partial,
                                  int count) {
    __local float localSum[256];
    const int gid = (int)get_global_id(0);
    const int lid = (int)get_local_id(0);
    const int group = (int)get_group_id(0);
    localSum[lid] = gid < count ? src[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) localSum[lid] += localSum[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[group] = localSum[0];
}

__kernel void lensDiffScaleRgba(__global float4* image, int count, float scale) {
    const int i = (int)get_global_id(0);
    if (i >= count) return;
    image[i].xyz *= scale;
}

__kernel void lensDiffAddImages(__global const float4* a,
                                __global const float4* b,
                                __global float4* dst,
                                int count,
                                float aGain,
                                float bGain) {
    const int i = (int)get_global_id(0);
    if (i >= count) return;
    const float3 rgb = a[i].xyz * aGain + b[i].xyz * bGain;
    dst[i] = (float4)(rgb.x, rgb.y, rgb.z, a[i].w);
}

__kernel void lensDiffCreativeFringe(__global const float4* src,
                                     __global float4* dst,
                                     __global float4* preview,
                                     int width,
                                     int height,
                                     float amount) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    const int idx = y * width + x;
    if (amount <= 1.0e-6f) {
        dst[idx] = src[idx];
        preview[idx] = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }
    const float cx = (float)max(0, width - 1) * 0.5f;
    const float cy = (float)max(0, height - 1) * 0.5f;
    const float dx = (float)x - cx;
    const float dy = (float)y - cy;
    const float len = hypot(dx, dy);
    const float invLen = len > 1.0e-6f ? 1.0f / len : 0.0f;
    const float sx = dx * invLen * amount;
    const float sy = dy * invLen * amount;
    const float4 red = sampleRgbaLinear(src, width, height, (float)x + sx, (float)y + sy);
    const float4 green = src[idx];
    const float4 blue = sampleRgbaLinear(src, width, height, (float)x - sx, (float)y - sy);
    const float4 out = (float4)(red.x, green.y, blue.z, green.w);
    dst[idx] = out;
    preview[idx] = (float4)(fabs(out.x - green.x), fabs(out.y - green.y), fabs(out.z - green.z), 1.0f);
}

__kernel void lensDiffComposite(__global const float4* linearSrcNative,
                                __global const float4* redistributed,
                                __global const float4* effect,
                                __global float4* finalImage,
                                int nativeWidth,
                                int nativeHeight,
                                int width,
                                int height,
                                float effectGain,
                                float coreCompensation,
                                float maxRedistributedSubtractScale) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= nativeWidth || y >= nativeHeight) return;
    const float sx = width == nativeWidth ? (float)x : ((float)x + 0.5f) * (float)width / (float)nativeWidth - 0.5f;
    const float sy = height == nativeHeight ? (float)y : ((float)y + 0.5f) * (float)height / (float)nativeHeight - 0.5f;
    const float4 src = linearSrcNative[y * nativeWidth + x];
    const float4 redist = sampleRgbaLinear(redistributed, width, height, sx, sy);
    const float4 eff = sampleRgbaLinear(effect, width, height, sx, sy);
    const float3 raw = fmax((float3)(0.0f), src.xyz - coreCompensation * redist.xyz + effectGain * eff.xyz);
    const float3 floorRgb = fmax((float3)(0.0f), src.xyz - redist.xyz * maxRedistributedSubtractScale);
    const float3 out = fmax(raw, floorRgb);
    finalImage[y * nativeWidth + x] = (float4)(out.x, out.y, out.z, src.w);
}

__kernel void lensDiffPackBuffer(__global const float4* image,
                                 __global const float4* linearSrcNative,
                                 __global uchar* dst,
                                 int nativeWidth,
                                 int nativeHeight,
                                 long rowBytes,
                                 int renderX1,
                                 int renderY1,
                                 int renderWidth,
                                 int renderHeight,
                                 int srcBoundsX1,
                                 int srcBoundsY1,
                                 int dstBoundsX1,
                                 int dstBoundsY1,
                                 int transfer,
                                 int encodeFinal,
                                 int preserveSourceAlpha) {
    const int gx = (int)get_global_id(0);
    const int gy = (int)get_global_id(1);
    if (gx >= renderWidth || gy >= renderHeight) return;
    const int hostX = renderX1 + gx;
    const int hostY = renderY1 + gy;
    const int sx = hostX - srcBoundsX1;
    const int sy = hostY - srcBoundsY1;
    if (sx < 0 || sy < 0 || sx >= nativeWidth || sy >= nativeHeight) return;
    float4 rgba = image[sy * nativeWidth + sx];
    if (encodeFinal) rgba = encodeTransfer(rgba, transfer);
    if (preserveSourceAlpha) rgba.w = linearSrcNative[sy * nativeWidth + sx].w;
    __global float* out = (__global float*)(dst + (long)(hostY - dstBoundsY1) * rowBytes + (long)(hostX - dstBoundsX1) * 16L);
    out[0] = rgba.x; out[1] = rgba.y; out[2] = rgba.z; out[3] = rgba.w;
}

__kernel void lensDiffPackImage(__global const float4* image,
                                __global const float4* linearSrcNative,
                                write_only image2d_t dst,
                                int nativeWidth,
                                int nativeHeight,
                                int renderX1,
                                int renderY1,
                                int renderWidth,
                                int renderHeight,
                                int srcBoundsX1,
                                int srcBoundsY1,
                                int dstBoundsX1,
                                int dstBoundsY1,
                                int transfer,
                                int encodeFinal,
                                int preserveSourceAlpha) {
    const int gx = (int)get_global_id(0);
    const int gy = (int)get_global_id(1);
    if (gx >= renderWidth || gy >= renderHeight) return;
    const int hostX = renderX1 + gx;
    const int hostY = renderY1 + gy;
    const int sx = hostX - srcBoundsX1;
    const int sy = hostY - srcBoundsY1;
    if (sx < 0 || sy < 0 || sx >= nativeWidth || sy >= nativeHeight) return;
    float4 rgba = image[sy * nativeWidth + sx];
    if (encodeFinal) rgba = encodeTransfer(rgba, transfer);
    if (preserveSourceAlpha) rgba.w = linearSrcNative[sy * nativeWidth + sx].w;
    write_imagef(dst, (int2)(hostX - dstBoundsX1, hostY - dstBoundsY1), rgba);
}

__kernel void lensDiffGrayToRgba(__global const float* gray,
                                 __global float4* rgba,
                                 int width,
                                 int height) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;
    const float v = saturateSafe(gray[y * width + x]);
    rgba[y * width + x] = (float4)(v, v, v, 1.0f);
}

inline float sampleGrayLinear(__global const float* gray, int width, int height, float fx, float fy) {
    const float x = clamp(fx, 0.0f, (float)max(0, width - 1));
    const float y = clamp(fy, 0.0f, (float)max(0, height - 1));
    const int x0 = (int)floor(x);
    const int y0 = (int)floor(y);
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float tx = x - (float)x0;
    const float ty = y - (float)y0;
    const float p00 = gray[y0 * width + x0];
    const float p10 = gray[y0 * width + x1];
    const float p01 = gray[y1 * width + x0];
    const float p11 = gray[y1 * width + x1];
    return mix(mix(p00, p10, tx), mix(p01, p11, tx), ty);
}

__kernel void lensDiffGrayToRgbaResample(__global const float* gray,
                                         __global float4* rgba,
                                         int srcWidth,
                                         int srcHeight,
                                         int dstWidth,
                                         int dstHeight) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= dstWidth || y >= dstHeight) return;
    const float sx = srcWidth == dstWidth ? (float)x : ((float)x + 0.5f) * (float)srcWidth / (float)dstWidth - 0.5f;
    const float sy = srcHeight == dstHeight ? (float)y : ((float)y + 0.5f) * (float)srcHeight / (float)dstHeight - 0.5f;
    const float v = saturateSafe(sampleGrayLinear(gray, srcWidth, srcHeight, sx, sy));
    rgba[y * dstWidth + x] = (float4)(v, v, v, 1.0f);
}
)CLC"};

struct ClMemDeleter {
    void operator()(cl_mem value) const noexcept {
        if (value != nullptr) {
            clReleaseMemObject(value);
        }
    }
};

struct ClKernelDeleter {
    void operator()(cl_kernel value) const noexcept {
        if (value != nullptr) {
            clReleaseKernel(value);
        }
    }
};

using UniqueMem = std::unique_ptr<std::remove_pointer<cl_mem>::type, ClMemDeleter>;
using UniqueKernel = std::unique_ptr<std::remove_pointer<cl_kernel>::type, ClKernelDeleter>;

std::string clErrorText(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        default: break;
    }
    std::ostringstream os;
    os << "CL_ERROR_" << static_cast<int>(err);
    return os.str();
}

bool setError(std::string* error, const std::string& text) {
    if (error != nullptr) {
        *error = text;
    }
    return false;
}

bool checkCl(cl_int err, const char* what, std::string* error) {
    if (err == CL_SUCCESS) {
        return true;
    }
    return setError(error, std::string(what) + ":" + clErrorText(err));
}

std::size_t roundUp(std::size_t value, std::size_t group) {
    return ((value + group - 1U) / group) * group;
}

int nextPowerOfTwo(int v) {
    int out = 1;
    while (out < v) {
        out <<= 1;
    }
    return out;
}

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

std::string sourceSpectrumCacheKey(cl_mem source, int paddedSize) {
    return std::to_string(reinterpret_cast<std::uintptr_t>(source)) + ":" + std::to_string(paddedSize);
}

std::string kernelSpectrumCacheKey(const LensDiffKernel& kernel, int paddedSize) {
    return std::to_string(paddedSize) + ":" + std::to_string(kernel.size) + ":" +
           std::to_string(hashKernelValues(kernel));
}

struct ProgramKey {
    cl_context context = nullptr;
    cl_device_id device = nullptr;

    bool operator==(const ProgramKey& other) const noexcept {
        return context == other.context && device == other.device;
    }
};

struct ProgramKeyHasher {
    std::size_t operator()(const ProgramKey& key) const noexcept {
        return reinterpret_cast<std::uintptr_t>(key.context) ^
               (reinterpret_cast<std::uintptr_t>(key.device) >> 4U);
    }
};

struct ProgramEntry {
    cl_program program = nullptr;

    ~ProgramEntry() {
        if (program != nullptr) {
            clReleaseProgram(program);
            program = nullptr;
        }
    }
};

std::mutex gProgramMutex;
std::unordered_map<ProgramKey, std::shared_ptr<ProgramEntry>, ProgramKeyHasher> gPrograms;

struct PersistentKernelSpectrum {
    UniqueMem spectrum;
    std::size_t bytes = 0;
};

std::mutex gPersistentKernelSpectrumMutex;
std::unordered_map<std::string, std::shared_ptr<PersistentKernelSpectrum>> gPersistentKernelSpectra;
std::size_t gPersistentKernelSpectrumBytes = 0;
constexpr std::size_t kPersistentKernelSpectrumBudgetBytes = 4ull * 1024ull * 1024ull * 1024ull;

std::string persistentKernelSpectrumCacheKey(cl_context context,
                                             cl_device_id device,
                                             const LensDiffKernel& kernel,
                                             int paddedSize) {
    return std::to_string(reinterpret_cast<std::uintptr_t>(context)) + ":" +
           std::to_string(reinterpret_cast<std::uintptr_t>(device)) + ":" +
           kernelSpectrumCacheKey(kernel, paddedSize);
}

std::shared_ptr<ProgramEntry> getProgram(cl_context context,
                                         cl_device_id device,
                                         std::string* error) {
    const ProgramKey key {context, device};
    {
        std::lock_guard<std::mutex> lock(gProgramMutex);
        const auto found = gPrograms.find(key);
        if (found != gPrograms.end()) {
            return found->second;
        }
    }

    cl_int err = CL_SUCCESS;
    std::array<std::size_t, kLensDiffOpenCLSourcePartCount> lengths {};
    std::array<const char*, kLensDiffOpenCLSourcePartCount> sources {};
    for (std::size_t i = 0; i < lengths.size(); ++i) {
        sources[i] = kLensDiffOpenCLSourceParts[i];
        lengths[i] = std::strlen(kLensDiffOpenCLSourceParts[i]);
    }
    cl_program program = clCreateProgramWithSource(context,
                                                   static_cast<cl_uint>(lengths.size()),
                                                   sources.data(),
                                                   lengths.data(),
                                                   &err);
    if (err != CL_SUCCESS || program == nullptr) {
        setError(error, "opencl-create-program:" + clErrorText(err));
        return nullptr;
    }

    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::string log;
        std::size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        if (logSize > 1U) {
            log.resize(logSize);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
        }
        clReleaseProgram(program);
        setError(error, "opencl-build-program:" + clErrorText(err) + ":" + log);
        return nullptr;
    }

    auto entry = std::make_shared<ProgramEntry>();
    entry->program = program;
    {
        std::lock_guard<std::mutex> lock(gProgramMutex);
        gPrograms.emplace(key, entry);
    }
    return entry;
}

UniqueKernel makeKernel(const std::shared_ptr<ProgramEntry>& program,
                        const char* name,
                        std::string* error) {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program->program, name, &err);
    if (err != CL_SUCCESS || kernel == nullptr) {
        setError(error, std::string("opencl-create-kernel:") + name + ":" + clErrorText(err));
        return UniqueKernel(nullptr);
    }
    return UniqueKernel(kernel);
}

template <typename T>
bool setKernelArg(cl_kernel kernel, cl_uint index, const T& value, const char* name, std::string* error) {
    return checkCl(clSetKernelArg(kernel, index, sizeof(T), &value), name, error);
}

bool enqueue2D(cl_command_queue queue,
               cl_kernel kernel,
               std::size_t width,
               std::size_t height,
               const char* name,
               std::string* error) {
    if (width == 0U || height == 0U) {
        return true;
    }
    const std::size_t local[2] = {16U, 16U};
    const std::size_t global[2] = {roundUp(width, local[0]), roundUp(height, local[1])};
    return checkCl(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr), name, error);
}

bool enqueue1D(cl_command_queue queue,
               cl_kernel kernel,
               std::size_t count,
               const char* name,
               std::string* error) {
    if (count == 0U) {
        return true;
    }
    const std::size_t local = static_cast<std::size_t>(kOpenCLWorkgroup1D);
    const std::size_t global = roundUp(count, local);
    return checkCl(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr), name, error);
}

UniqueMem makeBuffer(cl_context context,
                     cl_mem_flags flags,
                     std::size_t bytes,
                     void* hostPtr,
                     const char* name,
                     std::string* error) {
    cl_int err = CL_SUCCESS;
    cl_mem mem = clCreateBuffer(context, flags, std::max<std::size_t>(bytes, 1U), hostPtr, &err);
    if (err != CL_SUCCESS || mem == nullptr) {
        setError(error, std::string("opencl-create-buffer:") + name + ":" + clErrorText(err));
        return UniqueMem(nullptr);
    }
    return UniqueMem(mem);
}

UniqueMem makeFloatBuffer(cl_context context, std::size_t count, const char* name, std::string* error) {
    return makeBuffer(context, CL_MEM_READ_WRITE, count * sizeof(float), nullptr, name, error);
}

UniqueMem makeRgbaBuffer(cl_context context, std::size_t pixelCount, const char* name, std::string* error) {
    return makeBuffer(context, CL_MEM_READ_WRITE, pixelCount * sizeof(float) * 4U, nullptr, name, error);
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

std::vector<float> buildGaussianKernel(float radiusPx) {
    const float radius = std::max(0.25f, radiusPx);
    const int support = std::max(1, static_cast<int>(std::ceil(radius * 3.0f)));
    const int size = support * 2 + 1;
    std::vector<float> values(static_cast<std::size_t>(size) * size, 0.0f);
    const float sigma = std::max(0.25f, radius);
    const float invTwoSigma2 = 1.0f / (2.0f * sigma * sigma);
    float sum = 0.0f;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const float dx = static_cast<float>(x - support);
            const float dy = static_cast<float>(y - support);
            const float v = std::exp(-(dx * dx + dy * dy) * invTwoSigma2);
            values[static_cast<std::size_t>(y) * size + x] = v;
            sum += v;
        }
    }
    if (sum > 0.0f) {
        for (float& v : values) {
            v /= sum;
        }
    }
    return values;
}

bool uploadKernel(cl_context context,
                  const LensDiffKernel& kernel,
                  UniqueMem* out,
                  std::string* error) {
    if (kernel.size <= 0 || kernel.values.empty()) {
        return setError(error, "opencl-empty-kernel");
    }
    *out = makeBuffer(context,
                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      kernel.values.size() * sizeof(float),
                      const_cast<float*>(kernel.values.data()),
                      "kernel",
                      error);
    return static_cast<bool>(*out);
}

bool runClear(cl_command_queue queue,
              const std::shared_ptr<ProgramEntry>& program,
              cl_mem dst,
              int count,
              std::string* error) {
    UniqueKernel kernel = makeKernel(program, "lensDiffClearRgba", error);
    if (!kernel) return false;
    cl_uint arg = 0;
    return setKernelArg(kernel.get(), arg++, dst, "opencl-clear-arg-dst", error) &&
           setKernelArg(kernel.get(), arg++, count, "opencl-clear-arg-count", error) &&
           enqueue1D(queue, kernel.get(), static_cast<std::size_t>(count), "opencl-clear", error);
}

bool getKernelSpectrum(cl_command_queue queue,
                       cl_context context,
                       const std::shared_ptr<ProgramEntry>& program,
                       std::unordered_map<std::string, std::shared_ptr<PersistentKernelSpectrum>>& kernelSpectrumCache,
                       const LensDiffKernel& lensKernel,
                       int paddedSize,
                       cl_mem* outSpectrum,
                       std::string* error) {
    if (outSpectrum == nullptr) {
        return setError(error, "opencl-null-kernel-spectrum-output");
    }
    const std::string key = kernelSpectrumCacheKey(lensKernel, paddedSize);
    auto cached = kernelSpectrumCache.find(key);
    if (cached != kernelSpectrumCache.end() && cached->second && cached->second->spectrum) {
        *outSpectrum = cached->second->spectrum.get();
        return true;
    }
    cl_device_id device = nullptr;
    const cl_int queueErr = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device), &device, nullptr);
    if (!checkCl(queueErr, "opencl-kernel-spectrum-get-device", error)) return false;
    const std::string persistentKey = persistentKernelSpectrumCacheKey(context, device, lensKernel, paddedSize);
    {
        std::lock_guard<std::mutex> lock(gPersistentKernelSpectrumMutex);
        auto persistent = gPersistentKernelSpectra.find(persistentKey);
        if (persistent != gPersistentKernelSpectra.end() && persistent->second && persistent->second->spectrum) {
            kernelSpectrumCache.emplace(key, persistent->second);
            *outSpectrum = persistent->second->spectrum.get();
            return true;
        }
    }
    UniqueMem kernelBuffer;
    if (!uploadKernel(context, lensKernel, &kernelBuffer, error)) return false;
    const int paddedCount = paddedSize * paddedSize;
    const std::size_t spectrumBytes = static_cast<std::size_t>(paddedCount) * sizeof(float) * 2U;
    UniqueMem kernelSpectrum = makeBuffer(context,
                                          CL_MEM_READ_WRITE,
                                          spectrumBytes,
                                          nullptr,
                                          "kernel-spectrum",
                                          error);
    if (!kernelSpectrum) return false;
    UniqueKernel scatter = makeKernel(program, "lensDiffScatterKernelToComplex", error);
    if (!scatter) return false;
    const float zeroPattern[2] = {0.0f, 0.0f};
    if (!checkCl(clEnqueueFillBuffer(queue, kernelSpectrum.get(), zeroPattern, sizeof(zeroPattern), 0,
                                     static_cast<std::size_t>(paddedCount) * sizeof(float) * 2U,
                                     0, nullptr, nullptr),
                 "opencl-clear-kernel-spectrum", error)) {
        return false;
    }
    cl_uint arg = 0;
    const int kernelSize = lensKernel.size;
    if (!setKernelArg(scatter.get(), arg++, kernelBuffer.get(), "opencl-scatter-kernel-arg-kernel", error) ||
        !setKernelArg(scatter.get(), arg++, kernelSpectrum.get(), "opencl-scatter-kernel-arg-spectrum", error) ||
        !setKernelArg(scatter.get(), arg++, kernelSize, "opencl-scatter-kernel-arg-ksize", error) ||
        !setKernelArg(scatter.get(), arg++, paddedSize, "opencl-scatter-kernel-arg-padded", error) ||
        !enqueue2D(queue, scatter.get(), static_cast<std::size_t>(kernelSize), static_cast<std::size_t>(kernelSize),
                   "opencl-scatter-kernel-spectrum", error) ||
        !lensDiffOpenCLVkFFTEncodeSquare(queue, kernelSpectrum.get(), paddedSize, 1, false, error)) {
        return false;
    }
    if (!checkCl(clFinish(queue), "opencl-finish-kernel-spectrum-cache-fill", error)) {
        return false;
    }
    auto entry = std::make_shared<PersistentKernelSpectrum>();
    entry->spectrum = std::move(kernelSpectrum);
    entry->bytes = spectrumBytes;
    {
        std::lock_guard<std::mutex> lock(gPersistentKernelSpectrumMutex);
        auto existing = gPersistentKernelSpectra.find(persistentKey);
        if (existing != gPersistentKernelSpectra.end() && existing->second && existing->second->spectrum) {
            entry = existing->second;
        } else if (gPersistentKernelSpectrumBytes + spectrumBytes <= kPersistentKernelSpectrumBudgetBytes) {
            gPersistentKernelSpectrumBytes += spectrumBytes;
            gPersistentKernelSpectra.emplace(persistentKey, entry);
        }
    }
    kernelSpectrumCache.emplace(key, entry);
    *outSpectrum = entry->spectrum.get();
    return true;
}

UniqueMem makeRgbaSourceSpectrum(cl_command_queue queue,
                                 cl_context context,
                                 const std::shared_ptr<ProgramEntry>& program,
                                 cl_mem src,
                                 int width,
                                 int height,
                                 int paddedSize,
                                 std::string* error) {
    const int paddedCount = paddedSize * paddedSize;
    UniqueMem spectrum = makeBuffer(context,
                                    CL_MEM_READ_WRITE,
                                    static_cast<std::size_t>(paddedCount) * 3U * sizeof(float) * 2U,
                                    nullptr,
                                    "rgba-source-spectrum",
                                    error);
    if (!spectrum) return UniqueMem(nullptr);
    UniqueKernel pad = makeKernel(program, "lensDiffPadRgbaToComplexStack", error);
    if (!pad) return UniqueMem(nullptr);
    cl_uint arg = 0;
    if (!setKernelArg(pad.get(), arg++, src, "opencl-pad-source-rgba-arg-src", error) ||
        !setKernelArg(pad.get(), arg++, spectrum.get(), "opencl-pad-source-rgba-arg-spectrum", error) ||
        !setKernelArg(pad.get(), arg++, width, "opencl-pad-source-rgba-arg-width", error) ||
        !setKernelArg(pad.get(), arg++, height, "opencl-pad-source-rgba-arg-height", error) ||
        !setKernelArg(pad.get(), arg++, paddedSize, "opencl-pad-source-rgba-arg-padded", error) ||
        !enqueue2D(queue, pad.get(), static_cast<std::size_t>(paddedSize), static_cast<std::size_t>(paddedSize * 3),
                   "opencl-pad-source-rgba-spectrum", error) ||
        !lensDiffOpenCLVkFFTEncodeSquare(queue, spectrum.get(), paddedSize, 3, false, error)) {
        return UniqueMem(nullptr);
    }
    return spectrum;
}

bool getRgbaSourceSpectrum(cl_command_queue queue,
                           cl_context context,
                           const std::shared_ptr<ProgramEntry>& program,
                           std::unordered_map<std::string, UniqueMem>& sourceSpectrumCache,
                           cl_mem src,
                           int width,
                           int height,
                           int paddedSize,
                           cl_mem* outSpectrum,
                           std::string* error) {
    if (outSpectrum == nullptr) {
        return setError(error, "opencl-null-rgba-source-spectrum-output");
    }
    const std::string key = sourceSpectrumCacheKey(src, paddedSize);
    auto cached = sourceSpectrumCache.find(key);
    if (cached != sourceSpectrumCache.end()) {
        *outSpectrum = cached->second.get();
        return true;
    }
    UniqueMem spectrum = makeRgbaSourceSpectrum(queue, context, program, src, width, height, paddedSize, error);
    if (!spectrum) return false;
    auto inserted = sourceSpectrumCache.emplace(key, std::move(spectrum));
    *outSpectrum = inserted.first->second.get();
    return true;
}

bool runConvolveRgbaFromSpectra(cl_command_queue queue,
                                cl_context context,
                                const std::shared_ptr<ProgramEntry>& program,
                                cl_mem imageSpectrum,
                                cl_mem kernelSpectrum,
                                cl_mem alphaSource,
                                cl_mem dst,
                                int width,
                                int height,
                                int paddedSize,
                                float shoulder,
                                std::string* error) {
    const int paddedCount = paddedSize * paddedSize;
    const float scale = 1.0f / static_cast<float>(paddedCount);
    UniqueMem tempSpectrum = makeBuffer(context,
                                        CL_MEM_READ_WRITE,
                                        static_cast<std::size_t>(paddedCount) * 3U * sizeof(float) * 2U,
                                        nullptr,
                                        "rgba-temp-spectrum",
                                        error);
    if (!tempSpectrum) return false;

    UniqueKernel multiply = makeKernel(program, "lensDiffMultiplyComplexBroadcast", error);
    UniqueKernel extract = makeKernel(program, "lensDiffExtractRgbaStack", error);
    if (!multiply || !extract) return false;

    cl_uint arg = 0;
    const int batchCount = 3;
    if (!setKernelArg(multiply.get(), arg++, imageSpectrum, "opencl-mul-rgba-arg-image", error) ||
        !setKernelArg(multiply.get(), arg++, kernelSpectrum, "opencl-mul-rgba-arg-kernel", error) ||
        !setKernelArg(multiply.get(), arg++, tempSpectrum.get(), "opencl-mul-rgba-arg-dst", error) ||
        !setKernelArg(multiply.get(), arg++, paddedCount, "opencl-mul-rgba-arg-count", error) ||
        !setKernelArg(multiply.get(), arg++, batchCount, "opencl-mul-rgba-arg-batch", error) ||
        !enqueue2D(queue, multiply.get(), static_cast<std::size_t>(paddedCount), 3U, "opencl-multiply-rgba-spectrum", error)) {
        return false;
    }
    if (!lensDiffOpenCLVkFFTEncodeSquare(queue, tempSpectrum.get(), paddedSize, 3, true, error)) {
        return false;
    }
    arg = 0;
    return setKernelArg(extract.get(), arg++, tempSpectrum.get(), "opencl-extract-rgba-arg-spectrum", error) &&
           setKernelArg(extract.get(), arg++, alphaSource, "opencl-extract-rgba-arg-alpha", error) &&
           setKernelArg(extract.get(), arg++, dst, "opencl-extract-rgba-arg-dst", error) &&
           setKernelArg(extract.get(), arg++, width, "opencl-extract-rgba-arg-width", error) &&
           setKernelArg(extract.get(), arg++, height, "opencl-extract-rgba-arg-height", error) &&
           setKernelArg(extract.get(), arg++, paddedSize, "opencl-extract-rgba-arg-padded", error) &&
           setKernelArg(extract.get(), arg++, scale, "opencl-extract-rgba-arg-scale", error) &&
           setKernelArg(extract.get(), arg++, shoulder, "opencl-extract-rgba-arg-shoulder", error) &&
           enqueue2D(queue, extract.get(), static_cast<std::size_t>(width), static_cast<std::size_t>(height),
                     "opencl-extract-rgba", error);
}

bool runConvolveScalarPlane(cl_command_queue queue,
                            cl_context context,
                            const std::shared_ptr<ProgramEntry>& program,
                            std::unordered_map<std::string, std::shared_ptr<PersistentKernelSpectrum>>& kernelSpectrumCache,
                            cl_mem imageSpectrum,
                            const LensDiffKernel& lensKernel,
                            cl_mem planes,
                            int width,
                            int height,
                            int paddedSize,
                            int planeIndex,
                            int planeStride,
                            std::string* error) {
    const int paddedCount = paddedSize * paddedSize;
    const float scale = 1.0f / static_cast<float>(paddedCount);
    cl_mem kernelSpectrum = nullptr;
    if (!getKernelSpectrum(queue, context, program, kernelSpectrumCache, lensKernel, paddedSize,
                           &kernelSpectrum, error)) {
        return false;
    }
    UniqueMem tempSpectrum = makeBuffer(context,
                                        CL_MEM_READ_WRITE,
                                        static_cast<std::size_t>(paddedCount) * sizeof(float) * 2U,
                                        nullptr,
                                        "scalar-temp-spectrum",
                                        error);
    if (!tempSpectrum) return false;
    UniqueKernel multiply = makeKernel(program, "lensDiffMultiplyComplex", error);
    UniqueKernel extract = makeKernel(program, "lensDiffExtractScalarPlane", error);
    if (!multiply || !extract) return false;
    cl_uint arg = 0;
    if (!setKernelArg(multiply.get(), arg++, imageSpectrum, "opencl-mul-scalar-arg-image", error) ||
        !setKernelArg(multiply.get(), arg++, kernelSpectrum, "opencl-mul-scalar-arg-kernel", error) ||
        !setKernelArg(multiply.get(), arg++, tempSpectrum.get(), "opencl-mul-scalar-arg-dst", error) ||
        !setKernelArg(multiply.get(), arg++, paddedCount, "opencl-mul-scalar-arg-count", error) ||
        !enqueue1D(queue, multiply.get(), static_cast<std::size_t>(paddedCount), "opencl-multiply-scalar-spectrum", error)) {
        return false;
    }
    if (!lensDiffOpenCLVkFFTEncodeSquare(queue, tempSpectrum.get(), paddedSize, 1, true, error)) {
        return false;
    }
    arg = 0;
    return setKernelArg(extract.get(), arg++, tempSpectrum.get(), "opencl-extract-scalar-arg-spectrum", error) &&
           setKernelArg(extract.get(), arg++, planes, "opencl-extract-scalar-arg-planes", error) &&
           setKernelArg(extract.get(), arg++, width, "opencl-extract-scalar-arg-width", error) &&
           setKernelArg(extract.get(), arg++, height, "opencl-extract-scalar-arg-height", error) &&
           setKernelArg(extract.get(), arg++, paddedSize, "opencl-extract-scalar-arg-padded", error) &&
           setKernelArg(extract.get(), arg++, planeIndex, "opencl-extract-scalar-arg-plane", error) &&
           setKernelArg(extract.get(), arg++, planeStride, "opencl-extract-scalar-arg-stride", error) &&
           setKernelArg(extract.get(), arg++, scale, "opencl-extract-scalar-arg-scale", error) &&
           enqueue2D(queue, extract.get(), static_cast<std::size_t>(width), static_cast<std::size_t>(height),
                     "opencl-extract-scalar", error);
}

UniqueMem makeScalarSourceSpectrum(cl_command_queue queue,
                                   cl_context context,
                                   const std::shared_ptr<ProgramEntry>& program,
                                   cl_mem src,
                                   int width,
                                   int height,
                                   int paddedSize,
                                   std::string* error) {
    const int paddedCount = paddedSize * paddedSize;
    UniqueMem spectrum = makeBuffer(context,
                                    CL_MEM_READ_WRITE,
                                    static_cast<std::size_t>(paddedCount) * sizeof(float) * 2U,
                                    nullptr,
                                    "scalar-source-spectrum",
                                    error);
    if (!spectrum) return UniqueMem(nullptr);
    UniqueKernel pad = makeKernel(program, "lensDiffPadScalarToComplex", error);
    if (!pad) return UniqueMem(nullptr);
    cl_uint arg = 0;
    if (!setKernelArg(pad.get(), arg++, src, "opencl-pad-source-scalar-arg-src", error) ||
        !setKernelArg(pad.get(), arg++, spectrum.get(), "opencl-pad-source-scalar-arg-spectrum", error) ||
        !setKernelArg(pad.get(), arg++, width, "opencl-pad-source-scalar-arg-width", error) ||
        !setKernelArg(pad.get(), arg++, height, "opencl-pad-source-scalar-arg-height", error) ||
        !setKernelArg(pad.get(), arg++, paddedSize, "opencl-pad-source-scalar-arg-padded", error) ||
        !enqueue2D(queue, pad.get(), static_cast<std::size_t>(paddedSize), static_cast<std::size_t>(paddedSize),
                   "opencl-pad-source-scalar-spectrum", error) ||
        !lensDiffOpenCLVkFFTEncodeSquare(queue, spectrum.get(), paddedSize, 1, false, error)) {
        return UniqueMem(nullptr);
    }
    return spectrum;
}

bool getScalarSourceSpectrum(cl_command_queue queue,
                             cl_context context,
                             const std::shared_ptr<ProgramEntry>& program,
                             std::unordered_map<std::string, UniqueMem>& sourceSpectrumCache,
                             cl_mem src,
                             int width,
                             int height,
                             int paddedSize,
                             cl_mem* outSpectrum,
                             std::string* error) {
    if (outSpectrum == nullptr) {
        return setError(error, "opencl-null-scalar-source-spectrum-output");
    }
    const std::string key = sourceSpectrumCacheKey(src, paddedSize);
    auto cached = sourceSpectrumCache.find(key);
    if (cached != sourceSpectrumCache.end()) {
        *outSpectrum = cached->second.get();
        return true;
    }
    UniqueMem spectrum = makeScalarSourceSpectrum(queue, context, program, src, width, height, paddedSize, error);
    if (!spectrum) return false;
    auto inserted = sourceSpectrumCache.emplace(key, std::move(spectrum));
    *outSpectrum = inserted.first->second.get();
    return true;
}

bool runCombineSplit(cl_command_queue queue,
                     const std::shared_ptr<ProgramEntry>& program,
                     cl_mem core,
                     cl_mem structure,
                     cl_mem dst,
                     int count,
                     float coreGain,
                     float structureGain,
                     std::string* error) {
    UniqueKernel kernel = makeKernel(program, "lensDiffCombineSplit", error);
    if (!kernel) return false;
    cl_uint arg = 0;
    return setKernelArg(kernel.get(), arg++, core, "opencl-combine-arg-core", error) &&
           setKernelArg(kernel.get(), arg++, structure, "opencl-combine-arg-structure", error) &&
           setKernelArg(kernel.get(), arg++, dst, "opencl-combine-arg-dst", error) &&
           setKernelArg(kernel.get(), arg++, count, "opencl-combine-arg-count", error) &&
           setKernelArg(kernel.get(), arg++, coreGain, "opencl-combine-arg-core-gain", error) &&
           setKernelArg(kernel.get(), arg++, structureGain, "opencl-combine-arg-structure-gain", error) &&
           enqueue1D(queue, kernel.get(), static_cast<std::size_t>(count), "opencl-combine", error);
}

bool runApplyShoulder(cl_command_queue queue,
                      const std::shared_ptr<ProgramEntry>& program,
                      cl_mem image,
                      int count,
                      float shoulder,
                      std::string* error) {
    if (shoulder <= 0.0f) {
        return true;
    }
    UniqueKernel kernel = makeKernel(program, "lensDiffApplyShoulder", error);
    if (!kernel) return false;
    cl_uint arg = 0;
    return setKernelArg(kernel.get(), arg++, image, "opencl-shoulder-arg-image", error) &&
           setKernelArg(kernel.get(), arg++, count, "opencl-shoulder-arg-count", error) &&
           setKernelArg(kernel.get(), arg++, shoulder, "opencl-shoulder-arg-shoulder", error) &&
           enqueue1D(queue, kernel.get(), static_cast<std::size_t>(count), "opencl-apply-shoulder", error);
}

bool runAddImages(cl_command_queue queue,
                  const std::shared_ptr<ProgramEntry>& program,
                  cl_mem a,
                  cl_mem b,
                  cl_mem dst,
                  int count,
                  float aGain,
                  float bGain,
                  std::string* error) {
    UniqueKernel kernel = makeKernel(program, "lensDiffAddImages", error);
    if (!kernel) return false;
    cl_uint arg = 0;
    return setKernelArg(kernel.get(), arg++, a, "opencl-add-arg-a", error) &&
           setKernelArg(kernel.get(), arg++, b, "opencl-add-arg-b", error) &&
           setKernelArg(kernel.get(), arg++, dst, "opencl-add-arg-dst", error) &&
           setKernelArg(kernel.get(), arg++, count, "opencl-add-arg-count", error) &&
           setKernelArg(kernel.get(), arg++, aGain, "opencl-add-arg-again", error) &&
           setKernelArg(kernel.get(), arg++, bGain, "opencl-add-arg-bgain", error) &&
           enqueue1D(queue, kernel.get(), static_cast<std::size_t>(count), "opencl-add", error);
}

bool runMapSpectral(cl_command_queue queue,
                    cl_context context,
                    const std::shared_ptr<ProgramEntry>& program,
                    cl_mem planes,
                    cl_mem dst,
                    const LensDiffSpectrumConfig& config,
                    const LensDiffParams& params,
                    int width,
                    int height,
                    std::string* error) {
    std::array<float, kLensDiffMaxSpectralBins * 3> natural = config.naturalMatrix;
    std::array<float, kLensDiffMaxSpectralBins * 3> style = config.styleMatrix;
    UniqueMem naturalBuffer = makeBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         natural.size() * sizeof(float), natural.data(), "natural-spectrum", error);
    if (!naturalBuffer) return false;
    UniqueMem styleBuffer = makeBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       style.size() * sizeof(float), style.data(), "style-spectrum", error);
    if (!styleBuffer) return false;
    UniqueKernel kernel = makeKernel(program, "lensDiffMapSpectral", error);
    if (!kernel) return false;
    const int planeStride = width * height;
    const int binCount = std::max(1, std::min(config.binCount, kLensDiffMaxSpectralBins));
    const float spectrumForce = static_cast<float>(std::clamp(params.spectrumForce, 0.0, 1.0));
    const float spectrumSaturation = static_cast<float>(std::max(0.0, params.spectrumSaturation));
    const int chromaticAffectsLuma = params.chromaticAffectsLuma ? 1 : 0;
    cl_uint arg = 0;
    return setKernelArg(kernel.get(), arg++, planes, "opencl-map-arg-planes", error) &&
           setKernelArg(kernel.get(), arg++, dst, "opencl-map-arg-dst", error) &&
           setKernelArg(kernel.get(), arg++, naturalBuffer.get(), "opencl-map-arg-natural", error) &&
           setKernelArg(kernel.get(), arg++, styleBuffer.get(), "opencl-map-arg-style", error) &&
           setKernelArg(kernel.get(), arg++, width, "opencl-map-arg-width", error) &&
           setKernelArg(kernel.get(), arg++, height, "opencl-map-arg-height", error) &&
           setKernelArg(kernel.get(), arg++, planeStride, "opencl-map-arg-stride", error) &&
           setKernelArg(kernel.get(), arg++, binCount, "opencl-map-arg-bins", error) &&
           setKernelArg(kernel.get(), arg++, spectrumForce, "opencl-map-arg-force", error) &&
           setKernelArg(kernel.get(), arg++, spectrumSaturation, "opencl-map-arg-saturation", error) &&
           setKernelArg(kernel.get(), arg++, chromaticAffectsLuma, "opencl-map-arg-affects-luma", error) &&
           enqueue2D(queue, kernel.get(), static_cast<std::size_t>(width), static_cast<std::size_t>(height),
                     "opencl-map-spectral", error);
}

bool renderSpectral(cl_command_queue queue,
                    cl_context context,
                    const std::shared_ptr<ProgramEntry>& program,
                    cl_mem driver,
                    const std::vector<LensDiffPsfBin>& bins,
                    const LensDiffSpectrumConfig& spectrumConfig,
                    const LensDiffParams& params,
                    bool useCore,
                    bool useStructure,
                    cl_mem dst,
                    int width,
                    int height,
                    std::unordered_map<std::string, UniqueMem>& scalarSourceSpectrumCache,
                    std::unordered_map<std::string, std::shared_ptr<PersistentKernelSpectrum>>& kernelSpectrumCache,
                    std::string* error) {
    const int planeStride = width * height;
    const int binCount = std::max(1, std::min<int>(static_cast<int>(bins.size()), kLensDiffMaxSpectralBins));
    UniqueMem planes = makeFloatBuffer(context, static_cast<std::size_t>(planeStride) * binCount, "spectral-planes", error);
    if (!planes) return false;
    int maxKernelSize = 1;
    for (int i = 0; i < binCount; ++i) {
        const auto& bin = bins[static_cast<std::size_t>(i)];
        const LensDiffKernel* kernel = &bin.full;
        if (useCore && !useStructure) {
            kernel = &bin.core;
        } else if (!useCore && useStructure) {
            kernel = &bin.structure;
        }
        maxKernelSize = std::max(maxKernelSize, kernel->size);
    }
    const int paddedSize = nextPowerOfTwo(std::max(width, height) + maxKernelSize - 1);
    cl_mem sourceSpectrum = nullptr;
    if (!getScalarSourceSpectrum(queue, context, program, scalarSourceSpectrumCache, driver,
                                 width, height, paddedSize, &sourceSpectrum, error)) {
        return false;
    }
    for (int i = 0; i < binCount; ++i) {
        const auto& bin = bins[static_cast<std::size_t>(i)];
        const LensDiffKernel* kernel = &bin.full;
        if (useCore && !useStructure) {
            kernel = &bin.core;
        } else if (!useCore && useStructure) {
            kernel = &bin.structure;
        }
        if (!runConvolveScalarPlane(queue, context, program, kernelSpectrumCache, sourceSpectrum, *kernel, planes.get(),
                                    width, height, paddedSize, i, planeStride, error)) {
            return false;
        }
    }
    return runMapSpectral(queue, context, program, planes.get(), dst, spectrumConfig, params, width, height, error);
}

bool renderBins(cl_command_queue queue,
                cl_context context,
                const std::shared_ptr<ProgramEntry>& program,
                cl_mem redistributed,
                cl_mem driver,
                const std::vector<LensDiffPsfBin>& bins,
                const LensDiffParams& params,
                cl_mem outCore,
                cl_mem outStructure,
                cl_mem outEffect,
                int width,
                int height,
                std::unordered_map<std::string, UniqueMem>& rgbSourceSpectrumCache,
                std::unordered_map<std::string, UniqueMem>& scalarSourceSpectrumCache,
                std::unordered_map<std::string, std::shared_ptr<PersistentKernelSpectrum>>& kernelSpectrumCache,
                std::string* error) {
    if (bins.empty()) {
        return setError(error, "opencl-empty-psf-bins");
    }
    const int count = width * height;
    if (params.spectralMode == LensDiffSpectralMode::Mono) {
        int monoMaxKernelSize = bins.front().full.size;
        if (params.lookMode == LensDiffLookMode::Split || outCore != nullptr) {
            monoMaxKernelSize = std::max(monoMaxKernelSize, bins.front().core.size);
        }
        if (params.lookMode == LensDiffLookMode::Split || outStructure != nullptr) {
            monoMaxKernelSize = std::max(monoMaxKernelSize, bins.front().structure.size);
        }
        const int paddedSize = nextPowerOfTwo(std::max(width, height) + monoMaxKernelSize - 1);
        cl_mem sourceSpectrum = nullptr;
        if (!getRgbaSourceSpectrum(queue, context, program, rgbSourceSpectrumCache, redistributed,
                                   width, height, paddedSize, &sourceSpectrum, error)) {
            return false;
        }
        auto convolveRedistributed = [&](const LensDiffKernel& kernel, cl_mem dst, float shoulder) {
            cl_mem kernelSpectrum = nullptr;
            return getKernelSpectrum(queue, context, program, kernelSpectrumCache, kernel, paddedSize,
                                     &kernelSpectrum, error) &&
                   runConvolveRgbaFromSpectra(queue, context, program, sourceSpectrum, kernelSpectrum, redistributed, dst,
                                              width, height, paddedSize, shoulder, error);
        };
        if (params.lookMode == LensDiffLookMode::Split) {
            if (outCore == nullptr || outStructure == nullptr) {
                return setError(error, "opencl-split-missing-core-or-structure-output");
            }
            if (!convolveRedistributed(bins.front().core, outCore, static_cast<float>(params.coreShoulder)) ||
                !convolveRedistributed(bins.front().structure, outStructure, static_cast<float>(params.structureShoulder))) {
                return false;
            }
            return runCombineSplit(queue, program, outCore, outStructure, outEffect, count,
                                   static_cast<float>(std::max(0.0, params.coreGain)),
                                   static_cast<float>(std::max(0.0, params.structureGain)),
                                   error);
        }
        if (!convolveRedistributed(bins.front().full, outEffect, 0.0f)) {
            return false;
        }
        if (outCore != nullptr && !convolveRedistributed(bins.front().core, outCore, 0.0f)) {
            return false;
        }
        if (outStructure != nullptr && !convolveRedistributed(bins.front().structure, outStructure, 0.0f)) {
            return false;
        }
        return true;
    }

    const LensDiffSpectrumConfig zoneConfig = BuildLensDiffSpectrumConfig(params, bins);
    if (params.lookMode == LensDiffLookMode::Split) {
        if (outCore == nullptr || outStructure == nullptr) {
            return setError(error, "opencl-split-missing-spectral-core-or-structure-output");
        }
        if (!renderSpectral(queue, context, program, driver, bins, zoneConfig, params, true, false, outCore, width, height,
                            scalarSourceSpectrumCache, kernelSpectrumCache, error) ||
            !renderSpectral(queue, context, program, driver, bins, zoneConfig, params, false, true, outStructure, width, height,
                            scalarSourceSpectrumCache, kernelSpectrumCache, error)) {
            return false;
        }
        if (!runApplyShoulder(queue, program, outCore, count, static_cast<float>(params.coreShoulder), error) ||
            !runApplyShoulder(queue, program, outStructure, count, static_cast<float>(params.structureShoulder), error)) {
            return false;
        }
        return runCombineSplit(queue, program, outCore, outStructure, outEffect, count,
                               static_cast<float>(std::max(0.0, params.coreGain)),
                               static_cast<float>(std::max(0.0, params.structureGain)),
                               error);
    }
    if (!renderSpectral(queue, context, program, driver, bins, zoneConfig, params, true, true, outEffect, width, height,
                        scalarSourceSpectrumCache, kernelSpectrumCache, error)) {
        return false;
    }
    if (outCore != nullptr &&
        !renderSpectral(queue, context, program, driver, bins, zoneConfig, params, true, false, outCore, width, height,
                        scalarSourceSpectrumCache, kernelSpectrumCache, error)) {
        return false;
    }
    if (outStructure != nullptr &&
        !renderSpectral(queue, context, program, driver, bins, zoneConfig, params, false, true, outStructure, width, height,
                        scalarSourceSpectrumCache, kernelSpectrumCache, error)) {
        return false;
    }
    return true;
}

bool runAccumulate(cl_command_queue queue,
                   const std::shared_ptr<ProgramEntry>& program,
                   cl_mem src,
                   cl_mem dst,
                   int width,
                   int height,
                   int zoneX,
                   int zoneY,
                   std::string* error) {
    UniqueKernel kernel = makeKernel(program, "lensDiffAccumulateWeighted", error);
    if (!kernel) return false;
    cl_uint arg = 0;
    return setKernelArg(kernel.get(), arg++, src, "opencl-accumulate-arg-src", error) &&
           setKernelArg(kernel.get(), arg++, dst, "opencl-accumulate-arg-dst", error) &&
           setKernelArg(kernel.get(), arg++, width, "opencl-accumulate-arg-width", error) &&
           setKernelArg(kernel.get(), arg++, height, "opencl-accumulate-arg-height", error) &&
           setKernelArg(kernel.get(), arg++, zoneX, "opencl-accumulate-arg-zonex", error) &&
           setKernelArg(kernel.get(), arg++, zoneY, "opencl-accumulate-arg-zoney", error) &&
           enqueue2D(queue, kernel.get(), static_cast<std::size_t>(width), static_cast<std::size_t>(height),
                     "opencl-accumulate", error);
}

bool reduceLuma(cl_command_queue queue,
                cl_context context,
                const std::shared_ptr<ProgramEntry>& program,
                cl_mem image,
                int count,
                float* out,
                std::string* error) {
    if (count <= 0) {
        *out = 0.0f;
        return true;
    }
    UniqueKernel reduceImage = makeKernel(program, "lensDiffReduceLuma", error);
    UniqueKernel reduceFloat = makeKernel(program, "lensDiffReduceFloat", error);
    if (!reduceImage || !reduceFloat) return false;
    int currentCount = count;
    UniqueMem a = makeFloatBuffer(context, static_cast<std::size_t>((currentCount + 255) / 256), "reduce-a", error);
    if (!a) return false;
    {
        cl_uint arg = 0;
        if (!setKernelArg(reduceImage.get(), arg++, image, "opencl-reduce-luma-arg-image", error) ||
            !setKernelArg(reduceImage.get(), arg++, a.get(), "opencl-reduce-luma-arg-partial", error) ||
            !setKernelArg(reduceImage.get(), arg++, currentCount, "opencl-reduce-luma-arg-count", error) ||
            !enqueue1D(queue, reduceImage.get(), static_cast<std::size_t>(currentCount), "opencl-reduce-luma", error)) {
            return false;
        }
    }
    currentCount = (currentCount + 255) / 256;
    UniqueMem current = std::move(a);
    while (currentCount > 1) {
        const int nextCount = (currentCount + 255) / 256;
        UniqueMem next = makeFloatBuffer(context, static_cast<std::size_t>(nextCount), "reduce-next", error);
        if (!next) return false;
        cl_mem src = current.get();
        cl_mem dst = next.get();
        cl_uint arg = 0;
        if (!setKernelArg(reduceFloat.get(), arg++, src, "opencl-reduce-float-arg-src", error) ||
            !setKernelArg(reduceFloat.get(), arg++, dst, "opencl-reduce-float-arg-dst", error) ||
            !setKernelArg(reduceFloat.get(), arg++, currentCount, "opencl-reduce-float-arg-count", error) ||
            !enqueue1D(queue, reduceFloat.get(), static_cast<std::size_t>(currentCount), "opencl-reduce-float", error)) {
            return false;
        }
        currentCount = nextCount;
        current = std::move(next);
    }
    return checkCl(clEnqueueReadBuffer(queue, current.get(), CL_TRUE, 0, sizeof(float), out, 0, nullptr, nullptr),
                   "opencl-reduce-readback", error);
}

bool runScaleRgba(cl_command_queue queue,
                  const std::shared_ptr<ProgramEntry>& program,
                  cl_mem image,
                  int count,
                  float scale,
                  std::string* error) {
    UniqueKernel kernel = makeKernel(program, "lensDiffScaleRgba", error);
    if (!kernel) return false;
    cl_uint arg = 0;
    return setKernelArg(kernel.get(), arg++, image, "opencl-scale-arg-image", error) &&
           setKernelArg(kernel.get(), arg++, count, "opencl-scale-arg-count", error) &&
           setKernelArg(kernel.get(), arg++, scale, "opencl-scale-arg-scale", error) &&
           enqueue1D(queue, kernel.get(), static_cast<std::size_t>(count), "opencl-scale-rgba", error);
}

bool runCreativeFringe(cl_command_queue queue,
                       const std::shared_ptr<ProgramEntry>& program,
                       cl_mem src,
                       cl_mem dst,
                       cl_mem preview,
                       int width,
                       int height,
                       float amount,
                       std::string* error) {
    UniqueKernel kernel = makeKernel(program, "lensDiffCreativeFringe", error);
    if (!kernel) return false;
    cl_uint arg = 0;
    return setKernelArg(kernel.get(), arg++, src, "opencl-fringe-arg-src", error) &&
           setKernelArg(kernel.get(), arg++, dst, "opencl-fringe-arg-dst", error) &&
           setKernelArg(kernel.get(), arg++, preview, "opencl-fringe-arg-preview", error) &&
           setKernelArg(kernel.get(), arg++, width, "opencl-fringe-arg-width", error) &&
           setKernelArg(kernel.get(), arg++, height, "opencl-fringe-arg-height", error) &&
           setKernelArg(kernel.get(), arg++, amount, "opencl-fringe-arg-amount", error) &&
           enqueue2D(queue, kernel.get(), static_cast<std::size_t>(width), static_cast<std::size_t>(height),
                     "opencl-creative-fringe", error);
}

bool runComposite(cl_command_queue queue,
                  const std::shared_ptr<ProgramEntry>& program,
                  cl_mem linearSrcNative,
                  cl_mem redistributed,
                  cl_mem effect,
                  cl_mem finalImage,
                  int nativeWidth,
                  int nativeHeight,
                  int width,
                  int height,
                  const LensDiffParams& params,
                  std::string* error) {
    const float redistributionScale = 1.0f - static_cast<float>(std::clamp(params.corePreserve, 0.0, 1.0));
    const float protectedCoreFraction = std::max(0.2f, static_cast<float>(std::clamp(params.corePreserve, 0.0, 1.0)));
    const float maxRedistributedSubtractScale = redistributionScale > 1.0e-6f
        ? (1.0f - protectedCoreFraction) / redistributionScale
        : 0.0f;
    const float effectGain = params.energyMode == LensDiffEnergyMode::Preserve
        ? static_cast<float>(std::clamp(params.effectGain, 0.0, 1.0))
        : static_cast<float>(std::max(0.0, params.effectGain));
    const float coreCompensation = params.energyMode == LensDiffEnergyMode::Preserve
        ? effectGain
        : static_cast<float>(std::max(0.0, params.coreCompensation));
    UniqueKernel kernel = makeKernel(program, "lensDiffComposite", error);
    if (!kernel) return false;
    cl_uint arg = 0;
    return setKernelArg(kernel.get(), arg++, linearSrcNative, "opencl-composite-arg-src", error) &&
           setKernelArg(kernel.get(), arg++, redistributed, "opencl-composite-arg-redistributed", error) &&
           setKernelArg(kernel.get(), arg++, effect, "opencl-composite-arg-effect", error) &&
           setKernelArg(kernel.get(), arg++, finalImage, "opencl-composite-arg-final", error) &&
           setKernelArg(kernel.get(), arg++, nativeWidth, "opencl-composite-arg-native-width", error) &&
           setKernelArg(kernel.get(), arg++, nativeHeight, "opencl-composite-arg-native-height", error) &&
           setKernelArg(kernel.get(), arg++, width, "opencl-composite-arg-width", error) &&
           setKernelArg(kernel.get(), arg++, height, "opencl-composite-arg-height", error) &&
           setKernelArg(kernel.get(), arg++, effectGain, "opencl-composite-arg-effect-gain", error) &&
           setKernelArg(kernel.get(), arg++, coreCompensation, "opencl-composite-arg-core-comp", error) &&
           setKernelArg(kernel.get(), arg++, maxRedistributedSubtractScale, "opencl-composite-arg-floor", error) &&
           enqueue2D(queue, kernel.get(), static_cast<std::size_t>(nativeWidth), static_cast<std::size_t>(nativeHeight),
                     "opencl-composite", error);
}

bool runGrayToRgba(cl_command_queue queue,
                   const std::shared_ptr<ProgramEntry>& program,
                   cl_mem gray,
                   cl_mem rgba,
                   int width,
                   int height,
                   std::string* error) {
    UniqueKernel kernel = makeKernel(program, "lensDiffGrayToRgba", error);
    if (!kernel) return false;
    cl_uint arg = 0;
    return setKernelArg(kernel.get(), arg++, gray, "opencl-gray-rgba-arg-gray", error) &&
           setKernelArg(kernel.get(), arg++, rgba, "opencl-gray-rgba-arg-rgba", error) &&
           setKernelArg(kernel.get(), arg++, width, "opencl-gray-rgba-arg-width", error) &&
           setKernelArg(kernel.get(), arg++, height, "opencl-gray-rgba-arg-height", error) &&
           enqueue2D(queue, kernel.get(), static_cast<std::size_t>(width), static_cast<std::size_t>(height),
                     "opencl-gray-to-rgba", error);
}

bool runGrayToRgbaResample(cl_command_queue queue,
                           const std::shared_ptr<ProgramEntry>& program,
                           cl_mem gray,
                           cl_mem rgba,
                           int srcWidth,
                           int srcHeight,
                           int dstWidth,
                           int dstHeight,
                           std::string* error) {
    UniqueKernel kernel = makeKernel(program, "lensDiffGrayToRgbaResample", error);
    if (!kernel) return false;
    cl_uint arg = 0;
    return setKernelArg(kernel.get(), arg++, gray, "opencl-gray-rgba-resample-arg-gray", error) &&
           setKernelArg(kernel.get(), arg++, rgba, "opencl-gray-rgba-resample-arg-rgba", error) &&
           setKernelArg(kernel.get(), arg++, srcWidth, "opencl-gray-rgba-resample-arg-src-width", error) &&
           setKernelArg(kernel.get(), arg++, srcHeight, "opencl-gray-rgba-resample-arg-src-height", error) &&
           setKernelArg(kernel.get(), arg++, dstWidth, "opencl-gray-rgba-resample-arg-dst-width", error) &&
           setKernelArg(kernel.get(), arg++, dstHeight, "opencl-gray-rgba-resample-arg-dst-height", error) &&
           enqueue2D(queue, kernel.get(), static_cast<std::size_t>(dstWidth), static_cast<std::size_t>(dstHeight),
                     "opencl-gray-to-rgba-resample", error);
}

bool packOutput(cl_command_queue queue,
                cl_context context,
                const std::shared_ptr<ProgramEntry>& program,
                const LensDiffRenderRequest& request,
                const LensDiffParams& params,
                cl_mem output,
                cl_mem linearSrcNative,
                bool openCLImageMode,
                bool encodeFinal,
                bool preserveSourceAlpha,
                int nativeWidth,
                int nativeHeight,
                std::string* error) {
    const LensDiffImageRect outputRect = intersectRect(request.renderWindow, request.dst.bounds);
    const int renderWidth = outputRect.width();
    const int renderHeight = outputRect.height();
    if (renderWidth <= 0 || renderHeight <= 0) {
        return true;
    }
    const int transfer = params.inputTransfer == LensDiffInputTransfer::DavinciIntermediate ? 1 : 0;
    if (openCLImageMode) {
        cl_mem dstImage = static_cast<cl_mem>(request.dst.openCLImage);
        if (dstImage == nullptr) {
            return setError(error, "opencl-missing-destination-image");
        }
        UniqueKernel kernel = makeKernel(program, "lensDiffPackImage", error);
        if (!kernel) return false;
        cl_uint arg = 0;
        return setKernelArg(kernel.get(), arg++, output, "opencl-pack-image-arg-output", error) &&
               setKernelArg(kernel.get(), arg++, linearSrcNative, "opencl-pack-image-arg-src", error) &&
               setKernelArg(kernel.get(), arg++, dstImage, "opencl-pack-image-arg-dst", error) &&
               setKernelArg(kernel.get(), arg++, nativeWidth, "opencl-pack-image-arg-native-width", error) &&
               setKernelArg(kernel.get(), arg++, nativeHeight, "opencl-pack-image-arg-native-height", error) &&
               setKernelArg(kernel.get(), arg++, outputRect.x1, "opencl-pack-image-arg-render-x", error) &&
               setKernelArg(kernel.get(), arg++, outputRect.y1, "opencl-pack-image-arg-render-y", error) &&
               setKernelArg(kernel.get(), arg++, renderWidth, "opencl-pack-image-arg-render-width", error) &&
               setKernelArg(kernel.get(), arg++, renderHeight, "opencl-pack-image-arg-render-height", error) &&
               setKernelArg(kernel.get(), arg++, request.src.bounds.x1, "opencl-pack-image-arg-src-x1", error) &&
               setKernelArg(kernel.get(), arg++, request.src.bounds.y1, "opencl-pack-image-arg-src-y1", error) &&
               setKernelArg(kernel.get(), arg++, request.dst.bounds.x1, "opencl-pack-image-arg-dst-x1", error) &&
               setKernelArg(kernel.get(), arg++, request.dst.bounds.y1, "opencl-pack-image-arg-dst-y1", error) &&
               setKernelArg(kernel.get(), arg++, transfer, "opencl-pack-image-arg-transfer", error) &&
               setKernelArg(kernel.get(), arg++, static_cast<int>(encodeFinal ? 1 : 0), "opencl-pack-image-arg-encode", error) &&
               setKernelArg(kernel.get(), arg++, static_cast<int>(preserveSourceAlpha ? 1 : 0), "opencl-pack-image-arg-alpha", error) &&
               enqueue2D(queue, kernel.get(), static_cast<std::size_t>(renderWidth), static_cast<std::size_t>(renderHeight),
                         "opencl-pack-image", error);
    }

    cl_mem dstBuffer = static_cast<cl_mem>(request.dst.data);
    if (dstBuffer == nullptr) {
        return setError(error, "opencl-missing-destination-buffer");
    }
    const cl_long rowBytes = static_cast<cl_long>(request.dst.rowBytes);
    UniqueKernel kernel = makeKernel(program, "lensDiffPackBuffer", error);
    if (!kernel) return false;
    cl_uint arg = 0;
    return setKernelArg(kernel.get(), arg++, output, "opencl-pack-buffer-arg-output", error) &&
           setKernelArg(kernel.get(), arg++, linearSrcNative, "opencl-pack-buffer-arg-src", error) &&
           setKernelArg(kernel.get(), arg++, dstBuffer, "opencl-pack-buffer-arg-dst", error) &&
           setKernelArg(kernel.get(), arg++, nativeWidth, "opencl-pack-buffer-arg-native-width", error) &&
           setKernelArg(kernel.get(), arg++, nativeHeight, "opencl-pack-buffer-arg-native-height", error) &&
           setKernelArg(kernel.get(), arg++, rowBytes, "opencl-pack-buffer-arg-rowbytes", error) &&
           setKernelArg(kernel.get(), arg++, outputRect.x1, "opencl-pack-buffer-arg-render-x", error) &&
           setKernelArg(kernel.get(), arg++, outputRect.y1, "opencl-pack-buffer-arg-render-y", error) &&
           setKernelArg(kernel.get(), arg++, renderWidth, "opencl-pack-buffer-arg-render-width", error) &&
           setKernelArg(kernel.get(), arg++, renderHeight, "opencl-pack-buffer-arg-render-height", error) &&
           setKernelArg(kernel.get(), arg++, request.src.bounds.x1, "opencl-pack-buffer-arg-src-x1", error) &&
           setKernelArg(kernel.get(), arg++, request.src.bounds.y1, "opencl-pack-buffer-arg-src-y1", error) &&
           setKernelArg(kernel.get(), arg++, request.dst.bounds.x1, "opencl-pack-buffer-arg-dst-x1", error) &&
           setKernelArg(kernel.get(), arg++, request.dst.bounds.y1, "opencl-pack-buffer-arg-dst-y1", error) &&
           setKernelArg(kernel.get(), arg++, transfer, "opencl-pack-buffer-arg-transfer", error) &&
           setKernelArg(kernel.get(), arg++, static_cast<int>(encodeFinal ? 1 : 0), "opencl-pack-buffer-arg-encode", error) &&
           setKernelArg(kernel.get(), arg++, static_cast<int>(preserveSourceAlpha ? 1 : 0), "opencl-pack-buffer-arg-alpha", error) &&
           enqueue2D(queue, kernel.get(), static_cast<std::size_t>(renderWidth), static_cast<std::size_t>(renderHeight),
                     "opencl-pack-buffer", error);
}

} // namespace

bool RunLensDiffOpenCL(const LensDiffRenderRequest& request,
                       const LensDiffParams& params,
                       LensDiffPsfBankCache& cache,
                       std::string* error) {
    cl_command_queue queue = static_cast<cl_command_queue>(request.openCLCommandQueue);
    if (!request.hostEnabledOpenCLRender || queue == nullptr) {
        return setError(error, "opencl-render-not-enabled");
    }

    cl_context context = nullptr;
    cl_device_id device = nullptr;
    cl_int err = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(context), &context, nullptr);
    if (!checkCl(err, "opencl-get-context", error)) return false;
    err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device), &device, nullptr);
    if (!checkCl(err, "opencl-get-device", error)) return false;
    if (context == nullptr || device == nullptr) {
        return setError(error, "opencl-missing-context-or-device");
    }

    const bool imageMode = request.src.openCLImage != nullptr || request.dst.openCLImage != nullptr;
    if (imageMode) {
        if (request.src.openCLImage == nullptr || request.dst.openCLImage == nullptr) {
            return setError(error, "opencl-image-mode-requires-source-and-destination-images");
        }
    } else if (request.src.data == nullptr || request.dst.data == nullptr) {
        return setError(error, "opencl-buffer-mode-missing-source-or-destination-buffer");
    }

    const int nativeWidth = request.src.bounds.width();
    const int nativeHeight = request.src.bounds.height();
    if (nativeWidth <= 0 || nativeHeight <= 0 || request.dst.bounds.width() <= 0 || request.dst.bounds.height() <= 0) {
        return setError(error, "opencl-invalid-bounds");
    }
    if (!imageMode && (request.src.rowBytes <= 0 || request.dst.rowBytes <= 0)) {
        return setError(error, "opencl-invalid-rowbytes");
    }

    auto program = getProgram(context, device, error);
    if (!program) return false;

    LogLensDiffDiagnosticEvent("opencl-render-mode", imageMode ? "image" : "buffer");

    const double workingScale = ResolveLensDiffEffectWorkingScale(params);
    const bool resolutionAwareActive = params.resolutionAware && std::abs(workingScale - 1.0) > 1.0e-6;
    const int width = resolutionAwareActive ? std::max(1, static_cast<int>(std::lround(nativeWidth * workingScale))) : nativeWidth;
    const int height = resolutionAwareActive ? std::max(1, static_cast<int>(std::lround(nativeHeight * workingScale))) : nativeHeight;
    const int nativeCount = nativeWidth * nativeHeight;
    const int workingCount = width * height;
    const bool splitMode = params.lookMode == LensDiffLookMode::Split;
    const int debug = static_cast<int>(params.debugView);
    const bool needCore = splitMode || (!resolutionAwareActive && debug == kLensDiffDebugCore);
    const bool needStructure = splitMode || (!resolutionAwareActive && debug == kLensDiffDebugStructure);
    const bool staticDebug = params.debugView == LensDiffDebugView::Pupil ||
                             params.debugView == LensDiffDebugView::Psf ||
                             params.debugView == LensDiffDebugView::Otf ||
                             params.debugView == LensDiffDebugView::Phase ||
                             params.debugView == LensDiffDebugView::PhaseEdge ||
                             params.debugView == LensDiffDebugView::FieldPsf ||
                             params.debugView == LensDiffDebugView::ChromaticSplit;

    if (staticDebug) {
        EnsureLensDiffPsfBank(params, cache);
        const std::vector<float>& staticDebugRgba =
            GetLensDiffStaticDebugRgbaCached(params, &cache, nativeWidth, nativeHeight);
        if (staticDebugRgba.empty()) {
            return setError(error, "opencl-static-debug-empty");
        }
        UniqueMem debugUpload = makeBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           staticDebugRgba.size() * sizeof(float),
                                           const_cast<float*>(staticDebugRgba.data()), "debug-static", error);
        if (!debugUpload) return false;
        if (!packOutput(queue, context, program, request, params, debugUpload.get(), debugUpload.get(), imageMode,
                        false, false, nativeWidth, nativeHeight, error) ||
            !checkCl(clFinish(queue), "opencl-finish-static-debug", error)) {
            return false;
        }
        return true;
    }

    UniqueMem linearNative = makeRgbaBuffer(context, static_cast<std::size_t>(nativeCount), "linear-native", error);
    UniqueMem workingSrc = makeRgbaBuffer(context, static_cast<std::size_t>(workingCount), "working-src", error);
    UniqueMem mask = makeFloatBuffer(context, static_cast<std::size_t>(workingCount), "mask", error);
    UniqueMem redistributed = makeRgbaBuffer(context, static_cast<std::size_t>(workingCount), "redistributed", error);
    UniqueMem driver = makeFloatBuffer(context, static_cast<std::size_t>(workingCount), "driver", error);
    if (!linearNative || !workingSrc || !mask || !redistributed || !driver) {
        return false;
    }

    const int transfer = params.inputTransfer == LensDiffInputTransfer::DavinciIntermediate ? 1 : 0;
    if (imageMode) {
        UniqueKernel decode = makeKernel(program, "lensDiffDecodeImage", error);
        if (!decode) return false;
        cl_mem srcImage = static_cast<cl_mem>(request.src.openCLImage);
        cl_uint arg = 0;
        if (!setKernelArg(decode.get(), arg++, srcImage, "opencl-decode-image-arg-src", error) ||
            !setKernelArg(decode.get(), arg++, linearNative.get(), "opencl-decode-image-arg-dst", error) ||
            !setKernelArg(decode.get(), arg++, nativeWidth, "opencl-decode-image-arg-width", error) ||
            !setKernelArg(decode.get(), arg++, nativeHeight, "opencl-decode-image-arg-height", error) ||
            !setKernelArg(decode.get(), arg++, transfer, "opencl-decode-image-arg-transfer", error) ||
            !enqueue2D(queue, decode.get(), static_cast<std::size_t>(nativeWidth), static_cast<std::size_t>(nativeHeight),
                       "opencl-decode-image", error)) {
            return false;
        }
    } else {
        UniqueKernel decode = makeKernel(program, "lensDiffDecodeBuffer", error);
        if (!decode) return false;
        cl_mem srcBuffer = static_cast<cl_mem>(request.src.data);
        const cl_long rowBytes = static_cast<cl_long>(request.src.rowBytes);
        cl_uint arg = 0;
        if (!setKernelArg(decode.get(), arg++, srcBuffer, "opencl-decode-buffer-arg-src", error) ||
            !setKernelArg(decode.get(), arg++, linearNative.get(), "opencl-decode-buffer-arg-dst", error) ||
            !setKernelArg(decode.get(), arg++, nativeWidth, "opencl-decode-buffer-arg-width", error) ||
            !setKernelArg(decode.get(), arg++, nativeHeight, "opencl-decode-buffer-arg-height", error) ||
            !setKernelArg(decode.get(), arg++, rowBytes, "opencl-decode-buffer-arg-rowbytes", error) ||
            !setKernelArg(decode.get(), arg++, transfer, "opencl-decode-buffer-arg-transfer", error) ||
            !enqueue2D(queue, decode.get(), static_cast<std::size_t>(nativeWidth), static_cast<std::size_t>(nativeHeight),
                       "opencl-decode-buffer", error)) {
            return false;
        }
    }

    {
        UniqueKernel prepare = makeKernel(program, "lensDiffPrepare", error);
        if (!prepare) return false;
        const int extraction = params.extractionMode == LensDiffExtractionMode::Luma ? 1 : 0;
        const float threshold = static_cast<float>(params.threshold);
        const float softness = static_cast<float>(params.softnessStops);
        const float pointEmphasis = static_cast<float>(params.pointEmphasis);
        const float redistributionScale = 1.0f - static_cast<float>(std::clamp(params.corePreserve, 0.0, 1.0));
        cl_uint arg = 0;
        if (!setKernelArg(prepare.get(), arg++, linearNative.get(), "opencl-prepare-arg-native", error) ||
            !setKernelArg(prepare.get(), arg++, workingSrc.get(), "opencl-prepare-arg-working", error) ||
            !setKernelArg(prepare.get(), arg++, mask.get(), "opencl-prepare-arg-mask", error) ||
            !setKernelArg(prepare.get(), arg++, redistributed.get(), "opencl-prepare-arg-redistributed", error) ||
            !setKernelArg(prepare.get(), arg++, driver.get(), "opencl-prepare-arg-driver", error) ||
            !setKernelArg(prepare.get(), arg++, nativeWidth, "opencl-prepare-arg-native-width", error) ||
            !setKernelArg(prepare.get(), arg++, nativeHeight, "opencl-prepare-arg-native-height", error) ||
            !setKernelArg(prepare.get(), arg++, width, "opencl-prepare-arg-width", error) ||
            !setKernelArg(prepare.get(), arg++, height, "opencl-prepare-arg-height", error) ||
            !setKernelArg(prepare.get(), arg++, extraction, "opencl-prepare-arg-extraction", error) ||
            !setKernelArg(prepare.get(), arg++, threshold, "opencl-prepare-arg-threshold", error) ||
            !setKernelArg(prepare.get(), arg++, softness, "opencl-prepare-arg-softness", error) ||
            !setKernelArg(prepare.get(), arg++, pointEmphasis, "opencl-prepare-arg-point", error) ||
            !setKernelArg(prepare.get(), arg++, redistributionScale, "opencl-prepare-arg-redist-scale", error) ||
            !enqueue2D(queue, prepare.get(), static_cast<std::size_t>(width), static_cast<std::size_t>(height),
                       "opencl-prepare", error)) {
            return false;
        }
    }

    if (params.debugView == LensDiffDebugView::Selection) {
        UniqueMem debugSelection = makeRgbaBuffer(context, static_cast<std::size_t>(nativeCount), "debug-selection", error);
        if (!debugSelection) return false;
        if (!runGrayToRgbaResample(queue, program, mask.get(), debugSelection.get(),
                                   width, height, nativeWidth, nativeHeight, error) ||
            !packOutput(queue, context, program, request, params, debugSelection.get(), linearNative.get(), imageMode,
                        false, false, nativeWidth, nativeHeight, error) ||
            !checkCl(clFinish(queue), "opencl-finish-selection", error)) {
            return false;
        }
        return true;
    }

    EnsureLensDiffPsfBank(params, cache);
    if (cache.bins.empty()) {
        return setError(error, "opencl-psf-cache-empty");
    }

    const double scatterRadiusPx = ResolveLensDiffScatterRadiusPx(params);
    const bool scatterActive = params.scatterAmount > 1.0e-6 && scatterRadiusPx > 0.25;
    const bool needScatterPreview = scatterActive || (!resolutionAwareActive && debug == kLensDiffDebugScatter);
    const float creativeFringePx = static_cast<float>(std::max(0.0, ResolveLensDiffCreativeFringePx(params)));
    const bool creativeFringeActive = creativeFringePx > 1.0e-6f;
    const bool needCreativePreview = creativeFringeActive ||
                                     (!resolutionAwareActive && debug == kLensDiffDebugCreativeFringe);
    UniqueMem coreEffect = needCore ? makeRgbaBuffer(context, static_cast<std::size_t>(workingCount), "core-effect", error)
                                    : UniqueMem(nullptr);
    UniqueMem structureEffect = needStructure ? makeRgbaBuffer(context, static_cast<std::size_t>(workingCount), "structure-effect", error)
                                              : UniqueMem(nullptr);
    UniqueMem effect = makeRgbaBuffer(context, static_cast<std::size_t>(workingCount), "effect", error);
    UniqueMem finalImage = makeRgbaBuffer(context, static_cast<std::size_t>(nativeCount), "final", error);
    UniqueMem scatterPreview = needScatterPreview ? makeRgbaBuffer(context, static_cast<std::size_t>(workingCount), "scatter-preview", error)
                                                  : UniqueMem(nullptr);
    UniqueMem creativePreview = needCreativePreview ? makeRgbaBuffer(context, static_cast<std::size_t>(workingCount), "creative-preview", error)
                                                    : UniqueMem(nullptr);
    if ((needCore && !coreEffect) || (needStructure && !structureEffect) ||
        !effect || !finalImage || (needScatterPreview && !scatterPreview) ||
        (needCreativePreview && !creativePreview)) {
        return false;
    }

    std::unordered_map<std::string, UniqueMem> rgbSourceSpectrumCache;
    std::unordered_map<std::string, UniqueMem> scalarSourceSpectrumCache;
    std::unordered_map<std::string, std::shared_ptr<PersistentKernelSpectrum>> kernelSpectrumCache;

    if (cache.fieldZones.empty()) {
        if (!renderBins(queue, context, program, redistributed.get(), driver.get(), cache.bins, params,
                        needCore ? coreEffect.get() : nullptr,
                        needStructure ? structureEffect.get() : nullptr,
                        effect.get(), width, height,
                        rgbSourceSpectrumCache, scalarSourceSpectrumCache, kernelSpectrumCache, error)) {
            return false;
        }
    } else {
        if ((needCore && !runClear(queue, program, coreEffect.get(), workingCount, error)) ||
            (needStructure && !runClear(queue, program, structureEffect.get(), workingCount, error)) ||
            !runClear(queue, program, effect.get(), workingCount, error)) {
            return false;
        }
        UniqueMem zoneCore = needCore ? makeRgbaBuffer(context, static_cast<std::size_t>(workingCount), "zone-core", error)
                                      : UniqueMem(nullptr);
        UniqueMem zoneStructure = needStructure ? makeRgbaBuffer(context, static_cast<std::size_t>(workingCount), "zone-structure", error)
                                                : UniqueMem(nullptr);
        UniqueMem zoneEffect = makeRgbaBuffer(context, static_cast<std::size_t>(workingCount), "zone-effect", error);
        if ((needCore && !zoneCore) || (needStructure && !zoneStructure) || !zoneEffect) return false;
        for (const auto& zone : cache.fieldZones) {
            if (!renderBins(queue, context, program, redistributed.get(), driver.get(), zone.bins, zone.resolvedParams,
                            needCore ? zoneCore.get() : nullptr,
                            needStructure ? zoneStructure.get() : nullptr,
                            zoneEffect.get(), width, height,
                            rgbSourceSpectrumCache, scalarSourceSpectrumCache, kernelSpectrumCache, error) ||
                (needCore && !runAccumulate(queue, program, zoneCore.get(), coreEffect.get(), width, height, zone.zoneX, zone.zoneY, error)) ||
                (needStructure && !runAccumulate(queue, program, zoneStructure.get(), structureEffect.get(), width, height, zone.zoneX, zone.zoneY, error)) ||
                !runAccumulate(queue, program, zoneEffect.get(), effect.get(), width, height, zone.zoneX, zone.zoneY, error)) {
                return false;
            }
        }
    }

    if (params.energyMode == LensDiffEnergyMode::Preserve) {
        float inputEnergy = 0.0f;
        float effectEnergy = 0.0f;
        if (!reduceLuma(queue, context, program, redistributed.get(), workingCount, &inputEnergy, error) ||
            !reduceLuma(queue, context, program, effect.get(), workingCount, &effectEnergy, error)) {
            return false;
        }
        if (effectEnergy > 1.0e-6f) {
            if (!runScaleRgba(queue, program, effect.get(), workingCount, inputEnergy / effectEnergy, error)) {
                return false;
            }
        }
    }

    if (scatterActive) {
        const std::vector<float> gaussian = buildGaussianKernel(static_cast<float>(scatterRadiusPx));
        LensDiffKernel kernel {};
        kernel.size = static_cast<int>(std::sqrt(static_cast<double>(gaussian.size())));
        kernel.values = gaussian;
        const int paddedSize = nextPowerOfTwo(std::max(width, height) + kernel.size - 1);
        cl_mem sourceSpectrum = nullptr;
        cl_mem kernelSpectrum = nullptr;
        if (!getRgbaSourceSpectrum(queue, context, program, rgbSourceSpectrumCache, effect.get(),
                                   width, height, paddedSize, &sourceSpectrum, error) ||
            !getKernelSpectrum(queue, context, program, kernelSpectrumCache, kernel, paddedSize,
                               &kernelSpectrum, error) ||
            !runConvolveRgbaFromSpectra(queue, context, program, sourceSpectrum, kernelSpectrum, effect.get(),
                                        scatterPreview.get(), width, height, paddedSize, 0.0f, error) ||
            !runAddImages(queue, program, effect.get(), scatterPreview.get(), effect.get(), workingCount, 1.0f,
                          static_cast<float>(std::max(0.0, params.scatterAmount)), error)) {
            return false;
        }
    } else if (needScatterPreview && !runClear(queue, program, scatterPreview.get(), workingCount, error)) {
        return false;
    }

    if (creativeFringeActive || (!resolutionAwareActive && debug == kLensDiffDebugCreativeFringe)) {
        UniqueMem fringed = makeRgbaBuffer(context, static_cast<std::size_t>(workingCount), "fringed-effect", error);
        if (!fringed) return false;
        if (!runCreativeFringe(queue, program, effect.get(), fringed.get(), creativePreview.get(),
                               width, height, creativeFringePx, error)) {
            return false;
        }
        if (creativeFringeActive) {
            std::swap(effect, fringed);
        }
    }

    if (!runComposite(queue, program, linearNative.get(), redistributed.get(), effect.get(), finalImage.get(),
                      nativeWidth, nativeHeight, width, height, params, error)) {
        return false;
    }

    cl_mem output = finalImage.get();
    UniqueMem debugUpload;
    UniqueMem debugDynamic;
    bool encodeFinal = true;
    bool preserveSourceAlpha = true;
    if (debug != kLensDiffDebugFinal) {
        encodeFinal = false;
        preserveSourceAlpha = false;
        if (debug == kLensDiffDebugSelection) {
            debugDynamic = makeRgbaBuffer(context, static_cast<std::size_t>(nativeCount), "debug-selection", error);
            if (!debugDynamic) return false;
            if (resolutionAwareActive) {
                // Selection is a working-size scalar; debug it at working resolution only when dimensions match.
                // Static fallback is clearer than writing mismatched coordinates.
                const std::vector<float>& staticDebug =
                    GetLensDiffStaticDebugRgbaCached(params, &cache, nativeWidth, nativeHeight);
                debugUpload = makeBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         staticDebug.size() * sizeof(float),
                                         const_cast<float*>(staticDebug.data()), "debug-static", error);
                if (!debugUpload) return false;
                output = debugUpload.get();
            } else {
                if (!runGrayToRgba(queue, program, mask.get(), debugDynamic.get(), nativeWidth, nativeHeight, error)) {
                    return false;
                }
                output = debugDynamic.get();
            }
        } else if (!resolutionAwareActive && debug == kLensDiffDebugCore) {
            output = coreEffect.get();
        } else if (!resolutionAwareActive && debug == kLensDiffDebugStructure) {
            output = structureEffect.get();
        } else if (!resolutionAwareActive && debug == kLensDiffDebugEffect) {
            output = effect.get();
        } else if (!resolutionAwareActive && debug == kLensDiffDebugCreativeFringe) {
            output = creativePreview.get();
        } else if (!resolutionAwareActive && debug == kLensDiffDebugScatter) {
            output = scatterPreview.get();
        } else {
            const std::vector<float>& staticDebug =
                GetLensDiffStaticDebugRgbaCached(params, &cache, nativeWidth, nativeHeight);
            if (staticDebug.empty()) {
                return setError(error, "opencl-static-debug-empty");
            }
            debugUpload = makeBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     staticDebug.size() * sizeof(float),
                                     const_cast<float*>(staticDebug.data()), "debug-static", error);
            if (!debugUpload) return false;
            output = debugUpload.get();
        }
    }

    if (!packOutput(queue, context, program, request, params, output, linearNative.get(), imageMode,
                    encodeFinal, preserveSourceAlpha, nativeWidth, nativeHeight, error)) {
        return false;
    }

    if (!checkCl(clFinish(queue), "opencl-finish", error)) {
        return false;
    }
    return true;
}

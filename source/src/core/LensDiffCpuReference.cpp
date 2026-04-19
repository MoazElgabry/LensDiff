#include "LensDiffCpuReference.h"
#include "LensDiffApertureImage.h"
#include "LensDiffDiagnostics.h"
#include "LensDiffPhase.h"
#include "LensDiffSpectrum.h"
#include "LensDiffTransfer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

namespace {

using Complex = std::complex<float>;

constexpr float kPi = 3.14159265358979323846f;
constexpr float kGray18 = 0.18f;
constexpr float kMinimumSelectedCoreFloor = 0.2f;

std::mutex& referenceRawPsfCacheMutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<std::pair<int, int>, std::shared_ptr<const std::vector<float>>>& referenceRawPsfCache() {
    static std::map<std::pair<int, int>, std::shared_ptr<const std::vector<float>>> cache;
    return cache;
}

struct ScalarImage {
    int width = 0;
    int height = 0;
    std::vector<float> pixels;

    ScalarImage() = default;
    ScalarImage(int w, int h) : width(w), height(h), pixels(static_cast<std::size_t>(w) * h, 0.0f) {}

    float& at(int x, int y) { return pixels[static_cast<std::size_t>(y) * width + x]; }
    float at(int x, int y) const { return pixels[static_cast<std::size_t>(y) * width + x]; }
};

struct RgbaImage {
    int width = 0;
    int height = 0;
    std::vector<float> pixels;

    RgbaImage() = default;
    RgbaImage(int w, int h) : width(w), height(h), pixels(static_cast<std::size_t>(w) * h * 4, 0.0f) {}

    float* pixel(int x, int y) { return pixels.data() + (static_cast<std::size_t>(y) * width + x) * 4; }
    const float* pixel(int x, int y) const { return pixels.data() + (static_cast<std::size_t>(y) * width + x) * 4; }
};

template <typename T>
T clampValue(T v, T lo, T hi) {
    return std::max(lo, std::min(v, hi));
}

float saturate(float v) {
    return clampValue(v, 0.0f, 1.0f);
}

float safeLuma(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

float smoothStep01(float t) {
    const float x = saturate(t);
    return x * x * (3.0f - 2.0f * x);
}

float automaticCoreProtectionFraction(const LensDiffParams& params) {
    return std::max(kMinimumSelectedCoreFloor, static_cast<float>(clampValue(params.corePreserve, 0.0, 1.0)));
}

float softShoulder(float v, float shoulder) {
    if (shoulder <= 0.0f) {
        return std::max(0.0f, v);
    }
    const float x = std::max(0.0f, v);
    return shoulder * (1.0f - std::exp(-x / shoulder));
}

int nextPowerOfTwo(int v) {
    int n = 1;
    while (n < v) {
        n <<= 1;
    }
    return n;
}

bool isPowerOfTwo(int v) {
    return v > 0 && (v & (v - 1)) == 0;
}

int effectivePupilResolution(int requested) {
    return clampValue(requested, 64, 1024);
}

float shiftedSpectrumCenterCoordinate(int size) {
    return (size & 1) == 0 ? static_cast<float>(size) * 0.5f : static_cast<float>(size - 1) * 0.5f;
}

int chooseRawPsfSize(int pupilSize, int maxKernelRadiusPx) {
    const int requested = std::max(pupilSize * 2, std::max(512, maxKernelRadiusPx * 4));
    return clampValue(requested, 128, 16384);
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

std::array<float, 4> readPixel(const LensDiffImageView& view, int x, int y) {
    std::array<float, 4> rgba {0.0f, 0.0f, 0.0f, 0.0f};
    if (view.data == nullptr) {
        return rgba;
    }
    if (x < view.bounds.x1 || x >= view.bounds.x2 || y < view.bounds.y1 || y >= view.bounds.y2) {
        return rgba;
    }

    auto* base = static_cast<const std::uint8_t*>(view.data);
    const std::ptrdiff_t rowOffset = static_cast<std::ptrdiff_t>(y - view.bounds.y1) * view.rowBytes;
    const std::ptrdiff_t colOffset = static_cast<std::ptrdiff_t>(x - view.bounds.x1) *
                                     static_cast<std::ptrdiff_t>(view.components) *
                                     static_cast<std::ptrdiff_t>(view.bytesPerComponent);
    auto* pixel = reinterpret_cast<const float*>(base + rowOffset + colOffset);

    rgba[0] = pixel[0];
    rgba[1] = pixel[1];
    rgba[2] = pixel[2];
    rgba[3] = view.components > 3 ? pixel[3] : 1.0f;
    return rgba;
}

void writePixel(const LensDiffImageView& view, int x, int y, const std::array<float, 4>& rgba) {
    if (view.data == nullptr) {
        return;
    }
    if (x < view.bounds.x1 || x >= view.bounds.x2 || y < view.bounds.y1 || y >= view.bounds.y2) {
        return;
    }

    auto* base = static_cast<std::uint8_t*>(view.data);
    const std::ptrdiff_t rowOffset = static_cast<std::ptrdiff_t>(y - view.bounds.y1) * view.rowBytes;
    const std::ptrdiff_t colOffset = static_cast<std::ptrdiff_t>(x - view.bounds.x1) *
                                     static_cast<std::ptrdiff_t>(view.components) *
                                     static_cast<std::ptrdiff_t>(view.bytesPerComponent);
    auto* pixel = reinterpret_cast<float*>(base + rowOffset + colOffset);

    pixel[0] = rgba[0];
    pixel[1] = rgba[1];
    pixel[2] = rgba[2];
    if (view.components > 3) {
        pixel[3] = rgba[3];
    }
}

RgbaImage loadSourceRgba(const LensDiffImageView& src, LensDiffInputTransfer transfer) {
    const int w = src.bounds.width();
    const int h = src.bounds.height();
    RgbaImage image(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const auto rgba = readPixel(src, src.bounds.x1 + x, src.bounds.y1 + y);
            const WorkshopColor::Vec3f decoded = LensDiffTransfer::decodeToLinear(
                {rgba[0], rgba[1], rgba[2]},
                transfer);
            float* out = image.pixel(x, y);
            out[0] = decoded.x;
            out[1] = decoded.y;
            out[2] = decoded.z;
            out[3] = rgba[3];
        }
    }
    return image;
}

void radix2Fft1d(std::vector<Complex>& data, bool inverse) {
    const int n = static_cast<int>(data.size());
    if (!isPowerOfTwo(n)) {
        return;
    }

    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        const float angle = 2.0f * kPi / static_cast<float>(len) * (inverse ? 1.0f : -1.0f);
        const Complex wlen(std::cos(angle), std::sin(angle));
        for (int i = 0; i < n; i += len) {
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
        const float invN = 1.0f / static_cast<float>(n);
        for (auto& value : data) {
            value *= invN;
        }
    }
}

void fft1d(std::vector<Complex>& data, bool inverse) {
    const int n = static_cast<int>(data.size());
    if (n <= 1) {
        return;
    }
    if (isPowerOfTwo(n)) {
        radix2Fft1d(data, inverse);
        return;
    }

    const int m = nextPowerOfTwo(n * 2 - 1);
    std::vector<Complex> a(static_cast<std::size_t>(m), Complex(0.0f, 0.0f));
    std::vector<Complex> b(static_cast<std::size_t>(m), Complex(0.0f, 0.0f));
    const double sign = inverse ? 1.0 : -1.0;

    for (int i = 0; i < n; ++i) {
        const double index = static_cast<double>(i);
        const double phase = sign * static_cast<double>(kPi) * index * index / static_cast<double>(n);
        const Complex chirp(static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase)));
        const Complex chirpConjugate(static_cast<float>(std::cos(-phase)), static_cast<float>(std::sin(-phase)));
        a[static_cast<std::size_t>(i)] = data[static_cast<std::size_t>(i)] * chirp;
        b[static_cast<std::size_t>(i)] = chirpConjugate;
        if (i != 0) {
            b[static_cast<std::size_t>(m - i)] = chirpConjugate;
        }
    }

    radix2Fft1d(a, false);
    radix2Fft1d(b, false);
    for (int i = 0; i < m; ++i) {
        a[static_cast<std::size_t>(i)] *= b[static_cast<std::size_t>(i)];
    }
    radix2Fft1d(a, true);

    const float inverseScale = inverse ? (1.0f / static_cast<float>(n)) : 1.0f;
    for (int i = 0; i < n; ++i) {
        const double index = static_cast<double>(i);
        const double phase = sign * static_cast<double>(kPi) * index * index / static_cast<double>(n);
        const Complex chirp(static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase)));
        data[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i)] * chirp * inverseScale;
    }
}

void fft2d(std::vector<Complex>& data, int width, int height, bool inverse) {
    std::vector<Complex> scratch(static_cast<std::size_t>(std::max(width, height)));

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
        std::vector<Complex> col(static_cast<std::size_t>(height));
        for (int y = 0; y < height; ++y) {
            col[y] = data[static_cast<std::size_t>(y) * width + x];
        }
        fft1d(col, inverse);
        for (int y = 0; y < height; ++y) {
            data[static_cast<std::size_t>(y) * width + x] = col[y];
        }
    }
}

std::vector<float> fftShiftSquare(const std::vector<float>& src, int size) {
    std::vector<float> shifted(src.size(), 0.0f);
    const int half = size / 2;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const int sx = (x + half) % size;
            const int sy = (y + half) % size;
            shifted[static_cast<std::size_t>(y) * size + x] = src[static_cast<std::size_t>(sy) * size + sx];
        }
    }
    return shifted;
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

float sampleSquareFiltered(const std::vector<float>& image, int size, float x, float y, float footprint) {
    if (footprint <= 1.0f) {
        return sampleSquareBilinear(image, size, x, y);
    }
    const int taps = clampValue(static_cast<int>(std::ceil(footprint)), 2, 6);
    const float step = footprint / static_cast<float>(taps);
    float sum = 0.0f;
    for (int ty = 0; ty < taps; ++ty) {
        const float oy = (static_cast<float>(ty) + 0.5f) * step - footprint * 0.5f;
        for (int tx = 0; tx < taps; ++tx) {
            const float ox = (static_cast<float>(tx) + 0.5f) * step - footprint * 0.5f;
            sum += sampleSquareBilinear(image, size, x + ox, y + oy);
        }
    }
    return sum / static_cast<float>(taps * taps);
}

float sampleScalarBilinear(const ScalarImage& image, float x, float y) {
    if (x < 0.0f || y < 0.0f || x > static_cast<float>(image.width - 1) || y > static_cast<float>(image.height - 1)) {
        return 0.0f;
    }
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image.width - 1);
    const int y1 = std::min(y0 + 1, image.height - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float v00 = image.at(x0, y0);
    const float v10 = image.at(x1, y0);
    const float v01 = image.at(x0, y1);
    const float v11 = image.at(x1, y1);
    const float vx0 = v00 + (v10 - v00) * tx;
    const float vx1 = v01 + (v11 - v01) * tx;
    return vx0 + (vx1 - vx0) * ty;
}

std::array<float, 4> sampleRgbaBilinearImage(const RgbaImage& image, float x, float y) {
    std::array<float, 4> out {0.0f, 0.0f, 0.0f, 0.0f};
    if (x < 0.0f || y < 0.0f || x > static_cast<float>(image.width - 1) || y > static_cast<float>(image.height - 1)) {
        return out;
    }
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image.width - 1);
    const int y1 = std::min(y0 + 1, image.height - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float* p00 = image.pixel(x0, y0);
    const float* p10 = image.pixel(x1, y0);
    const float* p01 = image.pixel(x0, y1);
    const float* p11 = image.pixel(x1, y1);
    for (int c = 0; c < 4; ++c) {
        const float vx0 = p00[c] + (p10[c] - p00[c]) * tx;
        const float vx1 = p01[c] + (p11[c] - p01[c]) * tx;
        out[static_cast<std::size_t>(c)] = vx0 + (vx1 - vx0) * ty;
    }
    return out;
}

ScalarImage resampleScalarImage(const ScalarImage& src, int outWidth, int outHeight) {
    if (src.width == outWidth && src.height == outHeight) {
        return src;
    }
    ScalarImage out(outWidth, outHeight);
    const float scaleX = static_cast<float>(src.width) / static_cast<float>(std::max(1, outWidth));
    const float scaleY = static_cast<float>(src.height) / static_cast<float>(std::max(1, outHeight));
    for (int y = 0; y < outHeight; ++y) {
        for (int x = 0; x < outWidth; ++x) {
            const float sx = (static_cast<float>(x) + 0.5f) * scaleX - 0.5f;
            const float sy = (static_cast<float>(y) + 0.5f) * scaleY - 0.5f;
            out.at(x, y) = sampleScalarBilinear(src, sx, sy);
        }
    }
    return out;
}

RgbaImage resampleRgbaImage(const RgbaImage& src, int outWidth, int outHeight) {
    if (src.width == outWidth && src.height == outHeight) {
        return src;
    }
    RgbaImage out(outWidth, outHeight);
    const float scaleX = static_cast<float>(src.width) / static_cast<float>(std::max(1, outWidth));
    const float scaleY = static_cast<float>(src.height) / static_cast<float>(std::max(1, outHeight));
    for (int y = 0; y < outHeight; ++y) {
        for (int x = 0; x < outWidth; ++x) {
            const float sx = (static_cast<float>(x) + 0.5f) * scaleX - 0.5f;
            const float sy = (static_cast<float>(y) + 0.5f) * scaleY - 0.5f;
            const auto pixel = sampleRgbaBilinearImage(src, sx, sy);
            float* outPixel = out.pixel(x, y);
            outPixel[0] = pixel[0];
            outPixel[1] = pixel[1];
            outPixel[2] = pixel[2];
            outPixel[3] = pixel[3];
        }
    }
    return out;
}

float sumKernel(const std::vector<float>& kernel) {
    return std::accumulate(kernel.begin(), kernel.end(), 0.0f);
}

void normalizeKernel(std::vector<float>& kernel) {
    const float total = sumKernel(kernel);
    if (total <= 0.0f) {
        return;
    }
    const float inv = 1.0f / total;
    for (float& value : kernel) {
        value *= inv;
    }
}

float polygonMetric(float nx, float ny, int bladeCount, float roundness, float rotationRad, float outerRadius) {
    const float radius = std::sqrt(nx * nx + ny * ny);
    const float circleMetric = radius / std::max(outerRadius, 1e-5f);
    if (bladeCount < 3) {
        return circleMetric;
    }

    const float angle = std::atan2(ny, nx) - rotationRad;
    const float sector = 2.0f * kPi / static_cast<float>(bladeCount);
    float local = std::fmod(angle, sector);
    if (local < 0.0f) {
        local += sector;
    }
    local -= sector * 0.5f;
    const float polygonRadius = outerRadius * std::cos(kPi / static_cast<float>(bladeCount)) /
                                std::max(std::cos(local), 1e-4f);
    const float polygonMetricValue = radius / std::max(polygonRadius, 1e-5f);
    return circleMetric * roundness + polygonMetricValue * (1.0f - roundness);
}

bool blockedByVanes(float nx, float ny, int vaneCount, float thickness, float rotationRad, float outerRadius) {
    if (vaneCount <= 0 || thickness <= 0.0f) {
        return false;
    }
    const float scaledThickness = thickness * outerRadius;
    const int lineCount = std::max(1, vaneCount);
    for (int i = 0; i < lineCount; ++i) {
        const float angle = rotationRad + static_cast<float>(i) * kPi / static_cast<float>(lineCount);
        const float cs = std::cos(angle);
        const float sn = std::sin(angle);
        const float xr = nx * cs + ny * sn;
        const float yr = -nx * sn + ny * cs;
        if (std::abs(yr) <= scaledThickness && std::abs(xr) <= outerRadius) {
            return true;
        }
    }
    return false;
}

std::vector<std::array<float, 2>> buildStarVertices(int points,
                                                    float innerRadiusRatio,
                                                    float rotationRad,
                                                    float outerRadius) {
    const int pointCount = std::max(3, points);
    const float innerRadius = outerRadius * clampValue(innerRadiusRatio, 0.1f, 0.95f);
    const float angleStep = kPi / static_cast<float>(pointCount);
    const float startAngle = rotationRad - kPi * 0.5f;

    std::vector<std::array<float, 2>> vertices;
    vertices.reserve(static_cast<std::size_t>(pointCount) * 2U);
    for (int i = 0; i < pointCount * 2; ++i) {
        const float angle = startAngle + static_cast<float>(i) * angleStep;
        const float radius = (i & 1) == 0 ? outerRadius : innerRadius;
        vertices.push_back({radius * std::cos(angle), radius * std::sin(angle)});
    }
    return vertices;
}

float starMetric(float nx,
                 float ny,
                 int points,
                 float innerRadiusRatio,
                 float rotationRad,
                 float outerRadius) {
    const float radius = std::sqrt(nx * nx + ny * ny);
    if (radius <= 1e-6f) {
        return 0.0f;
    }

    const int pointCount = std::max(3, points);
    const float innerRadius = outerRadius * clampValue(innerRadiusRatio, 0.1f, 0.95f);
    const float angle = std::atan2(ny, nx) - rotationRad + kPi * 0.5f;
    const float sector = 2.0f * kPi / static_cast<float>(pointCount);
    float local = std::fmod(angle, sector);
    if (local < 0.0f) {
        local += sector;
    }
    const float halfSector = sector * 0.5f;
    const float t = local <= halfSector
        ? local / std::max(halfSector, 1e-6f)
        : (sector - local) / std::max(halfSector, 1e-6f);
    const float boundaryRadius = innerRadius + (outerRadius - innerRadius) * clampValue(t, 0.0f, 1.0f);
    return radius / std::max(boundaryRadius, 1e-5f);
}

bool pointInPolygon(const std::vector<std::array<float, 2>>& polygon, float x, float y) {
    if (polygon.size() < 3) {
        return false;
    }

    bool inside = false;
    std::size_t j = polygon.size() - 1;
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const float xi = polygon[i][0];
        const float yi = polygon[i][1];
        const float xj = polygon[j][0];
        const float yj = polygon[j][1];
        const bool intersects = ((yi > y) != (yj > y)) &&
                                (x < (xj - xi) * (y - yi) / std::max(yj - yi, 1e-6f) + xi);
        if (intersects) {
            inside = !inside;
        }
        j = i;
    }
    return inside;
}

float distancePointToSegment(float px, float py, float ax, float ay, float bx, float by) {
    const float vx = bx - ax;
    const float vy = by - ay;
    const float len2 = vx * vx + vy * vy;
    if (len2 <= 1e-10f) {
        const float dx = px - ax;
        const float dy = py - ay;
        return std::sqrt(dx * dx + dy * dy);
    }

    const float t = clampValue(((px - ax) * vx + (py - ay) * vy) / len2, 0.0f, 1.0f);
    const float cx = ax + t * vx;
    const float cy = ay + t * vy;
    const float dx = px - cx;
    const float dy = py - cy;
    return std::sqrt(dx * dx + dy * dy);
}

bool squareGridMaskOpen(float nx,
                        float ny,
                        int bladeCount,
                        float roundness,
                        float rotationRad,
                        float outerRadius) {
    const float cs = std::cos(-rotationRad);
    const float sn = std::sin(-rotationRad);
    const float rx = nx * cs - ny * sn;
    const float ry = nx * sn + ny * cs;
    const float squareHalf = outerRadius * 0.82f;
    if (std::abs(rx) > squareHalf || std::abs(ry) > squareHalf) {
        return false;
    }

    const int divisions = std::max(3, bladeCount);
    const float pitch = (2.0f * squareHalf) / static_cast<float>(divisions);
    const float barHalf = pitch * (0.06f + (1.0f - clampValue(roundness, 0.0f, 1.0f)) * 0.12f);

    auto distanceToNearestLine = [&](float v) {
        const float wrapped = std::fmod(v + squareHalf, pitch);
        const float positive = wrapped < 0.0f ? wrapped + pitch : wrapped;
        return std::min(positive, pitch - positive);
    };

    const float distX = distanceToNearestLine(rx);
    const float distY = distanceToNearestLine(ry);
    const float edgeDistX = std::abs(std::abs(rx) - squareHalf);
    const float edgeDistY = std::abs(std::abs(ry) - squareHalf);
    return distX <= barHalf || distY <= barHalf || edgeDistX <= barHalf || edgeDistY <= barHalf;
}

bool snowflakeMaskOpen(float nx,
                       float ny,
                       int bladeCount,
                       float roundness,
                       float rotationRad,
                       float outerRadius) {
    const float radius = std::sqrt(nx * nx + ny * ny);
    if (radius > outerRadius) {
        return false;
    }

    const int branchLevels = clampValue(std::max(2, bladeCount / 2), 2, 5);
    const float mainLength = outerRadius * 0.92f;
    const float branchAngle = kPi / 5.5f;
    const float baseThickness = outerRadius * (0.045f + (1.0f - clampValue(roundness, 0.0f, 1.0f)) * 0.05f);
    float minDistance = std::numeric_limits<float>::max();

    for (int arm = 0; arm < 6; ++arm) {
        const float angle = rotationRad - kPi * 0.5f + static_cast<float>(arm) * (kPi / 3.0f);
        const float cs = std::cos(angle);
        const float sn = std::sin(angle);
        const float ex = cs * mainLength;
        const float ey = sn * mainLength;
        minDistance = std::min(minDistance, distancePointToSegment(nx, ny, 0.0f, 0.0f, ex, ey));

        for (int level = 0; level < branchLevels; ++level) {
            const float t = 0.34f + static_cast<float>(level) * (0.44f / std::max(1, branchLevels - 1));
            const float mx = cs * mainLength * t;
            const float my = sn * mainLength * t;
            const float branchLength = outerRadius * (0.16f + 0.03f * static_cast<float>(level));
            for (int side = -1; side <= 1; side += 2) {
                const float branchTheta = angle + static_cast<float>(side) * branchAngle;
                const float bx = mx + std::cos(branchTheta) * branchLength;
                const float by = my + std::sin(branchTheta) * branchLength;
                minDistance = std::min(minDistance, distancePointToSegment(nx, ny, mx, my, bx, by));
            }
        }
    }

    return minDistance <= baseThickness;
}

bool spiralMaskOpen(float nx,
                    float ny,
                    int bladeCount,
                    float roundness,
                    float rotationRad,
                    float outerRadius) {
    const float radius = std::sqrt(nx * nx + ny * ny);
    if (radius > outerRadius) {
        return false;
    }

    const float radialNorm = radius / std::max(outerRadius, 1e-6f);
    const float angle = std::atan2(ny, nx) - rotationRad;
    const float twist = 4.0f + (1.0f - clampValue(roundness, 0.0f, 1.0f)) * 8.0f;
    const float opening = 0.18f + clampValue(roundness, 0.0f, 1.0f) * 0.26f;
    const float phase = static_cast<float>(std::max(3, bladeCount)) * angle + twist * radialNorm * (2.0f * kPi);
    const float band = 0.5f + 0.5f * std::cos(phase);
    return band >= (1.0f - opening);
}

float apodizationWeight(LensDiffApodizationMode mode, float radialNorm) {
    switch (mode) {
        case LensDiffApodizationMode::Cosine:
            return std::cos(std::min(radialNorm, 1.0f) * (kPi * 0.5f));
        case LensDiffApodizationMode::Gaussian:
            return std::exp(-4.0f * radialNorm * radialNorm);
        case LensDiffApodizationMode::Flat:
        default:
            return 1.0f;
    }
}

float sampleImageBilinear(const LensDiffApertureImage& image, float x, float y) {
    if (image.width <= 0 || image.height <= 0 || image.values.empty()) {
        return 0.0f;
    }
    if (x < 0.0f || y < 0.0f || x > static_cast<float>(image.width - 1) || y > static_cast<float>(image.height - 1)) {
        return 0.0f;
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image.width - 1);
    const int y1 = std::min(y0 + 1, image.height - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);

    const float v00 = image.values[static_cast<std::size_t>(y0) * image.width + x0];
    const float v10 = image.values[static_cast<std::size_t>(y0) * image.width + x1];
    const float v01 = image.values[static_cast<std::size_t>(y1) * image.width + x0];
    const float v11 = image.values[static_cast<std::size_t>(y1) * image.width + x1];

    const float vx0 = v00 + (v10 - v00) * tx;
    const float vx1 = v01 + (v11 - v01) * tx;
    return vx0 + (vx1 - vx0) * ty;
}

std::vector<float> buildCustomPupilAmplitude(const LensDiffParams& params, int size) {
    LensDiffApertureImage image;
    std::string error;
    if (!LoadLensDiffPreparedApertureImage(params.customAperturePath,
                                           params.customApertureNormalize,
                                           params.customApertureInvert,
                                           &image,
                                           &error)) {
        return std::vector<float>(static_cast<std::size_t>(size) * size, 0.0f);
    }

    std::vector<float> pupil(static_cast<std::size_t>(size) * size, 0.0f);
    const float outerRadius = 0.86f;
    const float rotationRad = static_cast<float>(params.rotationDeg * kPi / 180.0);
    const float cs = std::cos(-rotationRad);
    const float sn = std::sin(-rotationRad);
    const float imageAspect = static_cast<float>(image.width) / static_cast<float>(std::max(1, image.height));
    const float fitHalfWidth = imageAspect >= 1.0f ? 1.0f : imageAspect;
    const float fitHalfHeight = imageAspect >= 1.0f ? 1.0f / imageAspect : 1.0f;

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const float nx = ((static_cast<float>(x) + 0.5f) / static_cast<float>(size) - 0.5f) * 2.0f;
            const float ny = ((static_cast<float>(y) + 0.5f) / static_cast<float>(size) - 0.5f) * 2.0f;
            const float dx = nx - static_cast<float>(params.pupilDecenterX);
            const float dy = ny - static_cast<float>(params.pupilDecenterY);
            const float radius = std::sqrt(dx * dx + dy * dy);
            if (radius > outerRadius) {
                continue;
            }

            const float rx = dx * cs - dy * sn;
            const float ry = dx * sn + dy * cs;
            const float ux = rx / outerRadius;
            const float uy = ry / outerRadius;
            if (std::abs(ux) > fitHalfWidth || std::abs(uy) > fitHalfHeight) {
                continue;
            }

            const float sx = ((ux / fitHalfWidth) * 0.5f + 0.5f) * static_cast<float>(image.width - 1);
            const float sy = ((uy / fitHalfHeight) * 0.5f + 0.5f) * static_cast<float>(image.height - 1);
            const float radialNorm = radius / outerRadius;
            const float sample = sampleImageBilinear(image, sx, sy);
            pupil[static_cast<std::size_t>(y) * size + x] =
                sample * apodizationWeight(params.apodizationMode, radialNorm);
        }
    }

    return pupil;
}

std::vector<float> buildPupilAmplitude(const LensDiffParams& params) {
    const int size = effectivePupilResolution(params.pupilResolution);
    if (params.apertureMode == LensDiffApertureMode::Custom) {
        return buildCustomPupilAmplitude(params, size);
    }

    std::vector<float> pupil(static_cast<std::size_t>(size) * size, 0.0f);

    const float outerRadius = 0.86f;
    const float obstruction = static_cast<float>(std::max(0.0, std::min(0.95, params.centralObstruction))) * outerRadius;
    const float rotationRad = static_cast<float>(params.rotationDeg * kPi / 180.0);
    const float roundness = static_cast<float>(clampValue(params.roundness, 0.0, 1.0));
    const float vaneThickness = static_cast<float>(std::max(0.0, params.vaneThickness));
    const bool useHexagon = params.apertureMode == LensDiffApertureMode::Hexagon;
    const bool useStar = params.apertureMode == LensDiffApertureMode::Star;
    const bool useSpiral = params.apertureMode == LensDiffApertureMode::Spiral;
    const bool useSquareGrid = params.apertureMode == LensDiffApertureMode::SquareGrid;
    const bool useSnowflake = params.apertureMode == LensDiffApertureMode::Snowflake;
    const float starInnerRadiusRatio = 0.18f + roundness * 0.62f;

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const float nx = ((static_cast<float>(x) + 0.5f) / static_cast<float>(size) - 0.5f) * 2.0f;
            const float ny = ((static_cast<float>(y) + 0.5f) / static_cast<float>(size) - 0.5f) * 2.0f;
            const float dx = nx - static_cast<float>(params.pupilDecenterX);
            const float dy = ny - static_cast<float>(params.pupilDecenterY);
            const float radius = std::sqrt(dx * dx + dy * dy);

            bool insideShape = false;
            if (params.apertureMode == LensDiffApertureMode::Polygon || useHexagon) {
                const int sides = useHexagon ? 6 : std::max(3, params.bladeCount);
                const float metric = polygonMetric(dx, dy, sides, roundness, rotationRad, outerRadius);
                insideShape = metric <= 1.0f;
            } else if (useSquareGrid) {
                insideShape = squareGridMaskOpen(dx, dy, std::max(3, params.bladeCount), roundness, rotationRad, outerRadius);
            } else if (useSnowflake) {
                insideShape = snowflakeMaskOpen(dx, dy, std::max(3, params.bladeCount), roundness, rotationRad, outerRadius);
            } else if (useStar) {
                insideShape = starMetric(dx,
                                         dy,
                                         std::max(3, params.bladeCount),
                                         starInnerRadiusRatio,
                                         rotationRad,
                                         outerRadius) <= 1.0f;
            } else if (useSpiral) {
                insideShape = spiralMaskOpen(dx, dy, std::max(3, params.bladeCount), roundness, rotationRad, outerRadius);
            } else {
                const float metric = polygonMetric(dx, dy, std::max(3, params.bladeCount), roundness, rotationRad, outerRadius);
                insideShape = params.apertureMode == LensDiffApertureMode::Circle ? radius <= outerRadius : metric <= 1.0f;
            }

            if (!insideShape || radius < obstruction) {
                continue;
            }
            if (blockedByVanes(dx, dy, params.vaneCount, vaneThickness, rotationRad, outerRadius)) {
                continue;
            }

            const float radialNorm = radius / outerRadius;
            pupil[static_cast<std::size_t>(y) * size + x] = apodizationWeight(params.apodizationMode, radialNorm);
        }
    }

    return pupil;
}

std::vector<float> buildPupilPhaseWaves(const LensDiffParams& params) {
    return BuildLensDiffPupilPhaseWaves(params,
                                        effectivePupilResolution(params.pupilResolution),
                                        0.86f,
                                        params.rotationDeg);
}

std::vector<float> buildShiftedRawPsf(const std::vector<float>& pupil,
                                      const std::vector<float>& phaseWaves,
                                      int pupilSize,
                                      int rawSize) {
    std::vector<Complex> spectrum(static_cast<std::size_t>(rawSize) * rawSize, Complex(0.0f, 0.0f));
    const int offset = std::max(0, (rawSize - pupilSize) / 2);
    const bool usePhase = phaseWaves.size() == pupil.size();
    for (int y = 0; y < pupilSize; ++y) {
        for (int x = 0; x < pupilSize; ++x) {
            const std::size_t srcIndex = static_cast<std::size_t>(y) * pupilSize + x;
            const float amplitude = pupil[srcIndex];
            float phaseRadians = 0.0f;
            if (usePhase && amplitude > 0.0f) {
                phaseRadians = phaseWaves[srcIndex] * 2.0f * kPi;
            }
            spectrum[static_cast<std::size_t>(y + offset) * rawSize + (x + offset)] =
                Complex(amplitude * std::cos(phaseRadians), amplitude * std::sin(phaseRadians));
        }
    }
    fft2d(spectrum, rawSize, rawSize, false);

    std::vector<float> rawPsf(static_cast<std::size_t>(rawSize) * rawSize, 0.0f);
    for (std::size_t i = 0; i < rawPsf.size(); ++i) {
        rawPsf[i] = std::norm(spectrum[i]);
    }
    rawPsf = fftShiftSquare(rawPsf, rawSize);
    normalizeKernel(rawPsf);
    return rawPsf;
}

std::vector<float> buildShiftedRawPsf(const std::vector<float>& pupil, int pupilSize, int rawSize) {
    return buildShiftedRawPsf(pupil, std::vector<float>{}, pupilSize, rawSize);
}

std::vector<float> radialMeanProfile(const std::vector<float>& image, int size) {
    const int radiusMax = size / 2;
    std::vector<float> sums(static_cast<std::size_t>(radiusMax + 1), 0.0f);
    std::vector<int> counts(static_cast<std::size_t>(radiusMax + 1), 0);
    const float center = shiftedSpectrumCenterCoordinate(size);

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
        const int count = std::max(1, counts[static_cast<std::size_t>(r)]);
        profile[static_cast<std::size_t>(r)] = sums[static_cast<std::size_t>(r)] / static_cast<float>(count);
    }
    return profile;
}

float refineParabolicMinimum(float left, float center, float right) {
    const float denom = left - 2.0f * center + right;
    if (std::abs(denom) < 1e-6f) {
        return 0.0f;
    }
    return clampValue(0.5f * (left - right) / denom, -0.5f, 0.5f);
}

float estimateFirstMinimumRadius(const std::vector<float>& shiftedRawPsf, int size) {
    const std::vector<float> profile = radialMeanProfile(shiftedRawPsf, size);
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
            const float offset = refineParabolicMinimum(
                smooth[static_cast<std::size_t>(bestIndex - 1)],
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
        const float offset = refineParabolicMinimum(
            smooth[static_cast<std::size_t>(bestIndex - 1)],
            smooth[static_cast<std::size_t>(bestIndex)],
            smooth[static_cast<std::size_t>(bestIndex + 1)]);
        return std::max(1.0f, static_cast<float>(bestIndex) + offset);
    }
    return static_cast<float>(bestIndex);
}

void applySupportBoundaryTaper(LensDiffKernel& kernel, int supportRadius) {
    if (kernel.size <= 0 || kernel.values.empty() || supportRadius <= 0) {
        return;
    }

    const float center = static_cast<float>(kernel.size - 1) * 0.5f;
    const float extent = std::max(1.0f, static_cast<float>(supportRadius));
    const float fadeWidth = std::max(6.0f, std::min(24.0f, extent * 0.04f));
    const float fadeStart = std::max(0.0f, extent - fadeWidth);
    for (int y = 0; y < kernel.size; ++y) {
        for (int x = 0; x < kernel.size; ++x) {
            float weight = 1.0f;
            const float dx = static_cast<float>(x) - center;
            const float dy = static_cast<float>(y) - center;
            const float radius = std::sqrt(dx * dx + dy * dy);

            if (radius >= extent) {
                weight = 0.0f;
            } else {
                const float t = clampValue((radius - fadeStart) / std::max(fadeWidth, 1e-6f), 0.0f, 1.0f);
                weight = std::max(0.0f, std::cos(smoothStep01(t) * static_cast<float>(kPi * 0.5)));
            }
            kernel.values[static_cast<std::size_t>(y) * kernel.size + x] *= weight;
        }
    }
}

int estimateAdaptiveSupportRadius(const LensDiffKernel& kernel, int maxRadius) {
    if (kernel.size <= 0 || kernel.values.empty()) {
        return std::max(4, maxRadius);
    }

    const int radiusMax = std::min(maxRadius, kernel.size / 2);
    const float center = static_cast<float>(kernel.size - 1) * 0.5f;
    std::vector<float> ringEnergy(static_cast<std::size_t>(radiusMax + 1), 0.0f);
    std::vector<float> ringPeak(static_cast<std::size_t>(radiusMax + 1), 0.0f);
    float totalEnergy = 0.0f;
    float globalPeak = 0.0f;

    for (int y = 0; y < kernel.size; ++y) {
        for (int x = 0; x < kernel.size; ++x) {
            const float value = kernel.values[static_cast<std::size_t>(y) * kernel.size + x];
            const float dx = static_cast<float>(x) - center;
            const float dy = static_cast<float>(y) - center;
            const int r = std::min(radiusMax, static_cast<int>(std::ceil(std::sqrt(dx * dx + dy * dy))));
            ringEnergy[static_cast<std::size_t>(r)] += value;
            ringPeak[static_cast<std::size_t>(r)] = std::max(ringPeak[static_cast<std::size_t>(r)], value);
            totalEnergy += value;
            globalPeak = std::max(globalPeak, value);
        }
    }

    if (totalEnergy <= 1e-6f || globalPeak <= 1e-6f) {
        return std::max(4, radiusMax);
    }

    std::vector<float> outsidePeak(static_cast<std::size_t>(radiusMax + 2), 0.0f);
    for (int r = radiusMax; r >= 0; --r) {
        outsidePeak[static_cast<std::size_t>(r)] = std::max(
            ringPeak[static_cast<std::size_t>(r)],
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

int paddedAdaptiveSupportRadius(int estimatedRadius, int maxRadius) {
    const int padding = std::max(6, std::min(24, static_cast<int>(std::ceil(std::max(1, estimatedRadius) * 0.05f))));
    return clampValue(estimatedRadius + padding, 4, std::max(4, maxRadius));
}

LensDiffKernel cropKernelToRadius(const LensDiffKernel& kernel, int radius) {
    const int clampedRadius = clampValue(radius, 1, kernel.size / 2);
    const int newSize = clampedRadius * 2 + 1;
    const int srcCenter = kernel.size / 2;
    LensDiffKernel out {};
    out.size = newSize;
    out.values.assign(static_cast<std::size_t>(newSize) * newSize, 0.0f);

    for (int y = 0; y < newSize; ++y) {
        for (int x = 0; x < newSize; ++x) {
            const int sx = srcCenter - clampedRadius + x;
            const int sy = srcCenter - clampedRadius + y;
            out.values[static_cast<std::size_t>(y) * newSize + x] =
                kernel.values[static_cast<std::size_t>(sy) * kernel.size + sx];
        }
    }

    applySupportBoundaryTaper(out, clampedRadius);
    normalizeKernel(out.values);
    return out;
}

LensDiffKernel makeResampledKernel(const std::vector<float>& shiftedRawPsf,
                                   int rawSize,
                                   float scaleFactor,
                                   int supportRadius) {
    const int kernelSize = supportRadius * 2 + 1;
    LensDiffKernel kernel {};
    kernel.size = kernelSize;
    kernel.values.assign(static_cast<std::size_t>(kernelSize) * kernelSize, 0.0f);

    const float rawCenter = shiftedSpectrumCenterCoordinate(rawSize);
    const float kernelCenter = static_cast<float>(kernelSize - 1) * 0.5f;
    const float invScale = 1.0f / std::max(scaleFactor, 0.05f);

    for (int y = 0; y < kernelSize; ++y) {
        for (int x = 0; x < kernelSize; ++x) {
            const float dx = static_cast<float>(x) - kernelCenter;
            const float dy = static_cast<float>(y) - kernelCenter;
            const float sx = rawCenter + dx * invScale;
            const float sy = rawCenter + dy * invScale;
            kernel.values[static_cast<std::size_t>(y) * kernelSize + x] =
                sampleSquareFiltered(shiftedRawPsf, rawSize, sx, sy, invScale);
        }
    }

    normalizeKernel(kernel.values);
    return kernel;
}

LensDiffKernel azimuthalMeanKernel(const LensDiffKernel& kernel) {
    LensDiffKernel meanKernel {};
    meanKernel.size = kernel.size;
    meanKernel.values.assign(kernel.values.size(), 0.0f);

    const int radiusMax = kernel.size / 2;
    std::vector<float> sums(static_cast<std::size_t>(radiusMax + 1), 0.0f);
    std::vector<int> counts(static_cast<std::size_t>(radiusMax + 1), 0);
    const float center = static_cast<float>(kernel.size - 1) * 0.5f;

    for (int y = 0; y < kernel.size; ++y) {
        for (int x = 0; x < kernel.size; ++x) {
            const float dx = static_cast<float>(x) - center;
            const float dy = static_cast<float>(y) - center;
            const int r = std::min(radiusMax, static_cast<int>(std::round(std::sqrt(dx * dx + dy * dy))));
            sums[static_cast<std::size_t>(r)] += kernel.values[static_cast<std::size_t>(y) * kernel.size + x];
            counts[static_cast<std::size_t>(r)] += 1;
        }
    }

    for (int y = 0; y < kernel.size; ++y) {
        for (int x = 0; x < kernel.size; ++x) {
            const float dx = static_cast<float>(x) - center;
            const float dy = static_cast<float>(y) - center;
            const int r = std::min(radiusMax, static_cast<int>(std::round(std::sqrt(dx * dx + dy * dy))));
            const int count = std::max(1, counts[static_cast<std::size_t>(r)]);
            meanKernel.values[static_cast<std::size_t>(y) * kernel.size + x] = sums[static_cast<std::size_t>(r)] / static_cast<float>(count);
        }
    }

    normalizeKernel(meanKernel.values);
    return meanKernel;
}

LensDiffKernel reshapeAnisotropy(const LensDiffKernel& original, const LensDiffKernel& meanKernel, float emphasis) {
    LensDiffKernel shaped {};
    shaped.size = original.size;
    shaped.values.resize(original.values.size(), 0.0f);

    const float gain = 1.0f + std::max(0.0f, emphasis) * 4.0f;
    for (std::size_t i = 0; i < original.values.size(); ++i) {
        const float meanValue = meanKernel.values[i];
        const float residual = original.values[i] - meanValue;
        shaped.values[i] = std::max(0.0f, meanValue + residual * gain);
    }
    normalizeKernel(shaped.values);
    return shaped;
}

LensDiffKernel positiveResidualKernel(const LensDiffKernel& fullKernel, const LensDiffKernel& meanKernel) {
    LensDiffKernel residual {};
    residual.size = fullKernel.size;
    residual.values.resize(fullKernel.values.size(), 0.0f);

    for (std::size_t i = 0; i < fullKernel.values.size(); ++i) {
        residual.values[i] = std::max(0.0f, fullKernel.values[i] - meanKernel.values[i]);
    }

    const float total = sumKernel(residual.values);
    if (total > 0.0f) {
        normalizeKernel(residual.values);
    }
    return residual;
}

std::vector<Complex> paddedSpectrumFromKernel(const LensDiffKernel& kernel, int width, int height) {
    std::vector<Complex> data(static_cast<std::size_t>(width) * height, Complex(0.0f, 0.0f));
    const int center = kernel.size / 2;
    for (int y = 0; y < kernel.size; ++y) {
        for (int x = 0; x < kernel.size; ++x) {
            const int px = (x - center + width) % width;
            const int py = (y - center + height) % height;
            data[static_cast<std::size_t>(py) * width + px] = Complex(kernel.values[static_cast<std::size_t>(y) * kernel.size + x], 0.0f);
        }
    }
    fft2d(data, width, height, false);
    return data;
}

ScalarImage convolveScalar(const ScalarImage& src, const LensDiffKernel& kernel) {
    const int paddedWidth = nextPowerOfTwo(src.width + kernel.size - 1);
    const int paddedHeight = nextPowerOfTwo(src.height + kernel.size - 1);

    std::vector<Complex> imageSpectrum(static_cast<std::size_t>(paddedWidth) * paddedHeight, Complex(0.0f, 0.0f));
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            imageSpectrum[static_cast<std::size_t>(y) * paddedWidth + x] = Complex(src.at(x, y), 0.0f);
        }
    }

    fft2d(imageSpectrum, paddedWidth, paddedHeight, false);
    std::vector<Complex> kernelSpectrum = paddedSpectrumFromKernel(kernel, paddedWidth, paddedHeight);

    for (std::size_t i = 0; i < imageSpectrum.size(); ++i) {
        imageSpectrum[i] *= kernelSpectrum[i];
    }
    fft2d(imageSpectrum, paddedWidth, paddedHeight, true);

    ScalarImage out(src.width, src.height);
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            out.at(x, y) = std::max(0.0f, imageSpectrum[static_cast<std::size_t>(y) * paddedWidth + x].real());
        }
    }
    return out;
}

RgbaImage convolveRgba(const RgbaImage& src, const LensDiffKernel& kernel) {
    RgbaImage out(src.width, src.height);
    for (int c = 0; c < 3; ++c) {
        ScalarImage channel(src.width, src.height);
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                channel.at(x, y) = src.pixel(x, y)[c];
            }
        }
        const ScalarImage conv = convolveScalar(channel, kernel);
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                out.pixel(x, y)[c] = conv.at(x, y);
            }
        }
    }
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            out.pixel(x, y)[3] = src.pixel(x, y)[3];
        }
    }
    return out;
}

ScalarImage buildSelectionMask(const RgbaImage& src, const LensDiffParams& params) {
    ScalarImage mask(src.width, src.height);
    const float softness = static_cast<float>(std::max(0.01, params.softnessStops));
    const float thresholdStops = static_cast<float>(params.threshold);
    const float thresholdLinear = kGray18 * std::pow(2.0f, thresholdStops);

    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            const float* pixel = src.pixel(x, y);
            const float maxRgb = std::max({pixel[0], pixel[1], pixel[2]});
            const float signal = params.extractionMode == LensDiffExtractionMode::Luma
                ? safeLuma(pixel[0], pixel[1], pixel[2])
                : maxRgb;

            float gate = 0.0f;
            if (signal > 0.0f) {
                const float stops = std::log2(std::max(signal, 1e-6f) / kGray18);
                const float edge0 = thresholdStops - softness * 0.5f;
                const float edge1 = thresholdStops + softness * 0.5f;
                const float t = saturate((stops - edge0) / std::max(edge1 - edge0, 1e-4f));
                gate = t * t * (3.0f - 2.0f * t);
            }

            const float pointBoost = 1.0f + static_cast<float>(params.pointEmphasis) *
                                              std::max(0.0f, maxRgb / std::max(thresholdLinear, 1e-4f) - 1.0f);
            mask.at(x, y) = saturate(gate * pointBoost);
        }
    }

    return mask;
}

RgbaImage applyMask(const RgbaImage& src, const ScalarImage& mask, float scale) {
    RgbaImage out(src.width, src.height);
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            const float m = mask.at(x, y) * scale;
            const float* srcPixel = src.pixel(x, y);
            float* dstPixel = out.pixel(x, y);
            dstPixel[0] = srcPixel[0] * m;
            dstPixel[1] = srcPixel[1] * m;
            dstPixel[2] = srcPixel[2] * m;
            dstPixel[3] = srcPixel[3];
        }
    }
    return out;
}

ScalarImage applyMaskScalarDriver(const RgbaImage& src, const ScalarImage& mask, LensDiffExtractionMode mode, float scale) {
    ScalarImage out(src.width, src.height);
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            const float* pixel = src.pixel(x, y);
            const float signal = mode == LensDiffExtractionMode::Luma
                ? safeLuma(pixel[0], pixel[1], pixel[2])
                : std::max({pixel[0], pixel[1], pixel[2]});
            out.at(x, y) = signal * mask.at(x, y) * scale;
        }
    }
    return out;
}

float luminanceSum(const RgbaImage& image) {
    float sum = 0.0f;
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const float* pixel = image.pixel(x, y);
            sum += safeLuma(pixel[0], pixel[1], pixel[2]);
        }
    }
    return sum;
}

void scaleImage(RgbaImage& image, float scale) {
    for (float& value : image.pixels) {
        value *= scale;
    }
}

RgbaImage buildSpectralEffect(const ScalarImage& driver,
                              const std::vector<LensDiffPsfBin>& bins,
                              const LensDiffSpectrumConfig& spectrumConfig,
                              const LensDiffParams& params,
                              bool useCore,
                              bool useStructure) {
    std::vector<ScalarImage> convolved;
    convolved.reserve(bins.size());

    for (const auto& bin : bins) {
        const LensDiffKernel* kernel = &bin.full;
        if (useCore && !useStructure) {
            kernel = &bin.core;
        } else if (!useCore && useStructure) {
            kernel = &bin.structure;
        }
        convolved.push_back(convolveScalar(driver, *kernel));
    }

    RgbaImage out(driver.width, driver.height);
    for (int y = 0; y < driver.height; ++y) {
        for (int x = 0; x < driver.width; ++x) {
            std::array<float, kLensDiffMaxSpectralBins> values {};
            if (convolved.size() == 1U) {
                const float v = convolved[0].at(x, y);
                values[0] = v;
            } else {
                for (std::size_t i = 0; i < convolved.size() && i < values.size(); ++i) {
                    values[i] = convolved[i].at(x, y);
                }
            }
            const auto rgb = MapLensDiffSpectralBins(values, params, spectrumConfig);
            float* pixel = out.pixel(x, y);
            pixel[0] = rgb[0];
            pixel[1] = rgb[1];
            pixel[2] = rgb[2];
            pixel[3] = 1.0f;
        }
    }
    return out;
}

void applyShoulder(RgbaImage& image, float shoulder) {
    if (shoulder <= 0.0f) {
        return;
    }
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            float* pixel = image.pixel(x, y);
            pixel[0] = softShoulder(pixel[0], shoulder);
            pixel[1] = softShoulder(pixel[1], shoulder);
            pixel[2] = softShoulder(pixel[2], shoulder);
        }
    }
}

RgbaImage addImages(const RgbaImage& a, const RgbaImage& b, float aGain, float bGain) {
    RgbaImage out(a.width, a.height);
    for (int y = 0; y < a.height; ++y) {
        for (int x = 0; x < a.width; ++x) {
            const float* ap = a.pixel(x, y);
            const float* bp = b.pixel(x, y);
            float* op = out.pixel(x, y);
            op[0] = ap[0] * aGain + bp[0] * bGain;
            op[1] = ap[1] * aGain + bp[1] * bGain;
            op[2] = ap[2] * aGain + bp[2] * bGain;
            op[3] = ap[3];
        }
    }
    return out;
}

RgbaImage makeDebugFromScalar(const ScalarImage& scalar) {
    RgbaImage out(scalar.width, scalar.height);
    float maxValue = 0.0f;
    for (float value : scalar.pixels) {
        maxValue = std::max(maxValue, value);
    }
    const float invMax = maxValue > 0.0f ? 1.0f / maxValue : 0.0f;

    for (int y = 0; y < scalar.height; ++y) {
        for (int x = 0; x < scalar.width; ++x) {
            const float d = saturate(scalar.at(x, y) * invMax);
            float* pixel = out.pixel(x, y);
            pixel[0] = d;
            pixel[1] = d;
            pixel[2] = d;
            pixel[3] = 1.0f;
        }
    }
    return out;
}

ScalarImage makeScalarFromKernel(const LensDiffKernel& kernel) {
    ScalarImage out(kernel.size, kernel.size);
    out.pixels = kernel.values;
    return out;
}

ScalarImage makeScalarFromVector(const std::vector<float>& values, int size) {
    ScalarImage out(size, size);
    out.pixels = values;
    return out;
}

ScalarImage computeOtfMagnitude(const LensDiffKernel& kernel) {
    const int size = std::max(1, kernel.size);
    std::vector<Complex> spectrum = paddedSpectrumFromKernel(kernel, size, size);
    ScalarImage mag(size, size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            mag.at(x, y) = std::abs(spectrum[static_cast<std::size_t>(y) * size + x]);
        }
    }
    mag.pixels = fftShiftSquare(mag.pixels, size);
    return mag;
}

RgbaImage visualizeSquareCentered(const ScalarImage& square, int outWidth, int outHeight, bool logView) {
    RgbaImage out(outWidth, outHeight);
    const int squareSize = std::min(outWidth, outHeight);
    const int xOffset = (outWidth - squareSize) / 2;
    const int yOffset = (outHeight - squareSize) / 2;
    float maxValue = 0.0f;
    for (float value : square.pixels) {
        maxValue = std::max(maxValue, value);
    }
    const float logMax = std::log1p(std::max(0.0f, maxValue) * 1024.0f);

    for (int y = 0; y < outHeight; ++y) {
        for (int x = 0; x < outWidth; ++x) {
            float d = 0.0f;
            if (x >= xOffset && x < xOffset + squareSize && y >= yOffset && y < yOffset + squareSize) {
                const float nx = static_cast<float>(x - xOffset) / std::max(1, squareSize - 1);
                const float ny = static_cast<float>(y - yOffset) / std::max(1, squareSize - 1);
                const float sx = nx * static_cast<float>(square.width - 1);
                const float sy = ny * static_cast<float>(square.height - 1);
                float v = sampleSquareBilinear(square.pixels, square.width, sx, sy);
                if (logView) {
                    d = logMax > 0.0f ? std::log1p(std::max(0.0f, v) * 1024.0f) / logMax : 0.0f;
                } else {
                    d = maxValue > 0.0f ? v / maxValue : 0.0f;
                }
            }
            float* pixel = out.pixel(x, y);
            pixel[0] = d;
            pixel[1] = d;
            pixel[2] = d;
            pixel[3] = 1.0f;
        }
    }
    return out;
}

void accumulateWeightedImage(const RgbaImage& src, RgbaImage& dst, int zoneX, int zoneY, float gain = 1.0f) {
    if (src.width != dst.width || src.height != dst.height) {
        return;
    }
    const float denomX = std::max(1, src.width - 1);
    const float denomY = std::max(1, src.height - 1);
    for (int y = 0; y < src.height; ++y) {
        const float py = (static_cast<float>(y) / static_cast<float>(denomY)) * 2.0f;
        const float wy = std::max(0.0f, 1.0f - std::abs(py - static_cast<float>(zoneY)));
        if (wy <= 0.0f) continue;
        for (int x = 0; x < src.width; ++x) {
            const float px = (static_cast<float>(x) / static_cast<float>(denomX)) * 2.0f;
            const float wx = std::max(0.0f, 1.0f - std::abs(px - static_cast<float>(zoneX)));
            const float weight = wx * wy * gain;
            if (weight <= 0.0f) continue;
            const float* sp = src.pixel(x, y);
            float* dp = dst.pixel(x, y);
            dp[0] += sp[0] * weight;
            dp[1] += sp[1] * weight;
            dp[2] += sp[2] * weight;
            dp[3] = sp[3];
        }
    }
}

std::array<float, 4> sampleRgbaBilinear(const RgbaImage& image, float fx, float fy) {
    const float x = clampValue(fx, 0.0f, static_cast<float>(std::max(0, image.width - 1)));
    const float y = clampValue(fy, 0.0f, static_cast<float>(std::max(0, image.height - 1)));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image.width - 1);
    const int y1 = std::min(y0 + 1, image.height - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    std::array<float, 4> out {};
    const float* p00 = image.pixel(x0, y0);
    const float* p10 = image.pixel(x1, y0);
    const float* p01 = image.pixel(x0, y1);
    const float* p11 = image.pixel(x1, y1);
    for (int c = 0; c < 4; ++c) {
        const float a = p00[c] * (1.0f - tx) + p10[c] * tx;
        const float b = p01[c] * (1.0f - tx) + p11[c] * tx;
        out[static_cast<std::size_t>(c)] = a * (1.0f - ty) + b * ty;
    }
    return out;
}

RgbaImage applyCreativeFringe(const RgbaImage& image, float fringeAmount, RgbaImage* preview) {
    if (fringeAmount <= 1e-6f) {
        if (preview) *preview = RgbaImage(image.width, image.height);
        return image;
    }
    RgbaImage out(image.width, image.height);
    RgbaImage localPreview(image.width, image.height);
    const float cx = static_cast<float>(std::max(0, image.width - 1)) * 0.5f;
    const float cy = static_cast<float>(std::max(0, image.height - 1)) * 0.5f;
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const float dx = static_cast<float>(x) - cx;
            const float dy = static_cast<float>(y) - cy;
            const float length = std::sqrt(dx * dx + dy * dy);
            const float invLength = length > 1e-6f ? 1.0f / length : 0.0f;
            const float shiftX = dx * invLength * fringeAmount;
            const float shiftY = dy * invLength * fringeAmount;
            const auto red = sampleRgbaBilinear(image, static_cast<float>(x) + shiftX, static_cast<float>(y) + shiftY);
            const auto green = sampleRgbaBilinear(image, static_cast<float>(x), static_cast<float>(y));
            const auto blue = sampleRgbaBilinear(image, static_cast<float>(x) - shiftX, static_cast<float>(y) - shiftY);
            const float* src = image.pixel(x, y);
            float* dst = out.pixel(x, y);
            float* pv = localPreview.pixel(x, y);
            dst[0] = red[0];
            dst[1] = green[1];
            dst[2] = blue[2];
            dst[3] = src[3];
            pv[0] = std::abs(dst[0] - src[0]);
            pv[1] = std::abs(dst[1] - src[1]);
            pv[2] = std::abs(dst[2] - src[2]);
            pv[3] = 1.0f;
        }
    }
    if (preview) *preview = std::move(localPreview);
    return out;
}

RgbaImage visualizeSignedPhaseCentered(const ScalarImage& phase,
                                       const ScalarImage& pupilMask,
                                       int outWidth,
                                       int outHeight) {
    RgbaImage out(outWidth, outHeight);
    const int squareSize = std::min(outWidth, outHeight);
    const int xOffset = (outWidth - squareSize) / 2;
    const int yOffset = (outHeight - squareSize) / 2;

    float maxAbsValue = 0.0f;
    for (float value : phase.pixels) {
        maxAbsValue = std::max(maxAbsValue, std::abs(value));
    }

    for (int y = 0; y < outHeight; ++y) {
        for (int x = 0; x < outWidth; ++x) {
            float* pixel = out.pixel(x, y);
            pixel[0] = 0.0f;
            pixel[1] = 0.0f;
            pixel[2] = 0.0f;
            pixel[3] = 1.0f;

            if (x < xOffset || x >= xOffset + squareSize || y < yOffset || y >= yOffset + squareSize) {
                continue;
            }

            const float nx = static_cast<float>(x - xOffset) / std::max(1, squareSize - 1);
            const float ny = static_cast<float>(y - yOffset) / std::max(1, squareSize - 1);
            const float sx = nx * static_cast<float>(phase.width - 1);
            const float sy = ny * static_cast<float>(phase.height - 1);
            const float pupilValue = sampleSquareBilinear(pupilMask.pixels, pupilMask.width, sx, sy);
            if (pupilValue <= 1e-4f) {
                continue;
            }

            const float phaseValue = sampleSquareBilinear(phase.pixels, phase.width, sx, sy);
            const float normalized = maxAbsValue > 0.0f ? clampValue(phaseValue / maxAbsValue, -1.0f, 1.0f) : 0.0f;
            const float magnitude = std::abs(normalized);
            const float neutral = 0.18f;
            if (magnitude <= 1e-6f) {
                pixel[0] = neutral;
                pixel[1] = neutral;
                pixel[2] = neutral;
                continue;
            }

            if (normalized > 0.0f) {
                pixel[0] = neutral + 0.82f * magnitude;
                pixel[1] = neutral + 0.28f * magnitude;
                pixel[2] = std::max(0.0f, neutral - 0.12f * magnitude);
            } else {
                pixel[0] = std::max(0.0f, neutral - 0.10f * magnitude);
                pixel[1] = neutral + 0.34f * magnitude;
                pixel[2] = neutral + 0.80f * magnitude;
            }
        }
    }
    return out;
}

RgbaImage makeRgbaFromPacked(const std::vector<float>& rgba, int width, int height) {
    RgbaImage out(width, height);
    out.pixels = rgba;
    return out;
}

std::vector<float> wavelengthsForMode(LensDiffSpectralMode mode) {
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

void populatePsfBankCacheFromBaseKernels(const LensDiffParams& params,
                                         const LensDiffPsfBankKey& key,
                                         const std::vector<float>& pupil,
                                         int pupilSize,
                                         const std::vector<float>& wavelengths,
                                         std::vector<LensDiffKernel> baseKernels,
                                         LensDiffPsfBankCache& cache) {
    cache = {};
    cache.valid = true;
    cache.key = key;
    cache.supportRadiusPx = 4;
    cache.pupilDisplay = pupil;
    cache.pupilDisplaySize = pupilSize;
    cache.phaseDisplay = BuildLensDiffPupilPhaseForPsf(params);
    cache.phaseDisplaySize = pupilSize;
    cache.fieldGridSize = 0;
    cache.fieldZones.clear();
    for (std::size_t i = 0; i < wavelengths.size() && i < baseKernels.size(); ++i) {
        const float wavelength = wavelengths[i];
        LensDiffKernel baseKernel = std::move(baseKernels[i]);
        LensDiffKernel meanKernel = azimuthalMeanKernel(baseKernel);
        LensDiffKernel shapedKernel = reshapeAnisotropy(baseKernel, meanKernel, static_cast<float>(params.anisotropyEmphasis));
        LensDiffKernel structureKernel = positiveResidualKernel(shapedKernel, meanKernel);
        const int maxKernelRadiusPx = std::max(4, ResolveLensDiffMaxKernelRadiusPx(params));
        const int effectiveRadius = paddedAdaptiveSupportRadius(
            estimateAdaptiveSupportRadius(shapedKernel, maxKernelRadiusPx),
            maxKernelRadiusPx);

        meanKernel = cropKernelToRadius(meanKernel, effectiveRadius);
        shapedKernel = cropKernelToRadius(shapedKernel, effectiveRadius);
        structureKernel = cropKernelToRadius(structureKernel, effectiveRadius);
        cache.supportRadiusPx = std::max(cache.supportRadiusPx, effectiveRadius);

        LensDiffPsfBin bin {};
        bin.wavelengthNm = wavelength;
        bin.full = shapedKernel;
        bin.core = meanKernel;
        bin.structure = structureKernel;
        cache.bins.push_back(std::move(bin));
    }
}

void populatePsfBankCacheFromRawPsf(const LensDiffParams& params,
                                    const LensDiffPsfBankKey& key,
                                    const std::vector<float>& pupil,
                                    int pupilSize,
                                    const std::vector<float>& rawPsf,
                                    int rawPsfSize,
                                    const std::vector<float>& referenceRawPsf,
                                    LensDiffPsfBankCache& cache) {
    const float referenceFirstZeroRadius = estimateFirstMinimumRadius(referenceRawPsf, rawPsfSize);
    const float scaleBase = static_cast<float>(std::max(1.0, ResolveLensDiffDiffractionScalePx(params))) /
                            std::max(1.0f, referenceFirstZeroRadius);
    const int maxSupportRadius = std::max(4, ResolveLensDiffMaxKernelRadiusPx(params));
    const std::vector<float> wavelengths = wavelengthsForMode(params.spectralMode);
    std::vector<LensDiffKernel> baseKernels;
    baseKernels.reserve(wavelengths.size());
    for (float wavelength : wavelengths) {
        const float scaleFactor = scaleBase * (wavelength / 550.0f);
        baseKernels.push_back(makeResampledKernel(rawPsf, rawPsfSize, scaleFactor, maxSupportRadius));
    }
    populatePsfBankCacheFromBaseKernels(params, key, pupil, pupilSize, wavelengths, std::move(baseKernels), cache);
}

LensDiffKernel buildGaussianKernel(float radiusPx) {
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
        for (float& value : kernel.values) {
            value /= sum;
        }
    }
    return kernel;
}

void buildPsfBankGlobalOnly(const LensDiffParams& params, LensDiffPsfBankCache& cache) {
    const LensDiffPsfBankKey key = MakeLensDiffPsfBankKey(params);
    if (cache.valid && cache.key == key) {
        return;
    }
    LensDiffScopedTimer timer("cpu-psf-bank-global");

    const int pupilSize = GetLensDiffEffectivePupilResolution(params.pupilResolution);
    const int maxKernelRadiusPx = std::max(4, ResolveLensDiffMaxKernelRadiusPx(params));
    const int rawPsfSize = ChooseLensDiffRawPsfSize(pupilSize, maxKernelRadiusPx);
    const std::vector<float> pupil = BuildLensDiffPupilAmplitudeForPsf(params);
    const std::shared_ptr<const std::vector<float>> referenceRawPsf = GetLensDiffReferenceRawPsfCached(pupilSize, rawPsfSize);
    const float referenceFirstZeroRadius = std::max(1.0f, estimateFirstMinimumRadius(*referenceRawPsf, rawPsfSize));
    const float scaleBase = static_cast<float>(std::max(1.0, ResolveLensDiffDiffractionScalePx(params))) / referenceFirstZeroRadius;

    const std::vector<float> wavelengths = wavelengthsForMode(params.spectralMode);
    std::vector<LensDiffKernel> baseKernels;
    baseKernels.reserve(wavelengths.size());
    for (float wavelength : wavelengths) {
        LensDiffParams wavelengthParams = params;
        const float wavelengthOffset = (wavelength - 550.0f) / 110.0f;
        wavelengthParams.phaseDefocus += params.chromaticFocus * wavelengthOffset;
        if (std::abs(wavelengthParams.phaseDefocus) > 1e-6) {
            wavelengthParams.phaseMode = LensDiffPhaseMode::Enabled;
        }
        const std::vector<float> phaseWaves = BuildLensDiffPupilPhaseForPsf(wavelengthParams);
        const std::vector<float> rawPsf = buildShiftedRawPsf(pupil, phaseWaves, pupilSize, rawPsfSize);
        const float chromaticScale = std::max(0.25f, 1.0f + static_cast<float>(params.chromaticSpread) * wavelengthOffset);
        const float scaleFactor = scaleBase * (wavelength / 550.0f) * chromaticScale;
        baseKernels.push_back(makeResampledKernel(rawPsf, rawPsfSize, scaleFactor, maxKernelRadiusPx));
    }
    populatePsfBankCacheFromBaseKernels(params, key, pupil, pupilSize, wavelengths, std::move(baseKernels), cache);
}

LensDiffFieldZoneCache makeFieldZoneCache(const LensDiffParams& params, int zoneX, int zoneY) {
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
    buildPsfBankGlobalOnly(zone.resolvedParams, zoneCache);
    zone.key = zoneCache.key;
    zone.bins = std::move(zoneCache.bins);
    zone.supportRadiusPx = zoneCache.supportRadiusPx;
    zone.pupilDisplaySize = zoneCache.pupilDisplaySize;
    zone.pupilDisplay = std::move(zoneCache.pupilDisplay);
    zone.phaseDisplaySize = zoneCache.phaseDisplaySize;
    zone.phaseDisplay = std::move(zoneCache.phaseDisplay);
    return zone;
}

void buildPsfBankInternal(const LensDiffParams& params, LensDiffPsfBankCache& cache) {
    const LensDiffPsfBankKey key = MakeLensDiffPsfBankKey(params);
    const bool needFieldZones = HasLensDiffFieldPhase(params);
    const LensDiffFieldKey fieldKey = MakeLensDiffFieldKey(params);
    const bool canReuseFieldZones = needFieldZones &&
                                    cache.fieldGridSize == 3 &&
                                    cache.fieldZones.size() == 9U &&
                                    cache.fieldKey == fieldKey;
    if (cache.valid && cache.key == key &&
        ((!needFieldZones && cache.fieldZones.empty()) || canReuseFieldZones)) {
        return;
    }
    buildPsfBankGlobalOnly(params, cache);
    if (!needFieldZones) {
        cache.fieldGridSize = 0;
        cache.fieldKey = {};
        cache.fieldZones.clear();
        return;
    }
    LensDiffScopedTimer timer("cpu-field-zones");
    cache.fieldGridSize = 3;
    cache.fieldKey = fieldKey;
    cache.fieldZones.clear();
    cache.fieldZones.reserve(9);
    for (int zoneY = 0; zoneY < 3; ++zoneY) {
        for (int zoneX = 0; zoneX < 3; ++zoneX) {
            cache.fieldZones.push_back(makeFieldZoneCache(params, zoneX, zoneY));
        }
    }
}

RgbaImage renderDebugView(const LensDiffParams& params,
                          const ScalarImage& selectionMask,
                          LensDiffPsfBankCache& cache,
                          const RgbaImage& coreEffect,
                          const RgbaImage& structureEffect,
                          const RgbaImage& effect,
                          const RgbaImage& creativeFringePreview,
                          const RgbaImage& scatterPreview,
                          int outWidth,
                          int outHeight) {
    switch (params.debugView) {
        case LensDiffDebugView::Selection:
            return makeDebugFromScalar(selectionMask);
        case LensDiffDebugView::Pupil:
        case LensDiffDebugView::Psf:
        case LensDiffDebugView::Otf:
        case LensDiffDebugView::Phase:
        case LensDiffDebugView::PhaseEdge:
        case LensDiffDebugView::FieldPsf:
        case LensDiffDebugView::ChromaticSplit:
            return makeRgbaFromPacked(GetLensDiffStaticDebugRgbaCached(params, &cache, outWidth, outHeight), outWidth, outHeight);
        case LensDiffDebugView::Core:
            return coreEffect;
        case LensDiffDebugView::Structure:
            return structureEffect;
        case LensDiffDebugView::Effect:
            return effect;
        case LensDiffDebugView::CreativeFringe:
            return creativeFringePreview;
        case LensDiffDebugView::Scatter:
            return scatterPreview;
        case LensDiffDebugView::Final:
        default:
            return RgbaImage();
    }
}

} // namespace

int GetLensDiffEffectivePupilResolution(int requested) {
    return effectivePupilResolution(requested);
}

int ChooseLensDiffRawPsfSize(int pupilSize, int maxKernelRadiusPx) {
    return chooseRawPsfSize(pupilSize, maxKernelRadiusPx);
}

LensDiffPsfBankKey MakeLensDiffPsfBankKey(const LensDiffParams& params) {
    LensDiffPsfBankKey key {};
    const bool usePhase = HasLensDiffNonFlatPhase(params);
    key.apertureMode = params.apertureMode;
    key.apodizationMode = params.apodizationMode;
    key.spectralMode = params.spectralMode;
    key.bladeCount = params.apertureMode == LensDiffApertureMode::Custom ? 0 : params.bladeCount;
    key.vaneCount = params.apertureMode == LensDiffApertureMode::Custom ? 0 : params.vaneCount;
    key.pupilResolution = effectivePupilResolution(params.pupilResolution);
    key.frameShortSidePx = ResolveLensDiffOpticsShortSidePx(params);
    key.maxKernelRadiusPx = params.maxKernelRadiusPx;
    key.customAperturePath = params.apertureMode == LensDiffApertureMode::Custom ? params.customAperturePath : std::string();
    key.customApertureNormalize = params.apertureMode == LensDiffApertureMode::Custom ? params.customApertureNormalize : true;
    key.customApertureInvert = params.apertureMode == LensDiffApertureMode::Custom ? params.customApertureInvert : false;
    key.roundness = params.apertureMode == LensDiffApertureMode::Custom ? 0.0 : params.roundness;
    key.rotationDeg = params.rotationDeg;
    key.centralObstruction = params.apertureMode == LensDiffApertureMode::Custom ? 0.0 : params.centralObstruction;
    key.vaneThickness = params.apertureMode == LensDiffApertureMode::Custom ? 0.0 : params.vaneThickness;
    key.diffractionScalePx = params.diffractionScalePx;
    key.anisotropyEmphasis = params.anisotropyEmphasis;
    key.phaseDefocus = usePhase ? params.phaseDefocus : 0.0;
    key.phaseAstigmatism0 = usePhase ? params.phaseAstigmatism0 : 0.0;
    key.phaseAstigmatism45 = usePhase ? params.phaseAstigmatism45 : 0.0;
    key.phaseComaX = usePhase ? params.phaseComaX : 0.0;
    key.phaseComaY = usePhase ? params.phaseComaY : 0.0;
    key.phaseSpherical = usePhase ? params.phaseSpherical : 0.0;
    key.phaseTrefoilX = usePhase ? params.phaseTrefoilX : 0.0;
    key.phaseTrefoilY = usePhase ? params.phaseTrefoilY : 0.0;
    key.phaseSecondaryAstigmatism0 = usePhase ? params.phaseSecondaryAstigmatism0 : 0.0;
    key.phaseSecondaryAstigmatism45 = usePhase ? params.phaseSecondaryAstigmatism45 : 0.0;
    key.phaseQuadrafoil0 = usePhase ? params.phaseQuadrafoil0 : 0.0;
    key.phaseQuadrafoil45 = usePhase ? params.phaseQuadrafoil45 : 0.0;
    key.phaseSecondaryComaX = usePhase ? params.phaseSecondaryComaX : 0.0;
    key.phaseSecondaryComaY = usePhase ? params.phaseSecondaryComaY : 0.0;
    key.pupilDecenterX = usePhase ? params.pupilDecenterX : 0.0;
    key.pupilDecenterY = usePhase ? params.pupilDecenterY : 0.0;
    key.chromaticFocus = params.chromaticFocus;
    key.chromaticSpread = params.chromaticSpread;
    return key;
}

LensDiffFieldKey MakeLensDiffFieldKey(const LensDiffParams& params) {
    LensDiffFieldKey key {};
    if (!HasLensDiffFieldPhase(params)) {
        return key;
    }
    key.phaseFieldStrength = params.phaseFieldStrength;
    key.phaseFieldEdgeBias = params.phaseFieldEdgeBias;
    key.phaseFieldDefocus = params.phaseFieldDefocus;
    key.phaseFieldAstigRadial = params.phaseFieldAstigRadial;
    key.phaseFieldAstigTangential = params.phaseFieldAstigTangential;
    key.phaseFieldComaRadial = params.phaseFieldComaRadial;
    key.phaseFieldComaTangential = params.phaseFieldComaTangential;
    key.phaseFieldSpherical = params.phaseFieldSpherical;
    key.phaseFieldTrefoilRadial = params.phaseFieldTrefoilRadial;
    key.phaseFieldTrefoilTangential = params.phaseFieldTrefoilTangential;
    key.phaseFieldSecondaryAstigRadial = params.phaseFieldSecondaryAstigRadial;
    key.phaseFieldSecondaryAstigTangential = params.phaseFieldSecondaryAstigTangential;
    key.phaseFieldQuadrafoilRadial = params.phaseFieldQuadrafoilRadial;
    key.phaseFieldQuadrafoilTangential = params.phaseFieldQuadrafoilTangential;
    key.phaseFieldSecondaryComaRadial = params.phaseFieldSecondaryComaRadial;
    key.phaseFieldSecondaryComaTangential = params.phaseFieldSecondaryComaTangential;
    return key;
}

std::vector<float> BuildLensDiffPupilAmplitudeForPsf(const LensDiffParams& params) {
    return buildPupilAmplitude(params);
}

std::vector<float> BuildLensDiffPupilPhaseForPsf(const LensDiffParams& params) {
    return buildPupilPhaseWaves(params);
}

std::vector<float> BuildLensDiffStaticDebugRgba(const LensDiffParams& params,
                                                const LensDiffPsfBankCache& cache,
                                                int outWidth,
                                                int outHeight) {
    const LensDiffFieldZoneCache* edgeZone = cache.fieldZones.empty() ? nullptr : &cache.fieldZones.back();
    switch (params.debugView) {
        case LensDiffDebugView::Pupil:
            return visualizeSquareCentered(makeScalarFromVector(cache.pupilDisplay, cache.pupilDisplaySize), outWidth, outHeight, false).pixels;
        case LensDiffDebugView::Psf:
            return visualizeSquareCentered(makeScalarFromKernel(cache.bins.front().full), outWidth, outHeight, true).pixels;
        case LensDiffDebugView::Otf:
            return visualizeSquareCentered(computeOtfMagnitude(cache.bins.front().full), outWidth, outHeight, true).pixels;
        case LensDiffDebugView::Phase:
            return visualizeSignedPhaseCentered(makeScalarFromVector(cache.phaseDisplay, cache.phaseDisplaySize),
                                                makeScalarFromVector(cache.pupilDisplay, cache.pupilDisplaySize),
                                                outWidth,
                                                outHeight).pixels;
        case LensDiffDebugView::PhaseEdge:
            if (edgeZone != nullptr) {
                return visualizeSignedPhaseCentered(makeScalarFromVector(edgeZone->phaseDisplay, edgeZone->phaseDisplaySize),
                                                    makeScalarFromVector(edgeZone->pupilDisplay, edgeZone->pupilDisplaySize),
                                                    outWidth,
                                                    outHeight).pixels;
            }
            return visualizeSignedPhaseCentered(makeScalarFromVector(cache.phaseDisplay, cache.phaseDisplaySize),
                                                makeScalarFromVector(cache.pupilDisplay, cache.pupilDisplaySize),
                                                outWidth,
                                                outHeight).pixels;
        case LensDiffDebugView::FieldPsf:
            if (edgeZone != nullptr && !edgeZone->bins.empty()) {
                return visualizeSquareCentered(makeScalarFromKernel(edgeZone->bins.front().full), outWidth, outHeight, true).pixels;
            }
            return visualizeSquareCentered(makeScalarFromKernel(cache.bins.front().full), outWidth, outHeight, true).pixels;
        case LensDiffDebugView::ChromaticSplit: {
            ScalarImage preview(outWidth, outHeight);
            if (cache.bins.empty()) {
                return std::vector<float>(static_cast<std::size_t>(std::max(0, outWidth)) *
                                              static_cast<std::size_t>(std::max(0, outHeight)) * 4U,
                                          0.0f);
            }
            const LensDiffKernel& blue = cache.bins.front().full;
            const LensDiffKernel& green = cache.bins[cache.bins.size() / 2].full;
            const LensDiffKernel& red = cache.bins.back().full;
            const RgbaImage redImg = visualizeSquareCentered(makeScalarFromKernel(red), outWidth, outHeight, true);
            const RgbaImage greenImg = visualizeSquareCentered(makeScalarFromKernel(green), outWidth, outHeight, true);
            const RgbaImage blueImg = visualizeSquareCentered(makeScalarFromKernel(blue), outWidth, outHeight, true);
            RgbaImage out(outWidth, outHeight);
            for (int y = 0; y < outHeight; ++y) {
                for (int x = 0; x < outWidth; ++x) {
                    float* pixel = out.pixel(x, y);
                    pixel[0] = redImg.pixel(x, y)[0];
                    pixel[1] = greenImg.pixel(x, y)[1];
                    pixel[2] = blueImg.pixel(x, y)[2];
                    pixel[3] = 1.0f;
                }
            }
            return out.pixels;
        }
        default:
            return std::vector<float>(static_cast<std::size_t>(std::max(0, outWidth)) *
                                          static_cast<std::size_t>(std::max(0, outHeight)) * 4U,
                                      0.0f);
    }
}

const std::vector<float>& GetLensDiffStaticDebugRgbaCached(const LensDiffParams& params,
                                                           LensDiffPsfBankCache* cache,
                                                           int outWidth,
                                                           int outHeight) {
    static const std::vector<float> empty;
    if (cache == nullptr) {
        return empty;
    }
    if (cache->staticDebugView == static_cast<int>(params.debugView) &&
        cache->staticDebugWidth == outWidth &&
        cache->staticDebugHeight == outHeight &&
        !cache->staticDebugPixels.empty()) {
        return cache->staticDebugPixels;
    }
    LensDiffScopedTimer timer("static-debug-build");
    cache->staticDebugView = static_cast<int>(params.debugView);
    cache->staticDebugWidth = outWidth;
    cache->staticDebugHeight = outHeight;
    cache->staticDebugPixels = BuildLensDiffStaticDebugRgba(params, *cache, outWidth, outHeight);
    return cache->staticDebugPixels;
}

std::vector<float> BuildLensDiffShiftedRawPsf(const std::vector<float>& pupil,
                                              const std::vector<float>& phaseWaves,
                                              int pupilSize,
                                              int rawPsfSize) {
    return buildShiftedRawPsf(pupil, phaseWaves, pupilSize, rawPsfSize);
}

std::vector<float> BuildLensDiffShiftedRawPsf(const std::vector<float>& pupil,
                                              int pupilSize,
                                              int rawPsfSize) {
    return buildShiftedRawPsf(pupil, pupilSize, rawPsfSize);
}

float EstimateLensDiffFirstMinimumRadius(const std::vector<float>& shiftedRawPsf,
                                         int size) {
    return estimateFirstMinimumRadius(shiftedRawPsf, size);
}

std::vector<float> GetLensDiffSpectralWavelengths(LensDiffSpectralMode mode) {
    return wavelengthsForMode(mode);
}

std::shared_ptr<const std::vector<float>> GetLensDiffReferenceRawPsfCached(int pupilSize,
                                                                           int rawPsfSize) {
    const std::pair<int, int> key {pupilSize, rawPsfSize};
    std::lock_guard<std::mutex> lock(referenceRawPsfCacheMutex());
    auto& cache = referenceRawPsfCache();
    const auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }

    LensDiffParams referenceParams {};
    referenceParams.apertureMode = LensDiffApertureMode::Circle;
    referenceParams.apodizationMode = LensDiffApodizationMode::Flat;
    referenceParams.roundness = 1.0;
    referenceParams.pupilResolution = pupilSize;
    const std::vector<float> referencePupil = BuildLensDiffPupilAmplitudeForPsf(referenceParams);
    auto referenceRawPsf =
        std::make_shared<const std::vector<float>>(BuildLensDiffShiftedRawPsf(referencePupil, pupilSize, rawPsfSize));
    cache.emplace(key, referenceRawPsf);
    return referenceRawPsf;
}

void FinalizeLensDiffPsfBankFromRawPsf(const LensDiffParams& params,
                                       const LensDiffPsfBankKey& key,
                                       const std::vector<float>& pupil,
                                       int pupilSize,
                                       const std::vector<float>& rawPsf,
                                       int rawPsfSize,
                                       const std::vector<float>& referenceRawPsf,
                                       LensDiffPsfBankCache& cache) {
    populatePsfBankCacheFromRawPsf(params,
                                   key,
                                   pupil,
                                   pupilSize,
                                   rawPsf,
                                   rawPsfSize,
                                   referenceRawPsf,
                                   cache);
}

void FinalizeLensDiffPsfBankFromBaseKernels(const LensDiffParams& params,
                                            const LensDiffPsfBankKey& key,
                                            const std::vector<float>& pupil,
                                            int pupilSize,
                                            const std::vector<float>& wavelengths,
                                            std::vector<LensDiffKernel> baseKernels,
                                            LensDiffPsfBankCache& cache) {
    populatePsfBankCacheFromBaseKernels(params, key, pupil, pupilSize, wavelengths, std::move(baseKernels), cache);
}

void EnsureLensDiffPsfBank(const LensDiffParams& params,
                           LensDiffPsfBankCache& cache) {
    buildPsfBankInternal(params, cache);
}

bool RunLensDiffCpuReference(const LensDiffRenderRequest& request,
                             const LensDiffParams& params,
                             LensDiffPsfBankCache& cache,
                             std::string* error) {
    if (request.src.data == nullptr || request.dst.data == nullptr) {
        if (error) *error = "missing-source-or-destination-buffer";
        return false;
    }
    if (request.src.bounds.width() <= 0 || request.src.bounds.height() <= 0) {
        if (error) *error = "invalid-source-bounds";
        return false;
    }

    EnsureLensDiffPsfBank(params, cache);
    const LensDiffSpectrumConfig spectrumConfig = BuildLensDiffSpectrumConfig(params, cache.bins);

    const RgbaImage src = loadSourceRgba(request.src, params.inputTransfer);
    const double workingScale = ResolveLensDiffEffectWorkingScale(params);
    const bool resolutionAwareActive = params.resolutionAware && std::abs(workingScale - 1.0) > 1e-6;
    const int workingWidth = resolutionAwareActive ? std::max(1, static_cast<int>(std::lround(src.width * workingScale))) : src.width;
    const int workingHeight = resolutionAwareActive ? std::max(1, static_cast<int>(std::lround(src.height * workingScale))) : src.height;
    const RgbaImage workingSrc = resolutionAwareActive ? resampleRgbaImage(src, workingWidth, workingHeight) : src;
    const ScalarImage selectionMask = buildSelectionMask(workingSrc, params);
    const float redistributionScale = 1.0f - static_cast<float>(clampValue(params.corePreserve, 0.0, 1.0));
    const float protectedCoreFraction = automaticCoreProtectionFraction(params);
    const float maxRedistributedSubtractScale = redistributionScale > 1e-6f
        ? (1.0f - protectedCoreFraction) / redistributionScale
        : 0.0f;
    const RgbaImage redistributed = applyMask(workingSrc, selectionMask, redistributionScale);
    const ScalarImage redistributedDriver = applyMaskScalarDriver(workingSrc, selectionMask, params.extractionMode, redistributionScale);

    const float effectGain = params.energyMode == LensDiffEnergyMode::Preserve
        ? static_cast<float>(clampValue(params.effectGain, 0.0, 1.0))
        : static_cast<float>(std::max(0.0, params.effectGain));
    const float coreCompensation = params.energyMode == LensDiffEnergyMode::Preserve
        ? effectGain
        : static_cast<float>(std::max(0.0, params.coreCompensation));

    RgbaImage coreEffect(workingSrc.width, workingSrc.height);
    RgbaImage structureEffect(workingSrc.width, workingSrc.height);
    RgbaImage effect(workingSrc.width, workingSrc.height);
    auto renderEffectFromBins = [&](const std::vector<LensDiffPsfBin>& bins,
                                    RgbaImage* outCore,
                                    RgbaImage* outStructure,
                                    RgbaImage* outEffect) {
        const LensDiffSpectrumConfig zoneSpectrumConfig = BuildLensDiffSpectrumConfig(params, bins);
        RgbaImage zoneCore(workingSrc.width, workingSrc.height);
        RgbaImage zoneStructure(workingSrc.width, workingSrc.height);
        RgbaImage zoneEffect(workingSrc.width, workingSrc.height);
        if (params.spectralMode == LensDiffSpectralMode::Mono) {
            if (params.lookMode == LensDiffLookMode::Split) {
                zoneCore = convolveRgba(redistributed, bins.front().core);
                zoneStructure = convolveRgba(redistributed, bins.front().structure);
                applyShoulder(zoneCore, static_cast<float>(params.coreShoulder));
                applyShoulder(zoneStructure, static_cast<float>(params.structureShoulder));
                zoneEffect = addImages(zoneCore,
                                       zoneStructure,
                                       static_cast<float>(std::max(0.0, params.coreGain)),
                                       static_cast<float>(std::max(0.0, params.structureGain)));
            } else {
                zoneEffect = convolveRgba(redistributed, bins.front().full);
                zoneCore = convolveRgba(redistributed, bins.front().core);
                zoneStructure = convolveRgba(redistributed, bins.front().structure);
            }
        } else {
            if (params.lookMode == LensDiffLookMode::Split) {
                zoneCore = buildSpectralEffect(redistributedDriver, bins, zoneSpectrumConfig, params, true, false);
                zoneStructure = buildSpectralEffect(redistributedDriver, bins, zoneSpectrumConfig, params, false, true);
                applyShoulder(zoneCore, static_cast<float>(params.coreShoulder));
                applyShoulder(zoneStructure, static_cast<float>(params.structureShoulder));
                zoneEffect = addImages(zoneCore,
                                       zoneStructure,
                                       static_cast<float>(std::max(0.0, params.coreGain)),
                                       static_cast<float>(std::max(0.0, params.structureGain)));
            } else {
                zoneEffect = buildSpectralEffect(redistributedDriver, bins, zoneSpectrumConfig, params, true, true);
                zoneCore = buildSpectralEffect(redistributedDriver, bins, zoneSpectrumConfig, params, true, false);
                zoneStructure = buildSpectralEffect(redistributedDriver, bins, zoneSpectrumConfig, params, false, true);
            }
        }
        if (outCore) *outCore = std::move(zoneCore);
        if (outStructure) *outStructure = std::move(zoneStructure);
        if (outEffect) *outEffect = std::move(zoneEffect);
    };

    if (cache.fieldZones.empty()) {
        renderEffectFromBins(cache.bins, &coreEffect, &structureEffect, &effect);
    } else {
        for (const auto& zone : cache.fieldZones) {
            RgbaImage zoneCore(workingSrc.width, workingSrc.height);
            RgbaImage zoneStructure(workingSrc.width, workingSrc.height);
            RgbaImage zoneEffect(workingSrc.width, workingSrc.height);
            renderEffectFromBins(zone.bins, &zoneCore, &zoneStructure, &zoneEffect);
            accumulateWeightedImage(zoneCore, coreEffect, zone.zoneX, zone.zoneY);
            accumulateWeightedImage(zoneStructure, structureEffect, zone.zoneX, zone.zoneY);
            accumulateWeightedImage(zoneEffect, effect, zone.zoneX, zone.zoneY);
        }
    }

    if (params.energyMode == LensDiffEnergyMode::Preserve) {
        const float inputEnergy = luminanceSum(redistributed);
        const float effectEnergy = luminanceSum(effect);
        if (effectEnergy > 1e-6f) {
            scaleImage(effect, inputEnergy / effectEnergy);
        }
    }

    RgbaImage scatterPreview(workingSrc.width, workingSrc.height);
    const double scatterRadiusPx = ResolveLensDiffScatterRadiusPx(params);
    if (params.scatterAmount > 1e-6 && scatterRadiusPx > 0.25) {
        const LensDiffKernel scatterKernel = buildGaussianKernel(static_cast<float>(scatterRadiusPx));
        scatterPreview = convolveRgba(effect, scatterKernel);
        effect = addImages(effect, scatterPreview, 1.0f, static_cast<float>(std::max(0.0, params.scatterAmount)));
    }
    RgbaImage creativeFringePreview(workingSrc.width, workingSrc.height);
    effect = applyCreativeFringe(effect,
                                 static_cast<float>(std::max(0.0, ResolveLensDiffCreativeFringePx(params))),
                                 &creativeFringePreview);

    const ScalarImage selectionMaskDisplay = resolutionAwareActive ? resampleScalarImage(selectionMask, src.width, src.height) : selectionMask;
    const RgbaImage redistributedDisplay = resolutionAwareActive ? resampleRgbaImage(redistributed, src.width, src.height) : redistributed;
    const RgbaImage coreEffectDisplay = resolutionAwareActive ? resampleRgbaImage(coreEffect, src.width, src.height) : coreEffect;
    const RgbaImage structureEffectDisplay = resolutionAwareActive ? resampleRgbaImage(structureEffect, src.width, src.height) : structureEffect;
    const RgbaImage effectDisplay = resolutionAwareActive ? resampleRgbaImage(effect, src.width, src.height) : effect;
    const RgbaImage scatterPreviewDisplay = resolutionAwareActive ? resampleRgbaImage(scatterPreview, src.width, src.height) : scatterPreview;
    const RgbaImage creativeFringePreviewDisplay = resolutionAwareActive ? resampleRgbaImage(creativeFringePreview, src.width, src.height) : creativeFringePreview;

    RgbaImage finalImage(src.width, src.height);
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            const float* srcPixel = src.pixel(x, y);
            const float* redistPixel = redistributedDisplay.pixel(x, y);
            const float* effectPixel = effectDisplay.pixel(x, y);
            float* outPixel = finalImage.pixel(x, y);
            const float rawR = std::max(0.0f, srcPixel[0] - coreCompensation * redistPixel[0] + effectGain * effectPixel[0]);
            const float rawG = std::max(0.0f, srcPixel[1] - coreCompensation * redistPixel[1] + effectGain * effectPixel[1]);
            const float rawB = std::max(0.0f, srcPixel[2] - coreCompensation * redistPixel[2] + effectGain * effectPixel[2]);

            const float floorR = std::max(0.0f, srcPixel[0] - redistPixel[0] * maxRedistributedSubtractScale);
            const float floorG = std::max(0.0f, srcPixel[1] - redistPixel[1] * maxRedistributedSubtractScale);
            const float floorB = std::max(0.0f, srcPixel[2] - redistPixel[2] * maxRedistributedSubtractScale);

            outPixel[0] = std::max(rawR, floorR);
            outPixel[1] = std::max(rawG, floorG);
            outPixel[2] = std::max(rawB, floorB);
            outPixel[3] = srcPixel[3];
        }
    }

    const RgbaImage debugImage = renderDebugView(params,
                                                 selectionMaskDisplay,
                                                 cache,
                                                 coreEffectDisplay,
                                                 structureEffectDisplay,
                                                 effectDisplay,
                                                 creativeFringePreviewDisplay,
                                                 scatterPreviewDisplay,
                                                 src.width,
                                                 src.height);
    const bool useDebug = params.debugView != LensDiffDebugView::Final && debugImage.width == src.width;
    const RgbaImage& outputImage = useDebug ? debugImage : finalImage;

    const LensDiffImageRect outputRect = intersectRect(request.renderWindow, request.dst.bounds);
    for (int y = outputRect.y1; y < outputRect.y2; ++y) {
        for (int x = outputRect.x1; x < outputRect.x2; ++x) {
            const int sx = x - request.src.bounds.x1;
            const int sy = y - request.src.bounds.y1;
            std::array<float, 4> rgba {0.0f, 0.0f, 0.0f, 1.0f};
            if (sx >= 0 && sx < outputImage.width && sy >= 0 && sy < outputImage.height) {
                const float* pixel = outputImage.pixel(sx, sy);
                WorkshopColor::Vec3f encoded = {
                    pixel[0],
                    pixel[1],
                    pixel[2],
                };
                if (params.debugView == LensDiffDebugView::Final) {
                    encoded = LensDiffTransfer::encodeFromLinear(encoded, params.inputTransfer);
                }
                rgba[0] = encoded.x;
                rgba[1] = encoded.y;
                rgba[2] = encoded.z;
                rgba[3] = params.debugView == LensDiffDebugView::Final ? src.pixel(sx, sy)[3] : 1.0f;
            }
            writePixel(request.dst, x, y, rgba);
        }
    }

    return true;
}

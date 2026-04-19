#include "LensDiffSpectrum.h"

#include <algorithm>
#include <cmath>

namespace {

template <typename T>
T clampValue(T value, T lo, T hi) {
    return std::max(lo, std::min(value, hi));
}

bool approxEqual(float a, float b, float epsilon = 1e-3f) {
    return std::fabs(a - b) <= epsilon;
}

float safeLuma(const WorkshopColor::Vec3f& rgb) {
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

WorkshopColor::Vec3f add(const WorkshopColor::Vec3f& a, const WorkshopColor::Vec3f& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

WorkshopColor::Vec3f scale(const WorkshopColor::Vec3f& v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

WorkshopColor::Vec3f lerp(const WorkshopColor::Vec3f& a, const WorkshopColor::Vec3f& b, float t) {
    return {
        a.x * (1.0f - t) + b.x * t,
        a.y * (1.0f - t) + b.y * t,
        a.z * (1.0f - t) + b.z * t,
    };
}

WorkshopColor::Vec3f clampPositive(const WorkshopColor::Vec3f& v) {
    return {std::max(0.0f, v.x), std::max(0.0f, v.y), std::max(0.0f, v.z)};
}

WorkshopColor::Vec3f normalizeToLuma(const WorkshopColor::Vec3f& rgb, float targetLuma) {
    const float current = safeLuma(rgb);
    if (current <= 1e-6f || targetLuma <= 0.0f) {
        return rgb;
    }
    return scale(rgb, targetLuma / current);
}

WorkshopColor::Vec3f fallbackSpectralRgb(float wavelengthNm) {
    if (wavelengthNm <= 500.0f) {
        return {0.08f, 0.55f, 1.0f};
    }
    if (wavelengthNm <= 580.0f) {
        return {0.42f, 1.0f, 0.24f};
    }
    return {1.0f, 0.22f, 0.08f};
}

WorkshopColor::Vec3f spectralXyzAtWavelength(float wavelengthNm) {
    const auto& cmfs = WorkshopColor::cie1931XyzCmfs5nm();
    const int index = clampValue(static_cast<int>(std::lround((wavelengthNm - 380.0f) / 5.0f)), 0, static_cast<int>(cmfs.size() - 1));
    return cmfs[static_cast<std::size_t>(index)];
}

WorkshopColor::Vec3f naturalColumnFromWavelength(float wavelengthNm) {
    const WorkshopColor::Mat3f xyzToRgb = WorkshopColor::xyzToRgbMatrix(WorkshopColor::ColorPrimariesId::Rec709);
    WorkshopColor::Vec3f rgb = clampPositive(WorkshopColor::mul(xyzToRgb, spectralXyzAtWavelength(wavelengthNm)));
    if (rgb.x + rgb.y + rgb.z <= 1e-6f) {
        rgb = fallbackSpectralRgb(wavelengthNm);
    }
    return rgb;
}

bool invert3x3(const float m[3][3], float invOut[3][3]) {
    const float c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    const float c01 = m[0][2] * m[2][1] - m[0][1] * m[2][2];
    const float c02 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    const float c10 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    const float c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    const float c12 = m[0][2] * m[1][0] - m[0][0] * m[1][2];
    const float c20 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    const float c21 = m[0][1] * m[2][0] - m[0][0] * m[2][1];
    const float c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];

    const float det = m[0][0] * c00 + m[0][1] * c10 + m[0][2] * c20;
    if (std::fabs(det) <= 1e-8f) {
        return false;
    }

    const float invDet = 1.0f / det;
    invOut[0][0] = c00 * invDet;
    invOut[0][1] = c01 * invDet;
    invOut[0][2] = c02 * invDet;
    invOut[1][0] = c10 * invDet;
    invOut[1][1] = c11 * invDet;
    invOut[1][2] = c12 * invDet;
    invOut[2][0] = c20 * invDet;
    invOut[2][1] = c21 * invDet;
    invOut[2][2] = c22 * invDet;
    return true;
}

std::array<WorkshopColor::Vec3f, 3> balanceColumnsToWhite(const std::array<WorkshopColor::Vec3f, 3>& columns) {
    float basis[3][3] = {
        {columns[0].x, columns[1].x, columns[2].x},
        {columns[0].y, columns[1].y, columns[2].y},
        {columns[0].z, columns[1].z, columns[2].z},
    };
    float inv[3][3] = {};
    if (!invert3x3(basis, inv)) {
        return columns;
    }

    const float white[3] = {1.0f, 1.0f, 1.0f};
    std::array<float, 3> weights {};
    for (int i = 0; i < 3; ++i) {
        weights[static_cast<std::size_t>(i)] = inv[i][0] * white[0] + inv[i][1] * white[1] + inv[i][2] * white[2];
        if (!std::isfinite(weights[static_cast<std::size_t>(i)]) || weights[static_cast<std::size_t>(i)] <= 0.0f) {
            return columns;
        }
    }

    return {
        scale(columns[0], weights[0]),
        scale(columns[1], weights[1]),
        scale(columns[2], weights[2]),
    };
}

std::vector<WorkshopColor::Vec3f> balanceColumnsToNeutralSum(const std::vector<WorkshopColor::Vec3f>& columns) {
    if (columns.empty()) {
        return columns;
    }

    WorkshopColor::Vec3f sum {0.0f, 0.0f, 0.0f};
    for (const auto& column : columns) {
        sum = add(sum, column);
    }

    const float maxChannel = std::max(sum.x, std::max(sum.y, sum.z));
    if (maxChannel <= 1e-6f) {
        return columns;
    }

    const float targetChannel = (sum.x + sum.y + sum.z) / 3.0f;
    if (targetChannel <= 1e-6f) {
        return columns;
    }

    const WorkshopColor::Vec3f channelScale {
        targetChannel / std::max(sum.x, 1e-6f),
        targetChannel / std::max(sum.y, 1e-6f),
        targetChannel / std::max(sum.z, 1e-6f),
    };

    std::vector<WorkshopColor::Vec3f> balanced(columns.size());
    for (std::size_t i = 0; i < columns.size(); ++i) {
        WorkshopColor::Vec3f adjusted {
            columns[i].x * channelScale.x,
            columns[i].y * channelScale.y,
            columns[i].z * channelScale.z,
        };
        balanced[i] = normalizeToLuma(adjusted, safeLuma(columns[i]));
    }

    return balanced;
}

WorkshopColor::Vec3f sumColumns(const std::vector<WorkshopColor::Vec3f>& columns) {
    WorkshopColor::Vec3f sum {0.0f, 0.0f, 0.0f};
    for (const auto& column : columns) {
        sum = add(sum, column);
    }
    return sum;
}

std::vector<float> canonicalSpectral5Wavelengths() {
    return {440.0f, 490.0f, 540.0f, 590.0f, 640.0f};
}

std::vector<float> canonicalSpectral9Wavelengths() {
    return {420.0f, 450.0f, 480.0f, 510.0f, 540.0f, 570.0f, 600.0f, 630.0f, 660.0f};
}

bool matchesCanonicalWavelengths(const std::vector<LensDiffPsfBin>& bins,
                                 const std::vector<float>& wavelengths) {
    if (bins.size() != wavelengths.size()) {
        return false;
    }
    for (std::size_t i = 0; i < wavelengths.size(); ++i) {
        if (!approxEqual(bins[i].wavelengthNm, wavelengths[i])) {
            return false;
        }
    }
    return true;
}

std::vector<WorkshopColor::Vec3f> balanceColumnsToReferenceSum(const std::vector<WorkshopColor::Vec3f>& columns,
                                                               const WorkshopColor::Vec3f& targetSum) {
    if (columns.empty()) {
        return columns;
    }

    const WorkshopColor::Vec3f currentSum = sumColumns(columns);
    const float currentAvg = (currentSum.x + currentSum.y + currentSum.z) / 3.0f;
    const float targetAvg = (targetSum.x + targetSum.y + targetSum.z) / 3.0f;
    if (currentAvg <= 1e-6f || targetAvg <= 1e-6f) {
        return balanceColumnsToNeutralSum(columns);
    }

    const WorkshopColor::Vec3f normalizedTarget {
        targetSum.x * (currentAvg / targetAvg),
        targetSum.y * (currentAvg / targetAvg),
        targetSum.z * (currentAvg / targetAvg),
    };

    const WorkshopColor::Vec3f channelScale {
        normalizedTarget.x / std::max(currentSum.x, 1e-6f),
        normalizedTarget.y / std::max(currentSum.y, 1e-6f),
        normalizedTarget.z / std::max(currentSum.z, 1e-6f),
    };

    std::vector<WorkshopColor::Vec3f> balanced(columns.size());
    for (std::size_t i = 0; i < columns.size(); ++i) {
        WorkshopColor::Vec3f adjusted {
            columns[i].x * channelScale.x,
            columns[i].y * channelScale.y,
            columns[i].z * channelScale.z,
        };
        balanced[i] = normalizeToLuma(adjusted, safeLuma(columns[i]));
    }
    return balanced;
}

WorkshopColor::Vec3f cyanMagentaTarget(float wavelengthNm) {
    if (wavelengthNm <= 500.0f) return {0.18f, 0.84f, 1.0f};
    if (wavelengthNm <= 580.0f) return {0.90f, 0.89f, 0.96f};
    return {0.98f, 0.68f, 0.82f};
}

WorkshopColor::Vec3f warmCoolTarget(float wavelengthNm) {
    if (wavelengthNm <= 500.0f) return {0.24f, 0.74f, 1.0f};
    if (wavelengthNm <= 580.0f) return {0.92f, 0.86f, 0.76f};
    return {1.0f, 0.76f, 0.48f};
}

WorkshopColor::Vec3f styleColumnForWavelength(float wavelengthNm,
                                              LensDiffSpectrumStyle style,
                                              const WorkshopColor::Vec3f& naturalColumn) {
    WorkshopColor::Vec3f target = naturalColumn;
    float styleStrength = 1.0f;
    switch (style) {
        case LensDiffSpectrumStyle::CyanMagenta:
            target = cyanMagentaTarget(wavelengthNm);
            styleStrength = 0.56f;
            break;
        case LensDiffSpectrumStyle::WarmCool:
            target = warmCoolTarget(wavelengthNm);
            styleStrength = 0.52f;
            break;
        case LensDiffSpectrumStyle::Natural:
        default:
            target = naturalColumn;
            styleStrength = 0.0f;
            break;
    }
    target = lerp(naturalColumn, target, clampValue(styleStrength, 0.0f, 1.0f));
    return normalizeToLuma(target, safeLuma(naturalColumn));
}

WorkshopColor::Vec3f canonicalSpectral5ReferenceSum(LensDiffSpectrumStyle style) {
    std::vector<float> wavelengths = canonicalSpectral5Wavelengths();
    std::vector<WorkshopColor::Vec3f> columns(wavelengths.size());
    for (std::size_t i = 0; i < wavelengths.size(); ++i) {
        columns[i] = naturalColumnFromWavelength(wavelengths[i]);
    }
    columns = balanceColumnsToNeutralSum(columns);
    if (style != LensDiffSpectrumStyle::Natural) {
        std::vector<WorkshopColor::Vec3f> styled(columns.size());
        for (std::size_t i = 0; i < wavelengths.size(); ++i) {
            styled[i] = styleColumnForWavelength(wavelengths[i], style, columns[i]);
        }
        columns = balanceColumnsToNeutralSum(styled);
    }
    return sumColumns(columns);
}

std::vector<WorkshopColor::Vec3f> canonicalSpectral5ReferenceColumns(LensDiffSpectrumStyle style) {
    std::vector<float> wavelengths = canonicalSpectral5Wavelengths();
    std::vector<WorkshopColor::Vec3f> columns(wavelengths.size());
    for (std::size_t i = 0; i < wavelengths.size(); ++i) {
        columns[i] = naturalColumnFromWavelength(wavelengths[i]);
    }
    columns = balanceColumnsToNeutralSum(columns);
    if (style != LensDiffSpectrumStyle::Natural) {
        std::vector<WorkshopColor::Vec3f> styled(columns.size());
        for (std::size_t i = 0; i < wavelengths.size(); ++i) {
            styled[i] = styleColumnForWavelength(wavelengths[i], style, columns[i]);
        }
        columns = balanceColumnsToNeutralSum(styled);
    }
    return columns;
}

WorkshopColor::Vec3f interpolateReferenceColumn(float wavelengthNm,
                                                const std::vector<float>& referenceWavelengths,
                                                const std::vector<WorkshopColor::Vec3f>& referenceColumns) {
    if (referenceWavelengths.empty() || referenceColumns.empty()) {
        return {1.0f, 1.0f, 1.0f};
    }
    if (referenceWavelengths.size() == 1 || referenceColumns.size() == 1) {
        return referenceColumns.front();
    }
    if (wavelengthNm <= referenceWavelengths.front()) {
        return referenceColumns.front();
    }
    if (wavelengthNm >= referenceWavelengths.back()) {
        return referenceColumns.back();
    }

    for (std::size_t i = 1; i < referenceWavelengths.size() && i < referenceColumns.size(); ++i) {
        const float upperWavelength = referenceWavelengths[i];
        if (wavelengthNm <= upperWavelength) {
            const float lowerWavelength = referenceWavelengths[i - 1];
            const float span = std::max(upperWavelength - lowerWavelength, 1e-6f);
            const float t = clampValue((wavelengthNm - lowerWavelength) / span, 0.0f, 1.0f);
            return lerp(referenceColumns[i - 1], referenceColumns[i], t);
        }
    }

    return referenceColumns.back();
}

std::vector<WorkshopColor::Vec3f> interpolateReferenceColumnsForBins(
    const std::vector<LensDiffPsfBin>& bins,
    const std::vector<float>& referenceWavelengths,
    const std::vector<WorkshopColor::Vec3f>& referenceColumns) {
    std::vector<WorkshopColor::Vec3f> columns(bins.size());
    for (std::size_t i = 0; i < bins.size(); ++i) {
        columns[i] = interpolateReferenceColumn(bins[i].wavelengthNm, referenceWavelengths, referenceColumns);
    }
    return columns;
}

void writeMatrixColumns(const std::array<WorkshopColor::Vec3f, 3>& columns, std::array<float, 9>* matrix) {
    (*matrix)[0] = columns[0].x; (*matrix)[1] = columns[1].x; (*matrix)[2] = columns[2].x;
    (*matrix)[3] = columns[0].y; (*matrix)[4] = columns[1].y; (*matrix)[5] = columns[2].y;
    (*matrix)[6] = columns[0].z; (*matrix)[7] = columns[1].z; (*matrix)[8] = columns[2].z;
}

void writeMatrixColumns(const std::vector<WorkshopColor::Vec3f>& columns,
                        std::array<float, kLensDiffMaxSpectralBins * 3>* matrix) {
    matrix->fill(0.0f);
    const int count = std::min<int>(static_cast<int>(columns.size()), kLensDiffMaxSpectralBins);
    for (int i = 0; i < count; ++i) {
        (*matrix)[static_cast<std::size_t>(0 * kLensDiffMaxSpectralBins + i)] = columns[static_cast<std::size_t>(i)].x;
        (*matrix)[static_cast<std::size_t>(1 * kLensDiffMaxSpectralBins + i)] = columns[static_cast<std::size_t>(i)].y;
        (*matrix)[static_cast<std::size_t>(2 * kLensDiffMaxSpectralBins + i)] = columns[static_cast<std::size_t>(i)].z;
    }
}

std::array<float, 3> mulMatrix(const std::array<float, kLensDiffMaxSpectralBins * 3>& matrix,
                               const std::array<float, kLensDiffMaxSpectralBins>& bins,
                               int binCount) {
    if (binCount <= 1) {
        return {bins[0], bins[0], bins[0]};
    }
    std::array<float, 3> out {0.0f, 0.0f, 0.0f};
    const int count = std::min(binCount, kLensDiffMaxSpectralBins);
    for (int i = 0; i < count; ++i) {
        out[0] += matrix[static_cast<std::size_t>(0 * kLensDiffMaxSpectralBins + i)] * bins[static_cast<std::size_t>(i)];
        out[1] += matrix[static_cast<std::size_t>(1 * kLensDiffMaxSpectralBins + i)] * bins[static_cast<std::size_t>(i)];
        out[2] += matrix[static_cast<std::size_t>(2 * kLensDiffMaxSpectralBins + i)] * bins[static_cast<std::size_t>(i)];
    }
    return out;
}

} // namespace

LensDiffSpectrumConfig BuildLensDiffSpectrumConfig(const LensDiffParams& params,
                                                   const std::vector<LensDiffPsfBin>& bins) {
    LensDiffSpectrumConfig config {};
    config.binCount = static_cast<int>(std::min<std::size_t>(kLensDiffMaxSpectralBins, bins.size()));
    if (config.binCount <= 1) {
        config.binCount = 1;
        config.naturalMatrix.fill(0.0f);
        config.styleMatrix.fill(0.0f);
        config.naturalMatrix[0] = 1.0f;
        config.naturalMatrix[kLensDiffMaxSpectralBins] = 1.0f;
        config.naturalMatrix[kLensDiffMaxSpectralBins * 2] = 1.0f;
        config.styleMatrix = config.naturalMatrix;
        return config;
    }

    const bool useCanonical9ReferenceRefinement =
        config.binCount == static_cast<int>(canonicalSpectral9Wavelengths().size()) &&
        matchesCanonicalWavelengths(bins, canonicalSpectral9Wavelengths());

    std::vector<WorkshopColor::Vec3f> naturalColumns(static_cast<std::size_t>(config.binCount));
    if (useCanonical9ReferenceRefinement) {
        const std::vector<WorkshopColor::Vec3f> referenceColumns =
            canonicalSpectral5ReferenceColumns(LensDiffSpectrumStyle::Natural);
        naturalColumns = interpolateReferenceColumnsForBins(bins, canonicalSpectral5Wavelengths(), referenceColumns);
    } else {
        for (int i = 0; i < config.binCount; ++i) {
            naturalColumns[static_cast<std::size_t>(i)] =
                naturalColumnFromWavelength(bins[static_cast<std::size_t>(i)].wavelengthNm);
        }
    }
    if (config.binCount == 3) {
        std::array<WorkshopColor::Vec3f, 3> tri {
            naturalColumns[0], naturalColumns[1], naturalColumns[2]
        };
        tri = balanceColumnsToWhite(tri);
        naturalColumns[0] = tri[0];
        naturalColumns[1] = tri[1];
        naturalColumns[2] = tri[2];
    } else if (config.binCount >= 9) {
        naturalColumns = balanceColumnsToReferenceSum(
            naturalColumns,
            canonicalSpectral5ReferenceSum(LensDiffSpectrumStyle::Natural));
    } else {
        naturalColumns = balanceColumnsToNeutralSum(naturalColumns);
    }

    std::vector<WorkshopColor::Vec3f> styleColumns = naturalColumns;
    if (params.spectrumStyle != LensDiffSpectrumStyle::Natural) {
        if (useCanonical9ReferenceRefinement) {
            const std::vector<WorkshopColor::Vec3f> referenceColumns =
                canonicalSpectral5ReferenceColumns(params.spectrumStyle);
            styleColumns = interpolateReferenceColumnsForBins(bins, canonicalSpectral5Wavelengths(), referenceColumns);
        } else {
            for (int i = 0; i < config.binCount; ++i) {
                styleColumns[static_cast<std::size_t>(i)] = styleColumnForWavelength(
                    bins[static_cast<std::size_t>(i)].wavelengthNm,
                    params.spectrumStyle,
                    naturalColumns[static_cast<std::size_t>(i)]);
            }
        }
        if (config.binCount == 3) {
            std::array<WorkshopColor::Vec3f, 3> tri {
                styleColumns[0], styleColumns[1], styleColumns[2]
            };
            tri = balanceColumnsToWhite(tri);
            styleColumns[0] = tri[0];
            styleColumns[1] = tri[1];
            styleColumns[2] = tri[2];
        } else if (config.binCount >= 9) {
            styleColumns = balanceColumnsToReferenceSum(
                styleColumns,
                canonicalSpectral5ReferenceSum(params.spectrumStyle));
        } else {
            styleColumns = balanceColumnsToNeutralSum(styleColumns);
        }
    }

    writeMatrixColumns(naturalColumns, &config.naturalMatrix);
    writeMatrixColumns(styleColumns, &config.styleMatrix);
    return config;
}

std::array<float, 3> MapLensDiffSpectralBins(const std::array<float, kLensDiffMaxSpectralBins>& bins,
                                             const LensDiffParams& params,
                                             const LensDiffSpectrumConfig& config) {
    std::array<float, 3> natural = mulMatrix(config.naturalMatrix, bins, config.binCount);
    std::array<float, 3> style = mulMatrix(config.styleMatrix, bins, config.binCount);
    const float force = clampValue(static_cast<float>(params.spectrumForce), 0.0f, 1.0f);

    std::array<float, 3> rgb {
        natural[0] * (1.0f - force) + style[0] * force,
        natural[1] * (1.0f - force) + style[1] * force,
        natural[2] * (1.0f - force) + style[2] * force,
    };

    for (float& channel : rgb) {
        channel = std::max(0.0f, channel);
    }

    const float saturation = static_cast<float>(std::max(0.0, params.spectrumSaturation));
    const float gray = 0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2];
    rgb[0] = gray + (rgb[0] - gray) * saturation;
    rgb[1] = gray + (rgb[1] - gray) * saturation;
    rgb[2] = gray + (rgb[2] - gray) * saturation;

    if (!params.chromaticAffectsLuma) {
        const float targetLuma = 0.2126f * natural[0] + 0.7152f * natural[1] + 0.0722f * natural[2];
        const float currentLuma = 0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2];
        if (currentLuma > 1e-6f) {
            const float scale = targetLuma / currentLuma;
            rgb[0] *= scale;
            rgb[1] *= scale;
            rgb[2] *= scale;
        }
    }

    return rgb;
}

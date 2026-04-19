#include "LensDiffPhase.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kZernikeDefocusNorm = 1.7320508075688772f;   // sqrt(3)
constexpr float kZernikeAstigNorm = 2.4494897427831781f;     // sqrt(6)
constexpr float kZernikeComaNorm = 2.8284271247461901f;      // sqrt(8)
constexpr float kZernikeSphericalNorm = 2.2360679774997898f; // sqrt(5)
constexpr float kZernikeTrefoilNorm = 2.8284271247461901f;   // sqrt(8)
constexpr float kZernikeSecondaryAstigNorm = 3.1622776601683795f; // sqrt(10)
constexpr float kZernikeQuadrafoilNorm = 3.1622776601683795f;     // sqrt(10)
constexpr float kZernikeSecondaryComaNorm = 3.4641016151377544f;  // sqrt(12)

} // namespace

bool HasLensDiffNonFlatPhase(const LensDiffParams& params) {
    return params.phaseMode == LensDiffPhaseMode::Enabled &&
           (std::abs(params.phaseDefocus) > 1e-6 ||
            std::abs(params.phaseAstigmatism0) > 1e-6 ||
            std::abs(params.phaseAstigmatism45) > 1e-6 ||
            std::abs(params.phaseComaX) > 1e-6 ||
            std::abs(params.phaseComaY) > 1e-6 ||
            std::abs(params.phaseSpherical) > 1e-6 ||
            std::abs(params.phaseTrefoilX) > 1e-6 ||
            std::abs(params.phaseTrefoilY) > 1e-6 ||
            std::abs(params.phaseSecondaryAstigmatism0) > 1e-6 ||
            std::abs(params.phaseSecondaryAstigmatism45) > 1e-6 ||
            std::abs(params.phaseQuadrafoil0) > 1e-6 ||
            std::abs(params.phaseQuadrafoil45) > 1e-6 ||
            std::abs(params.phaseSecondaryComaX) > 1e-6 ||
            std::abs(params.phaseSecondaryComaY) > 1e-6 ||
            std::abs(params.pupilDecenterX) > 1e-6 ||
            std::abs(params.pupilDecenterY) > 1e-6);
}

bool HasLensDiffFieldPhase(const LensDiffParams& params) {
    return params.phaseMode == LensDiffPhaseMode::Enabled &&
           std::abs(params.phaseFieldStrength) > 1e-6 &&
           (std::abs(params.phaseFieldDefocus) > 1e-6 ||
            std::abs(params.phaseFieldAstigRadial) > 1e-6 ||
            std::abs(params.phaseFieldAstigTangential) > 1e-6 ||
            std::abs(params.phaseFieldComaRadial) > 1e-6 ||
            std::abs(params.phaseFieldComaTangential) > 1e-6 ||
            std::abs(params.phaseFieldSpherical) > 1e-6 ||
            std::abs(params.phaseFieldTrefoilRadial) > 1e-6 ||
            std::abs(params.phaseFieldTrefoilTangential) > 1e-6 ||
            std::abs(params.phaseFieldSecondaryAstigRadial) > 1e-6 ||
            std::abs(params.phaseFieldSecondaryAstigTangential) > 1e-6 ||
            std::abs(params.phaseFieldQuadrafoilRadial) > 1e-6 ||
            std::abs(params.phaseFieldQuadrafoilTangential) > 1e-6 ||
            std::abs(params.phaseFieldSecondaryComaRadial) > 1e-6 ||
            std::abs(params.phaseFieldSecondaryComaTangential) > 1e-6);
}

void DisableLensDiffPhase(LensDiffParams* params) {
    if (!params) {
        return;
    }
    params->phaseMode = LensDiffPhaseMode::Off;
    params->phaseDefocus = 0.0;
    params->phaseAstigmatism0 = 0.0;
    params->phaseAstigmatism45 = 0.0;
    params->phaseComaX = 0.0;
    params->phaseComaY = 0.0;
    params->phaseSpherical = 0.0;
    params->phaseTrefoilX = 0.0;
    params->phaseTrefoilY = 0.0;
    params->phaseSecondaryAstigmatism0 = 0.0;
    params->phaseSecondaryAstigmatism45 = 0.0;
    params->phaseQuadrafoil0 = 0.0;
    params->phaseQuadrafoil45 = 0.0;
    params->phaseSecondaryComaX = 0.0;
    params->phaseSecondaryComaY = 0.0;
    params->pupilDecenterX = 0.0;
    params->pupilDecenterY = 0.0;
    params->phaseFieldStrength = 0.0;
    params->phaseFieldEdgeBias = 0.0;
    params->phaseFieldDefocus = 0.0;
    params->phaseFieldAstigRadial = 0.0;
    params->phaseFieldAstigTangential = 0.0;
    params->phaseFieldComaRadial = 0.0;
    params->phaseFieldComaTangential = 0.0;
    params->phaseFieldSpherical = 0.0;
    params->phaseFieldTrefoilRadial = 0.0;
    params->phaseFieldTrefoilTangential = 0.0;
    params->phaseFieldSecondaryAstigRadial = 0.0;
    params->phaseFieldSecondaryAstigTangential = 0.0;
    params->phaseFieldQuadrafoilRadial = 0.0;
    params->phaseFieldQuadrafoilTangential = 0.0;
    params->phaseFieldSecondaryComaRadial = 0.0;
    params->phaseFieldSecondaryComaTangential = 0.0;
}

void addOrientationPair(double radialValue,
                        double tangentialValue,
                        double angle,
                        int harmonic,
                        double* x,
                        double* y) {
    if (x == nullptr || y == nullptr) return;
    const double base = static_cast<double>(harmonic) * angle;
    const double cs = std::cos(base);
    const double sn = std::sin(base);
    *x += radialValue * cs - tangentialValue * sn;
    *y += radialValue * sn + tangentialValue * cs;
}

LensDiffParams ResolveLensDiffFieldZoneParams(const LensDiffParams& params, float normalizedX, float normalizedY) {
    LensDiffParams resolved = params;
    if (!HasLensDiffFieldPhase(params)) {
        return resolved;
    }

    const double halfDiagonal = std::sqrt(2.0);
    const double radialNorm = std::min(1.0, std::sqrt(static_cast<double>(normalizedX) * normalizedX +
                                                      static_cast<double>(normalizedY) * normalizedY) / halfDiagonal);
    const double edgeBias = std::clamp(params.phaseFieldEdgeBias, 0.0, 1.0);
    const double fieldRamp = std::pow(radialNorm, 1.0 + edgeBias * 2.0);
    const double fieldScale = params.phaseFieldStrength * fieldRamp;
    if (std::abs(fieldScale) <= 1e-9) {
        return resolved;
    }

    const double angle = std::atan2(static_cast<double>(normalizedY), static_cast<double>(normalizedX));
    resolved.phaseDefocus += params.phaseFieldDefocus * fieldScale;
    resolved.phaseSpherical += params.phaseFieldSpherical * fieldScale;
    addOrientationPair(params.phaseFieldAstigRadial * fieldScale,
                       params.phaseFieldAstigTangential * fieldScale,
                       angle,
                       2,
                       &resolved.phaseAstigmatism0,
                       &resolved.phaseAstigmatism45);
    addOrientationPair(params.phaseFieldComaRadial * fieldScale,
                       params.phaseFieldComaTangential * fieldScale,
                       angle,
                       1,
                       &resolved.phaseComaX,
                       &resolved.phaseComaY);
    addOrientationPair(params.phaseFieldTrefoilRadial * fieldScale,
                       params.phaseFieldTrefoilTangential * fieldScale,
                       angle,
                       3,
                       &resolved.phaseTrefoilX,
                       &resolved.phaseTrefoilY);
    addOrientationPair(params.phaseFieldSecondaryAstigRadial * fieldScale,
                       params.phaseFieldSecondaryAstigTangential * fieldScale,
                       angle,
                       2,
                       &resolved.phaseSecondaryAstigmatism0,
                       &resolved.phaseSecondaryAstigmatism45);
    addOrientationPair(params.phaseFieldQuadrafoilRadial * fieldScale,
                       params.phaseFieldQuadrafoilTangential * fieldScale,
                       angle,
                       4,
                       &resolved.phaseQuadrafoil0,
                       &resolved.phaseQuadrafoil45);
    addOrientationPair(params.phaseFieldSecondaryComaRadial * fieldScale,
                       params.phaseFieldSecondaryComaTangential * fieldScale,
                       angle,
                       1,
                       &resolved.phaseSecondaryComaX,
                       &resolved.phaseSecondaryComaY);
    return resolved;
}

float EvaluateLensDiffPhaseWaves(const LensDiffParams& params, float px, float py) {
    if (!HasLensDiffNonFlatPhase(params)) {
        return 0.0f;
    }

    const float sx = px - static_cast<float>(params.pupilDecenterX);
    const float sy = py - static_cast<float>(params.pupilDecenterY);
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
    const float secondaryAstigRadial = (4.0f * r2 - 3.0f);
    const float secondaryAstig0 = kZernikeSecondaryAstigNorm * secondaryAstigRadial * (sx * sx - sy * sy);
    const float secondaryAstig45 = kZernikeSecondaryAstigNorm * secondaryAstigRadial * (2.0f * sx * sy);
    const float quadrafoil0 = kZernikeQuadrafoilNorm * (sx * sx * sx * sx - 6.0f * sx * sx * sy * sy + sy * sy * sy * sy);
    const float quadrafoil45 = kZernikeQuadrafoilNorm * (4.0f * sx * sy * (sx * sx - sy * sy));
    const float secondaryComaRadial = (10.0f * r2 * r2 - 12.0f * r2 + 3.0f);
    const float secondaryComaX = kZernikeSecondaryComaNorm * secondaryComaRadial * sx;
    const float secondaryComaY = kZernikeSecondaryComaNorm * secondaryComaRadial * sy;

    return static_cast<float>(params.phaseDefocus) * defocus +
           static_cast<float>(params.phaseAstigmatism0) * astig0 +
           static_cast<float>(params.phaseAstigmatism45) * astig45 +
           static_cast<float>(params.phaseComaX) * comaX +
           static_cast<float>(params.phaseComaY) * comaY +
           static_cast<float>(params.phaseSpherical) * spherical +
           static_cast<float>(params.phaseTrefoilX) * trefoilX +
           static_cast<float>(params.phaseTrefoilY) * trefoilY +
           static_cast<float>(params.phaseSecondaryAstigmatism0) * secondaryAstig0 +
           static_cast<float>(params.phaseSecondaryAstigmatism45) * secondaryAstig45 +
           static_cast<float>(params.phaseQuadrafoil0) * quadrafoil0 +
           static_cast<float>(params.phaseQuadrafoil45) * quadrafoil45 +
           static_cast<float>(params.phaseSecondaryComaX) * secondaryComaX +
           static_cast<float>(params.phaseSecondaryComaY) * secondaryComaY;
}

std::vector<float> BuildLensDiffPupilPhaseWaves(const LensDiffParams& params,
                                                int size,
                                                float outerRadius,
                                                double rotationDeg) {
    std::vector<float> phase(static_cast<std::size_t>(size) * size, 0.0f);
    if (!HasLensDiffNonFlatPhase(params) || size <= 0 || outerRadius <= 0.0f) {
        return phase;
    }

    const float rotationRad = static_cast<float>(rotationDeg * kPi / 180.0);
    const float cs = std::cos(-rotationRad);
    const float sn = std::sin(-rotationRad);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const float nx = ((static_cast<float>(x) + 0.5f) / static_cast<float>(size) - 0.5f) * 2.0f;
            const float ny = ((static_cast<float>(y) + 0.5f) / static_cast<float>(size) - 0.5f) * 2.0f;
            const float radius = std::sqrt(nx * nx + ny * ny);
            if (radius > outerRadius) {
                continue;
            }
            const float rx = nx * cs - ny * sn;
            const float ry = nx * sn + ny * cs;
            const float px = rx / outerRadius;
            const float py = ry / outerRadius;
            phase[static_cast<std::size_t>(y) * size + x] = EvaluateLensDiffPhaseWaves(params, px, py);
        }
    }
    return phase;
}

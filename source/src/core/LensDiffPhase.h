#pragma once

#include "LensDiffTypes.h"

#include <vector>

bool HasLensDiffNonFlatPhase(const LensDiffParams& params);
bool HasLensDiffFieldPhase(const LensDiffParams& params);

void DisableLensDiffPhase(LensDiffParams* params);

float EvaluateLensDiffPhaseWaves(const LensDiffParams& params, float px, float py);
LensDiffParams ResolveLensDiffFieldZoneParams(const LensDiffParams& params, float normalizedX, float normalizedY);

std::vector<float> BuildLensDiffPupilPhaseWaves(const LensDiffParams& params,
                                                int size,
                                                float outerRadius,
                                                double rotationDeg);

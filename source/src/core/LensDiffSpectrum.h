#pragma once

#include "LensDiffTypes.h"

#include <array>
#include <vector>

LensDiffSpectrumConfig BuildLensDiffSpectrumConfig(const LensDiffParams& params,
                                                   const std::vector<LensDiffPsfBin>& bins);

std::array<float, 3> MapLensDiffSpectralBins(const std::array<float, kLensDiffMaxSpectralBins>& bins,
                                             const LensDiffParams& params,
                                             const LensDiffSpectrumConfig& config);

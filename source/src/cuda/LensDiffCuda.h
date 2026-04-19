#pragma once

#include "../core/LensDiffTypes.h"

#include <string>

bool RunLensDiffCuda(const LensDiffRenderRequest& request,
                     const LensDiffParams& params,
                     LensDiffPsfBankCache& cache,
                     std::string* error);

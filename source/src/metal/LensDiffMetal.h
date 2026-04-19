#pragma once

#include "../core/LensDiffTypes.h"

#include <string>

bool RunLensDiffMetal(const LensDiffRenderRequest& request,
                      const LensDiffParams& params,
                      LensDiffPsfBankCache& cache,
                      std::string* error);

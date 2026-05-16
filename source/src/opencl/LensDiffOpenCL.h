#pragma once

#include "../core/LensDiffTypes.h"

#include <string>

// Full OpenCL render backend entry point. Uses the host-provided OpenCL command
// queue and renders into host OpenCL buffers/images without CPU image staging.
bool RunLensDiffOpenCL(const LensDiffRenderRequest& request,
                       const LensDiffParams& params,
                       LensDiffPsfBankCache& cache,
                       std::string* error);

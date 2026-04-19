#pragma once

#include "LensDiffTypes.h"

namespace LensDiffTransfer {

WorkshopColor::TransferFunctionId toWorkshopTransfer(LensDiffInputTransfer transfer);

WorkshopColor::Vec3f decodeToLinear(WorkshopColor::Vec3f rgb, LensDiffInputTransfer transfer);
WorkshopColor::Vec3f encodeFromLinear(WorkshopColor::Vec3f rgb, LensDiffInputTransfer transfer);

} // namespace LensDiffTransfer

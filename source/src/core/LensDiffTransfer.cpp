#include "LensDiffTransfer.h"

#include <algorithm>
#include <cmath>

namespace {

float encodeChannelFromLinear(float x, LensDiffInputTransfer transfer) {
    switch (transfer) {
        case LensDiffInputTransfer::DavinciIntermediate: {
            constexpr float kA = 0.0075f;
            constexpr float kB = 7.0f;
            constexpr float kC = 0.07329248f;
            constexpr float kM = 10.44426855f;
            constexpr float kLinCut = 0.00262409f;
            return x <= kLinCut ? (x * kM) : ((std::log2(std::max(x, 0.0f) + kA) + kB) * kC);
        }
        case LensDiffInputTransfer::Linear:
        default:
            return x;
    }
}

} // namespace

namespace LensDiffTransfer {

WorkshopColor::TransferFunctionId toWorkshopTransfer(LensDiffInputTransfer transfer) {
    switch (transfer) {
        case LensDiffInputTransfer::DavinciIntermediate:
            return WorkshopColor::TransferFunctionId::DavinciIntermediate;
        case LensDiffInputTransfer::Linear:
        default:
            return WorkshopColor::TransferFunctionId::Linear;
    }
}

WorkshopColor::Vec3f decodeToLinear(WorkshopColor::Vec3f rgb, LensDiffInputTransfer transfer) {
    return WorkshopColor::decodeToLinear(rgb, toWorkshopTransfer(transfer));
}

WorkshopColor::Vec3f encodeFromLinear(WorkshopColor::Vec3f rgb, LensDiffInputTransfer transfer) {
    return {
        encodeChannelFromLinear(rgb.x, transfer),
        encodeChannelFromLinear(rgb.y, transfer),
        encodeChannelFromLinear(rgb.z, transfer),
    };
}

} // namespace LensDiffTransfer

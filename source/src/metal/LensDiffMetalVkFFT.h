#pragma once

#if defined(__APPLE__)

#import <Metal/Metal.h>

#include <string>

bool lensDiffMetalVkFFTEncodeSquare(id<MTLCommandBuffer> commandBuffer,
                                    id<MTLComputeCommandEncoder> encoder,
                                    id<MTLBuffer> spectrum,
                                    int size,
                                    int imageCount,
                                    bool inverse,
                                    std::string* error);

#endif

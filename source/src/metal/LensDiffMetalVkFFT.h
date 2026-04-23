#pragma once

#if defined(__APPLE__)

#include <string>

@protocol MTLCommandBuffer;
@protocol MTLComputeCommandEncoder;
@protocol MTLBuffer;

bool lensDiffMetalVkFFTEncodeSquare(id<MTLCommandBuffer> commandBuffer,
                                    id<MTLComputeCommandEncoder> encoder,
                                    id<MTLBuffer> spectrum,
                                    int size,
                                    int imageCount,
                                    bool inverse,
                                    std::string* error);

#endif

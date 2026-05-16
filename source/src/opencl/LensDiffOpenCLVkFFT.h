#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>

// Encodes a square 2-D complex-to-complex FFT (forward or inverse) into
// commandQueue using a pooled VkFFT plan.  Mirrors lensDiffMetalVkFFTEncodeSquare
// and lensDiffCudaVkFFTExecC2C in contract:
//
//  - C2C, unnormalized (normalize=0).  The caller applies the 1/paddedArea
//    scale as part of the convolution math, consistent with the Metal and CUDA
//    backends.
//  - useLUT=1 for performance on all devices.
//  - Plans are cached per (device, context, size, imageCount) and pooled so
//    concurrent renders with the same geometry each get their own entry.
//
// commandQueue   - the CL queue to encode into; must remain valid until the
//                  enqueued work completes.
// spectrum       - device buffer of size*size*imageCount complex float pairs
//                  (interleaved real/imag, row-major).
// size           - width and height of the square FFT grid (power-of-two).
// imageCount     - number of planes batched in spectrum.
// inverse        - false = forward (-1 direction), true = inverse (+1 direction).
// error          - receives a diagnostic string on failure; may be nullptr.
bool lensDiffOpenCLVkFFTEncodeSquare(cl_command_queue commandQueue,
                                     cl_mem spectrum,
                                     int size,
                                     int imageCount,
                                     bool inverse,
                                     std::string* error);

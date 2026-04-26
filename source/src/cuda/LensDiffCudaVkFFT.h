#pragma once

#include <cstddef>
#include <string>

#include <cuda_runtime.h>

struct LensDiffCudaVkFFTPlanLease {
    void* repository = nullptr;
    void* entry = nullptr;
    cudaStream_t releaseStream = nullptr;
    bool persistent = false;

    LensDiffCudaVkFFTPlanLease() = default;
    LensDiffCudaVkFFTPlanLease(const LensDiffCudaVkFFTPlanLease&) = delete;
    LensDiffCudaVkFFTPlanLease& operator=(const LensDiffCudaVkFFTPlanLease&) = delete;
    LensDiffCudaVkFFTPlanLease(LensDiffCudaVkFFTPlanLease&& other) noexcept;
    LensDiffCudaVkFFTPlanLease& operator=(LensDiffCudaVkFFTPlanLease&& other) noexcept;
    ~LensDiffCudaVkFFTPlanLease();

    void release();
    std::size_t workBytes() const;
};

bool lensDiffCudaVkFFTAcquirePlan(bool persistentRepository,
                                  int repositoryTag,
                                  int width,
                                  int height,
                                  int batchCount,
                                  cudaStream_t stream,
                                  double* outInitMs,
                                  bool* outCacheHit,
                                  LensDiffCudaVkFFTPlanLease* outPlan,
                                  std::string* error);

bool lensDiffCudaVkFFTExecC2C(LensDiffCudaVkFFTPlanLease* plan,
                              void* buffer,
                              bool inverse,
                              std::string* error);

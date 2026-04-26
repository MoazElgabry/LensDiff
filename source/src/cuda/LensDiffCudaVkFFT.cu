#include "LensDiffCudaVkFFT.h"

#ifndef VKFFT_BACKEND
#define VKFFT_BACKEND 1
#endif

#include <vkFFT.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct RawCudaBuffer {
    void* ptr = nullptr;
    std::size_t bytes = 0;

    RawCudaBuffer() = default;
    RawCudaBuffer(const RawCudaBuffer&) = delete;
    RawCudaBuffer& operator=(const RawCudaBuffer&) = delete;

    RawCudaBuffer(RawCudaBuffer&& other) noexcept
        : ptr(other.ptr), bytes(other.bytes) {
        other.ptr = nullptr;
        other.bytes = 0;
    }

    RawCudaBuffer& operator=(RawCudaBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr = other.ptr;
            bytes = other.bytes;
            other.ptr = nullptr;
            other.bytes = 0;
        }
        return *this;
    }

    ~RawCudaBuffer() { release(); }

    bool allocate(std::size_t newBytes) {
        if (newBytes == 0) {
            release();
            return true;
        }
        if (ptr != nullptr && bytes == newBytes) {
            return true;
        }
        release();
        if (cudaMalloc(&ptr, newBytes) != cudaSuccess) {
            ptr = nullptr;
            bytes = 0;
            return false;
        }
        bytes = newBytes;
        return true;
    }

    void release() {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
            bytes = 0;
        }
    }
};

struct VkFFTPlanKey {
    int deviceId = 0;
    int repositoryTag = 0;
    int width = 0;
    int height = 0;
    int batchCount = 0;
    int useLut = 0;

    bool operator==(const VkFFTPlanKey& other) const {
        return deviceId == other.deviceId &&
               repositoryTag == other.repositoryTag &&
               width == other.width &&
               height == other.height &&
               batchCount == other.batchCount &&
               useLut == other.useLut;
    }
};

struct VkFFTPlanKeyHasher {
    std::size_t operator()(const VkFFTPlanKey& key) const noexcept {
        std::size_t hash = static_cast<std::size_t>(key.deviceId);
        hash = hash * 40503u + static_cast<std::size_t>(key.repositoryTag);
        hash = hash * 1315423911u + static_cast<std::size_t>(key.width);
        hash = hash * 2654435761u + static_cast<std::size_t>(key.height);
        hash = hash * 2246822519u + static_cast<std::size_t>(key.batchCount);
        hash = hash * 3266489917u + static_cast<std::size_t>(key.useLut);
        return hash;
    }
};

struct PersistentVkFFTPlanRepository {
    struct PlanEntry {
        VkFFTApplication app {};
        RawCudaBuffer placeholderBuffer;
        std::size_t placeholderBytes = 0;
        std::size_t tempBytes = 0;
        CUdevice device = 0;
        cudaStream_t stream = nullptr;
        bool initialized = false;
        bool inUse = false;
        std::uint64_t stamp = 0;
        cudaEvent_t readyEvent = nullptr;

        ~PlanEntry() {
            if (initialized) {
                deleteVkFFT(&app);
                initialized = false;
            }
            if (readyEvent != nullptr) {
                cudaEventDestroy(readyEvent);
                readyEvent = nullptr;
            }
        }
    };

    std::mutex mutex;
    std::unordered_map<VkFFTPlanKey, std::vector<std::unique_ptr<PlanEntry>>, VkFFTPlanKeyHasher> entries;
    std::uint64_t nextStamp = 0;
};

PersistentVkFFTPlanRepository& persistentVkFFTPlanRepository() {
    static PersistentVkFFTPlanRepository repository;
    return repository;
}

bool makePlanEntry(int width,
                   int height,
                   int batchCount,
                   cudaStream_t stream,
                   double* outInitMs,
                   std::unique_ptr<PersistentVkFFTPlanRepository::PlanEntry>* outEntry,
                   std::string* error) {
    if (outEntry == nullptr) {
        if (error) {
            *error = "cuda-vkfft-null-entry";
        }
        return false;
    }

    int deviceId = 0;
    if (cudaGetDevice(&deviceId) != cudaSuccess) {
        if (error) {
            *error = "cuda-vkfft-get-device-failed";
        }
        return false;
    }

    auto entry = std::make_unique<PersistentVkFFTPlanRepository::PlanEntry>();
    if (cuDeviceGet(&entry->device, deviceId) != CUDA_SUCCESS) {
        if (error) {
            *error = "cuda-vkfft-cuDeviceGet-failed";
        }
        return false;
    }

    const std::size_t placeholderBytes =
        static_cast<std::size_t>(width) *
        static_cast<std::size_t>(height) *
        static_cast<std::size_t>(batchCount) *
        sizeof(float) * 2u;
    if (!entry->placeholderBuffer.allocate(std::max<std::size_t>(placeholderBytes, 1u))) {
        if (error) {
            *error = "cuda-vkfft-placeholder-buffer-allocation-failed";
        }
        return false;
    }
    entry->placeholderBytes = placeholderBytes;
    entry->stream = stream;

    void* placeholderPtr = entry->placeholderBuffer.ptr;
    pfUINT bufferSize = static_cast<pfUINT>(placeholderBytes);
    pfUINT tempBufferSize = 0;
    VkFFTConfiguration configuration {};
    configuration.FFTdim = 2;
    configuration.size[0] = static_cast<pfUINT>(width);
    configuration.size[1] = static_cast<pfUINT>(height);
    configuration.numberBatches = static_cast<pfUINT>(batchCount);
    configuration.device = &entry->device;
    configuration.stream = &entry->stream;
    configuration.num_streams = 1;
    configuration.buffer = &placeholderPtr;
    configuration.bufferSize = &bufferSize;
    configuration.tempBufferSize = &tempBufferSize;
    configuration.useLUT = 1;
    configuration.performR2C = 0;
    configuration.makeForwardPlanOnly = 0;
    configuration.makeInversePlanOnly = 0;

    const auto initStart = std::chrono::steady_clock::now();
    const VkFFTResult result = initializeVkFFT(&entry->app, configuration);
    const double initMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                              std::chrono::steady_clock::now() - initStart)
                              .count();
    if (outInitMs != nullptr) {
        *outInitMs = initMs;
    }
    if (result != VKFFT_SUCCESS) {
        if (error != nullptr) {
            *error = "cuda-vkfft-init-failed:" + std::string(getVkFFTErrorString(result));
        }
        return false;
    }

    if (cudaEventCreateWithFlags(&entry->readyEvent, cudaEventDisableTiming) != cudaSuccess) {
        if (error != nullptr) {
            *error = "cuda-vkfft-ready-event-create-failed";
        }
        deleteVkFFT(&entry->app);
        entry->initialized = false;
        return false;
    }

    entry->tempBytes = static_cast<std::size_t>(tempBufferSize);
    entry->initialized = true;
    *outEntry = std::move(entry);
    return true;
}

PersistentVkFFTPlanRepository::PlanEntry* entryFromLease(const LensDiffCudaVkFFTPlanLease& lease) {
    return static_cast<PersistentVkFFTPlanRepository::PlanEntry*>(lease.entry);
}

PersistentVkFFTPlanRepository* repositoryFromLease(const LensDiffCudaVkFFTPlanLease& lease) {
    return static_cast<PersistentVkFFTPlanRepository*>(lease.repository);
}

}  // namespace

LensDiffCudaVkFFTPlanLease::LensDiffCudaVkFFTPlanLease(LensDiffCudaVkFFTPlanLease&& other) noexcept
    : repository(other.repository),
      entry(other.entry),
      releaseStream(other.releaseStream),
      persistent(other.persistent) {
    other.repository = nullptr;
    other.entry = nullptr;
    other.releaseStream = nullptr;
    other.persistent = false;
}

LensDiffCudaVkFFTPlanLease& LensDiffCudaVkFFTPlanLease::operator=(LensDiffCudaVkFFTPlanLease&& other) noexcept {
    if (this != &other) {
        release();
        repository = other.repository;
        entry = other.entry;
        releaseStream = other.releaseStream;
        persistent = other.persistent;
        other.repository = nullptr;
        other.entry = nullptr;
        other.releaseStream = nullptr;
        other.persistent = false;
    }
    return *this;
}

LensDiffCudaVkFFTPlanLease::~LensDiffCudaVkFFTPlanLease() {
    release();
}

void LensDiffCudaVkFFTPlanLease::release() {
    if (entry == nullptr) {
        repository = nullptr;
        releaseStream = nullptr;
        persistent = false;
        return;
    }

    auto* planEntry = entryFromLease(*this);
    if (persistent) {
        if (planEntry->readyEvent != nullptr && releaseStream != nullptr) {
            cudaEventRecord(planEntry->readyEvent, releaseStream);
        }
        auto* repositoryPtr = repositoryFromLease(*this);
        if (repositoryPtr != nullptr) {
            std::lock_guard<std::mutex> lock(repositoryPtr->mutex);
            planEntry->inUse = false;
        }
    } else {
        delete planEntry;
    }

    repository = nullptr;
    entry = nullptr;
    releaseStream = nullptr;
    persistent = false;
}

std::size_t LensDiffCudaVkFFTPlanLease::workBytes() const {
    const auto* planEntry = entryFromLease(*this);
    return planEntry != nullptr ? planEntry->tempBytes : 0;
}

bool lensDiffCudaVkFFTAcquirePlan(bool persistentRepository,
                                  int repositoryTag,
                                  int width,
                                  int height,
                                  int batchCount,
                                  cudaStream_t stream,
                                  double* outInitMs,
                                  bool* outCacheHit,
                                  LensDiffCudaVkFFTPlanLease* outPlan,
                                  std::string* error) {
    if (outPlan == nullptr || width <= 0 || height <= 0 || batchCount <= 0) {
        if (error != nullptr) {
            *error = "cuda-vkfft-invalid-plan-request";
        }
        return false;
    }

    outPlan->release();
    if (outCacheHit != nullptr) {
        *outCacheHit = false;
    }
    if (outInitMs != nullptr) {
        *outInitMs = 0.0;
    }

    if (!persistentRepository) {
        std::unique_ptr<PersistentVkFFTPlanRepository::PlanEntry> entry;
        if (!makePlanEntry(width, height, batchCount, stream, outInitMs, &entry, error)) {
            return false;
        }
        outPlan->entry = entry.release();
        outPlan->releaseStream = stream;
        outPlan->persistent = false;
        return true;
    }

    int deviceId = 0;
    if (cudaGetDevice(&deviceId) != cudaSuccess) {
        if (error != nullptr) {
            *error = "cuda-vkfft-get-device-failed";
        }
        return false;
    }

    PersistentVkFFTPlanRepository& repository = persistentVkFFTPlanRepository();
    const VkFFTPlanKey key {deviceId, repositoryTag, width, height, batchCount, 1};
    {
        std::lock_guard<std::mutex> lock(repository.mutex);
        auto& pool = repository.entries[key];
        for (auto& entryPtr : pool) {
            if (entryPtr != nullptr && !entryPtr->inUse) {
                entryPtr->inUse = true;
                entryPtr->stamp = ++repository.nextStamp;
                entryPtr->stream = stream;
                if (entryPtr->readyEvent != nullptr &&
                    cudaStreamWaitEvent(stream, entryPtr->readyEvent, 0) != cudaSuccess) {
                    entryPtr->inUse = false;
                    if (error != nullptr) {
                        *error = "cuda-vkfft-stream-wait-failed";
                    }
                    return false;
                }
                outPlan->repository = &repository;
                outPlan->entry = entryPtr.get();
                outPlan->releaseStream = stream;
                outPlan->persistent = true;
                if (outCacheHit != nullptr) {
                    *outCacheHit = true;
                }
                return true;
            }
        }
    }

    std::unique_ptr<PersistentVkFFTPlanRepository::PlanEntry> entry;
    if (!makePlanEntry(width, height, batchCount, stream, outInitMs, &entry, error)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(repository.mutex);
    auto& pool = repository.entries[key];
    entry->inUse = true;
    entry->stamp = ++repository.nextStamp;
    auto* entryRaw = entry.get();
    pool.push_back(std::move(entry));
    outPlan->repository = &repository;
    outPlan->entry = entryRaw;
    outPlan->releaseStream = stream;
    outPlan->persistent = true;
    return true;
}

bool lensDiffCudaVkFFTExecC2C(LensDiffCudaVkFFTPlanLease* plan,
                              void* buffer,
                              bool inverse,
                              std::string* error) {
    if (plan == nullptr || plan->entry == nullptr || buffer == nullptr) {
        if (error != nullptr) {
            *error = "cuda-vkfft-invalid-execute";
        }
        return false;
    }

    auto* planEntry = entryFromLease(*plan);
    void* launchBuffer = buffer;
    VkFFTLaunchParams launchParams {};
    launchParams.buffer = &launchBuffer;
    const VkFFTResult result = VkFFTAppend(&planEntry->app, inverse ? 1 : -1, &launchParams);
    if (result != VKFFT_SUCCESS) {
        if (error != nullptr) {
            *error = "cuda-vkfft-append-failed:" + std::string(getVkFFTErrorString(result));
        }
        return false;
    }
    return true;
}

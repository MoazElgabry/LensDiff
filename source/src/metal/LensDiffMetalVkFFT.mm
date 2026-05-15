#if defined(__APPLE__)

#include "LensDiffMetalVkFFT.h"

#ifndef VKFFT_BACKEND
#define VKFFT_BACKEND 5
#endif

#include "../../external/VkFFT/vkFFT/vkFFT.h"

#include <dispatch/dispatch.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct VkFFTPlanKey {
    std::uintptr_t device = 0;
    int size = 0;
    int imageCount = 0;

    bool operator==(const VkFFTPlanKey& other) const {
        return device == other.device &&
               size == other.size &&
               imageCount == other.imageCount;
    }
};

struct VkFFTPlanKeyHasher {
    std::size_t operator()(const VkFFTPlanKey& key) const noexcept {
        std::size_t hash = key.device;
        hash = hash * 2654435761u + static_cast<std::size_t>(key.size);
        hash = hash * 2246822519u + static_cast<std::size_t>(key.imageCount);
        return hash;
    }
};

struct CachedVkFFTPlan {
    VkFFTApplication app {};
    MTL::Buffer* configBuffer = nullptr;
    pfUINT configBufferSize = 0;
    // Guards one GPU use at a time. acquirePlan takes this before returning;
    // the caller releases it in the GPU completion handler. The pool grows on
    // demand so concurrent renders with the same geometry each get their own
    // entry and never have to wait on each other.
    dispatch_semaphore_t gpuSemaphore = dispatch_semaphore_create(1);

    ~CachedVkFFTPlan() {
        deleteVkFFT(&app);
        if (configBuffer != nullptr) {
            configBuffer->release();
            configBuffer = nullptr;
        }
    }
};

std::mutex gVkFFTPlanMutex;
// Pool per key: grows when concurrent renders need the same geometry simultaneously.
std::unordered_map<VkFFTPlanKey,
                   std::vector<std::shared_ptr<CachedVkFFTPlan>>,
                   VkFFTPlanKeyHasher> gVkFFTPlans;

std::string vkfftResultText(VkFFTResult result) {
    return std::string(getVkFFTErrorString(result));
}

// Creates and initializes a new VkFFT plan. Called outside the global mutex
// because initializeVkFFT can take ~100 ms and must not stall other threads.
std::shared_ptr<CachedVkFFTPlan> makePlan(MTL::Device* deviceCpp,
                                          MTL::CommandQueue* queueCpp,
                                          int size,
                                          int imageCount,
                                          std::string* error) {
    const NSUInteger bufferBytes = static_cast<NSUInteger>(size) *
                                   static_cast<NSUInteger>(size) *
                                   static_cast<NSUInteger>(imageCount) *
                                   sizeof(float) * 2u;

    std::shared_ptr<CachedVkFFTPlan> plan = std::make_shared<CachedVkFFTPlan>();
    plan->configBuffer = deviceCpp->newBuffer(bufferBytes, MTL::ResourceStorageModeShared);
    if (plan->configBuffer == nullptr) {
        if (error != nullptr) {
            *error = "metal-vkfft-placeholder-buffer-allocation-failed";
        }
        return nullptr;
    }
    plan->configBufferSize = static_cast<pfUINT>(bufferBytes);

    VkFFTConfiguration configuration {};
    configuration.FFTdim = 2;
    configuration.size[0] = static_cast<pfUINT>(size);
    configuration.size[1] = static_cast<pfUINT>(size);
    configuration.numberBatches = static_cast<pfUINT>(imageCount);
    configuration.normalize = 0;
    configuration.device = deviceCpp;
    configuration.queue = queueCpp;
    configuration.buffer = &plan->configBuffer;
    configuration.bufferSize = &plan->configBufferSize;
    configuration.useLUT = 1;
    configuration.performR2C = 0;
    configuration.makeForwardPlanOnly = 0;
    configuration.makeInversePlanOnly = 0;

    const VkFFTResult result = initializeVkFFT(&plan->app, configuration);
    if (result != VKFFT_SUCCESS) {
        if (error != nullptr) {
            *error = "metal-vkfft-init-failed:" + vkfftResultText(result);
        }
        return nullptr;
    }
    return plan;
}

// Returns a plan for the given geometry with gpuSemaphore already acquired.
// Scans the pool for a free entry first; if all entries are in flight, creates
// a new one so the caller never blocks waiting for a concurrent render to finish.
bool acquirePlan(id<MTLCommandBuffer> commandBuffer,
                 int size,
                 int imageCount,
                 std::shared_ptr<CachedVkFFTPlan>* outPlan,
                 std::string* error) {
    if (commandBuffer == nil || outPlan == nullptr || size <= 0 || imageCount <= 0) {
        if (error != nullptr) {
            *error = "metal-vkfft-invalid-init";
        }
        return false;
    }

    MTL::CommandBuffer* commandBufferCpp = (__bridge MTL::CommandBuffer*)commandBuffer;
    MTL::CommandQueue* queueCpp = commandBufferCpp != nullptr ? commandBufferCpp->commandQueue() : nullptr;
    MTL::Device* deviceCpp = commandBufferCpp != nullptr ? commandBufferCpp->device() : nullptr;
    if (queueCpp == nullptr || deviceCpp == nullptr) {
        if (error != nullptr) {
            *error = "metal-vkfft-missing-device-or-queue";
        }
        return false;
    }

    const VkFFTPlanKey key {
        reinterpret_cast<std::uintptr_t>(deviceCpp),
        size,
        imageCount
    };

    // Scan the pool for a free entry using a non-blocking semaphore try.
    // The pool mutex protects the vector; the semaphore itself atomically
    // prevents double-acquisition between threads.
    {
        std::lock_guard<std::mutex> lock(gVkFFTPlanMutex);
        auto it = gVkFFTPlans.find(key);
        if (it != gVkFFTPlans.end()) {
            for (const auto& candidate : it->second) {
                if (dispatch_semaphore_wait(candidate->gpuSemaphore, DISPATCH_TIME_NOW) == 0) {
                    *outPlan = candidate;
                    return true;
                }
            }
        }
    }

    // Every existing plan for this geometry is in flight. Build a new one.
    // makePlan runs outside the mutex to avoid stalling other threads during init.
    std::shared_ptr<CachedVkFFTPlan> plan = makePlan(deviceCpp, queueCpp, size, imageCount, error);
    if (!plan) {
        return false;
    }

    // Acquire the semaphore before publishing so no other thread can claim
    // this plan between insertion and our return. Succeeds immediately because
    // the semaphore is initialized to 1 and no other thread knows about the
    // plan yet.
    dispatch_semaphore_wait(plan->gpuSemaphore, DISPATCH_TIME_FOREVER);

    {
        std::lock_guard<std::mutex> lock(gVkFFTPlanMutex);
        gVkFFTPlans[key].push_back(plan);
    }
    *outPlan = plan;
    return true;
}

}  // namespace

bool lensDiffMetalVkFFTEncodeSquare(id<MTLCommandBuffer> commandBuffer,
                                    id<MTLComputeCommandEncoder> encoder,
                                    id<MTLBuffer> spectrum,
                                    int size,
                                    int imageCount,
                                    bool inverse,
                                    std::string* error) {
    if (commandBuffer == nil || encoder == nil || spectrum == nil || size <= 0 || imageCount <= 0) {
        if (error != nullptr) {
            *error = "metal-vkfft-invalid-execute";
        }
        return false;
    }

    // acquirePlan returns with gpuSemaphore already held.
    std::shared_ptr<CachedVkFFTPlan> plan;
    if (!acquirePlan(commandBuffer, size, imageCount, &plan, error)) {
        return false;
    }

    MTL::CommandBuffer* commandBufferCpp = (__bridge MTL::CommandBuffer*)commandBuffer;
    MTL::ComputeCommandEncoder* encoderCpp = (__bridge MTL::ComputeCommandEncoder*)encoder;
    MTL::Buffer* spectrumBuffer = (__bridge MTL::Buffer*)spectrum;
    if (commandBufferCpp == nullptr || encoderCpp == nullptr || spectrumBuffer == nullptr) {
        dispatch_semaphore_signal(plan->gpuSemaphore);
        if (error != nullptr) {
            *error = "metal-vkfft-invalid-metal-cpp-bridge";
        }
        return false;
    }

    VkFFTLaunchParams launchParams {};
    launchParams.commandBuffer = commandBufferCpp;
    launchParams.commandEncoder = encoderCpp;
    launchParams.buffer = &spectrumBuffer;

    // gpuSemaphore is already held by acquirePlan — no additional wait here.
    const int direction = inverse ? 1 : -1;
    const VkFFTResult result = VkFFTAppend(&plan->app, direction, &launchParams);
    if (result != VKFFT_SUCCESS) {
        dispatch_semaphore_signal(plan->gpuSemaphore);
        if (error != nullptr) {
            *error = "metal-vkfft-append-failed:" + vkfftResultText(result);
        }
        return false;
    }

    // Release the GPU slot when this command buffer finishes on the GPU.
    dispatch_semaphore_t semaphore = plan->gpuSemaphore;
    commandBufferCpp->addCompletedHandler(^(MTL::CommandBuffer*) {
        dispatch_semaphore_signal(semaphore);
    });
    return true;
}

#endif

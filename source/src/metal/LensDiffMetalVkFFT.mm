#if defined(__APPLE__)

#include "LensDiffMetalVkFFT.h"

#ifndef VKFFT_BACKEND
#define VKFFT_BACKEND 5
#endif

#include "../../external/VkFFT/vkFFT/vkFFT.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

struct VkFFTPlanKey {
    std::uintptr_t device = 0;
    std::uintptr_t queue = 0;
    int size = 0;
    int imageCount = 0;

    bool operator==(const VkFFTPlanKey& other) const {
        return device == other.device &&
               queue == other.queue &&
               size == other.size &&
               imageCount == other.imageCount;
    }
};

struct VkFFTPlanKeyHasher {
    std::size_t operator()(const VkFFTPlanKey& key) const noexcept {
        std::size_t hash = key.device;
        hash = hash * 1315423911u + key.queue;
        hash = hash * 2654435761u + static_cast<std::size_t>(key.size);
        hash = hash * 2246822519u + static_cast<std::size_t>(key.imageCount);
        return hash;
    }
};

struct CachedVkFFTPlan {
    VkFFTApplication app {};
    MTL::Buffer* configBuffer = nullptr;
    pfUINT configBufferSize = 0;
    std::mutex mutex;

    ~CachedVkFFTPlan() {
        deleteVkFFT(&app);
        if (configBuffer != nullptr) {
            configBuffer->release();
            configBuffer = nullptr;
        }
    }
};

std::mutex gVkFFTPlanMutex;
std::unordered_map<VkFFTPlanKey, std::shared_ptr<CachedVkFFTPlan>, VkFFTPlanKeyHasher> gVkFFTPlans;

std::string vkfftResultText(VkFFTResult result) {
    return std::string(getVkFFTErrorString(result));
}

bool initializePlan(id<MTLCommandBuffer> commandBuffer,
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
        reinterpret_cast<std::uintptr_t>(queueCpp),
        size,
        imageCount
    };

    {
        std::lock_guard<std::mutex> lock(gVkFFTPlanMutex);
        auto it = gVkFFTPlans.find(key);
        if (it != gVkFFTPlans.end()) {
            *outPlan = it->second;
            return true;
        }
    }

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
        return false;
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
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(gVkFFTPlanMutex);
        auto [it, inserted] = gVkFFTPlans.emplace(key, plan);
        *outPlan = inserted ? plan : it->second;
    }
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

    std::shared_ptr<CachedVkFFTPlan> plan;
    if (!initializePlan(commandBuffer, size, imageCount, &plan, error)) {
        return false;
    }

    MTL::Buffer* spectrumBuffer = (__bridge MTL::Buffer*)spectrum;
    VkFFTLaunchParams launchParams {};
    launchParams.commandBuffer = (__bridge MTL::CommandBuffer*)commandBuffer;
    launchParams.commandEncoder = (__bridge MTL::ComputeCommandEncoder*)encoder;
    launchParams.buffer = &spectrumBuffer;

    const int direction = inverse ? 1 : -1;
    std::lock_guard<std::mutex> lock(plan->mutex);
    const VkFFTResult result = VkFFTAppend(&plan->app, direction, &launchParams);
    if (result != VKFFT_SUCCESS) {
        if (error != nullptr) {
            *error = "metal-vkfft-append-failed:" + vkfftResultText(result);
        }
        return false;
    }
    return true;
}

#endif

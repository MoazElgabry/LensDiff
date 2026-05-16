#include "LensDiffOpenCLVkFFT.h"

#ifndef VKFFT_BACKEND
#define VKFFT_BACKEND 3
#endif

#include <vkFFT.h>

#include <cstdio>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Plan guard: cross-platform equivalent of Metal's dispatch_semaphore_t(1).
// tryAcquire() is non-blocking; release() is called from the CL event callback.
// ---------------------------------------------------------------------------

struct PlanGuard {
    std::mutex mutex;
    bool inUse = false;

    // Non-blocking: returns true and marks inUse if the plan is free.
    bool tryAcquire() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (inUse) {
            return false;
        }
        inUse = true;
        return true;
    }

    // Called from the GPU completion callback or on error cleanup.
    void release() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex);
            inUse = false;
        }
    }
};

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

struct CachedVkFFTPlan {
    VkFFTApplication app {};
    cl_mem configBuffer = nullptr;
    pfUINT configBufferBytes = 0;
    cl_context context = nullptr; // retained; used in destructor
    // Guards one GPU use at a time. acquirePlan takes this before returning;
    // the GPU completion callback releases it so the pool entry becomes
    // available for the next render.
    PlanGuard guard;

    ~CachedVkFFTPlan() {
        deleteVkFFT(&app);
        if (configBuffer != nullptr) {
            clReleaseMemObject(configBuffer);
            configBuffer = nullptr;
        }
        if (context != nullptr) {
            clReleaseContext(context);
            context = nullptr;
        }
    }
};

struct PlanCallbackState {
    std::shared_ptr<CachedVkFFTPlan> plan;
};

void CL_CALLBACK releasePlanAfterEvent(cl_event, cl_int, void* userData) {
    std::unique_ptr<PlanCallbackState> state(static_cast<PlanCallbackState*>(userData));
    if (state && state->plan) {
        state->plan->guard.release();
    }
}

// ---------------------------------------------------------------------------
// Plan pool
// ---------------------------------------------------------------------------

struct VkFFTPlanKey {
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    int size = 0;
    int imageCount = 0;

    bool operator==(const VkFFTPlanKey& other) const noexcept {
        return device == other.device &&
               context == other.context &&
               size == other.size &&
               imageCount == other.imageCount;
    }
};

struct VkFFTPlanKeyHasher {
    std::size_t operator()(const VkFFTPlanKey& key) const noexcept {
        auto h = [](std::size_t seed, std::size_t v) noexcept {
            return seed * 2654435761u ^ v;
        };
        std::size_t hash = reinterpret_cast<std::uintptr_t>(key.device);
        hash = h(hash, reinterpret_cast<std::uintptr_t>(key.context));
        hash = h(hash, static_cast<std::size_t>(key.size));
        hash = h(hash, static_cast<std::size_t>(key.imageCount));
        return hash;
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

std::string clErrorText(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                          return "CL_SUCCESS";
        case CL_INVALID_COMMAND_QUEUE:            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_CONTEXT:                  return "CL_INVALID_CONTEXT";
        case CL_INVALID_MEM_OBJECT:               return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_VALUE:                    return "CL_INVALID_VALUE";
        case CL_OUT_OF_RESOURCES:                 return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:               return "CL_OUT_OF_HOST_MEMORY";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        default: {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "CL_ERROR_%d", static_cast<int>(err));
            return buf;
        }
    }
}

// ---------------------------------------------------------------------------
// Plan creation: runs outside the pool mutex because initializeVkFFT ~100 ms.
// ---------------------------------------------------------------------------

std::shared_ptr<CachedVkFFTPlan> makePlan(cl_device_id device,
                                          cl_context context,
                                          int size,
                                          int imageCount,
                                          std::string* error) {
    const std::size_t bufferBytes =
        static_cast<std::size_t>(size) *
        static_cast<std::size_t>(size) *
        static_cast<std::size_t>(imageCount) *
        sizeof(float) * 2u;

    auto plan = std::make_shared<CachedVkFFTPlan>();

    // Retain context so the plan destructor can safely release it.
    clRetainContext(context);
    plan->context = context;

    cl_int clErr = CL_SUCCESS;
    plan->configBuffer = clCreateBuffer(context,
                                        CL_MEM_READ_WRITE,
                                        bufferBytes,
                                        nullptr,
                                        &clErr);
    if (clErr != CL_SUCCESS || plan->configBuffer == nullptr) {
        if (error != nullptr) {
            *error = "opencl-vkfft-placeholder-buffer-allocation-failed:" + clErrorText(clErr);
        }
        return nullptr;
    }
    plan->configBufferBytes = static_cast<pfUINT>(bufferBytes);

    VkFFTConfiguration configuration {};
    configuration.FFTdim = 2;
    configuration.size[0] = static_cast<pfUINT>(size);
    configuration.size[1] = static_cast<pfUINT>(size);
    configuration.numberBatches = static_cast<pfUINT>(imageCount);
    configuration.normalize = 0;
    configuration.device = &device;
    configuration.context = &context;
    configuration.buffer = &plan->configBuffer;
    configuration.bufferSize = &plan->configBufferBytes;
    configuration.useLUT = 1;
    configuration.performR2C = 0;
    configuration.makeForwardPlanOnly = 0;
    configuration.makeInversePlanOnly = 0;

    const VkFFTResult result = initializeVkFFT(&plan->app, configuration);
    if (result != VKFFT_SUCCESS) {
        if (error != nullptr) {
            *error = "opencl-vkfft-init-failed:" + vkfftResultText(result);
        }
        return nullptr;
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Pool acquisition: returns with plan->guard already acquired.
// ---------------------------------------------------------------------------

bool acquirePlan(cl_command_queue commandQueue,
                 int size,
                 int imageCount,
                 std::shared_ptr<CachedVkFFTPlan>* outPlan,
                 std::string* error) {
    if (commandQueue == nullptr || outPlan == nullptr || size <= 0 || imageCount <= 0) {
        if (error != nullptr) {
            *error = "opencl-vkfft-invalid-init";
        }
        return false;
    }

    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_int clErr = clGetCommandQueueInfo(commandQueue,
                                         CL_QUEUE_DEVICE,
                                         sizeof(device),
                                         &device,
                                         nullptr);
    if (clErr != CL_SUCCESS || device == nullptr) {
        if (error != nullptr) {
            *error = "opencl-vkfft-get-device-failed:" + clErrorText(clErr);
        }
        return false;
    }
    clErr = clGetCommandQueueInfo(commandQueue,
                                   CL_QUEUE_CONTEXT,
                                   sizeof(context),
                                   &context,
                                   nullptr);
    if (clErr != CL_SUCCESS || context == nullptr) {
        if (error != nullptr) {
            *error = "opencl-vkfft-get-context-failed:" + clErrorText(clErr);
        }
        return false;
    }

    const VkFFTPlanKey key { device, context, size, imageCount };

    // Scan the pool for a free entry using a non-blocking tryAcquire.
    {
        std::lock_guard<std::mutex> lock(gVkFFTPlanMutex);
        auto it = gVkFFTPlans.find(key);
        if (it != gVkFFTPlans.end()) {
            for (const auto& candidate : it->second) {
                if (candidate->guard.tryAcquire()) {
                    *outPlan = candidate;
                    return true;
                }
            }
        }
    }

    // Every existing plan for this geometry is in flight. Build a new one.
    // makePlan runs outside the mutex to avoid stalling other threads during init.
    std::shared_ptr<CachedVkFFTPlan> plan = makePlan(device, context, size, imageCount, error);
    if (!plan) {
        return false;
    }

    // Acquire before publishing so no other thread can claim this plan between
    // insertion and our return. tryAcquire always succeeds here because no
    // other thread knows about the plan yet.
    const bool acquired = plan->guard.tryAcquire();
    (void)acquired; // always true for a brand-new plan

    {
        std::lock_guard<std::mutex> lock(gVkFFTPlanMutex);
        gVkFFTPlans[key].push_back(plan);
    }
    *outPlan = plan;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

bool lensDiffOpenCLVkFFTEncodeSquare(cl_command_queue commandQueue,
                                     cl_mem spectrum,
                                     int size,
                                     int imageCount,
                                     bool inverse,
                                     std::string* error) {
    if (commandQueue == nullptr || spectrum == nullptr || size <= 0 || imageCount <= 0) {
        if (error != nullptr) {
            *error = "opencl-vkfft-invalid-execute";
        }
        return false;
    }

    // acquirePlan returns with plan->guard already acquired.
    std::shared_ptr<CachedVkFFTPlan> plan;
    if (!acquirePlan(commandQueue, size, imageCount, &plan, error)) {
        return false;
    }

    VkFFTLaunchParams launchParams {};
    launchParams.commandQueue = &commandQueue;
    launchParams.buffer = &spectrum;

    // plan->guard is already held; no additional acquire needed.
    const int direction = inverse ? 1 : -1;
    const VkFFTResult result = VkFFTAppend(&plan->app, direction, &launchParams);
    if (result != VKFFT_SUCCESS) {
        plan->guard.release();
        if (error != nullptr) {
            *error = "opencl-vkfft-append-failed:" + vkfftResultText(result);
        }
        return false;
    }

    cl_command_queue_properties queueProperties = 0;
    const cl_int propsErr = clGetCommandQueueInfo(commandQueue,
                                                  CL_QUEUE_PROPERTIES,
                                                  sizeof(queueProperties),
                                                  &queueProperties,
                                                  nullptr);
    if (propsErr == CL_SUCCESS && (queueProperties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) == 0) {
        cl_event marker = nullptr;
        const cl_int markerErr = clEnqueueMarkerWithWaitList(commandQueue, 0, nullptr, &marker);
        if (markerErr != CL_SUCCESS || marker == nullptr) {
            plan->guard.release();
            if (error != nullptr) {
                *error = "opencl-vkfft-marker-failed:" + clErrorText(markerErr);
            }
            return false;
        }
        auto* callbackState = new (std::nothrow) PlanCallbackState { plan };
        if (callbackState == nullptr) {
            clReleaseEvent(marker);
            const cl_int finishErr = clFinish(commandQueue);
            plan->guard.release();
            if (finishErr != CL_SUCCESS) {
                if (error != nullptr) {
                    *error = "opencl-vkfft-finish-after-callback-alloc-failed:" + clErrorText(finishErr);
                }
                return false;
            }
            if (error != nullptr) {
                *error = "opencl-vkfft-callback-state-allocation-failed";
            }
            return false;
        }
        const cl_int callbackErr = clSetEventCallback(marker, CL_COMPLETE, releasePlanAfterEvent, callbackState);
        clReleaseEvent(marker);
        if (callbackErr != CL_SUCCESS) {
            delete callbackState;
            const cl_int finishErr = clFinish(commandQueue);
            plan->guard.release();
            if (finishErr != CL_SUCCESS) {
                if (error != nullptr) {
                    *error = "opencl-vkfft-finish-after-callback-failed:" + clErrorText(finishErr);
                }
                return false;
            }
            if (error != nullptr) {
                *error = "opencl-vkfft-callback-failed:" + clErrorText(callbackErr);
            }
            return false;
        }
        return true;
    }

    const cl_int finishErr = clFinish(commandQueue);
    plan->guard.release();
    if (finishErr != CL_SUCCESS) {
        if (error != nullptr) {
            *error = "opencl-vkfft-finish-failed:" + clErrorText(finishErr);
        }
        return false;
    }

    return true;
}

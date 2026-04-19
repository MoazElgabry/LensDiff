#include "LensDiffApertureImage.h"
#include "LensDiffDiagnostics.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincodec.h>
#endif

#if defined(__APPLE__)
#include <ApplicationServices/ApplicationServices.h>
#include <ImageIO/ImageIO.h>
#endif

namespace {

struct CachedApertureImage {
    bool hasTimestamp = false;
    std::filesystem::file_time_type timestamp {};
    LensDiffApertureImage image {};
};

std::mutex& apertureImageCacheMutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<std::string, CachedApertureImage>& apertureImageCache() {
    static std::map<std::string, CachedApertureImage> cache;
    return cache;
}

bool getApertureTimestamp(const std::string& path,
                          std::filesystem::file_time_type* outTimestamp) {
    if (outTimestamp == nullptr) {
        return false;
    }
    std::error_code error;
    const auto timestamp = std::filesystem::last_write_time(path, error);
    if (error) {
        return false;
    }
    *outTimestamp = timestamp;
    return true;
}

float rgbaToAmplitude(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) {
    const float rf = static_cast<float>(r) / 255.0f;
    const float gf = static_cast<float>(g) / 255.0f;
    const float bf = static_cast<float>(b) / 255.0f;
    const float af = static_cast<float>(a) / 255.0f;
    const float luma = 0.2126f * rf + 0.7152f * gf + 0.0722f * bf;
    return std::clamp(luma * af, 0.0f, 1.0f);
}

#if defined(_WIN32)

template <typename T>
void safeRelease(T*& ptr) {
    if (ptr != nullptr) {
        ptr->Release();
        ptr = nullptr;
    }
}

std::wstring utf8ToWide(const std::string& text) {
    if (text.empty()) {
        return {};
    }
    const int length = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (length <= 0) {
        return {};
    }
    std::wstring wide(static_cast<std::size_t>(length - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), length);
    return wide;
}

bool loadWithWic(const std::string& path,
                 LensDiffApertureImage* outImage,
                 std::string* error) {
    const HRESULT initHr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool shouldUninitialize = initHr == S_OK || initHr == S_FALSE;
    if (FAILED(initHr) && initHr != RPC_E_CHANGED_MODE) {
        if (error) {
            *error = "wic-com-init-failed";
        }
        return false;
    }

    IWICImagingFactory* factory = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICFormatConverter* converter = nullptr;

    bool ok = false;
    do {
        const std::wstring widePath = utf8ToWide(path);
        if (widePath.empty()) {
            if (error) {
                *error = "wic-invalid-utf8-path";
            }
            break;
        }

        HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory,
                                      nullptr,
                                      CLSCTX_INPROC_SERVER,
                                      IID_PPV_ARGS(&factory));
        if (FAILED(hr) || factory == nullptr) {
            if (error) {
                *error = "wic-factory-create-failed";
            }
            break;
        }

        hr = factory->CreateDecoderFromFilename(widePath.c_str(),
                                                nullptr,
                                                GENERIC_READ,
                                                WICDecodeMetadataCacheOnDemand,
                                                &decoder);
        if (FAILED(hr) || decoder == nullptr) {
            if (error) {
                *error = "wic-open-failed";
            }
            break;
        }

        hr = decoder->GetFrame(0, &frame);
        if (FAILED(hr) || frame == nullptr) {
            if (error) {
                *error = "wic-frame-read-failed";
            }
            break;
        }

        hr = factory->CreateFormatConverter(&converter);
        if (FAILED(hr) || converter == nullptr) {
            if (error) {
                *error = "wic-converter-create-failed";
            }
            break;
        }

        hr = converter->Initialize(frame,
                                   GUID_WICPixelFormat32bppRGBA,
                                   WICBitmapDitherTypeNone,
                                   nullptr,
                                   0.0,
                                   WICBitmapPaletteTypeCustom);
        if (FAILED(hr)) {
            if (error) {
                *error = "wic-converter-init-failed";
            }
            break;
        }

        UINT width = 0;
        UINT height = 0;
        hr = converter->GetSize(&width, &height);
        if (FAILED(hr) || width == 0 || height == 0) {
            if (error) {
                *error = "wic-invalid-image-size";
            }
            break;
        }

        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 4U, 0);
        hr = converter->CopyPixels(nullptr,
                                   width * 4U,
                                   static_cast<UINT>(pixels.size()),
                                   pixels.data());
        if (FAILED(hr)) {
            if (error) {
                *error = "wic-copy-pixels-failed";
            }
            break;
        }

        outImage->width = static_cast<int>(width);
        outImage->height = static_cast<int>(height);
        outImage->values.assign(static_cast<std::size_t>(width) * height, 0.0f);
        for (std::size_t i = 0, j = 0; j < outImage->values.size(); i += 4, ++j) {
            outImage->values[j] = rgbaToAmplitude(pixels[i + 0], pixels[i + 1], pixels[i + 2], pixels[i + 3]);
        }

        ok = true;
    } while (false);

    safeRelease(converter);
    safeRelease(frame);
    safeRelease(decoder);
    safeRelease(factory);

    if (shouldUninitialize) {
        CoUninitialize();
    }
    return ok;
}

#endif

#if defined(__APPLE__)

bool loadWithImageIo(const std::string& path,
                     LensDiffApertureImage* outImage,
                     std::string* error) {
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault,
        reinterpret_cast<const UInt8*>(path.c_str()),
        static_cast<CFIndex>(path.size()),
        false);
    if (url == nullptr) {
        if (error) {
            *error = "imageio-invalid-path";
        }
        return false;
    }

    CGImageSourceRef source = CGImageSourceCreateWithURL(url, nullptr);
    if (source == nullptr) {
        CFRelease(url);
        if (error) {
            *error = "imageio-open-failed";
        }
        return false;
    }

    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    if (image == nullptr) {
        CFRelease(source);
        CFRelease(url);
        if (error) {
            *error = "imageio-frame-read-failed";
        }
        return false;
    }

    const std::size_t width = CGImageGetWidth(image);
    const std::size_t height = CGImageGetHeight(image);
    if (width == 0 || height == 0) {
        CGImageRelease(image);
        CFRelease(source);
        CFRelease(url);
        if (error) {
            *error = "imageio-invalid-image-size";
        }
        return false;
    }

    std::vector<std::uint8_t> pixels(width * height * 4U, 0);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(pixels.data(),
                                                 width,
                                                 height,
                                                 8,
                                                 width * 4U,
                                                 colorSpace,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(colorSpace);
    if (context == nullptr) {
        CGImageRelease(image);
        CFRelease(source);
        CFRelease(url);
        if (error) {
            *error = "imageio-context-create-failed";
        }
        return false;
    }

    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);

    outImage->width = static_cast<int>(width);
    outImage->height = static_cast<int>(height);
    outImage->values.assign(width * height, 0.0f);
    for (std::size_t i = 0, j = 0; j < outImage->values.size(); i += 4, ++j) {
        outImage->values[j] = rgbaToAmplitude(pixels[i + 0], pixels[i + 1], pixels[i + 2], pixels[i + 3]);
    }

    CGContextRelease(context);
    CGImageRelease(image);
    CFRelease(source);
    CFRelease(url);
    return true;
}

#endif

} // namespace

bool LoadLensDiffApertureImage(const std::string& path,
                               LensDiffApertureImage* outImage,
                               std::string* error) {
    LensDiffScopedTimer timer("custom-aperture-load");
    if (outImage == nullptr) {
        if (error) {
            *error = "aperture-image-output-null";
        }
        return false;
    }

    *outImage = LensDiffApertureImage {};
    if (path.empty()) {
        if (error) {
            *error = "aperture-image-path-empty";
        }
        return false;
    }

    std::filesystem::file_time_type timestamp {};
    const bool hasTimestamp = getApertureTimestamp(path, &timestamp);
    {
        std::lock_guard<std::mutex> lock(apertureImageCacheMutex());
        auto& cache = apertureImageCache();
        const auto it = cache.find(path);
        if (it != cache.end()) {
            const bool timestampMatches =
                (!hasTimestamp && !it->second.hasTimestamp) ||
                (hasTimestamp && it->second.hasTimestamp && it->second.timestamp == timestamp);
            if (timestampMatches) {
                *outImage = it->second.image;
                return true;
            }
        }
    }

    LensDiffApertureImage loadedImage {};
    bool ok = false;

#if defined(_WIN32)
    ok = loadWithWic(path, &loadedImage, error);
#elif defined(__APPLE__)
    ok = loadWithImageIo(path, &loadedImage, error);
#else
    if (error) {
        *error = "aperture-image-loading-not-supported-on-this-platform";
    }
    return false;
#endif

    if (!ok) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(apertureImageCacheMutex());
        CachedApertureImage& cached = apertureImageCache()[path];
        cached.hasTimestamp = hasTimestamp;
        cached.timestamp = timestamp;
        cached.image = loadedImage;
    }

    *outImage = std::move(loadedImage);
    return true;
}

bool LoadLensDiffPreparedApertureImage(const std::string& path,
                                       bool normalize,
                                       bool invert,
                                       LensDiffApertureImage* outImage,
                                       std::string* error) {
    if (!LoadLensDiffApertureImage(path, outImage, error)) {
        return false;
    }
    if (outImage == nullptr || outImage->values.empty()) {
        return true;
    }

    float minValue = outImage->values.front();
    float maxValue = outImage->values.front();
    for (float value : outImage->values) {
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }

    if (normalize && maxValue > minValue) {
        const float scale = 1.0f / (maxValue - minValue);
        for (float& value : outImage->values) {
            value = std::clamp((value - minValue) * scale, 0.0f, 1.0f);
        }
    }

    if (invert) {
        for (float& value : outImage->values) {
            value = 1.0f - value;
        }
    }
    return true;
}

#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

inline bool LensDiffTimingEnabled() {
    const char* value = std::getenv("LENSDIFF_TIMING");
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" && text != "OFF";
}

inline void LogLensDiffTimingStage(const char* stage, double elapsedMs, const std::string& note = std::string()) {
    if (!LensDiffTimingEnabled()) {
        return;
    }
    if (note.empty()) {
        std::fprintf(stderr, "[LensDiffTiming] stage=%s elapsedMs=%.3f\n", stage, elapsedMs);
    } else {
        std::fprintf(stderr, "[LensDiffTiming] stage=%s elapsedMs=%.3f note=%s\n", stage, elapsedMs, note.c_str());
    }
}

class LensDiffScopedTimer {
public:
    explicit LensDiffScopedTimer(const char* stage, std::string note = std::string())
        : stage_(stage)
        , note_(std::move(note))
        , enabled_(LensDiffTimingEnabled())
        , start_(enabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point()) {}

    ~LensDiffScopedTimer() {
        if (!enabled_) {
            return;
        }
        const auto end = std::chrono::steady_clock::now();
        const double elapsedMs =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start_).count();
        LogLensDiffTimingStage(stage_, elapsedMs, note_);
    }

private:
    const char* stage_ = "";
    std::string note_;
    bool enabled_ = false;
    std::chrono::steady_clock::time_point start_ {};
};

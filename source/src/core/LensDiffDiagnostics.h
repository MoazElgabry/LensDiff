#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

inline bool LensDiffEnvFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" && text != "OFF";
}

inline bool LensDiffTimingEnabled() {
    return LensDiffEnvFlagEnabled("LENSDIFF_TIMING");
}

inline bool LensDiffLogEnabled() {
    return LensDiffEnvFlagEnabled("LENSDIFF_LOG");
}

inline bool LensDiffDiagnosticsFileEnabled() {
    return LensDiffLogEnabled() || LensDiffTimingEnabled();
}

inline std::string LensDiffNowUtcIso8601() {
    const std::time_t now = std::time(nullptr);
    std::tm tmUtc {};
#ifdef _WIN32
    gmtime_s(&tmUtc, &now);
#else
    gmtime_r(&now, &tmUtc);
#endif
    char buffer[32] = {};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tmUtc);
    return buffer;
}

inline std::filesystem::path LensDiffDiagnosticsFilePath() {
    const char* overridePath = std::getenv("LENSDIFF_LOG_FILE");
    if (overridePath != nullptr && *overridePath != '\0') {
        return std::filesystem::path(overridePath);
    }
#ifdef _WIN32
    const char* base = std::getenv("LOCALAPPDATA");
    if ((base == nullptr || *base == '\0')) {
        base = std::getenv("APPDATA");
    }
    if (base != nullptr && *base != '\0') {
        return std::filesystem::path(base) / "LensDiff" / "Logs" / "LensDiff.log";
    }
#elif defined(__APPLE__)
    const char* home = std::getenv("HOME");
    if (home != nullptr && *home != '\0') {
        return std::filesystem::path(home) / "Library" / "Logs" / "LensDiff.log";
    }
#else
    const char* home = std::getenv("HOME");
    if (home != nullptr && *home != '\0') {
        return std::filesystem::path(home) / ".local" / "state" / "LensDiff" / "LensDiff.log";
    }
#endif
    return std::filesystem::path("LensDiff.log");
}

inline void WriteLensDiffDiagnosticLine(const std::string& line) {
    if (!LensDiffDiagnosticsFileEnabled()) {
        return;
    }

    static std::mutex logMutex;
    std::lock_guard<std::mutex> lock(logMutex);

    std::filesystem::path logPath = LensDiffDiagnosticsFilePath();
    std::error_code ec;
    if (logPath.has_parent_path()) {
        std::filesystem::create_directories(logPath.parent_path(), ec);
    }

    std::ofstream stream(logPath, std::ios::out | std::ios::app | std::ios::binary);
    if (!stream.is_open()) {
        return;
    }

    std::string cleanLine = line;
    while (!cleanLine.empty() && (cleanLine.back() == '\n' || cleanLine.back() == '\r')) {
        cleanLine.pop_back();
    }
    stream << "[" << LensDiffNowUtcIso8601() << "] " << cleanLine << '\n';
    stream.flush();
}

inline void LogLensDiffDiagnosticEvent(const char* event, const std::string& note = std::string()) {
    if (!LensDiffLogEnabled()) {
        return;
    }
    std::ostringstream ss;
    ss << "[LensDiffDiag] event=" << event;
    if (!note.empty()) {
        ss << " note=" << note;
    }
    std::string line = ss.str();
    std::fprintf(stderr, "%s\n", line.c_str());
    std::fflush(stderr);
    WriteLensDiffDiagnosticLine(line);
}

inline void LogLensDiffTimingStage(const char* stage, double elapsedMs, const std::string& note = std::string()) {
    if (!LensDiffTimingEnabled()) {
        return;
    }
    std::string line;
    if (note.empty()) {
        char buffer[256] = {};
        std::snprintf(buffer, sizeof(buffer), "[LensDiffTiming] stage=%s elapsedMs=%.3f", stage, elapsedMs);
        line = buffer;
        std::fprintf(stderr, "%s\n", buffer);
    } else {
        char buffer[512] = {};
        std::snprintf(buffer,
                      sizeof(buffer),
                      "[LensDiffTiming] stage=%s elapsedMs=%.3f note=%s",
                      stage,
                      elapsedMs,
                      note.c_str());
        line = buffer;
        std::fprintf(stderr, "%s\n", buffer);
    }
    std::fflush(stderr);
    WriteLensDiffDiagnosticLine(line);
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

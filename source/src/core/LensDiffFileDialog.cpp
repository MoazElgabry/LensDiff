#include "LensDiffFileDialog.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>

namespace {

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

std::string wideToUtf8(const std::wstring& text) {
    if (text.empty()) {
        return {};
    }
    const int length = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (length <= 0) {
        return {};
    }
    std::string utf8(static_cast<std::size_t>(length - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, utf8.data(), length, nullptr, nullptr);
    return utf8;
}

} // namespace

bool OpenLensDiffApertureFileDialog(std::string* outPath, std::string* error) {
    if (outPath == nullptr) {
        if (error) {
            *error = "file-dialog-output-null";
        }
        return false;
    }

    wchar_t fileBuffer[MAX_PATH] = L"";
    OPENFILENAMEW ofn {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFile = fileBuffer;
    ofn.nMaxFile = static_cast<DWORD>(std::size(fileBuffer));
    ofn.lpstrFilter =
        L"Image Files\0*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.gif;*.webp\0"
        L"All Files\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY;
    ofn.lpstrTitle = L"Select Custom Aperture";

    if (!GetOpenFileNameW(&ofn)) {
        const DWORD result = CommDlgExtendedError();
        if (result != 0 && error) {
            *error = "file-dialog-open-failed";
        }
        return false;
    }

    *outPath = wideToUtf8(fileBuffer);
    return !outPath->empty();
}

#elif defined(__linux__)

namespace {

std::string execAndReadLinuxCommand(const std::string& cmd) {
    std::string out;
    FILE* handle = popen(cmd.c_str(), "r");
    if (!handle) {
        return out;
    }
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), handle)) {
        out += buffer;
    }
    pclose(handle);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return out;
}

bool linuxCommandExists(const char* cmd) {
    if (cmd == nullptr || cmd[0] == '\0') {
        return false;
    }
    std::string probe = "command -v ";
    probe += cmd;
    probe += " >/dev/null 2>&1";
    return std::system(probe.c_str()) == 0;
}

} // namespace

bool OpenLensDiffApertureFileDialog(std::string* outPath, std::string* error) {
    if (outPath == nullptr) {
        if (error) {
            *error = "file-dialog-output-null";
        }
        return false;
    }

    if (linuxCommandExists("zenity")) {
        const std::string selectedPath =
            execAndReadLinuxCommand(
                "zenity --file-selection "
                "--title='Select Custom Aperture' "
                "--file-filter='Image files | *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.gif *.webp' "
                "--file-filter='All files | *' 2>/dev/null");
        if (!selectedPath.empty()) {
            *outPath = selectedPath;
            if (error) error->clear();
            return true;
        }
        return false;
    }

    if (linuxCommandExists("kdialog")) {
        const std::string selectedPath =
            execAndReadLinuxCommand(
                "kdialog --getopenfilename \"$HOME\" "
                "\"Image files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.gif *.webp)\" 2>/dev/null");
        if (!selectedPath.empty()) {
            *outPath = selectedPath;
            if (error) error->clear();
            return true;
        }
        return false;
    }

    if (error) {
        *error = "file-dialog-linux-helper-unavailable";
    }
    return false;
}

#else

bool OpenLensDiffApertureFileDialog(std::string* outPath, std::string* error) {
    if (outPath == nullptr) {
        if (error) {
            *error = "file-dialog-output-null";
        }
        return false;
    }
    if (error) {
        *error = "file-dialog-platform-implementation-unavailable";
    }
    return false;
}

#endif

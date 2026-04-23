#include "LensDiffFileDialog.h"

#if defined(__APPLE__)

#import <AppKit/AppKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

bool OpenLensDiffApertureFileDialog(std::string* outPath, std::string* error) {
    if (outPath == nullptr) {
        if (error) {
            *error = "file-dialog-output-null";
        }
        return false;
    }

    NSOpenPanel* panel = [NSOpenPanel openPanel];
    [panel setCanChooseFiles:YES];
    [panel setCanChooseDirectories:NO];
    [panel setAllowsMultipleSelection:NO];
    [panel setTitle:@"Select Custom Aperture"];
    [panel setAllowedContentTypes:@[
        UTTypePNG,
        UTTypeJPEG,
        UTTypeTIFF,
        UTTypeBMP,
        UTTypeGIF,
        UTTypeWebP
    ]];

    if ([panel runModal] != NSModalResponseOK) {
        return false;
    }

    NSURL* url = [[panel URLs] firstObject];
    if (url == nil) {
        if (error) {
            *error = "file-dialog-no-selection";
        }
        return false;
    }

    NSString* path = [url path];
    if (path == nil || [path length] == 0) {
        if (error) {
            *error = "file-dialog-empty-path";
        }
        return false;
    }

    *outPath = std::string([path UTF8String]);
    return !outPath->empty();
}

#endif

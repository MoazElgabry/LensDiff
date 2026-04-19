#pragma once

#include <string>
#include <vector>

struct LensDiffApertureImage {
    int width = 0;
    int height = 0;
    std::vector<float> values;
};

bool LoadLensDiffApertureImage(const std::string& path,
                               LensDiffApertureImage* outImage,
                               std::string* error);

bool LoadLensDiffPreparedApertureImage(const std::string& path,
                                       bool normalize,
                                       bool invert,
                                       LensDiffApertureImage* outImage,
                                       std::string* error);

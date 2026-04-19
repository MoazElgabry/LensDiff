#pragma once

#include "LensDiffTypes.h"

#include <memory>
#include <vector>
#include <string>

int GetLensDiffEffectivePupilResolution(int requested);

int ChooseLensDiffRawPsfSize(int pupilSize, int maxKernelRadiusPx);

LensDiffPsfBankKey MakeLensDiffPsfBankKey(const LensDiffParams& params);
LensDiffFieldKey MakeLensDiffFieldKey(const LensDiffParams& params);

std::vector<float> BuildLensDiffPupilAmplitudeForPsf(const LensDiffParams& params);

std::vector<float> BuildLensDiffPupilPhaseForPsf(const LensDiffParams& params);

std::vector<float> BuildLensDiffStaticDebugRgba(const LensDiffParams& params,
                                                const LensDiffPsfBankCache& cache,
                                                int outWidth,
                                                int outHeight);
const std::vector<float>& GetLensDiffStaticDebugRgbaCached(const LensDiffParams& params,
                                                           LensDiffPsfBankCache* cache,
                                                           int outWidth,
                                                           int outHeight);

std::vector<float> BuildLensDiffShiftedRawPsf(const std::vector<float>& pupil,
                                              const std::vector<float>& phaseWaves,
                                              int pupilSize,
                                              int rawPsfSize);

std::vector<float> BuildLensDiffShiftedRawPsf(const std::vector<float>& pupil,
                                              int pupilSize,
                                              int rawPsfSize);

float EstimateLensDiffFirstMinimumRadius(const std::vector<float>& shiftedRawPsf,
                                         int size);

std::vector<float> GetLensDiffSpectralWavelengths(LensDiffSpectralMode mode);
std::shared_ptr<const std::vector<float>> GetLensDiffReferenceRawPsfCached(int pupilSize,
                                                                           int rawPsfSize);

void FinalizeLensDiffPsfBankFromRawPsf(const LensDiffParams& params,
                                       const LensDiffPsfBankKey& key,
                                       const std::vector<float>& pupil,
                                       int pupilSize,
                                       const std::vector<float>& rawPsf,
                                       int rawPsfSize,
                                       const std::vector<float>& referenceRawPsf,
                                       LensDiffPsfBankCache& cache);
void FinalizeLensDiffPsfBankFromBaseKernels(const LensDiffParams& params,
                                            const LensDiffPsfBankKey& key,
                                            const std::vector<float>& pupil,
                                            int pupilSize,
                                            const std::vector<float>& wavelengths,
                                            std::vector<LensDiffKernel> baseKernels,
                                            LensDiffPsfBankCache& cache);

void EnsureLensDiffPsfBank(const LensDiffParams& params,
                           LensDiffPsfBankCache& cache);

bool RunLensDiffCpuReference(const LensDiffRenderRequest& request,
                             const LensDiffParams& params,
                             LensDiffPsfBankCache& cache,
                             std::string* error);

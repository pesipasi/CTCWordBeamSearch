#pragma once
#include "IMatrix.hpp"
#include "LanguageModel.hpp"
#include <stdint.h>
#include <cstddef>


// apply word beam search decoding on the matrix with given beam width
std::vector<std::vector<double>> wordBeamSearch(const IMatrix& mat, size_t beamWidth, const std::shared_ptr<LanguageModel>& lm, LanguageModelType lmType);


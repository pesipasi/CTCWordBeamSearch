#include "DataLoader.hpp"
#include "WordBeamSearch.hpp"
#include "Metrics.hpp"
#include "test.hpp"
#include <iostream>
#include <chrono>
// run unit tests: uncomment next line and run in debug mode
//#define UNITTESTS 


int main(int argc, char** argv)
{

#ifdef UNITTESTS
	test();
#else
	const std::string baseDir = "data/"; // dir containing corpus.txt, chars.txt, wordChars.txt, mat_x.csv, gt_x.txt with x=0, 1, ...
	const size_t sampleEach = 1; // only take each k*sampleEach sample from dataset, with k=0, 1, ...
	const double addK = 1.0; // add-k smoothing of bigram distribution
	const LanguageModelType lmType = LanguageModelType::Words; // scoring mode
	DataLoader loader{ baseDir, sampleEach, lmType, addK }; // load data
	const auto& lm = loader.getLanguageModel(); // get LM
	Metrics metrics{ lm->getWordChars() }; // CER and WER

	size_t ctr = 0;
	while (loader.hasNext())
	{
		// get data
		const auto data = loader.getNext();

		int a = strtol(argv[1], nullptr, 0);
		std::vector<std::vector<double>> res = wordBeamSearch(data.mat, a, lm, lmType);


		for (auto i = 0; i< res[0].size(); i++)
		{
			std::vector<uint32_t> char_t;
			char_t.push_back((int32_t) res[1][i]);
			std::cout<<res[0][i]<< " " << lm->labelToUtf8(char_t) << " " << res[2][i] <<'\n';
		}
		

		
		++ctr;
	}

#endif


	return 0;
}

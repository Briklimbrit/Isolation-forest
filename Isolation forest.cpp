#include "IsoForest.h"

#include <chrono>
#include <fstream>
#include <inttypes.h>
#include <iostream>
#include <stdlib.h>



using namespace IsolationForest;

void test(std::ofstream& outStream, size_t numTrainingSamples, size_t numTestSamples, uint32_t numTrees, uint32_t subSamplingSize, bool dump)
{
	Forest forest(numTrees, subSamplingSize);

	std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();

	// Create some training samples.
	for (size_t i = 0; i < numTrainingSamples; ++i)
	{
		Sample sample("training");
		FeaturePtrList features;

		uint32_t x = rand() % 25;
		uint32_t y = rand() % 25;

		features.push_back(new Feature("x", x));
		features.push_back(new Feature("y", y));

		sample.AddFeatures(features);
		forest.AddSample(sample);

		if (outStream.is_open())
		{
			outStream << "training," << x << "," << y << std::endl;
		}
	}

	// Create the isolation forest.
	forest.Create();

	// Test samples (similar to training samples).
	double avgControlScore = (double)0.0;
	double avgControlNormalizedScore = (double)0.0;
	for (size_t i = 0; i < numTestSamples; ++i)
	{
		Sample sample("control sample");
		FeaturePtrList features;

		uint32_t x = rand() % 25;
		uint32_t y = rand() % 25;

		features.push_back(new Feature("x", x));
		features.push_back(new Feature("y", y));
		sample.AddFeatures(features);

		// Run a test with the sample that doesn't contain outliers.
		double score = forest.Score(sample);
		double normalizedScore = forest.NormalizedScore(sample);
		avgControlScore += score;
		avgControlNormalizedScore += normalizedScore;

		if (outStream.is_open())
		{
			outStream << "control," << x << "," << y << std::endl;
		}
	}
	avgControlScore /= numTestSamples;
	avgControlNormalizedScore /= numTestSamples;

	// Outlier samples (different from training samples).
	double avgOutlierScore = (double)0.0;
	double avgOutlierNormalizedScore = (double)0.0;
	for (size_t i = 0; i < numTestSamples; ++i)
	{
		Sample sample("outlier sample");
		FeaturePtrList features;

		uint32_t x = 20 + (rand() % 25);
		uint32_t y = 20 + (rand() % 25);

		features.push_back(new Feature("x", x));
		features.push_back(new Feature("y", y));
		sample.AddFeatures(features);

		// Run a test with the sample that contains outliers.
		double score = forest.Score(sample);
		double normalizedScore = forest.NormalizedScore(sample);
		avgOutlierScore += score;
		avgOutlierNormalizedScore += normalizedScore;

		if (outStream.is_open())
		{
			outStream << "outlier," << x << "," << y << std::endl;
		}
	}
	avgOutlierScore /= numTestSamples;
	avgOutlierNormalizedScore /= numTestSamples;

	std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);

	std::cout << "Average of control test samples: " << avgControlScore << std::endl;
	std::cout << "Average of control test samples (normalized): " << avgControlNormalizedScore << std::endl;
	std::cout << "Average of outlier test samples: " << avgOutlierScore << std::endl;
	std::cout << "Average of outlier test samples (normalized): " << avgOutlierNormalizedScore << std::endl;
	std::cout << "Total time for Test 1: " << elapsedTime.count() << " seconds." << std::endl;

	if (dump)
	{
		forest.Dump();
	}
}

int main(int argc, const char* argv[])
{
	const char* ARG_OUTFILE = "outfile";
	const char* ARG_DUMP = "dump";
	const size_t NUM_TRAINING_SAMPLES = 100;
	const size_t NUM_TEST_SAMPLES = 10;
	const uint32_t NUM_TREES_IN_FOREST = 10;
	const uint32_t SUBSAMPLING_SIZE = 10;

	std::ofstream outStream;
	bool dump = false;

	// Parse the command line arguments.
	for (int i = 1; i < argc; ++i)
	{
		if ((strncmp(argv[i], ARG_OUTFILE, strlen(ARG_OUTFILE)) == 0) && (i + 1 < argc))
		{
			outStream.open(argv[i + 1]);
		}
		if (strncmp(argv[i], ARG_DUMP, strlen(ARG_DUMP)) == 0)
		{
			dump = true;
		}
	}

	srand((unsigned int)time(NULL));

	std::cout << "Test 1:" << std::endl;
	std::cout << "-------" << std::endl;
	test(outStream, NUM_TRAINING_SAMPLES, NUM_TEST_SAMPLES, NUM_TREES_IN_FOREST, SUBSAMPLING_SIZE, dump);
	std::cout << std::endl;

	std::cout << "Test 2:" << std::endl;
	std::cout << "-------" << std::endl;
	test(outStream, NUM_TRAINING_SAMPLES * 10, NUM_TEST_SAMPLES * 10, NUM_TREES_IN_FOREST * 10, SUBSAMPLING_SIZE * 10, dump);
	std::cout << std::endl;

	if (outStream.is_open())
	{
		outStream.close();
	}

	return 0;
}
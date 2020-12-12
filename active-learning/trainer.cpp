
/*
Implementation of Active Learning SVM with DLIB
*/

#include <iostream>
#include <dlib/svm.h>
#include <dlib/data_io.h>

using namespace std;
using namespace dlib;


int main(int argc, const char *argv[])
{
	// Load libsvm data file
	typedef std::vector<std::pair<unsigned int, double> > sparse_vect_type;
	std::vector<sparse_vect_type> samples;
	std::vector<double> labels;

	// Check the number of parameters
	if (argc < 3) {
		// Tell the user how to run the program
		std::cerr << "Usage: " << argv[0] << " <training_data_file_path> <output_file_path> [c_value]" << std::endl;
		return 1;
	}

	// Load training data file
	load_libsvm_formatted_data(argv[1], samples, labels);
	int N = samples.size();
	int kfold = 5; // k-fold crossvalidation
	int iteration = 10;
	int validation_size = ((kfold - 1) *N) / kfold;

	// Count positive samples
	int posN = 0;
	for (unsigned int i = 0; i < labels.size(); i++)
	{
		if (labels[i] == 1)
		{
			posN++;
		}
	}
	// Train and test with Linear SVM
	typedef sparse_linear_kernel<sparse_vect_type> kernel_type;
	svm_c_linear_dcd_trainer<kernel_type> linear_trainer;

	if (argc < 4)
	{
		// Write result output
		ofstream resultFile;
		resultFile.open(argv[2], ios::out | ios::binary);

		// Loop through multiple values of c parameter in SVM
		for (double c_value = 1; c_value <= 50; c_value = c_value + 1)
		{
			linear_trainer.set_c(c_value);
			randomize_samples(samples, labels);
			double precision = 0;
			double recall = 0;
			int countIter = 0;
			for (unsigned int iter = 0; iter < iteration; iter++)
			{
				std::vector<sparse_vect_type> tr_samples(samples.begin(), samples.begin() + validation_size);
				std::vector<double> tr_labels(labels.begin(), labels.begin() + validation_size);
				std::vector<sparse_vect_type> te_samples(samples.begin() + validation_size, samples.end());
				std::vector<double> te_labels(labels.begin() + validation_size, labels.end());

				decision_function<kernel_type> df = linear_trainer.train(tr_samples, tr_labels);

				if ((N < 50) || (posN < 10))
				{
					te_samples = samples;
					te_labels = labels;
				}

				// Count positive samples in test set
				int posN_te = 0;
				for (unsigned int i = 0; i < te_labels.size(); i++)
				{
					if (te_labels[i] == 1)
					{
						posN_te++;
					}
				}

				// Skip if there is no pos samples in test set
				if (posN_te == 0)
					continue;

				double true_pos, false_pos, false_neg;
				true_pos = 0.0;
				false_pos = 0.0;
				false_neg = 0.0;
				for (unsigned int i = 0; i < te_samples.size(); i++)
				{
					// Predict for each samples
					double pred = df(te_samples[i]);

					// Calculate number of true pos, false pos, and false neg
					if ((pred >= 0) && (te_labels[i] == 1)) // True positive classification
					{
						true_pos = true_pos + 1;
					}
					if ((pred >= 0) && (te_labels[i] == -1)) // False positive classification
					{
						false_pos = false_pos + 1;
					}
					if ((pred < 0) && (te_labels[i] == 1)) // False negative classification
					{
						false_neg = false_neg + 1;
					}
				}
				precision = precision + true_pos / (true_pos + false_pos);
				recall = recall + true_pos / (true_pos + false_neg);
				countIter++;
			}
			precision = precision / countIter;
			recall = recall / countIter;
			resultFile << c_value << "," << precision << "," << recall << endl;
		}
		resultFile.close();
	}
	else
	{ // train with spefic c_value and output classifier
		double c_value = atof(argv[3]);
		linear_trainer.set_c(c_value);
		decision_function<kernel_type> df = linear_trainer.train(samples, labels);
		ofstream fout(argv[2], ios::binary);
		serialize(df, fout);
		fout.close();
	}

	return 0;
}
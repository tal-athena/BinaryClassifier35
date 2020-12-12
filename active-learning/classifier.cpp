
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
	if (argc < 4) {
		// Tell the user how to run the program
		std::cerr << "Usage: " << argv[0] << " <trained_classifier_file_path> <testing_data_file_path> <output_file_path> [<threshold>]" << std::endl;
		return 1;
	}

	// Load data file
	load_libsvm_formatted_data(argv[2], samples, labels);

	// Load classifier
	typedef sparse_linear_kernel<sparse_vect_type> kernel_type;
	decision_function< kernel_type > df;
	ifstream fin(argv[1], ios::binary);
	deserialize(df, fin);

	// Classify test data
	ofstream resultFile;
	resultFile.open(argv[3], ios::out | ios::binary);
	double threshold = (argc > 4) ? atof(argv[4]) : 0.0;
	for (unsigned int i = 0; i < samples.size(); i++)
	{
		// Predict for each samples
		double pred = df(samples[i]);
		int classified = (pred > threshold) ? 1 : -1;
		resultFile << pred << "," << classified << endl;
	}
	resultFile.close();

	return 0;
}
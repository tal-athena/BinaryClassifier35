
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
	std::vector<sparse_vect_type> usamples;
	std::vector<double> ulabels;

	// Check the number of parameters
	if (argc < 5) {
		// Tell the user how to run the program
		std::cerr << "Usage: " << argv[0] << " <training_data_file_path> <new_data_file_path> <c_value> <output_file_path>" << std::endl;
		return 1;
	}

	// Load data file
	load_libsvm_formatted_data(argv[1], samples, labels);
	load_libsvm_formatted_data(argv[2], usamples, ulabels);

	// Define SVM trainer
	typedef sparse_linear_kernel<sparse_vect_type> kernel_type;
	svm_c_linear_dcd_trainer<kernel_type> trainer;
	double c_value = atof(argv[3]);
	trainer.set_c(c_value);
	decision_function<sparse_linear_kernel<sparse_vect_type> > df;

	// Ranking new samples
	std::vector<unsigned long> results;	
	active_learning_mode mode = max_min_margin;
	results = rank_unlabeled_training_samples(trainer, samples, labels, usamples, mode);

	// Write ranking output
	ofstream resultFile;
	resultFile.open(argv[4], ios::out | ios::binary);
	for (unsigned int i = 0; i < results.size(); i++)
	{
		resultFile << results[i] << endl;
	}
	resultFile.close();

	return 0;
}
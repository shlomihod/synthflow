#include <string>
#include <iostream>
#include <fstream>
#include <assert.h>
using namespace std;

#include "methods.h"

int main(int argc, char *argv[]) {
	// arguments
	if (argc < 3) {
		printf("incorrect args. ");
		// printf("./main <data> <sample> <iter> <theta1> <theta2> ...");
		printf("./main <data> <sample> <iter> <epsilon> <epsilon_split> <degree1> <degree2> ...");
		return 0;
	}
	string dataset = argv[1];
	cout << dataset << endl;

	int nsam = stoi(argv[2]);
	int niters = stoi(argv[3]);
    double epsilon = stod(argv[4]);
    double epsilon_split = stod(argv[5]);

	assert(epsilon > 0);
	assert(epsilon_split > 0 && epsilon_split < 1);

// 	vector<double> thetas;
// 	for (int i = 4; i < argc; i++) {
// 		thetas.push_back(stod(argv[i]));
// 		cout << thetas.back() << "\t";
// 	}
// 	cout << endl;
// 	// arguments

	vector<int> degrees;
	// for (int i = 4; i < argc; i++) {
	for (int i = 6; i < argc; i++) {
		degrees.push_back(stoi(argv[i]));
		cout << degrees.back() << "\t";
	}
	cout << endl;
	// arguments


	ofstream out("log/" + dataset + ".out");
	ofstream log("log/" + dataset + ".log");
	cout.rdbuf(log.rdbuf());

	random_device rd;						//non-deterministic random engine
	engine eng(rd());						//deterministic engine with a random seed

	table tbl("data/" + dataset, true);

    
	for (int degree : degrees) {
		cout << "degree: " << degree << endl;
		out << "degree: " << degree << endl;
		// for (double epsilon : {10}) {
			for (int iter = 0; iter < niters; iter++) {
				// cout << "epsilon: " << epsilon << " iter:" << iter + 1 << endl;
				cout << "epsilon: " << epsilon << " epsilon_split: " << epsilon_split << " iter:" << iter + 1 << endl;
				bayesian bayesian(eng, tbl, epsilon, epsilon_split, degree);
				bayesian.sampling(nsam);
				// bayesian.syn.printo_file("output/syn_" + dataset + "_eps" + to_string(int(epsilon)) +
				// 	"_degree" + to_string(degree) + "_iter" + to_string(iter) + ".dat");
				bayesian.syn.printo_file("output/syn_" + dataset + "_eps" + to_string(int(epsilon)) +
					"_split" + to_string(int(epsilon_split * 100)) + "_degree" + to_string(int(degree)) + "_iter" + to_string(iter) + ".dat");

			}
		// }
		cout << endl;
		out << endl;
        
// 	for (double theta : thetas) {
// 		cout << "theta: " << theta << endl;
// 		out << "theta: " << theta << endl;
// 		for (double epsilon : {10}) {
// 			for (int iter = 0; iter < niters; iter++) {
// 				cout << "epsilon: " << epsilon << " iter:" << iter + 1 << endl;
// 				bayesian bayesian(eng, tbl, epsilon, theta);
// 				bayesian.sampling(nsam);
// 				bayesian.syn.printo_file("output/syn_" + dataset + "_eps" + to_string(int(epsilon)) +
// 					"_theta" + to_string(int(theta)) + "_iter" + to_string(iter) + ".dat");
// 			}
// 		}
// 		cout << endl;
// 		out << endl;
	}
	out.close();
	log.close();
	return 0;
}

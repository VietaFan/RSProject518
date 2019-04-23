#include <iostream>
#include <fstream>
#include <unordered_map>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <algorithm>
using namespace std;

template<typename T>
ostream& operator<<(ostream &out, vector<T> &vec) {
	out << "[";
	if (vec.size() > 0) {		
		for (int i=0; i<vec.size()-1; ++i)
			out << vec[i] << ", ";
		out << vec[vec.size()-1];
	}
	out << "]";
	return out;
}

template<typename T>
ostream& operator<<(ostream &out, vector<T> &&vec) {
	out << "[";
	if (vec.size() > 0) {		
		for (int i=0; i<vec.size()-1; ++i)
			out << vec[i] << ", ";
		out << vec[vec.size()-1];
	}
	out << "]";
	return out;
}


struct sim_params {
	// user vars
	vector<double> user_cost, user_alpha, user_probs;
	// constraint function vars
	double cf_a, cf_b, cf_c;
	double cf_alpha, cf_beta, cf_gamma;
	double cf_t;
	// starting params
	uint64_t N, S;
};


inline double g(double a, double alpha, uint64_t N) {
	double x = pow(a*N, alpha);
	return x/(1.0+x);
}


double calc_constraint_func(sim_params &params, int n, uint64_t N, uint64_t S) {
	return params.cf_t*g(params.cf_a,params.cf_alpha,N) + (1.0-params.cf_t)*g(params.cf_b,params.cf_beta,n)*g(params.cf_c,params.cf_gamma,S-N);
}

bool construct_acc_func(sim_params &params, vector<int> &target_cts, vector<double> &f) {
	vector<int> tc_orig;
	for (int x: target_cts) {
		tc_orig.push_back(x);
	}
	//cout << "hi\n";
	//cout << target_cts << endl;
	f.clear();
	int nmax = 0;
	for (double cost: params.user_cost) {
		if (1.0/cost > nmax) {
			nmax = (int) (1.0/cost)+1;
		}
	}
	int k = params.user_cost.size();
	// order the n_i's so that they're in increasing order
	// we're using selection sort, but it doesn't matter since the rest is exponential anyway
	vector<int> permutation;
	vector<int> tc_copy;
	for (int x: target_cts)
		tc_copy.push_back(x);
	sort(tc_copy.begin(), tc_copy.end());
	vector<bool> visited(k, 0);
	for (int i=0; i<k; i++) {
		for (int j=0; j<k; j++) {
			if (visited[j]) continue;
			if (target_cts[j] == tc_copy[i]) {
				permutation.push_back(j);
				visited[j] = 1;
			}
		}
	}
	for (int i=0; i<=nmax; i++) {
		f.push_back(0);
	}
	int t, t_prev;
	target_cts.insert(target_cts.begin(),0);
	for (int i=0; i<k; i++) {
		t = permutation[i]+1;
		if (i == 0) {
			t_prev = 0;
		} else {
			t_prev = permutation[i-1]+1;
		}
		for (int j=target_cts[t_prev]+1; j<=target_cts[t]; j++) {
			f[j] = f[target_cts[t_prev]];
		}
		double reqval;
		for (int j=0; j<target_cts[t]; j++) {
			reqval = pow(pow(f[j], params.user_alpha[t-1])-params.user_cost[t-1]*j+params.user_cost[t-1]*target_cts[t], 1.0/params.user_alpha[t-1]);
	//		cout << "reqval = " << reqval << endl;
			if (reqval > f[target_cts[t]]) {
				f[target_cts[t]] = reqval;
			}
		}
	}
	//cout << "f = " << f << ", n_1=" << target_cts[permutation[k-1]+1] << endl;
	for (t=target_cts[permutation[k-1]+1]+1; t<=nmax; t++) {
		f[t] = f[target_cts[permutation[k-1]+1]];
	}
	//cout << "f = " << f << endl;
	vector<double> F;
	for (int n=0; n<=nmax; n++) {
		F.push_back(calc_constraint_func(params, n, params.N, params.S));
	}
	//cout << "F = " << F << endl;
	for (int n=1; n<=nmax; n++) {
		if (f[n] > F[n]) {
			f.clear();
			target_cts.clear();
			for (int x: tc_orig) {
				target_cts.push_back(x);
			}
			return false;
		}
	}
	target_cts.clear();
	for (int x: tc_orig) {
		target_cts.push_back(x);
	}			
	return true;
}

double find_optimal_acc_func(sim_params &params, vector<double> &bestf, bool verbose) {
	int nmax = 0;
	for (double cost: params.user_cost) {
		if (1.0/cost > nmax) {
			nmax = (int) (1.0/cost)+1;
		}
	}
	bestf.clear();
	for (int i=0; i<=nmax; i++) {
		bestf.push_back(0);
	}
	vector<double> f;
	double best_exp_nr = 0, exp_nr;
	int nsol = 1;
	int k = params.user_cost.size();
	for (int i=0; i<k; i++) {
		nsol *= nmax;
	}
	int x;
	for (int j=0; j<nsol; j++) {
		vector<int> target_cts(k);
		x = j;
		for (int i=0; i<k; i++) {
			target_cts[i] = x%nmax;
			x /= nmax;
		}
		exp_nr = 0;
		for (int t=0; t<k; t++) {
			exp_nr += params.user_probs[t]*target_cts[t];
		}
		// only test it if it could potentially be better
		if (exp_nr <= best_exp_nr) {
			continue;
		}
		if (construct_acc_func(params, target_cts, f)) {
			if(verbose)
				cout << "possible: " << target_cts << endl;
			//cout << "yay\n";
			best_exp_nr = exp_nr;
			for (int i=0; i<=nmax; i++) {
				bestf[i] = f[i];
			}
			if(verbose) {
				cout << "new best expected #ratings = " << best_exp_nr << endl;
				cout << "new best accuracy function = " << bestf << endl;
			}
	//		cout << bestf << endl;
		} else {
			//cout << "not possible: " << target_cts << endl;
		}
		//cout << j << endl;
	}
	//cout << "done\n";
	return best_exp_nr;
}		

int main(int argc, char **argv) {
	/*if (argc < 2) {
		cout << "syntax: 'simulation params.txt' or 'simulation params.txt outfile.txt'\n";
		return 0;
	}
	
	sim_params_vec vparams;
	load_params(string(argv[1]), vparams);
	
	if (argc == 3) {
		ofstream out(argv[2]);
		gen_solutions(vparams, out);
		out.close();
	} else {
		gen_solutions(vparams, cout);
	}*/
	
	sim_params params;
	
	params.N = 1000;
	params.S = 200000;
	params.cf_a = 0.0001;
	params.cf_b = 0.2;
	params.cf_c = 0.0001;
	params.cf_alpha = 1.0;
	params.cf_beta = 1.0;
	params.cf_gamma = 1.0;
	params.cf_t = 0.2;
	
		
	// with alpha=1.0 for all users, start at 34 with accuracy .68
	params.user_cost.push_back(0.02);
	params.user_alpha.push_back(1.0);
	params.user_probs.push_back(0.68);
	
	// with alpha=1.5 for all users, start at 26 with accuracy .646649
	params.user_cost.push_back(0.02);
	params.user_alpha.push_back(1.5);
	params.user_probs.push_back(0.18);
	
	// with alpha=2.0 for all users, start at 19 with accuracy .616441
	params.user_cost.push_back(0.02);
	params.user_alpha.push_back(2.0);
	params.user_probs.push_back(0.10);
	
	vector<double> bestf;
	cout << "max expected num ratings = " << find_optimal_acc_func(params, bestf, false) << endl;
	cout << "best accuracy function = " << bestf << endl;
	
	
	// try out a specific boundary constraint
	vector<int> tcts;
	tcts.push_back(18);
	tcts.push_back(6);
	tcts.push_back(11);
	cout << tcts << endl;
	vector<double> f;
	cout << construct_acc_func(params, tcts, f) << endl;
	cout << f << endl;
	cout << "exp util: " << (params.user_probs[0]*tcts[0]+params.user_probs[1]*tcts[1]+params.user_probs[2]*tcts[2]) << endl;
	return 0;
}

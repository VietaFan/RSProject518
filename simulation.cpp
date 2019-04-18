#include <iostream>
#include <fstream>
#include <unordered_map>
#include <cmath>
#include <vector>
#include <cstdlib>
using namespace std;

struct sim_result {
	uint64_t N, S;
	double util;
	vector<uint64_t> indiv_contribs;
	vector<double> indiv_utils;
};

enum sim_type {
	weighted, threshold
};

struct sim_params {
	// user vars
	double user_cost, user_alpha;
	// constraint function vars
	double cf_a, cf_b, cf_c;
	double cf_alpha, cf_beta, cf_gamma;
	double cf_t;
	// starting params
	uint64_t N_0, S_0;
	// accuracy function type
	sim_type type;
	// simulation parameters
	double threshold_ratio;
	double weighting_exp;
	uint64_t num_users;
};

struct sim_params_vec {
	// user vars
	vector<double> user_cost, user_alpha;
	// constraint function vars
	vector<double> cf_a, cf_b, cf_c;
	vector<double> cf_alpha, cf_beta, cf_gamma;
	vector<double> cf_t;
	// starting params
	vector<uint64_t> N_0, S_0;
	// accuracy function type
	sim_type type;
	// simulation parameters
	vector<double> threshold_ratio;
	vector<double> weighting_exp;
	vector<uint64_t> num_users;
};

int user_num_ratings(sim_params &params, vector<double> &f) {
	double util, max_util = 0.0;
	int best_n = 0;
	for (int n=0; n*params.user_cost < 1.0; n++) {
		util = pow(f[n], params.user_alpha)-n*params.user_cost;
		if (util > max_util) {
			best_n = n;
			max_util = util;
		}
	}
	return best_n;
}

int user_max_positive(sim_params &params, vector<double> &f) {
	double util;
	int best_n = 0;
	for (int n=0; n*params.user_cost < 1.0; n++) {
		util = pow(f[n], params.user_alpha)-n*params.user_cost;
		if (util > 0) {
			best_n = n;
		}
	}
	return best_n;
}

inline double g(double a, double alpha, uint64_t N) {
	double x = pow(a*N, alpha);
	return x/(1.0+x);
}

double calc_constraint_func(sim_params &params, int n, uint64_t N, uint64_t S) {
	return params.cf_t*g(params.cf_a,params.cf_alpha,N) + (1.0-params.cf_t)*g(params.cf_b,params.cf_beta,n)*g(params.cf_c,params.cf_gamma,S-N);
}

void run_sim(sim_params &params, sim_result &res) {
	res.util = 0;
	res.N = params.N_0;
	res.S = params.S_0;
	res.indiv_contribs = vector<uint64_t>();
	res.indiv_utils = vector<double>();
	for (int i=0; i<params.num_users; i++) {
		vector<double> F, f;
		for (int n=0; n*params.user_cost < 1.0; n++) {
			F.push_back(calc_constraint_func(params, n, res.N, res.S));
		}
		int n_min, n_max, n_chosen;
		n_min = user_num_ratings(params, F);
		n_max = user_max_positive(params, F);
		double r = 1.0*i/params.num_users, w;
		if (params.type == weighted) {
			w = pow(r, params.weighting_exp);
			n_chosen = (int) (n_min*w+n_max*(1.0-w));
		} else if (params.type == threshold) {
			if (r < params.threshold_ratio) {
				n_chosen = n_max;
			} else {
				n_chosen = n_min;
			}
		}
		for (int n=0; n<n_chosen; n++) {
			f.push_back(0.0);
		}
		for (int n=n_chosen; n*params.user_cost < 1.0; n++) {
			f.push_back(F[n]);
		}
		int n_contrib = user_num_ratings(params, f);
		double user_util = pow(f[n_contrib], params.user_alpha)-n_contrib*params.user_cost;
		res.indiv_contribs.push_back(n_contrib);
		res.indiv_utils.push_back(user_util);
		res.N += n_contrib;
		res.S += n_contrib*n_contrib;
		res.util += user_util;
	}
}

void load_params(string filename, sim_params_vec &params) {
	ifstream fin(filename);
	string token;
	unordered_map<string, vector<uint64_t>*> ntokenmap;
	unordered_map<string, vector<double>*> dtokenmap;
	dtokenmap["user_cost"] = &params.user_cost;
	dtokenmap["user_alpha"] = &params.user_alpha;
	dtokenmap["cf_a"] = &params.cf_a;
	dtokenmap["cf_b"] = &params.cf_b;
	dtokenmap["cf_c"] = &params.cf_c;
	dtokenmap["cf_alpha"] = &params.cf_alpha;
	dtokenmap["cf_beta"] = &params.cf_beta;
	dtokenmap["cf_gamma"] = &params.cf_gamma;
	dtokenmap["cf_t"] = &params.cf_t;
	dtokenmap["weighting_exp"] = &params.weighting_exp;
	dtokenmap["threshold_ratio"] = &params.threshold_ratio;
	ntokenmap["N_0"] = &params.N_0;
	ntokenmap["S_0"] = &params.S_0;
	ntokenmap["num_users"] = &params.num_users;
	bool current_double = false;
	vector<double>* current_dvec;
	vector<uint64_t>* current_nvec;
	while (!fin.eof()) {
		token = "";
		fin >> token;
		if (token.size() == 0) continue;
		if (!token.compare("weighted")) {
			params.type = weighted;
		} else if (!token.compare("threshold")) {
			params.type = threshold;
		} else if (ntokenmap.count(token)) {
			current_nvec = ntokenmap[token];
			current_double = false;
		} else if (dtokenmap.count(token)) {
			current_dvec = dtokenmap[token];
			current_double = true;
		} else {
			if (current_double) {
				current_dvec->push_back(stod(token));
			} else {
				current_nvec->push_back(stoull(token));
			}
		}
	}
	if (params.weighting_exp.size() == 0) {
		params.weighting_exp.push_back(0.0);
	}
	if (params.threshold_ratio.size() == 0) {
		params.threshold_ratio.push_back(0.0);
	}
}

void gensol_recsearch(sim_params &params, vector<string> &keys, vector<int> &sizes, int pos, unordered_map<string, vector<double>*> &vptrmap_d, 
					  unordered_map<string, vector<uint64_t>*> &vptrmap_n, unordered_map<string, double*> &pptrmap_d, unordered_map<string, uint64_t*> &pptrmap_n, ostream &out) {
	if (pos == keys.size()) {
		out << "{";
		for (int i=0; i<keys.size(); i++) {
			if (sizes[i] <= 1) {
				continue;
			}
			out << "'" << keys[i] << "': ";
			if (pptrmap_d.count(keys[i])) {
				out << (*pptrmap_d[keys[i]]);
			} else {
				out << (*pptrmap_n[keys[i]]);
			}
			out << ", ";
		}
		sim_result res;
		run_sim(params, res);
		out << "'utility': " << res.util << ", 'N': " << res.N << ", 'S': " << res.S;
		out << "}\n";
		return;
	}
	if (pptrmap_d.count(keys[pos])) {
		for (double d: *vptrmap_d[keys[pos]]) {
			*pptrmap_d[keys[pos]] = d;
			gensol_recsearch(params, keys, sizes, pos+1, vptrmap_d, vptrmap_n, pptrmap_d, pptrmap_n, out);
		}
	} else {
		for (uint64_t n: *vptrmap_n[keys[pos]]) {
			*pptrmap_n[keys[pos]] = n;
			gensol_recsearch(params, keys, sizes, pos+1, vptrmap_d, vptrmap_n, pptrmap_d, pptrmap_n, out);
		}
	}
}
void gen_solutions(sim_params_vec &vparams, ostream &out) {
	sim_params params;
	unordered_map<string, double*> pptrmap_d;
	unordered_map<string, uint64_t*> pptrmap_n;
	unordered_map<string, vector<double>*> vptrmap_d;
	unordered_map<string, vector<uint64_t>*> vptrmap_n;
	pptrmap_d["user_cost"] = &params.user_cost;
	pptrmap_d["user_alpha"] = &params.user_alpha;
	pptrmap_d["cf_a"] = &params.cf_a;
	pptrmap_d["cf_b"] = &params.cf_b;
	pptrmap_d["cf_c"] = &params.cf_c;
	pptrmap_d["cf_alpha"] = &params.cf_alpha;
	pptrmap_d["cf_beta"] = &params.cf_beta;
	pptrmap_d["cf_gamma"] = &params.cf_gamma;
	pptrmap_d["cf_t"] = &params.cf_t;
	pptrmap_d["weighting_exp"] = &params.weighting_exp;
	pptrmap_d["threshold_ratio"] = &params.threshold_ratio;
	pptrmap_n["N_0"] = &params.N_0;
	pptrmap_n["S_0"] = &params.S_0;
	pptrmap_n["num_users"] = &params.num_users;
	vptrmap_d["user_cost"] = &vparams.user_cost;
	vptrmap_d["user_alpha"] = &vparams.user_alpha;
	vptrmap_d["cf_a"] = &vparams.cf_a;
	vptrmap_d["cf_b"] = &vparams.cf_b;
	vptrmap_d["cf_c"] = &vparams.cf_c;
	vptrmap_d["cf_alpha"] = &vparams.cf_alpha;
	vptrmap_d["cf_beta"] = &vparams.cf_beta;
	vptrmap_d["cf_gamma"] = &vparams.cf_gamma;
	vptrmap_d["cf_t"] = &vparams.cf_t;
	vptrmap_d["weighting_exp"] = &vparams.weighting_exp;
	vptrmap_d["threshold_ratio"] = &vparams.threshold_ratio;
	vptrmap_n["N_0"] = &vparams.N_0;
	vptrmap_n["S_0"] = &vparams.S_0;
	vptrmap_n["num_users"] = &vparams.num_users;
	vector<string> keys({"user_cost", "user_alpha", "cf_a", "cf_b", "cf_c", 
		"cf_alpha", "cf_beta", "cf_gamma", "cf_t", "weighting_exp", 
		"threshold_ratio", "N_0", "S_0", "num_users"});
	vector<int> sizes;
	for (string key: keys) {
		if (pptrmap_d.count(key)) {
			sizes.push_back(vptrmap_d[key]->size());
		} else if (pptrmap_n.count(key)) {
			sizes.push_back(vptrmap_n[key]->size());
		}
		//cout << key << " " << sizes[sizes.size()-1] << endl;
	}
	params.type = vparams.type;
	gensol_recsearch(params, keys, sizes, 0, vptrmap_d, vptrmap_n, pptrmap_d, pptrmap_n, out);	
}

int main(int argc, char **argv) {
	if (argc < 2) {
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
	}
	return 0;
}

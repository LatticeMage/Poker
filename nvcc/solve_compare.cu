// Compares two sets of solved heads-up trees (the JSON of solve.cu): aggregate frequencies per depth and the
// hands whose play changes most. Host only; used to write the README table of the realisation run.
//
//   nvcc -O3 -arch=sm_89 -o solve_compare solve_compare.cu
//   ./solve_compare data/out headsup-%s.json data/out headsup-%s-real.json 5,10,20,40,100 [hands-at-depths]
//
// Arguments: dirA patternA dirB patternB stacks [depths for the per-hand lists, comma separated]. %s is the
// stack. Frequencies are weighted by combos (class sizes 6 / 4 / 12 out of 1326). A node a hand never reaches
// is not in the file; for the small blind's first decision and the big blind's first decisions every hand
// reaches it, so the aggregates below are complete.

#include "poker.cuh"
#include "ranges.cuh"

typedef std::vector<double> Vec;

// per class: sum of the frequencies of the actions whose label starts with `prefix` (NaN if the node or the
// class is missing). prefix "" matches every action but fold, "fold" matches fold.
static Vec share(const Json& nodes, const std::string& node, const std::vector<std::string>& prefixes) {
	Vec out(NCLASS, NAN);
	if (!nodes.has(node)) return out;
	const Json& nd = nodes.at(node);
	const Json& acts = nd.at("actions");
	for (int h = 0; h < NCLASS; h++) {
		if (!nd.at("freq").has(g_names[h])) continue;
		const Json& f = nd.at("freq").at(g_names[h]);
		double s = 0;
		for (size_t a = 0; a < acts.arr.size(); a++)
			for (const std::string& p : prefixes)
				if (acts.arr[a].str.compare(0, p.size(), p) == 0) s += f.arr[a].num;
		out[h] = s;
	}
	return out;
}

static double mean(const Vec& v) {  // combo weighted, NaN if any class is missing
	double s = 0, w = 0;
	for (int h = 0; h < NCLASS; h++) {
		if (std::isnan(v[h])) return NAN;
		s += v[h] * g_ncombo[h];
		w += g_ncombo[h];
	}
	return s / w;
}

struct Row {
	const char* name;
	std::string node;
	std::vector<std::string> prefixes;
};

static const Row ROWS[] = {
    {"SB folds", "root", {"fold"}},
    {"SB limps", "root", {"limp"}},
    {"SB raises (not all-in)", "root", {"raise"}},
    {"SB shoves", "root", {"allin"}},
    {"BB defends the 2.5 open", "open/2.5", {"call", "raise", "allin"}},
    {"BB 3-bets the 2.5 open", "open/2.5", {"raise", "allin"}},
    {"BB calls a shove", "open/allin", {"call"}},
    {"BB raises a limp", "limp", {"raise", "allin"}},
};

static std::string pct(double x) {
	if (std::isnan(x)) return "-";
	char b[16];
	snprintf(b, sizeof b, "%.1f", 100 * x);
	return b;
}

// how often the small blind makes the 2.5 open
static double opens(const Json& j) {
	return mean(share(j.at("nodes"), "root", {"raise 2.5"}));
}

static std::string path_of(const std::string& dir, const std::string& pattern, const std::string& stack) {
	std::string f = pattern;
	const size_t at = f.find("%s");
	f.replace(at, 2, stack);
	return dir + "/" + f;
}

int main(int argc, char** argv) {
	if (argc < 6) { fprintf(stderr, "usage: solve_compare dirA patternA dirB patternB stacks [hand-list depths]\n"); return 1; }
	setup_classes();
	std::vector<std::string> stacks, lists;
	auto split = [](const std::string& s) {
		std::vector<std::string> out;
		size_t from = 0;
		while (from < s.size()) {
			size_t to = s.find(',', from);
			if (to == std::string::npos) to = s.size();
			out.push_back(s.substr(from, to - from));
			from = to + 1;
		}
		return out;
	};
	stacks = split(argv[5]);
	if (argc > 6) lists = split(argv[6]);

	printf("| Depth | Quantity | Old formula | Table | Change (points) |\n|---|---|---|---|---|\n");
	for (const std::string& s : stacks) {
		const Json a = json_load(path_of(argv[1], argv[2], s)), b = json_load(path_of(argv[3], argv[4], s));
		bool first = true;
		for (const Row& r : ROWS) {
			double x = mean(share(a.at("nodes"), r.node, r.prefixes)), y = mean(share(b.at("nodes"), r.node, r.prefixes));
			// the big blind's answers to an open are only meaningful where the open is played
			if (r.node == "open/2.5" && (opens(a) < 0.01 || opens(b) < 0.01)) x = y = NAN;
			if (std::isnan(x) && std::isnan(y)) continue;
			if (!std::isnan(x) && !std::isnan(y) && x == 0 && y == 0) continue;
			char d[16] = "-";
			if (!std::isnan(x) && !std::isnan(y)) snprintf(d, sizeof d, "%+.1f", 100 * (y - x));
			printf("| %s | %s | %s | %s | %s |\n", first ? (s + " bb").c_str() : "", r.name, pct(x).c_str(), pct(y).c_str(), d);
			first = false;
		}
	}

	for (const std::string& s : lists) {
		const Json a = json_load(path_of(argv[1], argv[2], s)), b = json_load(path_of(argv[3], argv[4], s));
		// SB plays (1 - fold), BB defends the 2.5 open, BB 3-bets the 2.5 open
		for (int k = 0; k < 3; k++) {
			const Row& r = k == 0 ? ROWS[0] : k == 1 ? ROWS[4] : ROWS[5];
			const char* what = k == 0 ? "SB plays (limp, raise or shove)" : k == 1 ? "BB defends the 2.5 open" : "BB 3-bets the 2.5 open";
			Vec x = share(a.at("nodes"), r.node, r.prefixes), y = share(b.at("nodes"), r.node, r.prefixes);
			if (k == 0)
				for (int h = 0; h < NCLASS; h++) { x[h] = 1 - x[h]; y[h] = 1 - y[h]; }
			std::vector<std::pair<double, int> > d;
			for (int h = 0; h < NCLASS; h++)
				if (!std::isnan(x[h]) && !std::isnan(y[h])) d.push_back({y[h] - x[h], h});
			std::sort(d.begin(), d.end());
			// every class whose frequency moves by at least 25 points; a name alone means 0 -> 100% (or 100 -> 0)
			for (int dir = 1; dir >= -1; dir -= 2) {
				int count = 0;
				std::string line;
				for (int i = 0; i < (int)d.size(); i++) {
					const auto& e = dir > 0 ? d[d.size() - 1 - i] : d[i];
					if (dir * e.first < 0.25) break;
					count++;
					line += " " + g_names[e.second];
					if (std::fabs(x[e.second]) > 0.005 && std::fabs(x[e.second] - 1) > 0.005 || std::fabs(y[e.second]) > 0.005 && std::fabs(y[e.second] - 1) > 0.005)
						line += "(" + pct(x[e.second]) + "->" + pct(y[e.second]) + ")";
				}
				printf("%s bb, %s, %s %d classes:%s\n", s.c_str(), what, dir > 0 ? "plays more" : "plays less", count, line.c_str());
			}
		}
	}
	return 0;
}

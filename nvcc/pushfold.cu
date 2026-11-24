// Command line for the heads-up push/fold tables (2-20bb). Checked by test_pushfold.cu.
//
//   nvcc -O3 -fmad=false -arch=sm_89 -o pushfold pushfold.cu
//   ./pushfold --iters 200000 --out data/out/pushfold.json
//
//   --matrix FILE   169*169 float64 equity matrix (default data/equity-exact.bin)
//   --stacks A-B    stack depths in bb, a range "2-20" or a comma list (default 2-20)
//   --iters N       most CFR+ iterations (default 200000)
//   --tol X         stop early once the mixed exploitability of every stack is below X bb (default 1e-9)
//   --report N      print the exploitability every N iterations (default iters / 10)
//   --out FILE      the JSON of the tables (default data/out/pushfold.json)
//   --ckpt FILE     the relay file: state saved after every batch, a rerun carries on (needs a fixed --iters: --tol is then ignored)
//   --batch N       iterations per batch with --ckpt (default 1000)

#include "pushfold.cuh"

static double seconds_since(std::chrono::steady_clock::time_point t0) {
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static std::vector<double> parse_stacks(const std::string& s) {
	std::vector<double> out;
	const size_t dash = s.find('-');
	if (dash != std::string::npos) {
		for (int b = atoi(s.substr(0, dash).c_str()); b <= atoi(s.substr(dash + 1).c_str()); b++) out.push_back(b);
		return out;
	}
	size_t from = 0;
	while (from < s.size()) {
		size_t to = s.find(',', from);
		if (to == std::string::npos) to = s.size();
		out.push_back(atof(s.substr(from, to - from).c_str()));
		from = to + 1;
	}
	return out;
}

int main(int argc, char** argv) {
	std::string matrix = "data/equity-exact.bin", stacks = "2-20", out = "data/out/pushfold.json";
	long long iters = 200000, report = 0, batch_cap = 1000;
	std::string ckpt;
	double tol = 1e-9;
	for (int i = 1; i < argc; i++) {
		std::string a = argv[i];
		auto val = [&]() -> std::string {
			if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(1); }
			return argv[++i];
		};
		if (a == "--matrix") matrix = val();
		else if (a == "--stacks") stacks = val();
		else if (a == "--iters") iters = atoll(val().c_str());
		else if (a == "--tol") tol = atof(val().c_str());
		else if (a == "--report") report = atoll(val().c_str());
		else if (a == "--out") out = val();
		else if (a == "--ckpt") ckpt = val();
		else if (a == "--batch") batch_cap = atoll(val().c_str());
		else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
	}
	if (report <= 0) report = std::max(1ll, iters / 10);

	setup_classes();
	const std::vector<double> eq = read_matrix(matrix.c_str());
	// Every call is an all-in showdown, so the realisation heuristic never enters: the raw matrix is used.
	Problem pb = make_problem(eq, false);
	const std::vector<double> sizes = parse_stacks(stacks);
	std::vector<GameTree> trees;
	for (double s : sizes) trees.push_back(build_pushfold_tree(s));
	Solver sv(pb, trees);
	printf("%zu push/fold trees, %d nodes, exact matrix %s\n", trees.size(), sv.G, matrix.c_str());

	auto t0 = std::chrono::steady_clock::now();
	if (!ckpt.empty()) {  // relay: a fixed number of iterations, state saved after every batch
		const long long batch = pick_batch(batch_cap, report);
		if (relay_peek_done(ckpt) > 0) printf("relay file %s holds %llu of %lld iterations\n", ckpt.c_str(), (unsigned long long)relay_peek_done(ckpt), iters);
		sv.run(iters, ckpt, batch, [&](long long done) {
			if (done / report > (done - std::min(done, batch)) / report || done == iters) {
				auto ex = sv.exploitability();
				double worst = 0;
				for (auto& e : ex) worst = std::max(worst, e.first + e.second);
				printf("iter %lld  %.1fs  worst exploitability %.3e bb\n", done, seconds_since(t0), worst);
				fflush(stdout);
			}
		});
	}
	while (sv.done < iters) {
		sv.iterate(std::min(iters, (sv.done / report + 1) * report) - sv.done);
		auto ex = sv.exploitability();
		double worst = 0;
		for (auto& e : ex) worst = std::max(worst, e.first + e.second);
		printf("iter %lld  %.1fs  worst exploitability %.3e bb\n", sv.done, seconds_since(t0), worst);
		fflush(stdout);
		if (worst < tol) break;
	}

	std::vector<PfStack> rs = collect(sv, sizes, eq);
	printf("\n stack   push%%   call%%   exploitability: mixed, pure ranges (bb per hand)\n");
	for (const PfStack& r : rs)
		printf("%5gbb  %6.1f  %6.1f   %.3e  %.3e\n", r.stack, r.push_pct, r.call_pct, r.expl_mixed, r.expl_pure);

	std::string js = pushfold_json(rs, sv.done, equity_vs_random(pb, eq));
	const size_t slash = out.rfind('/');
	if (slash != std::string::npos && system(("mkdir -p '" + out.substr(0, slash) + "'").c_str()) != 0) return 1;
	FILE* f = fopen(out.c_str(), "wb");
	if (!f) { fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
	fwrite(js.data(), 1, js.size(), f);
	fclose(f);
	printf("wrote %s after %lld iterations\n", out.c_str(), sv.done);
	return 0;
}

// Command line for the heads-up preflop solver. Checked by test_solve.cu and test_solve_external.cu.
//
//   nvcc -O3 -fmad=false -arch=sm_89 -o solve solve.cu
//   ./solve --stacks 10,20,40,80 --iters 20000 --out out
//
//   --matrix FILE     169*169 float64 equity matrix (default data/equity-exact.bin)
//   --stacks LIST     stack depths in bb, comma separated (default 10,20,40,80)
//   --iters N         CFR+ iterations (default 4000)
//   --out DIR         where headsup-<stack>.json goes (default out); --prefix NAME changes "headsup", --suffix S appends S after the stack ("-real")
//   --raw             no realised-equity heuristic: every call is a showdown on raw equity
//   --real FILE       value calls that see a flop with the realisation table (data/realization.json) instead
//                     of the formula (solve_realization.cuh says how); the JSON gets a "realisation" field
//   --real-out MODE   classes outside the table's ranges: formula (default), cap, none
//   --real-limp MODE  limped pots (no data): formula (default) or srp (borrow the single-raised factors)
//   --tree ladder     the tree of tools/solve-preflop.ts (default); --open "2.5,9,22" and --limp "4,12" are its
//                     raise ladders ("," between raises, "/" between alternative sizes of one raise)
//   --tree mult       the tree of the external solver poker-cfr: raises of 2.5x, 3x, 3.5x, 4x the bet faced
//   --tree pushfold   the push/fold tree: the small blind folds or shoves, the big blind folds or calls (pushfold.cu
//                     builds the app's push/fold tables on top of it)
//   --ckpt PREFIX    the relay file PREFIX.ckpt: the state (regrets, strategy sums) is written after EVERY batch of --batch iterations
//                    and a run started again with the same options carries on from it (bit-identical to an uninterrupted run)
//   --batch N        iterations per batch (default 1000, about a second; with --report it is the largest divisor of the report interval up to N)
//   --ckpt-every N   accepted and ignored (the relay saves after every batch)
//   --report N        print the exploitability every N iterations (default iters / 5)
//   --samples N       value written to samples_per_matchup (0 = exact matrix)

#include "solve.cuh"
#include "solve_realization.cuh"

static std::vector<std::string> split(const std::string& s, char sep) {
	std::vector<std::string> out;
	size_t from = 0;
	while (from <= s.size()) {
		size_t to = s.find(sep, from);
		if (to == std::string::npos) to = s.size();
		if (to > from) out.push_back(s.substr(from, to - from));
		from = to + 1;
	}
	return out;
}

static Levels parse_levels(const std::string& s) {
	Levels l;
	for (const std::string& level : split(s, ',')) {
		std::vector<double> sizes;
		for (const std::string& x : split(level, '/')) sizes.push_back(atof(x.c_str()));
		l.push_back(sizes);
	}
	return l;
}

static double seconds_since(std::chrono::steady_clock::time_point t0) {
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
	std::string matrix = "data/equity-exact.bin", stacks = "10,20,40,80", out = "out", prefix = "headsup", suffix;
	std::string tree = "ladder", open = "2.5,9,22", limp = "4,12", ckpt, real, real_out = "formula", real_limp = "formula";
	long long iters = 4000, batch_cap = 1000, report = 0, samples = 0;
	bool raw = false;
	for (int i = 1; i < argc; i++) {
		std::string a = argv[i];
		auto val = [&]() -> std::string {
			if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(1); }
			return argv[++i];
		};
		if (a == "--matrix") matrix = val();
		else if (a == "--stacks") stacks = val();
		else if (a == "--iters") iters = atoll(val().c_str());
		else if (a == "--out") out = val();
		else if (a == "--prefix") prefix = val();
		else if (a == "--suffix") suffix = val();
		else if (a == "--raw") raw = true;
		else if (a == "--real") real = val();
		else if (a == "--real-out") real_out = val();
		else if (a == "--real-limp") real_limp = val();
		else if (a == "--tree") tree = val();
		else if (a == "--open") open = val();
		else if (a == "--limp") limp = val();
		else if (a == "--ckpt") ckpt = val();
		else if (a == "--ckpt-every") val();
		else if (a == "--batch") batch_cap = atoll(val().c_str());
		else if (a == "--report") report = atoll(val().c_str());
		else if (a == "--samples") samples = atoll(val().c_str());
		else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
	}
	if (report <= 0) report = std::max(1ll, iters / 5);

	setup_classes();
	const std::vector<double> equity = read_matrix(matrix.c_str());
	Problem pb = make_problem(equity, !raw);
	std::vector<double> sizes;
	std::vector<GameTree> trees;
	for (const std::string& s : split(stacks, ',')) {
		sizes.push_back(atof(s.c_str()));
		trees.push_back(tree == "mult"       ? build_mult_tree(sizes.back())
		                : tree == "pushfold" ? build_pushfold_tree(sizes.back())
		                                     : build_ladder_tree(sizes.back(), parse_levels(open), parse_levels(limp)));
	}
	RealOptions ro;
	std::string model = raw ? "raw equity (no realisation)" : "formula realised equity";
	std::string tag;
	if (!real.empty()) {
		if (raw) { fprintf(stderr, "--real and --raw exclude each other\n"); return 1; }
		if (real_out == "cap") ro.out = OUT_CAP;
		else if (real_out == "formula") ro.out = OUT_FORMULA;
		else if (real_out == "none") ro.out = OUT_NONE;
		else { fprintf(stderr, "--real-out: cap, formula or none\n"); return 1; }
		if (real_limp == "srp") ro.limp = LIMP_SRP;
		else if (real_limp != "formula") { fprintf(stderr, "--real-limp: formula or srp\n"); return 1; }
		try {
			const int sets = apply_realisation(pb, equity, trees, RealTable::load(real), ro);
			model = "realisation table, " + std::to_string(sets) + " matrix sets, " + ro.describe();
		} catch (const std::exception& e) { fprintf(stderr, "%s: %s\n", real.c_str(), e.what()); return 1; }
		tag = ro.describe();
	}
	Solver sv(pb, trees);
	sv.tag = tag;
	printf("%zu trees, %d nodes, %d action slots, %d levels, %s\n", trees.size(), sv.G, sv.S, sv.nlevels(), model.c_str());
	const std::string relay_file = ckpt.empty() ? "" : ckpt + ".ckpt";
	auto t0 = std::chrono::steady_clock::now();
	const long long batch = pick_batch(batch_cap, report);
	const long long start = relay_file.empty() ? 0 : (long long)relay_peek_done(relay_file);
	if (start > 0) printf("relay file %s holds %lld of %lld iterations\n", relay_file.c_str(), start, iters);
	sv.run(iters, relay_file, batch, [&](long long done) {
		if (done / report > (done - std::min(done, batch)) / report || done == iters) {  // a report interval ended in this batch
			auto ex = sv.exploitability();
			printf("iter %lld  %.1fs  exploitability (bb):", done, seconds_since(t0));
			for (size_t i = 0; i < ex.size(); i++) printf(" %g:%.5f", sizes[i], ex[i].first + ex[i].second);
			printf("\n");
			fflush(stdout);
		}
	});
	const double secs = seconds_since(t0);
	if (sv.done > start) printf("%lld iterations in %.2fs (%.3f ms per iteration)\n", sv.done - start, secs, 1000 * secs / (sv.done - start));

	auto ex = sv.exploitability();
	auto ev = sv.value();
	std::vector<double> avg = sv.averaged();
	std::string mk = "mkdir -p '" + out + "'";
	if (system(mk.c_str()) != 0) return 1;
	for (size_t i = 0; i < trees.size(); i++) {
		const double e = ex[i].first + ex[i].second;
		std::string path = out + "/" + prefix + "-" + num(sizes[i]) + suffix + ".json";
		std::string js = sv.json((int)i, sizes[i], sv.done, samples, e, avg, g_names);
		FILE* f = fopen(path.c_str(), "wb");
		if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); return 1; }
		fwrite(js.data(), 1, js.size(), f);
		fclose(f);
		printf("%gbb: exploitability %.5f bb (SB best response %+.5f, BB %+.5f), value of SB %+.5f bb -> %s\n", sizes[i], e,
		       ex[i].first, ex[i].second, ev[i].first, path.c_str());
	}
	return 0;
}

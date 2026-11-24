// Command line for the realisation factors. Checked by test_realization.cu.
//
//   nvcc -O3 -arch=sm_89 -o realization realization.cu
//   ./realization run 200000 data/realization.json       all pot types and depths -> JSON (resumable)
//   ./realization show srp 100 20000 [all]               one pot type and depth, print a summary (and every class)
//   ./realization compare data/realization.json          how far the table is from the old heuristic
//
// The ranges are read from src/lib/poker/data (charts.json, spots.json); POKER_DATA overrides that folder.

#include "realization.cuh"

static std::string data_dir() {
	const char* e = getenv("POKER_DATA");
	return e ? e : "../src/lib/poker/data";
}

static const PotDef* find_pot(const std::vector<PotDef>& pots, const std::string& key) {
	for (const auto& p : pots)
		if (p.key == key) return &p;
	fprintf(stderr, "unknown pot type %s (srp, 3bet, 4bet)\n", key.c_str());
	exit(1);
}

static int cmd_run(long long boards, const std::string& out, uint64_t seed) {
	std::vector<PotDef> pots = make_pots(data_dir());
	std::vector<PotRun> runs;
	for (const PotDef& def : pots) {
		PotRun r = make_run(def);
		const std::string state = out + "." + def.key + ".state";
		if (load_state(state, r, boards, seed)) {
			printf("%s: resumed from %s\n", def.key.c_str(), state.c_str());
		} else {
			printf("%s: %lld boards, depths", def.key.c_str(), boards);
			for (int d : r.depths) printf(" %d", d);
			printf("\n");
			r = run_pot(def, boards, seed, true);
			save_state(state, r, boards, seed);
		}
		printf("%s: %.0fs, identity errors: probability %.1e chips %.1e invested %.1e raw %.1e\n", def.key.c_str(), r.res.secs, r.res.err_prob,
		       r.res.err_chips, r.res.err_inv, r.res.err_raw);
		runs.push_back(r);
	}
	write_json(out, runs, boards, seed, default_pol());
	printf("wrote %s\n", out.c_str());
	return 0;
}

static int cmd_show(const std::string& pot, int depth, long long boards, bool all) {
	std::vector<PotDef> pots = make_pots(data_dir());
	const PotDef* pd = find_pot(pots, pot);
	std::vector<Scn> sc = {{(float)(2 * pd->c), (float)(depth - pd->c)}};
	SimResult r = simulate(pd->ip, pd->oop, sc, boards, 1, false);
	printf("%lld boards in %.1fs; errors prob %.2e chips %.2e inv %.2e raw %.2e\n", r.boards, r.secs, r.err_prob, r.err_chips, r.err_inv, r.err_raw);
	for (int side = 0; side < 2; side++) {
		Summary m = range_summary(r, 0, side, side ? pd->oop : pd->ip);
		printf("%s range: %.0f combos, raw %.4f, factor %.4f, ev %.3f\n", side ? "OOP" : "IP", m.weight, m.raw, m.factor, m.ev);
	}
	for (int side = 0; side < 2; side++) {
		printf("%s\n", side ? "OOP" : "IP");
		for (int c = 0; c < NCLASS; c++) {
			const Cell& x = r.cells[side * NCLASS + c];
			const double w = side ? pd->oop[c] : pd->ip[c];
			if (all || c < 40 || c % 6 == 0)
				printf("%-4s w=%.2f raw=%.3f real=%.3f factor=%.3f +-%.3f ev=%.2f pot=%.1f\n", g_names[c].c_str(), w, x.raw, x.real, x.factor, x.se, x.ev, x.pot);
		}
	}
	return 0;
}

// Reads a results file and prints how the table differs from the old heuristic realisedIp(eq).
static int cmd_compare(const std::string& path) {
	Json j = json_load(path);
	printf("%-5s %5s %-4s | %9s %9s | %9s %9s %9s %9s\n", "pot", "depth", "side", "factor", "old", "mean d", "rms d", "max |d|", "rms(169)");
	for (const Json& pot : j.at("pots").arr) {
		for (const Json& d : pot.at("depths").arr) {
			for (int side = 0; side < 2; side++) {
				const char* key = side ? "oop" : "ip";
				const Json& s = d.at(key);
				std::vector<double> raw, real, w;
				for (const Json& x : s.at("raw").arr) raw.push_back(x.num);
				for (const Json& x : s.at("real").arr) real.push_back(x.num);
				for (const Json& x : pot.at(side ? "oop_range" : "ip_range").arr) w.push_back(x.num);
				OldDiff od = diff_old(side, raw, real, w);
				printf("%-5s %5d %-4s | %9.4f %9.4f | %+9.4f %9.4f %9.4f %9.4f\n", pot.at("pot").str.c_str(), (int)d.at("depth_bb").num, key, od.new_factor,
				       od.old_factor, od.mean, od.rms, od.maxabs, od.rms_all);
			}
		}
	}
	printf("factor / old: equity-weighted factor of the range (realised share of the pot / raw equity), new table vs old formula.\n");
	printf("d = new realised share - old realised share, per class (weighted by the side's range, except rms(169): all classes).\n");
	return 0;
}

int main(int argc, char** argv) {
	std::string mode = argc > 1 ? argv[1] : "";
	try {
		setup_classes();
		setup_combos();
		if (mode == "run" && argc >= 3) return cmd_run(atoll(argv[2]), argc >= 4 ? argv[3] : "data/realization.json", argc >= 5 ? strtoull(argv[4], 0, 10) : 20261124ull);
		if (mode == "show" && argc >= 5) return cmd_show(argv[2], atoi(argv[3]), atoll(argv[4]), argc > 5);
		if (mode == "compare" && argc >= 3) return cmd_compare(argv[2]);
	} catch (const std::exception& e) {
		fprintf(stderr, "error: %s\n", e.what());
		return 1;
	}
	fprintf(stderr, "usage: realization run BOARDS [out.json [seed]] | show POT DEPTH BOARDS [all] | compare file.json\n");
	return 1;
}

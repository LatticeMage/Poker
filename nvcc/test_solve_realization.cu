// Checks for the solver's use of the realisation table (solve_realization.cuh). Prints PASS and exits 0, or
// FAIL and exits 1.
//
//   nvcc -O3 -fmad=false -arch=sm_89 -o test_solve_realization test_solve_realization.cu
//   ./test_solve_realization [matrix.bin [realization.json]]
//
// 1. Which pot type each call of the ladder tree is given, which showdowns stay raw.
// 2. The matchup split: hand-computed numbers, zero-sum for every set built from the real table (every
//    out-of-range mode), factors of 1 give the raw matrices exactly.
// 3. Depth: interpolation between table depths, towards 1 with nothing behind, flat beyond the deepest; the
//    out-of-range cap.
// 4. The GPU values a profile with the table's matrices as a host enumeration does (uniform play, all stacks).
// 5. Solved runs with the table on, each out-of-range rule (5 to 150 bb): frequencies sum to 1, exploitability below a bound,
//    zero-sum, best response >= value; a table of ones solves like raw equity; short stacks stay finite.
// 6. Option off: the old formula reproduces the numbers of the TypeScript solver (same as test_solve.cu).

#include "solve.cuh"
#include "solve_realization.cuh"

static int g_fail = 0;
static void check(bool ok, const char* what) {
	printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
	if (!ok) g_fail++;
}

static std::string& g_realization_path() {
	static std::string p = "data/realization.json";
	return p;
}

static const Levels OPEN = {{2.5}, {9}, {22}}, LIMP = {{4}, {12}};

static int find_node(const GameTree& t, const std::string& id) {
	for (size_t i = 0; i < t.nodes.size(); i++)
		if (t.nodes[i].kind == DECISION && t.nodes[i].id == id) return (int)i;
	return -1;
}

// A table made by hand: srp / 3bet / 4bet at the depths of the real file, every factor `v`.
static RealTable flat_table(double v) {
	RealTable t;
	const char* names[3] = {"srp", "3bet", "4bet"};
	const double c[3] = {2.5, 9, 22};
	const std::vector<double> depths[3] = {{20, 40, 80, 100}, {20, 40, 80, 100}, {40, 80, 100}};
	for (int p = 0; p < 3; p++) {
		RealTable::Pot pot;
		pot.name = names[p];
		pot.c = c[p];
		for (int s = 0; s < 2; s++) pot.inrange[s].assign(NCLASS, 1);
		for (double d : depths[p]) {
			RealTable::Depth dp;
			dp.depth = d;
			dp.x = d - c[p];
			for (int s = 0; s < 2; s++) {
				dp.factor[s].assign(NCLASS, v);
				dp.range_factor[s] = v;
			}
			pot.depths.push_back(dp);
		}
		t.pots.push_back(pot);
	}
	return t;
}

// -- 1. pot types ---------------------------------------------------------------------------------
static void test_mapping() {
	printf("pot types of the ladder tree (100 bb)\n");
	const GameTree t = build_ladder_tree(100, OPEN, LIMP);
	auto show_after = [&](const char* id, int action) {
		const int d = find_node(t, id);
		return d < 0 ? nullptr : &t.nodes[t.nodes[d].child[action]];
	};
	const GNode* srp = show_after("open/2.5", 1);
	const GNode* b3 = show_after("open/2.5/9", 1);
	const GNode* b4 = show_after("open/2.5/9/22", 1);
	const GNode* chk = show_after("limp", 0);
	const GNode* lr = show_after("limp/4", 1);
	check(srp && srp->kind == SHOW && srp->pot == 1 && srp->c0 == 2.5, "call of an open: single-raised, 2.5 each");
	check(b3 && b3->pot == 2 && b3->c0 == 9, "call of a 3-bet: 3-bet pot, 9 each");
	check(b4 && b4->pot == 3 && b4->c0 == 22, "call of a 4-bet: 4-bet pot, 22 each");
	check(chk && chk->pot == 0 && lr && lr->pot == 0, "limped pots (check, call of a raise after a limp): 0 = no table data");
	int allin = 0, bad = 0;
	for (const GNode& n : t.nodes)
		if (n.kind == SHOW && n.allin) { allin++; if (n.pot < 0) bad++; }
	check(allin > 0 && bad == 0, "all-in showdowns exist (they stay raw: mset is forced to 0 on the device)");
	const GameTree m = build_mult_tree(100);
	int pots[4] = {0, 0, 0, 0};
	for (const GNode& n : m.nodes)
		if (n.kind == SHOW && !n.allin) pots[n.pot]++;
	check(pots[0] > 0 && pots[1] > 0 && pots[2] > 0 && pots[3] > 0, "the poker-cfr tree also has all four pot kinds");
}

// -- 2. the matchup split -------------------------------------------------------------------------
static void test_split(const Problem& pb, const std::vector<double>& eq) {
	printf("matchup split\n");
	const int n = pb.n;
	std::vector<double> fa(n, 1.0), fb(n, 1.0);
	std::vector<char> none(n, 0);
	std::vector<double> m0, m1;
	build_real_set(pb, eq, fa, fb, none, none, m0, m1);
	bool exact0 = true, exact1 = true;
	double worst1 = 0;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			if (m0[i * n + j] != pb.M[0][0][i * n + j]) exact0 = false;
			if (m1[i * n + j] != pb.M[0][1][i * n + j]) exact1 = false;
			worst1 = std::max(worst1, std::fabs(m1[i * n + j] - pb.M[0][1][i * n + j]));
		}
	check(exact0 && exact1 && worst1 == 0, "all factors 1: both matrices are the raw ones, bit for bit");

	// by hand: e = 0.6, f_ip = 1.1, f_oop = 0.9: a = 0.66, b = 0.36, share = 0.66 / 1.02
	{
		const int i = 0, j = 5;
		std::vector<double> ga(n, 1.0), gb(n, 1.0), r0, r1;
		ga[i] = 1.1;
		gb[j] = 0.9;
		std::vector<double> e2 = eq;
		e2[i * n + j] = 0.6;
		build_real_set(pb, e2, ga, gb, none, none, r0, r1);
		const double w = pb.W[i * n + j], want = 0.66 / 1.02;
		check(std::fabs(r0[i * n + j] - w * want) < 1e-12 && std::fabs(r1[j * n + i] - w * (1 - want)) < 1e-12,
		      "e = 0.6, f_ip = 1.1, f_oop = 0.9: small blind gets 0.66 / 1.02 of the pot, big blind the rest");
		// formula class: small blind's hand i takes the old formula, the big blind gets its complement
		std::vector<char> fm(n, 0);
		fm[i] = 1;
		build_real_set(pb, e2, ga, gb, fm, none, r0, r1);
		const double a = realised_ip(0.6), b = 0.9 * 0.4;
		check(std::fabs(r0[i * n + j] - w * (a / (a + b))) < 1e-12, "a class on the formula contributes realised_ip(e) in the split");
		fm[i] = 0;
		fm[j] = 1;
		build_real_set(pb, e2, ga, gb, none, fm, r0, r1);
		const double a2 = 1.1 * 0.6, b2 = 1 - realised_ip(0.6);
		check(std::fabs(r1[j * n + i] - w * (b2 / (a2 + b2))) < 1e-12, "... and 1 - realised_ip(e) for the big blind");
	}

	// zero-sum: every set built from the real table, in every mode, and a lopsided random table
	const RealTable tab = RealTable::load(g_realization_path());
	for (int out = 0; out < 3; out++) {
		Problem p2 = pb;
		p2.extra.clear();
		std::vector<GameTree> trees;
		for (double s : {5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0}) trees.push_back(build_ladder_tree(s, OPEN, LIMP));
		RealOptions ro;
		ro.out = out;
		ro.limp = LIMP_SRP;
		const int sets = apply_realisation(p2, eq, trees, tab, ro);
		double worst = 0, lo = 1, hi = 0;
		for (int k = 0; k < sets; k++)
			for (int i = 0; i < n; i++)
				for (int j = 0; j < n; j++) {
					const double w = pb.W[i * n + j];
					const double s = p2.extra[2 * k][i * n + j] + p2.extra[2 * k + 1][j * n + i];
					worst = std::max(worst, std::fabs(s - w));
					if (w > 0) {
						const double share = p2.extra[2 * k][i * n + j] / w;
						lo = std::min(lo, share);
						hi = std::max(hi, share);
					}
				}
		char msg[160];
		snprintf(msg, sizeof msg, "out-of-range mode %d: %d sets, small blind + big blind = pot (largest error %.1e), shares in [%.3f, %.3f]", out, sets, worst, lo, hi);
		check(worst < 1e-12 && lo >= 0 && hi <= 1, msg);
	}
}

// -- 3. depth, cap --------------------------------------------------------------------------------
static void test_depth() {
	printf("depth and out-of-range rules\n");
	RealTable t = flat_table(1.0);
	// class 0 (AA) in range for IP in srp, factors 1.2 at 20 bb and 1.4 at 40 bb; class 1 out of range, 1.5 / 1.6
	RealTable::Pot& srp = t.pots[0];
	srp.inrange[0][1] = 0;  // class 0 stays in range
	srp.depths[0].factor[0][0] = 1.2;
	srp.depths[1].factor[0][0] = 1.4;
	srp.depths[0].factor[0][1] = 1.5;
	srp.depths[1].factor[0][1] = 1.6;
	for (RealTable::Depth& d : srp.depths) d.range_factor[0] = 1.1;
	std::vector<double> f;
	std::vector<char> fm;
	auto at = [&](double stack, int out, int h) {
		t.factors(0, 0, stack - 2.5, out, f, fm);
		return f[h];
	};
	check(std::fabs(at(20, OUT_CAP, 0) - 1.2) < 1e-12 && std::fabs(at(40, OUT_CAP, 0) - 1.4) < 1e-12, "table depths give the table factors");
	check(std::fabs(at(30, OUT_CAP, 0) - 1.3) < 1e-12, "30 bb: halfway between 20 and 40 bb (linear in the stack behind)");
	check(std::fabs(at(10, OUT_CAP, 0) - (1 + 0.2 * 7.5 / 17.5)) < 1e-12, "10 bb: towards 1 at x = 0, 7.5 / 17.5 of the way to the 20 bb factor");
	check(std::fabs(at(2.5, OUT_CAP, 0) - 1) < 1e-12 && std::fabs(at(1, OUT_CAP, 0) - 1) < 1e-12, "nothing behind: factor 1");
	check(std::fabs(at(150, OUT_CAP, 0) - at(100, OUT_CAP, 0)) < 1e-12, "150 bb: the deepest table depth, no extrapolation");
	check(std::fabs(at(40, OUT_CAP, 1) - 1.1) < 1e-12 && std::fabs(at(20, OUT_CAP, 1) - 1.1) < 1e-12, "out of range, cap: min(table, range factor)");
	check(std::fabs(at(40, OUT_NONE, 1) - 1.6) < 1e-12, "out of range, none: the table value");
	check(RealOptions().out == OUT_FORMULA && RealOptions().limp == LIMP_FORMULA, "defaults: out-of-range classes and limped pots keep the old formula");
	t.factors(0, 0, 37.5, OUT_FORMULA, f, fm);
	check(fm[1] == 1 && fm[0] == 0, "out of range, formula: the class is flagged, in-range ones are not");
}

// -- 4. GPU against a host enumeration ------------------------------------------------------------
// Expected value of the small blind under uniform play (every action equally likely at every decision),
// computed by walking the tree on the host with the matrices of the problem.
typedef std::vector<double> Vec;
static const std::vector<double>& mat(const Problem& pb, int set, int p) {
	return set < 2 ? pb.M[set][p] : pb.extra[2 * (set - 2) + p];
}
static double host_uniform_value(const Problem& pb, const GameTree& t, int node, Vec r0, Vec r1) {
	const GNode& nd = t.nodes[node];
	const int n = pb.n;
	if (nd.kind == DECISION) {
		const double k = (double)nd.child.size();
		double sum = 0;
		for (int c : nd.child) {
			Vec a = r0, b = r1;
			for (int h = 0; h < n; h++) (nd.player == 0 ? a : b)[h] /= k;
			sum += host_uniform_value(pb, t, c, a, b);
		}
		return sum;
	}
	double total = 0;
	for (int i = 0; i < n; i++) {
		double v = 0;
		if (nd.kind == FOLD) {
			double s = 0;
			for (int j = 0; j < n; j++) s += pb.W[i * n + j] * r1[j];
			v = s * (nd.player == 0 ? -nd.c0 : nd.c1);
		} else {
			const int set = nd.allin ? 0 : (nd.mset >= 0 ? nd.mset : 1);
			double g = 0, p = 0;
			for (int j = 0; j < n; j++) {
				g += mat(pb, set, 0)[i * n + j] * r1[j];
				p += pb.W[i * n + j] * r1[j];
			}
			v = g * (2 * nd.c0) - nd.c0 * p;
		}
		total += (pb.prior[i] * r0[i] * v) / pb.wsum[i];
	}
	return total;
}

static void test_gpu_values(const Problem& base, const std::vector<double>& eq) {
	printf("GPU against a host enumeration (uniform play, table on)\n");
	const RealTable tab = RealTable::load(g_realization_path());
	Problem pb = base;
	std::vector<GameTree> trees;
	const double stacks[5] = {10, 20, 40, 100, 150};
	for (double s : stacks) trees.push_back(build_ladder_tree(s, OPEN, LIMP));
	RealOptions ro;
	ro.limp = LIMP_SRP;  // so that every kind of call uses an extra set
	apply_realisation(pb, eq, trees, tab, ro);
	Solver sv(pb, trees);
	std::vector<double> sums((size_t)sv.S * NP, 1.0);  // equal sums: the averaged strategy is uniform
	sv.set_sums(sums);
	const auto ev = sv.value();
	double worst = 0;
	for (size_t i = 0; i < trees.size(); i++) {
		const double want = host_uniform_value(pb, trees[i], 0, Vec(pb.n, 1.0), Vec(pb.n, 1.0));
		printf("       %gbb: GPU %+.9f, host %+.9f\n", stacks[i], ev[i].first, want);
		worst = std::max(worst, std::fabs(ev[i].first - want));
	}
	check(worst < 1e-9, "values agree to 1e-9 at every depth");
}

// -- 5. solved runs -------------------------------------------------------------------------------
static void test_solved(const Problem& base, const std::vector<double>& eq, int mode) {
	static const char* NAMES[3] = {"cap", "formula (default)", "none"};
	printf("solved with the table, out-of-range classes: %s (5 to 150 bb, 3000 iterations)\n", NAMES[mode]);
	const RealTable tab = RealTable::load(g_realization_path());
	Problem pb = base;
	std::vector<GameTree> trees;
	const double stacks[11] = {5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150};
	for (double s : stacks) trees.push_back(build_ladder_tree(s, OPEN, LIMP));
	RealOptions ro;
	ro.out = mode;
	const int sets = apply_realisation(pb, eq, trees, tab, ro);
	printf("       %d matrix sets\n", sets);
	Solver sv(pb, trees);
	sv.iterate(3000);
	const std::vector<double> avg = sv.averaged();
	const auto ex = sv.exploitability();
	const auto ev = sv.value();
	double worst_sum = 0;
	bool finite = true;
	for (size_t i = 0; i < trees.size(); i++)
		for (size_t k = 0; k < trees[i].nodes.size(); k++) {
			if (trees[i].nodes[k].kind != DECISION) continue;
			const int slot = sv.slot_of((int)i, (int)k);
			for (int h = 0; h < pb.n; h++) {
				double s = 0;
				for (size_t a = 0; a < trees[i].nodes[k].child.size(); a++) {
					const double v = avg[(size_t)(slot + a) * NP + h];
					if (!std::isfinite(v)) finite = false;
					s += v;
				}
				worst_sum = std::max(worst_sum, std::fabs(s - 1));
			}
		}
	printf("       largest |sum of a hand's action frequencies - 1|: %.2e\n", worst_sum);
	check(finite && worst_sum < 1e-12, "frequencies are finite and sum to 1 for every hand at every node");
	bool zs = true, br = true, low = true;
	for (size_t i = 0; i < trees.size(); i++) {
		const double e = ex[i].first + ex[i].second;
		printf("       %gbb: exploitability %.6f bb, value of the SB %+.6f bb\n", stacks[i], e, ev[i].first);
		low = low && e >= -1e-12 && e < 0.01;
		zs = zs && std::fabs(ev[i].first + ev[i].second) < 1e-9;
		br = br && ex[i].first >= ev[i].first - 1e-9 && ex[i].second >= ev[i].second - 1e-9;
	}
	check(low, "exploitability below 0.01 bb at every depth");
	check(zs, "zero-sum: the two players' values add to 0");
	check(br, "best responses are worth at least the profile");
}

// A table of ones must solve like raw equity everywhere (here: the Problem whose set 1 is raw).
static void test_flat(const std::vector<double>& eq) {
	printf("a table of ones against raw equity (20 and 40 bb, 500 iterations)\n");
	Problem raw = make_problem(eq, false);  // set 1 = raw: every call on raw equity
	Problem flat = make_problem(eq, true);
	std::vector<GameTree> t1, t2;
	for (double s : {20.0, 40.0}) {
		t1.push_back(build_ladder_tree(s, OPEN, LIMP));
		t2.push_back(build_ladder_tree(s, OPEN, LIMP));
	}
	RealOptions ro;
	ro.limp = LIMP_SRP;
	apply_realisation(flat, eq, t2, flat_table(1.0), ro);
	Solver a(raw, t1), b(flat, t2);
	a.iterate(500);
	b.iterate(500);
	const std::vector<double> sa = a.sums(), sb = b.sums();
	double worst = 0, scale = 0;
	for (size_t i = 0; i < sa.size(); i++) {
		worst = std::max(worst, std::fabs(sa[i] - sb[i]));
		scale = std::max(scale, std::fabs(sa[i]));
	}
	printf("       largest difference of the strategy sums %.2e (values up to %.3g)\n", worst, scale);
	check(worst == 0, "identical to the bit");
}

// -- 6. option off: the TypeScript numbers --------------------------------------------------------
static void test_off(const Problem& pb) {
	printf("option off: tools/solve-preflop.ts (10, 20 bb, 3000 iterations)\n");
	std::vector<GameTree> trees = {build_ladder_tree(10, OPEN, LIMP), build_ladder_tree(20, OPEN, LIMP)};
	Solver sv(pb, trees);
	sv.iterate(3000);
	const auto ex = sv.exploitability();
	const double want[2] = {0.0000396, 0.0001309};
	for (int i = 0; i < 2; i++) {
		const double e = ex[i].first + ex[i].second;
		printf("       %gbb: exploitability GPU %.7f, TypeScript %.7f\n", i ? 20.0 : 10.0, e, want[i]);
		check(std::fabs(e - want[i]) < 1e-6, "exploitability equals the TypeScript number to 1e-6");
	}
	const std::vector<double> avg = sv.averaged();
	const std::string js = sv.json(1, 20, 3000, 0, ex[1].first + ex[1].second, avg, g_names);
	check(js.find("\"AA\":[0,0.19,0.81,0]") != std::string::npos && js.find("\"T9s\":[0,0.34,0,0.66]") != std::string::npos &&
	          js.find("realisation") == std::string::npos,
	      "same root frequencies as the TypeScript file, and no extra field");
}

int main(int argc, char** argv) {
	setup_classes();
	const char* path = argc > 1 ? argv[1] : "data/equity-exact.bin";
	if (argc > 2) g_realization_path() = argv[2];
	const std::vector<double> eq = read_matrix(path);
	const Problem pb = make_problem(eq, true);
	test_mapping();
	test_split(pb, eq);
	test_depth();
	test_gpu_values(pb, eq);
	test_flat(eq);
	for (int mode : {OUT_FORMULA, OUT_CAP, OUT_NONE}) test_solved(pb, eq, mode);
	test_off(pb);
	printf(g_fail ? "FAIL (%d)\n" : "PASS\n", g_fail);
	return g_fail ? 1 : 0;
}

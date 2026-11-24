// Checks for the heads-up preflop solver (solve.cuh). Prints PASS and exits 0, or FAIL and exits 1.
//
//   nvcc -O3 -fmad=false -arch=sm_89 -o test_solve test_solve.cu && ./test_solve [matrix.bin]
//
// 1. A two-hand game that can be solved by hand: the equilibrium, the exploitability of a given profile
//    against an enumeration written independently of the solver.
// 2. The GPU against a slow recursive host implementation of the same CFR+ (same tree, same matrices).
// 3. A solved run: the strategy of every hand sums to 1 at every node, exploitability stays below a stated
//    bound, the game is zero-sum, a best response is worth at least the profile's value.
// 4. Numbers from the TypeScript solver (tools/solve-preflop.ts) on the same exact matrix: node set, reach
//    and root frequencies, SB value, exploitability.
// 5. The JSON text has the node keys and the schema fields of src/lib/poker/data/headsup-*.json.

#include "solve.cuh"

static int g_fail = 0;
static void check(bool ok, const char* what) {
	printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
	if (!ok) g_fail++;
}

// -- 1. the two-hand game -------------------------------------------------------------------------
// Hands H and L, equally likely, H beats L every time, same hands tie. Stack 2, blinds 0.5 / 1. The small
// blind folds or shoves; the big blind folds or calls. By hand: H always shoves and always calls; L never
// calls (calling loses 2 to H and ties L, folding loses 1); with L never calling a shove with L costs
// nothing against folding (-0.5 either way), so the small blind may mix. The value of the game is 0 and the
// exploitability 0.
static GameTree tiny_tree() {
	GameTree t;
	GNode root;
	root.id = "root";
	root.player = 0;
	root.labels = {"fold", "allin"};
	const int r = t.add(root);
	GNode f0;
	f0.kind = FOLD; f0.player = 0; f0.c0 = 0.5; f0.c1 = 1;
	const int a = t.add(f0);
	GNode bb;
	bb.id = "allin";
	bb.player = 1;
	bb.labels = {"fold", "call"};
	const int b = t.add(bb);
	GNode f1;
	f1.kind = FOLD; f1.player = 1; f1.c0 = 2; f1.c1 = 1;
	GNode s;
	s.kind = SHOW; s.c0 = 2; s.allin = true;
	const int c = t.add(f1), d = t.add(s);
	t.nodes[b].child = {c, d};
	t.nodes[r].child = {a, b};
	return t;
}

static Problem tiny_problem() {
	Problem pb;
	pb.n = 2;
	pb.W = {1, 1, 1, 1};
	const double eq[4] = {0.5, 1, 0, 0.5};  // eq[i*2+j]: share of the pot hand i wins against hand j
	for (int r = 0; r < 2; r++) {
		pb.M[r][0].assign(4, 0);
		pb.M[r][1].assign(4, 0);
		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++) {
				pb.M[r][0][i * 2 + j] = eq[i * 2 + j];
				pb.M[r][1][i * 2 + j] = 1 - eq[j * 2 + i];
			}
	}
	pb.prior = {0.5, 0.5};
	pb.wsum = {2, 2};
	return pb;
}

// Best-response values by direct enumeration, no solver code: x[h] = chance the small blind shoves hand h,
// y[h] = chance the big blind calls hand h. Returns what a best response wins from both seats.
static double tiny_exploitability(const double* x, const double* y) {
	const double eq[2][2] = {{0.5, 1}, {0, 0.5}};
	double sb = 0, bb = 0;
	for (int h1 = 0; h1 < 2; h1++) {
		double shove = 0;
		for (int h2 = 0; h2 < 2; h2++) shove += 0.5 * ((1 - y[h2]) * 1 + y[h2] * (4 * eq[h1][h2] - 2));
		sb += 0.5 * std::max(-0.5, shove);
	}
	for (int h2 = 0; h2 < 2; h2++) {
		double fold = 0, call = 0;
		for (int h1 = 0; h1 < 2; h1++) {
			fold += 0.5 * ((1 - x[h1]) * 0.5 + x[h1] * -1);
			call += 0.5 * ((1 - x[h1]) * 0.5 + x[h1] * -(4 * eq[h1][h2] - 2));
		}
		bb += 0.5 * std::max(fold, call);
	}
	return sb + bb;
}

static void test_tiny() {
	printf("two-hand game\n");
	Solver sv(tiny_problem(), {tiny_tree()});
	sv.iterate(2000);
	std::vector<double> avg = sv.averaged();
	const int s0 = sv.slot_of(0, 0), s1 = sv.slot_of(0, 2);
	check(avg[(size_t)(s0 + 1) * NP + 0] > 0.99, "H shoves");
	check(avg[(size_t)(s1 + 1) * NP + 0] > 0.99, "H calls");
	check(avg[(size_t)(s1 + 1) * NP + 1] < 0.05, "L does not call");
	auto ex = sv.exploitability();
	auto ev = sv.value();
	printf("       exploitability %.2e, value of the small blind %+.2e\n", ex[0].first + ex[0].second, ev[0].first);
	check(ex[0].first + ex[0].second < 1e-3, "exploitability < 0.001");
	check(std::fabs(ev[0].first) < 1e-3 && std::fabs(ev[0].first + ev[0].second) < 1e-12, "value 0, zero-sum");

	// a profile that is not an equilibrium: the solver's best response against enumeration
	const double x[2] = {0.7, 0.2}, y[2] = {0.9, 0.3};
	std::vector<double> sum((size_t)sv.S * NP, 0.0);
	sum[(size_t)(s0 + 0) * NP + 0] = 0.3; sum[(size_t)(s0 + 1) * NP + 0] = 0.7;
	sum[(size_t)(s0 + 0) * NP + 1] = 0.8; sum[(size_t)(s0 + 1) * NP + 1] = 0.2;
	sum[(size_t)(s1 + 0) * NP + 0] = 0.1; sum[(size_t)(s1 + 1) * NP + 0] = 0.9;
	sum[(size_t)(s1 + 0) * NP + 1] = 0.7; sum[(size_t)(s1 + 1) * NP + 1] = 0.3;
	sv.set_sums(sum);
	ex = sv.exploitability();
	const double want = tiny_exploitability(x, y);
	printf("       profile x=(.7,.2) y=(.9,.3): GPU %.9f, enumeration %.9f\n", ex[0].first + ex[0].second, want);
	check(std::fabs(ex[0].first + ex[0].second - want) < 1e-12 && want > 0.1, "best response matches the enumeration");
}

// -- 2. host reference ----------------------------------------------------------------------------
typedef std::vector<double> Vec;

struct Ref {
	const Problem& pb;
	const GameTree& t;
	int n;
	std::vector<std::vector<Vec> > regret, sum;
	Ref(const Problem& p, const GameTree& tr) : pb(p), t(tr), n(p.n) {
		regret.resize(t.nodes.size());
		sum.resize(t.nodes.size());
		for (size_t i = 0; i < t.nodes.size(); i++)
			if (t.nodes[i].kind == DECISION) {
				regret[i].assign(t.nodes[i].child.size(), Vec(n, 0.0));
				sum[i] = regret[i];
			}
	}
	Vec times(const std::vector<double>& m, const Vec& r) const {
		Vec out(n);
		for (int i = 0; i < n; i++) {
			double s = 0;
			for (int j = 0; j < n; j++) s += m[i * n + j] * r[j];
			out[i] = s;
		}
		return out;
	}
	void terminal(const GNode& nd, const Vec& r0, const Vec& r1, Vec& v0, Vec& v1) const {
		if (nd.kind == FOLD) {
			const Vec s0 = times(pb.W, r1), s1 = times(pb.W, r0);
			const double loss = nd.player == 0 ? nd.c0 : nd.c1;
			v0.resize(n); v1.resize(n);
			for (int h = 0; h < n; h++) {
				v0[h] = s0[h] * (nd.player == 0 ? -loss : loss);
				v1[h] = s1[h] * (nd.player == 1 ? -loss : loss);
			}
			return;
		}
		const int m = nd.allin ? 0 : 1;
		const Vec g0 = times(pb.M[m][0], r1), p0 = times(pb.W, r1), g1 = times(pb.M[m][1], r0), p1 = times(pb.W, r0);
		v0.resize(n); v1.resize(n);
		for (int h = 0; h < n; h++) {
			v0[h] = g0[h] * (2 * nd.c0) - nd.c0 * p0[h];
			v1[h] = g1[h] * (2 * nd.c0) - nd.c0 * p1[h];
		}
	}
	std::vector<Vec> strategy(int node) const {
		const size_t k = regret[node].size();
		std::vector<Vec> out(k, Vec(n));
		for (int h = 0; h < n; h++) {
			double pos = 0;
			for (size_t a = 0; a < k; a++) pos += regret[node][a][h];
			for (size_t a = 0; a < k; a++) out[a][h] = pos > 0 ? regret[node][a][h] / pos : 1.0 / k;
		}
		return out;
	}
	std::vector<Vec> averaged(int node) const {
		const size_t k = sum[node].size();
		std::vector<Vec> out(k, Vec(n));
		for (int h = 0; h < n; h++) {
			double s = 0;
			for (size_t a = 0; a < k; a++) s += sum[node][a][h];
			for (size_t a = 0; a < k; a++) out[a][h] = s > 0 ? sum[node][a][h] / s : 1.0 / k;
		}
		return out;
	}
	void cfr(int node, const Vec& r0, const Vec& r1, double tt, Vec& v0, Vec& v1) {
		const GNode& nd = t.nodes[node];
		if (nd.kind != DECISION) { terminal(nd, r0, r1, v0, v1); return; }
		const std::vector<Vec> sigma = strategy(node);
		const int p = nd.player;
		Vec vp(n, 0.0), vo(n, 0.0);
		std::vector<Vec> av;
		for (size_t a = 0; a < nd.child.size(); a++) {
			Vec ra(n), c0, c1;
			const Vec& mine = p == 0 ? r0 : r1;
			for (int h = 0; h < n; h++) ra[h] = mine[h] * sigma[a][h];
			if (p == 0) cfr(nd.child[a], ra, r1, tt, c0, c1); else cfr(nd.child[a], r0, ra, tt, c0, c1);
			const Vec& cp = p == 0 ? c0 : c1;
			const Vec& co = p == 0 ? c1 : c0;
			av.push_back(cp);
			for (int h = 0; h < n; h++) { vp[h] += sigma[a][h] * cp[h]; vo[h] += co[h]; }
		}
		const Vec& mine = p == 0 ? r0 : r1;
		for (size_t a = 0; a < nd.child.size(); a++)
			for (int h = 0; h < n; h++) {
				regret[node][a][h] = std::max(0.0, regret[node][a][h] + av[a][h] - vp[h]);
				sum[node][a][h] += tt * mine[h] * sigma[a][h];
			}
		(p == 0 ? v0 : v1) = vp;
		(p == 0 ? v1 : v0) = vo;
	}
	// best response of `hero` against the averaged strategy of the other player
	Vec best(int node, int hero, const Vec& opp) const {
		const GNode& nd = t.nodes[node];
		const Vec ones(n, 1.0);
		if (nd.kind != DECISION) {
			Vec v0, v1;
			if (hero == 0) terminal(nd, ones, opp, v0, v1); else terminal(nd, opp, ones, v0, v1);
			return hero == 0 ? v0 : v1;
		}
		if (nd.player == hero) {
			Vec b = best(nd.child[0], hero, opp);
			for (size_t a = 1; a < nd.child.size(); a++) {
				const Vec v = best(nd.child[a], hero, opp);
				for (int h = 0; h < n; h++) b[h] = std::max(b[h], v[h]);
			}
			return b;
		}
		const std::vector<Vec> sigma = averaged(node);
		Vec out(n, 0.0);
		for (size_t a = 0; a < nd.child.size(); a++) {
			Vec r(n);
			for (int h = 0; h < n; h++) r[h] = opp[h] * sigma[a][h];
			const Vec v = best(nd.child[a], hero, r);
			for (int h = 0; h < n; h++) out[h] += v[h];
		}
		return out;
	}
	double exploitability() const {
		double total = 0;
		for (int hero = 0; hero < 2; hero++) {
			const Vec v = best(0, hero, Vec(n, 1.0));
			double s = 0;
			for (int h = 0; h < n; h++) s += (pb.prior[h] * v[h]) / pb.wsum[h];
			total += s;
		}
		return total;
	}
};

static void test_reference(const Problem& pb) {
	printf("GPU against the host reference (20bb tree, 150 iterations)\n");
	Levels open = {{2.5}, {9}, {22}}, limp = {{4}, {12}};
	std::vector<GameTree> trees = {build_ladder_tree(20, open, limp)};
	Solver sv(pb, trees);
	Ref ref(pb, trees[0]);
	const int iters = 150;
	for (int t = 1; t <= iters; t++) {
		Vec v0, v1;
		ref.cfr(0, Vec(pb.n, 1.0), Vec(pb.n, 1.0), t, v0, v1);
	}
	sv.iterate(iters);
	const std::vector<double> gr = sv.regrets(), gs = sv.sums();
	double dr = 0, ds = 0, scale = 0;
	for (size_t i = 0; i < trees[0].nodes.size(); i++) {
		if (trees[0].nodes[i].kind != DECISION) continue;
		const int slot = sv.slot_of(0, (int)i);
		for (size_t a = 0; a < ref.regret[i].size(); a++)
			for (int h = 0; h < pb.n; h++) {
				const double r = ref.regret[i][a][h], s = ref.sum[i][a][h];
				dr = std::max(dr, std::fabs(gr[(size_t)(slot + a) * NP + h] - r));
				ds = std::max(ds, std::fabs(gs[(size_t)(slot + a) * NP + h] - s));
				scale = std::max(scale, std::max(r, s));
			}
	}
	printf("       largest difference: regrets %.2e, strategy sums %.2e (values up to %.3g)\n", dr, ds, scale);
	check(dr <= 1e-9 * scale && ds <= 1e-9 * scale, "regrets and strategy sums agree to 1e-9 relative");
	const double gx = sv.exploitability()[0].first + sv.exploitability()[0].second, rx = ref.exploitability();
	printf("       exploitability: GPU %.9f, host %.9f\n", gx, rx);
	check(std::fabs(gx - rx) <= 1e-9, "exploitability agrees to 1e-9");
}

// -- 3 to 5. a solved run -------------------------------------------------------------------------
// Reference numbers from `npm run solve -- 0 3000` (tools/solve-preflop.ts) with the same exact matrix
// (tools/.cache/equity-0.json written from nvcc/data/equity-exact.bin), 10 and 20 bb, 3000 iterations.
// The exploitability comes from the same script with 7 digits printed instead of 3. Our output must also be
// identical text: the node keys in the same order and the same rounded numbers (the arithmetic is the same
// operation by operation, so the whole JSON file matches the TypeScript one).
struct TsRef {
	double stack, exploit;
	std::vector<std::string> nodes;     // node keys, in file order
	std::vector<std::string> contains;  // pieces of text that must occur
};
static const TsRef TS[] = {
    {10, 0.0000396,
     {"root", "limp", "limp/4", "limp/4/allin", "limp/allin", "open/2.5", "open/allin"},
     {"\"AA\":[0,1,0,0]", "\"T9s\":[0,0.91,0,0.09]", "\"72o\":[1,0,0,0]", "\"22\":[0,0,0,1]", "\"reach\":0.323", "\"reach\":0.042"}},
    {20, 0.0001309,
     {"root", "limp", "limp/4", "limp/4/12", "limp/4/allin", "limp/allin", "open/2.5", "open/2.5/9", "open/2.5/allin", "open/allin"},
     {"\"AA\":[0,0.19,0.81,0]", "\"T9s\":[0,0.34,0,0.66]", "\"A5s\":[0,0.16,0.84,0]", "\"reach\":0.712", "\"reach\":0.166"}},
};

static void test_solved(const Problem& pb) {
	printf("solved run: 10, 20, 40, 80 bb, 3000 iterations\n");
	Levels open = {{2.5}, {9}, {22}}, limp = {{4}, {12}};
	const double stacks[4] = {10, 20, 40, 80};
	std::vector<GameTree> trees;
	for (double s : stacks) trees.push_back(build_ladder_tree(s, open, limp));
	Solver sv(pb, trees);
	sv.iterate(3000);
	const std::vector<double> avg = sv.averaged();
	const auto ex = sv.exploitability();
	const auto ev = sv.value();
	double worst_sum = 0;
	for (size_t i = 0; i < trees.size(); i++)
		for (size_t k = 0; k < trees[i].nodes.size(); k++) {
			const GNode& nd = trees[i].nodes[k];
			if (nd.kind != DECISION) continue;
			const int slot = sv.slot_of((int)i, (int)k);
			for (int h = 0; h < pb.n; h++) {
				double s = 0;
				for (size_t a = 0; a < nd.child.size(); a++) s += avg[(size_t)(slot + a) * NP + h];
				worst_sum = std::max(worst_sum, std::fabs(s - 1));
			}
		}
	printf("       largest |sum of a hand's action frequencies - 1|: %.2e\n", worst_sum);
	check(worst_sum < 1e-12, "action frequencies sum to 1 for every hand at every node");
	for (size_t i = 0; i < trees.size(); i++) {
		const double e = ex[i].first + ex[i].second;
		printf("       %gbb: exploitability %.6f bb, value of the SB %+.6f bb\n", stacks[i], e, ev[i].first);
		check(e >= -1e-12 && e < 0.01, "exploitability below 0.01 bb");
		check(std::fabs(ev[i].first + ev[i].second) < 1e-9, "zero-sum");
		check(ex[i].first >= ev[i].first - 1e-9 && ex[i].second >= ev[i].second - 1e-9, "best responses are worth at least the profile");
	}
	printf("agreement with tools/solve-preflop.ts\n");
	for (const TsRef& ref : TS) {
		size_t i = 0;
		while (stacks[i] != ref.stack) i++;
		const double e = ex[i].first + ex[i].second;
		printf("       %gbb: exploitability GPU %.7f, TypeScript %.7f\n", ref.stack, e, ref.exploit);
		check(std::fabs(e - ref.exploit) < 1e-6, "exploitability equals the TypeScript number to 1e-6");
		const std::string js = sv.json((int)i, ref.stack, 3000, 0, e, avg, g_names);
		size_t at = 0;
		bool order = true;
		for (const std::string& id : ref.nodes) {
			const size_t f = js.find("\"" + id + "\":{\"player\"", at);
			if (f == std::string::npos) { order = false; break; }
			at = f;
		}
		size_t keys = 0;
		for (size_t f = js.find("\"player\""); f != std::string::npos; f = js.find("\"player\"", f + 1)) keys++;
		check(order && keys == ref.nodes.size(), "same node keys in the same order");
		bool all = true;
		for (const std::string& piece : ref.contains)
			if (js.find(piece) == std::string::npos) { all = false; printf("       missing: %s\n", piece.c_str()); }
		check(all, "same reach and root frequencies as the TypeScript file");
		check(js.find("\"stack_bb\":") == 1 && js.find("\"iterations\":3000,\"samples_per_matchup\":0,\"exploitability_bb\":") != std::string::npos,
		      "same header fields as src/lib/poker/data/headsup-*.json");
	}
}

int main(int argc, char** argv) {
	setup_classes();
	const char* path = argc > 1 ? argv[1] : "data/equity-exact.bin";
	const Problem pb = make_problem(read_matrix(path), true);
	test_tiny();
	test_reference(pb);
	test_solved(pb);
	printf(g_fail ? "FAIL (%d)\n" : "PASS\n", g_fail);
	return g_fail ? 1 : 0;
}

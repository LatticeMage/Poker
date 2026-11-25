// Checks for flopsolve.cuh: the solver on toy games whose solution is known (one round and three rounds of betting),
// its value and exploitability against a brute-force enumeration (every pair of hands, every pure plan; the payoffs
// of the one-round tree written out by hand here, the bigger trees walked pair by pair), the GPU matrix against a
// slow independent host evaluator, the GPU solve against the host solve, card removal by cards against the dense
// matrix, and the relay.
//
//   nvcc -O3 -arch=sm_89 -o test_flopsolve test_flopsolve.cu && ./test_flopsolve
#include "flopsolve.cuh"

static int failures = 0;
static void check(bool ok, const char* what) {
	printf("  %s  %s\n", ok ? "ok " : "BAD", what);
	failures += !ok;
}

// -- slow reference evaluator (host only, shares nothing with eval7) -----------------------------------------------
static uint64_t ref5(const int* c) {
	int r[5], cnt[13] = {0};
	bool flush = true;
	for (int i = 0; i < 5; i++) { r[i] = c[i] >> 2; cnt[r[i]]++; flush = flush && (c[i] & 3) == (c[0] & 3); }
	std::sort(r, r + 5, [&](int a, int b) { return cnt[a] != cnt[b] ? cnt[a] > cnt[b] : a > b; });
	int distinct = 0, hi = -1, lo = 99;
	for (int i = 0; i < 13; i++)
		if (cnt[i]) { distinct++; hi = std::max(hi, i); lo = std::min(lo, i); }
	int straight = -1;
	if (distinct == 5) {
		if (hi - lo == 4) straight = hi;
		else if (cnt[12] && cnt[0] && cnt[1] && cnt[2] && cnt[3]) straight = 3;
	}
	int cat;
	if (straight >= 0 && flush) cat = 8;
	else if (cnt[r[0]] == 4) cat = 7;
	else if (cnt[r[0]] == 3 && cnt[r[3]] == 2) cat = 6;
	else if (flush) cat = 5;
	else if (straight >= 0) cat = 4;
	else if (cnt[r[0]] == 3) cat = 3;
	else if (cnt[r[0]] == 2 && cnt[r[2]] == 2) cat = 2;
	else if (cnt[r[0]] == 2) cat = 1;
	else cat = 0;
	uint64_t v = cat;
	if (straight >= 0) return (v << 20) | ((uint64_t)straight << 16);
	for (int i = 0; i < 5; i++) v = (v << 4) | r[i];
	return v;
}
static uint64_t ref7(const int* c) {
	uint64_t best = 0;
	for (int a = 0; a < 7; a++)
		for (int b = a + 1; b < 7; b++) {
			int h[5], n = 0;
			for (int i = 0; i < 7; i++)
				if (i != a && i != b) h[n++] = c[i];
			best = std::max(best, ref5(h));
		}
	return best;
}
// The matrix of one flop with the reference evaluator; also checks that two live hands share 990 runouts.
static std::vector<uint16_t> ref_matrix(const FsSide& O, const FsSide& I, const int flop[3], bool& all_990) {
	std::vector<int> twice((size_t)O.n * I.n, 0), cnt((size_t)O.n * I.n, 0);
	for (int t = 0; t < 52; t++)
		for (int u = t + 1; u < 52; u++) {
			auto dead = [&](int c) { return c == flop[0] || c == flop[1] || c == flop[2] || c == t || c == u; };
			if (t == flop[0] || t == flop[1] || t == flop[2] || u == flop[0] || u == flop[1] || u == flop[2]) continue;
			std::vector<uint64_t> ro(O.n, 0), ri(I.n, 0);
			auto fill = [&](const FsSide& s, std::vector<uint64_t>& r) {
				for (int i = 0; i < s.n; i++) {
					const int h0 = s.cards[i * 2], h1 = s.cards[i * 2 + 1];
					if (dead(h0) || dead(h1)) continue;
					const int c[7] = {flop[0], flop[1], flop[2], t, u, h0, h1};
					r[i] = ref7(c) + 1;
				}
			};
			fill(O, ro); fill(I, ri);
			for (int i = 0; i < O.n; i++)
				for (int j = 0; j < I.n; j++) {
					if (!ro[i] || !ri[j]) continue;
					if (O.cards[i * 2] == I.cards[j * 2] || O.cards[i * 2] == I.cards[j * 2 + 1] || O.cards[i * 2 + 1] == I.cards[j * 2] ||
					    O.cards[i * 2 + 1] == I.cards[j * 2 + 1])
						continue;
					cnt[(size_t)i * I.n + j]++;
					twice[(size_t)i * I.n + j] += ro[i] > ri[j] ? 2 : ro[i] == ri[j] ? 1 : 0;
				}
		}
	std::vector<uint16_t> T((size_t)O.n * I.n);
	for (size_t x = 0; x < T.size(); x++) {
		if (cnt[x] != 0 && cnt[x] != 990) all_990 = false;
		T[x] = cnt[x] ? (uint16_t)twice[x] : FV_NONE;
	}
	return T;
}

// -- brute force on the one-round tree: sizes and payoffs written out here, independent of FvGame ------------------
struct Tree {
	double b[2], r[2];
	bool raise[2];
};
static Tree tree_of(double spr) {
	Tree t;
	for (int k = 0; k < 2; k++) {
		t.b[k] = std::min(spr, k == 0 ? 0.33 : 0.66);
		t.raise[k] = t.b[k] < spr - 1e-9;
		t.r[k] = std::min(spr, t.b[k] + 0.5 * (1 + 2 * t.b[k]));
	}
	return t;
}
// O's payoff for one pair of hands, strategies as 13 numbers each (the record's slots).
static double pay_o(double e, const Tree& t, const double* so, const double* si) {
	double v = so[0] * si[0] * e;  // check, check
	for (int k = 0; k < 2; k++) {
		const double b = t.b[k], r = t.r[k];
		const double* o1 = so + FV_OFF[1 + k];
		const double* o3 = so + FV_OFF[3 + k];
		const double* i1 = si + FV_OFF[1 + k];
		const double* i3 = si + FV_OFF[3 + k];
		// O checks, I bets, O folds / calls / raises (I folds / calls)
		v += so[0] * si[1 + k] * (o1[1] * (e * (1 + 2 * b) - b) + o1[2] * (i3[0] * (1 + b) + i3[1] * (e * (1 + 2 * r) - r)));
		// O bets, I folds / calls / raises (O folds / calls)
		v += so[1 + k] * (i1[0] * 1 + i1[1] * (e * (1 + 2 * b) - b) + i1[2] * (o3[0] * (-b) + o3[1] * (e * (1 + 2 * r) - r)));
	}
	return v;
}
// Every pure plan of a player on the one-round tree: one action at each of its five decisions.
static std::vector<std::vector<double>> pure_plans(const Tree& t) {
	std::vector<std::vector<double>> out;
	for (int a0 = 0; a0 < 3; a0++)
		for (int a1 = 0; a1 < 3; a1++)
			for (int a2 = 0; a2 < 3; a2++)
				for (int a3 = 0; a3 < 2; a3++)
					for (int a4 = 0; a4 < 2; a4++) {
						if ((a1 == 2 && !t.raise[0]) || (a2 == 2 && !t.raise[1])) continue;
						std::vector<double> p(FV_SLOTS, 0.0);
						p[FV_OFF[0] + a0] = p[FV_OFF[1] + a1] = p[FV_OFF[2] + a2] = p[FV_OFF[3] + a3] = p[FV_OFF[4] + a4] = 1;
						out.push_back(p);
					}
	return out;
}
struct Brute { double ev_o, expl; };
static Brute brute(const FvFlop& g, const uint16_t* T, const Tree& tree) {
	const int NA = g.n[0], NC = g.n[1];
	const std::vector<double> ao = g.average(0), ai = g.average(1);
	const std::vector<std::vector<double>> plans = pure_plans(tree);
	double W = 0, ev = 0, br_o = 0, br_i = 0;
	for (int i = 0; i < NA; i++) {
		double so[FV_SLOTS];
		g.flop_slots(0, ao, i, so);
		std::vector<double> plan_v(plans.size(), 0.0);
		for (int j = 0; j < NC; j++) {
			const uint16_t t = T[(size_t)i * NC + j];
			if (t == FV_NONE) continue;
			double si[FV_SLOTS];
			g.flop_slots(1, ai, j, si);
			const double e = (double)t / FV_FULL, pw = g.w[0][i] * g.w[1][j];
			W += pw;
			ev += pw * pay_o(e, tree, so, si);
			for (size_t p = 0; p < plans.size(); p++) plan_v[p] += pw * pay_o(e, tree, plans[p].data(), si);
		}
		br_o += *std::max_element(plan_v.begin(), plan_v.end());
	}
	for (int j = 0; j < NC; j++) {
		double si[FV_SLOTS];
		g.flop_slots(1, ai, j, si);
		std::vector<double> plan_v(plans.size(), 0.0);
		for (int i = 0; i < NA; i++) {
			const uint16_t t = T[(size_t)i * NC + j];
			if (t == FV_NONE) continue;
			double so[FV_SLOTS];
			g.flop_slots(0, ao, i, so);
			const double e = (double)t / FV_FULL, pw = g.w[0][i] * g.w[1][j];
			for (size_t p = 0; p < plans.size(); p++) plan_v[p] += pw * (1 - pay_o(e, tree, so, plans[p].data()));  // the two payoffs add up to the pot
		}
		br_i += *std::max_element(plan_v.begin(), plan_v.end());
	}
	return {ev / W, ((br_o + br_i) / W - 1) / 2};
}

// Any tree, one pair of hands: O's payoff by walking the tree with the two hands' own strategies. e = the pair's
// all-in equity on the flop; ek[k] = its equity when the k-th card is the turn, or -1 if one of the hands holds it.
static double walk(const FvGame& gm, int at, double e, const double* ek, const std::vector<double>& so, int NA, int i, const std::vector<double>& si, int NC,
                   int j) {
	const FvNode& nd = gm.nodes[at];
	const double share = nd.turn < 0 ? 1.0 : 1.0 / 45;  // the chance of that turn card for two live hands
	if (nd.type == 1) return share * ((nd.turn < 0 ? e : ek[nd.turn]) * (1 + 2 * nd.c) - nd.c);
	if (nd.type == 2) return share * (nd.player == 0 ? -nd.c : 1 + nd.c);
	double v = 0;
	if (nd.type == 3) {
		for (int k = 0; k < FV_TURNS; k++)
			if (ek[k] >= 0) v += walk(gm, gm.kids[nd.kids + k], e, ek, so, NA, i, si, NC, j);
		return v;
	}
	for (int a = 0; a < nd.nact; a++) {
		if (!nd.ok[a]) continue;
		const double p = nd.player == 0 ? so[(size_t)(nd.slot + a) * NA + i] : si[(size_t)(nd.slot + a) * NC + j];
		if (p > 0) v += p * walk(gm, nd.child[a], e, ek, so, NA, i, si, NC, j);
	}
	return v;
}
static double walk_value(const FvGame& gm, const FvFlop& g, const uint16_t* T, const uint8_t* Tk = nullptr) {
	const int NA = g.n[0], NC = g.n[1];
	const std::vector<double> ao = g.average(0), ai = g.average(1);
	double W = 0, ev = 0;
	for (int i = 0; i < NA; i++)
		for (int j = 0; j < NC; j++) {
			const uint16_t t = T[(size_t)i * NC + j];
			if (t == FV_NONE) continue;
			double ek[FV_TURNS];
			if (Tk)
				for (int k = 0; k < FV_TURNS; k++) {
					const uint8_t x = Tk[((size_t)k * NA + i) * NC + j];
					ek[k] = x == FV_TNONE ? -1 : (double)x / FV_TFULL;
				}
			const double pw = g.w[0][i] * g.w[1][j];
			W += pw;
			ev += pw * walk(gm, 0, (double)t / FV_FULL, Tk ? ek : nullptr, ao, NA, i, ai, NC, j);
		}
	return ev / W;
}

static int card(const char* s) { return rank_of(s[0]) * 4 + (int)(strchr("shdc", s[1]) - "shdc"); }
static double maxdiff(const float* a, const float* b, int from, int to) {
	double d = 0;
	for (int k = from; k < to; k++) d = std::max(d, (double)fabs(a[k] - b[k]));
	return d;
}

int main() {
	setup_classes();

	// 1. a toy with a known solution: O holds one bluff-catcher, I holds the nuts (30%) or air (70%). The polar
	// player bets the biggest size s with all the nuts and s / (1 + s) as many bluffs, the catcher calls 1 / (1 + s):
	// I's value is 0.3 * (1 + 2s) / (1 + s) pots.
	for (double spr : {10.0, 0.5}) {
		const uint16_t T[2] = {0, FV_FULL};
		const FvGame gm = fv_game(spr);
		FvFlop g;
		g.init(gm, 1, 2, {1.0}, {0.3, 0.7}, nullptr, nullptr);
		g.dense = T;
		float rec[FV_K];
		double sum = 0;
		fv_solve_host(g, T, 4000, rec, &sum);
		const double s = std::min(spr, 0.66), want = 1 - 0.3 * (1 + 2 * s) / (1 + s);
		const double bluff = fv_freq(rec, 5, 2, 0, 1), value = fv_freq(rec, 5, 2, 9, 10), call = fv_freq(rec, 2, 1);
		printf("      spr %.1f: ev of O %.4f (exact %.4f), exploitability %.5f, air bets big %.3f (exact %.3f), catcher calls %.3f (exact %.3f)\n", spr,
		       rec[0], want, rec[1], bluff, 0.3 * s / (1 + s) / 0.7, call, 1 / (1 + s));
		check(fabs(rec[0] - want) < 1e-3 && rec[1] < 1e-3 && rec[1] > -1e-6, "toy game: the value is the known one and the strategy cannot be exploited");
		check(fabs(bluff - 0.3 * s / (1 + s) / 0.7) < 0.01 && value > 0.99 && fabs(call - 1 / (1 + s)) < 0.01,
		      "toy game: the nuts always bet big, air bluffs and the catcher calls at the known frequencies");
		check(fabs(sum - 1) < 1e-5, "toy game: the two values add up to the pot (to float precision: the products are floats)");
	}

	// 2. a random abstract game against brute force: value and exploitability of the average strategies
	{
		const int NA = 6, NC = 7;
		std::vector<uint16_t> T((size_t)NA * NC);
		std::vector<double> wa(NA), wc(NC);
		uint64_t z = 12345;
		auto rnd = [&]() { z = z * 6364136223846793005ull + 1442695040888963407ull; return (double)(z >> 11) / 9007199254740992.0; };
		for (auto& t : T) t = rnd() < 0.1 ? FV_NONE : (uint16_t)(rnd() * FV_FULL);
		for (auto& x : wa) x = 0.2 + rnd();
		for (auto& x : wc) x = 0.2 + rnd();
		double worst_ev = 0, worst_ex = 0, worst_sum = 0, worst_walk = 0;
		for (int iters : {3, 40, 400}) {  // far from and near the equilibrium: the two computations must agree everywhere
			const FvGame gm = fv_game(3.0);
			FvFlop g;
			g.init(gm, NA, NC, wa, wc, nullptr, nullptr);
			g.dense = T.data();
			float rec[FV_K];
			double sum = 0;
			fv_solve_host(g, T.data(), iters, rec, &sum);
			const Brute b = brute(g, T.data(), tree_of(3.0));
			worst_walk = std::max(worst_walk, fabs(rec[0] - walk_value(gm, g, T.data())));
			printf("      %3d iterations: ev of O %.6f (brute force %.6f), exploitability %.6f (brute force %.6f)\n", iters, rec[0], b.ev_o, rec[1], b.expl);
			worst_ev = std::max(worst_ev, fabs(rec[0] - b.ev_o));
			worst_ex = std::max(worst_ex, fabs(rec[1] - b.expl));
			worst_sum = std::max(worst_sum, fabs(sum - 1));
			if (iters == 400) check(rec[1] < 2e-3, "after 400 iterations the random game is solved to under 0.2% of the pot");
		}
		check(worst_ev < 1e-6, "the value equals the sum over every pair of hands of the tree's payoffs");
		check(worst_ex < 1e-6, "the exploitability equals the one from trying every pure plan of each hand");
		check(worst_sum < 1e-5, "the two players' values add up to the pot");
		check(worst_walk < 1e-6, "walking the tree pair by pair gives the same value");

		// the same with two and three rounds of betting: value against the walk, and how well it gets solved
		for (int streets : {2, 3}) {
			const FvGame gm = fv_game(6.0, streets);
			FvFlop g;
			g.init(gm, NA, NC, wa, wc, nullptr, nullptr);
			g.dense = T.data();
			float rec[FV_K];
			double sum = 0;
			fv_solve_host(g, T.data(), 600, rec, &sum);
			const double wv = walk_value(gm, g, T.data());
			printf("      %d rounds (%d showdown lines): ev of O %.6f (walk %.6f), exploitability %.6f\n", streets, gm.lines, rec[0], wv, rec[1]);
			check(fabs(rec[0] - wv) < 1e-6 && fabs(sum - 1) < 1e-5 && rec[1] > -1e-6 && rec[1] < 3e-3,
			      streets == 2 ? "two rounds: value equals the walk, values add up to the pot, exploitability under 0.3% of the pot"
			                   : "three rounds: value equals the walk, values add up to the pot, exploitability under 0.3% of the pot");
		}
	}

	// 2b. three rounds, toy with a known solution: the polar player bets 66% on every round; its value is
	// 0.3 * ((1 + 2s) / (1 + s))^3 pots with s = 0.66.
	{
		const uint16_t T[2] = {0, FV_FULL};
		const FvGame gm = fv_game(50.0, 3);
		FvFlop g;
		g.init(gm, 1, 2, {1.0}, {0.3, 0.7}, nullptr, nullptr);
		g.dense = T;
		float rec[FV_K];
		fv_solve_host(g, T, 6000, rec);
		const double want = 1 - 0.3 * pow(2.32 / 1.66, 3);
		printf("      three rounds: ev of O %.4f (exact %.4f), exploitability %.5f\n", rec[0], want, rec[1]);
		check(fabs(rec[0] - want) < 2e-3 && rec[1] < 2e-3 && rec[1] > -1e-6, "three-round toy game: the value is the known one");
	}

	// 3. the spots
	{
		const FvSpot a = fv_spot("srp:BTN:BB"), b = fv_spot("3bet:BTN:BB"), c = fv_spot("4bet:BTN:BB"), d = fv_spot("srp:UTG:MP"), e = fv_spot("3bet:CO:BTN");
		check(a.aggressor_ip && fabs(a.pot - 4.5) < 1e-9 && fabs(a.stack - 98) < 1e-9, "BTN opens, BB calls: the aggressor is in position, pot 4.5bb, 98bb behind");
		check(!b.aggressor_ip && fabs(b.pot - 16.5) < 1e-9 && fabs(b.stack - 92) < 1e-9, "BB 3-bets BTN: the aggressor is out of position, pot 16.5bb");
		check(c.aggressor_ip && fabs(c.pot - 36.5) < 1e-9 && fabs(c.spr - 82 / 36.5) < 1e-9, "BTN 4-bets: pot 36.5bb, 82bb behind");
		check(!d.aggressor_ip && e.aggressor_ip && fabs(e.pot - 16.5) < 1e-9, "UTG against MP is out of position; BTN 3-betting CO is in position");
		const FvGame t = fv_game(c.spr), u = fv_game(0.5), t2 = fv_game(20.0, 2), t3 = fv_game(20.0, 3);
		check(t.lines == 9 && t.slots[0] == 13 && t.slots[1] == 13 && t.folds == 8, "one round: 9 showdown lines, 8 folds, 13 action slots per player");
		check(u.lines == 7 && u.slots[0] == 11 && u.slots[1] == 11, "with half a pot behind the big bet is all-in and cannot be raised: 7 lines");
		check(t2.lines == 27 && t3.lines == 81, "two rounds: 27 showdown lines; three rounds: 81");
		check(fv_flop_ids(false).size() == 383 && fv_flop_ids(true).size() == 1755, "383 flops for one per group, 1,755 for all");
	}

	// 4. the GPU: matrix against the reference evaluator, solve against the host solve
	FvDevice dev;
	{
		std::vector<double> a = parse_weighted("QQ+,AKs,AQs,KQs,JTs,76s,AKo,55"), b = parse_weighted("TT-88,AJs,KJs,QJs,T9s,98s,AQo,A5s");
		for (int c = 0; c < NCLASS; c++) {
			if (g_names[c] == "AKo") a[c] = 0.5;
			if (g_names[c] == "A5s") b[c] = 0.25;
		}
		const FsSide O = fs_side(a), I = fs_side(b);
		const char* texts[3][3] = {{"As", "7d", "2c"}, {"Jh", "Th", "9s"}, {"Qs", "8s", "3s"}};
		uint8_t cards[9];
		int fl[3][3];
		for (int f = 0; f < 3; f++) {
			for (int x = 0; x < 3; x++) fl[f][x] = card(texts[f][x]);
			std::sort(fl[f], fl[f] + 3);
			for (int x = 0; x < 3; x++) cards[f * 3 + x] = (uint8_t)fl[f][x];
		}
		const FvGame tree = fv_game(5.0), tree3 = fv_game(5.0, 3);
		const int iters = 200;
		std::vector<float> got(3 * FV_K);
		dev.keep_matrix = true;
		dev.solve(tree, O, I, cards, 3, iters, got.data());
		std::vector<float> got3(3 * FV_K);
		dev.keep_matrix = false;
		dev.solve(tree3, O, I, cards, 3, iters, got3.data());
		double worst3 = 0, worst_expl3 = 0;
		bool same_matrix = true, all_990 = true;
		double worst = 0, worst_cards = 0, worst_expl = 0, worst_sum = 0;
		FsDevice stats;
		double fs[3 * FS_K];
		stats.run(O, I, cards, 3, fs);
		double worst_eq = 0;
		for (int f = 0; f < 3; f++) {
			const std::vector<uint16_t> T = ref_matrix(O, I, fl[f], all_990);
			for (size_t x = 0; x < T.size(); x++) same_matrix = same_matrix && T[x] == dev.h_T[(size_t)f * O.n * I.n + x];
			FvFlop g, h;
			float host[FV_K], dense[FV_K];
			double sum = 0;
			g.init(tree, O.n, I.n, fv_live(O, cards + f * 3), fv_live(I, cards + f * 3), O.cards.data(), I.cards.data());
			fv_solve_host(g, T.data(), iters, host, &sum);
			h.init(tree, O.n, I.n, fv_live(O, cards + f * 3), fv_live(I, cards + f * 3), nullptr, nullptr);
			h.dense = T.data();
			fv_solve_host(h, T.data(), iters, dense);
			FvFlop g3;
			float host3[FV_K];
			g3.init(tree3, O.n, I.n, fv_live(O, cards + f * 3), fv_live(I, cards + f * 3), O.cards.data(), I.cards.data());
			fv_solve_host(g3, T.data(), iters, host3);
			worst3 = std::max(worst3, maxdiff(host3, &got3[f * FV_K], 0, FV_K));
			worst_expl3 = std::max(worst_expl3, (double)got3[f * FV_K + 1]);
			worst = std::max(worst, maxdiff(host, &got[f * FV_K], 0, FV_K));
			worst_cards = std::max(worst_cards, maxdiff(host, dense, 0, FV_K));
			worst_expl = std::max(worst_expl, (double)got[f * FV_K + 1]);
			worst_sum = std::max(worst_sum, fabs(sum - 1));
			worst_eq = std::max(worst_eq, fabs(got[f * FV_K + 2] - fs[f * FS_K]));
			printf("      %s%s%s: ev of O %.4f, all-in equity %.4f, exploitability %.5f; O bets %.3f, I bets after a check %.3f\n", texts[f][0], texts[f][1],
			       texts[f][2], got[f * FV_K], got[f * FV_K + 2], got[f * FV_K + 1], 1 - fv_freq(&got[f * FV_K], 0, 0), 1 - fv_freq(&got[f * FV_K], 5, 0));
		}
		printf("      GPU against host: largest difference in a record %.2e; cards against dense matrix %.2e\n", worst, worst_cards);
		printf("      values add up to the pot within %.2e; all-in equity differs from flopstats by at most %.2e\n", worst_sum, worst_eq);
		check(same_matrix && all_990, "the GPU matrix equals the reference evaluator's on three flops, and two live hands always share 990 runouts");
		check(worst < 2e-3, "the GPU solve gives the host solve's record (float against double products)");
		printf("      three rounds: GPU against host %.2e, largest exploitability %.5f\n", worst3, worst_expl3);
		check(worst3 < 2e-3 && worst_expl3 < 1e-2, "three rounds: the GPU solve gives the host solve's record, under 1% of the pot to exploit after 200 iterations");
		check(worst_cards < 1e-6, "card removal counted by cards equals card removal read from the matrix");
		check(worst_expl < 5e-3, "200 iterations leave under 0.5% of the pot to exploit on these flops");
		check(worst_sum < 1e-5 && worst_eq < 1e-5, "values add up to the pot, and the range's all-in equity is the one flopstats computes");
	}

	// 4b. the real turn card: 49 matrices against the reference evaluator, the solve against the host solve and the walk
	{
		const FsSide O = fs_side(parse_weighted("QQ-TT,AQs,KQs,JTs,76s,AKo")), I = fs_side(parse_weighted("99-88,AJs,KJs,T9s,98s,AQo"));
		int fl[3] = {card("As"), card("7d"), card("2c")};
		std::sort(fl, fl + 3);
		const uint8_t cards[3] = {(uint8_t)fl[0], (uint8_t)fl[1], (uint8_t)fl[2]};
		const FvGame tree = fv_game(8.0, 2, true);
		check(tree.turn && tree.tlines == 27 && tree.lines == 0, "flop and turn: 27 showdown lines after each of the 49 turn cards, none before");
		float got[FV_K];
		dev.keep_matrix = true;
		dev.solve(tree, O, I, cards, 1, 100, got);
		dev.keep_matrix = false;
		int tc[FV_TURNS], nt = 0;
		for (int c = 0; c < 52; c++)
			if (c != fl[0] && c != fl[1] && c != fl[2]) tc[nt++] = c;
		auto ranks = [&](const FsSide& sd) {  // strength of every hand on every turn and river, 0 if it holds one of them
			std::vector<uint64_t> r((size_t)sd.n * 49 * 49, 0);
			for (int i = 0; i < sd.n; i++)
				for (int a = 0; a < 49; a++)
					for (int b = a + 1; b < 49; b++) {
						const int h0 = sd.cards[i * 2], h1 = sd.cards[i * 2 + 1];
						bool dead = false;
						for (int x : {fl[0], fl[1], fl[2], tc[a], tc[b]}) dead = dead || h0 == x || h1 == x;
						if (dead) continue;
						const int c7[7] = {fl[0], fl[1], fl[2], tc[a], tc[b], h0, h1};
						r[((size_t)i * 49 + a) * 49 + b] = ref7(c7) + 1;
					}
			return r;
		};
		const std::vector<uint64_t> ro = ranks(O), ri = ranks(I);
		std::vector<uint8_t> Tk((size_t)FV_TURNS * O.n * I.n, FV_TNONE);
		bool all_44 = true, all_990 = true, sums = true;
		const std::vector<uint16_t> T = ref_matrix(O, I, fl, all_990);
		for (int i = 0; i < O.n; i++)
			for (int j = 0; j < I.n; j++) {
				if (O.cards[i * 2] == I.cards[j * 2] || O.cards[i * 2] == I.cards[j * 2 + 1] || O.cards[i * 2 + 1] == I.cards[j * 2] ||
				    O.cards[i * 2 + 1] == I.cards[j * 2 + 1])
					continue;
				int all = 0, live_turns = 0;
				for (int k = 0; k < FV_TURNS; k++) {
					int twice = 0, n = 0;
					for (int u = 0; u < FV_TURNS; u++) {
						if (u == k) continue;
						const int a = std::min(k, u), b = std::max(k, u);
						const uint64_t x = ro[((size_t)i * 49 + a) * 49 + b], y = ri[((size_t)j * 49 + a) * 49 + b];
						if (!x || !y) continue;
						n++;
						twice += x > y ? 2 : x == y ? 1 : 0;
					}
					if (n) { Tk[((size_t)k * O.n + i) * I.n + j] = (uint8_t)twice; all += twice; live_turns++; }
					all_44 = all_44 && (n == 0 || n == 44);
				}
				// every runout is counted once as (turn, river) and once the other way round
				if (T[(size_t)i * I.n + j] != FV_NONE) sums = sums && all == 2 * T[(size_t)i * I.n + j] && live_turns == 45;
			}
		check(Tk == dev.h_Tk && all_44, "the 49 GPU turn matrices equal the reference evaluator's, and two live hands always share 44 rivers");
		check(sums, "over the 45 turn cards two hands can see, the turn matrices add up to twice the flop matrix");
		FvFlop g;
		float host[FV_K];
		double sum = 0;
		g.init(tree, O.n, I.n, fv_live(O, cards), fv_live(I, cards), O.cards.data(), I.cards.data(), cards);
		fv_solve_host(g, T.data(), 100, host, &sum, Tk.data());
		const double d = maxdiff(host, got, 0, FV_K), wv = walk_value(tree, g, T.data(), Tk.data());
		printf("      turn card: ev of O %.5f (walk %.5f), exploitability %.5f, GPU against host %.2e, values add up within %.2e\n", host[0], wv, host[1], d,
		       fabs(sum - 1));
		check(fabs(host[0] - wv) < 1e-6, "turn card: the value equals the walk over every pair of hands and every turn card");
		check(d < 2e-3 && fabs(sum - 1) < 1e-5 && host[1] > -1e-6 && host[1] < 2e-2, "turn card: the GPU solve gives the host solve's record, values add up to the pot");
	}

	// 5. renaming the suits changes (almost) nothing
	{
		const FsSide O = fs_side(parse_weighted("JJ-22,AQs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,AQo-AJo,KQo")),
		             I = fs_side(parse_weighted("77+,ATs+,KTs+,QTs+,JTs,AJo+,KQo"));
		int f1[3] = {card("Kh"), card("9h"), card("6c")}, f2[3] = {card("Kd"), card("9d"), card("6s")};
		std::sort(f1, f1 + 3); std::sort(f2, f2 + 3);
		const uint8_t cards[6] = {(uint8_t)f1[0], (uint8_t)f1[1], (uint8_t)f1[2], (uint8_t)f2[0], (uint8_t)f2[1], (uint8_t)f2[2]};
		std::vector<float> got(2 * FV_K);
		const FvGame tree = fv_game(20.0, 3);
		dev.solve(tree, O, I, cards, 2, 200, got.data());
		const double d = maxdiff(&got[0], &got[FV_K], 0, FV_K);
		printf("      Kh9h6c against Kd9d6s: largest difference %.2e; ev of O %.4f, exploitability %.5f\n", d, got[0], got[1]);
		check(d < 2e-3, "Kh9h6c and Kd9d6s give the same record");
	}

	// 6. the relay: stopped twice and carried on equals an uninterrupted run; other inputs do not use the file
	{
		FsMatchup m;
		m.key = "srp:BTN:BB";
		m.aggressor = parse_weighted("TT+,AQs+,KQs,AKo");
		m.caller = parse_weighted("99-66,AJs,KJs,QJs,JTs");
		std::vector<int> ids;
		for (int i = 0; i < 150; i++) ids.push_back(i * 11);
		const std::string path = "/tmp/test_flopsolve.relay";
		remove(path.c_str());
		uint64_t done = 0;
		fv_run(dev, m, ids, 30, 1, path, false, 1, true, &done);
		const uint64_t first = done;
		fv_run(dev, m, ids, 30, 1, path, false, 1, true, &done);
		const uint64_t second = done;
		const std::vector<float> relayed = fv_run(dev, m, ids, 30, 1, path, false, 0, true, &done);
		const std::vector<float> straight = fv_run(dev, m, ids, 30, 1, "", false);
		double worst = 0;
		for (size_t i = 0; i < straight.size(); i++) worst = std::max(worst, (double)fabs(straight[i] - relayed[i]));
		const uint64_t per = fv_batch(fs_side(m.caller).n, fs_side(m.aggressor).n);
		check(first == per && second == 2 * per && done == ids.size(), "stopped after one batch, after two, then finished all 150 flops");
		check(worst == 0, "the relayed result is identical to the uninterrupted one");
		fv_run(dev, m, ids, 31, 1, path, false, 1, true, &done);
		check(done == per, "a relay file for another number of iterations is ignored");
		remove(path.c_str());
	}

	printf(failures ? "FAIL\n" : "PASS\n");
	return failures ? 1 : 0;
}

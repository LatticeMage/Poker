// Checks for the 6-max solver (sixmax.cuh). Prints PASS and exits 0, or FAIL and exits 1.
//
//   nvcc -O3 -arch=sm_89 -o test_sixmax test_sixmax.cu && ./test_sixmax
//
// Run it from nvcc/ (it reads data/equity-exact.bin, data/equity3.bin, data/realization.json and the ranges in
// ../src/lib/poker/data, or POKER_DATA). About two minutes.
//
// 1. A three-player game small enough to solve by hand (two hand classes, no card removal): the value of a
//    named profile worked out on paper, the engine against an enumeration of all deals written separately, a
//    best response against a separate best-response search, and the game being exactly zero-sum.
// 2. The real 100bb tree: card-removal tables (probabilities sum to 1, symmetric, known values), the
//    three-way table in the engine's layout, the pot-share matrices.
// 3. The engine against an independent host evaluator on the real tree (random profile, 8 chosen classes):
//    every player's value, and the best-response value of three players (the best-response gap).
// 4. A short real run: strategies sum to 1, frozen decisions never move, the result is bit-identical when
//    run twice, the zero-sum defect is small, the JSON parses and agrees with the strategies.
// 5. The relay: a run stopped and resumed (a new start each time, a half-written temp file left behind, an old-format
//    checkpoint converted) equals an uninterrupted run bit for bit.

#include "sixmax_run.cuh"

static int g_fail = 0;
static void check(bool ok, const char* what) {
	printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
	if (!ok) g_fail++;
}

// -- an independent host model --------------------------------------------------------------------------
// It is the same game (the same formulas are the specification), written as plain recursion over per-class
// vectors with direct table lookups, so that the kernels, the transposed matrices and the tensors are checked.
struct Ref {
	int n = 0;
	std::vector<double> Wn;
	std::vector<double> S[NCAT];
	std::vector<double> grole[3];
	std::function<double(int, int, int, double*)> e3;  // classes of slots 0, 1, 2 -> shares, returns the number of combo triples
	std::vector<double> NS;
};

typedef std::vector<std::vector<double> > VV;  // [player][class]

struct RefRun {
	const Ref& R;
	const std::vector<Node6>& nd;
	const std::vector<std::vector<std::vector<double> > >& sigma;  // [node][action][class]
	std::vector<int> H;                                            // the classes that are evaluated
	int hero;                                                      // -1: value of the profile; else: best response of this player

	VV terminal(const Node6& t, const VV& reach) const {
		const int n = R.n;
		VV val(NPLAYER, std::vector<double>(n, 0.0));
		double pot = 0;
		for (int q = 0; q < NPLAYER; q++) pot += t.contrib[q];
		const int nl = __builtin_popcount(t.live);
		for (int p = 0; p < NPLAYER; p++) {
			if (hero >= 0 && p != hero) continue;
			for (int h : H) {
				double m[NPLAYER];
				for (int q = 0; q < NPLAYER; q++) {
					m[q] = 0;
					for (int j = 0; j < n; j++) m[q] += R.Wn[(size_t)h * n + j] * reach[q][j];
				}
				const bool in = t.live >> p & 1;
				double prod = 1, v = 0;
				if (nl == 1 || !in) {
					for (int q = 0; q < NPLAYER; q++)
						if (q != p) prod *= m[q];
					v = ((nl == 1 && in) ? pot - t.contrib[p] : -t.contrib[p]) * prod;
				} else {
					for (int q = 0; q < NPLAYER; q++)
						if (!(t.live >> q & 1)) prod *= m[q];
					if (nl == 2) {
						int w = -1;
						for (int q = 0; q < NPLAYER; q++)
							if ((t.live >> q & 1) && q != p) w = q;
						double acc = 0;
						for (int j = 0; j < n; j++) {
							const double share = p == t.ip ? R.S[t.cat][(size_t)h * n + j] : 1 - R.S[t.cat][(size_t)j * n + h];
							acc += R.Wn[(size_t)h * n + j] * reach[w][j] * (pot * share - t.contrib[p]);
						}
						v = prod * acc;
					} else {
						int sl[3], k = 0, role[3];
						for (int q = 0; q < NPLAYER; q++)
							if (t.live >> q & 1) sl[k++] = q;
						int s = 0;
						for (int a = 0; a < 3; a++) {
							if (sl[a] == p) s = a;
							int rank = 0;
							for (int b = 0; b < 3; b++)
								if (POSTPOS[sl[b]] < POSTPOS[sl[a]]) rank++;
							role[a] = rank;
						}
						const int xs = s == 0 ? 1 : 0, ys = s == 2 ? 1 : 2;
						double acc = 0;
						for (int x = 0; x < n; x++) {
							if (reach[sl[xs]][x] == 0) continue;
							for (int y = 0; y < n; y++) {
								int cls[3];
								cls[s] = h;
								cls[xs] = x;
								cls[ys] = y;
								double E[3];
								const double N = R.e3(cls[0], cls[1], cls[2], E);
								if (N == 0) continue;
								double d = 0;
								for (int q = 0; q < 3; q++) d += E[q] * R.grole[role[q]][cls[q]];
								const double share = d > 0 ? E[s] * R.grole[role[s]][h] / d : 0;
								acc += reach[sl[xs]][x] * reach[sl[ys]][y] * N * (pot * share - t.contrib[p]);
							}
						}
						v = prod * acc / R.NS[h];
					}
				}
				val[p][h] = v;
			}
		}
		return val;
	}

	VV go(int g, const VV& reach) const {
		const Node6& t = nd[g];
		if (t.is_term()) return terminal(t, reach);
		const int p = t.player, k = (int)t.child.size();
		VV out(NPLAYER, std::vector<double>(R.n, 0.0));
		std::vector<VV> cv;
		for (int a = 0; a < k; a++) {
			VV r = reach;
			for (int h : H) (void)h;
			for (int j = 0; j < R.n; j++) r[p][j] = reach[p][j] * sigma[g][a][j];
			cv.push_back(go(t.child[a], r));
		}
		for (int h : H) {
			for (int q = 0; q < NPLAYER; q++) {
				if (hero >= 0 && q != hero) continue;
				if (q == p && hero < 0) {
					double s = 0;
					for (int a = 0; a < k; a++) s += sigma[g][a][h] * cv[a][q][h];
					out[q][h] = s;
				} else if (q == p && hero == q) {
					double best = cv[0][q][h];
					for (int a = 1; a < k; a++) best = std::max(best, cv[a][q][h]);
					out[q][h] = best;
				} else {
					double s = 0;
					for (int a = 0; a < k; a++) s += cv[a][q][h];
					out[q][h] = s;
				}
			}
		}
		return out;
	}
};

// Strategy slots from a [node][action][class] array.
static std::vector<double> to_slots(const Solver6& sol, const std::vector<std::vector<std::vector<double> > >& sg) {
	std::vector<double> s((size_t)sol.S * NPAD, 0.0);
	for (size_t i = 0; i < sol.nodes.size(); i++) {
		if (sol.nodes[i].is_term()) continue;
		for (size_t a = 0; a < sg[i].size(); a++)
			for (int h = 0; h < sol.pb.n; h++) s[(size_t)(sol.hn[i].aoff + a) * NPAD + h] = sg[i][a][h];
	}
	return s;
}

static std::vector<std::vector<std::vector<double> > > random_profile(const std::vector<Node6>& nodes, int n, uint64_t seed) {
	std::vector<std::vector<std::vector<double> > > sg(nodes.size());
	uint64_t s = seed;
	auto rnd = [&]() {
		s = s * 6364136223846793005ull + 1442695040888963407ull;
		return ((s >> 33) & 0xffffff) / 16777216.0;
	};
	for (size_t i = 0; i < nodes.size(); i++) {
		if (nodes[i].is_term()) continue;
		const size_t k = nodes[i].child.size();
		sg[i].assign(k, std::vector<double>(n, 0.0));
		for (int h = 0; h < n; h++) {
			double tot = 0;
			for (size_t a = 0; a < k; a++) {
				const double w = 0.05 + rnd();
				sg[i][a][h] = w;
				tot += w;
			}
			for (size_t a = 0; a < k; a++) sg[i][a][h] /= tot;
		}
	}
	return sg;
}

// Engine values at the root for a fixed profile: value[p][h]; best[p][h] for the listed heroes.
static void engine_eval(Solver6& sol, const std::vector<std::vector<std::vector<double> > >& sg, VV& value, std::vector<int> heroes, VV& best) {
	sol.set_sigma(to_slots(sol, sg));
	sol.sweep_down(sol.stream, 0, -1);
	sol.sweep_up(sol.stream, 1, 0);
	CK(cudaStreamSynchronize(sol.stream));
	value.assign(NPLAYER, std::vector<double>());
	for (int p = 0; p < NPLAYER; p++) value[p] = sol.root_vec(p);
	best.assign(NPLAYER, std::vector<double>());
	for (int p : heroes) {
		sol.sweep_down(sol.stream, 0, p);
		sol.sweep_up(sol.stream, 2, p);
		CK(cudaStreamSynchronize(sol.stream));
		best[p] = sol.root_vec(p);
	}
}

// -- 1. the three-player game ----------------------------------------------------------------------------
// Seats 0, 1, 2 (the other seats never act). Classes H and L, each player holds H or L with probability 1/2,
// independently. Everybody has put 1 in. Player 0 folds or bets (puts 2 in); player 1, then player 2 fold or call.
// The best class wins a showdown, equal classes split. If player 0 folds, players 1 and 2 show down the antes.
static std::vector<Node6> three_player_tree() {
	std::vector<Node6> t;
	auto add = [&](Node6 n) { t.push_back(n); return (int)t.size() - 1; };
	auto term = [&](int live, double c0, double c1, double c2, int cat) {
		Node6 x;
		x.type = NT_TERM;
		x.live = live;
		x.contrib[0] = c0; x.contrib[1] = c1; x.contrib[2] = c2;
		x.cat = cat;
		if (__builtin_popcount(live) == 2) {
			int a = -1, b = -1;
			for (int p = 0; p < NPLAYER; p++)
				if (live >> p & 1) (a < 0 ? a : b) = p;
			x.ip = POSTPOS[a] > POSTPOS[b] ? a : b;
			x.oop = x.ip == a ? b : a;
		}
		return add(x);
	};
	auto dec = [&](int p) {
		Node6 d;
		d.type = NT_RESP;
		d.player = p;
		return add(d);
	};
	auto act = [&](int d, ActKind k, int c) {
		t[d].akind.push_back(k);
		t[d].labels.push_back(k == A_FOLD ? "fold" : "go");
		t[d].child.push_back(c);
	};
	const int n0 = dec(0);
	act(n0, A_FOLD, term(6, 1, 1, 1, CAT_SRP));
	const int n1 = dec(1);
	act(n0, A_RAISE, n1);
	const int p2a = dec(2);  // after player 1 folds
	act(n1, A_FOLD, p2a);
	act(p2a, A_FOLD, term(1, 2, 1, 1, -1));
	act(p2a, A_CALL, term(5, 2, 1, 2, CAT_SRP));
	const int p2b = dec(2);  // after player 1 calls
	act(n1, A_CALL, p2b);
	act(p2b, A_FOLD, term(3, 2, 2, 1, CAT_SRP));
	act(p2b, A_CALL, term(7, 2, 2, 2, CAT_SRP));
	return t;
}

static void tiny_ref_and_problem(Problem6& pb, Ref& ref) {
	const int n = 2;
	pb.n = n;
	pb.prior = {0.5, 0.5};
	pb.Wn.assign(4, 0.5);
	std::vector<double> S = {0.5, 1.0, 0.0, 0.5};  // S[i][j]: the in-position player with i takes this share against j (0 = H)
	// class 0 = H, class 1 = L: H beats L
	S = {0.5, 1.0, 0.0, 0.5};
	pb.S[CAT_SRP] = S;
	pb.A_ip[CAT_SRP].assign(4, 0);
	pb.A_oop[CAT_SRP].assign(4, 0);
	for (int h = 0; h < n; h++)
		for (int j = 0; j < n; j++) {
			pb.A_ip[CAT_SRP][h * n + j] = 0.5 * S[h * n + j];
			pb.A_oop[CAT_SRP][h * n + j] = 0.5 * S[j * n + h];
		}
	auto shares = [](int c0, int c1, int c2, double* E) {  // lower class number wins
		const int best = std::min(c0, std::min(c1, c2));
		const int cls[3] = {c0, c1, c2};
		int cnt = 0;
		for (int q = 0; q < 3; q++) cnt += cls[q] == best;
		for (int q = 0; q < 3; q++) E[q] = cls[q] == best ? 1.0 / cnt : 0.0;
	};
	for (int s = 0; s < 3; s++) pb.T[s].assign(8, make_float4(0, 0, 0, 0));
	pb.NS.assign(n, 4.0);
	for (int a = 0; a < n; a++)
		for (int b = 0; b < n; b++)
			for (int c = 0; c < n; c++) {
				double E[3];
				shares(a, b, c, E);
				const float4 v = make_float4((float)E[0], (float)E[1], (float)E[2], 1.0f);
				pb.T[0][(a * n + b) * n + c] = v;
				pb.T[1][(b * n + a) * n + c] = v;
				pb.T[2][(c * n + a) * n + b] = v;
			}
	ref.n = n;
	ref.Wn = pb.Wn;
	ref.S[CAT_SRP] = S;
	for (int r = 0; r < 3; r++) ref.grole[r].assign(n, 1.0);
	ref.NS = pb.NS;
	ref.e3 = [shares](int c0, int c1, int c2, double* E) {
		shares(c0, c1, c2, E);
		return 1.0;
	};
}

// Written separately from everything above: expected chips of a hero of class i, by enumerating the other
// players' classes (probability 1/4) and walking the three-player game with the strategies.
static double enum_value(const std::vector<std::vector<std::vector<double> > >& sg, int hero, int i, bool best_response) {
	// strategies by node index of three_player_tree: 0 = player 0, 1 = player 1, 2 = player 2 (after fold), 4... see below
	// node order in the tree vector: [0] terminal(p0 fold) is created first, so node indices are looked up by building the tree again
	const std::vector<Node6> t = three_player_tree();
	// the hero's best response needs the other players' class tuples with weights: tuples of the other two players
	std::vector<int> others;
	for (int p = 0; p < 3; p++)
		if (p != hero) others.push_back(p);
	// payoff of a leaf for the hero given all three classes
	auto leaf = [&](const Node6& x, const int* cls) -> double {
		double pot = 0;
		for (int p = 0; p < 3; p++) pot += x.contrib[p];
		const bool in = x.live >> hero & 1;
		if (!in) return -x.contrib[hero];
		int best = 99, cnt = 0;
		for (int p = 0; p < 3; p++)
			if (x.live >> p & 1) best = std::min(best, cls[p]);
		for (int p = 0; p < 3; p++)
			if ((x.live >> p & 1) && cls[p] == best) cnt++;
		const double share = cls[hero] == best ? 1.0 / cnt : 0.0;
		return pot * share - x.contrib[hero];
	};
	// weights over the 4 tuples of the other players' classes, then the tree walk
	std::function<double(int, std::vector<double>)> walk = [&](int g, std::vector<double> w) -> double {
		const Node6& x = t[g];
		if (x.is_term()) {
			double s = 0;
			for (int tup = 0; tup < 4; tup++) {
				int cls[3];
				cls[hero] = i;
				cls[others[0]] = tup & 1;
				cls[others[1]] = tup >> 1;
				s += w[tup] * leaf(x, cls);
			}
			return s;
		}
		const int p = x.player;
		double out = best_response && p == hero ? -1e18 : 0;
		for (size_t a = 0; a < x.child.size(); a++) {
			std::vector<double> w2 = w;
			if (p != hero)
				for (int tup = 0; tup < 4; tup++) {
					const int cp = p == others[0] ? (tup & 1) : (tup >> 1);
					w2[tup] *= sg[g][a][cp];
				}
			double v = walk(x.child[a], w2);
			if (p == hero && !best_response) v *= sg[g][a][i];
			out = (best_response && p == hero) ? std::max(out, v) : out + v;
		}
		return out;
	};
	return walk(0, std::vector<double>(4, 0.25));
}

static void test_tiny() {
	printf("1. three-player game\n");
	std::vector<Node6> t = three_player_tree();
	Problem6 pb;
	Ref ref;
	tiny_ref_and_problem(pb, ref);
	Solver6 sol(pb, t, std::vector<bool>(t.size(), false));
	// node indices: 0 = player 0's decision
	std::vector<std::vector<std::vector<double> > > sg(t.size());
	// profile by paper: players continue (bet / call) with H and fold L
	for (size_t g = 0; g < t.size(); g++) {
		if (t[g].is_term()) continue;
		sg[g].assign(t[g].child.size(), std::vector<double>(2, 0));
		for (size_t a = 0; a < t[g].child.size(); a++) {
			const bool cont = t[g].akind[a] != A_FOLD;
			sg[g][a][0] = cont ? 1 : 0;  // class 0 = H
			sg[g][a][1] = cont ? 0 : 1;
		}
	}
	VV value, best;
	engine_eval(sol, sg, value, {0, 1, 2}, best);
	// player 0 with H: both others L (1/4) wins the 2 antes (+2); one H (1/2): heads-up for 2.5 chips each... net +0.5; both H (1/4): 6/3 - 2 = 0
	check(std::fabs(value[0][0] - 0.75) < 1e-12, "player 0 holding H, everyone continues with H only: value 0.75 (paper)");
	check(std::fabs(value[0][1] + 1.0) < 1e-12, "player 0 holding L folds: value -1 (paper)");
	// player 2 (last to act) with H: P0 bets with H (1/2). Derived by the enumeration below as well; here the exact paper value:
	// P0 L (1/2): fold, then P1 and P2 show the antes: P1 H (1/2): P1 and P2 tie -> 0; P1 L: P2 H wins +1 (pot 3, share 1: 3-1... see enum)
	const double v00 = enum_value(sg, 2, 0, false);
	check(std::fabs(value[2][0] - v00) < 1e-12, "player 2 holding H against the enumeration of all deals");
	// random profile, all players, all classes, and best responses
	double worst = 0, worst_br = 0;
	for (int seed = 1; seed <= 20; seed++) {
		sg = random_profile(t, 2, seed);
		engine_eval(sol, sg, value, {0, 1, 2}, best);
		for (int p = 0; p < 3; p++)
			for (int h = 0; h < 2; h++) {
				worst = std::max(worst, std::fabs(value[p][h] - enum_value(sg, p, h, false)));
				worst_br = std::max(worst_br, std::fabs(best[p][h] - enum_value(sg, p, h, true)));
			}
	}
	printf("     largest difference engine / enumeration: value %.2e, best response %.2e\n", worst, worst_br);
	check(worst < 1e-12, "20 random profiles: every player's and class's value equals the enumeration of all deals");
	check(worst_br < 1e-12, "20 random profiles: best-response values equal a separate best-response search");
	// zero-sum: no card removal here, so the sum over players of the average value is exactly 0
	double zs = 0;
	sg = random_profile(t, 2, 99);
	engine_eval(sol, sg, value, {}, best);
	for (int p = 0; p < 3; p++)
		for (int h = 0; h < 2; h++) zs += 0.5 * value[p][h];
	printf("     sum of values %.2e\n", zs);
	check(std::fabs(zs) < 1e-12, "the game is exactly zero-sum when there is no card removal");
	// a solve of the tiny game: H always continues (never folds), regrets small
	sol.set_frozen(std::vector<bool>(t.size(), false));
	std::vector<std::vector<std::vector<double> > > uni(t.size());
	for (size_t g = 0; g < t.size(); g++) {
		if (t[g].is_term()) continue;
		uni[g].assign(t[g].child.size(), std::vector<double>(2, 1.0 / t[g].child.size()));
	}
	sol.seed(to_slots(sol, uni), 1.0);
	sol.iterate(4000);
	const Solver6::Gap gp = sol.exploit();
	printf("     after 4000 iterations: total gap %.2e\n", gp.total());
	check(gp.total() < 5e-3, "CFR+ on the three-player game: the best-response gap is small");
	check(std::fabs(gp.sum_value()) < 1e-12, "and the values still sum to 0");
}

// -- 2. tables of the real problem -----------------------------------------------------------------------
static void test_tables(const Problem6& pb, const std::vector<uint16_t>& e3) {
	printf("2. card removal and tables\n");
	const int n = NCLASS;
	double worst_row = 0, worst_sym = 0;
	for (int i = 0; i < n; i++) {
		double s = 0;
		for (int j = 0; j < n; j++) {
			s += pb.Wn[(size_t)i * n + j];
			worst_sym = std::max(worst_sym, std::fabs(pb.Wn[(size_t)i * n + j] * g_ncombo[i] - pb.Wn[(size_t)j * n + i] * g_ncombo[j]));
		}
		worst_row = std::max(worst_row, std::fabs(s - 1));
	}
	check(worst_row < 1e-12, "each hand's chances of the other hand classes sum to 1");
	check(worst_sym < 1e-12, "class i's combos x chance of j = class j's combos x chance of i");
	const int AA = class_index("AA"), AKs = class_index("AKs"), KK = class_index("KK"), AKo = class_index("AKo");
	// hero AA: another player holds AA with 1 combo of 1225 (the two other aces), KK with 6/1225, AKs with 2 combos (kings of the 2 other suits... K with ace of the other two suits)
	check(std::fabs(pb.Wn[(size_t)AA * n + AA] - 1.0 / 1225) < 1e-15, "hero AA: the other player holds AA with probability 1/1225");
	check(std::fabs(pb.Wn[(size_t)AA * n + KK] - 6.0 / 1225) < 1e-15, "hero AA: KK with probability 6/1225");
	check(std::fabs(pb.Wn[(size_t)AA * n + AKs] - 2.0 / 1225) < 1e-15 && std::fabs(pb.Wn[(size_t)AA * n + AKo] - 6.0 / 1225) < 1e-15,
	      "hero AA: AKs 2/1225 and AKo 6/1225");
	double ws = 0;
	for (int i = 0; i < n; i++) ws += pb.prior[i];
	check(std::fabs(ws - 1) < 1e-12, "class probabilities sum to 1");
	// three-way table in the engine's layout
	double worst_ns = 0, worst_sum = 0;
	for (int a = 0; a < n; a++) worst_ns = std::max(worst_ns, std::fabs(pb.NS[a] - (double)g_ncombo[a] * 1225 * 1128) / (g_ncombo[a] * 1225.0 * 1128));
	long long bad = 0, cnt = 0;
	for (int a = 0; a < n; a += 7)
		for (int b = 0; b < n; b++)
			for (int c = 0; c < n; c++) {
				const float4 v = pb.T[0][((size_t)a * n + b) * n + c];
				if (v.w == 0) continue;
				cnt++;
				worst_sum = std::max(worst_sum, (double)std::fabs(v.x + v.y + v.z - 1));
				const float4 w1 = pb.T[1][((size_t)b * n + a) * n + c], w2 = pb.T[2][((size_t)c * n + a) * n + b];
				if (w1.x != v.x || w1.y != v.y || w1.z != v.z || w1.w != v.w || w2.x != v.x || w2.w != v.w) bad++;
			}
	printf("     N sums: worst relative error %.1e; shares sum to 1 within %.1e (%lld cells)\n", worst_ns, worst_sum, cnt);
	check(worst_ns < 1e-12, "three-way counts: for every class sum of N = combos x 1225 x 1128");
	check(worst_sum < 1e-4 && bad == 0, "the three layouts hold the same cell, shares sum to 1");
	// pot-share matrices
	double lo = 1, hi = 0;
	for (int c = 0; c < NCAT; c++) {
		if (pb.S[c].empty()) continue;
		for (double x : pb.S[c]) { lo = std::min(lo, x); hi = std::max(hi, x); }
	}
	check(lo >= 0 && hi <= 1, "every pot share lies in [0, 1]");
	const int s72 = class_index("72o");
	check(pb.S[CAT_SRP][(size_t)AA * n + s72] > 0.8 && pb.S[CAT_SRP][(size_t)s72 * n + AA] < 0.35, "single-raised pot: AA in position takes more than 0.8 of the pot against 72o, 72o less than 0.35 against AA");
	// the tie between equal classes: in position the player takes at least as much as out of position (position is worth something)
	double ipmore = 0;
	for (int i = 0; i < n; i++) ipmore += pb.S[CAT_SRP][(size_t)i * n + i] - 0.5;
	check(ipmore > 0, "in a single-raised pot, equal hands: position takes more than half on average");
	(void)e3;
}

// -- 3. the engine against the host model on the real tree ----------------------------------------------
static void test_reference(Solver6& sol, const std::vector<uint16_t>& e3) {
	printf("3. engine against the independent host evaluator (real 100bb tree)\n");
	Ref ref;
	const Problem6& pb = sol.pb;
	ref.n = NCLASS;
	ref.Wn = pb.Wn;
	for (int c = 0; c < NCAT; c++) ref.S[c] = pb.S[c];
	for (int r = 0; r < 3; r++) ref.grole[r] = pb.grole[r];
	ref.NS = pb.NS;
	ref.e3 = [&e3](int a, int b, int c, double* E) {
		float f[3];
		const uint32_t N = e3_lookup(e3.data(), a, b, c, f);
		for (int q = 0; q < 3; q++) E[q] = f[q];
		return (double)N;
	};
	const char* hs[8] = {"AA", "72o", "AKs", "55", "T9s", "KQo", "A5s", "J4o"};
	std::vector<int> H;
	for (const char* h : hs) H.push_back(class_index(h));
	const auto sg = random_profile(sol.nodes, NCLASS, 12345);
	VV value, best;
	engine_eval(sol, sg, value, {0, 4, 5}, best);
	RefRun rv{ref, sol.nodes, sg, H, -1};
	VV root(NPLAYER, std::vector<double>(NCLASS, 1.0));
	const VV rval = rv.go(0, root);
	double worst = 0, worst_br = 0, scale = 0;
	for (int p = 0; p < NPLAYER; p++)
		for (int h : H) {
			worst = std::max(worst, std::fabs(rval[p][h] - value[p][h]));
			scale = std::max(scale, std::fabs(value[p][h]));
		}
	for (int p : {0, 4, 5}) {
		RefRun rb{ref, sol.nodes, sg, H, p};
		const VV bv = rb.go(0, root);
		for (int h : H) worst_br = std::max(worst_br, std::fabs(bv[p][h] - best[p][h]));
	}
	printf("     largest difference: values %.2e (largest value %.3f), best responses %.2e\n", worst, scale, worst_br);
	check(worst < 1e-9, "values of a random profile: all six players, 8 classes (includes three-way showdowns, all pot types)");
	check(worst_br < 1e-9, "best-response values of UTG, SB and BB (so the best-response gap) for the same profile");
	// the best response is worth at least the profile
	bool ge = true;
	for (int p : {0, 4, 5})
		for (int h : H)
			if (best[p][h] < value[p][h] - 1e-12) ge = false;
	check(ge, "a best response is worth at least the profile's value");
}

// -- 4. a short real run ---------------------------------------------------------------------------------
static std::vector<double> run_short(Solver6& sol, const Source& src, long long iters, std::vector<bool> frozen) {
	const std::vector<Node6>& nodes = sol.nodes;
	sol.set_frozen(frozen);
	sol.seed(init_sigma(nodes, sol.hn, sol.S, src, frozen, 0.02), 1.0);
	sol.iterate(iters);
	return sol.averaged();
}

static void test_run(Solver6& sol, const Source& src) {
	printf("4. a short real run\n");
	const std::vector<Node6>& nodes = sol.nodes;
	std::vector<bool> frozen(nodes.size(), false);
	for (size_t i = 0; i < nodes.size(); i++) frozen[i] = !nodes[i].is_term() && nodes[i].type != NT_RFI;
	const std::vector<double> a1 = run_short(sol, src, 150, frozen);
	const std::vector<double> before = init_sigma(nodes, sol.hn, sol.S, src, frozen, 0.02);
	// frozen decisions stay exactly at the source
	double moved = 0, rfi_moved = 0;
	for (size_t i = 0; i < nodes.size(); i++) {
		if (nodes[i].is_term()) continue;
		for (size_t a = 0; a < nodes[i].child.size(); a++)
			for (int h = 0; h < NCLASS; h++) {
				const size_t s = (size_t)(sol.hn[i].aoff + a) * NPAD + h;
				if (frozen[i]) moved = std::max(moved, std::fabs(a1[s] - before[s]));
				else rfi_moved = std::max(rfi_moved, std::fabs(a1[s] - before[s]));
			}
	}
	check(moved == 0, "stage 1: decisions fixed to the source charts did not move");
	check(rfi_moved > 0.01, "stage 1: the first-in decisions did move");
	// the best-response gap of a frozen reply is not counted: gaps after stage 1 are small for everybody
	const Solver6::Gap g1 = sol.exploit();
	printf("     stage 1 gap bb/100 (first-in decisions only): ");
	for (int p = 0; p < NPLAYER; p++) printf("%s %.3f ", SEAT[p], 100 * g1.gap(p));
	printf("\n");
	// stage 2, twice, bit for bit
	std::vector<bool> none(nodes.size(), false);
	const std::vector<double> b1 = run_short(sol, src, 150, none);
	const std::vector<double> b2 = run_short(sol, src, 150, none);
	check(memcmp(b1.data(), b2.data(), b1.size() * sizeof(double)) == 0, "determinism: 150 iterations from the same start give bit-identical strategies");
	double worst = 0;
	for (size_t i = 0; i < nodes.size(); i++) {
		if (nodes[i].is_term()) continue;
		for (int h = 0; h < NCLASS; h++) {
			double s = 0;
			for (size_t a = 0; a < nodes[i].child.size(); a++) s += b1[(size_t)(sol.hn[i].aoff + a) * NPAD + h];
			worst = std::max(worst, std::fabs(s - 1));
		}
	}
	printf("     largest |sum of a hand's frequencies - 1|: %.2e\n", worst);
	check(worst < 1e-12, "the frequencies of each hand sum to 1 at every decision");
	bool nonneg = true;
	for (size_t i = 0; i < nodes.size(); i++)
		if (!nodes[i].is_term())
			for (size_t a = 0; a < nodes[i].child.size(); a++)
				for (int h = 0; h < NCLASS; h++)
					if (b1[(size_t)(sol.hn[i].aoff + a) * NPAD + h] < 0) nonneg = false;
	check(nonneg, "no negative frequency");
	const Solver6::Gap g2 = sol.exploit();
	printf("     stage 2 after 150 iterations: gap bb/100 total %.3f, sum of values %.4f bb\n", 100 * g2.total(), g2.sum_value());
	check(std::fabs(g2.sum_value()) < 0.1, "zero-sum defect from the pairwise card-removal model below 0.1 bb per hand");
	bool br = true;
	for (int p = 0; p < NPLAYER; p++)
		if (g2.gap(p) < -1e-9) br = false;
	check(br, "every player's best-response gap is non-negative");
	// JSON
	std::vector<ConvPoint> conv;
	conv.push_back({150, g2});
	RunOpts o;
	Rules R;
	const std::string js = write_result(o, R, sol, src, conv, 150);
	JsonReader jr(js);
	bool parsed = true;
	Json root;
	try { root = jr.value(); } catch (...) { parsed = false; }
	check(parsed, "the JSON parses");
	if (parsed) {
		check(root.at("schema").str == "sixmax/v1" && root.has("rfi") && root.has("vs_open") && root.has("vs_3bet") && root.has("convergence"),
		      "schema fields present");
		check(root.at("vs_open").obj.size() == 15 && root.at("vs_3bet").obj.size() == 15 && root.at("rfi").obj.size() == 5,
		      "5 first-in seats, 15 facing-open and 15 facing-3-bet spots, as in the app");
		// the percentage written for UTG equals the one recomputed from its map
		const Json& mix = root.at("rfi").at("UTG").at("mix");
		double s = 0;
		for (const auto& kv : mix.obj) s += kv.second.num * g_ncombo[class_index(kv.first)];
		check(std::fabs(100 * s / 1326 - root.at("rfi").at("UTG").at("pct").num) < 0.2, "UTG open % in the JSON equals the combos of its mix");
		// a spot: call + bluff + value <= 1 per hand
		bool le1 = true;
		for (const auto& kv : root.at("vs_open").obj)
			for (const auto& h : kv.second.at("call").obj) {
				const std::string& nm = h.first;
				double t = h.second.num;
				if (kv.second.at("bluff").has(nm)) t += kv.second.at("bluff").at(nm).num;
				if (kv.second.at("value").has(nm)) t += kv.second.at("value").at(nm).num;
				if (t > 1.02) le1 = false;
			}
		check(le1, "call + 3-bet <= 1 per hand in every facing-open spot");
	}
}

// -- 5. the relay -------------------------------------------------------------------------------------
static bool same_bits6(const std::vector<double>& a, const std::vector<double>& b) {
	return a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

static void test_relay(Solver6& sol, const Source& src) {
	printf("5. the relay: 120 iterations in batches of 20, stopped and resumed\n");
	const std::vector<Node6>& nodes = sol.nodes;
	std::vector<bool> none(nodes.size(), false);
	const size_t cnt = (size_t)sol.S * NPAD;
	auto fresh = [&]() {  // what a new process starts from
		sol.set_frozen(none);
		sol.seed(init_sigma(nodes, sol.hn, sol.S, src, none, 0.02), 1.0);
	};
	fresh();
	sol.iterate(120);
	const std::vector<double> ref_r = sol.download(sol.d_regret, cnt), ref_s = sol.download(sol.d_sum, cnt);
	const std::vector<double> ref_avg = sol.averaged();
	// the caller's extra state: the list of batch ends seen (stands for the convergence points)
	std::vector<long long> seen;
	auto after = [&](long long d) { seen.push_back(d); };
	auto save_x = [&](FILE* f) {
		const uint64_t n = seen.size();
		fwrite(&n, sizeof n, 1, f);
		fwrite(seen.data(), sizeof(long long), n, f);
	};
	auto load_x = [&](FILE* f) {
		uint64_t n = 0;
		if (fread(&n, sizeof n, 1, f) != 1 || n > 1000) return false;
		std::vector<long long> v(n);
		if (fread(v.data(), sizeof(long long), n, f) != n) return false;
		seen = v;
		return true;
	};
	const char* path = "/tmp/test_sixmax_relay.ckpt";
	remove(path);
	fresh();
	sol.run(120, "", 20);
	check(same_bits6(sol.download(sol.d_regret, cnt), ref_r) && same_bits6(sol.download(sol.d_sum, cnt), ref_s),
	      "run() in batches without a file equals iterate()");
	fresh();
	seen.clear();
	sol.run(120, path, 20, after, save_x, load_x, 2);
	check(sol.done == 40 && relay_peek_done(path) == 40, "stopped after 2 batches: the file holds 40 iterations");
	fresh();  // a new process
	seen.clear();
	sol.run(120, path, 20, after, save_x, load_x, 2);
	check(sol.done == 80 && relay_peek_done(path) == 80 && seen.size() == 4, "second run restored 40, did 2 more batches, extra state restored");
	{
		FILE* t = fopen((std::string(path) + ".tmp").c_str(), "wb");  // a hard kill during a save leaves this behind
		fwrite("garbage", 1, 7, t);
		fclose(t);
	}
	fresh();
	seen.clear();
	sol.run(120, path, 20, after, save_x, load_x);
	check(sol.done == 120 && relay_peek_done(path) == 120, "third run finished at 120");
	check(same_bits6(sol.download(sol.d_regret, cnt), ref_r) && same_bits6(sol.download(sol.d_sum, cnt), ref_s),
	      "regrets and strategy sums equal the uninterrupted run, bit for bit");
	check(same_bits6(sol.averaged(), ref_avg), "the averaged strategy equals the uninterrupted run, bit for bit");
	check(seen.size() == 6 && seen[0] == 20 && seen[5] == 120, "the extra state (list of batch ends) is the same as in an uninterrupted run");
	// an old-format checkpoint (the format before the relay) at iteration 100 is converted and finished
	remove(path);
	fresh();
	sol.iterate(100);
	{
		const std::vector<double> r = sol.download(sol.d_regret, cnt), s = sol.download(sol.d_sum, cnt);
		FILE* f = fopen(path, "wb");
		long long hdr[4] = {0x36584d41583031ll, sol.G, sol.S, 100};
		fwrite(hdr, sizeof hdr, 1, f);
		fwrite(r.data(), sizeof(double), cnt, f);
		fwrite(s.data(), sizeof(double), cnt, f);
		fclose(f);
	}
	fresh();
	sol.run(120, path, 20);
	check(sol.done == 120 && same_bits6(sol.download(sol.d_regret, cnt), ref_r) && same_bits6(sol.download(sol.d_sum, cnt), ref_s),
	      "an old-format checkpoint at 100 is converted and finished: same result, bit for bit");
	remove(path);
	remove((std::string(path) + ".tmp").c_str());
}

int main() {
	setup_classes();
	test_tiny();
	const std::vector<double> eq = sx_read_matrix("data/equity-exact.bin");
	const std::vector<uint16_t> e3 = e3_load_table("data/equity3.bin");
	const Json real = json_load("data/realization.json");
	Problem6 pb = make_problem6(eq, &real, &e3, 100, true);
	test_tables(pb, e3);
	Rules R;
	std::vector<Node6> nodes = build_tree(R);
	Source src = load_source(data_dir());
	Solver6 sol(pb, nodes, std::vector<bool>(nodes.size(), false));
	test_reference(sol, e3);
	test_run(sol, src);
	test_relay(sol, src);
	printf("\n%s\n", g_fail ? "FAIL" : "PASS");
	return g_fail ? 1 : 0;
}

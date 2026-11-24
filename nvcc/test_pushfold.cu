// Checks for the push/fold tables (pushfold.cuh). Prints PASS and exits 0, or FAIL and exits 1.
//
//   nvcc -O3 -fmad=false -arch=sm_89 -o test_pushfold test_pushfold.cu && ./test_pushfold [matrix.bin] [iterations]
//
// 1. The evaluator (PfEval, the thing that measures every other check) against numbers worked out by hand on
//    a two-hand game.
// 2. A solved run on the exact matrix, 2 to 20bb: the exploitability the solver reports for its averaged
//    strategy equals the evaluator's on the same strategy; the game is zero-sum; the averaged strategy is
//    tiny in exploitability, and so are the pure ranges made from it.
// 3. Poker sense: ranges only shrink as the stacks deepen, the best hands are in every range, the numbers sit
//    where published push/fold charts have them.
// 4. The JSON has the schema of src/lib/poker/data/tables.json and pushfold-mix.json, and its percentages,
//    lists and frequencies agree with each other.
// 5. equity_vs_random: known values, and the average over all hands is exactly one half.

#include "pushfold.cuh"

static int g_fail = 0;
static void check(bool ok, const char* what) {
	printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
	if (!ok) g_fail++;
}

// -- 1. the evaluator on a two-hand game ------------------------------------------------------------
// Hands H and L, equally likely, H beats L every time, a hand ties itself. Stack 2, blinds 0.5 / 1.
static void test_evaluator() {
	printf("evaluator against a two-hand game worked out by hand\n");
	Problem pb;
	pb.n = 2;
	pb.W = {1, 1, 1, 1};
	pb.prior = {0.5, 0.5};
	pb.wsum = {2, 2};
	const std::vector<double> eq = {0.5, 1, 0, 0.5};  // eq[i*2+j]
	PfEval ev = {pb, eq};
	// Shove everything, call everything: shoving H is worth 1 (ties H for 0, beats L for +2), shoving L is worth
	// -1 and should fold for -0.5, so the small blind's best response gains 0.5 with L, which is half the deals.
	// The big blind calls with H, and with L calling and folding are worth the same: no gain there.
	check(std::fabs(ev.exploitability(2, {1, 1}, {1, 1}) - 0.25) < 1e-12, "shove everything, call everything: 0.25");
	// Nobody shoves: shoving H into callers is worth 1 instead of -0.5, half the deals, 1.5 each; L is right to
	// fold; the big blind has nothing to answer.
	check(std::fabs(ev.exploitability(2, {0, 0}, {1, 1}) - 0.75) < 1e-12, "fold everything: 0.75");
	// The equilibrium: H shoves and calls, L folds. Shoving L is worth exactly -0.5 (as folding), nobody gains.
	check(std::fabs(ev.exploitability(2, {1, 0}, {1, 0})) < 1e-12, "the equilibrium: 0");
	// Its value for the small blind: H shoves for 0.5 (0 against H, +1 when L folds), L folds for -0.5.
	check(std::fabs(ev.sb_value(2, {1, 0}, {1, 0})) < 1e-12, "and its value for the small blind: 0");
}

// -- 2. and 3. a solved run -------------------------------------------------------------------------
static std::vector<PfStack> g_solved;
static std::vector<double> g_stacks;
static std::vector<double> g_eq;
static Problem g_pb;
static long long g_iters = 0;

static void test_solved(long long iterations) {
	printf("solved run, exact matrix, %lld iterations\n", iterations);
	for (int s = 2; s <= 20; s++) g_stacks.push_back(s);
	std::vector<GameTree> trees;
	for (double s : g_stacks) trees.push_back(build_pushfold_tree(s));
	Solver sv(g_pb, trees);
	sv.iterate(iterations);
	g_iters = sv.done;
	g_solved = collect(sv, g_stacks, g_eq);
	PfEval host = {g_pb, g_eq};
	auto ev = sv.value();
	auto ex = sv.exploitability();
	double worst_mixed = 0, worst_pure = 0, worst_diff = 0, worst_val = 0;
	bool at_least = true;
	for (size_t i = 0; i < g_stacks.size(); i++) {
		const PfStack& r = g_solved[i];
		worst_mixed = std::max(worst_mixed, r.expl_mixed);
		worst_pure = std::max(worst_pure, r.expl_pure);
		// The evaluator on the averaged strategy itself (not the thresholded one).
		worst_diff = std::max(worst_diff, std::fabs(r.expl_mixed - host.exploitability(r.stack, r.push_mix, r.call_mix)));
		worst_val = std::max(worst_val, std::fabs(r.sb_value - host.sb_value(r.stack, r.push_mix, r.call_mix)));
		at_least = at_least && ex[i].first >= ev[i].first - 1e-9 && ex[i].second >= ev[i].second - 1e-9;
	}
	printf("       worst over 2-20bb: mixed %.3e, pure ranges %.3e bb; solver vs evaluator %.2e\n", worst_mixed, worst_pure,
	       worst_diff);
	check(worst_diff < 1e-9, "the solver's exploitability equals the independent evaluator's (1e-9)");
	check(worst_val < 1e-9, "the solver's value of the profile equals the evaluator's (1e-9)");
	check(at_least, "a best response is worth at least the profile's value, for both seats");
	check(worst_mixed < 1e-5, "averaged strategy: exploitability below 1e-5 bb per hand at every stack");
	check(worst_pure < 2e-3, "pure ranges: exploitability below 0.002 bb per hand at every stack");
	// Zero-sum: the values of the two seats (relative to the blinds) add up to 0.
	bool zero = true;
	for (size_t i = 0; i < g_stacks.size(); i++) zero = zero && std::fabs(ev[i].first + ev[i].second) < 1e-9;
	check(zero, "zero-sum at every stack");
}

static double pct_at(int s, bool push) { return push ? g_solved[s - 2].push_pct : g_solved[s - 2].call_pct; }
static bool has(const std::vector<int>& v, const char* name) {
	for (int h : v)
		if (g_names[h] == name) return true;
	return false;
}

static void test_poker_sense() {
	printf("poker sense\n");
	bool shrink_push = true, shrink_call = true, top = true, junk = true;
	for (int s = 3; s <= 20; s++) {
		shrink_push = shrink_push && pct_at(s, true) <= pct_at(s - 1, true) + 0.5;
		shrink_call = shrink_call && pct_at(s, false) <= pct_at(s - 1, false) + 0.5;
	}
	for (const PfStack& r : g_solved) {
		for (const char* h : {"AA", "KK", "QQ", "AKs", "AKo"}) top = top && has(r.push, h);
		top = top && has(r.call, "AA") && has(r.call, "KK");
		if (r.stack >= 10) junk = junk && !has(r.push, "32o") && !has(r.call, "72o");
	}
	check(shrink_push, "the shoving range does not grow as the stacks deepen (within half a point)");
	check(shrink_call, "the calling range does not grow as the stacks deepen (within half a point)");
	check(top, "AA, KK, QQ, AKs, AKo shove at every depth; AA and KK always call");
	check(junk, "from 10bb on 32o never shoves and 72o never calls");
	// at 2bb the big blind calls (nearly) everything, so the small blind shoves all but the worst hands
	check(pct_at(2, true) > 85 && pct_at(2, false) > 95, "2bb: shove over 85%, call over 95%");
	check(pct_at(5, true) > 65 && pct_at(5, true) < 80, "5bb: SB shoves about 70% (65 to 80)");
	check(pct_at(10, true) > 52 && pct_at(10, true) < 66, "10bb: SB shoves in the high 50s (52 to 66)");
	check(pct_at(10, false) > 30 && pct_at(10, false) < 45, "10bb: BB calls about 38% (30 to 45)");
	check(pct_at(20, true) > 30 && pct_at(20, true) < 50, "20bb: SB shoves about 40% (30 to 50)");
	check(pct_at(20, false) > 15 && pct_at(20, false) < 28, "20bb: BB calls about 22% (15 to 28)");
}

// -- 4. the JSON ----------------------------------------------------------------------------------------
static size_t count(const std::string& s, const std::string& piece) {
	size_t n = 0;
	for (size_t f = s.find(piece); f != std::string::npos; f = s.find(piece, f + 1)) n++;
	return n;
}

static void test_json() {
	printf("JSON\n");
	const std::vector<double> evr = equity_vs_random(g_pb, g_eq);
	const std::string js = pushfold_json(g_solved, g_iters, evr);
	check(js.find("{\"samples_per_matchup\":0,\"equity_vs_random\":{\"AA\":") == 0, "starts with samples_per_matchup and equity_vs_random");
	check(js.find(",\"pushfold\":{\"2\":{\"push\":[") != std::string::npos && js.find(",\"mix\":{\"2\":{\"push\":{") != std::string::npos,
	      "pushfold and mix sections start at 2bb");
	check(count(js, "\"push_pct\":") == 19 && count(js, "\"call_pct\":") == 19 && count(js, "\"exploitability_bb\":") == 19,
	      "19 rows, each with push_pct, call_pct and exploitability_bb");
	check(count(js, "\"exploitability_mixed_bb\":") == 19, "19 solver reports");
	// the lists, the percentages and the frequencies must tell the same story
	bool pct_ok = true, mix_ok = true, rows_ok = true;
	for (const PfStack& r : g_solved) {
		double a = 0, b = 0, ma = 0, mb = 0;
		for (int h : r.push) a += g_ncombo[h];
		for (int h : r.call) b += g_ncombo[h];
		pct_ok = pct_ok && std::fabs(100 * a / 1326 - r.push_pct) < 1e-9 && std::fabs(100 * b / 1326 - r.call_pct) < 1e-9;
		for (int h = 0; h < NCLASS; h++) {
			const double p = r.push_mix[h], c = r.call_mix[h];
			mix_ok = mix_ok && p >= -1e-12 && p <= 1 + 1e-12 && c >= -1e-12 && c <= 1 + 1e-12;
			// a class is in the pure range exactly when its mix is above one half
			rows_ok = rows_ok && ((p > 0.5) == has(r.push, g_names[h].c_str())) && ((c > 0.5) == has(r.call, g_names[h].c_str()));
			ma += p * g_ncombo[h];
			mb += c * g_ncombo[h];
		}
		// the mixed ranges are about as wide as the pure ones (the edge hands split about evenly)
		mix_ok = mix_ok && std::fabs(100 * ma / 1326 - r.push_pct) < 3 && std::fabs(100 * mb / 1326 - r.call_pct) < 3;
	}
	check(pct_ok, "push_pct and call_pct are the combo shares of the lists");
	check(rows_ok, "a class is in the list exactly when its frequency is above one half");
	check(mix_ok, "frequencies are in [0, 1], and the mixed ranges are as wide as the pure ones (within 3 points)");
}

// -- 5. equity against a random hand --------------------------------------------------------------------
static void test_evr() {
	printf("equity against a random hand\n");
	const std::vector<double> evr = equity_vs_random(g_pb, g_eq);
	auto at = [&](const char* name) {
		for (int i = 0; i < NCLASS; i++)
			if (g_names[i] == name) return evr[i];
		return -1.0;
	};
	// published: AA 85.2%, KK 82.4%, AKs 67.0%, 72o 34.6%
	check(std::fabs(at("AA") - 0.852) < 0.001 && std::fabs(at("KK") - 0.824) < 0.001, "AA 85.2%, KK 82.4%");
	check(std::fabs(at("AKs") - 0.670) < 0.001 && std::fabs(at("72o") - 0.346) < 0.001, "AKs 67.0%, 72o 34.6%");
	double mean = 0;
	for (int i = 0; i < NCLASS; i++) mean += g_pb.prior[i] * evr[i];
	check(std::fabs(mean - 0.5) < 1e-9, "averaged over all hands it is exactly one half");
}

int main(int argc, char** argv) {
	setup_classes();
	const char* path = argc > 1 ? argv[1] : "data/equity-exact.bin";
	const long long iterations = argc > 2 ? atoll(argv[2]) : 20000;
	g_eq = read_matrix(path);
	g_pb = make_problem(g_eq, false);
	test_evaluator();
	test_solved(iterations);
	test_poker_sense();
	test_json();
	test_evr();
	printf(g_fail ? "FAIL (%d)\n" : "PASS\n", g_fail);
	return g_fail ? 1 : 0;
}

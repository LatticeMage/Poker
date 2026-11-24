// Cross-check of the solver against the external solver poker-cfr (BSD-2, by Wataru Inariba; a
// read-only clone is under .tmp/poker-cfr; its code is never run, only its output files are read). It solves heads-up preflop with checks to showdown, i.e. raw equity and no
// realisation, with raises of 2.5x / 3x / 3.5x / 4x the bet faced plus all-in. We run our solver on that
// same tree (build_mult_tree) with the realisation switched off, and compare node by node.
//
//   nvcc -O3 -fmad=false -arch=sm_89 -o test_solve_external test_solve_external.cu
//   ./test_solve_external <stack bb> <iterations> <preflop-<bb>-<iters>.bin> [matrix.bin]
//
// File format (bincode 1.x, fixed-width little endian; main_preflop.rs):
//   (HashMap<Vec<u8>, Vec<Vec<Vec<f64>>>>, f64 ev_sb, f64 exploitability)
// key = the actions taken so far (0 fold, 1 call/check, 2.. raises); value[action][a][b] = average share of
// the combos of a 13x13 cell that take the action. Cell: pair [r][r], suited [low][high], offsuit [high][low]
// (rank 0 = deuce). It also prints the differences that are expected: its equity table is its own Monte
// Carlo (heads_up_pre_flop_equity.bin), ours is the exact enumeration.
//
// This is a sanity check, not a pass/fail test: it exits 1 only if the file cannot be read or the trees do
// not line up.

#include <map>

#include "solve.cuh"

struct Reader {
	std::vector<uint8_t> b;
	size_t at = 0;
	uint64_t u64() {
		uint64_t v;
		if (at + 8 > b.size()) { fprintf(stderr, "file ends early\n"); exit(1); }
		memcpy(&v, &b[at], 8);
		at += 8;
		return v;
	}
	double f64() {
		double v;
		if (at + 8 > b.size()) { fprintf(stderr, "file ends early\n"); exit(1); }
		memcpy(&v, &b[at], 8);
		at += 8;
		return v;
	}
};
typedef std::vector<std::vector<std::vector<double> > > Strat;  // [action][13][13]

int main(int argc, char** argv) {
	if (argc < 4) { fprintf(stderr, "usage: test_solve_external <bb> <iterations> <preflop-bb-iters.bin> [matrix.bin]\n"); return 1; }
	const double stack = atof(argv[1]);
	const long long iters = atoll(argv[2]);
	Reader rd;
	{
		FILE* f = fopen(argv[3], "rb");
		if (!f) { fprintf(stderr, "cannot read %s\n", argv[3]); return 1; }
		uint8_t buf[1 << 16];
		size_t k;
		while ((k = fread(buf, 1, sizeof buf, f)) > 0) rd.b.insert(rd.b.end(), buf, buf + k);
		fclose(f);
	}
	std::map<std::vector<uint8_t>, Strat> ext;
	const uint64_t entries = rd.u64();
	for (uint64_t e = 0; e < entries; e++) {
		std::vector<uint8_t> key(rd.u64());
		for (auto& c : key) c = rd.b[rd.at++];
		Strat s(rd.u64());
		for (auto& act : s) {
			act.resize(rd.u64());
			for (auto& row : act) {
				row.resize(rd.u64());
				for (auto& x : row) x = rd.f64();
			}
		}
		ext[key] = s;
	}
	const double ext_ev = rd.f64(), ext_exploit = rd.f64();
	printf("external: %llu decision nodes, SB value %+.4f bb, exploitability %.3e bb\n", (unsigned long long)entries, ext_ev, ext_exploit);

	setup_classes();
	const Problem pb = make_problem(read_matrix(argc > 4 ? argv[4] : "data/equity-exact.bin"), false);
	std::vector<GameTree> trees = {build_mult_tree(stack)};
	const GameTree& t = trees[0];
	size_t ours = 0;
	for (const GNode& nd : t.nodes) ours += nd.kind == DECISION;
	printf("ours:     %zu decision nodes (same tree rules, raw equity)\n", ours);
	if (ours != entries) { fprintf(stderr, "the trees differ in size\n"); return 1; }

	Solver sv(pb, trees);
	auto t0 = std::chrono::steady_clock::now();
	sv.iterate(iters);
	const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	const auto ex = sv.exploitability();
	const auto ev = sv.value();
	const std::vector<double> avg = sv.averaged();
	printf("ours after %lld iterations (%.1fs): SB value %+.4f bb, exploitability %.3e bb\n", iters, secs, ev[0].first,
	       ex[0].first + ex[0].second);

	// map a class to its cell in the external 13x13 tables
	std::vector<int> ra(NCLASS), rb(NCLASS);
	for (int h = 0; h < NCLASS; h++) {
		const std::string& s = g_names[h];
		const int x = (int)(strchr(RANKS, s[0]) - RANKS), y = (int)(strchr(RANKS, s[1]) - RANKS);
		if (s.size() == 2) ra[h] = rb[h] = x;
		else if (s[2] == 's') { ra[h] = y; rb[h] = x; }
		else { ra[h] = x; rb[h] = y; }
	}

	// walk our tree; reach of both players under our averaged strategy
	double tv_all = 0, w_all = 0;
	struct Row { std::string id; double reach, tv; std::vector<double> mine, theirs; std::vector<std::string> labels; };
	std::vector<Row> rows;
	std::function<void(int, std::vector<std::vector<double> >)> walk = [&](int node, std::vector<std::vector<double> > reach) {
		const GNode& nd = t.nodes[node];
		if (nd.kind != DECISION) return;
		auto it = ext.find(nd.path);
		if (it == ext.end() || it->second.size() != nd.child.size()) { fprintf(stderr, "node %s not in the external file\n", nd.id.c_str()); exit(1); }
		const int slot = sv.slot_of(0, node);
		const std::vector<double>& mine = reach[nd.player];
		// An off-path node (one nobody plays into) has no determined equilibrium strategy, so nodes are weighted by
		// how likely they are to be reached: the actor's hand weight times the other player's total reach.
		double other_played = 0;
		for (int h = 0; h < NCLASS; h++) other_played += pb.prior[h] * reach[1 - nd.player][h];
		double wsum = 0, tv = 0;
		std::vector<double> m(nd.child.size(), 0.0), th(nd.child.size(), 0.0);
		for (int h = 0; h < NCLASS; h++) {
			const double w = pb.prior[h] * mine[h] * other_played;  // chance this hand is at this node
			double d = 0;
			for (size_t a = 0; a < nd.child.size(); a++) {
				const double p = avg[(size_t)(slot + a) * NP + h], q = it->second[a][ra[h]][rb[h]];
				d += std::fabs(p - q);
				m[a] += w * p;
				th[a] += w * q;
			}
			tv += w * 0.5 * d;
			wsum += w;
		}
		if (wsum > 1e-9) {
			for (size_t a = 0; a < m.size(); a++) { m[a] /= wsum; th[a] /= wsum; }
			// the external file stores 0 for hands that never get here; weights keep those out
			tv_all += tv;
			w_all += wsum;
			rows.push_back({nd.id, wsum, tv / wsum, m, th, nd.labels});
		}
		for (size_t a = 0; a < nd.child.size(); a++) {
			std::vector<std::vector<double> > r = reach;
			for (int h = 0; h < NCLASS; h++) r[nd.player][h] = mine[h] * avg[(size_t)(slot + a) * NP + h];
			walk(nd.child[a], r);
		}
	};
	walk(0, std::vector<std::vector<double> >(2, std::vector<double>(NCLASS, 1.0)));

	printf("\nper node: share of hands that get there (prior x reach of the actor), mean total-variation distance of the\n"
	       "per-hand strategies (0 = identical, 1 = disjoint), and the action mix over those hands (ours / external)\n");
	int shown = 0;
	for (const Row& r : rows) {
		if (r.reach < 0.02 || shown >= 14) continue;
		shown++;
		printf("%-14s reach %.3f  tv %.3f  ", r.id.c_str(), r.reach, r.tv);
		for (size_t a = 0; a < r.mine.size(); a++) printf(" %s %.1f/%.1f%%", r.labels[a].c_str(), 100 * r.mine[a], 100 * r.theirs[a]);
		printf("\n");
	}
	printf("\nweighted over all %zu nodes (weight = share of hands at the node): mean tv %.4f\n", rows.size(), tv_all / w_all);
	printf("SB value: ours %+.4f bb, external %+.4f bb (difference %.4f)\n", ev[0].first, ext_ev, ev[0].first - ext_ev);
	return 0;
}

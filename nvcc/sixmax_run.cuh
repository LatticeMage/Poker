// The driver of the 6-max solver: the two stages, the reading of the result (the app's shapes) and the JSON.
// Included by sixmax.cu. The checks use the pieces in sixmax.cuh directly.
#pragma once

#include "sixmax.cuh"

struct RunOpts {
	double stack = 100;
	long long iters1 = 3000, iters2 = 20000, report = 1000;
	std::string out = "data/out", ckpt;
	double seed_k = 1.0;  // regret mass given to the starting strategy
};

static std::string data_dir() {
	const char* e = getenv("POKER_DATA");
	return e ? e : "../src/lib/poker/data";
}

// What a solved strategy says, read from the averaged sigma slots.
struct Reading {
	const std::vector<Node6>* nodes;
	const std::vector<DNode>* hn;
	const Problem6* pb;
	std::vector<double> avg, reach;  // sigma slots; reach[(node * 6 + p) * NPAD + h]

	double f(int node, int a, int h) const { return avg[(size_t)((*hn)[node].aoff + a) * NPAD + h]; }
	int act(int node, int kind) const {  // index of the first action of this kind, or -1
		const Node6& n = (*nodes)[node];
		for (size_t a = 0; a < n.akind.size(); a++)
			if (n.akind[a] == kind) return (int)a;
		return -1;
	}
	// the aggressive action (raise or all-in) of a node
	int aggr(int node) const {
		int a = act(node, A_RAISE);
		return a >= 0 ? a : act(node, A_ALLIN);
	}
	double fold(int node, int h) const { return f(node, 0, h); }
	double call(int node, int h) const {
		const int a = act(node, A_CALL);
		return a >= 0 ? f(node, a, h) : 0.0;
	}
	double agg(int node, int h) const {
		const int a = aggr(node);
		return a >= 0 ? f(node, a, h) : 0.0;
	}
	double limp(int node, int h) const {
		const int a = act(node, A_LIMP);
		return a >= 0 ? f(node, a, h) : 0.0;
	}
	// share of all hands that get to this decision (the acting player's class counted with his own reach)
	double node_prob(int node) const {
		const Node6& nd = (*nodes)[node];
		double tot = 0;
		for (int h = 0; h < pb->n; h++) {
			double prod = reach[((size_t)node * NPLAYER + nd.player) * NPAD + h];
			for (int q = 0; q < NPLAYER; q++) {
				if (q == nd.player) continue;
				double m = 0;
				for (int j = 0; j < pb->n; j++) m += pb->Wn[(size_t)h * pb->n + j] * reach[((size_t)node * NPLAYER + q) * NPAD + j];
				prod *= m;
			}
			tot += pb->prior[h] * prod;
		}
		return tot;
	}
	int find(NodeType type, int player, int opener, int raiser, bool clean) const {
		for (size_t i = 0; i < nodes->size(); i++) {
			const Node6& n = (*nodes)[i];
			if (n.type != type || n.player != player) continue;
			if (opener >= 0 && n.opener != opener) continue;
			if (raiser >= 0 && n.raiser != raiser) continue;
			if (clean && !n.callers.empty()) continue;
			return (int)i;
		}
		return -1;
	}
};

static double pct_of(const std::vector<double>& w) {  // share of the 1326 combos, in percent
	double s = 0;
	for (int h = 0; h < NCLASS; h++) s += w[h] * g_ncombo[h];
	return 100.0 * s / 1326.0;
}

static std::string jnum(double x, int digits = 3) {
	char b[48];
	snprintf(b, sizeof b, "%.*f", digits, x);
	std::string s = b;
	if (s.find('.') != std::string::npos) {
		while (s.back() == '0') s.pop_back();
		if (s.back() == '.') s.pop_back();
	}
	if (s == "-0") s = "0";
	return s;
}

static std::string jmap(const std::vector<double>& w, double min = 0.005) {  // {"AKs":0.5,...}, strongest first
	std::string out = "{";
	bool first = true;
	for (int h = 0; h < NCLASS; h++) {
		if (w[h] < min) continue;
		if (!first) out += ",";
		first = false;
		out += "\"" + g_names[h] + "\":" + jnum(w[h]);
	}
	return out + "}";
}

static std::string jstr(const std::string& s) { return "\"" + s + "\""; }

struct ConvPoint {
	long long iter;
	Solver6::Gap gap;
};

struct Names {  // the app's names of the seats are the SEAT array
	static std::string pair(int a, int b) { return std::string(SEAT[a]) + "|" + SEAT[b]; }
};

// Builds the JSON and prints the summary. `src` gives the comparison with the source charts.
static std::string write_result(const RunOpts& o, const Rules& R, Solver6& sol, const Source& src, const std::vector<ConvPoint>& conv,
                                long long done2) {
	Reading rd;
	rd.nodes = &sol.nodes;
	rd.hn = &sol.hn;
	rd.pb = &sol.pb;
	rd.avg = sol.averaged();
	rd.reach = sol.reaches();
	const int n = NCLASS;
	const Solver6::Gap fin = sol.exploit();
	std::string js = "{\"schema\":\"sixmax/v1\",\"kind\":\"approximation\",\"stack_bb\":" + jnum(R.stack) +
	                 ",\"description\":\"6-max preflop charts solved by CFR+ on one six-player tree (nvcc/sixmax); an approximation: "
	                 "no equilibrium guarantee for six players, realisation factors from the fixed-policy table, cold calls and cold 4-bets "
	                 "absent, at most two callers of an open.\"";
	js += ",\"sizes\":{\"open\":{";
	for (int p = 0; p < 5; p++) js += (p ? "," : "") + jstr(SEAT[p]) + ":" + jnum(R.open[p]);
	js += "},\"bb_raise_over_sb_limp\":" + jnum(R.iso) + ",\"threebet_x_open_in_position\":" + jnum(R.m3_ip) +
	      ",\"threebet_x_open_out_of_position\":" + jnum(R.m3_oop) + ",\"squeeze_adds_open_per_caller\":true,\"fourbet_x_threebet\":" +
	      jnum(R.m4) + ",\"allin_when_raise_at_least_share_of_stack\":" + jnum(R.allin_frac) + "}";
	js += ",\"iterations\":{\"stage1_first_in_only\":" + std::to_string(o.iters1) + ",\"stage2_all\":" + std::to_string(done2) + "}";
	js += ",\"classes\":[";
	for (int h = 0; h < n; h++) js += (h ? "," : "") + jstr(g_names[h]);
	js += "]";
	js += ",\"convergence\":{\"unit\":\"bb per hand; multiply by 100 for bb/100\",\"gap\":{";
	for (int p = 0; p < NPLAYER; p++) js += (p ? "," : "") + jstr(SEAT[p]) + ":" + jnum(fin.gap(p), 5);
	js += "},\"value\":{";
	for (int p = 0; p < NPLAYER; p++) js += (p ? "," : "") + jstr(SEAT[p]) + ":" + jnum(fin.value[p], 5);
	js += "},\"sum_of_values\":" + jnum(fin.sum_value(), 5) + ",\"total_gap\":" + jnum(fin.total(), 5) + ",\"history\":[";
	for (size_t k = 0; k < conv.size(); k++) {
		js += (k ? "," : "") + std::string("{\"iter\":") + std::to_string(conv[k].iter) + ",\"total_gap\":" + jnum(conv[k].gap.total(), 5) + ",\"gap\":[";
		for (int p = 0; p < NPLAYER; p++) js += (p ? "," : "") + jnum(conv[k].gap.gap(p), 5);
		js += "]}";
	}
	js += "]}";

	printf("\n== result at %gbb (%lld stage-2 iterations) ==\n", R.stack, done2);
	printf("best-response gap bb/100: ");
	for (int p = 0; p < NPLAYER; p++) printf("%s %.3f  ", SEAT[p], 100 * fin.gap(p));
	printf("| total %.3f | sum of values %.4f bb\n", 100 * fin.total(), fin.sum_value());
	printf("value bb/100:             ");
	for (int p = 0; p < NPLAYER; p++) printf("%s %.2f  ", SEAT[p], 100 * fin.value[p]);
	printf("\n");

	// first in
	js += ",\"rfi\":{";
	printf("\nfirst in: solved open %% / source chart %% (stated) / hands that changed by more than 0.5\n");
	for (int s = 0; s < 5; s++) {
		const int nd = rd.find(NT_RFI, s, -1, -1, false);
		std::vector<double> open(n), limp(n);
		for (int h = 0; h < n; h++) {
			open[h] = rd.agg(nd, h);
			limp[h] = rd.limp(nd, h);
		}
		const double p = pct_of(open), ps = pct_of(src.rfi[s]);
		std::string added, dropped;
		double moved = 0;
		for (int h = 0; h < n; h++) {
			moved += std::fabs(open[h] - src.rfi[s][h]) * g_ncombo[h];
			if (open[h] >= 0.5 && src.rfi[s][h] < 0.5) added += g_names[h] + " ";
			if (open[h] < 0.5 && src.rfi[s][h] >= 0.5) dropped += g_names[h] + " ";
		}
		printf("  %-3s %5.1f%% / %5.1f%% (%.1f)  limp %4.1f%%  moved %.1f%% of combos\n    in:  %s\n    out: %s\n", SEAT[s], p, ps,
		       src.stated[s], pct_of(limp), 100.0 * moved / 1326.0, added.c_str(), dropped.c_str());
		js += std::string(s ? "," : "") + jstr(SEAT[s]) + ":{\"pct\":" + jnum(p, 2) + ",\"source_pct\":" + jnum(ps, 2) + ",\"source_stated_pct\":" +
		      jnum(src.stated[s], 1) + ",\"source_chart\":" + jstr(src.key[s]) + ",\"limp_pct\":" + jnum(pct_of(limp), 2) +
		      ",\"changed_share_of_combos_pct\":" + jnum(100.0 * moved / 1326.0, 2) + ",\"reach\":" + jnum(rd.node_prob(nd), 4) +
		      ",\"hands_added\":" + jstr(added) + ",\"hands_dropped\":" + jstr(dropped) + ",\"mix\":" + jmap(open) +
		      (s == 4 ? ",\"limp\":" + jmap(limp) : "") + "}";
	}
	js += "}";

	// facing an open, facing a 3-bet
	js += ",\"vs_open\":{";
	printf("\nfacing an open (hero|opener): call%% / 3-bet%% (bluff%% + value%%)   source in brackets\n");
	bool first = true;
	for (int r = 1; r < 6; r++)
		for (int o2 = 0; o2 < r && o2 < 5; o2++) {
			const int nd = rd.find(NT_RESP, r, o2, -1, true), n4 = rd.find(NT_FACE4, r, o2, r, true);
			if (nd < 0) continue;
			std::vector<double> c(n), b(n), v(n), fo(n);
			for (int h = 0; h < n; h++) {
				c[h] = rd.call(nd, h);
				const double t3 = rd.agg(nd, h);
				const double f4 = n4 >= 0 ? rd.fold(n4, h) : 1.0;
				b[h] = t3 * f4;
				v[h] = t3 * (1 - f4);
				fo[h] = rd.fold(nd, h);
			}
			std::vector<double> sc = src.vo_call[r][o2], sb = src.vo_bluff[r][o2], sv = src.vo_value[r][o2];
			printf("  %-3s|%-3s call %5.1f 3bet %5.1f (%4.1f+%4.1f)   [call %5.1f 3bet %5.1f]  size %g\n", SEAT[r], SEAT[o2], pct_of(c),
			       pct_of(b) + pct_of(v), pct_of(b), pct_of(v), pct_of(sc), pct_of(sb) + pct_of(sv), rd.node_prob(nd) >= 0 ? sol.nodes[nd].bet : 0);
			if (!first) js += ",";
			first = false;
			const int ag = rd.aggr(nd);
			js += jstr(Names::pair(r, o2)) + ":{\"hero\":" + jstr(SEAT[r]) + ",\"opener\":" + jstr(SEAT[o2]) + ",\"open_size_bb\":" +
			      jnum(sol.nodes[nd].bet) + ",\"threebet_size_bb\":" + (ag >= 0 ? jnum(sol.nodes[sol.nodes[nd].child[ag]].bet) : "0") +
			      ",\"reach\":" + jnum(rd.node_prob(nd), 4) + ",\"call_pct\":" + jnum(pct_of(c), 2) + ",\"threebet_pct\":" +
			      jnum(pct_of(b) + pct_of(v), 2) + ",\"source_call_pct\":" + jnum(pct_of(sc), 2) + ",\"source_threebet_pct\":" +
			      jnum(pct_of(sb) + pct_of(sv), 2) + ",\"call\":" + jmap(c) + ",\"bluff\":" + jmap(b) + ",\"value\":" + jmap(v) + "}";
		}
	js += "},\"vs_3bet\":{";
	printf("\nfacing a 3-bet (opener|3-bettor): call%% / 4-bet%% (bluff%% + value%%)   source in brackets\n");
	first = true;
	for (int o2 = 0; o2 < 5; o2++)
		for (int r = o2 + 1; r < 6; r++) {
			const int nd = rd.find(NT_FACE3, o2, o2, r, true), n5 = rd.find(NT_FACE5, o2, o2, r, true);
			if (nd < 0) continue;
			std::vector<double> c(n), b(n), v(n);
			for (int h = 0; h < n; h++) {
				c[h] = rd.call(nd, h);
				const double f4 = rd.agg(nd, h);
				const double fo5 = n5 >= 0 ? rd.fold(n5, h) : 1.0;
				b[h] = f4 * fo5;
				v[h] = f4 * (1 - fo5);
			}
			std::vector<double> sc = src.v3_call[o2][r], sb = src.v3_bluff[o2][r], sv = src.v3_value[o2][r];
			printf("  %-3s|%-3s call %5.1f 4bet %5.1f (%4.1f+%4.1f)   [call %5.1f 4bet %5.1f]  3bet %g\n", SEAT[o2], SEAT[r], pct_of(c),
			       pct_of(b) + pct_of(v), pct_of(b), pct_of(v), pct_of(sc), pct_of(sb) + pct_of(sv), sol.nodes[nd].bet);
			if (!first) js += ",";
			first = false;
			js += jstr(Names::pair(o2, r)) + ":{\"hero\":" + jstr(SEAT[o2]) + ",\"threebettor\":" + jstr(SEAT[r]) + ",\"threebet_size_bb\":" +
			      jnum(sol.nodes[nd].bet) + ",\"reach\":" + jnum(rd.node_prob(nd), 4) + ",\"call_pct\":" + jnum(pct_of(c), 2) +
			      ",\"fourbet_pct\":" + jnum(pct_of(b) + pct_of(v), 2) + ",\"source_call_pct\":" + jnum(pct_of(sc), 2) +
			      ",\"source_fourbet_pct\":" + jnum(pct_of(sb) + pct_of(sv), 2) + ",\"call\":" + jmap(c) + ",\"bluff\":" + jmap(b) +
			      ",\"value\":" + jmap(v) + "}";
		}
	js += "}";

	// every other decision that is reached at least a little: squeezes, facing a 4-bet or 5-bet, the limp pots
	js += ",\"other_nodes\":[";
	first = true;
	int kept = 0;
	double capmass = 0, openmass = 0;
	for (size_t i = 0; i < sol.nodes.size(); i++) {
		const Node6& nd = sol.nodes[i];
		if (nd.is_term()) continue;
		const double pr = rd.node_prob((int)i);
		if (nd.type == NT_RFI && nd.player < 5) {
			(void)0;
		}
		if (nd.type == NT_RESP && (int)nd.callers.size() == R.max_callers) capmass += pr;
		if (nd.type == NT_RESP && nd.callers.empty()) openmass += pr;
		const bool shown = nd.type == NT_RFI || (nd.type == NT_RESP && nd.callers.empty()) || (nd.type == NT_FACE3 && nd.callers.empty());
		if (shown || pr < 0.0005) continue;
		if (!first) js += ",";
		first = false;
		kept++;
		js += "{\"id\":" + jstr(nd.hist) + ",\"type\":" +
		      jstr(nd.type == NT_LIMP_BB ? "bb_vs_limp" : nd.type == NT_LIMP_SB ? "sb_vs_iso" : nd.type == NT_RESP ? "vs_open_after_callers" :
		           nd.type == NT_FACE3 ? "vs_3bet_after_callers" : nd.type == NT_FACE3C ? "caller_vs_squeeze" : nd.type == NT_FACE4 ? "vs_4bet" : "vs_5bet") +
		      ",\"player\":" + jstr(SEAT[nd.player]) + ",\"reach\":" + jnum(pr, 4) + ",\"actions\":[";
		for (size_t a = 0; a < nd.labels.size(); a++) js += (a ? "," : "") + jstr(nd.labels[a]);
		js += "],\"freq\":{";
		bool f2 = true;
		for (int h = 0; h < n; h++) {
			if (rd.reach[((size_t)i * NPLAYER + nd.player) * NPAD + h] < 0.01) continue;
			if (!f2) js += ",";
			f2 = false;
			js += jstr(g_names[h]) + ":[";
			for (size_t a = 0; a < nd.labels.size(); a++) js += (a ? "," : "") + jnum(rd.f((int)i, (int)a, h), 2);
			js += "]";
		}
		js += "}}";
	}
	js += "]";
	printf("\n%d further decision nodes written (reach >= 0.05%%).\n", kept);
	printf("Share of 'facing an open' decisions where a third caller was not allowed: %.2f%% of reach\n",
	       openmass > 0 ? 100 * capmass / openmass : 0.0);
	js += ",\"limits\":{\"third_caller_forbidden_reach_share\":" + jnum(openmass > 0 ? capmass / openmass : 0, 5) + "}}";
	return js;
}

static bool is_rfi(const Node6& n) { return n.type == NT_RFI; }

static int run_solve(const RunOpts& o) {
	const std::string dd = data_dir();
	Rules R;
	R.stack = o.stack;
	std::vector<Node6> nodes = build_tree(R);
	printf("tree: %zu nodes at %gbb\n", nodes.size(), R.stack);
	const std::vector<double> eq = sx_read_matrix("data/equity-exact.bin");
	const std::vector<uint16_t> e3 = e3_load_table("data/equity3.bin");
	const Json real = json_load("data/realization.json");
	Source src = load_source(dd);
	Problem6 pb = make_problem6(eq, &real, &e3, R.stack, true);
	printf("inputs loaded\n");
	std::vector<bool> frozen1(nodes.size(), false), none(nodes.size(), false);
	for (size_t i = 0; i < nodes.size(); i++) frozen1[i] = !nodes[i].is_term() && !is_rfi(nodes[i]);
	Solver6 sol(pb, nodes, frozen1);
	const double eps = 0.02;
	sol.seed(init_sigma(nodes, sol.hn, sol.S, src, frozen1, eps), o.seed_k);

	auto line = [&](const char* tag, long long it, const Solver6::Gap& g) {
		printf("%s %7lld  gap bb/100:", tag, it);
		for (int p = 0; p < NPLAYER; p++) printf(" %s %.3f", SEAT[p], 100 * g.gap(p));
		printf("  total %.3f  sum of values %.4f\n", 100 * g.total(), g.sum_value());
		fflush(stdout);
	};
	// stage 1: first-in decisions only, every reply as in the source charts
	for (long long it = 0; it < o.iters1;) {
		const long long step = std::min(o.report, o.iters1 - it);
		sol.iterate(step);
		it += step;
		line("stage1", it, sol.exploit());
	}
	// stage 2: everything, starting from stage 1
	std::vector<double> sigma = sol.averaged();
	std::vector<bool> fz2(nodes.size(), false);
	for (size_t i = 0; i < nodes.size(); i++) {
		if (nodes[i].is_term()) continue;
		const int k = (int)nodes[i].child.size();
		for (int a = 0; a < k; a++)
			for (int h = 0; h < NCLASS; h++) {
				double& x = sigma[(size_t)(sol.hn[i].aoff + a) * NPAD + h];
				x = (1 - eps) * x + eps / k;
			}
	}
	sol.set_frozen(fz2);
	sol.seed(sigma, o.seed_k);
	if (!o.ckpt.empty() && sol.load((o.ckpt + ".ckpt").c_str())) printf("resumed stage 2 at iteration %lld\n", sol.done);
	std::vector<ConvPoint> conv;
	while (sol.done < o.iters2) {
		const long long step = std::min(o.report, o.iters2 - sol.done);
		sol.iterate(step);
		const Solver6::Gap g = sol.exploit();
		conv.push_back({sol.done, g});
		line("stage2", sol.done, g);
		if (!o.ckpt.empty()) sol.save((o.ckpt + ".ckpt").c_str());
	}
	std::string js = write_result(o, R, sol, src, conv, sol.done);
	const std::string path = o.out + "/sixmax-" + fmtnum(R.stack) + ".json";
	FILE* f = fopen(path.c_str(), "wb");
	if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); return 1; }
	fwrite(js.data(), 1, js.size(), f);
	fclose(f);
	printf("wrote %s (%zu bytes)\n", path.c_str(), js.size());
	return 0;
}

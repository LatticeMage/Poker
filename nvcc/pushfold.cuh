// Heads-up push/fold tables for the app (2-20bb): the small blind shoves or folds, the big blind calls or folds,
// cEV, no ICM. The game is solved by the CFR+ solver of solve.cuh on the push/fold tree (build_pushfold_tree)
// with the exact equity matrix; this file turns the averaged strategies into the data the app reads
// (src/lib/poker/data/tables.json "pushfold" and pushfold-mix.json) and measures how good they are with an
// evaluator that does not share code with the solver. Used by pushfold.cu and test_pushfold.cu.
//
// Payoffs, in bb, relative to the posted blinds (SB 0.5, BB 1): SB folds -0.5; SB shoves and BB folds +1;
// both all-in: stack * (2 * equity - 1). They are those of tools/make-tables.ts, which solved the same game by
// fictitious play on a noisy matrix.
#pragma once

#include "solve.cuh"

// One stack depth, solved.
struct PfStack {
	double stack = 0;
	std::vector<double> push_mix, call_mix;  // averaged strategy of the solver, per class: how often it shoves / calls
	std::vector<int> push, call;             // the pure ranges: the classes whose mix is above one half
	double push_pct = 0, call_pct = 0;       // share of the 1326 combos in each range
	double expl_mixed = 0;                   // best-response gain of the averaged (mixed) strategy, bb per hand
	double expl_pure = 0;                    // the same for the pure ranges, which is what the app shows
	double sb_value = 0;                     // value of the averaged profile for the small blind
};

// -- evaluator (host) ------------------------------------------------------------------------------
// Best-response gain of a profile, in bb per hand dealt, both seats added: for each hand the best of its two
// actions against the other side's profile, minus what the profile earns, weighted by how often the hand is
// dealt. push[h] is the probability that SB class h shoves, call[g] that BB class g calls a shove; these may
// be mixed or 0/1. The big blind's values count only the hands in which he is shoved at (his gain is not
// divided by how often that happens), which is the per-hand-dealt figure the solver also reports.
struct PfEval {
	const Problem& pb;
	const std::vector<double>& eq;  // eq[i*n+j]: share of the pot class i wins against class j

	double sb_push_value(int h, double stack, const std::vector<double>& call) const {
		const int n = pb.n;
		double v = 0;
		for (int g = 0; g < n; g++) {
			const double w = pb.W[h * n + g];
			v += w * (call[g] * (stack * (2 * eq[h * n + g] - 1)) + (1 - call[g]) * 1.0);
		}
		return v / pb.wsum[h];
	}
	// BB class g: value of calling and of folding, each summed over the shoving hands only.
	void bb_values(int g, double stack, const std::vector<double>& push, double* call_v, double* fold_v) const {
		const int n = pb.n;
		double c = 0, f = 0;
		for (int h = 0; h < n; h++) {
			const double w = pb.W[h * n + g] * push[h];
			c += w * (stack * (2 * (1 - eq[h * n + g]) - 1));
			f -= w;
		}
		*call_v = c / pb.wsum[g];
		*fold_v = f / pb.wsum[g];
	}
	double exploitability(double stack, const std::vector<double>& push, const std::vector<double>& call) const {
		double gap_sb = 0, gap_bb = 0;
		for (int h = 0; h < pb.n; h++) {
			const double v = sb_push_value(h, stack, call), fold = -0.5;
			gap_sb += pb.prior[h] * (std::max(v, fold) - (push[h] * v + (1 - push[h]) * fold));
		}
		for (int g = 0; g < pb.n; g++) {
			double c, f;
			bb_values(g, stack, push, &c, &f);
			gap_bb += pb.prior[g] * (std::max(c, f) - (call[g] * c + (1 - call[g]) * f));
		}
		return gap_sb + gap_bb;
	}
	// The value of the profile for the small blind, bb per hand (relative to the blinds).
	double sb_value(double stack, const std::vector<double>& push, const std::vector<double>& call) const {
		double s = 0;
		for (int h = 0; h < pb.n; h++)
			s += pb.prior[h] * (push[h] * sb_push_value(h, stack, call) + (1 - push[h]) * -0.5);
		return s;
	}
};

static std::vector<double> to_pure(const std::vector<double>& mix) {
	std::vector<double> p(mix.size());
	for (size_t i = 0; i < mix.size(); i++) p[i] = mix[i] > 0.5 ? 1 : 0;
	return p;
}

static double range_pct(const std::vector<double>& flags) {
	double hit = 0, all = 0;
	for (size_t i = 0; i < flags.size(); i++) {
		hit += flags[i] * g_ncombo[i];
		all += g_ncombo[i];
	}
	return 100 * hit / all;
}

// -- solving -----------------------------------------------------------------------------------------
// The node ids of a push/fold tree (see build_pushfold_tree): the small blind decides at node 0, the big
// blind at node 2. In each node action 1 is the aggressive one (allin / call).
static const int PF_SB_NODE = 0, PF_BB_NODE = 2;

// Everything is read from one Solver whose trees are the push/fold trees of `stacks` (in this order).
static std::vector<PfStack> collect(Solver& sv, const std::vector<double>& stacks, const std::vector<double>& eq) {
	const int n = sv.pb.n;
	std::vector<double> avg = sv.averaged();
	auto ex = sv.exploitability();
	auto ev = sv.value();
	PfEval host = {sv.pb, eq};
	std::vector<PfStack> out(stacks.size());
	for (size_t i = 0; i < stacks.size(); i++) {
		PfStack& r = out[i];
		r.stack = stacks[i];
		const int sb = sv.slot_of((int)i, PF_SB_NODE), bb = sv.slot_of((int)i, PF_BB_NODE);
		r.push_mix.resize(n);
		r.call_mix.resize(n);
		for (int h = 0; h < n; h++) {
			r.push_mix[h] = avg[(size_t)(sb + 1) * NP + h];
			r.call_mix[h] = avg[(size_t)(bb + 1) * NP + h];
		}
		const std::vector<double> push = to_pure(r.push_mix), call = to_pure(r.call_mix);
		for (int h = 0; h < n; h++) {
			if (push[h]) r.push.push_back(h);
			if (call[h]) r.call.push_back(h);
		}
		r.push_pct = range_pct(push);
		r.call_pct = range_pct(call);
		r.expl_mixed = ex[i].first + ex[i].second;
		r.expl_pure = host.exploitability(stacks[i], push, call);
		r.sb_value = ev[i].first;
	}
	return out;
}

// -- the app's data (JSON) -----------------------------------------------------------------------------
// Math.round(x * 10^digits) / 10^digits, printed without trailing zeros, like JSON.stringify of the TypeScript
// tool's numbers.
static std::string fmt(double x, int digits) {
	char b[64];
	snprintf(b, sizeof b, "%.*f", digits, std::floor(x * std::pow(10.0, digits) + 0.5) / std::pow(10.0, digits));
	std::string s = b;
	if (s.find('.') != std::string::npos) {
		while (s.back() == '0') s.pop_back();
		if (s.back() == '.') s.pop_back();
	}
	return s == "-0" ? "0" : s;
}

// Share of each class's non-conflicting combos in which it wins, against a random hand: the strength order of
// the opening charts (tools/make-tables.ts computed the same sums from the noisy matrix).
static std::vector<double> equity_vs_random(const Problem& pb, const std::vector<double>& eq) {
	std::vector<double> out(pb.n);
	for (int i = 0; i < pb.n; i++) {
		double w = 0, t = 0;
		for (int j = 0; j < pb.n; j++) {
			w += pb.W[i * pb.n + j] * eq[i * pb.n + j];
			t += pb.W[i * pb.n + j];
		}
		out[i] = w / t;
	}
	return out;
}

// One JSON document with the three things the app reads plus the solver's own report:
//   equity_vs_random, pushfold (push/call lists in class order, percentages, exploitability of the pure ranges),
//   mix (class -> frequency, only the classes at or above 2%), and solver (not read by the app).
static std::string pushfold_json(const std::vector<PfStack>& rs, long long iterations, const std::vector<double>& evr) {
	std::string pf, mix, sol, eqs;
	for (int i = 0; i < NCLASS; i++) eqs += std::string(i ? "," : "") + "\"" + g_names[i] + "\":" + fmt(evr[i], 4);
	auto freqs = [](const std::vector<double>& f) {
		std::string s = "{";
		bool first = true;
		for (int i = 0; i < NCLASS; i++) {
			if (f[i] < 0.02) continue;
			s += std::string(first ? "" : ",") + "\"" + g_names[i] + "\":" + fmt(std::min(1.0, f[i]), 2);
			first = false;
		}
		return s + "}";
	};
	auto names = [](const std::vector<int>& v) {
		std::string s = "[";
		for (size_t k = 0; k < v.size(); k++) s += std::string(k ? "," : "") + "\"" + g_names[v[k]] + "\"";
		return s + "]";
	};
	for (size_t i = 0; i < rs.size(); i++) {
		const PfStack& r = rs[i];
		const std::string key = "\"" + num(r.stack) + "\":", sep = i ? "," : "";
		pf += sep + key + "{\"push\":" + names(r.push) + ",\"call\":" + names(r.call) + ",\"push_pct\":" + fmt(r.push_pct, 1) +
		      ",\"call_pct\":" + fmt(r.call_pct, 1) + ",\"exploitability_bb\":" + fmt(r.expl_pure, 6) + "}";
		mix += sep + key + "{\"push\":" + freqs(r.push_mix) + ",\"call\":" + freqs(r.call_mix) + "}";
		sol += sep + key + "{\"exploitability_mixed_bb\":" + fmt(r.expl_mixed, 9) + ",\"exploitability_pure_bb\":" +
		       fmt(r.expl_pure, 9) + ",\"sb_value_bb\":" + fmt(r.sb_value, 6) + "}";
	}
	return "{\"samples_per_matchup\":0,\"equity_vs_random\":{" + eqs + "},\"pushfold\":{" + pf + "},\"mix\":{" + mix +
	       "},\"solver\":{\"iterations\":" + std::to_string(iterations) + ",\"stacks\":{" + sol + "}}}\n";
}

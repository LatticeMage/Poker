// The realisation-factor table as the solver's value of a call that sees a flop (solve.cuh, host side).
//
// Today's solver values every such call with one formula, realisedIp(eq) (solve.cuh). Here the value comes
// from data/realization.json (tool `realization`): a factor per hand class, side (in position = the small
// blind, out of position = the big blind), pot type and depth, "chips won / final pot over raw equity".
//
// How a tree node becomes a matrix (the design, in the order of the questions it answers):
//
// 1. Pot type. A call that sees a flop is single-raised after one raise (an open), a 3-bet pot after two, a
//    4-bet pot after three or more (the GNode::pot of the tree builders). A call of an all-in keeps the raw
//    equity. The table has no limped pots (nobody opened, or the big blind raised a limp): those calls keep
//    the old formula (default), or borrow the single-raised factors (`limp = LIMP_SRP`, a sensitivity run).
//
// 2. Depth. The table holds depths 20, 40, 80, 100 bb (the 4-bet pot: 40, 80, 100). What matters is the
//    stack behind on the flop, x = stack - c (c = what each has put in, the node's own c0); the table's
//    x_k = depth_k - c_table. Between two table depths the factor is interpolated linearly in x. Above the
//    deepest it is the deepest one (150 bb uses the 100 bb factors: no extrapolation). Below the shallowest
//    it is interpolated towards 1 at x = 0: with nothing behind the flop cannot change the result (the
//    realisation test shows factors within 2% of 1 at SPR 0.02), and the distance from 1 grows with the
//    stack. So 5 to 15 bb get a fraction of the 20 bb deviation, never an extrapolated one.
//
// 3. Zero-sum. The table gives a factor per class and side, not per matchup. For small-blind hand i and
//    big-blind hand j with raw equity e of the small blind, the realised shares are
//        a = f_ip(i) e,   b = f_oop(j) (1 - e),   small blind gets a / (a + b),  big blind b / (a + b).
//    The two shares add to exactly 1 by construction (so every call is zero-sum, the pot is split fully), and
//    with f_ip = f_oop = 1 the small blind's share is e. This is an approximation: the table factor of a class
//    is its average against the opponent's whole range at the table's ranges, not against hand j; the
//    normalisation spreads the gap between the two sides' factors evenly (in ratio) instead of from a model of
//    that matchup; and the average over a range of the matchup shares is not the table's range factor.
//
// 4. Classes outside the range the table was made with (the ranges are the 6-max charts at 100 bb; the
//    table says which classes those are) have inflated factors, up to 1.7. `out = OUT_CAP` (default)
//    limits their factor to the equity-weighted factor of the side's whole range at that depth (an
//    out-of-range hand realises at most like an average hand of the range). `OUT_FORMULA` gives them the
//    old formula instead, in every matchup; `OUT_NONE` takes the table as it is (sensitivity run only).
#pragma once

#include <map>
#include <tuple>

#include "ranges.cuh"
#include "solve.cuh"

enum RealOut { OUT_CAP = 0, OUT_FORMULA = 1, OUT_NONE = 2 };
enum RealLimp { LIMP_FORMULA = 0, LIMP_SRP = 1 };

struct RealOptions {
	int out = OUT_CAP;
	int limp = LIMP_FORMULA;
	std::string describe() const {
		static const char* O[3] = {"cap", "formula", "none"};
		return std::string("table:realization/v1,out-of-range=") + O[out] + ",limped=" + (limp == LIMP_SRP ? "single-raised" : "formula");
	}
};

struct RealTable {
	struct Depth {
		double depth = 0, x = 0;  // depth_bb, stack behind on the flop (depth - c)
		std::vector<double> factor[2];  // [side: 0 = IP / small blind, 1 = OOP / big blind][class]
		double range_factor[2] = {1, 1};
	};
	struct Pot {
		std::string name;
		double c = 0;  // each player's chips in before the flop
		std::vector<char> inrange[2];
		std::vector<Depth> depths;  // ascending
	};
	std::vector<Pot> pots;  // srp, 3bet, 4bet in file order

	static RealTable load(const std::string& path) {
		RealTable t;
		const Json j = json_load(path);
		if (j.at("schema").str != "realization/v1") throw std::runtime_error(path + ": not a realization/v1 file");
		if (j.at("classes").arr.size() != (size_t)NCLASS) throw std::runtime_error("class count");
		for (int h = 0; h < NCLASS; h++)
			if (j.at("classes").arr[h].str != g_names[h]) throw std::runtime_error("class order differs from poker.cuh");
		for (const Json& p : j.at("pots").arr) {
			Pot pot;
			pot.name = p.at("pot").str;
			pot.c = p.at("preflop_each_bb").num;
			for (int s = 0; s < 2; s++) {
				const Json& r = p.at(s ? "oop_range" : "ip_range");
				for (const Json& v : r.arr) pot.inrange[s].push_back(v.num > 0 ? 1 : 0);
			}
			for (const Json& d : p.at("depths").arr) {
				Depth dp;
				dp.depth = d.at("depth_bb").num;
				dp.x = dp.depth - pot.c;
				for (int s = 0; s < 2; s++) {
					const Json& side = d.at(s ? "oop" : "ip");
					for (const Json& v : side.at("factor").arr) dp.factor[s].push_back(v.num);
					dp.range_factor[s] = side.at("range_summary").at("factor").num;
					if (dp.factor[s].size() != (size_t)NCLASS) throw std::runtime_error("factor count");
				}
				pot.depths.push_back(dp);
			}
			t.pots.push_back(pot);
		}
		return t;
	}

	// Factors of one side for stack-behind x. `formula[h]` is set for the classes that use the old formula
	// (out of range with OUT_FORMULA); their factor entry is 1 and unused.
	void factors(int pot, int side, double x, int out, std::vector<double>& f, std::vector<char>& formula) const {
		const Pot& p = pots[pot];
		f.assign(NCLASS, 1.0);
		formula.assign(NCLASS, 0);
		// anchors: x = 0 (factor 1), then the table depths; each anchor's per-class value after the range rule
		auto anchor = [&](const Depth& d, int h) {
			double v = d.factor[side][h];
			if (!p.inrange[side][h] && out == OUT_CAP) v = std::min(v, d.range_factor[side]);
			return v;
		};
		const Depth* lo = nullptr;  // anchor below x (nullptr = the x = 0 anchor)
		const Depth* hi = nullptr;  // anchor at or above x
		for (const Depth& d : p.depths) {
			if (d.x >= x) { hi = &d; break; }
			lo = &d;
		}
		for (int h = 0; h < NCLASS; h++) {
			if (!p.inrange[side][h] && out == OUT_FORMULA) { formula[h] = 1; continue; }
			double v;
			if (x <= 0) v = 1;
			else if (!hi) v = anchor(*lo, h);  // deeper than the table: its deepest depth
			else {
				const double x0 = lo ? lo->x : 0, v0 = lo ? anchor(*lo, h) : 1.0, v1 = anchor(*hi, h);
				v = v0 + (v1 - v0) * (x - x0) / (hi->x - x0);
			}
			f[h] = v;
		}
	}
};

// One matrix set for a call with given factors. Layout of Problem::M: [own hand * n + opponent hand],
// small blind in m0, big blind in m1. fa / fb: small blind's / big blind's factors; fma / fmb: use the
// formula for that class. Where both factors are exactly 1 the entry is the raw one (same expression as
// make_problem), so factors of 1 reproduce raw equity exactly.
static void build_real_set(const Problem& pb, const std::vector<double>& equity, const std::vector<double>& fa,
                           const std::vector<double>& fb, const std::vector<char>& fma, const std::vector<char>& fmb,
                           std::vector<double>& m0, std::vector<double>& m1) {
	const int n = pb.n;
	m0.assign((size_t)n * n, 0);
	m1.assign((size_t)n * n, 0);
	for (int i = 0; i < n; i++)  // i: small blind's hand
		for (int j = 0; j < n; j++) {  // j: big blind's hand
			const double w = pb.W[i * n + j], e = equity[i * n + j];
			if (!fma[i] && !fmb[j] && fa[i] == 1.0 && fb[j] == 1.0) {
				m0[i * n + j] = w * e;
				m1[j * n + i] = w * (1 - e);  // the expressions of make_problem
				continue;
			}
			const double a = fma[i] ? realised_ip(e) : fa[i] * e;
			const double b = fmb[j] ? 1 - realised_ip(e) : fb[j] * (1 - e);
			const double share = a / (a + b);  // small blind's realised share of the pot
			m0[i * n + j] = w * share;
			m1[j * n + i] = w * (1 - share);
		}
}

// Gives every non-all-in showdown of every tree a matrix set (GNode::mset), appending the sets to
// pb.extra. A set is shared by all nodes with the same stack, pot type and chips in.
static int apply_realisation(Problem& pb, const std::vector<double>& equity, std::vector<GameTree>& trees,
                             const RealTable& tab, const RealOptions& opt) {
	std::map<std::tuple<double, int, double>, int> sets;
	for (GameTree& t : trees)
		for (GNode& nd : t.nodes) {
			if (nd.kind != SHOW || nd.allin) continue;
			int pot = nd.pot;  // 0 limped, 1..3
			if (pot == 0) {
				if (opt.limp == LIMP_FORMULA) continue;
				pot = 1;
			}
			pot = std::min(pot, (int)tab.pots.size()) - 1;
			const auto key = std::make_tuple(t.stack, pot, nd.c0);
			auto it = sets.find(key);
			if (it == sets.end()) {
				const double x = t.stack - nd.c0;
				std::vector<double> fa, fb;
				std::vector<char> ma, mb;
				tab.factors(pot, 0, x, opt.out, fa, ma);
				tab.factors(pot, 1, x, opt.out, fb, mb);
				std::vector<double> m0, m1;
				build_real_set(pb, equity, fa, fb, ma, mb, m0, m1);
				pb.extra.push_back(m0);
				pb.extra.push_back(m1);
				it = sets.emplace(key, (int)pb.extra.size() / 2 + 1).first;  // set number = 2 + k
			}
			nd.mset = it->second;
		}
	return (int)sets.size();
}

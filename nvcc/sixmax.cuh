// The 6-max preflop solver: game tree, CFR+ kernels, host driver. Used by sixmax.cu (the command line) and by
// test_sixmax.cu (the checks). Inputs: the exact 169x169 equity matrix, the three-way table (equity3), the
// realisation factors (realization.json) and the source charts (charts.json, spots.json).
//
// WHAT IS SOLVED (read together with the "6-max preflop solver" section of README.md)
//
// Six seats UTG, MP (the app's name for the hijack), CO, BTN, SB, BB, equal stacks, blinds 0.5 / 1. Every
// player has a strategy over the 169 hand classes at every decision, and all six are trained at once by
// vector-form CFR+ (the heads-up solver's algorithm, see solve.cuh) on ONE tree that contains the whole hand:
//
//   first in      UTG, MP, CO, BTN: fold or raise to 2.5.  SB: fold, limp or raise to 3.  If it folds to the BB
//                 the BB wins the blinds. After an SB limp the BB checks or raises to 4 (SB then folds or calls).
//   responding    after an open the seats behind it act in turn: fold, call, or 3-bet. The size is 3x the open
//                 when the 3-bettor will be in position after the flop, 4x when out of position, plus one open
//                 per caller already in (a squeeze). At most two players may call an open (three see the flop).
//   3-bet         everybody except the opener and the 3-bettor folds (cold calls and cold 4-bets are not in
//                 the game; players who called the open fold to the 3-bet unless the opener folds first, then they
//                 may call it, in turn). The opener folds, calls or 4-bets (2.2x the 3-bet).
//   4-bet         the 3-bettor folds, calls or goes all-in; the opener then folds or calls the all-in.
//   a raise of 70% of the stack or more is an all-in.
//
// Showdowns: two players use a payoff matrix per pot type (raw equity if all-in, realisation-adjusted
// otherwise, see Problem6); three players (open called twice) use the exact three-way table. Players who
// folded enter only through their reach (see "card removal" below).
//
// CARD REMOVAL. A value is the expected chips of a hero holding class h, given h. Another player's chance of
// holding class j is Wn[h][j] = (combo pairs of h and j that do not clash) / (combos of h * 1225), so a folded
// player q contributes the factor m_q[h] = sum_j Wn[h][j] * reach_q[j] (the probability that q takes the path
// to this terminal). Two players in a showdown use the conflict-free pair weights (heads-up) or the exact
// triple counts N of equity3 (three-way). What is approximated: the removal among the OTHER players (their
// hands do not clash with each other in reality, here only with the hero's). The error is of the order of one
// card in fifty; the game is therefore only zero-sum up to that, and the checks measure it.
#pragma once

#include <functional>
#include <map>
#include <set>

#include "equity3.cuh"
#include "poker.cuh"
#include "ranges.cuh"

static const int NPAD = 192;    // hand slots per vector (169 padded to 6 warps; thread h = class h)
static const int MAXACT = 6;    // most actions at one decision
static const int NPLAYER = 6;
static const char* SEAT[NPLAYER] = {"UTG", "MP", "CO", "BTN", "SB", "BB"};
// Order of action after the flop: SB first, BTN last. Higher = later = in position.
static const int POSTPOS[NPLAYER] = {2, 3, 4, 5, 0, 1};

// Pot types of a showdown.
enum Cat { CAT_LIMP = 0, CAT_SRP, CAT_3BET, CAT_4BET, CAT_ALLIN, NCAT };
static const char* CAT_NAME[NCAT] = {"limp", "srp", "3bet", "4bet", "allin"};

// -- the game tree (host) --------------------------------------------------------------------------
enum NodeType { NT_RFI, NT_LIMP_BB, NT_LIMP_SB, NT_RESP, NT_FACE3, NT_FACE4, NT_FACE5, NT_FACE3C, NT_TERM };
enum ActKind { A_FOLD, A_CALL, A_RAISE, A_ALLIN, A_LIMP };

struct Node6 {
	NodeType type = NT_TERM;
	int player = -1;
	std::vector<int> child, akind;
	std::vector<std::string> labels;
	int opener = -1, raiser = -1;  // opener: the first raiser; raiser: the 3-bettor (face3, face4, face5)
	std::vector<int> callers;      // players who called the open before this node
	std::string hist;              // what happened so far, for the output
	double bet = 0;                // the amount to match at a decision (to-size of the last raise)
	// terminals
	int live = 0;                  // bit mask of the players still in the hand
	double contrib[NPLAYER] = {};  // chips each player has put in (folded blinds included)
	int cat = -1;                  // showdown pot type
	int ip = -1, oop = -1;         // heads-up showdown: who acts last after the flop, who first
	bool is_term() const { return type == NT_TERM; }
};

struct Rules {
	double stack = 100;
	double open[NPLAYER] = {2.5, 2.5, 2.5, 2.5, 3.0, 0};  // raise-first-in size; the SB's is the source charts' 3
	double iso = 4.0;                                      // the BB's raise over an SB limp
	double m3_ip = 3.0, m3_oop = 4.0, m4 = 2.2;            // 3-bet: x the open (in / out of position), 4-bet: x the 3-bet
	double allin_frac = 0.7;                               // a raise to this share of the stack or more is all-in
	int max_callers = 2;
};

static std::string fmtnum(double x) {
	char b[32];
	snprintf(b, sizeof b, "%g", x);
	return b;
}

struct TreeBuilder {
	Rules R;
	std::vector<Node6> nodes;

	int add(const Node6& n) { nodes.push_back(n); return (int)nodes.size() - 1; }
	static double half(double x) { return std::floor(x * 2 + 0.5) / 2; }

	int term(int live, const double* c, int cat, const std::string& hist) {
		Node6 t;
		t.type = NT_TERM;
		t.live = live;
		memcpy(t.contrib, c, sizeof t.contrib);
		t.cat = cat;
		t.hist = hist;
		if (__builtin_popcount(live) == 2) {
			int a = -1, b = -1;
			for (int p = 0; p < NPLAYER; p++)
				if (live >> p & 1) (a < 0 ? a : b) = p;
			t.ip = POSTPOS[a] > POSTPOS[b] ? a : b;
			t.oop = t.ip == a ? b : a;
		}
		return add(t);
	}

	double size3(int o, int r, int ncall) const {
		const double base = R.open[o] * (POSTPOS[r] > POSTPOS[o] ? R.m3_ip : R.m3_oop) + ncall * R.open[o];
		const double s = half(base);
		return s >= R.allin_frac * R.stack ? R.stack : s;
	}
	double size4(double t3) const {
		const double s = half(R.m4 * t3);
		return (s >= R.allin_frac * R.stack || s <= t3) ? R.stack : s;
	}

	static std::string step(const std::string& h, int seat, const std::string& what) {
		return h + (h.empty() ? "" : " ") + SEAT[seat] + ":" + what;
	}

	int decision(NodeType type, int player, const std::string& hist) {
		Node6 d;
		d.type = type;
		d.player = player;
		d.hist = hist;
		return add(d);
	}
	void action(int d, ActKind kind, const std::string& label, int child) {
		nodes[d].akind.push_back(kind);
		nodes[d].labels.push_back(label);
		nodes[d].child.push_back(child);
	}

	int firstin(int s, const double* c, const std::string& hist) {
		if (s == 5) return term(1 << 5, c, -1, hist + " all fold");  // the big blind wins the blinds
		const int d = decision(NT_RFI, s, hist);
		action(d, A_FOLD, "fold", firstin(s + 1, c, step(hist, s, "f")));
		if (s == 4) {  // the small blind may complete
			double c2[NPLAYER];
			memcpy(c2, c, sizeof c2);
			c2[4] = 1;
			const int bb = decision(NT_LIMP_BB, 5, step(hist, 4, "limp"));
			nodes[bb].opener = 4;
			double c3[NPLAYER];
			memcpy(c3, c2, sizeof c3);
			action(bb, A_CALL, "check", term((1 << 4) | (1 << 5), c2, CAT_LIMP, step(nodes[bb].hist, 5, "check")));
			c3[5] = R.iso;
			const int sb = decision(NT_LIMP_SB, 4, step(nodes[bb].hist, 5, "r" + fmtnum(R.iso)));
			nodes[sb].opener = 5;
			double c4[NPLAYER];
			memcpy(c4, c3, sizeof c4);
			action(sb, A_FOLD, "fold", term(1 << 5, c3, -1, step(nodes[sb].hist, 4, "f")));
			c4[4] = R.iso;
			action(sb, A_CALL, "call", term((1 << 4) | (1 << 5), c4, CAT_SRP, step(nodes[sb].hist, 4, "c")));
			action(bb, A_RAISE, "raise " + fmtnum(R.iso), sb);
			action(d, A_LIMP, "limp", bb);
		}
		double c2[NPLAYER];
		memcpy(c2, c, sizeof c2);
		c2[s] = R.open[s];
		action(d, A_RAISE, "raise " + fmtnum(R.open[s]), respond(s, R.open[s], c2, {}, s + 1, step(hist, s, "r" + fmtnum(R.open[s]))));
		return d;
	}

	// The seats behind the opener answer in turn.
	int respond(int o, double bet, const double* c, const std::vector<int>& callers, int r, const std::string& hist) {
		if (r > 5) {  // nobody raised: the opener and the callers see the flop
			int live = 1 << o;
			for (int q : callers) live |= 1 << q;
			if (callers.empty()) return term(live, c, -1, hist);
			return term(live, c, CAT_SRP, hist);
		}
		const int d = decision(NT_RESP, r, hist);
		nodes[d].opener = o;
		nodes[d].callers = callers;
		nodes[d].bet = bet;
		action(d, A_FOLD, "fold", respond(o, bet, c, callers, r + 1, step(hist, r, "f")));
		if ((int)callers.size() < R.max_callers) {
			double c2[NPLAYER];
			memcpy(c2, c, sizeof c2);
			c2[r] = bet;
			std::vector<int> cl = callers;
			cl.push_back(r);
			action(d, A_CALL, "call", respond(o, bet, c2, cl, r + 1, step(hist, r, "c")));
		}
		const double t3 = size3(o, r, (int)callers.size());
		double c3[NPLAYER];
		memcpy(c3, c, sizeof c3);
		c3[r] = t3;
		action(d, t3 >= R.stack ? A_ALLIN : A_RAISE, t3 >= R.stack ? "allin" : "3bet " + fmtnum(t3),
		       face3(o, r, callers, t3, c3, step(hist, r, t3 >= R.stack ? "allin" : "3b" + fmtnum(t3))));
		return d;
	}

	// The opener folded to a squeeze: the players who called his open answer it, one after the other, with fold or
	// call (their open is dead money in the pot). Without callers the 3-bettor just wins.
	int callers_answer(int o, int r, const std::vector<int>& callers, size_t idx, double t3, const double* c, const std::string& hist) {
		if (idx >= callers.size()) return term(1 << r, c, -1, hist);
		const int q = callers[idx];
		const int d = decision(NT_FACE3C, q, hist);
		nodes[d].opener = o;
		nodes[d].raiser = r;
		nodes[d].callers = callers;
		nodes[d].bet = t3;
		action(d, A_FOLD, "fold", callers_answer(o, r, callers, idx + 1, t3, c, step(hist, q, "f")));
		double c2[NPLAYER];
		memcpy(c2, c, sizeof c2);
		c2[q] = t3;
		action(d, A_CALL, "call", term((1 << q) | (1 << r), c2, t3 >= R.stack ? CAT_ALLIN : CAT_3BET, step(hist, q, "c")));
		return d;
	}

	int face3(int o, int r, const std::vector<int>& callers, double t3, const double* c, const std::string& hist) {
		const int d = decision(NT_FACE3, o, hist);
		nodes[d].opener = o;
		nodes[d].raiser = r;
		nodes[d].callers = callers;
		nodes[d].bet = t3;
		action(d, A_FOLD, "fold", callers_answer(o, r, callers, 0, t3, c, step(hist, o, "f")));
		double c2[NPLAYER];
		memcpy(c2, c, sizeof c2);
		c2[o] = t3;
		action(d, A_CALL, "call", term((1 << o) | (1 << r), c2, t3 >= R.stack ? CAT_ALLIN : CAT_3BET, step(hist, o, "c")));
		if (t3 < R.stack) {
			const double t4 = size4(t3);
			double c4[NPLAYER];
			memcpy(c4, c, sizeof c4);
			c4[o] = t4;
			action(d, t4 >= R.stack ? A_ALLIN : A_RAISE, t4 >= R.stack ? "allin" : "4bet " + fmtnum(t4),
			       face4(o, r, callers, t4, c4, step(hist, o, t4 >= R.stack ? "allin" : "4b" + fmtnum(t4))));
		}
		return d;
	}

	int face4(int o, int r, const std::vector<int>& callers, double t4, const double* c, const std::string& hist) {
		const int d = decision(NT_FACE4, r, hist);
		nodes[d].opener = o;
		nodes[d].raiser = r;
		nodes[d].callers = callers;
		nodes[d].bet = t4;
		action(d, A_FOLD, "fold", term(1 << o, c, -1, step(hist, r, "f")));
		double c2[NPLAYER];
		memcpy(c2, c, sizeof c2);
		c2[r] = t4;
		action(d, A_CALL, "call", term((1 << o) | (1 << r), c2, t4 >= R.stack ? CAT_ALLIN : CAT_4BET, step(hist, r, "c")));
		if (t4 < R.stack) {
			double c5[NPLAYER];
			memcpy(c5, c, sizeof c5);
			c5[r] = R.stack;
			const int f5 = decision(NT_FACE5, o, step(hist, r, "allin"));
			nodes[f5].opener = o;
			nodes[f5].raiser = r;
			nodes[f5].callers = callers;
			nodes[f5].bet = R.stack;
			action(f5, A_FOLD, "fold", term(1 << r, c5, -1, step(nodes[f5].hist, o, "f")));
			double c6[NPLAYER];
			memcpy(c6, c5, sizeof c6);
			c6[o] = R.stack;
			action(f5, A_CALL, "call", term((1 << o) | (1 << r), c6, CAT_ALLIN, step(nodes[f5].hist, o, "c")));
			action(d, A_ALLIN, "allin", f5);
		}
		return d;
	}

	std::vector<Node6> build() {
		nodes.clear();
		double c[NPLAYER] = {0, 0, 0, 0, 0.5, 1};
		firstin(0, c, "");
		return nodes;
	}
};

static std::vector<Node6> build_tree(const Rules& r) {
	TreeBuilder b;
	b.R = r;
	return b.build();
}

// -- the numbers a tree is played with (host) --------------------------------------------------------
// All matrices are n x n row-major, the first index is the hero's class.
struct Problem6 {
	int n = 0;
	std::vector<double> prior;  // share of all combos, sums to 1
	std::vector<double> Wn;     // Wn[h][j]: chance that another player holds class j when the hero holds h
	// showdown of two: A_ip[c][h][j] = Wn[h][j] * S_c[h][j], A_oop[c][h][i] = Wn[h][i] * S_c[i][h], where S_c[i][j] is
	// the share of the pot that the player in position with class i takes against class j (out of position) in a
	// pot of type c; the player out of position takes 1 - S.
	std::vector<double> A_ip[NCAT], A_oop[NCAT];
	std::vector<double> S[NCAT];  // kept for the output and the checks
	// showdown of three: T[s][(own * n + x) * n + y] = {share of slot 0, 1, 2, number of combo triples}, where own is
	// the class of slot s and x, y the classes of the other two slots in slot order
	std::vector<float4> T[3];
	std::vector<double> NS;        // NS[h] = sum over x, y of N(h, x, y)
	std::vector<double> grole[3];  // realisation factor of the class for the player who acts first / second / last after the flop
};

// -- the engine --------------------------------------------------------------------------------------
struct DNode {
	int type;  // 0 decision, 1 terminal
	int player, nact, aoff, frozen;
	int child[MAXACT];
	int live, cat, ip, oop;
	int sl[3], role[3];  // three-way terminal: players in seat order, their order of action after the flop
	double contrib[NPLAYER], pot;
};

struct Ctx6 {
	const DNode* nodes;
	double *reach, *val, *sigma, *regret, *sum;
	const double* t;
	const double* Wn;  // transposed, padded: [j * NPAD + h]
	const double *A_ip[NCAT], *A_oop[NCAT];
	const float4* T[3];
	const double *NS, *grole;
	int n;
};

// Regret matching, then the reach of the children (the acting player's reach is multiplied by his strategy).
__global__ void k_top(Ctx6 c, const int* ids, int regmatch) {
	const int g = ids[blockIdx.x], h = threadIdx.x;
	if (h >= c.n) return;
	const DNode& nd = c.nodes[g];
	const int k = nd.nact, p = nd.player;
	if (regmatch && !nd.frozen) {
		double pos = 0;
		for (int a = 0; a < k; a++) pos += c.regret[(size_t)(nd.aoff + a) * NPAD + h];
		for (int a = 0; a < k; a++)
			c.sigma[(size_t)(nd.aoff + a) * NPAD + h] = pos > 0 ? c.regret[(size_t)(nd.aoff + a) * NPAD + h] / pos : 1.0 / k;
	}
	double r[NPLAYER];
	for (int q = 0; q < NPLAYER; q++) r[q] = c.reach[((size_t)g * NPLAYER + q) * NPAD + h];
	for (int a = 0; a < k; a++) {
		const double s = c.sigma[(size_t)(nd.aoff + a) * NPAD + h];
		const size_t ch = (size_t)nd.child[a] * NPLAYER;
		for (int q = 0; q < NPLAYER; q++) c.reach[(ch + q) * NPAD + h] = q == p ? r[q] * s : r[q];
	}
}

// Terminals: the value of every player (or of `only`) for every class. For a three-way showdown only the
// factor of the players who folded is written here; k_term3 multiplies in the showdown itself.
__global__ void k_term(Ctx6 c, const int* ids, int only) {
	__shared__ double r[NPLAYER][NPAD];
	const int g = ids[blockIdx.x], h = threadIdx.x;
	for (int q = 0; q < NPLAYER; q++) r[q][h] = c.reach[((size_t)g * NPLAYER + q) * NPAD + h];
	__syncthreads();
	if (h >= c.n) return;
	const DNode& nd = c.nodes[g];
	double m[NPLAYER];
	for (int q = 0; q < NPLAYER; q++) {
		double s = 0;
		for (int j = 0; j < c.n; j++) s += c.Wn[(size_t)j * NPAD + h] * r[q][j];
		m[q] = s;
	}
	const int nl = __popc(nd.live);
	for (int p = 0; p < NPLAYER; p++) {
		if (only >= 0 && p != only) continue;
		const bool in = nd.live >> p & 1;
		double v;
		if (nl == 1 || !in) {
			double prod = 1;
			for (int q = 0; q < NPLAYER; q++)
				if (q != p) prod *= m[q];
			const double pay = (nl == 1 && in) ? nd.pot - nd.contrib[p] : -nd.contrib[p];
			v = pay * prod;
		} else {
			double pif = 1;
			for (int q = 0; q < NPLAYER; q++)
				if (!(nd.live >> q & 1)) pif *= m[q];
			if (nl == 2) {
				const int w = nd.live & ~(1 << p) ? __ffs(nd.live & ~(1 << p)) - 1 : 0;
				const double* A = (p == nd.ip ? c.A_ip[nd.cat] : c.A_oop[nd.cat]);
				double dot = 0;
				for (int j = 0; j < c.n; j++) dot += A[(size_t)j * NPAD + h] * r[w][j];
				if (p == nd.ip) v = pif * (nd.pot * dot - nd.contrib[p] * m[w]);
				else v = pif * (nd.pot * (m[w] - dot) - nd.contrib[p] * m[w]);
			} else {
				v = pif;
			}
		}
		c.val[((size_t)g * NPLAYER + p) * NPAD + h] = v;
	}
}

// Three-way showdowns. Grid (terminal, slot, class of that slot); one thread per class y of the third player.
__global__ void k_term3(Ctx6 c, const int* ids, int only) {
	__shared__ double rs[2][NPAD], red[NPAD];
	const int g = ids[blockIdx.x], s = blockIdx.y, h = blockIdx.z, y = threadIdx.x;
	const DNode& nd = c.nodes[g];
	if (only >= 0 && nd.sl[s] != only) return;
	const int xs = s == 0 ? 1 : 0, ys = s == 2 ? 1 : 2;  // the other two slots, in order
	rs[0][y] = c.reach[((size_t)g * NPLAYER + nd.sl[xs]) * NPAD + y];
	rs[1][y] = c.reach[((size_t)g * NPLAYER + nd.sl[ys]) * NPAD + y];
	__syncthreads();
	double acc = 0;
	if (y < c.n) {
		const double pot = nd.pot, cs = nd.contrib[nd.sl[s]];
		for (int x = 0; x < c.n; x++) {
			const double rx = rs[0][x];
			if (rx == 0) continue;
			const float4 e = c.T[s][((size_t)h * c.n + x) * c.n + y];
			if (e.w == 0) continue;
			int cls[3];
			cls[s] = h;
			cls[xs] = x;
			cls[ys] = y;
			const double E[3] = {e.x, e.y, e.z};
			double d = 0, mine = 0;
			for (int q = 0; q < 3; q++) {
				const double gq = c.grole[(size_t)nd.role[q] * NPAD + cls[q]];
				d += E[q] * gq;
				if (q == s) mine = E[q] * gq;
			}
			const double share = d > 0 ? mine / d : 0;
			acc += rx * rs[1][y] * (double)e.w * (pot * share - cs);
		}
	}
	red[y] = acc;
	__syncthreads();
	if (y < 64) red[y] += red[y + 128];
	__syncthreads();
	for (int o = 64; o > 0; o >>= 1) {
		if (y < o) red[y] += red[y + o];
		__syncthreads();
	}
	if (y == 0) {
		double* v = &c.val[((size_t)g * NPLAYER + nd.sl[s]) * NPAD + h];
		*v = *v * red[0] / c.NS[h];
	}
}

// mode 0: training (values, regret and strategy-sum updates); 1: value of the profile in the sigma slots;
// 2: best-response value of player `hero` against the strategies in the sigma slots.
__global__ void k_bottom(Ctx6 c, const int* ids, int mode, int hero) {
	const int g = ids[blockIdx.x], h = threadIdx.x;
	if (h >= c.n) return;
	const DNode& nd = c.nodes[g];
	const int k = nd.nact, p = nd.player;
	double* val = c.val;
	if (mode == 2) {
		double out;
		if (p == hero && nd.frozen) {  // a frozen decision plays its strategy even in the best response
			out = 0;
			for (int a = 0; a < k; a++)
				out += c.sigma[(size_t)(nd.aoff + a) * NPAD + h] * val[((size_t)nd.child[a] * NPLAYER + hero) * NPAD + h];
		} else if (p == hero) {
			out = val[((size_t)nd.child[0] * NPLAYER + hero) * NPAD + h];
			for (int a = 1; a < k; a++) out = fmax(out, val[((size_t)nd.child[a] * NPLAYER + hero) * NPAD + h]);
		} else {
			out = 0;
			for (int a = 0; a < k; a++) out += val[((size_t)nd.child[a] * NPLAYER + hero) * NPAD + h];
		}
		val[((size_t)g * NPLAYER + hero) * NPAD + h] = out;
		return;
	}
	double vp = 0, vo[NPLAYER];
	for (int q = 0; q < NPLAYER; q++) vo[q] = 0;
	for (int a = 0; a < k; a++) {
		const size_t ch = (size_t)nd.child[a] * NPLAYER;
		vp += c.sigma[(size_t)(nd.aoff + a) * NPAD + h] * val[(ch + p) * NPAD + h];
		for (int q = 0; q < NPLAYER; q++)
			if (q != p) vo[q] += val[(ch + q) * NPAD + h];
	}
	for (int q = 0; q < NPLAYER; q++) val[((size_t)g * NPLAYER + q) * NPAD + h] = q == p ? vp : vo[q];
	if (mode == 1 || nd.frozen) return;
	const double t = *c.t, reach = c.reach[((size_t)g * NPLAYER + p) * NPAD + h];
	for (int a = 0; a < k; a++) {
		const size_t s = (size_t)(nd.aoff + a) * NPAD + h;
		const double av = val[((size_t)nd.child[a] * NPLAYER + p) * NPAD + h];
		c.regret[s] = fmax(0.0, c.regret[s] + av - vp);
		c.sum[s] += t * reach * c.sigma[s];  // linear averaging
	}
}

// sigma slots := the averaged strategy (the current one where a class never gets there).
__global__ void k_average(Ctx6 c, const int* ids) {
	const int g = ids[blockIdx.x], h = threadIdx.x;
	if (h >= c.n) return;
	const DNode& nd = c.nodes[g];
	if (nd.frozen) return;
	double s = 0, pos = 0;
	for (int a = 0; a < nd.nact; a++) {
		s += c.sum[(size_t)(nd.aoff + a) * NPAD + h];
		pos += c.regret[(size_t)(nd.aoff + a) * NPAD + h];
	}
	for (int a = 0; a < nd.nact; a++) {
		const size_t i = (size_t)(nd.aoff + a) * NPAD + h;
		c.sigma[i] = s > 0 ? c.sum[i] / s : (pos > 0 ? c.regret[i] / pos : 1.0 / nd.nact);
	}
}

__global__ void k_tick(double* t) { *t += 1; }

// -- the solver (host) -------------------------------------------------------------------------------
struct Solver6 {
	Problem6 pb;
	std::vector<Node6> nodes;
	std::vector<DNode> hn;
	std::vector<int> dec_ids, term_ids, term3_ids;
	std::vector<std::vector<int> > levels;
	std::vector<int> lv_off;
	int G = 0, S = 0;
	long long done = 0;
	Ctx6 ctx;
	int *d_lv = nullptr, *d_term = nullptr, *d_term3 = nullptr, *d_dec = nullptr;
	double *d_reach = nullptr, *d_val = nullptr, *d_sigma = nullptr, *d_regret = nullptr, *d_sum = nullptr, *d_t = nullptr;
	DNode* d_nodes = nullptr;
	double *d_Wn = nullptr, *d_A[2][NCAT] = {}, *d_NS = nullptr, *d_grole = nullptr;
	float4* d_T[3] = {};
	cudaGraphExec_t exec = nullptr;
	cudaStream_t stream = nullptr;

	Solver6(const Problem6& problem, const std::vector<Node6>& tree, const std::vector<bool>& frozen) : pb(problem), nodes(tree) {
		if (pb.n > NPAD) { fprintf(stderr, "too many hands\n"); exit(1); }
		G = (int)nodes.size();
		std::vector<int> dep(G, 0);
		for (int i = 0; i < G; i++)
			for (int c : nodes[i].child) dep[c] = dep[i] + 1;
		int aoff = 0;
		for (int i = 0; i < G; i++) {
			const Node6& s = nodes[i];
			DNode d = {};
			d.type = s.is_term();
			d.player = s.player;
			double pot = 0;
			for (int p = 0; p < NPLAYER; p++) {
				d.contrib[p] = s.contrib[p];
				pot += s.contrib[p];
			}
			d.pot = pot;
			if (!s.is_term()) {
				if ((int)s.child.size() > MAXACT) { fprintf(stderr, "too many actions\n"); exit(1); }
				d.nact = (int)s.child.size();
				d.aoff = aoff;
				aoff += d.nact;
				d.frozen = i < (int)frozen.size() && frozen[i];
				for (int a = 0; a < d.nact; a++) d.child[a] = s.child[a];
				dec_ids.push_back(i);
				if (levels.size() <= (size_t)dep[i]) levels.resize(dep[i] + 1);
				levels[dep[i]].push_back(i);
			} else {
				d.live = s.live;
				d.cat = s.cat;
				d.ip = s.ip;
				d.oop = s.oop;
				term_ids.push_back(i);
				if (__builtin_popcount(s.live) == 3) {
					int k = 0;
					for (int p = 0; p < NPLAYER; p++)
						if (s.live >> p & 1) d.sl[k++] = p;
					for (int a = 0; a < 3; a++) {
						int rank = 0;
						for (int b = 0; b < 3; b++)
							if (POSTPOS[d.sl[b]] < POSTPOS[d.sl[a]]) rank++;
						d.role[a] = rank;
					}
					term3_ids.push_back(i);
				}
			}
			hn.push_back(d);
		}
		S = std::max(aoff, 1);
		upload();
	}

	~Solver6() {
		if (exec) cudaGraphExecDestroy(exec);
		if (stream) cudaStreamDestroy(stream);
		cudaFree(d_lv); cudaFree(d_term); cudaFree(d_term3); cudaFree(d_dec); cudaFree(d_reach); cudaFree(d_val);
		cudaFree(d_sigma); cudaFree(d_regret); cudaFree(d_sum); cudaFree(d_t); cudaFree(d_nodes); cudaFree(d_Wn);
		cudaFree(d_NS); cudaFree(d_grole);
		for (int s = 0; s < 2; s++)
			for (int c = 0; c < NCAT; c++) cudaFree(d_A[s][c]);
		for (int s = 0; s < 3; s++) cudaFree(d_T[s]);
	}
	Solver6(const Solver6&) = delete;

	template <class T> static T* put(const std::vector<T>& v) {
		T* d;
		CK(cudaMalloc(&d, std::max<size_t>(v.size(), 1) * sizeof(T)));
		CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
		return d;
	}
	static double* zeros(size_t count) {
		double* d;
		CK(cudaMalloc(&d, count * sizeof(double)));
		CK(cudaMemset(d, 0, count * sizeof(double)));
		return d;
	}
	// n x n row-major -> padded transposed [j * NPAD + h]
	std::vector<double> transposed(const std::vector<double>& m) const {
		std::vector<double> t((size_t)NPAD * NPAD, 0.0);
		const int n = pb.n;
		for (int h = 0; h < n; h++)
			for (int j = 0; j < n; j++) t[(size_t)j * NPAD + h] = m[(size_t)h * n + j];
		return t;
	}

	void upload() {
		const int n = pb.n;
		d_nodes = put(hn);
		std::vector<int> flat;
		for (auto& l : levels) {
			lv_off.push_back((int)flat.size());
			flat.insert(flat.end(), l.begin(), l.end());
		}
		lv_off.push_back((int)flat.size());
		d_lv = put(flat);
		d_term = put(term_ids);
		d_term3 = put(term3_ids);
		d_dec = put(dec_ids);
		d_Wn = put(transposed(pb.Wn));
		for (int c = 0; c < NCAT; c++) {
			d_A[0][c] = put(transposed(pb.A_ip[c].empty() ? std::vector<double>((size_t)n * n, 0.0) : pb.A_ip[c]));
			d_A[1][c] = put(transposed(pb.A_oop[c].empty() ? std::vector<double>((size_t)n * n, 0.0) : pb.A_oop[c]));
		}
		if (!term3_ids.empty()) {
			for (int s = 0; s < 3; s++) d_T[s] = put(pb.T[s]);
		}
		std::vector<double> ns(NPAD, 1.0), gr((size_t)3 * NPAD, 1.0);
		for (int h = 0; h < n; h++) {
			if (!pb.NS.empty()) ns[h] = pb.NS[h];
			for (int r = 0; r < 3; r++)
				if (!pb.grole[r].empty()) gr[(size_t)r * NPAD + h] = pb.grole[r][h];
		}
		d_NS = put(ns);
		d_grole = put(gr);
		d_reach = zeros((size_t)G * NPLAYER * NPAD);
		d_val = zeros((size_t)G * NPLAYER * NPAD);
		d_sigma = zeros((size_t)S * NPAD);
		d_regret = zeros((size_t)S * NPAD);
		d_sum = zeros((size_t)S * NPAD);
		d_t = zeros(1);
		set_t(1);
		std::vector<double> ones((size_t)G * NPLAYER * NPAD, 0.0);  // root reach: 1 for every real class
		for (int p = 0; p < NPLAYER; p++)
			for (int h = 0; h < n; h++) ones[((size_t)0 * NPLAYER + p) * NPAD + h] = 1;
		CK(cudaMemcpy(d_reach, ones.data(), ones.size() * sizeof(double), cudaMemcpyHostToDevice));
		ctx.nodes = d_nodes;
		ctx.reach = d_reach; ctx.val = d_val; ctx.sigma = d_sigma; ctx.regret = d_regret; ctx.sum = d_sum;
		ctx.t = d_t;
		ctx.Wn = d_Wn;
		for (int c = 0; c < NCAT; c++) { ctx.A_ip[c] = d_A[0][c]; ctx.A_oop[c] = d_A[1][c]; }
		for (int s = 0; s < 3; s++) ctx.T[s] = d_T[s];
		ctx.NS = d_NS;
		ctx.grole = d_grole;
		ctx.n = n;
		CK(cudaStreamCreate(&stream));
	}

	void set_t(double t) { CK(cudaMemcpy(d_t, &t, sizeof t, cudaMemcpyHostToDevice)); }
	int nlevels() const { return (int)levels.size(); }
	int lv_count(int l) const { return lv_off[l + 1] - lv_off[l]; }

	// Start (again) from a strategy: the sigma slots take it, and the regrets become K times it, so that
	// regret matching plays it first. Strategy sums and the iteration count restart.
	void seed(const std::vector<double>& sigma, double K) {
		std::vector<double> reg((size_t)S * NPAD, 0.0), zero((size_t)S * NPAD, 0.0);
		for (int i = 0; i < G; i++) {
			const DNode& d = hn[i];
			if (d.type) continue;
			for (int a = 0; a < d.nact; a++)
				for (int h = 0; h < pb.n; h++) {
					const size_t s = (size_t)(d.aoff + a) * NPAD + h;
					reg[s] = d.frozen ? 0.0 : K * sigma[s];
				}
		}
		CK(cudaMemcpy(d_sigma, sigma.data(), sigma.size() * sizeof(double), cudaMemcpyHostToDevice));
		CK(cudaMemcpy(d_regret, reg.data(), reg.size() * sizeof(double), cudaMemcpyHostToDevice));
		CK(cudaMemcpy(d_sum, zero.data(), zero.size() * sizeof(double), cudaMemcpyHostToDevice));
		done = 0;
		set_t(1);
		if (exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
	}

	// Freeze or unfreeze decisions (frozen ones keep the strategy in their sigma slots and are not trained).
	void set_frozen(const std::vector<bool>& frozen) {
		for (int i = 0; i < G; i++)
			if (!hn[i].type) hn[i].frozen = i < (int)frozen.size() && frozen[i];
		CK(cudaMemcpy(d_nodes, hn.data(), hn.size() * sizeof(DNode), cudaMemcpyHostToDevice));
		if (exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
	}

	void sweep_down(cudaStream_t s, int regmatch, int only) {
		for (int l = 0; l < nlevels(); l++)
			if (lv_count(l)) k_top<<<lv_count(l), NPAD, 0, s>>>(ctx, d_lv + lv_off[l], regmatch);
		if (!term_ids.empty()) k_term<<<(int)term_ids.size(), NPAD, 0, s>>>(ctx, d_term, only);
		if (!term3_ids.empty()) k_term3<<<dim3((unsigned)term3_ids.size(), 3, (unsigned)pb.n), NPAD, 0, s>>>(ctx, d_term3, only);
	}
	void sweep_up(cudaStream_t s, int mode, int hero) {
		for (int l = nlevels() - 1; l >= 0; l--)
			if (lv_count(l)) k_bottom<<<lv_count(l), NPAD, 0, s>>>(ctx, d_lv + lv_off[l], mode, hero);
	}
	void enqueue_iteration(cudaStream_t s) {
		sweep_down(s, 1, -1);
		sweep_up(s, 0, 0);
		k_tick<<<1, 1, 0, s>>>(d_t);
	}

	void ensure_graph() {
		if (exec) return;
		cudaGraph_t graph;
		CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
		enqueue_iteration(stream);
		CK(cudaStreamEndCapture(stream, &graph));
		CK(cudaGraphInstantiate(&exec, graph, 0));
		CK(cudaGraphDestroy(graph));
	}

	// `count` more iterations in one go (tests, short runs; no relay, no thermal gate).
	void iterate(long long count) {
		ensure_graph();
		for (long long i = 0; i < count; i++) CK(cudaGraphLaunch(exec, stream));
		CK(cudaStreamSynchronize(stream));
		CK(cudaGetLastError());
		done += count;
	}

	// A hash of everything the iterations depend on: the tree, the frozen flags, the tables, and the state the run
	// starts from (regrets, sums, sigma slots), so that a relay file of another start is not used.
	uint64_t problem_tag() {
		uint64_t h = relay_hash_u64(14695981039346656037ull, 0x36584d4158ull);
		auto u = [&](uint64_t v) { h = relay_hash_u64(h, v); };
		auto vec = [&](const std::vector<double>& v) { u(v.size()); h = relay_hash(h, v.data(), v.size() * sizeof(double)); };
		u(G); u(S); u(pb.n);
		for (const DNode& d : hn) {
			u(d.type); u(d.player); u(d.nact); u(d.aoff); u(d.frozen);
			for (int a = 0; a < MAXACT; a++) u((uint64_t)(int64_t)d.child[a]);
			u(d.live); u(d.cat); u(d.ip); u(d.oop);
			for (int k = 0; k < 3; k++) { u(d.sl[k]); u(d.role[k]); }
			for (int p = 0; p < NPLAYER; p++) h = relay_hash(h, &d.contrib[p], sizeof(double));
			h = relay_hash(h, &d.pot, sizeof(double));
		}
		vec(pb.prior);
		vec(pb.Wn);
		for (int c = 0; c < NCAT; c++) { vec(pb.A_ip[c]); vec(pb.A_oop[c]); }
		for (int s = 0; s < 3; s++) {
			u(pb.T[s].size());
			h = relay_hash(h, pb.T[s].data(), pb.T[s].size() * sizeof(float4));
			vec(pb.grole[s]);
		}
		vec(pb.NS);
		const size_t cnt = (size_t)S * NPAD;
		vec(download(d_regret, cnt));
		vec(download(d_sum, cnt));
		vec(download(d_sigma, cnt));
		return h;
	}

	// The relay: run iterations 0 .. total in batches of `batch` iterations (a second or two each), writing the state
	// (regrets and strategy sums; the iteration count is the relay's `done`; `save_extra` / `load_extra` add the
	// caller's own, e.g. the convergence points) to `path` after EVERY batch. Started again with the same problem and
	// total it restores that state and goes on; the result is identical to an uninterrupted run, bit for bit.
	// `after(done)` is called after each batch (reports); it may compute gaps but must not touch regrets or sums.
	// A file in the old format is converted when its iteration count is a multiple of `batch`. path empty = no relay.
	// `stop_after` is for tests: stop cleanly after that many batches of this call.
	void run(long long total, const std::string& path, long long batch, const std::function<void(long long)>& after = nullptr,
	         const std::function<void(FILE*)>& save_extra = nullptr, const std::function<bool(FILE*)>& load_extra = nullptr,
	         uint64_t stop_after = 0) {
		ensure_graph();
		if (batch < 1) batch = 50;
		const size_t cnt = (size_t)S * NPAD;
		Relay relay;
		relay.path = path;
		relay.tag = path.empty() ? 0 : problem_tag();
		relay.stop_after = stop_after;
		relay.save_state = [&](FILE* f) {
			const std::vector<double> r = download(d_regret, cnt), s = download(d_sum, cnt);
			fwrite(r.data(), sizeof(double), cnt, f);
			fwrite(s.data(), sizeof(double), cnt, f);
			if (save_extra) save_extra(f);
		};
		relay.load_state = [&](FILE* f) {
			std::vector<double> r(cnt), s(cnt);
			if (fread(r.data(), sizeof(double), cnt, f) != cnt || fread(s.data(), sizeof(double), cnt, f) != cnt) return false;
			if (load_extra && !load_extra(f)) return false;
			CK(cudaMemcpy(d_regret, r.data(), cnt * sizeof(double), cudaMemcpyHostToDevice));
			CK(cudaMemcpy(d_sum, s.data(), cnt * sizeof(double), cudaMemcpyHostToDevice));
			return true;
		};
		if (!path.empty()) {  // an old-format checkpoint (stage 2 of earlier versions)
			const std::vector<double> r0 = download(d_regret, cnt), s0 = download(d_sum, cnt);
			const long long done0 = done;
			if (done == 0 && load_legacy(path.c_str())) {
				if (done <= total && done % batch == 0) {
					fprintf(stderr, "[relay] %s: converting an old checkpoint at iteration %lld\n", path.c_str(), done);
					// the tag was made from the start state, before load_legacy replaced it; the caller's extra state
					// (convergence points) is whatever it holds now: the old format has none
					relay_save(relay, (uint64_t)total, (uint64_t)batch, (uint64_t)done);
				} else {
					fprintf(stderr, "[relay] %s: old checkpoint at %lld does not fit a batch of %lld, starting from scratch\n", path.c_str(), done, batch);
				}
				CK(cudaMemcpy(d_regret, r0.data(), cnt * sizeof(double), cudaMemcpyHostToDevice));  // back to the start; dispatch restores
				CK(cudaMemcpy(d_sum, s0.data(), cnt * sizeof(double), cudaMemcpyHostToDevice));
				done = done0;
				set_t(1);
			}
		}
		dispatch((uint64_t)total, (uint64_t)batch, [&](uint64_t first, uint64_t n) {
			set_t((double)first + 1);
			for (uint64_t i = 0; i < n; i++) CK(cudaGraphLaunch(exec, stream));
			CK(cudaStreamSynchronize(stream));
			done = (long long)(first + n);
			if (after) after(done);
		}, "iter", false, path.empty() ? nullptr : &relay);
		done = (stop_after && !path.empty()) ? (long long)relay.done : total;
		set_t((double)done + 1);
	}

	void average() {
		if (!dec_ids.empty()) k_average<<<(int)dec_ids.size(), NPAD, 0, stream>>>(ctx, d_dec);
		CK(cudaStreamSynchronize(stream));
	}

	double root_gain(int p) {  // expected value of player p at the root, in bb
		std::vector<double> v(NPAD);
		CK(cudaMemcpy(v.data(), d_val + ((size_t)0 * NPLAYER + p) * NPAD, NPAD * sizeof(double), cudaMemcpyDeviceToHost));
		double s = 0;
		for (int h = 0; h < pb.n; h++) s += pb.prior[h] * v[h];
		return s;
	}
	std::vector<double> root_vec(int p) {
		std::vector<double> v(NPAD);
		CK(cudaMemcpy(v.data(), d_val + ((size_t)0 * NPLAYER + p) * NPAD, NPAD * sizeof(double), cudaMemcpyDeviceToHost));
		return v;
	}

	struct Gap {
		double value[NPLAYER], best[NPLAYER];  // bb per hand: the averaged profile's value, a best response's value
		double gap(int p) const { return best[p] - value[p]; }
		double total() const { double s = 0; for (int p = 0; p < NPLAYER; p++) s += gap(p); return s; }
		double sum_value() const { double s = 0; for (int p = 0; p < NPLAYER; p++) s += value[p]; return s; }
	};

	// Each player's value under the averaged strategies, and what he would get by best-responding to the
	// others' averaged strategies (all his decisions at once).
	Gap exploit() {
		average();
		Gap g;
		sweep_down(stream, 0, -1);
		sweep_up(stream, 1, 0);
		CK(cudaStreamSynchronize(stream));
		for (int p = 0; p < NPLAYER; p++) g.value[p] = root_gain(p);
		for (int p = 0; p < NPLAYER; p++) {
			sweep_down(stream, 0, p);
			sweep_up(stream, 2, p);
			CK(cudaStreamSynchronize(stream));
			g.best[p] = root_gain(p);
		}
		return g;
	}

	std::vector<double> download(const double* d, size_t count) {
		std::vector<double> v(count);
		CK(cudaMemcpy(v.data(), d, count * sizeof(double), cudaMemcpyDeviceToHost));
		return v;
	}
	std::vector<double> sigma_slots() { return download(d_sigma, (size_t)S * NPAD); }
	std::vector<double> averaged() {
		average();
		return sigma_slots();
	}
	// Reach of every player at every node under the strategies in the sigma slots: [(node * 6 + p) * NPAD + h].
	std::vector<double> reaches() {
		sweep_down(stream, 0, 0);  // the terminal values are not needed, only the reaches
		CK(cudaStreamSynchronize(stream));
		return download(d_reach, (size_t)G * NPLAYER * NPAD);
	}
	void set_sigma(const std::vector<double>& s) {
		CK(cudaMemcpy(d_sigma, s.data(), s.size() * sizeof(double), cudaMemcpyHostToDevice));
	}

	// -- the old checkpoint format (PREFIX.ckpt before the relay): read only, run() converts it
	bool load_legacy(const char* path) {
		FILE* f = fopen(path, "rb");
		if (!f) return false;
		long long hdr[4];
		std::vector<double> r((size_t)S * NPAD), s((size_t)S * NPAD);
		bool ok = fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == 0x36584d41583031ll && hdr[1] == G && hdr[2] == S &&
		          fread(r.data(), sizeof(double), r.size(), f) == r.size() && fread(s.data(), sizeof(double), s.size(), f) == s.size();
		fclose(f);
		if (!ok) return false;
		CK(cudaMemcpy(d_regret, r.data(), r.size() * sizeof(double), cudaMemcpyHostToDevice));
		CK(cudaMemcpy(d_sum, s.data(), s.size() * sizeof(double), cudaMemcpyHostToDevice));
		done = hdr[3];
		set_t((double)done + 1);
		return true;
	}
};

#include "sixmax_data.cuh"

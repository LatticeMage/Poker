// The flop as a solved game: for a heads-up matchup (two preflop ranges, who is in position, pot and stacks) and a
// flop, the equilibrium of a small betting tree, hand by hand. This is what stands at a leaf of the decision tree
// (flop group x matchup) instead of a hand-made betting rule.
//
// The game (pot = 1, stacks = spr pots), FvGame:
//   flop    the player out of position (O) checks, bets 33% or bets 66%; after a check the player in position (I)
//           checks, bets 33% or bets 66%; facing a bet a player folds, calls or raises (the raise adds half of the
//           pot as it is after the call); facing a raise a player folds or calls;
//   later   with `streets` = 2 or 3 the betting goes on for one or two more rounds: check or bet 66%, fold or call;
//   bets and raises are capped at the stack (then there is no raise over them, and no further round).
// After the last round the hands are shown: a showdown pays the exact all-in equity of the two hands (every turn
// and river). No card comes between the rounds: the later rounds are betting on the same information (a "static"
// model). They are there because with one round every strong hand has to get its money in on the flop, which makes
// the flop bets too big and the raises too many; with more rounds a hand can wait.
// The model knows range advantage, nut advantage, how much a bet folds out, what calling costs and that the
// betting goes on; it does not know which cards come (draws are just hands with some equity).
//   turn    (fv_game(spr, 2, true)) the real second round: after the flop betting one of the 49 turn cards comes,
//           then one round of check or bet 66%, fold or call, and the hands are shown at their equity on that turn
//           (every river). Here hands change value between the rounds, so a check gives a free card and a bet
//           denies it: what makes a flop bet worth making. Costs 49 matrices per flop instead of one.
//           fv_game(spr, 3, true) adds a third round of betting on the same turn card (the river's betting without
//           its card), so a strong hand has two more rounds to get its money in.
//
// Method: CFR+ with alternating updates and linear averaging on the exact hand-against-hand matrix.
//   matrix   T[i][j] = 2 * wins + ties of O's combo i against I's combo j over the 990 runouts both are live in
//            (FV_NONE if the two hands share a card or one of them uses a flop card), built on the GPU;
//   products each iteration needs, for every showdown line of the tree, the sum over the opponent's combos of
//            (their reach) x (equity): two kernel launches per iteration for a whole batch of flops;
//   the rest (regrets, strategies, fold lines, which only need card removal) is host work.
// Checks (test_flopsolve.cu): toy games with a known solution, brute-force enumeration, the host reference.
//
// Result: FV_K floats per (matchup, flop): ev of O (share of the pot), exploitability (pots), all-in equity of O's
// range, iterations, then for each of the ten flop decisions, each of ten equity bins of the hand and each action
// the share of the range that is in that bin, reaches the decision by its own actions and takes the action.
#pragma once

#include "flopstats.cuh"

#include <thread>

static const int FV_MAXL = 96;         // most showdown lines a tree may have (columns of one product)
static const uint16_t FV_NONE = 0xFFFF;
static const int FV_FULL = 1980;       // 2 x C(45,2): twice the runouts two live hands share
static const int FV_TURNS = 49;        // turn cards
static const uint8_t FV_TNONE = 0xFF;
static const int FV_TFULL = 88;        // 2 x 44: twice the rivers two live hands share on a turn
static const int FV_TLIVE = 45;        // turn cards two live hands can see
static const double FV_BET[2] = {0.33, 0.66};
static const double FV_LATER_BET = 0.66;
static const double FV_RAISE = 0.5;    // a raise adds this share of the pot after the call
static const int FV_SLOTS = 13;        // action slots of one player on the flop, in the record's order
static const int FV_OFF[5] = {0, 3, 6, 9, 11};
static const int FV_NACT[5] = {3, 3, 3, 2, 2};
static const int FV_NODES = 10;        // flop decisions: five of O, then five of I
static const int FV_HEAD = 4;
static const int FV_K = FV_HEAD + FV_NODES * FS_BINS * 3;
static const int FV_THREADS = 4;       // host threads over the flops of a batch
// Flop decisions of a player (nd), in this order:
//   0  the first decision (O at the start, I after O's check): check, bet 33%, bet 66%
//   1  facing a 33% bet: fold, call, raise        2  facing a 66% bet: fold, call, raise
//   3  bet 33% and was raised: fold, call         4  bet 66% and was raised: fold, call
static const char* FV_NODE_NAME[5] = {"first", "vs 33%", "vs 66%", "33% raised", "66% raised"};

// -- device -------------------------------------------------------------------------------------------------------
// T[(f * NA + i) * NC + j], from the rank tables of the two sides (fs_rank_kernel).
__global__ void fv_matrix_kernel(const uint32_t* rank_a, const uint32_t* rank_b, const uint8_t* ca, const uint8_t* cb, int NA, int NC, int F,
                                 uint16_t* T) {
	const uint64_t id = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	if (id >= (uint64_t)F * NA * NC) return;
	const int j = (int)(id % NC), i = (int)((id / NC) % NA), f = (int)(id / ((uint64_t)NA * NC));
	const int a0 = ca[i * 2], a1 = ca[i * 2 + 1], b0 = cb[j * 2], b1 = cb[j * 2 + 1];
	uint16_t out = FV_NONE;
	if (a0 != b0 && a0 != b1 && a1 != b0 && a1 != b1) {
		const uint32_t* ra = rank_a + ((uint64_t)f * NA + i) * FS_R;
		const uint32_t* rb = rank_b + ((uint64_t)f * NC + j) * FS_R;
		int twice = 0, n = 0;
		for (int r = 0; r < FS_R; r++) {
			const uint32_t a = ra[r], b = rb[r];
			const int live = (a != 0) & (b != 0);
			n += live;
			twice += live * (2 * (a > b) + (a == b));
		}
		if (n) out = (uint16_t)twice;  // n is 990 whenever it is not 0
	}
	T[id] = out;
}

// Tt[(f * NC + j) * NA + i] = T[(f * NA + i) * NC + j]
__global__ void fv_transpose_kernel(const uint16_t* T, int NA, int NC, int F, uint16_t* Tt) {
	const uint64_t id = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	if (id >= (uint64_t)F * NA * NC) return;
	const int i = (int)(id % NA), j = (int)((id / NA) % NC), f = (int)(id / ((uint64_t)NA * NC));
	Tt[id] = T[((uint64_t)f * NA + i) * NC + j];
}

// out[(f * N + i) * L + c] = sum over k of m(i, k) * v[(f * M + k) * L + c], where m = M2[(f * N + i) * M + k],
// or FV_FULL - that when `complement` (the other player's side of the same matrix); FV_NONE entries are skipped.
__global__ void fv_mul_kernel(const uint16_t* M2, const float* v, int N, int M, int F, int L, int complement, float* out) {
	const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= (uint32_t)(F * N)) return;
	const int f = id / N;
	const uint16_t* row = M2 + (uint64_t)id * M;
	const float* vf = v + (uint64_t)f * M * L;
	float acc[FV_MAXL];
	for (int c = 0; c < L; c++) acc[c] = 0;
	for (int k = 0; k < M; k++) {
		const uint16_t t = row[k];
		if (t == FV_NONE) continue;
		const float m = complement ? (float)(FV_FULL - t) : (float)t;
		const float* x = vf + (uint64_t)k * L;
		for (int c = 0; c < L; c++) acc[c] += m * x[c];
	}
	float* o = out + (uint64_t)id * L;
	for (int c = 0; c < L; c++) o[c] = acc[c];
}

// Tk[((f * 49 + k) * NA + i) * NC + j] = 2 * wins + ties of i against j over the rivers, when the k-th card that is
// not on the flop is the turn; FV_TNONE if the hands share a card or one of them uses a flop card or the turn card.
__global__ void fv_turn_matrix_kernel(const uint32_t* rank_a, const uint32_t* rank_b, const uint8_t* ca, const uint8_t* cb, int NA, int NC, int F,
                                      uint8_t* Tk) {
	const uint64_t id = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	if (id >= (uint64_t)F * FV_TURNS * NA * NC) return;
	const int j = (int)(id % NC), i = (int)((id / NC) % NA), k = (int)((id / ((uint64_t)NA * NC)) % FV_TURNS);
	const int f = (int)(id / ((uint64_t)NA * NC * FV_TURNS));
	const int a0 = ca[i * 2], a1 = ca[i * 2 + 1], b0 = cb[j * 2], b1 = cb[j * 2 + 1];
	uint8_t out = FV_TNONE;
	if (a0 != b0 && a0 != b1 && a1 != b0 && a1 != b1) {
		const uint32_t* ra = rank_a + ((uint64_t)f * NA + i) * FS_R;
		const uint32_t* rb = rank_b + ((uint64_t)f * NC + j) * FS_R;
		int twice = 0, n = 0;
		for (int u = 0; u < FV_TURNS; u++) {
			if (u == k) continue;
			const int lo = u < k ? u : k, hi = u < k ? k : u;
			const int r = 48 * lo - lo * (lo - 1) / 2 + (hi - lo - 1);  // the pair's place among the 1,176 runouts
			const uint32_t a = ra[r], b = rb[r];
			const int live = (a != 0) & (b != 0);
			n += live;
			twice += live * (2 * (a > b) + (a == b));
		}
		if (n) out = (uint8_t)twice;  // n is 44 whenever it is not 0
	}
	Tk[id] = out;
}
__global__ void fv_turn_transpose_kernel(const uint8_t* Tk, int NA, int NC, int FK, uint8_t* Tt) {
	const uint64_t id = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	if (id >= (uint64_t)FK * NA * NC) return;
	const int i = (int)(id % NA), j = (int)((id / NA) % NC);
	const uint64_t fk = id / ((uint64_t)NA * NC);
	Tt[id] = Tk[(fk * NA + i) * NC + j];
}
// As fv_mul_kernel, for the turn matrices: one block of L columns per (flop, turn card).
__global__ void fv_mul_turn_kernel(const uint8_t* M2, const float* v, int N, int M, int FK, int L, int complement, float* out) {
	const uint64_t id = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	if (id >= (uint64_t)FK * N) return;
	const uint64_t fk = id / N;
	const uint8_t* row = M2 + id * M;
	const float* vf = v + fk * M * L;
	float acc[FV_MAXL];
	for (int c = 0; c < L; c++) acc[c] = 0;
	for (int k = 0; k < M; k++) {
		const uint8_t t = row[k];
		if (t == FV_TNONE) continue;
		const float m = complement ? (float)(FV_TFULL - t) : (float)t;
		const float* x = vf + (uint64_t)k * L;
		for (int c = 0; c < L; c++) acc[c] += m * x[c];
	}
	float* o = out + id * L;
	for (int c = 0; c < L; c++) o[c] = acc[c];
}

// -- the spot: who is in position, pot and stacks -----------------------------------------------------------------
// The betting tree. Decisions of a player own consecutive action slots; a showdown has a line (its column in the
// products), a fold has an index of its own.
struct FvNode {
	int type = 0;                  // 0 decision, 1 showdown, 2 fold, 3 the turn card (49 children, FvGame::kids)
	int player = 0;                // who acts, or who folded
	int nact = 0, child[3] = {-1, -1, -1};
	bool ok[3] = {true, true, true};
	int slot = -1;                 // decision: its first slot among the player's
	int nd = -1;                   // flop decision in the record's order (0..4); -1 on later rounds
	double c = 0;                  // showdown: what each player put in; fold: what the folder put in
	int line = -1;                 // showdown: its column (among the lines of its matrix); fold: its index
	int turn = -1;                 // the turn card (0..48) the node comes after; -1 before the turn
	int kids = -1;                 // turn card node: where its 49 children start in FvGame::kids
};
struct FvGame {
	std::vector<FvNode> nodes;
	std::vector<int> kids;
	int slots[2] = {0, 0}, lines = 0, folds = 0, streets = 1;
	int tlines = 0;                // showdown lines after each turn card (the same number for every card)
	bool turn = false;             // the second round is played on a real turn card
	int at_turn = -1, tline[FV_TURNS] = {0};
	double spr = 0;

	int add(FvNode n) { n.turn = at_turn; nodes.push_back(n); return (int)nodes.size() - 1; }
	int showdown(double c) {
		FvNode n;
		n.type = 1; n.c = c;
		n.line = at_turn < 0 ? lines++ : tline[at_turn]++;
		return add(n);
	}
	int fold(int who, double c) { FvNode n; n.type = 2; n.player = who; n.c = c; n.line = folds++; return add(n); }
	int decision(int player, int nact, int nd) {
		FvNode n;
		n.player = player; n.nact = nact; n.nd = nd;
		n.slot = slots[player];
		slots[player] += nact;
		return add(n);
	}
	// Player p faces a bet of b in round s (each has c in before it): fold, call, or raise where the round allows.
	int facing(int s, double c, double b, int p, int nd, int nd_raised) {
		const double pot = 1 + 2 * c, left = spr - c;
		const bool can = s == 0 && b < left - 1e-9;  // raises only on the flop
		const int n = decision(p, s == 0 ? 3 : 2, nd);
		const int f = fold(p, c);
		const int call = street(s + 1, c + b);
		nodes[n].child[0] = f;
		nodes[n].child[1] = call;
		if (s == 0) nodes[n].ok[2] = can;
		if (can) {
			const double r = std::min(left, b + FV_RAISE * (pot + 2 * b));
			const int m = decision(1 - p, 2, nd_raised);
			const int f2 = fold(1 - p, c + b);
			const int call2 = street(s + 1, c + r);
			nodes[m].child[0] = f2;
			nodes[m].child[1] = call2;
			nodes[n].child[2] = m;
		}
		return n;
	}
	// Round s with c in from each player: O's first decision, or the showdown when the betting is over.
	int street(int s, double c) {
		if (s == streets || c > spr - 1e-9) return showdown(c);
		if (s == 1 && turn && at_turn < 0) {  // the turn card comes: the same round after each of the 49 cards
			FvNode n;
			n.type = 3;
			n.kids = (int)kids.size();
			kids.resize(kids.size() + FV_TURNS);
			const int at = add(n);
			for (int k = 0; k < FV_TURNS; k++) {
				at_turn = k;
				const int r = street(1, c);
				kids[nodes[at].kids + k] = r;
			}
			at_turn = -1;
			return at;
		}
		const double pot = 1 + 2 * c, left = spr - c;
		const int K = s == 0 ? 2 : 1;
		const int o = decision(0, 1 + K, s == 0 ? 0 : -1);
		const int i = decision(1, 1 + K, s == 0 ? 0 : -1);
		nodes[o].child[0] = i;
		const int through = street(s + 1, c);
		nodes[i].child[0] = through;
		for (int k = 0; k < K; k++) {
			const double b = std::min(left, (s == 0 ? FV_BET[k] : FV_LATER_BET) * pot);
			const int vs_o = facing(s, c, b, 1, s == 0 ? 1 + k : -1, s == 0 ? 3 + k : -1);  // O bets, I answers
			nodes[o].child[1 + k] = vs_o;
			const int vs_i = facing(s, c, b, 0, s == 0 ? 1 + k : -1, s == 0 ? 3 + k : -1);  // I bets after the check
			nodes[i].child[1 + k] = vs_i;
		}
		return o;
	}
};
static inline FvGame fv_game(double spr, int streets = 1, bool turn = false) {
	FvGame g;
	g.spr = spr;
	g.streets = turn ? std::max(2, streets) : streets;
	g.turn = turn;
	g.street(0, 0.0);  // node 0 is O's first decision
	g.tlines = g.tline[0];
	if (g.lines > FV_MAXL || g.tlines > FV_MAXL) throw std::runtime_error("the tree has more showdown lines than FV_MAXL");
	return g;
}

// From a matchup key "kind:opener:other" (kind = srp, 3bet, 4bet). Sizes: 100bb stacks; the open is the chart's
// (UTG 3, MP 3, CO 2.5, BTN 2, SB 3 big blinds); a 3-bet is 3 opens in position and 4 out of position; a 4-bet is
// 2.25 times the 3-bet; the blinds of players who folded stay in the pot.
struct FvSpot {
	bool aggressor_ip = false;
	double pot = 0, stack = 0, spr = 0;
};
static inline FvSpot fv_spot(const std::string& key) {
	const size_t p1 = key.find(':'), p2 = key.find(':', p1 + 1);
	if (p1 == std::string::npos || p2 == std::string::npos) throw std::runtime_error("matchup key " + key);
	const std::string kind = key.substr(0, p1), opener = key.substr(p1 + 1, p2 - p1 - 1), other = key.substr(p2 + 1);
	auto order = [&](const std::string& p) {  // order of acting after the flop
		const char* names[6] = {"SB", "BB", "UTG", "MP", "CO", "BTN"};
		for (int i = 0; i < 6; i++)
			if (p == names[i]) return i;
		throw std::runtime_error("position " + p + " in " + key);
	};
	const int po = order(opener), pt = order(other);
	const double opens[6] = {3, 0, 3, 3, 2.5, 2};
	const double open = opens[po], three = open * (pt > po ? 3 : 4), four = three * 2.25;
	const double in = kind == "srp" ? open : kind == "3bet" ? three : four;
	const double dead = (opener != "SB" && other != "SB" ? 0.5 : 0) + (opener != "BB" && other != "BB" ? 1 : 0);
	FvSpot s;
	s.pot = 2 * in + dead;
	s.stack = 100 - in;
	s.spr = s.stack / s.pot;
	const int aggressor = kind == "3bet" ? pt : po, caller = kind == "3bet" ? po : pt;
	s.aggressor_ip = aggressor > caller;
	return s;
}

// -- one flop of one matchup, host side ---------------------------------------------------------------------------
// Side 0 = O (out of position), side 1 = I. Vectors over a side's combos; strategies as [slot * n + i].
struct FvFlop {
	const FvGame* game = nullptr;
	int n[2] = {0, 0}, L = 0, TL = 0;                // columns of the full matrix (at least 1) and of each turn matrix
	int tcard[FV_TURNS];                             // the turn cards, in the kernels' order
	std::vector<double> w[2];                        // range weight, 0 for a combo that uses a flop card
	const uint8_t* cards[2] = {nullptr, nullptr};    // two per combo; null = an abstract game (no card removal by cards)
	const uint16_t* dense = nullptr;                 // the matrix [i * n[1] + j]; used for the valid sums if no cards
	std::vector<double> reg[2], sum[2], cur[2];      // regrets, summed strategy, current strategy
	std::vector<double> vq[2], vf[2];                // for side s: the other side's reach it can meet, per showdown line and per fold
	std::vector<double> eq[2], tot[2];               // all-in equity of each combo and the opponent weight it can meet
	std::vector<int> same[2];                        // the identical combo in the other range, or -1
	double W = 0;                                    // sum over the valid pairs of the two weights

	void init(const FvGame& g, int n_o, int n_i, const std::vector<double>& w_o, const std::vector<double>& w_i, const uint8_t* c_o, const uint8_t* c_i,
	          const uint8_t* flop = nullptr) {
		game = &g;
		L = std::max(1, g.lines);
		TL = g.tlines;
		if (g.turn && !(flop && c_o && c_i)) throw std::runtime_error("a tree with a turn card needs real cards");
		if (flop) {
			int k = 0;
			for (int c = 0; c < 52; c++)
				if (c != flop[0] && c != flop[1] && c != flop[2]) tcard[k++] = c;
		}
		n[0] = n_o; n[1] = n_i;
		w[0] = w_o; w[1] = w_i;
		cards[0] = c_o; cards[1] = c_i;
		for (int s = 0; s < 2; s++) {
			reg[s].assign((size_t)g.slots[s] * n[s], 0.0);
			sum[s].assign((size_t)g.slots[s] * n[s], 0.0);
			cur[s].assign((size_t)g.slots[s] * n[s], 0.0);
			vq[s].assign(((size_t)L + (size_t)FV_TURNS * TL) * n[s], 0.0);
			vf[s].assign((size_t)g.folds * n[s], 0.0);
			same[s].assign(n[s], -1);
			match(s);
		}
		if (c_o && c_i) {
			std::vector<int> at(52 * 52, -1);
			for (int j = 0; j < n[1]; j++) at[c_i[j * 2] * 52 + c_i[j * 2 + 1]] = at[c_i[j * 2 + 1] * 52 + c_i[j * 2]] = j;
			for (int i = 0; i < n[0]; i++) {
				same[0][i] = at[c_o[i * 2] * 52 + c_o[i * 2 + 1]];
				if (same[0][i] >= 0) same[1][same[0][i]] = i;
			}
		}
	}

	// A strategy from non-negative numbers per action (regrets, or summed strategy): their shares, uniform if all are 0.
	void shares(int s, const std::vector<double>& from, std::vector<double>& to) const {
		const int N = n[s];
		for (const FvNode& nd : game->nodes) {
			if (nd.type != 0 || nd.player != s) continue;
			for (int i = 0; i < N; i++) {
				double pos = 0;
				int ok = 0;
				for (int a = 0; a < nd.nact; a++)
					if (nd.ok[a]) { pos += from[(size_t)(nd.slot + a) * N + i]; ok++; }
				for (int a = 0; a < nd.nact; a++)
					to[(size_t)(nd.slot + a) * N + i] = !nd.ok[a] ? 0 : pos > 0 ? from[(size_t)(nd.slot + a) * N + i] / pos : 1.0 / ok;
			}
		}
	}
	void match(int s) { shares(s, reg[s], cur[s]); }
	std::vector<double> average(int s) const {
		std::vector<double> av((size_t)game->slots[s] * n[s]);
		shares(s, sum[s], av);
		return av;
	}

	// out[i] = sum of v over the other side's combos that share no card with combo i of side s.
	void valid(int s, const double* v, double* out) const {
		const int o = 1 - s, N = n[s], M = n[o];
		if (cards[0]) {
			double S = 0, C[52] = {0};
			for (int j = 0; j < M; j++) {
				S += v[j];
				C[cards[o][j * 2]] += v[j];
				C[cards[o][j * 2 + 1]] += v[j];
			}
			for (int i = 0; i < N; i++) out[i] = S - C[cards[s][i * 2]] - C[cards[s][i * 2 + 1]] + (same[s][i] >= 0 ? v[same[s][i]] : 0);
		} else {
			for (int i = 0; i < N; i++) {
				double x = 0;
				for (int j = 0; j < M; j++)
					if ((s == 0 ? dense[(size_t)i * n[1] + j] : dense[(size_t)j * n[1] + i]) != FV_NONE) x += v[j];
				out[i] = x;
			}
		}
	}

	bool holds(int s, int i, int k) const { return cards[s][i * 2] == tcard[k] || cards[s][i * 2 + 1] == tcard[k]; }
	// The row of vq a showdown uses.
	size_t qrow(const FvNode& nd) const { return nd.turn < 0 ? (size_t)nd.line : (size_t)L + (size_t)nd.turn * TL + nd.line; }

	// Down the tree with side s's reach r under strategy st: at every showdown the reach goes to the kernel's input
	// (out[i * L + line], or after turn card k out_t[(k * N + i) * TL + line]) and what the other side can meet of
	// it to vq; at every fold to vf. A hand that holds the turn card does not reach that card's branch.
	void down(int at, int s, const std::vector<double>& st, const std::vector<double>& r, float* out, float* out_t) {
		const FvNode& nd = game->nodes[at];
		const int N = n[s], o = 1 - s;
		if (nd.type == 1) {
			if (nd.turn < 0)
				for (int i = 0; i < N; i++) out[(size_t)i * L + nd.line] = (float)r[i];
			else
				for (int i = 0; i < N; i++) out_t[((size_t)nd.turn * N + i) * TL + nd.line] = (float)r[i];
			valid(o, r.data(), &vq[o][qrow(nd) * n[o]]);
		} else if (nd.type == 2) {
			valid(o, r.data(), &vf[o][(size_t)nd.line * n[o]]);
		} else if (nd.type == 3) {
			std::vector<double> x(N);
			for (int k = 0; k < FV_TURNS; k++) {
				for (int i = 0; i < N; i++) x[i] = holds(s, i, k) ? 0 : r[i];
				down(game->kids[nd.kids + k], s, st, x, out, out_t);
			}
		} else if (nd.player != s) {
			for (int a = 0; a < nd.nact; a++)
				if (nd.ok[a]) down(nd.child[a], s, st, r, out, out_t);
		} else {
			std::vector<double> x(N);
			for (int a = 0; a < nd.nact; a++) {
				if (!nd.ok[a]) continue;
				const double* p = &st[(size_t)(nd.slot + a) * N];
				for (int i = 0; i < N; i++) x[i] = r[i] * p[i];
				down(nd.child[a], s, st, x, out, out_t);
			}
		}
	}
	// The reach of side s at every showdown line (current strategy, or the average) for the kernels.
	void fill(int s, bool avg, float* out, float* out_t = nullptr) {
		if (avg) down(0, s, average(s), w[s], out, out_t);
		else down(0, s, cur[s], w[s], out, out_t);
	}

	// Up the tree for side s: the value of every hand at a node against the other side's reach (fill(1 - s) before,
	// prod = its products with the matrix). A decision of s is worth its strategy st, or its best action when st is
	// null (a best response). With `own` (s's reach by its own actions) it also does the CFR+ update of iteration t.
	// After a turn card every payoff counts 1/45: the chance of that card for two live hands.
	std::vector<double> up(int at, int s, const float* prod, const float* prod_t, const double* st, const std::vector<double>* own, int t) {
		const FvNode& nd = game->nodes[at];
		const int N = n[s];
		std::vector<double> v(N, 0.0);
		if (nd.type == 1) {
			const double* q = &vq[s][qrow(nd) * N];
			if (nd.turn < 0)
				for (int i = 0; i < N; i++) v[i] = (1 + 2 * nd.c) * ((double)prod[(size_t)i * L + nd.line] / FV_FULL) - nd.c * q[i];
			else
				for (int i = 0; i < N; i++)
					v[i] = ((1 + 2 * nd.c) * ((double)prod_t[((size_t)nd.turn * N + i) * TL + nd.line] / FV_TFULL) - nd.c * q[i]) / FV_TLIVE;
		} else if (nd.type == 2) {
			const double* q = &vf[s][(size_t)nd.line * N];
			// the folder loses what it put in, the other wins the pot and that
			const double pay = (nd.player == s ? -nd.c : 1 + nd.c) / (nd.turn < 0 ? 1 : FV_TLIVE);
			for (int i = 0; i < N; i++) v[i] = pay * q[i];
		} else if (nd.type == 3) {
			std::vector<double> x;
			for (int k = 0; k < FV_TURNS; k++) {
				if (own) {
					x = *own;
					for (int i = 0; i < N; i++)
						if (holds(s, i, k)) x[i] = 0;
				}
				const std::vector<double> c = up(game->kids[nd.kids + k], s, prod, prod_t, st, own ? &x : nullptr, t);
				for (int i = 0; i < N; i++)
					if (!holds(s, i, k)) v[i] += c[i];
			}
		} else if (nd.player != s) {
			for (int a = 0; a < nd.nact; a++) {
				if (!nd.ok[a]) continue;
				const std::vector<double> c = up(nd.child[a], s, prod, prod_t, st, own, t);
				for (int i = 0; i < N; i++) v[i] += c[i];
			}
		} else {
			std::vector<double> va[3];
			if (!st) std::fill(v.begin(), v.end(), -1e300);
			for (int a = 0; a < nd.nact; a++) {
				if (!nd.ok[a]) continue;
				if (own) {
					std::vector<double> x(N);
					const double* p = &cur[s][(size_t)(nd.slot + a) * N];
					for (int i = 0; i < N; i++) x[i] = (*own)[i] * p[i];
					va[a] = up(nd.child[a], s, prod, prod_t, st, &x, t);
				} else {
					va[a] = up(nd.child[a], s, prod, prod_t, st, nullptr, t);
				}
				if (st) {
					const double* p = &st[(size_t)(nd.slot + a) * N];
					for (int i = 0; i < N; i++) v[i] += p[i] * va[a][i];
				} else {
					for (int i = 0; i < N; i++) v[i] = std::max(v[i], va[a][i]);
				}
			}
			if (own)
				for (int a = 0; a < nd.nact; a++) {
					if (!nd.ok[a]) continue;
					const size_t k = (size_t)(nd.slot + a) * N;
					for (int i = 0; i < N; i++) {
						sum[s][k + i] += t * (*own)[i] * cur[s][k + i];
						reg[s][k + i] = std::max(0.0, reg[s][k + i] + va[a][i] - v[i]);
					}
				}
		}
		return v;
	}
	double total(int s, const std::vector<double>& v) const {
		double x = 0;
		for (int i = 0; i < n[s]; i++) x += w[s][i] * v[i];
		return x;
	}

	// The whole range in line 0: the product gives every hand's all-in equity.
	void fill_range(int s, float* out) const {
		for (int i = 0; i < n[s]; i++)
			for (int c = 0; c < L; c++) out[(size_t)i * L + c] = c == 0 ? (float)w[s][i] : 0.0f;
	}
	void set_equity(int s, const float* prod) {
		const int N = n[s];
		tot[s].assign(N, 0.0);
		eq[s].assign(N, 0.5);
		valid(s, w[1 - s].data(), tot[s].data());
		W = 0;
		for (int i = 0; i < N; i++) {
			if (tot[s][i] > 0) eq[s][i] = (double)prod[(size_t)i * L] / FV_FULL / tot[s][i];
			W += w[s][i] * tot[s][i];
		}
	}

	// One CFR+ update of side s (iteration t, counted from 1) from the products of the other side's reach.
	void step(int s, const float* prod, int t, const float* prod_t = nullptr) {
		const std::vector<double> one(n[s], 1.0);
		up(0, s, prod, prod_t, cur[s].data(), &one, t);
		match(s);
	}
	// Against the other side's average (fill(1 - s, true) before): the best response value and the value of s's average.
	double best(int s, const float* prod, const float* prod_t = nullptr) { return total(s, up(0, s, prod, prod_t, nullptr, nullptr, 0)); }
	double value(int s, const float* prod, const float* prod_t = nullptr) {
		const std::vector<double> av = average(s);
		return total(s, up(0, s, prod, prod_t, av.data(), nullptr, 0));
	}

	// A strategy of side s for hand i on the flop, in the record's 13 slots (FV_OFF); 0 where the tree has no such decision.
	void flop_slots(int s, const std::vector<double>& st, int i, double* out) const {
		for (int k = 0; k < FV_SLOTS; k++) out[k] = 0;
		for (const FvNode& nd : game->nodes)
			if (nd.type == 0 && nd.player == s && nd.nd >= 0)
				for (int a = 0; a < nd.nact; a++) out[FV_OFF[nd.nd] + a] = st[(size_t)(nd.slot + a) * n[s] + i];
	}

	// The result record. best_o, best_i: best response values; value_o: O's value when both play their average.
	void result(double best_o, double best_i, double value_o, int iters, float* out) const {
		for (int k = 0; k < FV_K; k++) out[k] = 0;
		if (W <= 0) return;
		double eo = 0;
		for (int i = 0; i < n[0]; i++) eo += w[0][i] * tot[0][i] * eq[0][i];
		out[0] = (float)(value_o / W);
		out[1] = (float)(((best_o + best_i) / W - 1) / 2);
		out[2] = (float)(eo / W);
		out[3] = (float)iters;
		for (int s = 0; s < 2; s++) {
			const std::vector<double> av = average(s);
			for (int i = 0; i < n[s]; i++) {
				const double mass = w[s][i] * tot[s][i] / W;
				if (mass <= 0) continue;
				const int bin = std::min(FS_BINS - 1, (int)(eq[s][i] * FS_BINS + 1e-9));
				double x[FV_SLOTS];
				flop_slots(s, av, i, x);
				for (int nd = 0; nd < 5; nd++) {
					// how often the hand reaches the decision by its own actions
					const double own = nd == 0 ? 1 : nd <= 2 ? (s == 0 ? x[0] : 1) : x[nd - 2];
					for (int a = 0; a < FV_NACT[nd]; a++) out[FV_HEAD + ((s * 5 + nd) * FS_BINS + bin) * 3 + a] += (float)(mass * own * x[FV_OFF[nd] + a]);
				}
			}
		}
	}
};

// Reading a result record: the share of the hands at a decision (node = side * 5 + nd) that take action a,
// over the equity bins [bin_from, bin_to); -1 when no hand of those bins gets there.
static inline double fv_freq(const float* rec, int node, int a, int bin_from = 0, int bin_to = FS_BINS) {
	double x = 0, all = 0;
	for (int b = bin_from; b < bin_to; b++)
		for (int k = 0; k < 3; k++) {
			const double v = rec[FV_HEAD + (node * FS_BINS + b) * 3 + k];
			all += v;
			if (k == a) x += v;
		}
	return all > 1e-12 ? x / all : -1;
}
// The share of the range that reaches a decision by its own actions.
static inline double fv_reach(const float* rec, int node) {
	double all = 0;
	for (int k = 0; k < FS_BINS * 3; k++) all += rec[FV_HEAD + node * FS_BINS * 3 + k];
	return all;
}

// -- slow host reference: the same solver with plain loops for the products ---------------------------------------
// out[i * L + c] = sum over k of m(i, k) * v[k * L + c]; rows = true: the matrix as stored (O's side), false: the
// complement of its transpose (I's side).
static inline void fv_host_mul(const uint16_t* T, int NA, int NC, int L, bool rows, const float* v, float* out) {
	const int N = rows ? NA : NC, M = rows ? NC : NA;
	std::vector<double> acc(L);
	for (int i = 0; i < N; i++) {
		std::fill(acc.begin(), acc.end(), 0.0);
		for (int k = 0; k < M; k++) {
			const uint16_t t = rows ? T[(size_t)i * NC + k] : T[(size_t)k * NC + i];
			if (t == FV_NONE) continue;
			const double m = rows ? t : FV_FULL - t;
			for (int c = 0; c < L; c++) acc[c] += m * v[(size_t)k * L + c];
		}
		for (int c = 0; c < L; c++) out[(size_t)i * L + c] = (float)acc[c];
	}
}
// The same for the 49 turn matrices Tk[(k * NA + i) * NC + j]: block k of the input and of the output.
static inline void fv_host_mul_turn(const uint8_t* Tk, int NA, int NC, int L, bool rows, const float* v, float* out) {
	const int N = rows ? NA : NC, M = rows ? NC : NA;
	std::vector<double> acc(L);
	for (int k = 0; k < FV_TURNS; k++)
		for (int i = 0; i < N; i++) {
			std::fill(acc.begin(), acc.end(), 0.0);
			for (int j = 0; j < M; j++) {
				const uint8_t t = rows ? Tk[((size_t)k * NA + i) * NC + j] : Tk[((size_t)k * NA + j) * NC + i];
				if (t == FV_TNONE) continue;
				const double m = rows ? t : FV_TFULL - t;
				for (int c = 0; c < L; c++) acc[c] += m * v[((size_t)k * M + j) * L + c];
			}
			for (int c = 0; c < L; c++) out[((size_t)k * N + i) * L + c] = (float)acc[c];
		}
}
static inline void fv_solve_host(FvFlop& g, const uint16_t* T, int iters, float* out, double* value_sum = nullptr, const uint8_t* Tk = nullptr) {
	const int NA = g.n[0], NC = g.n[1], L = g.L, TL = g.TL;
	std::vector<float> q((size_t)NC * L), x((size_t)NA * L), p((size_t)NA * L), y((size_t)NC * L);
	std::vector<float> qt((size_t)FV_TURNS * NC * TL), xt((size_t)FV_TURNS * NA * TL), pt((size_t)FV_TURNS * NA * TL), yt((size_t)FV_TURNS * NC * TL);
	auto mul_o = [&]() {
		fv_host_mul(T, NA, NC, L, true, q.data(), x.data());
		if (TL) fv_host_mul_turn(Tk, NA, NC, TL, true, qt.data(), xt.data());
	};
	auto mul_i = [&]() {
		fv_host_mul(T, NA, NC, L, false, p.data(), y.data());
		if (TL) fv_host_mul_turn(Tk, NA, NC, TL, false, pt.data(), yt.data());
	};
	g.fill_range(1, q.data());
	fv_host_mul(T, NA, NC, L, true, q.data(), x.data());
	g.fill_range(0, p.data());
	fv_host_mul(T, NA, NC, L, false, p.data(), y.data());
	g.set_equity(0, x.data());
	g.set_equity(1, y.data());
	for (int t = 1; t <= iters; t++) {
		g.fill(1, false, q.data(), qt.data());
		mul_o();
		g.step(0, x.data(), t, xt.data());
		g.fill(0, false, p.data(), pt.data());
		mul_i();
		g.step(1, y.data(), t, yt.data());
	}
	g.fill(1, true, q.data(), qt.data());
	mul_o();
	const double best_o = g.best(0, x.data(), xt.data()), value_o = g.value(0, x.data(), xt.data());
	g.fill(0, true, p.data(), pt.data());
	mul_i();
	const double best_i = g.best(1, y.data(), yt.data());
	if (value_sum) *value_sum = (value_o + g.value(1, y.data(), yt.data())) / g.W;
	g.result(best_o, best_i, value_o, iters, out);
}

// -- device driver ------------------------------------------------------------------------------------------------
// Flops per batch for a matchup: fewer when the matrix or the tree is big, so a batch stays a matter of seconds.
static inline int fv_batch(int n_o, int n_i, int lines = 9, int turn_lines = 0) {
	if (turn_lines) {  // 49 matrices per flop: a few flops at a time
		const double work = (double)n_o * n_i * FV_TURNS * turn_lines * 2;
		return std::max(1, std::min(16, (int)(2.0e9 / std::max(1.0, work))));
	}
	const double cells = (double)n_o * n_i * std::max(1.0, lines / 9.0);
	return std::max(4, std::min(FS_BATCH, (int)(FS_BATCH * 60000.0 / std::max(1.0, cells))));
}

// The weights of a side on a flop: 0 for the combos that use a flop card.
static inline std::vector<double> fv_live(const FsSide& s, const uint8_t* fl) {
	std::vector<double> w(s.n);
	for (int i = 0; i < s.n; i++) {
		const int a = s.cards[i * 2], b = s.cards[i * 2 + 1];
		w[i] = (a == fl[0] || a == fl[1] || a == fl[2] || b == fl[0] || b == fl[1] || b == fl[2]) ? 0 : s.w[i];
	}
	return w;
}

// fn(f) for every f in [0, count), on a few host threads.
template <class F>
static inline void fv_each(int count, F&& fn) {
	const int T = std::min(count, FV_THREADS);
	if (T <= 1) {
		for (int f = 0; f < count; f++) fn(f);
		return;
	}
	std::vector<std::thread> th;
	for (int t = 0; t < T; t++)
		th.emplace_back([&fn, t, T, count]() {
			for (int f = t; f < count; f += T) fn(f);
		});
	for (std::thread& x : th) x.join();
}

struct FvDevice {
	int cap_n = 0;
	uint8_t *flops = nullptr, *co = nullptr, *ci = nullptr;
	uint32_t *ro = nullptr, *ri = nullptr;
	uint16_t *T = nullptr, *Tt = nullptr;
	float *vin = nullptr, *vout = nullptr;
	std::vector<float> h_in, h_out;
	std::vector<uint16_t> h_T;  // filled by solve() when keep_matrix is set (tests)
	std::vector<uint8_t> h_Tk;  // the same for the turn matrices
	bool keep_matrix = false;
	// the turn matrices and their products, sized for the batch at hand
	size_t cap_tk = 0, cap_tv = 0;
	uint8_t *Tk = nullptr, *Tkt = nullptr;
	float *tin = nullptr, *tout = nullptr;
	std::vector<float> h_tin, h_tout;

	void reserve_turn(size_t cells, size_t vals) {
		if (cells > cap_tk) {
			if (cap_tk) { cudaFree(Tk); cudaFree(Tkt); }
			cap_tk = cells;
			CK(cudaMalloc(&Tk, cells));
			CK(cudaMalloc(&Tkt, cells));
		}
		if (vals > cap_tv) {
			if (cap_tv) { cudaFree(tin); cudaFree(tout); }
			cap_tv = vals;
			CK(cudaMalloc(&tin, vals * 4));
			CK(cudaMalloc(&tout, vals * 4));
			h_tin.resize(vals);
			h_tout.resize(vals);
		}
	}

	void reserve(int n) {
		if (n <= cap_n) return;
		release();
		cap_n = n;
		CK(cudaMalloc(&flops, FS_BATCH * 3));
		CK(cudaMalloc(&co, (size_t)n * 2));
		CK(cudaMalloc(&ci, (size_t)n * 2));
		CK(cudaMalloc(&ro, (size_t)FS_BATCH * n * FS_R * 4));
		CK(cudaMalloc(&ri, (size_t)FS_BATCH * n * FS_R * 4));
		CK(cudaMalloc(&T, (size_t)FS_BATCH * n * n * 2));
		CK(cudaMalloc(&Tt, (size_t)FS_BATCH * n * n * 2));
		CK(cudaMalloc(&vin, (size_t)FS_BATCH * n * FV_MAXL * 4));
		CK(cudaMalloc(&vout, (size_t)FS_BATCH * n * FV_MAXL * 4));
		h_in.resize((size_t)FS_BATCH * n * FV_MAXL);
		h_out.resize((size_t)FS_BATCH * n * FV_MAXL);
	}
	void release() {
		if (!cap_n) return;
		cudaFree(flops); cudaFree(co); cudaFree(ci); cudaFree(ro); cudaFree(ri); cudaFree(T); cudaFree(Tt); cudaFree(vin); cudaFree(vout);
		cap_n = 0;
		if (cap_tk) { cudaFree(Tk); cudaFree(Tkt); cap_tk = 0; }
		if (cap_tv) { cudaFree(tin); cudaFree(tout); cap_tv = 0; }
	}

	// The products for side s of every flop of the batch: h_in holds the other side's reach, h_out gets the result
	// (and h_tin, h_tout for the turn matrices when the tree has a turn card: TL > 0).
	void mul(int s, int NO, int NI, int L, int count, int TL = 0, bool full = true) {
		const int N = s == 0 ? NO : NI, M = s == 0 ? NI : NO;
		if (full) {
			CK(cudaMemcpy(vin, h_in.data(), (size_t)count * M * L * 4, cudaMemcpyHostToDevice));
			fv_mul_kernel<<<(count * N + 127) / 128, 128>>>(s == 0 ? T : Tt, vin, N, M, count, L, s, vout);
			CK(cudaMemcpy(h_out.data(), vout, (size_t)count * N * L * 4, cudaMemcpyDeviceToHost));
		}
		if (TL) {
			const uint64_t fk = (uint64_t)count * FV_TURNS;
			CK(cudaMemcpy(tin, h_tin.data(), fk * M * TL * 4, cudaMemcpyHostToDevice));
			fv_mul_turn_kernel<<<(unsigned)((fk * N + 127) / 128), 128>>>(s == 0 ? Tk : Tkt, tin, N, M, (int)fk, TL, s, tout);
			CK(cudaMemcpy(h_tout.data(), tout, fk * N * TL * 4, cudaMemcpyDeviceToHost));
		}
	}

	// Solves `count` flops (3 ascending cards each) of one matchup; O = out of position. FV_K floats per flop.
	void solve(const FvGame& game, const FsSide& O, const FsSide& I, const uint8_t* flop_cards, int count, int iters, float* out) {
		fs_setup();
		reserve(std::max(O.n, I.n));
		const int NO = O.n, NI = I.n, L = std::max(1, game.lines);
		CK(cudaMemcpy(flops, flop_cards, (size_t)count * 3, cudaMemcpyHostToDevice));
		CK(cudaMemcpy(co, O.cards.data(), (size_t)NO * 2, cudaMemcpyHostToDevice));
		CK(cudaMemcpy(ci, I.cards.data(), (size_t)NI * 2, cudaMemcpyHostToDevice));
		const uint64_t no = (uint64_t)count * NO * FS_R, ni = (uint64_t)count * NI * FS_R, cells = (uint64_t)count * NO * NI;
		fs_rank_kernel<<<(unsigned)((no + 255) / 256), 256>>>(flops, co, NO, count, ro);
		fs_rank_kernel<<<(unsigned)((ni + 255) / 256), 256>>>(flops, ci, NI, count, ri);
		fv_matrix_kernel<<<(unsigned)((cells + 127) / 128), 128>>>(ro, ri, co, ci, NO, NI, count, T);
		fv_transpose_kernel<<<(unsigned)((cells + 255) / 256), 256>>>(T, NO, NI, count, Tt);
		CK(cudaGetLastError());
		if (keep_matrix) {
			h_T.resize(cells);
			CK(cudaMemcpy(h_T.data(), T, cells * 2, cudaMemcpyDeviceToHost));
		}
		const int TL = game.tlines;
		const bool full = game.lines > 0;  // otherwise the full matrix is only needed once, for the hands' equities
		const size_t so = (size_t)NO * L, si = (size_t)NI * L, to = (size_t)FV_TURNS * NO * TL, ti = (size_t)FV_TURNS * NI * TL;
		if (TL) {
			const uint64_t tcells = cells * FV_TURNS;
			reserve_turn(tcells, (size_t)count * std::max(to, ti));
			fv_turn_matrix_kernel<<<(unsigned)((tcells + 127) / 128), 128>>>(ro, ri, co, ci, NO, NI, count, Tk);
			fv_turn_transpose_kernel<<<(unsigned)((tcells + 255) / 256), 256>>>(Tk, NO, NI, count * FV_TURNS, Tkt);
			CK(cudaGetLastError());
			if (keep_matrix) {
				h_Tk.resize(tcells);
				CK(cudaMemcpy(h_Tk.data(), Tk, tcells, cudaMemcpyDeviceToHost));
			}
		}
		float* nil = nullptr;
		auto tin_of = [&](int f, size_t per) { return TL ? &h_tin[f * per] : nil; };
		auto tout_of = [&](int f, size_t per) { return TL ? &h_tout[f * per] : nil; };
		std::vector<FvFlop> g(count);
		fv_each(count, [&](int f) {
			g[f].init(game, NO, NI, fv_live(O, flop_cards + f * 3), fv_live(I, flop_cards + f * 3), O.cards.data(), I.cards.data(), flop_cards + f * 3);
		});
		// every hand's all-in equity
		fv_each(count, [&](int f) { g[f].fill_range(1, &h_in[f * si]); });
		mul(0, NO, NI, L, count);
		fv_each(count, [&](int f) { g[f].set_equity(0, &h_out[f * so]); });
		fv_each(count, [&](int f) { g[f].fill_range(0, &h_in[f * so]); });
		mul(1, NO, NI, L, count);
		fv_each(count, [&](int f) { g[f].set_equity(1, &h_out[f * si]); });
		for (int t = 1; t <= iters; t++) {
			fv_each(count, [&](int f) { g[f].fill(1, false, &h_in[f * si], tin_of(f, ti)); });
			mul(0, NO, NI, L, count, TL, full);
			fv_each(count, [&](int f) { g[f].step(0, &h_out[f * so], t, tout_of(f, to)); g[f].fill(0, false, &h_in[f * so], tin_of(f, to)); });
			mul(1, NO, NI, L, count, TL, full);
			fv_each(count, [&](int f) { g[f].step(1, &h_out[f * si], t, tout_of(f, ti)); });
		}
		std::vector<double> best_o(count), value_o(count);
		fv_each(count, [&](int f) { g[f].fill(1, true, &h_in[f * si], tin_of(f, ti)); });
		mul(0, NO, NI, L, count, TL, full);
		fv_each(count, [&](int f) {
			best_o[f] = g[f].best(0, &h_out[f * so], tout_of(f, to));
			value_o[f] = g[f].value(0, &h_out[f * so], tout_of(f, to));
			g[f].fill(0, true, &h_in[f * so], tin_of(f, to));
		});
		mul(1, NO, NI, L, count, TL, full);
		fv_each(count, [&](int f) { g[f].result(best_o[f], g[f].best(1, &h_out[f * si], tout_of(f, ti)), value_o[f], iters, out + (size_t)f * FV_K); });
		CK(cudaGetLastError());
	}
};

// -- a matchup as a relay -----------------------------------------------------------------------------------------
// `streets` everywhere below: 1 to 3 rounds without a card between them, FV_TURN_MODE = flop and a real turn, or
// FV_TURN3_MODE = flop, a real turn, and a third round of betting on that turn (the river's betting, without its card).
static const int FV_TURN_MODE = 12, FV_TURN3_MODE = 13;
static inline FvGame fv_game_of(double spr, int streets) {
	return streets == FV_TURN_MODE ? fv_game(spr, 2, true) : streets == FV_TURN3_MODE ? fv_game(spr, 3, true) : fv_game(spr, streets);
}
static inline uint64_t fv_tag(const FsMatchup& m, const std::vector<int>& flop_ids, int iters, int streets) {
	uint64_t h = relay_hash_u64(relay_hash_u64(relay_hash_u64(relay_hash_u64(13, FV_K), FS_R), iters), streets);
	h = relay_hash(h, FV_BET, sizeof FV_BET);
	h = relay_hash(h, &FV_LATER_BET, sizeof FV_LATER_BET);
	h = relay_hash(h, &FV_RAISE, sizeof FV_RAISE);
	const FvSpot s = fv_spot(m.key);
	h = relay_hash(h, &s.spr, sizeof s.spr);
	h = relay_hash_u64(h, s.aggressor_ip);
	h = relay_hash(h, m.key.data(), m.key.size());
	h = relay_hash(h, m.aggressor.data(), NCLASS * 8);
	h = relay_hash(h, m.caller.data(), NCLASS * 8);
	h = relay_hash(h, flop_ids.data(), flop_ids.size() * sizeof(int));
	return h;
}

// One matchup on the flops flop_ids (indices into all_flops()): FV_K floats per flop, in that order.
static inline std::vector<float> fv_run(FvDevice& dev, const FsMatchup& m, const std::vector<int>& flop_ids, int iters, int streets,
                                        const std::string& relay_path, bool verbose = false, uint64_t stop_after = 0, bool return_on_stop = false,
                                        uint64_t* done = nullptr) {
	const std::vector<Flop>& flops = all_flops();
	const FvSpot spot = fv_spot(m.key);
	const FvGame game = fv_game_of(spot.spr, streets);
	const FsSide A = fs_side(m.aggressor), C = fs_side(m.caller);
	const FsSide& O = spot.aggressor_ip ? C : A;
	const FsSide& I = spot.aggressor_ip ? A : C;
	const uint64_t total = flop_ids.size();
	std::vector<float> res(total * FV_K, 0.0f);
	Relay relay;
	relay.path = relay_path;
	relay.tag = fv_tag(m, flop_ids, iters, streets);
	relay.stop_after = stop_after;
	relay.return_on_stop = return_on_stop;
	relay.save_state = [&](FILE* f) { fwrite(res.data(), 4, res.size(), f); };
	relay.load_state = [&](FILE* f) { return fread(res.data(), 4, res.size(), f) == res.size(); };
	dispatch(total, fv_batch(O.n, I.n, game.lines, game.tlines), [&](uint64_t first, uint64_t count) {
		uint8_t cards[FS_BATCH * 3];
		for (uint64_t k = 0; k < count; k++)
			for (int x = 0; x < 3; x++) cards[k * 3 + x] = (uint8_t)flops[flop_ids[first + k]].cards[x];
		dev.solve(game, O, I, cards, (int)count, iters, &res[first * FV_K]);
	}, m.key.c_str(), verbose, &relay);
	if (done) *done = relay.done;
	return res;
}

// The flops a run covers: all 1,755 classes, or one per group of the tree (level 7: the first of each group in
// shape order).
static inline std::vector<int> fv_flop_ids(bool all) {
	std::vector<int> ids;
	if (all) {
		for (int i = 0; i < (int)all_flops().size(); i++) ids.push_back(i);
		return ids;
	}
	std::vector<int> group;
	fs_groups(7, group);
	int last = -1;
	for (int i : flop_shape_order())
		if (group[i] != last) { ids.push_back(i); last = group[i]; }
	return ids;
}

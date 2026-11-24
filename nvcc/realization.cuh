// Realisation factors: how much of its raw equity a hand class actually turns into chips when it sees a flop
// heads-up. Kernels and host drivers; used by realization.cu (command line) and test_realization.cu (checks).
//
// MODEL (an approximation, labelled as such on the site)
//   Two ranges (IP = small blind / button, OOP = big blind) meet on a random flop, turn and river. Each side
//   plays a FIXED, simple policy driven by the hand's equity against the other side's current range:
//     - the hand is ranked by its equity (all runouts to come) against the opponent's range at that node;
//       q = share of the side's own current range that is stronger (0 = best hand, 1 = worst);
//     - betting (OOP leading, or IP after a check): the top V of the range bets for value, the bottom
//       B = V * b / (1 + b) bluffs (the balanced ratio for a bet of b pots); everything else checks;
//     - facing a bet of b pots: the top MDF = 1 / (1 + b) of the range continues (fold the rest); the top rho
//       raises all-in when the stack left after a call is at most RAISE_SPR pots; a raise all-in is answered the
//       same way with the pot odds it offers;
//     - a bet of at least COMMIT of the remaining stack is an all-in; one bet size per street.
//   After every action each side's range is narrowed by the probability that the hand takes that action (exact,
//   no sampling of actions), so later streets use the ranges that really reach them. Only the board is random.
//   Equities use exact card removal and all runouts (flop: 1081 turn+river pairs, turn: 46 rivers), computed on the
//   GPU with a sort of the 1081 live combos by hand strength and prefix sums of the opponent's weights.
//
// OUTPUT per hero class (169) / side / pot type / depth, all against the opponent's range on a random board
//   raw    all-in equity (pre-flop all-in, five cards to come)
//   real   chips won / final pot, i.e. the share of the pot it actually collects, folds counted as 0
//   factor real / raw   (the classic "equity realisation")
//   ev     chips won minus chips put in after the flop (bb), so EV of the call = ev - preflop commitment
//   pot    average final pot (bb)
// Chips are exact per deal: on every deal the two sides' chips won add up to the final pot.
#pragma once

#include <cub/cub.cuh>
#include "poker.cuh"

static const int NC = 1326;          // two-card combos
static const int NT = 256;           // threads per block
static const int ITEMS = 6;          // sort items per thread: NT * ITEMS >= NC
static const int MAXN = 1184;        // most live combos in a sort: C(49,2) = 1176 on the flop
static const uint32_t SENT = 1u << 24;  // sort key of a dead combo (eval7 < 2^24)

// -- tables ----------------------------------------------------------------------------------------
__constant__ uint8_t k_c0[NC], k_c1[NC];
__constant__ uint16_t k_list[52 * 51];   // k_list[card * 51 + t]: the combos that contain `card`
__constant__ uint16_t k_cls[NCLASS][12]; // combo indices of each hand class

// -- policy ----------------------------------------------------------------------------------------
struct Pol {
	float bet[3];      // bet size, fraction of the pot (flop, turn, river)
	float vo[3];       // OOP leading: share of its range betting for value
	float vi[3];       // IP betting after a check: share betting for value
	float rho;         // share of the range raising all-in (when allowed)
	float commit;      // a bet of at least this share of the stack left is all-in
	float raise_spr;   // raising allowed when the stack left after a call is at most this many pots
	float defend;      // scale on the minimum defence frequency
};
__constant__ Pol c_pol;

static Pol default_pol() {
	Pol p;
	const float bet[3] = {0.33f, 0.66f, 0.75f}, vo[3] = {0.10f, 0.14f, 0.18f}, vi[3] = {0.32f, 0.30f, 0.30f};
	for (int i = 0; i < 3; i++) { p.bet[i] = bet[i]; p.vo[i] = vo[i]; p.vi[i] = vi[i]; }
	p.rho = 0.05f;
	p.commit = 0.60f;
	p.raise_spr = 2.0f;
	p.defend = 1.0f;
	return p;
}

struct Scn {
	float pot;    // pot at the flop
	float stack;  // effective stack behind, each side
};

// -- shared memory -----------------------------------------------------------------------------------
typedef cub::BlockRadixSort<uint32_t, NT, ITEMS, uint16_t> Sorter;
typedef cub::BlockScan<float, NT> Scanner;
typedef cub::BlockReduce<double, NT> Reducer;
union TmpStorage {
	Sorter::TempStorage sort;
	Scanner::TempStorage scan;
	Reducer::TempStorage red;
};
__shared__ TmpStorage s_tmp;
__shared__ uint8_t s_c0[NC], s_c1[NC];
__shared__ uint16_t s_list[52 * 51];
__shared__ uint32_t s_str[NC];            // strength of every combo on the loaded board (SENT if dead)
__shared__ uint32_t s_key[MAXN];          // sorted keys
__shared__ uint16_t s_idx[MAXN];          // combo index at each sorted position
__shared__ float s_pm0[MAXN + 1], s_pm1[MAXN + 1];  // exclusive prefix sums of weights in sorted order
__shared__ float s_csum[2][52];           // weight on each card
__shared__ double s_red, s_tot[2], s_diag[2];

// -- scratch layout: slots of NC floats per block ---------------------------------------------------
enum {
	G_WIN_I, G_TOT_I, G_WIN_O, G_TOT_O,            // scan outputs
	G_M_I, G_M_O,                                  // transient weights
	G_KEY,                                         // percentile keys
	G_RAWWIN_I, G_RAWTOT_I, G_RAWWIN_O, G_RAWTOT_O,
	G_CO_I, G_INV_I, G_PF_I, G_CO_O, G_INV_O, G_PF_O,  // chips won / put in / final pot, per side (3 each)
	G_V0, G_V1, G_V2,                              // 1 if the combo is alive on the flop / turn / river board
	G_ONES,
	G_ROOT_I, G_ROOT_O,                            // flop equity at the root (shared by all depths)
	G_FIN_STR, G_FIN_KEY, G_FIN_IDX,               // sorted structure of the final board
	G_LEVEL0
};
enum { L_MI, L_MO, L_EQI, L_EQO, L_QI, L_QO, L_ROB, L_ROC, L_RIB, L_RICC, L_TA, L_TB, L_TC, L_TM, L_COUNT };
static const int N_SLOTS = G_LEVEL0 + 3 * L_COUNT;

struct Blk {
	float* scr;
	const float* w0[2];   // range weight per combo, side 0 = IP, 1 = OOP
	int cards[5];         // flop, turn, river
	float P0, S0;         // flop pot, stack behind
};

__device__ __forceinline__ float* sl(const Blk& B, int slot) { return B.scr + (size_t)slot * NC; }
__device__ __forceinline__ float* lv(const Blk& B, int level, int k) { return sl(B, G_LEVEL0 + level * L_COUNT + k); }

// -- block primitives ---------------------------------------------------------------------------------
__device__ double block_total(double x) {
	double r = Reducer(s_tmp.red).Sum(x);
	if (threadIdx.x == 0) s_red = r;
	__syncthreads();
	double out = s_red;
	__syncthreads();
	return out;
}

__device__ __forceinline__ int lower_b(uint32_t v, int n) {
	int lo = 0, hi = n;
	while (lo < hi) {
		int mid = (lo + hi) >> 1;
		if (s_key[mid] < v) lo = mid + 1; else hi = mid;
	}
	return lo;
}
__device__ __forceinline__ int upper_b(uint32_t v, int n) {
	int lo = 0, hi = n;
	while (lo < hi) {
		int mid = (lo + hi) >> 1;
		if (s_key[mid] <= v) lo = mid + 1; else hi = mid;
	}
	return lo;
}

// Sorts the NC keys ascending into s_key / s_idx (first n positions). Dead keys are SENT and land last.
__device__ void block_sort(const uint32_t* keys, int n) {
	const int tid = threadIdx.x;
	uint32_t k[ITEMS];
	uint16_t v[ITEMS];
#pragma unroll
	for (int i = 0; i < ITEMS; i++) {
		int j = tid * ITEMS + i;
		k[i] = j < NC ? keys[j] : SENT;
		v[i] = (uint16_t)j;
	}
	__syncthreads();
	Sorter(s_tmp.sort).Sort(k, v, 0, 25);
#pragma unroll
	for (int i = 0; i < ITEMS; i++) {
		int pos = tid * ITEMS + i;
		if (pos < n) { s_key[pos] = k[i]; s_idx[pos] = v[i]; }
	}
	__syncthreads();
}

// pm[p] = sum of m over the first p sorted combos (exclusive), pm[n] = total.
__device__ void block_prefix(const float* m, float* pm, int n) {
	const int tid = threadIdx.x;
	const int CH = 5;  // NT * CH >= MAXN
	float loc[CH], sum = 0.f;
#pragma unroll
	for (int i = 0; i < CH; i++) {
		int p = tid * CH + i;
		float x = p < n ? m[s_idx[p]] : 0.f;
		loc[i] = x;
		sum += x;
	}
	float off, total;
	__syncthreads();
	Scanner(s_tmp.scan).ExclusiveSum(sum, off, total);
	float run = off;
#pragma unroll
	for (int i = 0; i < CH; i++) {
		int p = tid * CH + i;
		if (p < n) pm[p] = run;
		run += loc[i];
	}
	if (tid == 0) pm[n] = total;
	__syncthreads();
}

// Scores every combo on a five-card board and sorts the live ones (1081 of them) by strength.
__device__ void prepare_board(const int* b5) {
	uint64_t bm = 0;
	for (int i = 0; i < 5; i++) bm |= 1ull << b5[i];
	for (int j = threadIdx.x; j < NC; j += NT) {
		const int a = s_c0[j], b = s_c1[j];
		if (((bm >> a) | (bm >> b)) & 1) {
			s_str[j] = SENT;
		} else {
			int h[7] = {a, b, b5[0], b5[1], b5[2], b5[3], b5[4]};
			s_str[j] = eval7(h);
		}
	}
	__syncthreads();
	block_sort(s_str, 1081);
}

// For every live combo h: against the weights `mO` (for h on side IP) and `mI` (for h on side OOP), add the
// weight of opponents it beats (+ half the ties) to win*, and the weight of all opponents that do not share a
// card with it to tot*. Strictly-less and tie weights come from the sorted prefix sums; combos that share a
// card with h are taken out by walking the 51 combos on each of its two cards.
__device__ void scan_both(const float* mI, const float* mO, float* winI, float* totI, float* winO, float* totO) {
	const int n = 1081;
	block_prefix(mO, s_pm0, n);
	block_prefix(mI, s_pm1, n);
	for (int j = threadIdx.x; j < NC; j += NT) {
		const uint32_t s = s_str[j];
		if (s == SENT) continue;
		const int lo = lower_b(s, n), hi = upper_b(s, n);
		float lO = 0.f, tO = 0.f, aO = 0.f, lI = 0.f, tI = 0.f, aI = 0.f;
		for (int side = 0; side < 2; side++) {
			const uint16_t* lst = s_list + (side ? s_c1[j] : s_c0[j]) * 51;
			for (int t = 0; t < 51; t++) {
				const int o = lst[t];
				const uint32_t so = s_str[o];
				if (so == SENT) continue;
				const float xo = mO[o], xi = mI[o];
				aO += xo;
				aI += xi;
				if (so < s) { lO += xo; lI += xi; }
				else if (so == s) { tO += xo; tI += xi; }
			}
		}
		{
			const float less = s_pm0[lo] - lO;
			const float tie = s_pm0[hi] - s_pm0[lo] - tO + mO[j];
			winI[j] += less + 0.5f * tie;
			totI[j] += s_pm0[n] - aO + mO[j];
		}
		{
			const float less = s_pm1[lo] - lI;
			const float tie = s_pm1[hi] - s_pm1[lo] - tI + mI[j];
			winO[j] += less + 0.5f * tie;
			totO[j] += s_pm1[n] - aI + mI[j];
		}
	}
	__syncthreads();
}

// Equity of every combo, on both sides, against the weights mI / mO (zero where a combo is dead on the board
// shown so far), averaged over all runouts: ratio of the summed win weight to the summed opponent weight.
__device__ void node_eq(const Blk& B, int st, const float* mI, const float* mO, float* eqI, float* eqO) {
	const int tid = threadIdx.x;
	float *wI = sl(B, G_WIN_I), *tI = sl(B, G_TOT_I), *wO = sl(B, G_WIN_O), *tO = sl(B, G_TOT_O);
	for (int j = tid; j < NC; j += NT) { wI[j] = 0.f; tI[j] = 0.f; wO[j] = 0.f; tO[j] = 0.f; }
	__syncthreads();
	int b5[5];
	for (int i = 0; i < 5; i++) b5[i] = B.cards[i];
	if (st == 2) {
		prepare_board(b5);
		scan_both(mI, mO, wI, tI, wO, tO);
	} else if (st == 1) {
		for (int r = 0; r < 52; r++) {
			bool used = false;
			for (int i = 0; i < 4; i++) used |= (b5[i] == r);
			if (used) continue;
			b5[4] = r;
			prepare_board(b5);
			scan_both(mI, mO, wI, tI, wO, tO);
		}
	} else {
		for (int t = 0; t < 52; t++) {
			bool used = false;
			for (int i = 0; i < 3; i++) used |= (b5[i] == t);
			if (used) continue;
			for (int r = t + 1; r < 52; r++) {
				if (b5[0] == r || b5[1] == r || b5[2] == r) continue;
				b5[3] = t;
				b5[4] = r;
				prepare_board(b5);
				scan_both(mI, mO, wI, tI, wO, tO);
			}
		}
	}
	for (int j = tid; j < NC; j += NT) {
		eqI[j] = tI[j] > 1e-9f ? wI[j] / tI[j] : 0.5f;
		eqO[j] = tO[j] > 1e-9f ? wO[j] / tO[j] : 0.5f;
	}
	__syncthreads();
}

// q[j] = share of the weight m that is strictly stronger than combo j (plus half of the equal ones), where
// strength is eq. Computed for every live combo, in the range or not. n = live combos on the street's board.
__device__ void percentile(const Blk& B, const float* eq, const float* m, const float* valid, int n, float* q) {
	const int tid = threadIdx.x;
	uint32_t* key = (uint32_t*)sl(B, G_KEY);
	for (int j = tid; j < NC; j += NT) {
		uint32_t k = (uint32_t)(fminf(fmaxf(eq[j], 0.f), 1.f) * 16777214.f);
		key[j] = valid[j] > 0.f ? k : SENT;
	}
	__syncthreads();
	block_sort(key, n);
	block_prefix(m, s_pm0, n);
	const float M = s_pm0[n];
	for (int j = tid; j < NC; j += NT) {
		if (key[j] == SENT) { q[j] = 1.f; continue; }
		const int lo = lower_b(key[j], n), hi = upper_b(key[j], n);
		q[j] = M > 1e-9f ? ((M - s_pm0[hi]) + 0.5f * (s_pm0[hi] - s_pm0[lo])) / M : 0.5f;
	}
	__syncthreads();
}

// Weight on each card of the weights a and b (live combos only), and the weight of the pairs (h, o) with h on a,
// o on b that share no card. Leaves the card sums in s_csum and the totals in s_tot.
__device__ double pair_stats(const float* a, const float* b) {
	const int tid = threadIdx.x;
	if (tid < 52) {
		float sa = 0.f, sb = 0.f;
		for (int t = 0; t < 51; t++) {
			const int o = s_list[tid * 51 + t];
			sa += a[o];
			sb += b[o];
		}
		s_csum[0][tid] = sa;
		s_csum[1][tid] = sb;
	}
	__syncthreads();
	double ta = 0, tb = 0, same = 0;
	for (int j = tid; j < NC; j += NT) { ta += a[j]; tb += b[j]; same += (double)a[j] * b[j]; }
	ta = block_total(ta);
	tb = block_total(tb);
	same = block_total(same);
	double cross = 0;
	if (tid < 52) cross = (double)s_csum[0][tid] * s_csum[1][tid];
	cross = block_total(cross);
	if (tid == 0) { s_tot[0] = ta; s_tot[1] = tb; }
	__syncthreads();
	return ta * tb - (cross - same);
}

// The loaded sorted structure of the final board lives in global memory between showdowns.
__device__ void save_final(const Blk& B) {
	uint32_t* fs = (uint32_t*)sl(B, G_FIN_STR);
	uint32_t* fk = (uint32_t*)sl(B, G_FIN_KEY);
	uint16_t* fi = (uint16_t*)sl(B, G_FIN_IDX);
	for (int j = threadIdx.x; j < NC; j += NT) fs[j] = s_str[j];
	for (int j = threadIdx.x; j < 1081; j += NT) { fk[j] = s_key[j]; fi[j] = s_idx[j]; }
	__syncthreads();
}
__device__ void load_final(const Blk& B) {
	const uint32_t* fs = (const uint32_t*)sl(B, G_FIN_STR);
	const uint32_t* fk = (const uint32_t*)sl(B, G_FIN_KEY);
	const uint16_t* fi = (const uint16_t*)sl(B, G_FIN_IDX);
	for (int j = threadIdx.x; j < NC; j += NT) s_str[j] = fs[j];
	for (int j = threadIdx.x; j < 1081; j += NT) { s_key[j] = fk[j]; s_idx[j] = fi[j]; }
	__syncthreads();
}

// -- terminals ------------------------------------------------------------------------------------------
// `win` side takes the pot. rW / rL: the probability that each combo of the winner / loser reaches this
// terminal. pf: final pot; invW / invL: chips each side put in after the flop.
__device__ void term_fold(const Blk& B, int win, const float* rW, const float* rL, float pf, float invW, float invL) {
	const int tid = threadIdx.x, los = 1 - win;
	float* mW = sl(B, G_M_I);
	float* mL = sl(B, G_M_O);
	const float* v = sl(B, G_V2);
	for (int j = tid; j < NC; j += NT) {
		mW[j] = B.w0[win][j] * rW[j] * v[j];
		mL[j] = B.w0[los][j] * rL[j] * v[j];
	}
	__syncthreads();
	const double pairs = pair_stats(mW, mL);
	const float totW = (float)s_tot[0], totL = (float)s_tot[1];
	float *coW = sl(B, G_CO_I + 3 * win), *invWa = sl(B, G_INV_I + 3 * win), *pfW = sl(B, G_PF_I + 3 * win);
	float *invLa = sl(B, G_INV_I + 3 * los), *pfL = sl(B, G_PF_I + 3 * los);
	for (int j = tid; j < NC; j += NT) {
		if (v[j] == 0.f) continue;
		const int c0 = s_c0[j], c1 = s_c1[j];
		const float oppW = totL - s_csum[1][c0] - s_csum[1][c1] + mL[j];   // loser weight that does not block winner's hand j
		const float oppL = totW - s_csum[0][c0] - s_csum[0][c1] + mW[j];
		coW[j] += rW[j] * pf * oppW;
		pfW[j] += rW[j] * pf * oppW;
		invWa[j] += rW[j] * invW * oppW;
		pfL[j] += rL[j] * pf * oppL;
		invLa[j] += rL[j] * invL * oppL;
	}
	if (tid == 0) { s_diag[0] += pairs; s_diag[1] += (double)pf * pairs; }
	__syncthreads();
}

__device__ void showdown(const Blk& B, const float* rI, const float* rO, float pf, float invI, float invO) {
	const int tid = threadIdx.x;
	float* mI = sl(B, G_M_I);
	float* mO = sl(B, G_M_O);
	const float* v = sl(B, G_V2);
	float *wI = sl(B, G_WIN_I), *tI = sl(B, G_TOT_I), *wO = sl(B, G_WIN_O), *tO = sl(B, G_TOT_O);
	for (int j = tid; j < NC; j += NT) {
		mI[j] = B.w0[0][j] * rI[j] * v[j];
		mO[j] = B.w0[1][j] * rO[j] * v[j];
		wI[j] = 0.f; tI[j] = 0.f; wO[j] = 0.f; tO[j] = 0.f;
	}
	__syncthreads();
	load_final(B);
	scan_both(mI, mO, wI, tI, wO, tO);
	const double pairs = pair_stats(mI, mO);
	float *coI = sl(B, G_CO_I), *invIa = sl(B, G_INV_I), *pfI = sl(B, G_PF_I);
	float *coO = sl(B, G_CO_O), *invOa = sl(B, G_INV_O), *pfO = sl(B, G_PF_O);
	for (int j = tid; j < NC; j += NT) {
		if (v[j] == 0.f) continue;
		coI[j] += rI[j] * pf * wI[j];
		pfI[j] += rI[j] * pf * tI[j];
		invIa[j] += rI[j] * invI * tI[j];
		coO[j] += rO[j] * pf * wO[j];
		pfO[j] += rO[j] * pf * tO[j];
		invOa[j] += rO[j] * invO * tO[j];
	}
	if (tid == 0) { s_diag[0] += pairs; s_diag[1] += (double)pf * pairs; }
	__syncthreads();
}

// -- the betting tree --------------------------------------------------------------------------------------
__device__ void street_node(const Blk& B, int st, float pot, float inv, const float* rI, const float* rO, bool root);

__device__ void child(const Blk& B, int st, float pot, float inv, const float* rI, const float* rO, bool allin) {
	if (allin || st == 2) showdown(B, rI, rO, pot, inv, inv);
	else street_node(B, st + 1, pot, inv, rI, rO, false);
}

// One street. rI / rO: probability that each combo has arrived here with the actions taken so far (every combo,
// in the range or not). pot: chips in the middle, inv: chips each side has put in since the flop.
__device__ void street_node(const Blk& B, int st, float pot, float inv, const float* rI, const float* rO, bool root) {
	const int tid = threadIdx.x;
	const float S = B.S0 - inv;
	float *mI = lv(B, st, L_MI), *mO = lv(B, st, L_MO), *qI = lv(B, st, L_QI), *qO = lv(B, st, L_QO);
	float *rOb = lv(B, st, L_ROB), *rOc = lv(B, st, L_ROC), *rIb = lv(B, st, L_RIB), *rIc = lv(B, st, L_RICC);
	float *tA = lv(B, st, L_TA), *tB = lv(B, st, L_TB), *tC = lv(B, st, L_TC), *tM = lv(B, st, L_TM);
	const float* eqI = root ? sl(B, G_ROOT_I) : lv(B, st, L_EQI);
	const float* eqO = root ? sl(B, G_ROOT_O) : lv(B, st, L_EQO);
	const float* valid = sl(B, G_V0 + st);
	const float* w0I = B.w0[0];
	const float* w0O = B.w0[1];
	const int nval = (49 - st) * (48 - st) / 2;  // live combos on this street's board

	for (int j = tid; j < NC; j += NT) {
		mI[j] = w0I[j] * rI[j] * valid[j];
		mO[j] = w0O[j] * rO[j] * valid[j];
	}
	__syncthreads();
	if (!root) node_eq(B, st, mI, mO, lv(B, st, L_EQI), lv(B, st, L_EQO));
	percentile(B, eqI, mI, valid, nval, qI);
	percentile(B, eqO, mO, valid, nval, qO);

	// sizes and thresholds
	float x = c_pol.bet[st] * pot;
	const bool allin = x >= c_pol.commit * S;
	if (allin) x = S;
	const float be = x / pot;
	const float vo = c_pol.vo[st], vi = c_pol.vi[st];
	const float bo = vo * be / (1.f + be), bi = vi * be / (1.f + be);
	const bool can_raise = !allin && (S - x <= c_pol.raise_spr * (pot + 2.f * x));
	const float mdf = pot / (pot + x) * c_pol.defend;                       // continue share facing the bet
	const float mdf_r = (pot + x + S) / ((pot + x + S) + (S - x));          // continue share facing an all-in raise
	const float rho = c_pol.rho;

	// who bets
	// Bluffs: on the river the weakest hands (no showdown value, nothing to lose by being called). On the flop and
	// turn the best of the hands that would fold to the bet, i.e. the ones with the most equity to fall back on
	// (draws), so a hand far below the range never bluffs.
	const float band_lo = st == 2 ? 1.f - 2.f : mdf;  // river: bottom share, handled by q > 1 - B below
	auto bluff = [&](float q, float B) { return st == 2 ? (q > 1.f - B) : (q >= band_lo && q < band_lo + B); };
	for (int j = tid; j < NC; j += NT) {
		const float bo_j = (qO[j] < vo || bluff(qO[j], bo)) ? 1.f : 0.f;
		const float bi_j = (qI[j] < vi || bluff(qI[j], bi)) ? 1.f : 0.f;
		rOb[j] = rO[j] * bo_j;
		rOc[j] = rO[j] * (1.f - bo_j);
		rIb[j] = rI[j] * bi_j;
		rIc[j] = rI[j] * (1.f - bi_j);
	}
	__syncthreads();

	// --- OOP bets x ---
	for (int j = tid; j < NC; j += NT) {
		const bool rz = can_raise && qI[j] < rho;
		const bool cl = !rz && qI[j] < mdf;
		tA[j] = rI[j] * ((rz || cl) ? 0.f : 1.f);
	}
	__syncthreads();
	term_fold(B, 1, rOb, tA, pot + x, inv + x, inv);                         // IP folds
	for (int j = tid; j < NC; j += NT) {
		const bool rz = can_raise && qI[j] < rho;
		tA[j] = rI[j] * ((!rz && qI[j] < mdf) ? 1.f : 0.f);
	}
	__syncthreads();
	child(B, st, pot + 2.f * x, inv + x, tA, rOb, allin);                     // IP calls
	if (can_raise) {                                                          // IP raises all-in
		for (int j = tid; j < NC; j += NT) {
			tA[j] = rI[j] * ((qI[j] < rho) ? 1.f : 0.f);
			tM[j] = w0O[j] * rOb[j] * valid[j];                                // OOP's betting range
		}
		__syncthreads();
		percentile(B, eqO, tM, valid, nval, tB);
		for (int j = tid; j < NC; j += NT) tC[j] = rOb[j] * (tB[j] < mdf_r ? 0.f : 1.f);
		__syncthreads();
		term_fold(B, 0, tA, tC, pot + x + S, inv + S, inv + x);                // OOP folds
		for (int j = tid; j < NC; j += NT) tC[j] = rOb[j] * (tB[j] < mdf_r ? 1.f : 0.f);
		__syncthreads();
		showdown(B, tA, tC, pot + 2.f * S, inv + S, inv + S);                  // OOP calls
	}

	// --- OOP checks ---
	child(B, st, pot, inv, rIc, rOc, false);                                  // IP checks behind
	// OOP's checking range answers IP's bet
	for (int j = tid; j < NC; j += NT) tM[j] = w0O[j] * rOc[j] * valid[j];
	__syncthreads();
	percentile(B, eqO, tM, valid, nval, tB);
	for (int j = tid; j < NC; j += NT) {
		const bool rz = can_raise && tB[j] < rho;
		const bool cl = !rz && tB[j] < mdf;
		tA[j] = rOc[j] * ((rz || cl) ? 0.f : 1.f);
	}
	__syncthreads();
	term_fold(B, 0, rIb, tA, pot + x, inv + x, inv);                          // OOP folds to IP's bet
	for (int j = tid; j < NC; j += NT) {
		const bool rz = can_raise && tB[j] < rho;
		tA[j] = rOc[j] * ((!rz && tB[j] < mdf) ? 1.f : 0.f);
	}
	__syncthreads();
	child(B, st, pot + 2.f * x, inv + x, rIb, tA, allin);                     // OOP calls
	if (can_raise) {                                                          // OOP check-raises all-in
		for (int j = tid; j < NC; j += NT) {
			tA[j] = rOc[j] * ((tB[j] < rho) ? 1.f : 0.f);
			tM[j] = w0I[j] * rIb[j] * valid[j];                                // IP's betting range
		}
		__syncthreads();
		percentile(B, eqI, tM, valid, nval, tB);
		for (int j = tid; j < NC; j += NT) tC[j] = rIb[j] * (tB[j] < mdf_r ? 0.f : 1.f);
		__syncthreads();
		term_fold(B, 1, tA, tC, pot + x + S, inv + S, inv + x);                // IP folds
		for (int j = tid; j < NC; j += NT) tC[j] = rIb[j] * (tB[j] < mdf_r ? 1.f : 0.f);
		__syncthreads();
		showdown(B, tC, tA, pot + 2.f * S, inv + S, inv + S);                  // IP calls
	}
}

// -- the kernel -----------------------------------------------------------------------------------------------
// One block per board at a time (persistent blocks stride over the boards). Output per board and scenario and
// side and class: five floats {chips won, final pot, raw equity, chips put in, opponent weight} summed over the
// class's live combos. Dividing sums over boards by the summed opponent weight gives the average over (hero hand,
// opponent hand, board) dealt together; dividing each board by its own weight first would bias the boards that remove
// part of the opponent's range.
static const int STAT_N = 5;
static const int DIAG_N = 8;

__global__ void __launch_bounds__(NT, 2)
rz_kernel(const Scn* scn, int nscn, const float* w0I, const float* w0O, uint64_t seed, long long board0, int nboards,
          float* scratch, float* stats, double* diag) {
	const int tid = threadIdx.x;
	for (int j = tid; j < NC; j += NT) { s_c0[j] = k_c0[j]; s_c1[j] = k_c1[j]; }
	for (int j = tid; j < 52 * 51; j += NT) s_list[j] = k_list[j];
	__syncthreads();
	Blk B;
	B.scr = scratch + (size_t)blockIdx.x * N_SLOTS * NC;
	B.w0[0] = w0I;
	B.w0[1] = w0O;
	for (int bi = blockIdx.x; bi < nboards; bi += gridDim.x) {
		Rng r;
		r.s = mix64(seed ^ mix64((uint64_t)(board0 + bi) * 0xD1B54A32D192ED03ull + 77));
		uint64_t used = 0;
		for (int i = 0; i < 5; i++) {
			int c;
			do { c = r.below(52); } while ((used >> c) & 1);
			used |= 1ull << c;
			B.cards[i] = c;
		}
		// live-combo masks per street
		float *v0 = sl(B, G_V0), *v1 = sl(B, G_V1), *v2 = sl(B, G_V2), *ones = sl(B, G_ONES);
		uint64_t m3 = 0, m4 = 0, m5 = 0;
		for (int i = 0; i < 5; i++) {
			const uint64_t b = 1ull << B.cards[i];
			if (i < 3) m3 |= b;
			if (i < 4) m4 |= b;
			m5 |= b;
		}
		for (int j = tid; j < NC; j += NT) {
			const uint64_t cm = (1ull << s_c0[j]) | (1ull << s_c1[j]);
			v0[j] = (cm & m3) ? 0.f : 1.f;
			v1[j] = (cm & m4) ? 0.f : 1.f;
			v2[j] = (cm & m5) ? 0.f : 1.f;
			ones[j] = 1.f;
		}
		__syncthreads();

		// raw equity on the actual five cards (the final board), against the whole range
		float *mI = sl(B, G_M_I), *mO = sl(B, G_M_O);
		float *rwI = sl(B, G_RAWWIN_I), *rtI = sl(B, G_RAWTOT_I), *rwO = sl(B, G_RAWWIN_O), *rtO = sl(B, G_RAWTOT_O);
		for (int j = tid; j < NC; j += NT) {
			mI[j] = w0I[j] * v2[j];
			mO[j] = w0O[j] * v2[j];
			rwI[j] = 0.f; rtI[j] = 0.f; rwO[j] = 0.f; rtO[j] = 0.f;
		}
		__syncthreads();
		prepare_board(B.cards);
		save_final(B);
		scan_both(mI, mO, rwI, rtI, rwO, rtO);
		const double pairs_all = pair_stats(mI, mO);
		double raw_i = 0, raw_o = 0;
		for (int j = tid; j < NC; j += NT) { raw_i += (double)w0I[j] * v2[j] * rwI[j]; raw_o += (double)w0O[j] * v2[j] * rwO[j]; }
		raw_i = block_total(raw_i);
		raw_o = block_total(raw_o);

		// flop equity at the root: the same for every depth
		for (int j = tid; j < NC; j += NT) { mI[j] = w0I[j] * v0[j]; mO[j] = w0O[j] * v0[j]; }
		__syncthreads();
		node_eq(B, 0, mI, mO, sl(B, G_ROOT_I), sl(B, G_ROOT_O));

		for (int d = 0; d < nscn; d++) {
			B.P0 = scn[d].pot;
			B.S0 = scn[d].stack;
			for (int s = G_CO_I; s <= G_PF_O; s++)
				for (int j = tid; j < NC; j += NT) sl(B, s)[j] = 0.f;
			if (tid == 0) { s_diag[0] = 0; s_diag[1] = 0; }
			__syncthreads();
			street_node(B, 0, B.P0, 0.f, ones, ones, true);

			// per class sums
			for (int ci = tid; ci < 2 * NCLASS; ci += NT) {
				const int side = ci / NCLASS, cls = ci % NCLASS;
				const float *co = sl(B, G_CO_I + 3 * side), *pf = sl(B, G_PF_I + 3 * side), *iv = sl(B, G_INV_I + 3 * side);
				const float *rw = side ? rwO : rwI, *rt = side ? rtO : rtI;
				float aC = 0, aP = 0, aE = 0, aI = 0, n = 0;
				for (int m = 0; m < c_ncombo[cls]; m++) {
					const int j = k_cls[cls][m];
					const float z = rt[j];
					if (v2[j] == 0.f || z <= 1e-9f) continue;
					aC += co[j];
					aP += pf[j];
					aE += rw[j];
					aI += iv[j];
					n += z;  // a board counts in proportion to the opponent weight it leaves, as when dealing opponent and board together
				}
				float* o = stats + ((((size_t)bi * nscn + d) * 2 + side) * NCLASS + cls) * STAT_N;
				o[0] = aC; o[1] = aP; o[2] = aE; o[3] = aI; o[4] = n;
			}
			// bookkeeping identities (see test_realization.cu)
			double co_i = 0, co_o = 0, iv_i = 0, iv_o = 0;
			const float *coI = sl(B, G_CO_I), *coO = sl(B, G_CO_O), *ivI = sl(B, G_INV_I), *ivO = sl(B, G_INV_O);
			for (int j = tid; j < NC; j += NT) {
				co_i += (double)w0I[j] * v2[j] * coI[j];
				co_o += (double)w0O[j] * v2[j] * coO[j];
				iv_i += (double)w0I[j] * v2[j] * ivI[j];
				iv_o += (double)w0O[j] * v2[j] * ivO[j];
			}
			co_i = block_total(co_i);
			co_o = block_total(co_o);
			iv_i = block_total(iv_i);
			iv_o = block_total(iv_o);
			if (tid == 0) {
				double* o = diag + ((size_t)bi * nscn + d) * DIAG_N;
				o[0] = pairs_all; o[1] = s_diag[0]; o[2] = s_diag[1]; o[3] = co_i; o[4] = co_o;
				o[5] = raw_i + raw_o; o[6] = iv_i + iv_o; o[7] = (double)B.P0;
			}
			__syncthreads();
		}
	}
}

// Test hook: the strength sort and scan on one board with caller-chosen weights.
__global__ void scan_test_kernel(const int* b5, const float* mI, const float* mO, float* winI, float* totI, float* winO,
                                 float* totO, uint32_t* strength) {
	for (int j = threadIdx.x; j < NC; j += NT) { s_c0[j] = k_c0[j]; s_c1[j] = k_c1[j]; }
	for (int j = threadIdx.x; j < 52 * 51; j += NT) s_list[j] = k_list[j];
	__syncthreads();
	int c[5];
	for (int i = 0; i < 5; i++) c[i] = b5[i];
	prepare_board(c);
	for (int j = threadIdx.x; j < NC; j += NT) { winI[j] = 0; totI[j] = 0; winO[j] = 0; totO[j] = 0; strength[j] = s_str[j]; }
	__syncthreads();
	scan_both(mI, mO, winI, totI, winO, totO);
}

// Test hook: percentile ranks.
__global__ void percentile_test_kernel(const float* eq, const float* m, const float* valid, int n, float* q, float* scratch) {
	for (int j = threadIdx.x; j < NC; j += NT) { s_c0[j] = k_c0[j]; s_c1[j] = k_c1[j]; }
	__syncthreads();
	Blk B;
	B.scr = scratch;
	percentile(B, eq, m, valid, n, q);
}

// ====================================================================================================
// host
// ====================================================================================================
static int g_pair_idx[52][52];
static uint8_t g_c0[NC], g_c1[NC];
static int g_combo_class[NC];

static void setup_combos() {
	int n = 0;
	for (int a = 0; a < 52; a++)
		for (int b = a + 1; b < 52; b++) {
			g_c0[n] = a;
			g_c1[n] = b;
			g_pair_idx[a][b] = g_pair_idx[b][a] = n;
			n++;
		}
	uint16_t list[52 * 51];
	int cnt[52] = {};
	for (int j = 0; j < NC; j++) {
		list[g_c0[j] * 51 + cnt[g_c0[j]]++] = j;
		list[g_c1[j] * 51 + cnt[g_c1[j]]++] = j;
	}
	uint16_t cls[NCLASS][12] = {};
	for (int c = 0; c < NCLASS; c++)
		for (int m = 0; m < g_ncombo[c]; m++) {
			int j = g_pair_idx[g_combos[c][m][0]][g_combos[c][m][1]];
			cls[c][m] = j;
			g_combo_class[j] = c;
		}
	CK(cudaMemcpyToSymbol(k_c0, g_c0, sizeof(g_c0)));
	CK(cudaMemcpyToSymbol(k_c1, g_c1, sizeof(g_c1)));
	CK(cudaMemcpyToSymbol(k_list, list, sizeof(list)));
	CK(cudaMemcpyToSymbol(k_cls, cls, sizeof(cls)));
}

static void set_policy(const Pol& p) { CK(cudaMemcpyToSymbol(c_pol, &p, sizeof(p))); }

// class weights -> per-combo weights
static std::vector<float> combo_weights(const std::vector<double>& w) {
	std::vector<float> out(NC);
	for (int j = 0; j < NC; j++) out[j] = (float)w[g_combo_class[j]];
	return out;
}

struct Cell {
	double raw = 0, real = 0, factor = 0, se = 0, ev = 0, pot = 0, n = 0, se_raw = 0;
};

struct SimResult {
	std::vector<Cell> cells;  // [scenario][side][class]
	long long boards = 0;
	// worst relative error of the bookkeeping identities over all boards
	double err_prob = 0;   // sum of path probabilities = all pairs
	double err_chips = 0;  // chips won by both sides = final pot
	double err_inv = 0;    // chips put in after the flop = final pot - flop pot
	double err_raw = 0;    // raw equities of both sides add up to 1
	double secs = 0;
	std::vector<double> acc;  // reduction state: [scenario][side][class][15], then boards
};

static const int ACC_N = 15;

// Reduction state -> cells.
static void finish_cells(SimResult& r, int nscn) {
	r.cells.assign((size_t)nscn * 2 * NCLASS, Cell());
	const double nb = (double)r.boards;
	for (size_t i = 0; i < r.cells.size(); i++) {
		const double* a = &r.acc[i * ACC_N];
		const double N = a[3];
		Cell& c = r.cells[i];
		if (N <= 0) continue;
		const double C = a[0] / N, P = a[1] / N, E = a[2] / N, I = a[4] / N;
		c.raw = E;
		c.pot = P;
		c.real = P > 0 ? C / P : 0;
		c.factor = E > 0 ? c.real / E : 0;
		c.ev = C - I;
		c.n = N;
		if (nb > 1) {
			// standard error of the raw equity: boards are the independent units
			const double* q = a + 5;
			double v = (q[7] - 2 * E * q[8] + E * E * q[9]) * nb / (nb - 1) / (N * N);
			c.se_raw = v > 0 ? sqrt(v) : 0;
		}
		if (C > 1e-9 && E > 1e-9 && P > 1e-9 && nb > 1) {
			// moments: a[5..14] = CC CP CE CN PP PE PN EE EN NN
			const double* q = a + 5;
			const double k1 = 1 / C, k2 = 1 / E, k3 = 1 / P;
			// sum over boards of rho^2, rho = A_C/C - A_E/E - A_P/P + N (the first-order error of log(C / (E P)))
			double s = k1 * k1 * q[0] + k2 * k2 * q[7] + k3 * k3 * q[4] + q[9]
			  - 2 * k1 * k2 * q[2] - 2 * k1 * k3 * q[1] + 2 * k1 * q[3]
			  + 2 * k2 * k3 * q[5] - 2 * k2 * q[8] - 2 * k3 * q[6];
			c.se = 0;
			double var = s * nb / (nb - 1) / (N * N);
			if (var > 0) c.se = c.factor * sqrt(var);
		}
	}
}

// Runs `nboards` random boards for the ranges ip / oop (169 class weights) in every scenario.
// seed fixes the boards; board0 is the first board number (so a run can be continued).
static SimResult simulate(const std::vector<double>& ip, const std::vector<double>& oop, const std::vector<Scn>& scns,
                          long long nboards, uint64_t seed, bool verbose, const Pol* pol = nullptr) {
	SimResult res;
	const int nscn = (int)scns.size();
	set_policy(pol ? *pol : default_pol());
	std::vector<float> wi = combo_weights(ip), wo = combo_weights(oop);
	float *d_wi, *d_wo, *d_scratch, *d_stats;
	double* d_diag;
	Scn* d_scn;
	CK(cudaMalloc(&d_wi, NC * 4));
	CK(cudaMalloc(&d_wo, NC * 4));
	CK(cudaMalloc(&d_scn, nscn * sizeof(Scn)));
	CK(cudaMemcpy(d_wi, wi.data(), NC * 4, cudaMemcpyHostToDevice));
	CK(cudaMemcpy(d_wo, wo.data(), NC * 4, cudaMemcpyHostToDevice));
	CK(cudaMemcpy(d_scn, scns.data(), nscn * sizeof(Scn), cudaMemcpyHostToDevice));
	int sms = 0, per_sm = 0;
	CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
	CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, rz_kernel, NT, 0));
	const int chunk = 1024;
	const int grid = std::max(1, std::min(chunk, sms * std::max(1, per_sm)));
	CK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));
	CK(cudaMalloc(&d_scratch, (size_t)grid * N_SLOTS * NC * 4));
	CK(cudaMalloc(&d_stats, (size_t)chunk * nscn * 2 * NCLASS * STAT_N * 4));
	CK(cudaMalloc(&d_diag, (size_t)chunk * nscn * DIAG_N * 8));
	std::vector<float> stats((size_t)chunk * nscn * 2 * NCLASS * STAT_N);
	std::vector<double> diag((size_t)chunk * nscn * DIAG_N);
	res.acc.assign((size_t)nscn * 2 * NCLASS * ACC_N, 0.0);
	auto t0 = std::chrono::steady_clock::now();
	for (long long b0 = 0; b0 < nboards; b0 += chunk) {
		const int nb = (int)std::min<long long>(chunk, nboards - b0);
		rz_kernel<<<std::min(grid, nb), NT>>>(d_scn, nscn, d_wi, d_wo, seed, b0, nb, d_scratch, d_stats, d_diag);
		CK(cudaGetLastError());
		CK(cudaDeviceSynchronize());
		CK(cudaMemcpy(stats.data(), d_stats, (size_t)nb * nscn * 2 * NCLASS * STAT_N * 4, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(diag.data(), d_diag, (size_t)nb * nscn * DIAG_N * 8, cudaMemcpyDeviceToHost));
		for (int b = 0; b < nb; b++) {
			for (int d = 0; d < nscn; d++) {
				const double* g = &diag[((size_t)b * nscn + d) * DIAG_N];
				const double pairs = g[0];
				if (pairs > 0) {
					res.err_prob = std::max(res.err_prob, fabs(g[1] - pairs) / pairs);
					res.err_raw = std::max(res.err_raw, fabs(g[5] - pairs) / pairs);
					if (g[2] > 0) res.err_chips = std::max(res.err_chips, fabs(g[3] + g[4] - g[2]) / g[2]);
					res.err_inv = std::max(res.err_inv, fabs(g[6] - (g[2] - g[7] * g[1])) / std::max(1e-9, g[2]));
				}
				for (int side = 0; side < 2; side++)
					for (int cls = 0; cls < NCLASS; cls++) {
						const float* s = &stats[((((size_t)b * nscn + d) * 2 + side) * NCLASS + cls) * STAT_N];
						double* a = &res.acc[(((size_t)d * 2 + side) * NCLASS + cls) * ACC_N];
						const double x[4] = {s[0], s[1], s[2], s[4]};  // C P E N
						a[0] += x[0]; a[1] += x[1]; a[2] += x[2]; a[3] += x[3]; a[4] += s[3];
						int k = 5;
						for (int p = 0; p < 4; p++)
							for (int q = p; q < 4; q++) a[k++] += x[p] * x[q];
					}
			}
		}
		res.boards += nb;
		if (verbose) {
			double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
			printf("  %lld/%lld boards  %.1fs\n", res.boards, nboards, secs);
			fflush(stdout);
		}
	}
	res.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	cudaFree(d_wi); cudaFree(d_wo); cudaFree(d_scn); cudaFree(d_scratch); cudaFree(d_stats); cudaFree(d_diag);
	finish_cells(res, nscn);
	return res;
}

// -- scenarios: heads-up, small blind (button, IP) against the big blind (OOP) -------------------------------
// Sizes follow tools/solve-preflop.ts: open 2.5, 3-bet 9, 4-bet 22, a raise of 70% of the stack or more is an
// all-in. Each side has put `c` in, so the flop pot is 2c. Ranges come from data/charts.json and spots.json.
#include "ranges.cuh"

struct PotDef {
	std::string key;                 // "srp", "3bet", "4bet"
	double c;                        // preflop chips each side has in
	std::vector<double> ip, oop;     // class weights
	std::string ip_from, oop_from;   // where the ranges come from
};

static std::vector<double> chart_range(const Json& charts, const std::string& key) {
	for (const Json& c : charts.at("charts").arr)
		if (c.at("key").str == key) return parse_weighted(c.at("text").str);
	throw std::runtime_error("no chart " + key);
}

static std::vector<PotDef> make_pots(const std::string& dir) {
	Json charts = json_load(dir + "/charts.json"), spots = json_load(dir + "/spots.json");
	auto spot = [&](const char* kind, const char* key, const char* part) {
		return parse_weighted(spots.at(kind).at(key).at(part).str);
	};
	std::vector<PotDef> pots;
	pots.push_back({"srp", 2.5, chart_range(charts, "SB6"), spot("vs_open", "BB|SB", "call"), "chart SB6",
	                "spots vs_open BB|SB call"});
	pots.push_back({"3bet", 9.0, spot("vs_3bet", "SB|BB", "call"),
	                range_add(spot("vs_open", "BB|SB", "bluff"), spot("vs_open", "BB|SB", "value")),
	                "spots vs_3bet SB|BB call", "spots vs_open BB|SB bluff + value"});
	pots.push_back({"4bet", 22.0, range_add(spot("vs_3bet", "SB|BB", "bluff"), spot("vs_3bet", "SB|BB", "value")),
	                spot("vs_open", "BB|SB", "value"), "spots vs_3bet SB|BB bluff + value", "spots vs_open BB|SB value"});
	return pots;
}

static const int DEPTHS[4] = {20, 40, 80, 100};

// Depths at which the pot type reaches the flop: a raise of at least 70% of the stack is all-in pre-flop.
static bool pot_reaches_flop(const PotDef& p, int depth) { return p.key == "srp" || p.key == "3bet" || p.c < 0.7 * depth; }

// Range-level means over a side's range (weights are class share x combos): raw equity, the equity-weighted
// factor (= realised share of the pot / raw equity of the whole range) and ev.
struct Summary { double raw = 0, factor = 0, ev = 0, weight = 0; };
static Summary range_summary(const SimResult& r, int scn, int side, const std::vector<double>& w) {
	Summary s;
	double sr = 0, sf = 0;
	for (int c = 0; c < NCLASS; c++) {
		const Cell& x = r.cells[((size_t)scn * 2 + side) * NCLASS + c];
		const double k = w[c] * combos_of_class(c);
		s.weight += k;
		sr += k * x.raw;
		sf += k * x.raw * x.factor;
		s.ev += k * x.ev;
	}
	if (s.weight > 0) { s.raw = sr / s.weight; s.ev /= s.weight; }
	s.factor = sr > 0 ? sf / sr : 0;
	return s;
}

// -- the old heuristic, for comparison -----------------------------------------------------------------------
// tools/solve-preflop.ts: realisedIp(eq) = clamp(0.5 + 1.08 (eq - 0.5) + 0.025) for the side in position, and
// 1 - realisedIp(1 - eq) for the other.
static double old_real(int side, double eq) {
	auto ip = [](double e) { return std::min(1.0, std::max(0.0, 0.5 + 1.08 * (e - 0.5) + 0.025)); };
	return side == 0 ? ip(eq) : 1.0 - ip(1.0 - eq);
}

struct OldDiff {
	double mean = 0, rms = 0, maxabs = 0;   // new realised share minus old, weighted by the side's range
	double mean_all = 0, rms_all = 0;       // same over all 169 classes, unweighted
	double old_factor = 0, new_factor = 0;  // equity-weighted factor of the range under each
};

// `raw` and `real` per class of one side (169 each); w = the side's range.
static OldDiff diff_old(int side, const std::vector<double>& raw, const std::vector<double>& real, const std::vector<double>& w) {
	OldDiff d;
	double sw = 0, so = 0, sn = 0, sr = 0;
	for (int c = 0; c < NCLASS; c++) {
		const double k = w[c] * combos_of_class(c);
		const double delta = real[c] - old_real(side, raw[c]);
		sw += k;
		d.mean += k * delta;
		d.rms += k * delta * delta;
		d.maxabs = std::max(d.maxabs, k > 0 ? fabs(delta) : 0.0);
		d.mean_all += delta;
		d.rms_all += delta * delta;
		sr += k * raw[c];
		so += k * old_real(side, raw[c]);
		sn += k * real[c];
	}
	if (sw > 0) { d.mean /= sw; d.rms = sqrt(d.rms / sw); }
	d.mean_all /= NCLASS;
	d.rms_all = sqrt(d.rms_all / NCLASS);
	d.old_factor = sr > 0 ? so / sr : 0;
	d.new_factor = sr > 0 ? sn / sr : 0;
	return d;
}

// -- running a pot type at several depths, results file ------------------------------------------------------------
struct PotRun {
	PotDef def;
	std::vector<int> depths;
	std::vector<Scn> scns;
	SimResult res;
};

static PotRun make_run(const PotDef& def) {
	PotRun r;
	r.def = def;
	for (int d : DEPTHS)
		if (pot_reaches_flop(def, d)) {
			r.depths.push_back(d);
			r.scns.push_back({(float)(2 * def.c), (float)(d - def.c)});
		}
	return r;
}

static PotRun run_pot(const PotDef& def, long long boards, uint64_t seed, bool verbose, const Pol* pol = nullptr) {
	PotRun r = make_run(def);
	r.res = simulate(def.ip, def.oop, r.scns, boards, seed, verbose, pol);
	return r;
}

static std::string num_array(const std::vector<double>& v, int decimals) {
	std::string s = "[";
	char buf[48];
	for (size_t i = 0; i < v.size(); i++) {
		snprintf(buf, sizeof(buf), "%.*f", decimals, v[i]);
		std::string t = buf;
		if (t.find('.') != std::string::npos) {  // trim trailing zeros to keep the file small
			while (t.back() == '0') t.pop_back();
			if (t.back() == '.') t.pop_back();
		}
		if (t == "-0") t = "0";
		s += (i ? "," : "") + t;
	}
	return s + "]";
}

static std::vector<double> cell_column(const SimResult& r, int scn, int side, double Cell::*field) {
	std::vector<double> v(NCLASS);
	for (int c = 0; c < NCLASS; c++) v[c] = r.cells[((size_t)scn * 2 + side) * NCLASS + c].*field;
	return v;
}

static void write_json(const std::string& path, const std::vector<PotRun>& runs, long long boards, uint64_t seed, const Pol& pol) {
	FILE* f = fopen(path.c_str(), "w");
	if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
	fprintf(f, "{\n\t\"schema\": \"realization/v1\",\n\t\"kind\": \"approximation\",\n");
	fprintf(f, "\t\"description\": \"Equity realisation per hand class, heads-up (small blind / button in position, big blind out of position), from a Monte-Carlo simulation of flop, turn and river with a fixed equity-driven betting policy. An approximation, not a postflop solve.\",\n");
	fprintf(f, "\t\"classes\": [");
	for (int c = 0; c < NCLASS; c++) fprintf(f, "%s\"%s\"", c ? "," : "", g_names[c].c_str());
	fprintf(f, "],\n\t\"boards_per_pot_type\": %lld,\n\t\"seed\": %llu,\n", boards, (unsigned long long)seed);
	fprintf(f, "\t\"policy\": {\"bet_pot_fraction\": [%g,%g,%g], \"oop_lead_value_share\": [%g,%g,%g], \"ip_bet_value_share\": [%g,%g,%g], "
	           "\"raise_share\": %g, \"all_in_if_bet_at_least_share_of_stack\": %g, \"raise_allowed_if_stack_after_call_at_most_pots\": %g, \"defence_scale\": %g},\n",
	        pol.bet[0], pol.bet[1], pol.bet[2], pol.vo[0], pol.vo[1], pol.vo[2], pol.vi[0], pol.vi[1], pol.vi[2], pol.rho, pol.commit, pol.raise_spr, pol.defend);
	fprintf(f, "\t\"pots\": [\n");
	for (size_t pi = 0; pi < runs.size(); pi++) {
		const PotRun& pr = runs[pi];
		fprintf(f, "\t\t{\n\t\t\t\"pot\": \"%s\",\n\t\t\t\"preflop_each_bb\": %g,\n", pr.def.key.c_str(), pr.def.c);
		fprintf(f, "\t\t\t\"ip_range_from\": \"%s\",\n\t\t\t\"oop_range_from\": \"%s\",\n", pr.def.ip_from.c_str(), pr.def.oop_from.c_str());
		fprintf(f, "\t\t\t\"ip_range\": %s,\n\t\t\t\"oop_range\": %s,\n", num_array(pr.def.ip, 4).c_str(), num_array(pr.def.oop, 4).c_str());
		fprintf(f, "\t\t\t\"identity_errors\": {\"path_probability\": %.2e, \"chips\": %.2e, \"invested\": %.2e, \"raw_equity\": %.2e},\n",
		        pr.res.err_prob, pr.res.err_chips, pr.res.err_inv, pr.res.err_raw);
		fprintf(f, "\t\t\t\"depths\": [\n");
		for (size_t di = 0; di < pr.depths.size(); di++) {
			fprintf(f, "\t\t\t\t{\n\t\t\t\t\t\"depth_bb\": %d, \"pot_bb\": %g, \"stack_bb\": %g, \"spr\": %.3f,\n", pr.depths[di], pr.scns[di].pot,
			        pr.scns[di].stack, pr.scns[di].stack / pr.scns[di].pot);
			for (int side = 0; side < 2; side++) {
				const auto& w = side ? pr.def.oop : pr.def.ip;
				const Summary m = range_summary(pr.res, (int)di, side, w);
				const OldDiff dd = diff_old(side, cell_column(pr.res, (int)di, side, &Cell::raw), cell_column(pr.res, (int)di, side, &Cell::real), w);
				fprintf(f, "\t\t\t\t\t\"%s\": {\n", side ? "oop" : "ip");
				fprintf(f, "\t\t\t\t\t\t\"range_summary\": {\"combos\": %.0f, \"raw\": %.4f, \"factor\": %.4f, \"ev_bb\": %.3f, \"old_formula_factor\": %.4f, \"mean_share_change_vs_old\": %.4f, \"rms_share_change_vs_old\": %.4f},\n",
				        m.weight, m.raw, m.factor, m.ev, dd.old_factor, dd.mean, dd.rms);
				const struct { const char* name; double Cell::*field; int decimals; } cols[] = {
					{"raw", &Cell::raw, 5}, {"real", &Cell::real, 5}, {"factor", &Cell::factor, 5}, {"se", &Cell::se, 5},
					{"ev_bb", &Cell::ev, 4}, {"final_pot_bb", &Cell::pot, 3}};
				const size_t ncols = sizeof(cols) / sizeof(cols[0]);
				for (size_t k = 0; k < ncols; k++)
					fprintf(f, "\t\t\t\t\t\t\"%s\": %s%s\n", cols[k].name, num_array(cell_column(pr.res, (int)di, side, cols[k].field), cols[k].decimals).c_str(),
					        k + 1 < ncols ? "," : "");
				fprintf(f, "\t\t\t\t\t}%s\n", side ? "" : ",");
			}
			fprintf(f, "\t\t\t\t}%s\n", di + 1 < pr.depths.size() ? "," : "");
		}
		fprintf(f, "\t\t\t]\n\t\t}%s\n", pi + 1 < runs.size() ? "," : "");
	}
	fprintf(f, "\t]\n}\n");
	fclose(f);
}

// Reduction state of a pot run, so an interrupted `run` resumes with the pot types it has finished.
static void save_state(const std::string& path, const PotRun& r, long long boards, uint64_t seed) {
	FILE* f = fopen(path.c_str(), "wb");
	if (!f) return;
	long long hdr[3] = {boards, (long long)seed, (long long)r.res.acc.size()};
	fwrite(hdr, sizeof(hdr), 1, f);
	double errs[5] = {r.res.err_prob, r.res.err_chips, r.res.err_inv, r.res.err_raw, r.res.secs};
	fwrite(errs, sizeof(errs), 1, f);
	fwrite(r.res.acc.data(), sizeof(double), r.res.acc.size(), f);
	fclose(f);
}

static bool load_state(const std::string& path, PotRun& r, long long boards, uint64_t seed) {
	FILE* f = fopen(path.c_str(), "rb");
	if (!f) return false;
	long long hdr[3];
	double errs[5];
	bool ok = fread(hdr, sizeof(hdr), 1, f) == 1 && hdr[0] == boards && hdr[1] == (long long)seed &&
	          hdr[2] == (long long)(r.scns.size() * 2 * NCLASS * ACC_N) && fread(errs, sizeof(errs), 1, f) == 1;
	if (ok) {
		r.res.acc.resize(hdr[2]);
		ok = fread(r.res.acc.data(), sizeof(double), hdr[2], f) == (size_t)hdr[2];
	}
	fclose(f);
	if (!ok) return false;
	r.res.boards = boards;
	r.res.err_prob = errs[0]; r.res.err_chips = errs[1]; r.res.err_inv = errs[2]; r.res.err_raw = errs[3]; r.res.secs = errs[4];
	finish_cells(r.res, (int)r.scns.size());
	return true;
}

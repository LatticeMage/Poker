// Range against range on every flop: for a heads-up matchup (two preflop ranges) and each of the 1,755 flop
// classes, the exact all-in equity of every combo of one range against the other range (every turn and river,
// card removal exact), and from that
//   - the equity of the aggressor's range and of the caller's range (they add up to 1: a check),
//   - how each range's equity is spread: the share of the range in ten equity bins (0-10%, ..., 90-100%).
// The share of a range above 80% is its "nut" part; aggressor minus caller is the nut advantage.
//
// These are the numbers under the flop groups of the decision tree (flops.cuh, flop_shape): a group's value is the
// weighted mean over its flops, and how much the flops of one group differ says how good the grouping is.
//
// Layout of a result: FS_K doubles per (matchup, flop class) = eq_aggressor, eq_caller, 10 bins aggressor,
// 10 bins caller. Flop classes in all_flops() order.
//
// Cost: per flop, 1,176 runouts x (combos of A x combos of B) comparisons, twice (once per side). The biggest
// matchup (574 x 592 combos) is about 1.4e12 comparisons over the 1,755 flops.
#pragma once

#include "dispatch.cuh"
#include "flops.cuh"
#include "ranges.cuh"

static const int FS_R = 1176;  // C(49,2): turn and river from the 49 cards not on the flop
static const int FS_BINS = 10;
static const int FS_K = 2 + 2 * FS_BINS;
static const int FS_BATCH = 64;  // flops per batch

__constant__ uint8_t c_fs_pair[FS_R][2];

static inline void fs_setup() {
	static bool done = false;
	if (done) return;
	done = true;
	uint8_t pair[FS_R][2];
	int n = 0;
	for (int a = 0; a < 49; a++)
		for (int b = a + 1; b < 49; b++) { pair[n][0] = a; pair[n][1] = b; n++; }
	CK(cudaMemcpyToSymbol(c_fs_pair, pair, sizeof(pair)));
}

// rank[(f * N + k) * FS_R + r] = strength of combo k on flop f with runout r, or 0 if the combo uses a board card.
// flops: 3 cards each, ascending.
__global__ void fs_rank_kernel(const uint8_t* flops, const uint8_t* combos, int N, int F, uint32_t* rank) {
	const uint64_t id = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	if (id >= (uint64_t)F * N * FS_R) return;
	const int r = (int)(id % FS_R);
	const int k = (int)((id / FS_R) % N);
	const int f = (int)(id / ((uint64_t)FS_R * N));
	const uint8_t* fl = flops + f * 3;
	// the r-th pair of the 49 cards that are not on the flop
	int t = c_fs_pair[r][0], u = c_fs_pair[r][1];
#pragma unroll
	for (int x = 0; x < 3; x++) {
		if (t >= fl[x]) t++;
		if (u >= fl[x]) u++;
	}
	const int h0 = combos[k * 2], h1 = combos[k * 2 + 1];
	uint32_t out = 0;
	if (h0 != fl[0] && h0 != fl[1] && h0 != fl[2] && h0 != t && h0 != u && h1 != fl[0] && h1 != fl[1] && h1 != fl[2] && h1 != t && h1 != u) {
		const int c[7] = {fl[0], fl[1], fl[2], t, u, h0, h1};
		out = eval7(c) + 1;
	}
	rank[id] = out;
}

// One thread per (flop, combo i of this side): against every combo j of the other side, over every runout.
// win[f * N + i] = sum over j of w_j * (wins + ties / 2), tot[...] = sum over j of w_j * runouts both hands are live in.
__global__ void fs_compare_kernel(const uint32_t* rank_me, const uint32_t* rank_them, const uint8_t* comb_me, const uint8_t* comb_them,
                                  const double* w_them, int N, int M, int F, double* win, double* tot) {
	const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= (uint32_t)(F * N)) return;
	const int i = id % N, f = id / N;
	const int a0 = comb_me[i * 2], a1 = comb_me[i * 2 + 1];
	const uint32_t* ra = rank_me + ((uint64_t)f * N + i) * FS_R;
	double w_sum = 0, t_sum = 0;
	for (int j = 0; j < M; j++) {
		const int b0 = comb_them[j * 2], b1 = comb_them[j * 2 + 1];
		if (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1) continue;
		const uint32_t* rb = rank_them + ((uint64_t)f * M + j) * FS_R;
		int twice = 0, n = 0;  // twice: 2 per win, 1 per tie
		for (int r = 0; r < FS_R; r++) {
			const uint32_t a = ra[r], b = rb[r];
			const int live = (a != 0) & (b != 0);
			n += live;
			twice += live * (2 * (a > b) + (a == b));
		}
		w_sum += w_them[j] * 0.5 * twice;
		t_sum += w_them[j] * n;
	}
	win[id] = w_sum;
	tot[id] = t_sum;
}

// -- host ---------------------------------------------------------------------------------------------------------
struct FsMatchup {
	std::string key;                 // "srp:BTN:BB"
	std::vector<double> aggressor;   // 169 class weights
	std::vector<double> caller;
};

static inline std::vector<FsMatchup> fs_load_matchups(const std::string& path) {
	const Json j = json_load(path);
	std::vector<FsMatchup> out;
	for (const Json& m : j.arr) {
		FsMatchup x;
		x.key = m.at("key").str;
		for (const Json& v : m.at("aggressor").arr) x.aggressor.push_back(v.num);
		for (const Json& v : m.at("caller").arr) x.caller.push_back(v.num);
		if (x.aggressor.size() != NCLASS || x.caller.size() != NCLASS) throw std::runtime_error("matchup " + x.key + ": expected 169 weights");
		out.push_back(x);
	}
	return out;
}

// The combos of a range (every combo of a class with weight > 0) and their weights.
struct FsSide {
	std::vector<uint8_t> cards;
	std::vector<double> w;
	int n = 0;
};
static inline FsSide fs_side(const std::vector<double>& cls) {
	FsSide s;
	for (int c = 0; c < NCLASS; c++) {
		if (cls[c] <= 0) continue;
		for (int k = 0; k < g_ncombo[c]; k++) {
			s.cards.push_back(g_combos[c][k][0]);
			s.cards.push_back(g_combos[c][k][1]);
			s.w.push_back(cls[c]);
		}
	}
	s.n = (int)s.w.size();
	return s;
}

// From the per-combo sums of one side: the range's equity and its ten bins. A combo counts with its weight times
// the weight of the opponent hands it can meet (tot), which is how often it is actually held on this board.
static inline void fs_reduce(const FsSide& s, const double* win, const double* tot, double& eq, double* bins) {
	double W = 0, T = 0;
	for (int b = 0; b < FS_BINS; b++) bins[b] = 0;
	for (int i = 0; i < s.n; i++) {
		if (tot[i] <= 0) continue;
		const double mass = s.w[i] * tot[i];
		W += s.w[i] * win[i];
		T += mass;
		const double e = win[i] / tot[i];
		bins[std::min(FS_BINS - 1, (int)(e * FS_BINS + 1e-9))] += mass;
	}
	eq = T > 0 ? W / T : 0.5;
	for (int b = 0; b < FS_BINS; b++) bins[b] = T > 0 ? bins[b] / T : 0;
}

// Device work for `count` flops (3 ascending cards each) of one matchup; writes FS_K doubles per flop to out.
struct FsDevice {
	int cap_n = 0;
	uint8_t *flops = nullptr, *ca = nullptr, *cb = nullptr;
	uint32_t *ra = nullptr, *rb = nullptr;
	double *wa = nullptr, *wb = nullptr, *win = nullptr, *tot = nullptr;
	std::vector<double> h_win, h_tot;

	void reserve(int n) {  // n = the largest number of combos a side can have
		if (n <= cap_n) return;
		release();
		cap_n = n;
		CK(cudaMalloc(&flops, FS_BATCH * 3));
		CK(cudaMalloc(&ca, (size_t)n * 2));
		CK(cudaMalloc(&cb, (size_t)n * 2));
		CK(cudaMalloc(&ra, (size_t)FS_BATCH * n * FS_R * 4));
		CK(cudaMalloc(&rb, (size_t)FS_BATCH * n * FS_R * 4));
		CK(cudaMalloc(&wa, (size_t)n * 8));
		CK(cudaMalloc(&wb, (size_t)n * 8));
		CK(cudaMalloc(&win, (size_t)FS_BATCH * n * 8));
		CK(cudaMalloc(&tot, (size_t)FS_BATCH * n * 8));
		h_win.resize((size_t)FS_BATCH * n);
		h_tot.resize((size_t)FS_BATCH * n);
	}
	void release() {
		if (!cap_n) return;
		cudaFree(flops); cudaFree(ca); cudaFree(cb); cudaFree(ra); cudaFree(rb); cudaFree(wa); cudaFree(wb); cudaFree(win); cudaFree(tot);
		cap_n = 0;
	}

	void run(const FsSide& A, const FsSide& B, const uint8_t* flop_cards, int count, double* out) {
		fs_setup();
		reserve(std::max(A.n, B.n));
		CK(cudaMemcpy(flops, flop_cards, (size_t)count * 3, cudaMemcpyHostToDevice));
		CK(cudaMemcpy(ca, A.cards.data(), (size_t)A.n * 2, cudaMemcpyHostToDevice));
		CK(cudaMemcpy(cb, B.cards.data(), (size_t)B.n * 2, cudaMemcpyHostToDevice));
		CK(cudaMemcpy(wa, A.w.data(), (size_t)A.n * 8, cudaMemcpyHostToDevice));
		CK(cudaMemcpy(wb, B.w.data(), (size_t)B.n * 8, cudaMemcpyHostToDevice));
		const uint64_t na = (uint64_t)count * A.n * FS_R, nb = (uint64_t)count * B.n * FS_R;
		fs_rank_kernel<<<(unsigned)((na + 255) / 256), 256>>>(flops, ca, A.n, count, ra);
		fs_rank_kernel<<<(unsigned)((nb + 255) / 256), 256>>>(flops, cb, B.n, count, rb);
		// the aggressor's combos against the caller's range
		fs_compare_kernel<<<(count * A.n + 127) / 128, 128>>>(ra, rb, ca, cb, wb, A.n, B.n, count, win, tot);
		CK(cudaMemcpy(h_win.data(), win, (size_t)count * A.n * 8, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(h_tot.data(), tot, (size_t)count * A.n * 8, cudaMemcpyDeviceToHost));
		for (int f = 0; f < count; f++) fs_reduce(A, &h_win[(size_t)f * A.n], &h_tot[(size_t)f * A.n], out[f * FS_K], &out[f * FS_K + 2]);
		// and the caller's combos against the aggressor's range
		fs_compare_kernel<<<(count * B.n + 127) / 128, 128>>>(rb, ra, cb, ca, wa, B.n, A.n, count, win, tot);
		CK(cudaMemcpy(h_win.data(), win, (size_t)count * B.n * 8, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(h_tot.data(), tot, (size_t)count * B.n * 8, cudaMemcpyDeviceToHost));
		for (int f = 0; f < count; f++)
			fs_reduce(B, &h_win[(size_t)f * B.n], &h_tot[(size_t)f * B.n], out[f * FS_K + 1], &out[f * FS_K + 2 + FS_BINS]);
		CK(cudaGetLastError());
	}
};

static inline uint64_t fs_tag(const std::vector<FsMatchup>& ms) {
	uint64_t h = relay_hash_u64(relay_hash_u64(relay_hash_u64(7, FS_K), FS_R), ms.size());
	for (const FsMatchup& m : ms) {
		h = relay_hash(h, m.key.data(), m.key.size());
		h = relay_hash(h, m.aggressor.data(), NCLASS * 8);
		h = relay_hash(h, m.caller.data(), NCLASS * 8);
	}
	return h;
}

// All matchups x all 1,755 flops, as one relay: results[(m * 1755 + flop index) * FS_K ...]. An item is one flop of
// one matchup; a batch (FS_BATCH items) may span two matchups, the launch splits it.
static inline std::vector<double> fs_run(const std::vector<FsMatchup>& ms, const std::string& relay_path, bool verbose = false,
                                         uint64_t stop_after = 0, bool return_on_stop = false, uint64_t* done = nullptr) {
	const std::vector<Flop>& flops = all_flops();
	const uint64_t NF = flops.size(), total = ms.size() * NF;
	std::vector<double> res(total * FS_K, 0.0);
	std::vector<FsSide> A, B;
	for (const FsMatchup& m : ms) { A.push_back(fs_side(m.aggressor)); B.push_back(fs_side(m.caller)); }
	FsDevice dev;
	Relay relay;
	relay.path = relay_path;
	relay.tag = fs_tag(ms);
	relay.stop_after = stop_after;
	relay.return_on_stop = return_on_stop;
	relay.save_state = [&](FILE* f) { fwrite(res.data(), 8, res.size(), f); };
	relay.load_state = [&](FILE* f) { return fread(res.data(), 8, res.size(), f) == res.size(); };
	dispatch(total, FS_BATCH, [&](uint64_t first, uint64_t count) {
		uint64_t at = first;
		while (at < first + count) {
			const uint64_t m = at / NF;
			const uint64_t end = std::min<uint64_t>(first + count, (m + 1) * NF);
			const int n = (int)(end - at);
			uint8_t cards[FS_BATCH * 3];
			for (int k = 0; k < n; k++)
				for (int x = 0; x < 3; x++) cards[k * 3 + x] = (uint8_t)flops[at % NF + k].cards[x];
			dev.run(A[m], B[m], cards, n, &res[at * FS_K]);
			at = end;
		}
	}, "flops", verbose, &relay);
	if (done) *done = relay.done;
	dev.release();
	return res;
}

// -- groups -------------------------------------------------------------------------------------------------------
// group[flop index] = number of the flop's group (0-based) at a level of flop_shape (1..7); returns the group names.
static inline std::vector<std::string> fs_groups(int depth, std::vector<int>& group) {
	const std::vector<Flop>& all = all_flops();
	group.assign(all.size(), -1);
	std::vector<std::string> names;
	for (int i : flop_shape_order()) {
		const std::string n = flop_shape_name(all[i], depth);
		if (names.empty() || names.back() != n) names.push_back(n);
		group[i] = (int)names.size() - 1;
	}
	return names;
}

// The share of a range at or above an equity (a multiple of 10%), from its bins.
static inline double fs_share_from(const double* bins, int first_bin) {
	double s = 0;
	for (int b = first_bin; b < FS_BINS; b++) s += bins[b];
	return s;
}

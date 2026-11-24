// Fused CUDA kernel for the 169x169 hand-class equity matrix.
//
//   nvcc -O3 -arch=sm_89 -o equity equity.cu
//   ./equity pair AKs QQ 8000000          equity of one class pair
//   ./equity matrix 1000000 out.bin       all 14,196 pairs; out.bin = 169*169 float64, row-major
//   ./equity exact out.bin [A B]          every combo pair on every board: no sampling noise
//   ./equity selftest [N]                 GPU evaluator vs a slow host reference on random deals
//
// Cards are 0..51 = rank index (0 = deuce) * 4 + suit, as in src/lib/poker/ranges.ts. Class order is
// ALL_HANDS from ranges.ts (AA, AKs, AKo, AQs, ..., 22), so the matrix lines up with the TypeScript tools.
// Each sample: a random combo of each class (skipped if they share a card), a random board from the other
// 48 cards, both hands scored by a branch-light bit-mask evaluator. Equity = (wins + ties/2) / valid.
//
// Scores are category << 20 | tie-break nibbles; only the ordering matters.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                              \
	do {                                                                                   \
		cudaError_t e_ = (x);                                                              \
		if (e_ != cudaSuccess) {                                                           \
			fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
			exit(1);                                                                       \
		}                                                                                  \
	} while (0)

static const int NCLASS = 169;
__constant__ uint8_t c_combos[NCLASS][12][2];
__constant__ uint8_t c_ncombo[NCLASS];

__device__ __forceinline__ int hi_bit(uint32_t m) { return 31 - __clz(m); }  // m != 0

__device__ __forceinline__ uint32_t topk(uint32_t m, int k) {
	uint32_t v = 0;
	for (int i = 0; i < k; i++) {
		if (m) {
			int h = hi_bit(m);
			v = (v << 4) | h;
			m ^= 1u << h;
		} else {
			v <<= 4;
		}
	}
	return v;
}

// Top rank + 1 of the best straight in a 13-bit rank mask, or 0 (the ace also plays low).
__device__ __forceinline__ int straight_top(uint32_t m) {
	uint32_t e = (m << 1) | ((m >> 12) & 1);
	uint32_t x = e & (e >> 1) & (e >> 2) & (e >> 3) & (e >> 4);
	return x ? hi_bit(x) + 4 : 0;
}

__device__ uint32_t eval7(const int* c) {
	uint32_t m1 = 0, m2 = 0, m3 = 0, m4 = 0, sm[4] = {0, 0, 0, 0};
#pragma unroll
	for (int i = 0; i < 7; i++) {
		uint32_t b = 1u << (c[i] >> 2);
		m4 |= m3 & b;
		m3 |= m2 & b;
		m2 |= m1 & b;
		m1 |= b;
		sm[c[i] & 3] |= b;
	}
	const uint32_t q = m4, t = m3 & ~m4, p = m2 & ~m3, rm = m1;
	uint32_t fm = 0;
#pragma unroll
	for (int s = 0; s < 4; s++)
		if (__popc(sm[s]) >= 5) fm = sm[s];
	if (fm) {
		int s = straight_top(fm);
		if (s) return (8u << 20) | ((s - 1) << 16);
	}
	if (q) {
		int qh = hi_bit(q);
		return (7u << 20) | (qh << 16) | (hi_bit(rm & ~(1u << qh)) << 12);
	}
	int th = 0;
	if (t) {
		th = hi_bit(t);
		uint32_t pc = (t & ~(1u << th)) | p;
		if (pc) return (6u << 20) | (th << 16) | (hi_bit(pc) << 12);
	}
	if (fm) return (5u << 20) | topk(fm, 5);
	int s = straight_top(rm);
	if (s) return (4u << 20) | ((s - 1) << 16);
	if (t) return (3u << 20) | (th << 16) | (topk(rm & ~(1u << th), 2) << 8);
	if (__popc(p) >= 2) {
		int h1 = hi_bit(p);
		int h2 = hi_bit(p & ~(1u << h1));
		int kick = hi_bit(rm & ~((1u << h1) | (1u << h2)));
		return (2u << 20) | (h1 << 16) | (h2 << 12) | (kick << 8);
	}
	if (p) {
		int ph = hi_bit(p);
		return (1u << 20) | (ph << 16) | (topk(rm & ~(1u << ph), 3) << 4);
	}
	return topk(rm, 5);
}

__device__ __forceinline__ uint64_t mix64(uint64_t z) {
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}

struct Rng {
	uint64_t s;
	__device__ __forceinline__ uint32_t next() {
		s += 0x9E3779B97F4A7C15ull;
		return (uint32_t)(mix64(s) >> 32);
	}
	__device__ __forceinline__ uint32_t below(uint32_t n) { return (uint32_t)(((uint64_t)next() * n) >> 32); }
};

__global__ void matchup_kernel(const int2* pairs, int per_thread, uint64_t seed, unsigned long long* wins2,
                               unsigned long long* valid) {
	const int pi = blockIdx.y;
	const int a = pairs[pi].x, b = pairs[pi].y;
	const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	Rng r;
	r.s = mix64(seed ^ mix64(((uint64_t)a * NCLASS + b) * 0xD1B54A32D192ED03ull) ^ mix64(tid + 0x1234567ull));
	const uint32_t na = c_ncombo[a], nb = c_ncombo[b];
	unsigned long long w2 = 0, v = 0;
	for (int s = 0; s < per_thread; s++) {
		const uint32_t ia = r.below(na), ib = r.below(nb);
		const int a0 = c_combos[a][ia][0], a1 = c_combos[a][ia][1];
		const int b0 = c_combos[b][ib][0], b1 = c_combos[b][ib][1];
		if (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1) continue;
		uint64_t used = (1ull << a0) | (1ull << a1) | (1ull << b0) | (1ull << b1);
		int ha[7], hb[7];
		ha[0] = a0; ha[1] = a1; hb[0] = b0; hb[1] = b1;
#pragma unroll
		for (int k = 0; k < 5; k++) {
			int c;
			do { c = r.below(52); } while ((used >> c) & 1);
			used |= 1ull << c;
			ha[2 + k] = c;
			hb[2 + k] = c;
		}
		const uint32_t sa = eval7(ha), sb = eval7(hb);
		w2 += sa > sb ? 2 : (sa == sb ? 1 : 0);
		v++;
	}
	for (int off = 16; off > 0; off >>= 1) {
		w2 += __shfl_down_sync(0xffffffffu, w2, off);
		v += __shfl_down_sync(0xffffffffu, v, off);
	}
	if ((threadIdx.x & 31) == 0) {
		atomicAdd(&wins2[pi], w2);
		atomicAdd(&valid[pi], v);
	}
}

__global__ void test_kernel(const uint8_t* deals, int n, int8_t* out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const uint8_t* d = deals + 9 * i;
	int ha[7], hb[7];
	ha[0] = d[0]; ha[1] = d[1]; hb[0] = d[2]; hb[1] = d[3];
	for (int k = 0; k < 5; k++) ha[2 + k] = hb[2 + k] = d[4 + k];
	uint32_t sa = eval7(ha), sb = eval7(hb);
	out[i] = sa > sb ? 1 : (sa < sb ? -1 : 0);
}


// -- exact enumeration --------------------------------------------------------------------------
// Every non-conflicting (combo of a, combo of b) pair is a job; each job is split into CHUNKS slices of the
// C(48,5) boards, one slice per thread. Board k is the k-th 5-subset of the 48 free cards in lexicographic order.
static const int CHUNKS = 512;
static const uint32_t TOTAL_BOARDS = 1712304;  // C(48,5)
__constant__ uint32_t c_binom[49][6];

__global__ void exact_kernel(const uint32_t* jobs, const int2* pairs, uint64_t first, uint64_t njobs,
                             unsigned long long* wins2) {
	const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= njobs * CHUNKS) return;
	const uint32_t job = jobs[first + tid / CHUNKS];
	const uint32_t chunk = (uint32_t)(tid % CHUNKS);
	const int pi = job >> 8, ia = (job >> 4) & 15, ib = job & 15;
	const int a = pairs[pi].x, b = pairs[pi].y;
	const int a0 = c_combos[a][ia][0], a1 = c_combos[a][ia][1];
	const int b0 = c_combos[b][ib][0], b1 = c_combos[b][ib][1];
	const uint64_t used = (1ull << a0) | (1ull << a1) | (1ull << b0) | (1ull << b1);
	uint8_t rem[48];
	int n = 0;
	for (int c = 0; c < 52; c++)
		if (!((used >> c) & 1)) rem[n++] = c;
	const uint32_t start = (uint32_t)((uint64_t)chunk * TOTAL_BOARDS / CHUNKS);
	const uint32_t end = (uint32_t)((uint64_t)(chunk + 1) * TOTAL_BOARDS / CHUNKS);
	// unrank `start`
	int idx[5];
	uint32_t r = start;
	int x = 0;
	for (int i = 0; i < 5; i++) {
		int v = x;
		for (;; v++) {
			uint32_t cnt = c_binom[47 - v][4 - i];
			if (r < cnt) break;
			r -= cnt;
		}
		idx[i] = v;
		x = v + 1;
	}
	int ha[7], hb[7];
	ha[0] = a0; ha[1] = a1; hb[0] = b0; hb[1] = b1;
	unsigned long long w2 = 0;
	for (uint32_t k = start; k < end; k++) {
#pragma unroll
		for (int j = 0; j < 5; j++) {
			ha[2 + j] = rem[idx[j]];
			hb[2 + j] = ha[2 + j];
		}
		const uint32_t sa = eval7(ha), sb = eval7(hb);
		w2 += sa > sb ? 2 : (sa == sb ? 1 : 0);
		int i = 4;
		while (i >= 0 && idx[i] == 43 + i) i--;
		if (i < 0) break;
		idx[i]++;
		for (int j = i + 1; j < 5; j++) idx[j] = idx[j - 1] + 1;
	}
	atomicAdd(&wins2[pi], w2);
}

// -- host ---------------------------------------------------------------------------------------
static const char* RANKS = "23456789TJQKA";
static std::vector<std::string> g_names;
static uint8_t g_combos[NCLASS][12][2];
static uint8_t g_ncombo[NCLASS];

static void setup_classes() {
	uint8_t combos[NCLASS][12][2] = {};
	uint8_t n[NCLASS] = {};
	auto add = [&](int idx, int c0, int c1) {
		combos[idx][n[idx]][0] = c0;
		combos[idx][n[idx]][1] = c1;
		n[idx]++;
	};
	int idx = 0;
	for (int hi = 12; hi >= 0; hi--) {
		for (int lo = hi; lo >= 0; lo--) {
			if (hi == lo) {
				g_names.push_back(std::string(2, RANKS[hi]));
				for (int a = 0; a < 4; a++)
					for (int b = a + 1; b < 4; b++) add(idx, hi * 4 + a, hi * 4 + b);
				idx++;
			} else {
				g_names.push_back(std::string() + RANKS[hi] + RANKS[lo] + "s");
				for (int s = 0; s < 4; s++) add(idx, hi * 4 + s, lo * 4 + s);
				idx++;
				g_names.push_back(std::string() + RANKS[hi] + RANKS[lo] + "o");
				for (int a = 0; a < 4; a++)
					for (int b = 0; b < 4; b++)
						if (a != b) add(idx, hi * 4 + a, lo * 4 + b);
				idx++;
			}
		}
	}
	memcpy(g_combos, combos, sizeof(combos));
	memcpy(g_ncombo, n, sizeof(n));
	CK(cudaMemcpyToSymbol(c_combos, combos, sizeof(combos)));
	CK(cudaMemcpyToSymbol(c_ncombo, n, sizeof(n)));
}

static int class_index(const std::string& name) {
	for (int i = 0; i < NCLASS; i++)
		if (g_names[i] == name) return i;
	fprintf(stderr, "unknown hand class %s\n", name.c_str());
	exit(1);
}


// -- slow reference for the self test (host only) -----------------------------------------------
// Independent of eval7: ranks 5-card hands by sorting rank groups, then takes the best of the 21 subsets.
static uint64_t eval5_ref(const int* c) {
	int r[5], su[5];
	for (int i = 0; i < 5; i++) { r[i] = c[i] >> 2; su[i] = c[i] & 3; }
	bool flush = true;
	for (int i = 1; i < 5; i++) flush = flush && su[i] == su[0];
	int cnt[13] = {};
	for (int i = 0; i < 5; i++) cnt[r[i]]++;
	int grank[5], gcnt[5], ng = 0;
	for (int k = 5; k >= 1; k--)  // groups by size desc, then rank desc
		for (int rk = 12; rk >= 0; rk--)
			if (cnt[rk] == k) { grank[ng] = rk; gcnt[ng] = k; ng++; }
	int distinct = 0, hi = -1, lo = 13;
	for (int rk = 0; rk < 13; rk++)
		if (cnt[rk]) { distinct++; hi = rk; if (lo == 13) lo = rk; }
	int straight_top = -1;
	if (distinct == 5) {
		if (hi - lo == 4) straight_top = hi;
		else if (hi == 12 && cnt[0] && cnt[1] && cnt[2] && cnt[3]) straight_top = 3;
	}
	auto pack = [&](int n) { uint64_t v = 0; for (int i = 0; i < n; i++) v = v << 4 | grank[i]; return v; };
	if (straight_top >= 0 && flush) return 8ull << 40 | straight_top;
	if (gcnt[0] == 4) return 7ull << 40 | pack(ng);
	if (gcnt[0] == 3 && gcnt[1] == 2) return 6ull << 40 | pack(ng);
	if (flush) return 5ull << 40 | pack(ng);
	if (straight_top >= 0) return 4ull << 40 | straight_top;
	if (gcnt[0] == 3) return 3ull << 40 | pack(ng);
	if (gcnt[0] == 2 && gcnt[1] == 2) return 2ull << 40 | pack(ng);
	if (gcnt[0] == 2) return 1ull << 40 | pack(ng);
	return pack(ng);
}

static uint64_t eval7_ref(const int* c) {
	uint64_t best = 0;
	for (int skip1 = 0; skip1 < 7; skip1++)
		for (int skip2 = skip1 + 1; skip2 < 7; skip2++) {
			int five[5], n = 0;
			for (int i = 0; i < 7; i++)
				if (i != skip1 && i != skip2) five[n++] = c[i];
			uint64_t v = eval5_ref(five);
			if (v > best) best = v;
		}
	return best;
}

static uint64_t g_rng = 88172645463325252ull;
static uint32_t rnd(uint32_t n) {
	g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
	return (uint32_t)(((g_rng >> 32) * n) >> 32);
}

// Random deals (a0 a1 b0 b1 board x5), all distinct cards, drawn from `deck`.
static std::vector<uint8_t> random_deals(int n, const std::vector<int>& deck) {
	std::vector<uint8_t> out(9 * (size_t)n);
	for (int i = 0; i < n; i++) {
		std::vector<int> d = deck;
		for (int k = 0; k < 9; k++) {
			int j = k + rnd((uint32_t)d.size() - k);
			std::swap(d[k], d[j]);
			out[9 * (size_t)i + k] = (uint8_t)d[k];
		}
	}
	return out;
}

static long long selftest_deals(const std::vector<uint8_t>& deals, int n) {
	uint8_t* d_in;
	int8_t* d_out;
	CK(cudaMalloc(&d_in, deals.size()));
	CK(cudaMalloc(&d_out, n));
	CK(cudaMemcpy(d_in, deals.data(), deals.size(), cudaMemcpyHostToDevice));
	test_kernel<<<(n + 255) / 256, 256>>>(d_in, n, d_out);
	CK(cudaGetLastError());
	std::vector<int8_t> got(n);
	CK(cudaMemcpy(got.data(), d_out, n, cudaMemcpyDeviceToHost));
	cudaFree(d_in);
	cudaFree(d_out);
	long long bad = 0;
	for (int i = 0; i < n; i++) {
		const uint8_t* d = &deals[9 * (size_t)i];
		int ha[7], hb[7];
		ha[0] = d[0]; ha[1] = d[1]; hb[0] = d[2]; hb[1] = d[3];
		for (int k = 0; k < 5; k++) ha[2 + k] = hb[2 + k] = d[4 + k];
		uint64_t ra = eval7_ref(ha), rb = eval7_ref(hb);
		int ref = ra > rb ? 1 : (ra < rb ? -1 : 0);
		bad += ref != got[i];
	}
	return bad;
}

static const int BLOCK = 256;
static const int BLOCKS_PER_PAIR = 64;

// Runs `pairs` for about `samples` samples each; returns equity per pair and valid counts.
static void run_pairs(const std::vector<int2>& pairs, long long samples, uint64_t seed, std::vector<double>& eq,
                      std::vector<unsigned long long>& valid_out) {
	const int P = (int)pairs.size();
	const int per_thread = (int)((samples + (long long)BLOCKS_PER_PAIR * BLOCK - 1) / ((long long)BLOCKS_PER_PAIR * BLOCK));
	int2* d_pairs;
	unsigned long long *d_w, *d_v;
	CK(cudaMalloc(&d_pairs, P * sizeof(int2)));
	CK(cudaMalloc(&d_w, P * sizeof(unsigned long long)));
	CK(cudaMalloc(&d_v, P * sizeof(unsigned long long)));
	CK(cudaMemcpy(d_pairs, pairs.data(), P * sizeof(int2), cudaMemcpyHostToDevice));
	CK(cudaMemset(d_w, 0, P * sizeof(unsigned long long)));
	CK(cudaMemset(d_v, 0, P * sizeof(unsigned long long)));
	dim3 grid(BLOCKS_PER_PAIR, P);
	matchup_kernel<<<grid, BLOCK>>>(d_pairs, per_thread, seed, d_w, d_v);
	CK(cudaGetLastError());
	CK(cudaDeviceSynchronize());
	std::vector<unsigned long long> w(P), v(P);
	CK(cudaMemcpy(w.data(), d_w, P * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
	CK(cudaMemcpy(v.data(), d_v, P * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
	cudaFree(d_pairs); cudaFree(d_w); cudaFree(d_v);
	eq.resize(P);
	valid_out = v;
	for (int i = 0; i < P; i++) eq[i] = v[i] ? (double)w[i] / (2.0 * (double)v[i]) : 0.5;
}

int main(int argc, char** argv) {
	if (argc < 2) { fprintf(stderr, "usage: equity pair|matrix|test ...\n"); return 1; }
	setup_classes();
	std::string mode = argv[1];
	if (mode == "pair" && argc >= 5) {
		int a = class_index(argv[2]), b = class_index(argv[3]);
		long long samples = atoll(argv[4]);
		std::vector<double> eq;
		std::vector<unsigned long long> v;
		auto t0 = std::chrono::steady_clock::now();
		run_pairs({make_int2(a, b)}, samples, 12345, eq, v);
		double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		printf("%s vs %s: %.5f  (%llu valid samples, %.2fs)\n", argv[2], argv[3], eq[0], v[0], secs);
	} else if (mode == "matrix" && argc >= 4) {
		long long samples = atoll(argv[2]);
		std::vector<double> m(NCLASS * NCLASS, 0.5);
		std::vector<int2> all;
		for (int i = 0; i < NCLASS; i++)
			for (int j = i + 1; j < NCLASS; j++) all.push_back(make_int2(i, j));
		const int PB = 256;
		auto t0 = std::chrono::steady_clock::now();
		for (size_t s = 0; s < all.size(); s += PB) {
			std::vector<int2> chunk(all.begin() + s, all.begin() + std::min(all.size(), s + PB));
			std::vector<double> eq;
			std::vector<unsigned long long> v;
			run_pairs(chunk, samples, 20261124, eq, v);
			for (size_t k = 0; k < chunk.size(); k++) {
				m[chunk[k].x * NCLASS + chunk[k].y] = eq[k];
				m[chunk[k].y * NCLASS + chunk[k].x] = 1.0 - eq[k];
			}
			double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
			size_t done = s + chunk.size();
			printf("%zu/%zu pairs  %.1fs  eta %.1fs\n", done, all.size(), secs, secs / done * (all.size() - done));
			fflush(stdout);
		}
		FILE* f = fopen(argv[3], "wb");
		fwrite(m.data(), sizeof(double), m.size(), f);
		fclose(f);
		printf("wrote %s\n", argv[3]);
	} else if (mode == "exact" && argc >= 3) {
		// ./equity exact out.bin [A B]   (the optional pair restricts the run to that one matchup)
		uint32_t binom[49][6] = {};
		for (int nn = 0; nn <= 48; nn++) {
			binom[nn][0] = 1;
			for (int k = 1; k <= 5 && k <= nn; k++) binom[nn][k] = binom[nn - 1][k - 1] + (k <= nn - 1 ? binom[nn - 1][k] : 0);
		}
		CK(cudaMemcpyToSymbol(c_binom, binom, sizeof(binom)));
		std::vector<int2> pairs;
		if (argc >= 5) pairs.push_back(make_int2(class_index(argv[3]), class_index(argv[4])));
		else
			for (int i = 0; i < NCLASS; i++)
				for (int j = i + 1; j < NCLASS; j++) pairs.push_back(make_int2(i, j));
		std::vector<uint32_t> jobs;
		std::vector<unsigned long long> njob(pairs.size(), 0);
		for (size_t p = 0; p < pairs.size(); p++) {
			int a = pairs[p].x, b = pairs[p].y;
			for (int ia = 0; ia < g_ncombo[a]; ia++)
				for (int ib = 0; ib < g_ncombo[b]; ib++) {
					int a0 = g_combos[a][ia][0], a1 = g_combos[a][ia][1], b0 = g_combos[b][ib][0], b1 = g_combos[b][ib][1];
					if (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1) continue;
					jobs.push_back((uint32_t)p << 8 | ia << 4 | ib);
					njob[p]++;
				}
		}
		printf("%zu pairs, %zu jobs, %.3e boards\n", pairs.size(), jobs.size(), (double)jobs.size() * TOTAL_BOARDS);
		uint32_t* d_jobs;
		int2* d_pairs;
		unsigned long long* d_w;
		CK(cudaMalloc(&d_jobs, jobs.size() * 4));
		CK(cudaMalloc(&d_pairs, pairs.size() * sizeof(int2)));
		CK(cudaMalloc(&d_w, pairs.size() * 8));
		CK(cudaMemcpy(d_jobs, jobs.data(), jobs.size() * 4, cudaMemcpyHostToDevice));
		CK(cudaMemcpy(d_pairs, pairs.data(), pairs.size() * sizeof(int2), cudaMemcpyHostToDevice));
		CK(cudaMemset(d_w, 0, pairs.size() * 8));
		const uint64_t BATCH = 32768;  // jobs per launch
		auto t0 = std::chrono::steady_clock::now();
		for (uint64_t first = 0; first < jobs.size(); first += BATCH) {
			uint64_t nj = std::min<uint64_t>(BATCH, jobs.size() - first);
			uint64_t threads = nj * CHUNKS;
			exact_kernel<<<(unsigned)((threads + 255) / 256), 256>>>(d_jobs, d_pairs, first, nj, d_w);
			CK(cudaGetLastError());
			CK(cudaDeviceSynchronize());
			double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
			uint64_t done = first + nj;
			printf("%llu/%zu jobs  %.1fs  eta %.0fs\n", (unsigned long long)done, jobs.size(), secs, secs / done * (jobs.size() - done));
			fflush(stdout);
		}
		std::vector<unsigned long long> w(pairs.size());
		CK(cudaMemcpy(w.data(), d_w, pairs.size() * 8, cudaMemcpyDeviceToHost));
		std::vector<double> m(NCLASS * NCLASS, 0.5);
		for (size_t p = 0; p < pairs.size(); p++) {
			double e = (double)w[p] / (2.0 * (double)njob[p] * (double)TOTAL_BOARDS);
			m[pairs[p].x * NCLASS + pairs[p].y] = e;
			m[pairs[p].y * NCLASS + pairs[p].x] = 1.0 - e;
			if (pairs.size() == 1) printf("%s vs %s: %.6f exact (%llu combo pairs)\n", argv[3], argv[4], e, njob[p]);
		}
		FILE* f = fopen(argv[2], "wb");
		fwrite(m.data(), sizeof(double), m.size(), f);
		fclose(f);
		printf("wrote %s\n", argv[2]);
	} else if (mode == "selftest") {
		int n = argc >= 3 ? atoi(argv[2]) : 200000;
		std::vector<int> full, small;  // small: A..8 only, so quads, full houses and straight flushes are common
		for (int c = 0; c < 52; c++) {
			full.push_back(c);
			if ((c >> 2) <= 6 || (c >> 2) == 12) small.push_back(c);
		}
		long long bad_full = selftest_deals(random_deals(n, full), n);
		long long bad_small = selftest_deals(random_deals(n, small), n);
		printf("selftest: %lld mismatches in %d random deals, %lld in %d deals from a 32-card deck\n", bad_full, n, bad_small, n);
		return (bad_full || bad_small) ? 1 : 0;
	} else {
		fprintf(stderr, "bad arguments\n");
		return 1;
	}
	return 0;
}

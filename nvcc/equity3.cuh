// Three-way equity: kernels, table format, lookup, range-vs-range equity and the resumable generator. Used by
// equity3.cu (the command line) and test_equity3.cu (the checks).
//
// What a cell is. For three hand classes (a, b, c) the cell holds the share of the pot each player wins when
// all five board cards are dealt (ties split among the tied players, a three-way tie is 1/3 each), averaged
// over every ordered choice (combo of a, combo of b, combo of c) whose six cards are all different, and over
// all boards. The count N of those combo triples is stored with the cell, because the range formula needs it.
//
// Indexing. Cells are unordered triples of classes i <= j <= k (class numbers as in poker.cuh, AA = 0). Classes
// may repeat when the cards allow it: AA AA KK is fine (two different pairs of aces), AA AA AA is not (N = 0).
// idx = C(k+2,3) + C(j+1,2) + i, which is 0 .. 818,804 in the order "k outer, j middle, i inner". The three
// numbers are for the players in that sorted order; a lookup for any other player order sorts the classes
// (ties broken by player number) and reads the numbers back, see e3_lookup. Players holding the same class
// have the same equity by symmetry, and the table stores the average of their sampled values so they match.
//
// File (equity3.bin): 16-byte header, then 818,805 records of 4 x uint16 (little endian):
//   rec[0] = N (0 .. 1728, 0 means the triple is impossible), rec[1..3] = share of the sorted slot 0, 1, 2 in
//   units of 1/65535. Impossible triples have all four fields 0. The three shares sum to 65535 +- 2.
//   Header: "EQ3\0", version (1), number of triples (818805), number of classes (169), each uint32.
// equity3-se.bin: one byte per triple, the largest standard error of its three shares in units of 0.002%
//   (255 = 0.51% or more). Same index. Used by the checks, not needed to read the table.
#pragma once

#include <array>

#include "dispatch.cuh"

static const uint32_t E3_NTRIPLE = (uint32_t)NCLASS * (NCLASS + 1) * (NCLASS + 2) / 6;  // 818,805
static const double E3_SE_UNIT = 2e-5;

__host__ __device__ inline uint32_t e3_index_sorted(int i, int j, int k) {  // i <= j <= k
	return (uint32_t)((k + 2) * (k + 1) * k / 6 + (j + 1) * j / 2 + i);
}

// Sort three classes; slot[p] is the position of player p in the sorted order, ties by player number.
__host__ __device__ inline uint32_t e3_index(const int* c, int* slot) {
	int key[3] = {c[0] * 4 + 0, c[1] * 4 + 1, c[2] * 4 + 2};
	if (key[0] > key[1]) { int t = key[0]; key[0] = key[1]; key[1] = t; }
	if (key[1] > key[2]) { int t = key[1]; key[1] = key[2]; key[2] = t; }
	if (key[0] > key[1]) { int t = key[0]; key[0] = key[1]; key[1] = t; }
	for (int s = 0; s < 3; s++) slot[key[s] & 3] = s;
	return e3_index_sorted(key[0] >> 2, key[1] >> 2, key[2] >> 2);
}

// Looks up players' classes c0, c1, c2 (player order). eq[p] is player p's share, returns N (0 = impossible).
__host__ __device__ inline uint32_t e3_lookup(const uint16_t* t, int c0, int c1, int c2, float* eq) {
	int c[3] = {c0, c1, c2}, slot[3];
	const uint16_t* r = t + 4 * (size_t)e3_index(c, slot);
	for (int p = 0; p < 3; p++) eq[p] = r[1 + slot[p]] * (1.0f / 65535.0f);
	return r[0];
}

// -- table file ----------------------------------------------------------------------------------
static std::vector<uint16_t> e3_load_table(const char* path) {
	FILE* f = fopen(path, "rb");
	if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
	uint32_t h[4];
	if (fread(h, 4, 4, f) != 4 || memcmp(h, "EQ3\0", 4) != 0 || h[1] != 1 || h[2] != E3_NTRIPLE || h[3] != NCLASS) {
		fprintf(stderr, "%s is not an equity3 table\n", path);
		exit(1);
	}
	std::vector<uint16_t> t(4 * (size_t)E3_NTRIPLE);
	if (fread(t.data(), 2, t.size(), f) != t.size()) { fprintf(stderr, "%s is truncated\n", path); exit(1); }
	fclose(f);
	return t;
}

static void e3_write_table(const char* path, const std::vector<uint16_t>& t) {
	FILE* f = fopen(path, "wb");
	if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
	uint32_t h[4] = {0, 1, E3_NTRIPLE, NCLASS};
	memcpy(h, "EQ3\0", 4);
	fwrite(h, 4, 4, f);
	fwrite(t.data(), 2, t.size(), f);
	fclose(f);
}

// -- counting combo triples (host) --------------------------------------------------------------
static uint64_t g_cmask[NCLASS][12];

static void e3_setup_masks() {
	for (int c = 0; c < NCLASS; c++)
		for (int k = 0; k < g_ncombo[c]; k++) g_cmask[c][k] = (1ull << g_combos[c][k][0]) | (1ull << g_combos[c][k][1]);
}

// Number of ordered (combo of a, combo of b, combo of c) with six different cards.
static uint32_t e3_count(int a, int b, int c) {
	uint32_t n = 0;
	for (int x = 0; x < g_ncombo[a]; x++)
		for (int y = 0; y < g_ncombo[b]; y++) {
			if (g_cmask[a][x] & g_cmask[b][y]) continue;
			const uint64_t m = g_cmask[a][x] | g_cmask[b][y];
			for (int z = 0; z < g_ncombo[c]; z++) n += !(m & g_cmask[c][z]);
		}
	return n;
}

// All triples in index order: cls[idx] = (i, j, k, N).
static std::vector<int4> e3_all_triples() {
	e3_setup_masks();
	std::vector<int4> cls;
	cls.reserve(E3_NTRIPLE);
	for (int k = 0; k < NCLASS; k++)
		for (int j = 0; j <= k; j++)
			for (int i = 0; i <= j; i++) {
				if ((uint32_t)cls.size() != e3_index_sorted(i, j, k)) { fprintf(stderr, "index mismatch\n"); exit(1); }
				cls.push_back(make_int4(i, j, k, (int)e3_count(i, j, k)));
			}
	return cls;
}

// -- sampling ------------------------------------------------------------------------------------
// acc layout per triple: 3 sums of 6*share (shares are 0, 1/3, 1/2, 1 so 6*share is an integer 0, 2, 3, 6),
// 3 sums of squares of that, and the number of samples n.
static const int E3_ACC = 7;
static const int E3_TPT = 2048;  // threads per triple per launch
static const int E3_BLOCK = 256;

// One job = one triple of classes in a given player order (cls[job] = (class of player 0, 1, 2, N)). Each of its
// E3_TPT threads draws `quota[job]` valid samples: a random combo for every player (retry if two share a card, which
// keeps the draw uniform over the valid combo triples), then a random board from the 46 cards left.
__global__ void e3_sample_kernel(const int4* cls, const uint32_t* jobs, const uint32_t* quota, uint32_t round,
                                 uint64_t seed, unsigned long long* acc) {
	const uint32_t job = blockIdx.y;
	const uint32_t ti = jobs[job];
	const int4 c = cls[ti];
	const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	Rng r;
	r.s = mix64(seed ^ mix64(((uint64_t)ti * 0x9E3779B1ull + round) * 0xD1B54A32D192ED03ull) ^ mix64(tid + 0x1234567ull));
	const uint32_t n0 = c_ncombo[c.x], n1 = c_ncombo[c.y], n2 = c_ncombo[c.z];
	const uint32_t q = quota[job];
	unsigned long long s[3] = {0, 0, 0}, sq[3] = {0, 0, 0};
	for (uint32_t done = 0; done < q;) {
		const uint32_t i0 = r.below(n0), i1 = r.below(n1), i2 = r.below(n2);
		const int a0 = c_combos[c.x][i0][0], a1 = c_combos[c.x][i0][1];
		const int b0 = c_combos[c.y][i1][0], b1 = c_combos[c.y][i1][1];
		const int d0 = c_combos[c.z][i2][0], d1 = c_combos[c.z][i2][1];
		const uint64_t ma = (1ull << a0) | (1ull << a1), mb = (1ull << b0) | (1ull << b1), md = (1ull << d0) | (1ull << d1);
		if ((ma & mb) | (ma & md) | (mb & md)) continue;
		uint64_t used = ma | mb | md;
		int ha[7], hb[7], hd[7];
		ha[0] = a0; ha[1] = a1; hb[0] = b0; hb[1] = b1; hd[0] = d0; hd[1] = d1;
#pragma unroll
		for (int k = 0; k < 5; k++) {
			int x;
			do { x = r.below(52); } while ((used >> x) & 1);
			used |= 1ull << x;
			ha[2 + k] = hb[2 + k] = hd[2 + k] = x;
		}
		const uint32_t sa = eval7(ha), sb = eval7(hb), sd = eval7(hd);
		const uint32_t best = max(sa, max(sb, sd));
		const int wa = sa == best, wb = sb == best, wd = sd == best;
		const unsigned w = 6 / (wa + wb + wd);  // 6, 3 or 2
		if (wa) { s[0] += w; sq[0] += w * w; }
		if (wb) { s[1] += w; sq[1] += w * w; }
		if (wd) { s[2] += w; sq[2] += w * w; }
		done++;
	}
#pragma unroll
	for (int p = 0; p < 3; p++)
		for (int off = 16; off > 0; off >>= 1) {
			s[p] += __shfl_down_sync(0xffffffffu, s[p], off);
			sq[p] += __shfl_down_sync(0xffffffffu, sq[p], off);
		}
	if ((threadIdx.x & 31) == 0) {
		unsigned long long* a = acc + (size_t)ti * E3_ACC;
#pragma unroll
		for (int p = 0; p < 3; p++) {
			atomicAdd(&a[p], s[p]);
			atomicAdd(&a[3 + p], sq[p]);
		}
		atomicAdd(&a[6], (unsigned long long)q * 32);
	}
}

struct E3Est {
	double eq[3];  // share of each player, in the order the classes were given
	double se[3];
	uint64_t n;
};

static E3Est e3_est_from_acc(const unsigned long long* a) {
	E3Est e = {};
	e.n = a[6];
	for (int p = 0; p < 3; p++) {
		if (!e.n) { e.eq[p] = 1.0 / 3; e.se[p] = 1; continue; }
		const double m6 = (double)a[p] / (double)e.n;
		e.eq[p] = m6 / 6;
		double v6 = (double)a[3 + p] / (double)e.n - m6 * m6;
		if (v6 < 0) v6 = 0;
		e.se[p] = sqrt(v6 / 36 / (double)e.n);
	}
	return e;
}

// Samples every given (ordered) class triple with about `samples` valid samples (rounded up to a multiple of
// E3_TPT); the result is in the given player order. Triples that are impossible get n = 0.
static std::vector<E3Est> e3_sample_ordered(const std::vector<int3>& triples, uint64_t samples, uint64_t seed) {
	e3_setup_masks();
	const int T = (int)triples.size();
	std::vector<int4> cls(T);
	std::vector<uint32_t> jobs, quota;
	const uint32_t q = (uint32_t)((samples + E3_TPT - 1) / E3_TPT);
	for (int t = 0; t < T; t++) {
		cls[t] = make_int4(triples[t].x, triples[t].y, triples[t].z, (int)e3_count(triples[t].x, triples[t].y, triples[t].z));
		if (cls[t].w) { jobs.push_back(t); quota.push_back(q); }
	}
	std::vector<unsigned long long> acc((size_t)T * E3_ACC, 0);
	if (!jobs.empty()) {
		int4* d_cls; uint32_t *d_jobs, *d_q; unsigned long long* d_acc;
		CK(cudaMalloc(&d_cls, T * sizeof(int4)));
		CK(cudaMalloc(&d_jobs, jobs.size() * 4));
		CK(cudaMalloc(&d_q, jobs.size() * 4));
		CK(cudaMalloc(&d_acc, acc.size() * 8));
		CK(cudaMemcpy(d_cls, cls.data(), T * sizeof(int4), cudaMemcpyHostToDevice));
		CK(cudaMemcpy(d_jobs, jobs.data(), jobs.size() * 4, cudaMemcpyHostToDevice));
		CK(cudaMemcpy(d_q, quota.data(), jobs.size() * 4, cudaMemcpyHostToDevice));
		CK(cudaMemset(d_acc, 0, acc.size() * 8));
		dispatch(jobs.size(), 4096, [&](uint64_t first, uint64_t nj) {
			// the kernel reads jobs[blockIdx.y], so hand it the slice starting at `first`
			e3_sample_kernel<<<dim3(E3_TPT / E3_BLOCK, (unsigned)nj), E3_BLOCK>>>(d_cls, d_jobs + first, d_q + first, 0, seed, d_acc);
		});
		CK(cudaMemcpy(acc.data(), d_acc, acc.size() * 8, cudaMemcpyDeviceToHost));
		cudaFree(d_cls); cudaFree(d_jobs); cudaFree(d_q); cudaFree(d_acc);
	}
	std::vector<E3Est> out(T);
	for (int t = 0; t < T; t++) out[t] = e3_est_from_acc(&acc[(size_t)t * E3_ACC]);
	return out;
}

// -- exact enumeration ---------------------------------------------------------------------------
// Every valid combo triple of a class triple on all C(46,5) boards. A job is (triple, three combo numbers);
// each job is split into E3_CHUNKS slices of the boards, one slice per thread (same scheme as equity.cuh).
static const int E3_CHUNKS = 512;
static const uint32_t E3_BOARDS = 1370754;  // C(46,5)
__constant__ uint32_t c_binom46[47][6];

__global__ void e3_exact_kernel(const int4* cls, const uint2* jobs, uint64_t first, uint64_t njobs, unsigned long long* sum6) {
	const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= njobs * E3_CHUNKS) return;
	const uint2 job = jobs[first + tid / E3_CHUNKS];
	const uint32_t chunk = (uint32_t)(tid % E3_CHUNKS);
	const int4 c = cls[job.x];
	const int ia = (job.y >> 8) & 15, ib = (job.y >> 4) & 15, ic = job.y & 15;
	int ha[7], hb[7], hd[7];
	ha[0] = c_combos[c.x][ia][0]; ha[1] = c_combos[c.x][ia][1];
	hb[0] = c_combos[c.y][ib][0]; hb[1] = c_combos[c.y][ib][1];
	hd[0] = c_combos[c.z][ic][0]; hd[1] = c_combos[c.z][ic][1];
	uint64_t used = 0;
	for (int k = 0; k < 2; k++) used |= (1ull << ha[k]) | (1ull << hb[k]) | (1ull << hd[k]);
	uint8_t rem[46];
	int n = 0;
	for (int x = 0; x < 52; x++)
		if (!((used >> x) & 1)) rem[n++] = x;
	const uint32_t start = (uint32_t)((uint64_t)chunk * E3_BOARDS / E3_CHUNKS);
	const uint32_t end = (uint32_t)((uint64_t)(chunk + 1) * E3_BOARDS / E3_CHUNKS);
	int idx[5];
	uint32_t r = start;
	int x = 0;
	for (int i = 0; i < 5; i++) {
		int v = x;
		for (;; v++) {
			const uint32_t cnt = c_binom46[45 - v][4 - i];
			if (r < cnt) break;
			r -= cnt;
		}
		idx[i] = v;
		x = v + 1;
	}
	unsigned long long s[3] = {0, 0, 0};
	for (uint32_t k = start; k < end; k++) {
#pragma unroll
		for (int j = 0; j < 5; j++) ha[2 + j] = hb[2 + j] = hd[2 + j] = rem[idx[j]];
		const uint32_t sa = eval7(ha), sb = eval7(hb), sd = eval7(hd);
		const uint32_t best = max(sa, max(sb, sd));
		const int wa = sa == best, wb = sb == best, wd = sd == best;
		const unsigned w = 6 / (wa + wb + wd);
		s[0] += wa * w; s[1] += wb * w; s[2] += wd * w;
		int i = 4;
		while (i >= 0 && idx[i] == 41 + i) i--;
		if (i < 0) break;
		idx[i]++;
		for (int j = i + 1; j < 5; j++) idx[j] = idx[j - 1] + 1;
	}
	for (int p = 0; p < 3; p++) atomicAdd(&sum6[(size_t)job.x * 3 + p], s[p]);
}

// Exact share of each player (given order) for each class triple, over all valid combo triples and all boards.
// An impossible triple gets 1/3 each and njob = 0.
static void e3_exact(const std::vector<int3>& triples, std::vector<std::array<double, 3>>& eq, std::vector<uint64_t>& njob_out) {
	e3_setup_masks();
	uint32_t binom[47][6] = {};
	for (int nn = 0; nn <= 46; nn++) {
		binom[nn][0] = 1;
		for (int k = 1; k <= 5 && k <= nn; k++) binom[nn][k] = binom[nn - 1][k - 1] + (k <= nn - 1 ? binom[nn - 1][k] : 0);
	}
	CK(cudaMemcpyToSymbol(c_binom46, binom, sizeof(binom)));
	const int T = (int)triples.size();
	std::vector<int4> cls(T);
	std::vector<uint2> jobs;
	std::vector<uint64_t> njob(T, 0);
	for (int t = 0; t < T; t++) {
		const int a = triples[t].x, b = triples[t].y, c = triples[t].z;
		cls[t] = make_int4(a, b, c, 0);
		for (int x = 0; x < g_ncombo[a]; x++)
			for (int y = 0; y < g_ncombo[b]; y++)
				for (int z = 0; z < g_ncombo[c]; z++) {
					const uint64_t ma = g_cmask[a][x], mb = g_cmask[b][y], md = g_cmask[c][z];
					if ((ma & mb) | (ma & md) | (mb & md)) continue;
					jobs.push_back(make_uint2((uint32_t)t, (uint32_t)(x << 8 | y << 4 | z)));
					njob[t]++;
				}
	}
	std::vector<unsigned long long> sum6((size_t)T * 3, 0);
	if (!jobs.empty()) {
		int4* d_cls; uint2* d_jobs; unsigned long long* d_s;
		CK(cudaMalloc(&d_cls, T * sizeof(int4)));
		CK(cudaMalloc(&d_jobs, jobs.size() * sizeof(uint2)));
		CK(cudaMalloc(&d_s, sum6.size() * 8));
		CK(cudaMemcpy(d_cls, cls.data(), T * sizeof(int4), cudaMemcpyHostToDevice));
		CK(cudaMemcpy(d_jobs, jobs.data(), jobs.size() * sizeof(uint2), cudaMemcpyHostToDevice));
		CK(cudaMemset(d_s, 0, sum6.size() * 8));
		dispatch(jobs.size(), 8192, [&](uint64_t first, uint64_t nj) {
			e3_exact_kernel<<<(unsigned)((nj * E3_CHUNKS + 255) / 256), 256>>>(d_cls, d_jobs, first, nj, d_s);
		});
		CK(cudaMemcpy(sum6.data(), d_s, sum6.size() * 8, cudaMemcpyDeviceToHost));
		cudaFree(d_cls); cudaFree(d_jobs); cudaFree(d_s);
	}
	eq.assign(T, {1.0 / 3, 1.0 / 3, 1.0 / 3});
	for (int t = 0; t < T; t++)
		if (njob[t])
			for (int p = 0; p < 3; p++) eq[t][p] = (double)sum6[(size_t)t * 3 + p] / (6.0 * (double)njob[t] * (double)E3_BOARDS);
	njob_out = njob;
}

// -- range against range -------------------------------------------------------------------------
// The solver call. w[p][c] is player p's weight on class c (every combo of a class equally likely). For player p
// and each class i of his:
//     eq[p][i]   = sum_{j,k} w[q][j] w[r][k] N(i,j,k) E_p(i,j,k) / mass[p][i]
//     mass[p][i] = sum_{j,k} w[q][j] w[r][k] N(i,j,k)
// where q, r are the other two players, N is the number of combo triples without a shared card and E_p the
// table's share for player p. This is exact for class-uniform ranges, not an approximation: the probability of
// a combo triple is proportional to w[p][i] w[q][j] w[r][k] among the non-conflicting ones (card removal is the
// N factor), and every combo triple of a class triple has the same average equity because suit permutations
// map one into another and keep class-uniform ranges unchanged. Weights are not normalised (mass scales with
// them). mass = 0 means the class cannot occur against these ranges; eq is then 0.
static void e3_ranges_cpu(const std::vector<uint16_t>& t, const double* w /*[3][169]*/, double* eq /*[3][169]*/, double* mass) {
	for (int p = 0; p < 3; p++) {
		const int q = (p + 1) % 3, r = (p + 2) % 3;
		for (int i = 0; i < NCLASS; i++) {
			double num = 0, den = 0;
			for (int j = 0; j < NCLASS; j++) {
				if (w[q * NCLASS + j] == 0) continue;
				for (int k = 0; k < NCLASS; k++) {
					const double ww = w[q * NCLASS + j] * w[r * NCLASS + k];
					if (ww == 0) continue;
					int c[3];
					c[p] = i; c[q] = j; c[r] = k;
					float e[3];
					const uint32_t N = e3_lookup(t.data(), c[0], c[1], c[2], e);
					den += ww * N;
					num += ww * N * e[p];
				}
			}
			mass[p * NCLASS + i] = den;
			eq[p * NCLASS + i] = den > 0 ? num / den : 0;
		}
	}
}

// Same on the GPU; grid (3 players, 169 classes). Callable from other kernels' host code: d_table is the
// 4*E3_NTRIPLE uint16 array (records only, no header), d_w the 3*169 weights, results on the device.
__global__ void e3_ranges_kernel(const uint16_t* t, const double* w, double* eq, double* mass) {
	const int p = blockIdx.x, i = blockIdx.y, q = (p + 1) % 3, r = (p + 2) % 3;
	__shared__ double num, den;
	if (threadIdx.x == 0) { num = 0; den = 0; }
	__syncthreads();
	double n = 0, d = 0;
	for (int m = threadIdx.x; m < NCLASS * NCLASS; m += blockDim.x) {
		const int j = m / NCLASS, k = m % NCLASS;
		const double ww = w[q * NCLASS + j] * w[r * NCLASS + k];
		if (ww == 0) continue;
		int c[3];
		c[p] = i; c[q] = j; c[r] = k;
		float e[3];
		const uint32_t N = e3_lookup(t, c[0], c[1], c[2], e);
		d += ww * N;
		n += ww * N * e[p];
	}
	atomicAdd(&num, n);
	atomicAdd(&den, d);
	__syncthreads();
	if (threadIdx.x == 0) {
		mass[p * NCLASS + i] = den;
		eq[p * NCLASS + i] = den > 0 ? num / den : 0;
	}
}

static void e3_ranges_gpu(const uint16_t* d_table, const double* w, double* eq, double* mass) {
	double *d_w, *d_eq, *d_m;
	CK(cudaMalloc(&d_w, 3 * NCLASS * 8));
	CK(cudaMalloc(&d_eq, 3 * NCLASS * 8));
	CK(cudaMalloc(&d_m, 3 * NCLASS * 8));
	CK(cudaMemcpy(d_w, w, 3 * NCLASS * 8, cudaMemcpyHostToDevice));
	e3_ranges_kernel<<<dim3(3, NCLASS), 256>>>(d_table, d_w, d_eq, d_m);
	CK(cudaGetLastError());
	CK(cudaMemcpy(eq, d_eq, 3 * NCLASS * 8, cudaMemcpyDeviceToHost));
	CK(cudaMemcpy(mass, d_m, 3 * NCLASS * 8, cudaMemcpyDeviceToHost));
	cudaFree(d_w); cudaFree(d_eq); cudaFree(d_m);
}

static uint16_t* e3_upload_table(const std::vector<uint16_t>& t) {
	uint16_t* d;
	CK(cudaMalloc(&d, t.size() * 2));
	CK(cudaMemcpy(d, t.data(), t.size() * 2, cudaMemcpyHostToDevice));
	return d;
}

// Range text: comma-separated classes, each "AKs", "AKs:0.5" (weight), or with a plus: "77+" = 77..AA,
// "ATs+" = ATs AKs, "ATo+" = ATo..AKo. Weight defaults to 1.
static std::vector<double> e3_parse_range(const std::string& spec) {
	std::vector<double> w(NCLASS, 0.0);
	size_t pos = 0;
	while (pos < spec.size()) {
		size_t end = spec.find(',', pos);
		if (end == std::string::npos) end = spec.size();
		std::string tok = spec.substr(pos, end - pos);
		pos = end + 1;
		if (tok.empty()) continue;
		double wt = 1;
		size_t colon = tok.find(':');
		if (colon != std::string::npos) { wt = atof(tok.c_str() + colon + 1); tok = tok.substr(0, colon); }
		bool plus = !tok.empty() && tok.back() == '+';
		if (plus) tok.pop_back();
		std::vector<std::string> names;
		if (!plus) {
			names.push_back(tok);
		} else {
			int hi = (int)(strchr(RANKS, tok[0]) - RANKS), lo = (int)(strchr(RANKS, tok[1]) - RANKS);
			if (tok.size() == 2) {
				for (int r = lo; r <= 12; r++) names.push_back(std::string(2, RANKS[r]));
			} else {
				for (int r = lo; r < hi; r++) names.push_back(std::string() + RANKS[hi] + RANKS[r] + tok[2]);
			}
		}
		for (auto& n : names) w[class_index(n)] = wt;
	}
	return w;
}

// -- generator -----------------------------------------------------------------------------------
// Adaptive and resumable. Round 0 gives every possible triple n0 samples. Every later round gives the triples
// whose largest standard error is still above the target about as many samples again (at most doubling, and never
// more than the estimate of what is still needed), until they all meet the target or reach nmax. State
// (per-triple accumulators and the number of rounds finished) is written to dir/ckpt.bin; a restart reads it and
// carries on. Seeds depend on (triple, round), so a chunk redone after a crash gives the same samples.
struct E3Gen {
	std::vector<int4> cls;
	std::vector<unsigned long long> acc;
	std::vector<uint8_t> rounds;  // rounds finished per triple
};

static bool e3_save_ckpt(const std::string& dir, const E3Gen& g) {
	std::string tmp = dir + "/ckpt.tmp", fin = dir + "/ckpt.bin";
	FILE* f = fopen(tmp.c_str(), "wb");
	if (!f) return false;
	uint32_t h[2] = {0x4B433345u, E3_NTRIPLE};
	fwrite(h, 4, 2, f);
	fwrite(g.rounds.data(), 1, g.rounds.size(), f);
	fwrite(g.acc.data(), 8, g.acc.size(), f);
	bool ok = fclose(f) == 0;
	return ok && rename(tmp.c_str(), fin.c_str()) == 0;
}

static bool e3_load_ckpt(const std::string& dir, E3Gen& g) {
	FILE* f = fopen((dir + "/ckpt.bin").c_str(), "rb");
	if (!f) return false;
	uint32_t h[2];
	bool ok = fread(h, 4, 2, f) == 2 && h[0] == 0x4B433345u && h[1] == E3_NTRIPLE;
	ok = ok && fread(g.rounds.data(), 1, g.rounds.size(), f) == g.rounds.size();
	ok = ok && fread(g.acc.data(), 8, g.acc.size(), f) == g.acc.size();
	fclose(f);
	if (!ok) { fprintf(stderr, "checkpoint in %s is unreadable\n", dir.c_str()); exit(1); }
	return true;
}

struct E3Stats {
	uint64_t valid = 0, impossible = 0, total_n = 0;
	double se_max = 0, se_median = 0, se_p99 = 0, se_mean = 0;
};

// Builds the table and se files' contents from the accumulators.
static E3Stats e3_finalize(const E3Gen& g, std::vector<uint16_t>& table, std::vector<uint8_t>& se_bytes) {
	table.assign(4 * (size_t)E3_NTRIPLE, 0);
	se_bytes.assign(E3_NTRIPLE, 0);
	E3Stats st;
	std::vector<double> ses;
	for (uint32_t t = 0; t < E3_NTRIPLE; t++) {
		const int4 c = g.cls[t];
		if (!c.w) { st.impossible++; continue; }
		st.valid++;
		E3Est e = e3_est_from_acc(&g.acc[(size_t)t * E3_ACC]);
		st.total_n += e.n;
		// players with the same class are exchangeable: average their values (the sum stays 1)
		const int cl[3] = {c.x, c.y, c.z};
		double v[3] = {e.eq[0], e.eq[1], e.eq[2]};
		for (int p = 0; p < 3; p++) {
			double s = 0; int m = 0;
			for (int q = 0; q < 3; q++)
				if (cl[q] == cl[p]) { s += e.eq[q]; m++; }
			v[p] = s / m;
		}
		table[4 * (size_t)t] = (uint16_t)c.w;
		double worst = 0;
		for (int p = 0; p < 3; p++) {
			table[4 * (size_t)t + 1 + p] = (uint16_t)lround(v[p] * 65535.0);
			worst = std::max(worst, e.se[p]);
		}
		se_bytes[t] = (uint8_t)std::min(255.0, ceil(worst / E3_SE_UNIT));
		ses.push_back(worst);
	}
	std::sort(ses.begin(), ses.end());
	if (!ses.empty()) {
		st.se_max = ses.back();
		st.se_median = ses[ses.size() / 2];
		st.se_p99 = ses[(size_t)(ses.size() * 0.99)];
		double s = 0;
		for (double x : ses) s += x;
		st.se_mean = s / ses.size();
	}
	return st;
}

static void e3_generate(const std::string& dir, double target_se, uint64_t nmax, uint64_t n0) {
	const uint64_t SEED = 20261124;
	E3Gen g;
	g.cls = e3_all_triples();
	g.acc.assign((size_t)E3_NTRIPLE * E3_ACC, 0);
	g.rounds.assign(E3_NTRIPLE, 0);
	const bool resumed = e3_load_ckpt(dir, g);
	uint64_t nvalid = 0;
	for (auto& c : g.cls) nvalid += c.w > 0;
	printf("%u triples, %llu possible; target SE %.4f%%, n0 %llu, nmax %llu%s\n", E3_NTRIPLE, (unsigned long long)nvalid,
	       target_se * 100, (unsigned long long)n0, (unsigned long long)nmax, resumed ? " (resumed from checkpoint)" : "");
	fflush(stdout);

	int4* d_cls;
	unsigned long long* d_acc;
	CK(cudaMalloc(&d_cls, g.cls.size() * sizeof(int4)));
	CK(cudaMalloc(&d_acc, g.acc.size() * 8));
	CK(cudaMemcpy(d_cls, g.cls.data(), g.cls.size() * sizeof(int4), cudaMemcpyHostToDevice));
	CK(cudaMemcpy(d_acc, g.acc.data(), g.acc.size() * 8, cudaMemcpyHostToDevice));
	const size_t CH = 4096;  // triples per launch
	uint32_t *d_jobs, *d_q;
	CK(cudaMalloc(&d_jobs, CH * 4));
	CK(cudaMalloc(&d_q, CH * 4));

	auto t_start = std::chrono::steady_clock::now();
	auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); };
	double last_ckpt = 0, last_print = 0;
	uint64_t samples_done = 0;
	for (uint32_t round = 0; round < 40; round++) {
		std::vector<uint32_t> jobs, quota;
		for (uint32_t t = 0; t < E3_NTRIPLE; t++) {
			if (!g.cls[t].w || g.rounds[t] != round) continue;
			uint64_t add;
			if (round == 0) {
				add = n0;
			} else {
				E3Est e = e3_est_from_acc(&g.acc[(size_t)t * E3_ACC]);
				const double se = std::max(e.se[0], std::max(e.se[1], e.se[2]));
				if (se <= target_se || e.n >= nmax) continue;
				// n scales as 1/se^2; the estimate is noisy, so ask for 10% more and never more than double
				const double need = e.n * (se * se) / (target_se * target_se) * 1.1 - e.n;
				add = (uint64_t)std::min<double>(std::min<double>(need, (double)e.n), (double)(nmax - e.n));
				add = std::max<uint64_t>(add, 4 * E3_TPT);
			}
			jobs.push_back(t);
			quota.push_back((uint32_t)((add + E3_TPT - 1) / E3_TPT));
		}
		if (jobs.empty()) break;
		printf("round %u: %zu triples\n", round, jobs.size());
		fflush(stdout);
		dispatch(jobs.size(), CH, [&](uint64_t first_, uint64_t count) {
			const size_t first = (size_t)first_;
			const unsigned nj = (unsigned)count;
			CK(cudaMemcpy(d_jobs, &jobs[first], nj * 4, cudaMemcpyHostToDevice));
			CK(cudaMemcpy(d_q, &quota[first], nj * 4, cudaMemcpyHostToDevice));
			e3_sample_kernel<<<dim3(E3_TPT / E3_BLOCK, nj), E3_BLOCK>>>(d_cls, d_jobs, d_q, round, SEED, d_acc);
			CK(cudaGetLastError());
			CK(cudaDeviceSynchronize());
			for (unsigned k = 0; k < nj; k++) {
				g.rounds[jobs[first + k]] = (uint8_t)(round + 1);
				samples_done += (uint64_t)quota[first + k] * E3_TPT;
			}
			const double now = secs();
			if (now - last_print > 15) {
				printf("  round %u  %zu/%zu  %.0fs  %.2e samples (%.2e/s)\n", round, first + nj, jobs.size(), now,
				       (double)samples_done, samples_done / now);
				fflush(stdout);
				last_print = now;
			}
			if (now - last_ckpt > 120) {
				CK(cudaMemcpy(g.acc.data(), d_acc, g.acc.size() * 8, cudaMemcpyDeviceToHost));
				if (!e3_save_ckpt(dir, g)) fprintf(stderr, "warning: cannot write checkpoint\n");
				last_ckpt = now;
			}
		});
		CK(cudaMemcpy(g.acc.data(), d_acc, g.acc.size() * 8, cudaMemcpyDeviceToHost));
		if (!e3_save_ckpt(dir, g)) fprintf(stderr, "warning: cannot write checkpoint\n");
		last_ckpt = secs();
		printf("round %u done at %.0fs\n", round, secs());
		fflush(stdout);
	}
	cudaFree(d_cls); cudaFree(d_acc); cudaFree(d_jobs); cudaFree(d_q);
	std::vector<uint16_t> table;
	std::vector<uint8_t> se;
	E3Stats st = e3_finalize(g, table, se);
	e3_write_table((dir + "/equity3.bin").c_str(), table);
	FILE* f = fopen((dir + "/equity3-se.bin").c_str(), "wb");
	if (!f) { fprintf(stderr, "cannot write se file\n"); exit(1); }
	fwrite(se.data(), 1, se.size(), f);
	fclose(f);
	printf("done: %llu possible triples, %llu impossible, %.3e samples in total\n", (unsigned long long)st.valid,
	       (unsigned long long)st.impossible, (double)st.total_n);
	printf("standard error per triple (largest of the 3 players): mean %.4f%%, median %.4f%%, 99th pct %.4f%%, max %.4f%%\n",
	       st.se_mean * 100, st.se_median * 100, st.se_p99 * 100, st.se_max * 100);
	printf("wrote %s/equity3.bin and equity3-se.bin  (%.1fs this run)\n", dir.c_str(), secs());
}

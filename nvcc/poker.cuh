// Shared by the nvcc tools: card encoding, the 7-card evaluator, a counter-based RNG and the 169 hand classes.
//
// Cards are 0..51 = rank index (0 = deuce) * 4 + suit, as in src/lib/poker/ranges.ts. Class order is
// ALL_HANDS from ranges.ts (AA, AKs, AKo, AQs, ..., 22), so results line up with the TypeScript tools.
// eval7 scores a hand as category << 20 | tie-break nibbles; only the ordering matters.
//
// One .cu file per program includes this header (the __constant__ tables are per translation unit).
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
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


// -- hand classes (host) -------------------------------------------------------------------------
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

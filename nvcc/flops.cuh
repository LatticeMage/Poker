// The 1,755 strategically distinct flops: the 22,100 three-card flops grouped by suit permutation (24 of them
// map a flop to another flop with the same strategic meaning). Each class has a representative flop and a weight
// = how many of the 22,100 flops it stands for (4, 12 or 24). Host code, no GPU needed.
//
//   const std::vector<Flop>& flops = all_flops();   // 1,755 entries, canonical order
//   Flop f: cards[3] (0..51, rank index * 4 + suit, sorted), weight, kind (rainbow/two-tone/monotone),
//           texture bits (paired, high card rank, connectedness) for grouping by board type.
//
// Used to weight a flop-by-flop computation so that it equals the average over all 22,100 flops, with 1,755
// computations instead of 22,100.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

struct Flop {
	int cards[3];        // sorted ascending; the representative of the class
	int weight;          // number of the 22,100 flops this class stands for
	int suit_kind;       // 0 rainbow (3 suits), 1 two-tone (2 suits), 2 monotone (1 suit)
	bool paired;         // two cards of the same rank
	bool trips;          // three cards of the same rank
	int high;            // highest rank index (0 = deuce .. 12 = ace)
	int span;            // highest rank - lowest rank (connectedness: 2 = three in a row if all distinct)
};

// Canonical form of a card triple under the 24 suit permutations: the smallest sorted triple.
static inline void flop_canon(const int in[3], int out[3]) {
	static int perms[24][4];
	static bool built = false;
	if (!built) {
		int p[4] = {0, 1, 2, 3}, n = 0;
		do {
			for (int i = 0; i < 4; i++) perms[n][i] = p[i];
			n++;
		} while (std::next_permutation(p, p + 4));
		built = true;
	}
	bool have = false;
	for (int k = 0; k < 24; k++) {
		int t[3];
		for (int i = 0; i < 3; i++) t[i] = (in[i] & ~3) | perms[k][in[i] & 3];
		std::sort(t, t + 3);
		if (!have || t[0] < out[0] || (t[0] == out[0] && (t[1] < out[1] || (t[1] == out[1] && t[2] < out[2])))) {
			out[0] = t[0]; out[1] = t[1]; out[2] = t[2];
			have = true;
		}
	}
}

static inline const std::vector<Flop>& all_flops() {
	static std::vector<Flop> flops;
	if (!flops.empty()) return flops;
	// every canonical triple appears as its own canon: collect the canon triples and count the flops mapping to each
	for (int a = 0; a < 52; a++)
		for (int b = a + 1; b < 52; b++)
			for (int c = b + 1; c < 52; c++) {
				const int in[3] = {a, b, c};
				int canon[3];
				flop_canon(in, canon);
				if (canon[0] == a && canon[1] == b && canon[2] == c) {
					Flop f;
					f.cards[0] = a; f.cards[1] = b; f.cards[2] = c;
					f.weight = 0;
					int suits = 0;
					for (int i = 0; i < 3; i++) suits |= 1 << (in[i] & 3);
					const int ns = __builtin_popcount(suits);
					f.suit_kind = ns == 3 ? 0 : (ns == 2 ? 1 : 2);
					const int r0 = a >> 2, r1 = b >> 2, r2 = c >> 2;
					f.paired = (r0 == r1) || (r1 == r2);
					f.trips = (r0 == r1) && (r1 == r2);
					f.high = std::max(r0, std::max(r1, r2));
					f.span = f.high - std::min(r0, std::min(r1, r2));
					flops.push_back(f);
				}
			}
	// weights: count every flop under its canonical class
	for (int a = 0; a < 52; a++)
		for (int b = a + 1; b < 52; b++)
			for (int c = b + 1; c < 52; c++) {
				const int in[3] = {a, b, c};
				int canon[3];
				flop_canon(in, canon);
				for (Flop& f : flops)
					if (f.cards[0] == canon[0] && f.cards[1] == canon[1] && f.cards[2] == canon[2]) { f.weight++; break; }
			}
	return flops;
}

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
	int index;           // position in all_flops() (kept in a subset, so results can be mapped back)
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
					f.index = (int)flops.size();
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

// -- a representative subset ------------------------------------------------------------------------------------
// Computing something for all 1,755 classes is often too much; a subset of k flops, each weighted by the weight of
// the group of similar flops it stands for, gives nearly the same average. This is OUR subset (not the list a
// commercial solver ships), chosen by weighted k-medoids on the board structure so that every kind of flop is
// represented, and every one of the 1,755 classes belongs to exactly one group (subset_partition): the weights of
// the k flops add up to 22,100 again. Deterministic: integer distances, fixed start,
// ties broken by index, so every machine gets the same list. subset_flops(184) is the one used first.
struct FlopFeatures { int v[7]; };  // r0 >= r1 >= r2 (0..12), suit kind x6, paired x6, trips x6

static inline FlopFeatures flop_features(const Flop& f) {
	int r[3] = {f.cards[0] >> 2, f.cards[1] >> 2, f.cards[2] >> 2};
	std::sort(r, r + 3, [](int a, int b) { return a > b; });
	FlopFeatures x;
	x.v[0] = r[0]; x.v[1] = r[1]; x.v[2] = r[2];
	x.v[3] = f.suit_kind * 6;
	x.v[4] = f.paired && !f.trips ? 6 : 0;
	x.v[5] = f.trips ? 6 : 0;
	x.v[6] = 0;
	return x;
}

static inline int flop_dist(const FlopFeatures& a, const FlopFeatures& b) {
	int d = 0;
	for (int i = 0; i < 6; i++) d += (a.v[i] - b.v[i]) * (a.v[i] - b.v[i]);
	return d;
}

// The subset as a PARTITION: every one of the 1,755 classes belongs to exactly one group, and each group has one
// representative flop. rep[g] is the representative of group g (its weight = the total weight of the group);
// group_of[i] is the group of class i of all_flops(); members of a group are the classes with group_of == g.
struct FlopPartition {
	std::vector<Flop> rep;        // k representatives, ordered by index in all_flops()
	std::vector<int> group_of;    // size 1755: index into rep
};

static inline FlopPartition subset_partition(int k) {
	const std::vector<Flop>& all = all_flops();
	const int n = (int)all.size();
	if (k >= n) {
		FlopPartition id;
		id.rep = all;
		id.group_of.resize(n);
		for (int i = 0; i < n; i++) id.group_of[i] = i;
		return id;
	}
	std::vector<FlopFeatures> x(n);
	for (int i = 0; i < n; i++) x[i] = flop_features(all[i]);
	// start: farthest-first, beginning with the heaviest class
	std::vector<int> med;
	std::vector<int> nearest(n, 1 << 30);
	int first = 0;
	for (int i = 1; i < n; i++)
		if (all[i].weight > all[first].weight) first = i;
	med.push_back(first);
	while ((int)med.size() < k) {
		const int last = med.back();
		int best = -1, best_d = -1;
		for (int i = 0; i < n; i++) {
			nearest[i] = std::min(nearest[i], flop_dist(x[i], x[last]));
			if (nearest[i] > best_d) { best_d = nearest[i]; best = i; }
		}
		med.push_back(best);
	}
	std::vector<int> assign(n, 0);
	for (int iter = 0; iter < 100; iter++) {
		bool changed = false;
		for (int i = 0; i < n; i++) {  // assign each flop to its nearest medoid (ties: the lower medoid position)
			int bj = 0, bd = flop_dist(x[i], x[med[0]]);
			for (int j = 1; j < k; j++) {
				const int d = flop_dist(x[i], x[med[j]]);
				if (d < bd) { bd = d; bj = j; }
			}
			if (assign[i] != bj) { assign[i] = bj; changed = true; }
		}
		for (int j = 0; j < k; j++) {  // new medoid: the member that minimises the weighted distance to the others
			std::vector<int> members;
			for (int i = 0; i < n; i++)
				if (assign[i] == j) members.push_back(i);
			if (members.empty()) continue;
			long long best = -1;
			int bi = med[j];
			for (int a : members) {
				long long cost = 0;
				for (int b : members) cost += (long long)all[b].weight * flop_dist(x[a], x[b]);
				if (best < 0 || cost < best) { best = cost; bi = a; }
			}
			if (bi != med[j]) { med[j] = bi; changed = true; }
		}
		if (!changed) break;
	}
	// groups in the order of their representatives' index in all_flops()
	std::vector<int> order(k);
	for (int j = 0; j < k; j++) order[j] = j;
	std::sort(order.begin(), order.end(), [&](int a, int b) { return all[med[a]].index < all[med[b]].index; });
	std::vector<int> new_pos(k, -1);
	FlopPartition out;
	for (int pos = 0; pos < k; pos++) {
		const int j = order[pos];
		Flop f = all[med[j]];
		f.weight = 0;
		for (int i = 0; i < n; i++)
			if (assign[i] == j) f.weight += all[i].weight;
		if (f.weight > 0) { new_pos[j] = (int)out.rep.size(); out.rep.push_back(f); }
	}
	out.group_of.resize(n);
	for (int i = 0; i < n; i++) out.group_of[i] = new_pos[assign[i]];
	return out;
}

// The representatives alone (each weighted by its group).
static inline std::vector<Flop> subset_flops(int k) { return subset_partition(k).rep; }

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
#include <string>
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

// -- names ------------------------------------------------------------------------------------------------------
// "AsKsQd": ranks high to low, suits as s h d c (the canonical representative's suits, nothing more).
static inline std::string flop_name(const Flop& f) {
	const char* ranks = "23456789TJQKA";
	const char* suits = "shdc";
	int c[3] = {f.cards[0], f.cards[1], f.cards[2]};
	std::sort(c, c + 3, [](int a, int b) { return (a >> 2) != (b >> 2) ? (a >> 2) > (b >> 2) : (a & 3) < (b & 3); });
	std::string out;
	for (int i = 0; i < 3; i++) {
		out += ranks[c[i] >> 2];
		out += suits[c[i] & 3];
	}
	return out;
}

// Number of broadway cards (T J Q K A) on the flop.
static inline int flop_broadway(const Flop& f) {
	int n = 0;
	for (int i = 0; i < 3; i++) n += (f.cards[i] >> 2) >= 8;
	return n;
}

// Span of the three ranks counting the ace both high and low (A-2-3 and A-K-Q are both tight).
static inline int flop_span_ace_low(const Flop& f) {
	int r[3] = {f.cards[0] >> 2, f.cards[1] >> 2, f.cards[2] >> 2};
	const int hi = std::max(r[0], std::max(r[1], r[2])), lo = std::min(r[0], std::min(r[1], r[2]));
	int span = hi - lo;
	if (hi == 12) {
		int l[3];
		for (int i = 0; i < 3; i++) l[i] = r[i] == 12 ? -1 : r[i];
		span = std::min(span, std::max(l[0], std::max(l[1], l[2])) - std::min(l[0], std::min(l[1], l[2])));
	}
	return span;
}

// -- wetness: a RESULT, not a category ------------------------------------------------------------------------------
// How many draws a board offers. Points from the flush side (rainbow 0, two-tone 1, monotone 2) and from the straight
// side, measured on the DISTINCT ranks of the board with the ace both high and low (span <= 2: 2 points; 3-4: 1; else 0).
// A paired board has only two distinct ranks, so its straight side is the span of those two (98x9 is tight), and a
// paired board with a flush draw and a straight draw is wet. 0 = 乾, 1 = 偏乾, 2 = 中, 3 = 濕, 4 = 很濕.
static inline int flop_wetness(const Flop& f) {
	int r[3] = {f.cards[0] >> 2, f.cards[1] >> 2, f.cards[2] >> 2};
	std::sort(r, r + 3);
	std::vector<int> d;
	for (int i = 0; i < 3; i++)
		if (d.empty() || d.back() != r[i]) d.push_back(r[i]);
	int span = d.back() - d.front();
	if (d.back() == 12) {  // the ace also plays low
		std::vector<int> l;
		for (int x : d) l.push_back(x == 12 ? -1 : x);
		span = std::min(span, *std::max_element(l.begin(), l.end()) - *std::min_element(l.begin(), l.end()));
	}
	const int flush_pts = f.suit_kind == 2 ? 2 : (f.suit_kind == 1 ? 1 : 0);
	const int straight_pts = span <= 2 ? 2 : (span <= 4 ? 1 : 0);
	return flush_pts + straight_pts;
}
static inline const char* flop_wetness_name(int w) {
	static const char* n[5] = {"乾", "偏乾", "中", "濕", "很濕"};
	return n[w];
}

// -- a named taxonomy: the refinement of the 1,755 into groups people can name ------------------------------------
// Every flop falls into one group of a fixed hierarchy; the levels, in this order:
//   1. broadway cards (T J Q K A) on the board: 0高 / 1高 / 2高 / 3高
//   2. pairing: 未成對 / 成對 / 三條
//   3. suits: 彩虹 (three suits) / 雙花 (two suits: a flush draw) / 單色 (one suit: a flush board, "天花")
//   4. straight structure, unpaired boards only (ace counts high and low): 順緊 span <= 2 / 順可 3-4 / 順聽 5-6 / 順遠 >= 7
//      ("dry" and "wet" are not a category: they are a result of the suits and the straight structure, see flop_wetness)
//   5. the highest card: A高 / K高 / Q高 / JT高 / 98高 / 7低 (7 or lower)
//   6. two-tone boards only: 花含頂牌 (the two suited cards include the highest card) / 花不含頂牌
// A group is the set of flops that agree on all levels, e.g. "2高|未成對|雙花|近連|K高|花含頂牌". Groups beyond what
// the user needs can be merged by dropping the last levels (flop_group_name(f, depth)); flop_taxonomy(k) refines
// further, splitting the heaviest group by the middle card until there are exactly k groups.
static inline int flop_two_tone_top_suited(const Flop& f) {  // 1 if the two suited cards include the highest card
	int cnt[4] = {0, 0, 0, 0};
	for (int i = 0; i < 3; i++) cnt[f.cards[i] & 3]++;
	int fs = -1;
	for (int s = 0; s < 4; s++)
		if (cnt[s] == 2) fs = s;
	if (fs < 0) return -1;
	int top = -1, top_suit = -1;
	for (int i = 0; i < 3; i++)
		if ((f.cards[i] >> 2) > top) { top = f.cards[i] >> 2; top_suit = f.cards[i] & 3; }
	return top_suit == fs ? 1 : 0;
}

static inline std::string flop_group_name(const Flop& f, int depth = 6) {
	std::string n;
	const char* bw[4] = {"0高", "1高", "2高", "3高"};
	n += bw[flop_broadway(f)];
	if (depth >= 2) n += std::string("|") + (f.trips ? "三條" : (f.paired ? "成對" : "未成對"));
	if (depth >= 3) n += std::string("|") + (f.suit_kind == 0 ? "彩虹" : (f.suit_kind == 1 ? "雙花" : "單色"));
	if (depth >= 4) {
		const int sp = flop_span_ace_low(f);
		// straight structure of the board (a fact about the ranks; "dry" and "wet" are RESULTS, see flop_wetness):
		// span <= 2 tight, 3-4 a straight is possible with two hole cards, 5-6 only draws, >= 7 far apart
		n += std::string("|") + (f.paired ? "-" : (sp <= 2 ? "順緊" : (sp <= 4 ? "順可" : (sp <= 6 ? "順聽" : "順遠"))));
	}
	if (depth >= 5) {
		const int h = f.high;
		n += std::string("|") + (h == 12 ? "A高" : (h == 11 ? "K高" : (h == 10 ? "Q高" : (h >= 8 ? "JT高" : (h >= 6 ? "98高" : "7低")))));
	}
	if (depth >= 6) {
		const int t = flop_two_tone_top_suited(f);
		n += std::string("|") + (t < 0 ? "-" : (t ? "花含頂牌" : "花不含頂牌"));
	}
	return n;
}

struct FlopTaxonomy {
	std::vector<std::string> name;   // group names
	std::vector<Flop> rep;           // one representative per group (the member closest to the others), weight = the group's
	std::vector<int> group_of;       // size 1755: group of class i of all_flops()
	std::vector<double> wetness;     // per group: the weighted mean wetness of its members, 0 (dry) .. 4 (very wet): the group's place on the dry-wet spectrum
};

static inline FlopTaxonomy flop_taxonomy(int k) {
	const std::vector<Flop>& all = all_flops();
	const int n = (int)all.size();
	std::vector<std::string> label(n);
	for (int i = 0; i < n; i++) label[i] = flop_group_name(all[i], 6);
	// refinement: while there are fewer than k groups, split the heaviest splittable group by the middle card
	// (at or below / above the middle card of the group's lightest... fixed rule: middle rank <= its median)
	for (;;) {
		std::vector<std::string> names;
		for (int i = 0; i < n; i++)
			if (std::find(names.begin(), names.end(), label[i]) == names.end()) names.push_back(label[i]);
		if ((int)names.size() >= k) break;
		std::string heavy;
		long long hw = -1;
		for (const std::string& g : names) {
			long long w = 0;
			int cnt = 0;
			for (int i = 0; i < n; i++)
				if (label[i] == g) { w += all[i].weight; cnt++; }
			if (cnt > 1 && w > hw) { hw = w; heavy = g; }
		}
		if (hw < 0) break;
		std::vector<int> mid;
		std::vector<int> members;
		for (int i = 0; i < n; i++)
			if (label[i] == heavy) {
				members.push_back(i);
				int r[3] = {all[i].cards[0] >> 2, all[i].cards[1] >> 2, all[i].cards[2] >> 2};
				std::sort(r, r + 3);
				mid.push_back(r[1]);
			}
		std::vector<int> sorted_mid = mid;
		std::sort(sorted_mid.begin(), sorted_mid.end());
		const int median = sorted_mid[sorted_mid.size() / 2 - (sorted_mid.size() % 2 == 0 ? 1 : 0)];
		const char* hi_lo[2] = {"|中張高", "|中張低"};
		bool any_low = false, any_high = false;
		for (size_t m = 0; m < members.size(); m++) { if (mid[m] <= median) any_low = true; else any_high = true; }
		if (!any_low || !any_high) {  // cannot split by the middle card: split by the lowest card instead
			for (size_t m = 0; m < members.size(); m++) {
				int r[3] = {all[members[m]].cards[0] >> 2, all[members[m]].cards[1] >> 2, all[members[m]].cards[2] >> 2};
				std::sort(r, r + 3);
				mid[m] = r[0];
			}
			sorted_mid = mid;
			std::sort(sorted_mid.begin(), sorted_mid.end());
		}
		const int cut = sorted_mid[sorted_mid.size() / 2 - (sorted_mid.size() % 2 == 0 ? 1 : 0)];
		for (size_t m = 0; m < members.size(); m++) label[members[m]] = heavy + hi_lo[mid[m] <= cut ? 1 : 0];
	}
	FlopTaxonomy out;
	std::vector<std::string> names;
	for (int i = 0; i < n; i++)
		if (std::find(names.begin(), names.end(), label[i]) == names.end()) names.push_back(label[i]);
	std::sort(names.begin(), names.end());
	out.name = names;
	out.group_of.assign(n, 0);
	out.rep.assign(names.size(), Flop());
	std::vector<long long> gw(names.size(), 0);
	for (int i = 0; i < n; i++) {
		const int g = (int)(std::find(names.begin(), names.end(), label[i]) - names.begin());
		out.group_of[i] = g;
		gw[g] += all[i].weight;
	}
	// the representative: the member with the smallest weighted distance (in board structure) to the others
	for (size_t g = 0; g < names.size(); g++) {
		long long best = -1;
		int bi = -1;
		for (int a = 0; a < n; a++) {
			if (out.group_of[a] != (int)g) continue;
			long long cost = 0;
			const FlopFeatures fa = flop_features(all[a]);
			for (int b = 0; b < n; b++)
				if (out.group_of[b] == (int)g) cost += (long long)all[b].weight * flop_dist(fa, flop_features(all[b]));
			if (best < 0 || cost < best) { best = cost; bi = a; }
		}
		out.rep[g] = all[bi];
		out.rep[g].weight = (int)gw[g];
	}
	// the dry-wet spectrum: wetness is a result computed per flop; a group sits at the weighted mean of its members
	out.wetness.assign(names.size(), 0.0);
	for (int i = 0; i < n; i++) out.wetness[out.group_of[i]] += (double)all[i].weight * flop_wetness(all[i]);
	for (size_t g = 0; g < names.size(); g++) out.wetness[g] /= (double)gw[g];
	return out;
}


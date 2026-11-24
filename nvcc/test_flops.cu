// Checks for flops.cuh (host only, no GPU needed): 1,755 classes, weights that add up to 22,100, and the structure
// the classes must have.
//
//   nvcc -O3 -o test_flops test_flops.cu && ./test_flops

#include "flops.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

static int failures = 0;
static void check(bool ok, const char* what) {
	printf("  %s  %s\n", ok ? "ok " : "BAD", what);
	failures += !ok;
}

int main() {
	const std::vector<Flop>& f = all_flops();
	check(f.size() == 1755, "1,755 strategically distinct flops");
	long long total = 0;
	bool weights_ok = true;
	for (const Flop& x : f) {
		total += x.weight;
		weights_ok = weights_ok && (x.weight == 4 || x.weight == 12 || x.weight == 24);
	}
	check(total == 22100, "the weights add up to C(52,3) = 22,100");
	check(weights_ok, "every weight is 4, 12 or 24");

	// weight by suit structure: rainbow 8,788 flops, two-tone 12,168, monotone 1,144 (counted by hand: 13^3-... see below)
	long long w[3] = {0, 0, 0};
	int n[3] = {0, 0, 0};
	for (const Flop& x : f) { w[x.suit_kind] += x.weight; n[x.suit_kind]++; }
	check(w[2] == 4LL * (13 * 12 * 11 / 6), "monotone flops: 4 suits x C(13,3) = 1,144");
	check(w[0] == 22100 - 1144 - w[1] && n[0] == 455 && n[1] == 1014 && n[2] == 286,
	      "classes by suits: 455 rainbow, 1,014 two-tone, 286 monotone");
	check(w[1] == 12168 && w[0] == 8788, "flops by suits: 8,788 rainbow, 12,168 two-tone");

	// pairs: 13 ranks x C(4,2) pair-suit choices x 48 third cards = 13 * 6 * 48 = 3,744 paired flops incl. trips counted apart
	long long paired_w = 0, trips_w = 0;
	for (const Flop& x : f) {
		if (x.trips) trips_w += x.weight;
		else if (x.paired) paired_w += x.weight;
	}
	check(trips_w == 13 * 4, "trips: 13 ranks x 4 = 52 flops");
	check(paired_w == 13 * 6 * 48, "paired (not trips): 13 x C(4,2) x 48 = 3,744 flops");

	// the representatives really are distinct classes
	std::set<std::tuple<int, int, int>> seen;
	bool distinct = true;
	for (const Flop& x : f) distinct = distinct && seen.insert({x.cards[0], x.cards[1], x.cards[2]}).second;
	check(distinct, "the 1,755 representatives are all different");

	// canonical form: any suit permutation of a flop gives the same canon, and canon(canon) = canon
	bool canon_ok = true;
	const int tests[][3] = {{0, 5, 10}, {51, 47, 43}, {3, 7, 11}, {12, 13, 40}, {4, 9, 50}};
	for (auto& t : tests) {
		int a[3], b[3], c[3];
		flop_canon(t, a);
		const int swapped[3] = {(t[0] & ~3) | (3 - (t[0] & 3)), (t[1] & ~3) | (3 - (t[1] & 3)), (t[2] & ~3) | (3 - (t[2] & 3))};
		flop_canon(swapped, b);
		flop_canon(a, c);
		canon_ok = canon_ok && a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[0] == c[0] && a[1] == c[1] && a[2] == c[2];
	}
	check(canon_ok, "a suit relabelling gives the same canonical flop, and canon is idempotent");

	// the 184-flop subset: our representative list
	{
		std::vector<Flop> sub = subset_flops(184);
		long long sw = 0;
		bool kinds[3] = {false, false, false};
		bool has_paired = false, has_trips = false, has_ace = false, has_deuce = false, has_connected = false, has_gapped = false;
		std::set<int> idx;
		double mean_high_full = 0, mean_high_sub = 0, mean_span_full = 0, mean_span_sub = 0, w_two_full = 0, w_two_sub = 0;
		for (const Flop& x : sub) {
			sw += x.weight;
			kinds[x.suit_kind] = true;
			has_paired = has_paired || (x.paired && !x.trips);
			has_trips = has_trips || x.trips;
			has_ace = has_ace || x.high == 12;
			has_deuce = has_deuce || x.high == 2;
			has_connected = has_connected || (!x.paired && x.span <= 3);
			has_gapped = has_gapped || (!x.paired && x.span >= 8);
			idx.insert(x.index);
			mean_high_sub += (double)x.weight * x.high;
			mean_span_sub += (double)x.weight * x.span;
			w_two_sub += x.suit_kind == 1 ? x.weight : 0;
		}
		for (const Flop& x : f) {
			mean_high_full += (double)x.weight * x.high;
			mean_span_full += (double)x.weight * x.span;
			w_two_full += x.suit_kind == 1 ? x.weight : 0;
		}
		check(sub.size() == 184, "the subset has 184 flops");
		check(sw == 22100, "the subset weights add up to 22,100 again");
		check(idx.size() == 184, "the 184 are different classes (their indices point into the 1,755)");
		check(kinds[0] && kinds[1] && kinds[2], "all three suit structures are represented");
		check(has_paired && has_trips, "paired boards and trips are represented");
		check(has_ace && has_deuce && has_connected && has_gapped, "ace-high, low, connected and disconnected flops are represented");
		check(std::fabs(mean_high_sub - mean_high_full) / 22100 < 0.15, "the weighted mean high card is close to the full set's (within 0.15 of a rank)");
		check(std::fabs(mean_span_sub - mean_span_full) / 22100 < 0.15, "the weighted mean span is close to the full set's (within 0.15)");
		check(std::fabs(w_two_sub - w_two_full) / 22100 < 0.02, "the share of two-tone flops is within 2 points of the full set's");
		// every class belongs to exactly one of the 184 groups; a representative belongs to its own group
		FlopPartition part = subset_partition(184);
		std::vector<long long> gw(184, 0);
		std::vector<int> gn(184, 0);
		bool mapped = part.group_of.size() == f.size() && part.rep.size() == 184;
		for (size_t i = 0; mapped && i < f.size(); i++) {
			const int g = part.group_of[i];
			mapped = g >= 0 && g < 184;
			if (mapped) { gw[g] += f[i].weight; gn[g]++; }
		}
		bool weights_match = mapped, reps_self = mapped, nonempty = mapped;
		int total_members = 0;
		for (int g = 0; mapped && g < 184; g++) {
			weights_match = weights_match && gw[g] == part.rep[g].weight;
			reps_self = reps_self && part.group_of[part.rep[g].index] == g;
			nonempty = nonempty && gn[g] >= 1;
			total_members += gn[g];
		}
		check(mapped && total_members == 1755, "every one of the 1,755 classes belongs to exactly one of the 184 groups");
		check(nonempty && reps_self, "no group is empty, and each representative is in its own group");
		check(weights_match, "each representative's weight is the total weight of its group");
		int biggest = 0, single = 0;
		for (int g = 0; mapped && g < 184; g++) { biggest = std::max(biggest, gn[g]); single += gn[g] == 1; }
		printf("  (groups: %d classes in the biggest group, %d groups with one class)\n", biggest, single);

		std::vector<Flop> again = subset_flops(184);
		bool same = again.size() == sub.size();
		for (size_t i = 0; same && i < sub.size(); i++) same = again[i].index == sub[i].index && again[i].weight == sub[i].weight;
		check(same, "the subset is deterministic");
		printf("  (mean high card: full %.3f, subset %.3f; mean span: full %.3f, subset %.3f; two-tone share: full %.4f, subset %.4f)\n",
		       mean_high_full / 22100, mean_high_sub / 22100, mean_span_full / 22100, mean_span_sub / 22100, w_two_full / 22100,
		       w_two_sub / 22100);
	}

	printf(failures ? "FAIL\n" : "PASS\n");
	return failures ? 1 : 0;
}

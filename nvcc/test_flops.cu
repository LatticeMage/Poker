// Checks for flops.cuh (host only, no GPU needed): 1,755 classes, weights that add up to 22,100, and the structure
// the classes must have.
//
//   nvcc -O3 -o test_flops test_flops.cu && ./test_flops

#include "flops.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <string>

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

	// names: ranks high to low, then the suits as s h d c
	{
		bool names_ok = true;
		std::set<std::string> names;
		for (const Flop& x : f) names_ok = names_ok && names.insert(flop_name(x)).second && flop_name(x).size() == 6;
		check(names_ok, "the 1,755 flops have 1,755 different six-character names (e.g. AsKsQd)");
		int broadway_hist[4] = {0, 0, 0, 0};
		long long bw_weight[4] = {0, 0, 0, 0};
		for (const Flop& x : f) { broadway_hist[flop_broadway(x)]++; bw_weight[flop_broadway(x)] += x.weight; }
		// by hand: no broadway card = three of the 32 cards 2..9 = C(32,3) = 4,960; three broadway = three of the 20 cards T..A = C(20,3) = 1,140
		check(bw_weight[0] == 4960 && bw_weight[3] == 1140 && bw_weight[0] + bw_weight[1] + bw_weight[2] + bw_weight[3] == 22100,
		      "flops by number of broadway cards: 0 -> 4,960 and 3 -> 1,140 (counted by hand), all four add up to 22,100");
	}

	// the taxonomy: every level partitions all 1,755 classes, each level refines the one before, 184 groups at the end
	{
		bool nested = true, partitions = true;
		size_t prev_groups = 0;
		for (int depth = 1; depth <= 6; depth++) {
			std::set<std::string> names;
			long long wsum = 0;
			for (const Flop& x : f) {
				names.insert(flop_group_name(x, depth));
				wsum += x.weight;
				if (depth > 1) {  // the name one level up is the prefix of this level's name
					const std::string up = flop_group_name(x, depth - 1), me = flop_group_name(x, depth);
					nested = nested && me.compare(0, up.size(), up) == 0;
				}
			}
			partitions = partitions && wsum == 22100;
			nested = nested && names.size() >= prev_groups;
			prev_groups = names.size();
		}
		check(partitions, "every level of the taxonomy covers all 1,755 classes (22,100 flops)");
		check(nested, "every level refines the one before (the group name is extended, the number of groups never falls)");
		FlopTaxonomy t184 = flop_taxonomy(184);
		long long tw = 0;
		bool in_range = t184.group_of.size() == 1755;
		std::vector<int> members(t184.name.size(), 0);
		for (size_t i = 0; in_range && i < 1755; i++) {
			in_range = t184.group_of[i] >= 0 && t184.group_of[i] < (int)t184.name.size();
			if (in_range) members[t184.group_of[i]]++;
		}
		bool nonempty = true, weights_ok = true;
		std::vector<long long> gw(t184.name.size(), 0);
		for (size_t i = 0; in_range && i < 1755; i++) gw[t184.group_of[i]] += f[i].weight;
		for (size_t g = 0; g < t184.name.size(); g++) { nonempty = nonempty && members[g] > 0; weights_ok = weights_ok && gw[g] == t184.rep[g].weight; tw += gw[g]; }
		check(t184.name.size() == 184 && in_range && nonempty, "the refinement ends with exactly 184 non-empty named groups");
		check(weights_ok && tw == 22100, "the 184 groups' weights are the sums of their members' and add up to 22,100");
		std::set<std::string> distinct(t184.name.begin(), t184.name.end());
		check(distinct.size() == 184, "the 184 group names are different");
	}

	// wetness is a result, not a category: no group name contains it, and it behaves as it should
	{
		// rainbow, far apart ranks: dry. monotone and tight: the wettest. paired with a flush draw and tight ranks: wet.
		const Flop* dry = nullptr;
		for (const Flop& x : f) {
			const int r0 = x.cards[0] >> 2, r1 = x.cards[1] >> 2, r2 = x.cards[2] >> 2;
			if (!dry && x.suit_kind == 0 && r2 == 11 && r1 == 5 && r0 == 0) dry = &x;  // ranks K, 7, 2 in three suits
		}
		const Flop* wettest = nullptr;
		const Flop* paired_wet = nullptr;
		for (const Flop& x : f) {
			if (!wettest && x.suit_kind == 2 && flop_span_ace_low(x) <= 2 && !x.paired) wettest = &x;
			if (!paired_wet && x.paired && !x.trips && x.suit_kind == 1 && flop_wetness(x) >= 3) paired_wet = &x;
		}
		bool names_clean = true;
		for (int depth = 1; depth <= 6; depth++)
			for (const Flop& x : f) names_clean = names_clean && flop_group_name(x, depth).find("乾") == std::string::npos &&
			                                  flop_group_name(x, depth).find("濕") == std::string::npos;
		check(names_clean, "no group name carries dry or wet: they are results, not categories");
		check(dry && flop_wetness(*dry) == 0, "K-7-2 rainbow is dry (wetness 0)");
		bool trips_dry = true;
		for (const Flop& x : f)
			if (x.trips) trips_dry = trips_dry && flop_wetness(x) == 0;
		check(trips_dry, "trips boards (one rank, no draws of any kind) are dry");
		check(wettest && flop_wetness(*wettest) == 4, "a tight monotone board is the wettest (wetness 4)");
		check(paired_wet != nullptr, "a paired board with a flush draw and a tight straight structure is wet (>= 3)");
		int wc[5] = {0, 0, 0, 0, 0};
		for (const Flop& x : f) wc[flop_wetness(x)] += x.weight;
		check(wc[0] > 0 && wc[4] > 0 && wc[0] + wc[1] + wc[2] + wc[3] + wc[4] == 22100, "all five wetness levels occur and cover the 22,100 flops");
		printf("  (flops by wetness: dry %d, fairly dry %d, medium %d, wet %d, very wet %d)\n", wc[0], wc[1], wc[2], wc[3], wc[4]);
	}

	// the dry-wet spectrum of the 184 (or 182) groups: a result placed on the groups after the classification
	{
		FlopTaxonomy t = flop_taxonomy(182);
		double lo = 99, hi = -1;
		bool in_range = t.wetness.size() == t.name.size();
		for (double w : t.wetness) { lo = std::min(lo, w); hi = std::max(hi, w); in_range = in_range && w >= 0 && w <= 4; }
		check(in_range && lo < 0.5 && hi > 3.0, "the groups spread over the dry-wet spectrum (driest below 0.5, wettest above 3)");
		// a rainbow group with far-apart ranks is drier than a monotone group with tight ranks
		double rb = -1, mono = -1;
		for (size_t g = 0; g < t.name.size(); g++) {
			if (t.name[g].find("彩虹") != std::string::npos && t.name[g].find("順遠") != std::string::npos) rb = std::max(rb, t.wetness[g]);
			if (t.name[g].find("單色") != std::string::npos && t.name[g].find("順緊") != std::string::npos) mono = std::max(mono, t.wetness[g]);
		}
		check(rb >= 0 && mono >= 0 && rb < 1.0 && mono > 3.0, "rainbow far-apart groups sit at the dry end (< 1), monotone tight groups at the wet end (> 3)");
		// a group's wetness is the weighted mean of its members': the weighted mean over all groups equals the mean over all flops
		double lhs = 0, rhs = 0;
		for (size_t g = 0; g < t.name.size(); g++) lhs += t.wetness[g] * t.rep[g].weight;
		for (const Flop& x : f) rhs += (double)x.weight * flop_wetness(x);
		check(std::fabs(lhs - rhs) < 1e-6 * 22100, "the groups' wetness averages back to the mean wetness of all flops");
	}

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

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

	// wetness is a result, not a category: it is counted over the 1,176 hole-card hands of each flop
	{
		const Flop* dry = nullptr;
		const Flop* mono_tight = nullptr;
		const Flop* mono_far = nullptr;
		for (const Flop& x : f) {
			const int r0 = x.cards[0] >> 2, r1 = x.cards[1] >> 2, r2 = x.cards[2] >> 2;
			if (!dry && x.suit_kind == 0 && r2 == 11 && r1 == 5 && r0 == 0) dry = &x;  // K, 7, 2 in three suits
			if (!mono_tight && x.suit_kind == 2 && r2 == 7 && r1 == 6 && r0 == 5) mono_tight = &x;  // 9, 8, 7 in one suit
			if (!mono_far && x.suit_kind == 2 && r2 == 11 && r1 == 5 && r0 == 0) mono_far = &x;  // K, 7, 2 in one suit
		}
		check(dry && mono_tight && mono_far, "found K72 rainbow, 987 monotone, K72 monotone");
		FlopWetness d = flop_wetness_detail(*dry), m = flop_wetness_detail(*mono_tight), mf = flop_wetness_detail(*mono_far);
		// by hand, K72 rainbow: no suit has four cards and no five ranks can be in a row, so no flush, straight or draw of any kind
		check(d.count[0] == 0 && d.count[1] == 0 && d.count[2] == 0 && d.count[3] == 0 && d.count[4] == 0 && d.share == 0,
		      "K72 rainbow: no hand makes a flush, a straight or any draw (counted: 0 of 1,176)");
		// by hand, a monotone board: 10 cards of the suit are left. A flush needs both hole cards in it: C(10,2) = 45 hands.
		// A flush draw (four of the suit) needs exactly one: 10 * 39 = 390 hands. Three of the suit in hand cannot happen (2 cards).
		check(mf.count[0] == 45 && mf.count[2] == 390, "monotone board: 45 hands make a flush and 390 have a flush draw (C(10,2) and 10 x 39, counted by hand)");
		check(m.count[0] == 45 && m.count[2] == 390 && m.count[3] > 0 && m.count[1] > 0, "987 monotone: the same flush counts, and also open-ended draws and made straights");
		check(m.share > mf.share && mf.share > d.share, "wetness order: 987 monotone > K72 monotone > K72 rainbow");
		check(flop_wetness(*dry) == 0 && flop_wetness(*mono_tight) == 4, "K72 rainbow is dry (0), 987 monotone is very wet (4)");
		check(m.types == 5 && mf.types == 2 && d.types == 0, "types that occur: 987 monotone 5 (flush, straight, FD, OESD, GUT), K72 monotone 2 (flush, FD), K72 rainbow 0 (its 198 backdoor flush draws are not a type)");

		int wc[5] = {0, 0, 0, 0, 0};
		for (const Flop& x : f) wc[flop_wetness(x)] += x.weight;
		check(wc[0] > 0 && wc[4] > 0 && wc[0] + wc[1] + wc[2] + wc[3] + wc[4] == 22100, "all five wetness levels occur and cover the 22,100 flops");
		printf("  (flops by wetness: dry %d, fairly dry %d, medium %d, wet %d, very wet %d)\n", wc[0], wc[1], wc[2], wc[3], wc[4]);
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

	{
		// the order by characteristics: suits -> high cards -> connectedness of the high cards -> of the low cards -> ...
		const std::vector<int> order = flop_shape_order();
		std::set<int> seen(order.begin(), order.end());
		check(order.size() == 1755 && seen.size() == 1755 && *seen.begin() == 0 && *seen.rbegin() == 1754,
		      "the order by characteristics lists every one of the 1,755 classes once");
		const int want[7] = {3, 12, 28, 55, 73, 123, 146};
		bool counts = true, contiguous = true;
		for (int depth = 1; depth <= 7; depth++) {
			std::set<std::string> names;
			std::string last;
			int runs = 0;
			for (int i : order) {
				const std::string n = flop_shape_name(f[i], depth);
				names.insert(n);
				if (n != last) { runs++; last = n; }
			}
			counts = counts && (int)names.size() == want[depth - 1];
			contiguous = contiguous && runs == (int)names.size();
		}
		check(counts, "groups per level: 3 suits, 12 with the high cards, 28, 55, 73 with three-in-one-straight, 123, 146");
		check(contiguous, "in the order every group of every level is one unbroken run");
		auto name_of = [&](const char* flop, int depth) {
			for (const Flop& x : f)
				if (flop_name(x) == flop) return flop_shape_name(x, depth);
			return std::string("?");
		};
		check(name_of("AsQs3s", 4) == "單色｜2高｜高跳" && name_of("KsQs6s", 4) == "單色｜2高｜高連" && name_of("KsJsTs", 4) == "單色｜3高｜高一連一跳",
		      "A-Q-3 two high cards with a gap, K-Q-6 two connected high cards, K-J-T three high cards");
		check(name_of("AsAhJs", 7) == "雙花｜3高｜高對＋跳｜同花雙高" && name_of("AsKsKh", 7) == "雙花｜3高｜高對＋連｜同花雙高",
		      "A-A-J a pair with a gap, A-K-K a pair with a connected card, both with two high cards suited");
		check(name_of("4s3s2s", 4) == "單色｜0高｜低三連" && name_of("6s4s3s", 4) == "單色｜0高｜低一連一跳" && name_of("Kd7h2s", 6) == "三花｜1高｜低斷｜不同順｜高低斷",
		      "4-3-2 three low in a row, 6-4-3 one connected one gapped, K-7-2 nothing connects");
		check(name_of("Ad5h4s", 5) == "三花｜1高｜低連｜同順" && name_of("Ad3h2s", 5) == "三花｜1高｜低連｜同順" && name_of("Ad6h5s", 5) == "三花｜1高｜低連｜不同順",
		      "the ace also sits under the low cards: A-5-4 and A-3-2 are in one straight (A2345), A-6-5 is not");
		check(name_of("AdKh2s", 5) == "三花｜2高｜高連｜不同順" && name_of("JdTh9s", 5) == "三花｜2高｜高連｜同順" && name_of("7d3h2s", 5) == "三花｜0高｜低一連一跳｜不同順" &&
		          name_of("7d5h4s", 5) == "三花｜0高｜低一連一跳｜同順",
		      "A-K-2 and J-T-9 are apart, 7-3-2 and 7-5-4 are apart (they were together before the fifth level)");
	}

	printf(failures ? "FAIL\n" : "PASS\n");
	return failures ? 1 : 0;
}

// Checks for flops.cuh (host only, no GPU needed): 1,755 classes, weights that add up to 22,100, and the structure
// the classes must have.
//
//   nvcc -O3 -o test_flops test_flops.cu && ./test_flops

#include "flops.cuh"

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

	printf(failures ? "FAIL\n" : "PASS\n");
	return failures ? 1 : 0;
}

// Checks for flopstats.cuh: the GPU numbers against a slow, independent host computation (its own evaluator, plain
// loops over every runout), the two sides adding up to 1, suit-renamed flops giving the same answer, and the relay.
//
//   nvcc -O3 -arch=sm_89 -o test_flopstats test_flopstats.cu && ./test_flopstats
#include "flopstats.cuh"

static int failures = 0;
static void check(bool ok, const char* what) {
	printf("  %s  %s\n", ok ? "ok " : "BAD", what);
	failures += !ok;
}

// -- slow reference (host only, shares nothing with eval7) ----------------------------------------------------------
static uint64_t ref5(const int* c) {
	int r[5], cnt[13] = {0};
	bool flush = true;
	for (int i = 0; i < 5; i++) { r[i] = c[i] >> 2; cnt[r[i]]++; flush = flush && (c[i] & 3) == (c[0] & 3); }
	// ranks ordered by (count, rank) descending
	std::sort(r, r + 5, [&](int a, int b) { return cnt[a] != cnt[b] ? cnt[a] > cnt[b] : a > b; });
	int distinct = 0, hi = -1, lo = 99;
	for (int i = 0; i < 13; i++)
		if (cnt[i]) { distinct++; hi = std::max(hi, i); lo = std::min(lo, i); }
	int straight = -1;
	if (distinct == 5) {
		if (hi - lo == 4) straight = hi;
		else if (cnt[12] && cnt[0] && cnt[1] && cnt[2] && cnt[3]) straight = 3;  // the wheel
	}
	int cat;
	if (straight >= 0 && flush) cat = 8;
	else if (cnt[r[0]] == 4) cat = 7;
	else if (cnt[r[0]] == 3 && cnt[r[3]] == 2) cat = 6;
	else if (flush) cat = 5;
	else if (straight >= 0) cat = 4;
	else if (cnt[r[0]] == 3) cat = 3;
	else if (cnt[r[0]] == 2 && cnt[r[2]] == 2) cat = 2;
	else if (cnt[r[0]] == 2) cat = 1;
	else cat = 0;
	uint64_t v = cat;
	if (straight >= 0) return (v << 20) | ((uint64_t)straight << 16);
	for (int i = 0; i < 5; i++) v = (v << 4) | r[i];
	return v;
}
static uint64_t ref7(const int* c) {
	uint64_t best = 0;
	for (int a = 0; a < 7; a++)
		for (int b = a + 1; b < 7; b++) {
			int h[5], n = 0;
			for (int i = 0; i < 7; i++)
				if (i != a && i != b) h[n++] = c[i];
			best = std::max(best, ref5(h));
		}
	return best;
}

static void reference(const FsSide& A, const FsSide& B, const int flop[3], double* out) {
	auto side = [&](const FsSide& me, const FsSide& them, double& eq, double* bins) {
		std::vector<double> win(me.n, 0), tot(me.n, 0);
		for (int t = 0; t < 52; t++)
			for (int u = t + 1; u < 52; u++) {
				if (t == flop[0] || t == flop[1] || t == flop[2] || u == flop[0] || u == flop[1] || u == flop[2]) continue;
				auto dead = [&](int c) { return c == flop[0] || c == flop[1] || c == flop[2] || c == t || c == u; };
				std::vector<uint64_t> rm(me.n, 0), rt(them.n, 0);
				auto fill = [&](const FsSide& s, std::vector<uint64_t>& r) {
					for (int i = 0; i < s.n; i++) {
						const int h0 = s.cards[i * 2], h1 = s.cards[i * 2 + 1];
						if (dead(h0) || dead(h1)) continue;
						const int c[7] = {flop[0], flop[1], flop[2], t, u, h0, h1};
						r[i] = ref7(c) + 1;
					}
				};
				fill(me, rm); fill(them, rt);
				for (int i = 0; i < me.n; i++) {
					if (!rm[i]) continue;
					for (int j = 0; j < them.n; j++) {
						if (!rt[j]) continue;
						if (me.cards[i * 2] == them.cards[j * 2] || me.cards[i * 2] == them.cards[j * 2 + 1] || me.cards[i * 2 + 1] == them.cards[j * 2] ||
						    me.cards[i * 2 + 1] == them.cards[j * 2 + 1])
							continue;
						tot[i] += them.w[j];
						win[i] += them.w[j] * (rm[i] > rt[j] ? 1.0 : (rm[i] == rt[j] ? 0.5 : 0.0));
					}
				}
			}
		fs_reduce(me, win.data(), tot.data(), eq, bins);
	};
	side(A, B, out[0], out + 2);
	side(B, A, out[1], out + 2 + FS_BINS);
}

static std::vector<double> range_of(const char* text) { return parse_weighted(text); }
static int card(const char* s) { return rank_of(s[0]) * 4 + (int)(strchr("shdc", s[1]) - "shdc"); }

int main() {
	setup_classes();
	FsDevice dev;

	// 1. against the reference: weighted ranges, flops of every suit kind, paired and trips
	{
		std::vector<double> a = range_of("QQ+,AKs,AQs,KQs,JTs,76s,AKo"), b = range_of("TT-88,AJs,KJs,QJs,T9s,98s,AQo,55");
		for (int c = 0; c < NCLASS; c++) {  // mixed weights, as the charts have
			if (g_names[c] == "AKo") a[c] = 0.5;
			if (g_names[c] == "55") b[c] = 0.25;
		}
		const FsSide A = fs_side(a), B = fs_side(b);
		const char* texts[6][3] = {{"As", "7d", "2c"}, {"Jh", "Th", "9s"}, {"Qs", "8s", "3s"}, {"Ks", "Kd", "4c"}, {"9c", "9d", "9s"}, {"5d", "4h", "As"}};
		double worst = 0, worst_sum = 0;
		for (auto& t : texts) {
			int flop[3] = {card(t[0]), card(t[1]), card(t[2])};
			std::sort(flop, flop + 3);
			const uint8_t cards[3] = {(uint8_t)flop[0], (uint8_t)flop[1], (uint8_t)flop[2]};
			double got[FS_K], want[FS_K];
			dev.run(A, B, cards, 1, got);
			reference(A, B, flop, want);
			for (int k = 0; k < FS_K; k++) worst = std::max(worst, fabs(got[k] - want[k]));
			worst_sum = std::max(worst_sum, fabs(got[0] + got[1] - 1));
		}
		printf("      largest difference from the reference %.2e, largest |eq + eq - 1| %.2e\n", worst, worst_sum);
		check(worst < 1e-12, "equity and the ten bins of both ranges equal the host reference on six flops");
		check(worst_sum < 1e-12, "the two ranges' equities add up to 1");
	}

	// 2. a known spot: aces against kings on a dry flop, and on a king-high flop
	{
		const FsSide A = fs_side(range_of("AA")), B = fs_side(range_of("KK"));
		int f1[3] = {card("2c"), card("7d"), card("9h")}, f2[3] = {card("2c"), card("7d"), card("Kh")};
		std::sort(f1, f1 + 3); std::sort(f2, f2 + 3);
		const uint8_t cards[6] = {(uint8_t)f1[0], (uint8_t)f1[1], (uint8_t)f1[2], (uint8_t)f2[0], (uint8_t)f2[1], (uint8_t)f2[2]};
		double got[2 * FS_K];
		dev.run(A, B, cards, 2, got);
		printf("      AA vs KK: %.4f on 9-7-2, %.4f on K-7-2\n", got[0], got[FS_K]);
		check(got[0] > 0.89 && got[0] < 0.93, "aces are about 91% against kings on 9-7-2 rainbow");
		check(got[FS_K] < 0.12, "and under 12% once a king is on the flop");
		check(fs_share_from(got + 2, 8) > 0.99 && fs_share_from(got + FS_K + 2, 8) < 0.01, "the bins say the same: all of AA is above 80% on the first flop, none on the second");
	}

	// 3. renaming the suits changes nothing (the ranges are suit-symmetric), so one flop per class is enough
	{
		const FsSide A = fs_side(range_of("77+,ATs+,KTs+,QTs+,JTs,AJo+,KQo")), B = fs_side(range_of("JJ-22,AQs-A2s,KQs-K9s,QJs-Q9s,JTs-J9s,T9s,98s,87s,76s,AQo-AJo,KQo"));
		int f1[3] = {card("Kh"), card("9h"), card("6c")}, f2[3] = {card("Kd"), card("9d"), card("6s")};
		std::sort(f1, f1 + 3); std::sort(f2, f2 + 3);
		const uint8_t cards[6] = {(uint8_t)f1[0], (uint8_t)f1[1], (uint8_t)f1[2], (uint8_t)f2[0], (uint8_t)f2[1], (uint8_t)f2[2]};
		double got[2 * FS_K], worst = 0;
		dev.run(A, B, cards, 2, got);
		for (int k = 0; k < FS_K; k++) worst = std::max(worst, fabs(got[k] - got[FS_K + k]));
		check(worst < 1e-12, "Kh9h6c and Kd9d6s give the same numbers");
	}

	// 4. the relay: a run that is stopped twice and carried on equals an uninterrupted one
	{
		std::vector<FsMatchup> ms(2);
		ms[0].key = "a"; ms[0].aggressor = range_of("AA,AKs"); ms[0].caller = range_of("KK,QJs");
		ms[1].key = "b"; ms[1].aggressor = range_of("QQ,JTs"); ms[1].caller = range_of("99,AQs");
		const std::string path = "/tmp/test_flopstats.relay";
		remove(path.c_str());
		uint64_t done = 0;
		fs_run(ms, path, false, 5, true, &done);
		const uint64_t first = done;
		fs_run(ms, path, false, 30, true, &done);
		const uint64_t second = done;
		const std::vector<double> relayed = fs_run(ms, path, false, 0, true, &done);
		const std::vector<double> straight = fs_run(ms, "", false);
		double worst = 0, worst_sum = 0;
		for (size_t i = 0; i < straight.size(); i++) worst = std::max(worst, fabs(straight[i] - relayed[i]));
		for (size_t i = 0; i < straight.size() / FS_K; i++) worst_sum = std::max(worst_sum, fabs(straight[i * FS_K] + straight[i * FS_K + 1] - 1));
		check(first == 5 * FS_BATCH && second == 35 * FS_BATCH && done == 2 * 1755, "stopped after 5 and after 35 batches, then finished all 3,510 items");
		check(worst == 0, "the relayed result is identical to the uninterrupted one");
		check(worst_sum < 1e-12, "on all 1,755 flops of both matchups the equities add up to 1");
		// a restart with other ranges must not use the file
		ms[1].caller = range_of("99,AJs");
		fs_run(ms, path, false, 1, true, &done);
		check(done == FS_BATCH, "a relay file for other ranges is ignored");
		remove(path.c_str());
	}

	// 5. the groups line up with flops.cu: 73 at level 5, 383 at level 7 (the tree), every flop in one
	{
		std::vector<int> group, base;
		const std::vector<std::string> names = fs_groups(7, group), names5 = fs_groups(5, base);
		bool all = true;
		for (int g : group) all = all && g >= 0 && g < (int)names.size();
		check(names5.size() == 73 && names.size() == 383 && all, "73 groups at level 5, 383 at level 7, and every flop class has one");
	}

	printf(failures ? "FAIL\n" : "PASS\n");
	return failures ? 1 : 0;
}

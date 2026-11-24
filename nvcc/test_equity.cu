// Checks for the equity tool: the GPU evaluator against a slow, independent host reference, the hand class
// tables, and the equity computation itself (exact, sampled). Prints PASS and exits 0, or exits 1 on any mismatch.
//
//   nvcc -O3 -arch=sm_89 -o test_equity test_equity.cu && ./test_equity [deals]

#include "equity.cuh"

__global__ void test_kernel(const uint8_t* deals, int n, int8_t* out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const uint8_t* d = deals + 9 * i;
	int ha[7], hb[7];
	ha[0] = d[0]; ha[1] = d[1]; hb[0] = d[2]; hb[1] = d[3];
	for (int k = 0; k < 5; k++) ha[2 + k] = hb[2 + k] = d[4 + k];
	uint32_t sa = eval7(ha), sb = eval7(hb);
	out[i] = sa > sb ? 1 : (sa < sb ? -1 : 0);
}


// -- slow reference for the self test (host only) -----------------------------------------------
// Independent of eval7: ranks 5-card hands by sorting rank groups, then takes the best of the 21 subsets.
static uint64_t eval5_ref(const int* c) {
	int r[5], su[5];
	for (int i = 0; i < 5; i++) { r[i] = c[i] >> 2; su[i] = c[i] & 3; }
	bool flush = true;
	for (int i = 1; i < 5; i++) flush = flush && su[i] == su[0];
	int cnt[13] = {};
	for (int i = 0; i < 5; i++) cnt[r[i]]++;
	int grank[5], gcnt[5], ng = 0;
	for (int k = 5; k >= 1; k--)  // groups by size desc, then rank desc
		for (int rk = 12; rk >= 0; rk--)
			if (cnt[rk] == k) { grank[ng] = rk; gcnt[ng] = k; ng++; }
	int distinct = 0, hi = -1, lo = 13;
	for (int rk = 0; rk < 13; rk++)
		if (cnt[rk]) { distinct++; hi = rk; if (lo == 13) lo = rk; }
	int straight_top = -1;
	if (distinct == 5) {
		if (hi - lo == 4) straight_top = hi;
		else if (hi == 12 && cnt[0] && cnt[1] && cnt[2] && cnt[3]) straight_top = 3;
	}
	auto pack = [&](int n) { uint64_t v = 0; for (int i = 0; i < n; i++) v = v << 4 | grank[i]; return v; };
	if (straight_top >= 0 && flush) return 8ull << 40 | straight_top;
	if (gcnt[0] == 4) return 7ull << 40 | pack(ng);
	if (gcnt[0] == 3 && gcnt[1] == 2) return 6ull << 40 | pack(ng);
	if (flush) return 5ull << 40 | pack(ng);
	if (straight_top >= 0) return 4ull << 40 | straight_top;
	if (gcnt[0] == 3) return 3ull << 40 | pack(ng);
	if (gcnt[0] == 2 && gcnt[1] == 2) return 2ull << 40 | pack(ng);
	if (gcnt[0] == 2) return 1ull << 40 | pack(ng);
	return pack(ng);
}

static uint64_t eval7_ref(const int* c) {
	uint64_t best = 0;
	for (int skip1 = 0; skip1 < 7; skip1++)
		for (int skip2 = skip1 + 1; skip2 < 7; skip2++) {
			int five[5], n = 0;
			for (int i = 0; i < 7; i++)
				if (i != skip1 && i != skip2) five[n++] = c[i];
			uint64_t v = eval5_ref(five);
			if (v > best) best = v;
		}
	return best;
}

static uint64_t g_rng = 88172645463325252ull;
static uint32_t rnd(uint32_t n) {
	g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
	return (uint32_t)(((g_rng >> 32) * n) >> 32);
}

// Random deals (a0 a1 b0 b1 board x5), all distinct cards, drawn from `deck`.
static std::vector<uint8_t> random_deals(int n, const std::vector<int>& deck) {
	std::vector<uint8_t> out(9 * (size_t)n);
	for (int i = 0; i < n; i++) {
		std::vector<int> d = deck;
		for (int k = 0; k < 9; k++) {
			int j = k + rnd((uint32_t)d.size() - k);
			std::swap(d[k], d[j]);
			out[9 * (size_t)i + k] = (uint8_t)d[k];
		}
	}
	return out;
}

static long long selftest_deals(const std::vector<uint8_t>& deals, int n) {
	uint8_t* d_in;
	int8_t* d_out;
	CK(cudaMalloc(&d_in, deals.size()));
	CK(cudaMalloc(&d_out, n));
	CK(cudaMemcpy(d_in, deals.data(), deals.size(), cudaMemcpyHostToDevice));
	test_kernel<<<(n + 255) / 256, 256>>>(d_in, n, d_out);
	CK(cudaGetLastError());
	std::vector<int8_t> got(n);
	CK(cudaMemcpy(got.data(), d_out, n, cudaMemcpyDeviceToHost));
	cudaFree(d_in);
	cudaFree(d_out);
	long long bad = 0;
	for (int i = 0; i < n; i++) {
		const uint8_t* d = &deals[9 * (size_t)i];
		int ha[7], hb[7];
		ha[0] = d[0]; ha[1] = d[1]; hb[0] = d[2]; hb[1] = d[3];
		for (int k = 0; k < 5; k++) ha[2 + k] = hb[2 + k] = d[4 + k];
		uint64_t ra = eval7_ref(ha), rb = eval7_ref(hb);
		int ref = ra > rb ? 1 : (ra < rb ? -1 : 0);
		bad += ref != got[i];
	}
	return bad;
}

int main(int argc, char** argv) {
	setup_classes();
	int n = argc >= 2 ? atoi(argv[1]) : 200000;
	int failures = 0;

	// 1. evaluator vs reference. The second batch draws from a 32-card deck (A and 2 to 8) so that quads,
	//    full houses and straight flushes are common.
	std::vector<int> full, small;
	for (int c = 0; c < 52; c++) {
		full.push_back(c);
		if ((c >> 2) <= 6 || (c >> 2) == 12) small.push_back(c);
	}
	long long bad_full = selftest_deals(random_deals(n, full), n);
	long long bad_small = selftest_deals(random_deals(n, small), n);
	printf("evaluator: %lld mismatches in %d random deals, %lld in %d deals from a 32-card deck\n", bad_full, n, bad_small, n);
	failures += (bad_full || bad_small);

	// 2. class tables: 169 classes, 1326 distinct two-card combos, the right size per class, ranges.ts order.
	int total = 0;
	bool ok = g_names.size() == NCLASS && g_names[0] == "AA" && g_names[1] == "AKs" && g_names[2] == "AKo" &&
	          g_names[168] == "22";
	bool seen[52][52] = {};
	for (int i = 0; i < NCLASS && ok; i++) {
		int want = g_names[i].size() == 2 ? 6 : (g_names[i][2] == 's' ? 4 : 12);
		ok = ok && g_ncombo[i] == want;
		for (int k = 0; k < g_ncombo[i]; k++) {
			int a = g_combos[i][k][0], b = g_combos[i][k][1];
			int lo = a < b ? a : b, hi = a < b ? b : a;
			ok = ok && a != b && !seen[lo][hi];
			seen[lo][hi] = true;
			total++;
		}
	}
	ok = ok && total == 1326;
	printf("class tables: %s (%d combos)\n", ok ? "ok" : "BAD", total);
	failures += !ok;


	// 3. exact enumeration: reference values (from this tool's first full run, close to the usual 81.9% and
	//    46.0%), a class against itself is exactly 50%, and swapping the two classes gives 1 - equity.
	{
		auto exact1 = [&](const char* a, const char* b) {
			std::vector<double> eq;
			std::vector<unsigned long long> nj;
			exact_pairs({make_int2(class_index(a), class_index(b))}, eq, nj, false);
			return eq[0];
		};
		double aakk = exact1("AA", "KK"), kkaa = exact1("KK", "AA"), akq = exact1("AKs", "QQ");
		double aa_aa = exact1("AA", "AA"), aks_aks = exact1("AKs", "AKs");
		bool ex = fabs(aakk - 0.819461) < 1e-6 && fabs(akq - 0.460485) < 1e-6 && fabs(aakk + kkaa - 1) < 1e-12 &&
		          fabs(aa_aa - 0.5) < 1e-12 && fabs(aks_aks - 0.5) < 1e-12;
		printf("exact: AA v KK %.6f, KK v AA %.6f, AKs v QQ %.6f, AA v AA %.6f, AKs v AKs %.6f  %s\n", aakk, kkaa, akq,
		       aa_aa, aks_aks, ex ? "ok" : "BAD");
		failures += !ex;

		// 4. sampling against exact on a spread of matchups: within 5 standard errors at 4,000,000 samples.
		const char* names[][2] = {{"AA", "72o"}, {"AKs", "QQ"}, {"JTs", "AKo"}, {"22", "43s"}, {"A5s", "KQo"}, {"87s", "AJo"}};
		double worst = 0;
		for (auto& pr : names) {
			std::vector<double> eq;
			std::vector<unsigned long long> v;
			run_pairs({make_int2(class_index(pr[0]), class_index(pr[1]))}, 4000000, 99, eq, v);
			double e = exact1(pr[0], pr[1]);
			double z = fabs(eq[0] - e) / sqrt(e * (1 - e) / (double)v[0]);
			if (z > worst) worst = z;
		}
		printf("sampled vs exact: worst difference %.2f standard errors (limit 5)  %s\n", worst, worst < 5 ? "ok" : "BAD");
		failures += worst >= 5;
	}

	// 5. relay: an interrupted run, carried on from its checkpoint, gives exactly the numbers of an uninterrupted one
	{
		const char* ck1 = "/tmp/test_equity_exact.ckpt";
		const char* ck2 = "/tmp/test_equity_sampled.ckpt";
		remove(ck1);
		remove(ck2);
		std::vector<int2> pr = {make_int2(class_index("AKs"), class_index("QQ")), make_int2(class_index("AA"), class_index("KK")),
		                        make_int2(class_index("JTs"), class_index("AKo")), make_int2(class_index("22"), class_index("43s")),
		                        make_int2(class_index("87s"), class_index("AJo"))};
		std::vector<double> whole, part, resumed;
		std::vector<unsigned long long> nj, nj2, v;
		exact_pairs(pr, whole, nj, false);
		exact_pairs(pr, part, nj2, false, ck1, 16, 4);     // 4 batches of 16 jobs, then "killed"
		exact_pairs(pr, resumed, nj2, false, ck1, 16, 0);  // the same command again
		bool same = whole.size() == resumed.size() && part != whole;
		for (size_t i = 0; i < whole.size(); i++) same = same && whole[i] == resumed[i];
		printf("relay, exact: interrupted after 4 batches, then resumed: %s\n", same ? "identical to the uninterrupted run, ok" : "BAD");
		failures += !same;

		std::vector<double> sw, sp, sr;
		run_pairs(pr, 1000000, 5, sw, v);
		run_pairs(pr, 1000000, 5, sp, v, false, 2, ck2, 1);  // 1 batch of 2 pairs, then "killed"
		run_pairs(pr, 1000000, 5, sr, v, false, 2, ck2, 0);
		bool same2 = sw.size() == sr.size() && sp != sw;
		for (size_t i = 0; i < sw.size(); i++) same2 = same2 && sw[i] == sr[i];
		printf("relay, sampled: interrupted after 1 batch, then resumed: %s\n", same2 ? "identical to the uninterrupted run, ok" : "BAD");
		failures += !same2;
		remove(ck1);
		remove(ck2);
	}

	printf(failures ? "FAIL\n" : "PASS\n");
	return failures ? 1 : 0;
}

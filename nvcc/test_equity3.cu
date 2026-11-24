// Checks for the three-way equity table. Prints PASS and exits 0, or FAIL and exits 1.
//
//   nvcc -O3 -arch=sm_89 -o test_equity3 test_equity3.cu
//   ./test_equity3 equity3.bin equity3-se.bin [host_deals_per_triple]
//
// 1. Table structure: every triple's N equals an independent count, N = 0 exactly for impossible triples, the
//    three shares of every possible triple sum to 1, symmetric players have equal shares.
// 2. Lookup: any player order gives the permuted answer.
// 3. Exact enumeration (GPU, every combo triple on all C(46,5) boards) against the table, in standard errors.
// 4. Independent slow host reference on random deals (21-subset evaluator, own dealing code) against the table.
// 5. Sampler symmetry: the sampling kernel on permuted class orders gives permuted answers.
// 6. A provable identity: with all combos equally likely for all three players, each player averages exactly
//    1/3 (the players are exchangeable and the shares sum to 1).
// 8. The relay: sampling, exact enumeration and the generator, stopped and resumed (also from a checkpoint of the old
//    format), give the same results as an uninterrupted run, bit for bit.
// 7. The range function: GPU = CPU, permuting the ranges permutes the answers, per-player masses agree, and
//    against a direct combo-level Monte Carlo on realistic ranges (error reported).

#include "equity3.cuh"

// -- slow reference (host only), copied from test_equity.cu so that this test stands alone ---------
static uint64_t eval5_ref(const int* c) {
	int r[5], su[5];
	for (int i = 0; i < 5; i++) { r[i] = c[i] >> 2; su[i] = c[i] & 3; }
	bool flush = true;
	for (int i = 1; i < 5; i++) flush = flush && su[i] == su[0];
	int cnt[13] = {};
	for (int i = 0; i < 5; i++) cnt[r[i]]++;
	int grank[5], gcnt[5], ng = 0;
	for (int k = 5; k >= 1; k--)
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
	for (int s1 = 0; s1 < 7; s1++)
		for (int s2 = s1 + 1; s2 < 7; s2++) {
			int five[5], n = 0;
			for (int i = 0; i < 7; i++)
				if (i != s1 && i != s2) five[n++] = c[i];
			best = std::max(best, eval5_ref(five));
		}
	return best;
}

static uint64_t g_rng = 88172645463325252ull;
static uint32_t rnd(uint32_t n) {
	g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
	return (uint32_t)(((g_rng >> 32) * n) >> 32);
}

static int failures = 0;
static void report(bool ok, const char* what) {
	printf("  %s: %s\n", what, ok ? "ok" : "BAD");
	failures += !ok;
}

// -- 6 players' combo-level Monte Carlo on the GPU (own sampling code, independent of e3_sample_kernel) --
struct ComboEntry { int cls, c0, c1; };

__global__ void mc_kernel(const ComboEntry* combos, const double* cdf, int ncombo, uint32_t per_thread, uint64_t seed,
                          unsigned long long* sum6, unsigned long long* sq6, unsigned long long* cnt) {
	__shared__ unsigned long long s_s[3 * NCLASS], s_q[3 * NCLASS], s_c[3 * NCLASS];
	for (int i = threadIdx.x; i < 3 * NCLASS; i += blockDim.x) { s_s[i] = 0; s_q[i] = 0; s_c[i] = 0; }
	__syncthreads();
	Rng r;
	r.s = mix64(seed ^ mix64((uint64_t)blockIdx.x * blockDim.x + threadIdx.x + 77));
	for (uint32_t it = 0; it < per_thread; it++) {
		int pick[3];
		for (int p = 0; p < 3; p++) {
			const double u = r.next() * (1.0 / 4294967296.0) * cdf[p * ncombo + ncombo - 1];
			int lo = 0, hi = ncombo - 1;
			while (lo < hi) {
				int mid = (lo + hi) >> 1;
				if (cdf[p * ncombo + mid] > u) hi = mid; else lo = mid + 1;
			}
			pick[p] = lo;
		}
		const ComboEntry e0 = combos[pick[0]], e1 = combos[pick[1]], e2 = combos[pick[2]];
		const uint64_t m0 = (1ull << e0.c0) | (1ull << e0.c1), m1 = (1ull << e1.c0) | (1ull << e1.c1),
		               m2 = (1ull << e2.c0) | (1ull << e2.c1);
		if ((m0 & m1) | (m0 & m2) | (m1 & m2)) continue;
		uint64_t used = m0 | m1 | m2;
		int h[3][7];
		h[0][0] = e0.c0; h[0][1] = e0.c1; h[1][0] = e1.c0; h[1][1] = e1.c1; h[2][0] = e2.c0; h[2][1] = e2.c1;
		for (int k = 0; k < 5; k++) {
			int x;
			do { x = r.below(52); } while ((used >> x) & 1);
			used |= 1ull << x;
			h[0][2 + k] = h[1][2 + k] = h[2][2 + k] = x;
		}
		uint32_t sc[3] = {eval7(h[0]), eval7(h[1]), eval7(h[2])};
		const uint32_t best = max(sc[0], max(sc[1], sc[2]));
		const int nw = (sc[0] == best) + (sc[1] == best) + (sc[2] == best);
		const int cl[3] = {e0.cls, e1.cls, e2.cls};
		for (int p = 0; p < 3; p++) {
			const unsigned w = sc[p] == best ? 6 / nw : 0;
			atomicAdd(&s_s[p * NCLASS + cl[p]], (unsigned long long)w);
			atomicAdd(&s_q[p * NCLASS + cl[p]], (unsigned long long)(w * w));
			atomicAdd(&s_c[p * NCLASS + cl[p]], 1ull);
		}
	}
	__syncthreads();
	for (int i = threadIdx.x; i < 3 * NCLASS; i += blockDim.x) {
		atomicAdd(&sum6[i], s_s[i]);
		atomicAdd(&sq6[i], s_q[i]);
		atomicAdd(&cnt[i], s_c[i]);
	}
}

// -- 8. the relay: stopped and resumed equals uninterrupted, bit for bit ---------------------------------------
static std::vector<uint8_t> slurp(const std::string& p) {
	std::vector<uint8_t> v;
	FILE* f = fopen(p.c_str(), "rb");
	if (!f) return v;
	uint8_t buf[1 << 16];
	size_t n;
	while ((n = fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
	fclose(f);
	return v;
}

static void test_relay() {
	printf("8. the relay\n");
	// sampling: 9,000 triples (two batches of 4096 and a rest), stopped after the first batch
	{
		std::vector<int3> tr;
		uint64_t s = 12345;
		auto rnd = [&]() { s = s * 6364136223846793005ull + 1442695040888963407ull; return (int)((s >> 33) % NCLASS); };
		for (int i = 0; i < 9000; i++) tr.push_back(make_int3(rnd(), rnd(), rnd()));
		const char* p = "/tmp/test_equity3_sample.ckpt";
		remove(p);
		const std::vector<E3Est> ref = e3_sample_ordered(tr, 4096, 777);
		const std::vector<E3Est> a = e3_sample_ordered(tr, 4096, 777, p, 1);
		report(relay_peek_done(p) == 4096, "sampling: stopped after 1 batch, the file holds 4096 jobs");
		const std::vector<E3Est> b = e3_sample_ordered(tr, 4096, 777, p);
		bool same = true;
		for (size_t i = 0; i < tr.size(); i++)
			same = same && b[i].n == ref[i].n && memcmp(b[i].eq, ref[i].eq, sizeof b[i].eq) == 0 && memcmp(b[i].se, ref[i].se, sizeof b[i].se) == 0;
		report(same, "sampling: resumed result equals the uninterrupted one, bit for bit");
		(void)a;
		remove(p);
	}
	// exact enumeration: 30 triples, more than 8192 combo triples in all (several batches)
	{
		std::vector<int3> tr;
		for (int i = 0; i < 30; i++) tr.push_back(make_int3(20 + i, 40 + 3 * i, 90 + 2 * i));  // together more than 8192 combo triples
		const char* p = "/tmp/test_equity3_exact.ckpt";
		remove(p);
		std::vector<std::array<double, 3>> ref, a, b;
		std::vector<uint64_t> n0, n1, n2;
		e3_exact(tr, ref, n0);
		uint64_t total = 0;
		for (uint64_t n : n0) total += n;
		e3_exact(tr, a, n1, p, 1);
		report(total > 8192 && relay_peek_done(p) == 8192, "exact: stopped after 1 batch, the file holds 8192 jobs");
		e3_exact(tr, b, n2, p);
		bool same = n0 == n2;
		for (size_t i = 0; i < tr.size(); i++) same = same && memcmp(ref[i].data(), b[i].data(), 3 * sizeof(double)) == 0;
		report(same, "exact: resumed result equals the uninterrupted one, bit for bit");
		remove(p);
	}
	// the generator: 2 rounds on a coarse target; uninterrupted, stopped and resumed twice, and from an old-format file
	{
		const char* A = "/tmp/test_e3_gen_a";
		const char* B = "/tmp/test_e3_gen_b";
		const char* C = "/tmp/test_e3_gen_c";
		for (const char* d : {A, B, C}) {
			if (system((std::string("rm -rf ") + d + " && mkdir -p " + d).c_str()) != 0) exit(1);
		}
		const double target = 0.003;
		const uint64_t nmax = 16384, n0 = 4096;
		e3_generate(A, target, nmax, n0);
		const std::vector<uint8_t> ta = slurp(std::string(A) + "/equity3.bin"), sa = slurp(std::string(A) + "/equity3-se.bin");
		report(ta.size() > 1000000 && sa.size() == E3_NTRIPLE, "generator: an uninterrupted run wrote the table and the se file");
		const bool s1 = !e3_generate(B, target, nmax, n0, 3);
		report(s1 && relay_peek_done(std::string(B) + "/ckpt.bin") == 3 * 60000, "generator: stopped after 3 batches (60,000 triples each)");
		const bool s2 = !e3_generate(B, target, nmax, n0, 20);  // restores 3 batches, round 0 has 14: ends it, a few batches of round 1
		report(s2, "generator: second run stopped in a later round");
		e3_generate(B, target, nmax, n0);
		report(slurp(std::string(B) + "/equity3.bin") == ta && slurp(std::string(B) + "/equity3-se.bin") == sa,
		       "generator: stopped twice and resumed gives the same table and se files, byte for byte");
		// a checkpoint of the old format taken at the end of round 0
		uint64_t nvalid = 0;
		for (const int4& c : e3_all_triples()) nvalid += c.w > 0;
		e3_generate(C, target, nmax, n0, (nvalid + 59999) / 60000);
		{
			FILE* f = fopen((std::string(C) + "/ckpt.bin").c_str(), "rb");
			fseek(f, 4 + 32, SEEK_SET);
			E3Gen g;
			g.acc.assign((size_t)E3_NTRIPLE * E3_ACC, 0);
			g.rounds.assign(E3_NTRIPLE, 0);
			uint32_t r = 0;
			const bool ok = e3_read_state(f, r, g);
			fclose(f);
			report(ok && r == 0, "generator: relay file at the end of round 0 is readable");
			f = fopen((std::string(C) + "/ckpt.bin").c_str(), "wb");
			uint32_t h[2] = {0x4B433345u, E3_NTRIPLE};
			fwrite(h, 4, 2, f);
			fwrite(g.rounds.data(), 1, g.rounds.size(), f);
			fwrite(g.acc.data(), 8, g.acc.size(), f);
			fclose(f);
		}
		e3_generate(C, target, nmax, n0);
		report(slurp(std::string(C) + "/equity3.bin") == ta && slurp(std::string(C) + "/equity3-se.bin") == sa,
		       "generator: carried on from an old-format checkpoint gives the same table, byte for byte");
		for (const char* d : {A, B, C})
			if (system((std::string("rm -rf ") + d).c_str()) != 0) exit(1);
	}
}

int main(int argc, char** argv) {
	if (argc < 3) { fprintf(stderr, "usage: test_equity3 equity3.bin equity3-se.bin [host_deals_per_triple]\n"); return 1; }
	setup_classes();
	e3_setup_masks();
	std::vector<uint16_t> table = e3_load_table(argv[1]);
	std::vector<uint8_t> se_b(E3_NTRIPLE);
	{
		FILE* f = fopen(argv[2], "rb");
		if (!f || fread(se_b.data(), 1, E3_NTRIPLE, f) != E3_NTRIPLE) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
		fclose(f);
	}
	const int host_deals = argc >= 4 ? atoi(argv[3]) : 20000;
	std::vector<int4> cls = e3_all_triples();
	auto se_of = [&](uint32_t idx) { return se_b[idx] * E3_SE_UNIT; };
	auto tri = [&](const char* a, const char* b, const char* c) { return make_int3(class_index(a), class_index(b), class_index(c)); };

	// 1. structure -------------------------------------------------------------------------------
	printf("1. table structure\n");
	{
		long long bad_n = 0, bad_flag = 0, bad_sum = 0, bad_sym = 0, nvalid = 0, nimp = 0;
		double se_max = 0;
		for (uint32_t t = 0; t < E3_NTRIPLE; t++) {
			const uint16_t* r = &table[4 * (size_t)t];
			bad_n += r[0] != cls[t].w;
			if (!cls[t].w) {
				nimp++;
				bad_flag += r[1] || r[2] || r[3];
				continue;
			}
			nvalid++;
			const int s = r[1] + r[2] + r[3];
			bad_sum += abs(s - 65535) > 2;
			const int c[3] = {cls[t].x, cls[t].y, cls[t].z};
			for (int p = 0; p < 3; p++)
				for (int q = p + 1; q < 3; q++) bad_sym += c[p] == c[q] && r[1 + p] != r[1 + q];
			se_max = std::max(se_max, se_of(t));
		}
		printf("  %lld possible, %lld impossible triples, largest SE %.4f%%\n", nvalid, nimp, se_max * 100);
		report(bad_n == 0, "N equals an independent count for every triple");
		report(bad_flag == 0 && nimp > 0, "impossible triples are all zero");
		report(bad_sum == 0, "the three shares sum to 1 (+-2/65535)");
		report(bad_sym == 0, "players with the same class have equal shares");
		report(se_max <= 0.001, "largest standard error is at most 0.1%");
		// AA AA AA, AA AA AA is impossible; AKs x4 suits: 4 AKs combos, three players -> possible
		report(cls[e3_index_sorted(0, 0, 0)].w == 0 && cls[e3_index_sorted(1, 1, 1)].w == 24, "AA AA AA impossible, AKs AKs AKs has 24 combo triples");
	}

	// 2. lookup in every player order ---------------------------------------------------------------
	printf("2. lookup\n");
	{
		long long bad = 0, n = 0;
		for (int s = 0; s < 20000; s++) {
			int c[3] = {(int)rnd(NCLASS), (int)rnd(NCLASS), (int)rnd(NCLASS)};
			if (s % 3 == 0) c[1] = c[0];
			float e[3];
			uint32_t N = e3_lookup(table.data(), c[0], c[1], c[2], e);
			static const int perm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
			for (auto& pm : perm) {
				float f[3];
				uint32_t M = e3_lookup(table.data(), c[pm[0]], c[pm[1]], c[pm[2]], f);
				bad += M != N;
				for (int p = 0; p < 3; p++) bad += f[p] != e[pm[p]];
				n++;
			}
		}
		report(bad == 0, "all 6 player orders of 20,000 random class triples agree");
	}

	// 3. exact enumeration vs table ---------------------------------------------------------------------
	printf("3. exact enumeration vs table\n");
	{
		std::vector<int3> ts = {tri("AA", "KK", "QQ"), tri("AA", "AA", "KK"), tri("AKs", "AKo", "QQ"), tri("AKs", "AKs", "AKs"),
		                       tri("JTs", "98s", "22"), tri("72o", "32o", "AA"), tri("KK", "KK", "QQ"), tri("AKo", "AKo", "AKo"),
		                       tri("T9s", "87s", "76s"), tri("A5s", "KQo", "55"),
		                       tri("QQ", "AA", "KK"), tri("T9s", "AKo", "22")};
		std::vector<std::array<double, 3>> eq;
		std::vector<uint64_t> nj;
		auto t0 = std::chrono::steady_clock::now();
		e3_exact(ts, eq, nj);
		double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		double worst_z = 0, worst_abs = 0, sumz2 = 0;
		int nzz = 0;
		bool njok = true;
		for (size_t t = 0; t < ts.size(); t++) {
			int c[3] = {ts[t].x, ts[t].y, ts[t].z};
			float e[3];
			uint32_t N = e3_lookup(table.data(), c[0], c[1], c[2], e);
			njok = njok && N == nj[t];
			int slot[3];
			uint32_t idx = e3_index(c, slot);
			// table shares are symmetrised for equal classes; the exact values already are
			for (int p = 0; p < 3; p++) {
				double d = fabs(e[p] - eq[t][p]);
				worst_abs = std::max(worst_abs, d);
				worst_z = std::max(worst_z, d / std::max(se_of(idx), 1e-5));
				sumz2 += d * d / (se_of(idx) * se_of(idx)); nzz++;
			}
			printf("  %s %s %s  exact %.5f %.5f %.5f  table %.5f %.5f %.5f  (%llu combo triples, se<=%.4f%%)\n",
			       g_names[c[0]].c_str(), g_names[c[1]].c_str(), g_names[c[2]].c_str(), eq[t][0], eq[t][1], eq[t][2], e[0], e[1],
			       e[2], (unsigned long long)N, se_of(idx) * 100);
		}
		printf("  %zu triples enumerated in %.1fs, worst difference %.4f%% = %.2f standard errors (limit 4.5), rms %.2f (the SE used is the triple's largest, so a bit below 1 is expected)\n",
		       ts.size(), secs, worst_abs * 100, worst_z, sqrt(sumz2 / nzz));
		report(njok, "combo-triple counts equal the enumeration's");
		report(worst_z < 4.5, "table within 4.5 standard errors of the exact value");
	}
	// the same on 150 random possible triples in random player order: exact enumeration of everything
	{
		std::vector<int3> ts;
		for (int t = 0; t < 150; t++) {
			uint32_t idx;
			do { idx = rnd(E3_NTRIPLE); } while (!cls[idx].w);
			int c[3] = {cls[idx].x, cls[idx].y, cls[idx].z};
			for (int k = 2; k > 0; k--) std::swap(c[k], c[rnd(k + 1)]);
			ts.push_back(make_int3(c[0], c[1], c[2]));
		}
		std::vector<std::array<double, 3>> eq;
		std::vector<uint64_t> nj;
		auto t0 = std::chrono::steady_clock::now();
		e3_exact(ts, eq, nj);
		double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		double worst_z = 0, worst_abs = 0, sumz2 = 0, sumz = 0;
		int n = 0;
		bool njok = true;
		for (size_t t = 0; t < ts.size(); t++) {
			int c[3] = {ts[t].x, ts[t].y, ts[t].z}, slot[3];
			float e[3];
			njok = njok && e3_lookup(table.data(), c[0], c[1], c[2], e) == nj[t];
			const double se = std::max(se_of(e3_index(c, slot)), 1e-5);
			for (int p = 0; p < 3; p++) {
				double d = e[p] - eq[t][p];
				worst_abs = std::max(worst_abs, fabs(d));
				worst_z = std::max(worst_z, fabs(d) / se);
				sumz += d / se; sumz2 += d * d / (se * se); n++;
			}
		}
		printf("  150 random triples exact in %.1fs: worst difference %.4f%% = %.2f standard errors, mean z %.2f, rms z %.2f\n", secs,
		       worst_abs * 100, worst_z, sumz / n, sqrt(sumz2 / n));
		report(njok && worst_z < 5 && fabs(sumz / n) < 0.3, "150 random triples: counts equal, worst < 5 SE, no bias");
	}

	// 4. independent host reference on random deals --------------------------------------------------
	printf("4. host reference on random deals (%d per triple)\n", host_deals);
	{
		const int NT = 30;
		double sumz = 0, sumz2 = 0;
		int nz = 0;
		for (int t = 0; t < NT; t++) {
			uint32_t idx;
			do { idx = rnd(E3_NTRIPLE); } while (!cls[idx].w);
			int c[3] = {cls[idx].x, cls[idx].y, cls[idx].z};
			// random player order, so that unsorted lookups are exercised too
			for (int k = 2; k > 0; k--) std::swap(c[k], c[rnd(k + 1)]);
			double s[3] = {0, 0, 0}, sq[3] = {0, 0, 0};
			int done = 0;
			while (done < host_deals) {
				int cards[3][2];
				uint64_t used = 0;
				bool bad = false;
				for (int p = 0; p < 3 && !bad; p++) {
					int k = (int)rnd(g_ncombo[c[p]]);
					cards[p][0] = g_combos[c[p]][k][0]; cards[p][1] = g_combos[c[p]][k][1];
					uint64_t m = (1ull << cards[p][0]) | (1ull << cards[p][1]);
					bad = (used & m) != 0;
					used |= m;
				}
				if (bad) continue;
				// own dealing: partial shuffle of the remaining deck
				int deck[52], nd = 0;
				for (int x = 0; x < 52; x++)
					if (!((used >> x) & 1)) deck[nd++] = x;
				int board[5];
				for (int k = 0; k < 5; k++) {
					int j = k + (int)rnd(nd - k);
					std::swap(deck[k], deck[j]);
					board[k] = deck[k];
				}
				uint64_t sc[3];
				for (int p = 0; p < 3; p++) {
					int h[7] = {cards[p][0], cards[p][1], board[0], board[1], board[2], board[3], board[4]};
					sc[p] = eval7_ref(h);
				}
				uint64_t best = std::max(sc[0], std::max(sc[1], sc[2]));
				int nw = (sc[0] == best) + (sc[1] == best) + (sc[2] == best);
				for (int p = 0; p < 3; p++) {
					double v = sc[p] == best ? 1.0 / nw : 0;
					s[p] += v; sq[p] += v * v;
				}
				done++;
			}
			float e[3];
			e3_lookup(table.data(), c[0], c[1], c[2], e);
			for (int p = 0; p < 3; p++) {
				double m = s[p] / host_deals, var = sq[p] / host_deals - m * m;
				double se = sqrt(std::max(var, 1e-12) / host_deals);
				double z = (m - e[p]) / se;
				sumz += z; sumz2 += z * z; nz++;
			}
		}
		// 3 players' shares of one deal sum to 1, so the z values of a triple are not independent (2 degrees of freedom)
		double rms = sqrt(sumz2 / nz);
		printf("  %d triples, mean z %.2f, rms z %.2f (expect about 0 and 1; the three shares of a triple are correlated)\n", NT, sumz / nz, rms);
		report(fabs(sumz / nz) < 0.5 && rms < 1.3, "host reference agrees with the table (mean z < 0.5, rms z < 1.4)");
	}

	// 5. sampler symmetry ---------------------------------------------------------------------------
	printf("5. sampling kernel under permuted player order\n");
	{
		bool ok = true;
		double worst = 0;
		const int3 base[] = {tri("AKs", "QQ", "T9s"), tri("AA", "AA", "76o"), tri("KJo", "KJo", "KJo")};
		for (auto& b : base) {
			const int c[3] = {b.x, b.y, b.z};
			static const int perm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
			std::vector<int3> ts;
			for (auto& pm : perm) ts.push_back(make_int3(c[pm[0]], c[pm[1]], c[pm[2]]));
			auto est = e3_sample_ordered(ts, 20000000, 4242);
			for (int a = 0; a < 6; a++)
				for (int p = 0; p < 3; p++) {
					// player p of ordering a holds class c[perm[a][p]]; compare with ordering 0 at that player
					double ref = est[0].eq[perm[a][p]], se = sqrt(est[a].se[p] * est[a].se[p] + est[0].se[perm[a][p]] * est[0].se[perm[a][p]]);
					double z = fabs(est[a].eq[p] - ref) / std::max(se, 1e-6);
					worst = std::max(worst, z);
				}
		}
		printf("  worst difference %.2f standard errors (limit 5)\n", worst);
		ok = worst < 5;
		report(ok, "6 player orders of 3 triples agree");
	}

	// 6. exchangeability identity and the range function -------------------------------------------------
	printf("6. range function\n");
	uint16_t* d_table = e3_upload_table(table);
	{
		std::vector<double> w(3 * NCLASS, 1.0), eq(3 * NCLASS), mass(3 * NCLASS);
		e3_ranges_gpu(d_table, w.data(), eq.data(), mass.data());
		double tot[3];
		for (int p = 0; p < 3; p++) {
			double num = 0, den = 0;
			for (int i = 0; i < NCLASS; i++) { num += mass[p * NCLASS + i] * eq[p * NCLASS + i]; den += mass[p * NCLASS + i]; }
			tot[p] = num / den;
		}
		printf("  all combos equally likely: average share %.6f %.6f %.6f (exactly 1/3 by exchangeability)\n", tot[0], tot[1], tot[2]);
		bool ok = true;
		for (int p = 0; p < 3; p++) ok = ok && fabs(tot[p] - 1.0 / 3) < 2e-5;
		report(ok, "average share of three random hands is 1/3");
		printf("  AA vs two random hands: %.4f   72o vs two random hands: %.4f\n", eq[0 * NCLASS + class_index("AA")], eq[0 * NCLASS + class_index("72o")]);
	}
	// realistic ranges
	struct Setup { const char* name; std::string r[3]; };
	std::vector<Setup> setups = {
	    {"squeeze: open 77+ / flat / 3bettor", {"77+,ATs+,KQs,AJo+,KQo", "22+,A2s+,K9s+,Q9s+,J9s+,T9s,98s,87s,76s,ATo+,KJo+", "TT+,AQs+,AKo:0.5,A5s:0.5,A4s:0.5"}},
	    {"cold call: UTG open / HJ call / BTN wide", {"66+,A9s+,KTs+,QTs+,JTs,ATo+,KQo", "22-JJ,suited broadways,suited connectors", "22+,A2s+,K5s+,Q8s+,J8s+,T8s+,97s+,86s+,75s+,65s,A7o+,KTo+,QTo+,JTo"}},
	    {"mixed weights", {"AA,KK,QQ:0.5,AKs,AKo:0.7,JTs:0.3,22:0.2", "99+,AQs+,AKo,T9s:0.5,KQs:0.8", "55+,A8s+,KJs+,QJs,JTs,ATo+,KQo:0.6"}},
	};
	// the second setup's middle range uses words; replace by explicit lists
	setups[1].r[1] = "22,33,44,55,66,77,88,99,TT,JJ,AJs,AQs,AKs,KQs,KJs,QJs,JTs,T9s,98s,87s,76s,65s,54s";
	std::vector<ComboEntry> combos;
	for (int c = 0; c < NCLASS; c++)
		for (int k = 0; k < g_ncombo[c]; k++) combos.push_back({c, g_combos[c][k][0], g_combos[c][k][1]});
	ComboEntry* d_combos;
	CK(cudaMalloc(&d_combos, combos.size() * sizeof(ComboEntry)));
	CK(cudaMemcpy(d_combos, combos.data(), combos.size() * sizeof(ComboEntry), cudaMemcpyHostToDevice));
	double worst_z_all = 0, worst_abs_all = 0;
	for (auto& su : setups) {
		std::vector<double> w;
		for (int p = 0; p < 3; p++) {
			std::vector<double> r = e3_parse_range(su.r[p]);
			w.insert(w.end(), r.begin(), r.end());
		}
		std::vector<double> eq(3 * NCLASS), mass(3 * NCLASS), eqg(3 * NCLASS), massg(3 * NCLASS);
		e3_ranges_cpu(table, w.data(), eq.data(), mass.data());
		e3_ranges_gpu(d_table, w.data(), eqg.data(), massg.data());
		double dmax = 0;
		for (int i = 0; i < 3 * NCLASS; i++) dmax = std::max(dmax, std::max(fabs(eq[i] - eqg[i]), fabs(mass[i] - massg[i]) / std::max(1.0, mass[i])));
		report(dmax < 1e-9, "GPU and CPU range function agree");

		// permuting the ranges permutes the answers
		std::vector<double> w2(3 * NCLASS), eq2(3 * NCLASS), mass2(3 * NCLASS);
		for (int p = 0; p < 3; p++) memcpy(&w2[p * NCLASS], &w[((p + 1) % 3) * NCLASS], NCLASS * 8);
		e3_ranges_cpu(table, w2.data(), eq2.data(), mass2.data());
		double dperm = 0;
		for (int p = 0; p < 3; p++)
			for (int i = 0; i < NCLASS; i++) dperm = std::max(dperm, fabs(eq2[p * NCLASS + i] - eq[((p + 1) % 3) * NCLASS + i]));
		report(dperm < 1e-9, "permuting the three ranges permutes the equities");

		// whole-range equities add up to 1 and every player sees the same number of combo triples
		double rng_eq[3], rng_mass[3];
		for (int p = 0; p < 3; p++) {
			double num = 0, den = 0;
			for (int i = 0; i < NCLASS; i++) { num += w[p * NCLASS + i] * mass[p * NCLASS + i] * eq[p * NCLASS + i]; den += w[p * NCLASS + i] * mass[p * NCLASS + i]; }
			rng_eq[p] = num / den; rng_mass[p] = den;
		}
		report(fabs(rng_eq[0] + rng_eq[1] + rng_eq[2] - 1) < 2e-5 && fabs(rng_mass[0] - rng_mass[1]) < 1e-6 * rng_mass[0] &&
		           fabs(rng_mass[0] - rng_mass[2]) < 1e-6 * rng_mass[0], "range equities sum to 1, equal total mass");

		// direct combo-level Monte Carlo: each combo drawn with its class weight, same cards removed, same showdown
		std::vector<double> cdf(3 * combos.size());
		for (int p = 0; p < 3; p++) {
			double acc = 0;
			for (size_t k = 0; k < combos.size(); k++) { acc += w[p * NCLASS + combos[k].cls]; cdf[p * combos.size() + k] = acc; }
		}
		double* d_cdf;
		unsigned long long *d_s, *d_q, *d_c;
		CK(cudaMalloc(&d_cdf, cdf.size() * 8));
		CK(cudaMalloc(&d_s, 3 * NCLASS * 8)); CK(cudaMalloc(&d_q, 3 * NCLASS * 8)); CK(cudaMalloc(&d_c, 3 * NCLASS * 8));
		CK(cudaMemcpy(d_cdf, cdf.data(), cdf.size() * 8, cudaMemcpyHostToDevice));
		CK(cudaMemset(d_s, 0, 3 * NCLASS * 8)); CK(cudaMemset(d_q, 0, 3 * NCLASS * 8)); CK(cudaMemset(d_c, 0, 3 * NCLASS * 8));
		const int blocks = 1024, per_thread = 3000;  // 1024*256*3000 = 7.9e8 draws
		auto t0 = std::chrono::steady_clock::now();
		mc_kernel<<<blocks, 256>>>(d_combos, d_cdf, (int)combos.size(), per_thread, 31337, d_s, d_q, d_c);
		CK(cudaGetLastError());
		CK(cudaDeviceSynchronize());
		double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		std::vector<unsigned long long> hs(3 * NCLASS), hq(3 * NCLASS), hc(3 * NCLASS);
		CK(cudaMemcpy(hs.data(), d_s, 3 * NCLASS * 8, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(hq.data(), d_q, 3 * NCLASS * 8, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(hc.data(), d_c, 3 * NCLASS * 8, cudaMemcpyDeviceToHost));
		cudaFree(d_cdf); cudaFree(d_s); cudaFree(d_q); cudaFree(d_c);
		double worst_z = 0, worst_abs = 0, sum_sq = 0, wsum_abs = 0, wtot = 0;
		int ncmp = 0;
		for (int p = 0; p < 3; p++)
			for (int i = 0; i < NCLASS; i++) {
				const unsigned long long n = hc[p * NCLASS + i];
				if (n < 50000 || w[p * NCLASS + i] == 0) continue;
				const double m = hs[p * NCLASS + i] / (6.0 * n), v6 = (double)hq[p * NCLASS + i] / n - 36 * m * m;
				const double se = sqrt(std::max(v6, 0.0) / 36 / n);
				const double d = m - eq[p * NCLASS + i];
				worst_z = std::max(worst_z, fabs(d) / se);
				worst_abs = std::max(worst_abs, fabs(d));
				sum_sq += d * d / (se * se);
				// frequency-weighted mean absolute difference (what a solver feels)
				wsum_abs += n * fabs(d); wtot += n;
				ncmp++;
			}
		printf("  [%s] %.1fs: %d class comparisons, worst |diff| %.4f%%, worst %.2f SE, rms z %.2f, weighted mean |diff| %.4f%%\n",
		       su.name, secs, ncmp, worst_abs * 100, worst_z, sqrt(sum_sq / ncmp), wsum_abs / wtot * 100);
		worst_z_all = std::max(worst_z_all, worst_z);
		worst_abs_all = std::max(worst_abs_all, worst_abs);
	}
	report(worst_z_all < 5, "table-based range equity agrees with combo-level Monte Carlo (< 5 SE of the Monte Carlo)");
	printf("  worst absolute difference to combo-level Monte Carlo: %.4f%%\n", worst_abs_all * 100);

	test_relay();
	printf(failures ? "FAIL\n" : "PASS\n");
	return failures ? 1 : 0;
}

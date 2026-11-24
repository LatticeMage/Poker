// Checks for the realisation factors: the range parsing, the GPU sort-and-scan equity against a naive host loop, the
// bookkeeping identities of the simulation (zero-sum), the raw equities against the exact 169x169 matrix,
// determinism, the sanity properties the table must have, and, when it exists, the results file itself.
// Prints PASS and exits 0, or FAIL and exits 1.
//
//   nvcc -O3 -arch=sm_89 -o test_realization test_realization.cu && ./test_realization [boards]
//
// Needs the ranges (../src/lib/poker/data, or POKER_DATA) and data/equity-exact.bin. [boards] is the number of random
// boards per simulated case (default 8000); a case takes about 10 seconds.

#include "realization.cuh"

static int failures = 0;
static void check(bool ok, const char* what) {
	printf("  %-4s %s\n", ok ? "ok" : "BAD", what);
	if (!ok) failures++;
}

static std::string data_dir() {
	const char* e = getenv("POKER_DATA");
	return e ? e : "../src/lib/poker/data";
}

// ------------------------------------------------------------------------------------------------------
// 1. ranges: the parser against the combo counts the source pages state
// ------------------------------------------------------------------------------------------------------
static void test_ranges() {
	printf("ranges\n");
	Json spots = json_load(data_dir() + "/spots.json"), charts = json_load(data_dir() + "/charts.json");
	int bad = 0, n = 0;
	const char* kinds[] = {"vs_open", "vs_3bet"};
	for (const char* kind : kinds)
		for (const auto& kv : spots.at(kind).obj) {
			const char* parts[] = {"call", "bluff", "value"};
			for (int p = 0; p < 3; p++) {
				int got = combo_count(parse_weighted(kv.second.at(parts[p]).str));
				int want = (int)kv.second.at("stated").arr[p].num;
				if (got != want) { bad++; printf("    %s %s %s: %d combos, page says %d\n", kind, kv.first.c_str(), parts[p], got, want); }
				n++;
			}
		}
	check(bad == 0, "every call / bluff / value list has the combo count the page states (30 spots x 3)");
	bad = 0;
	for (const Json& c : charts.at("charts").arr) {
		double pct = 100.0 * combo_count(parse_weighted(c.at("text").str)) / 1326.0;
		if (fabs(pct - c.at("stated_pct").num) > 0.5) { bad++; printf("    chart %s: %.2f%% vs stated %.2f%%\n", c.at("key").str.c_str(), pct, c.at("stated_pct").num); }
	}
	check(bad == 0, "every chart has the percentage it states (within 0.5: the sources round, PSBTN says 62 for 62.3)");
	auto one = parse_weighted("AsAc, AKs, KK-QQ, A5s-A3s, JTs-98s");
	check(fabs(one[class_index("AA")] - 1.0 / 6) < 1e-12 && one[class_index("AKs")] == 1 && one[class_index("AKo")] == 0 &&
	      one[class_index("QQ")] == 1 && one[class_index("A4s")] == 1 && one[class_index("T9s")] == 1 && one[class_index("87s")] == 0,
	      "single combos, spans and class names parse as in spots.ts");
}

// ------------------------------------------------------------------------------------------------------
// 2. the GPU scan and percentile against naive host loops
// ------------------------------------------------------------------------------------------------------
static void test_scan() {
	printf("sort-and-scan equity\n");
	uint32_t seedr = 12345;
	auto rnd = [&]() { seedr = seedr * 1664525u + 1013904223u; return (seedr >> 8) / 16777216.0; };
	int *d_b5;
	float *d_mI, *d_mO, *d_wi, *d_ti, *d_wo, *d_to;
	uint32_t* d_str;
	CK(cudaMalloc(&d_b5, 5 * 4));
	CK(cudaMalloc(&d_mI, NC * 4)); CK(cudaMalloc(&d_mO, NC * 4));
	CK(cudaMalloc(&d_wi, NC * 4)); CK(cudaMalloc(&d_ti, NC * 4)); CK(cudaMalloc(&d_wo, NC * 4)); CK(cudaMalloc(&d_to, NC * 4));
	CK(cudaMalloc(&d_str, NC * 4));
	double worst = 0;
	int cases = 0, ties_seen = 0;
	// random boards, and a board that plays for everybody (every combo ties: the royal flush)
	const int royal[5] = {12 * 4, 11 * 4, 10 * 4, 9 * 4, 8 * 4};
	for (int t = 0; t < 25; t++) {
		int b5[5];
		if (t == 0) {
			memcpy(b5, royal, sizeof(b5));
		} else if (t == 1) {  // a paired, monotone board: many ties and flushes
			const int fixed[5] = {12 * 4 + 1, 12 * 4 + 2, 5 * 4 + 1, 5 * 4 + 2, 3 * 4 + 1};
			memcpy(b5, fixed, sizeof(b5));
		} else {
			uint64_t used = 0;
			for (int i = 0; i < 5; i++) {
				int c;
				do { c = (int)(rnd() * 52); } while ((used >> c) & 1);
				used |= 1ull << c;
				b5[i] = c;
			}
		}
		std::vector<float> mI(NC), mO(NC);
		for (int j = 0; j < NC; j++) {
			mI[j] = rnd() < 0.5 ? 0.f : (float)rnd();
			mO[j] = rnd() < 0.3 ? 0.f : (rnd() < 0.5 ? 1.f : (float)rnd());
		}
		CK(cudaMemcpy(d_b5, b5, 5 * 4, cudaMemcpyHostToDevice));
		CK(cudaMemcpy(d_mI, mI.data(), NC * 4, cudaMemcpyHostToDevice));
		CK(cudaMemcpy(d_mO, mO.data(), NC * 4, cudaMemcpyHostToDevice));
		scan_test_kernel<<<1, NT>>>(d_b5, d_mI, d_mO, d_wi, d_ti, d_wo, d_to, d_str);
		CK(cudaGetLastError());
		CK(cudaDeviceSynchronize());
		std::vector<float> wi(NC), ti(NC), wo(NC), to(NC);
		std::vector<uint32_t> str(NC);
		CK(cudaMemcpy(wi.data(), d_wi, NC * 4, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(ti.data(), d_ti, NC * 4, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(wo.data(), d_wo, NC * 4, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(to.data(), d_to, NC * 4, cudaMemcpyDeviceToHost));
		CK(cudaMemcpy(str.data(), d_str, NC * 4, cudaMemcpyDeviceToHost));
		for (int j = 0; j < NC; j++) {
			if (str[j] >= SENT) continue;
			double a = 0, at = 0, b = 0, bt = 0;
			for (int o = 0; o < NC; o++) {
				if (str[o] >= SENT) continue;
				if (g_c0[o] == g_c0[j] || g_c0[o] == g_c1[j] || g_c1[o] == g_c0[j] || g_c1[o] == g_c1[j]) continue;  // shares a card (incl. itself)
				const double beats = str[j] > str[o] ? 1.0 : (str[j] == str[o] ? 0.5 : 0.0);
				if (str[j] == str[o]) ties_seen++;
				a += mO[o] * beats; at += mO[o];
				b += mI[o] * beats; bt += mI[o];
			}
			worst = std::max({worst, fabs(a - wi[j]), fabs(at - ti[j]), fabs(b - wo[j]), fabs(bt - to[j])});
		}
		cases++;
	}
	printf("    %d boards, %d tied pairs compared, largest difference %.2e\n", cases, ties_seen, worst);
	check(worst < 2e-3 && ties_seen > 1000, "win and opponent weight match the naive loop on random boards, a paired board and an all-tie board");
	cudaFree(d_b5); cudaFree(d_mI); cudaFree(d_mO); cudaFree(d_wi); cudaFree(d_ti); cudaFree(d_wo); cudaFree(d_to); cudaFree(d_str);

	// percentile
	float *d_eq, *d_m, *d_valid, *d_q, *d_scr;
	CK(cudaMalloc(&d_eq, NC * 4)); CK(cudaMalloc(&d_m, NC * 4)); CK(cudaMalloc(&d_valid, NC * 4)); CK(cudaMalloc(&d_q, NC * 4));
	CK(cudaMalloc(&d_scr, (size_t)N_SLOTS * NC * 4));
	double worstq = 0;
	for (int t = 0; t < 10; t++) {
		int b3[3] = {(int)(rnd() * 52), 0, 0};
		do { b3[1] = (int)(rnd() * 52); } while (b3[1] == b3[0]);
		do { b3[2] = (int)(rnd() * 52); } while (b3[2] == b3[0] || b3[2] == b3[1]);
		std::vector<float> eq(NC), m(NC), valid(NC);
		int n = 0;
		for (int j = 0; j < NC; j++) {
			bool dead = false;
			for (int i = 0; i < 3; i++) dead |= g_c0[j] == b3[i] || g_c1[j] == b3[i];
			valid[j] = dead ? 0.f : 1.f;
			n += !dead;
			eq[j] = t % 2 ? (float)(int)(rnd() * 20) / 20.f : (float)rnd();  // odd cases: many ties
			m[j] = rnd() < 0.4 ? 0.f : (float)rnd();
			if (dead) m[j] = 0.f;
		}
		CK(cudaMemcpy(d_eq, eq.data(), NC * 4, cudaMemcpyHostToDevice));
		CK(cudaMemcpy(d_m, m.data(), NC * 4, cudaMemcpyHostToDevice));
		CK(cudaMemcpy(d_valid, valid.data(), NC * 4, cudaMemcpyHostToDevice));
		percentile_test_kernel<<<1, NT>>>(d_eq, d_m, d_valid, n, d_q, d_scr);
		CK(cudaGetLastError());
		CK(cudaDeviceSynchronize());
		std::vector<float> q(NC);
		CK(cudaMemcpy(q.data(), d_q, NC * 4, cudaMemcpyDeviceToHost));
		auto key = [&](int j) { return (uint32_t)(fminf(fmaxf(eq[j], 0.f), 1.f) * 16777214.f); };
		double M = 0;
		for (int j = 0; j < NC; j++) M += m[j];
		for (int j = 0; j < NC; j++) {
			if (valid[j] == 0.f) continue;
			double stronger = 0, equal = 0;
			for (int o = 0; o < NC; o++) {
				if (valid[o] == 0.f) continue;
				if (key(o) > key(j)) stronger += m[o];
				else if (key(o) == key(j)) equal += m[o];
			}
			worstq = std::max(worstq, fabs((stronger + 0.5 * equal) / M - q[j]));
		}
	}
	printf("    10 boards, largest difference %.2e\n", worstq);
	check(worstq < 1e-4, "percentile ranks match the naive loop (including many ties)");
	cudaFree(d_eq); cudaFree(d_m); cudaFree(d_valid); cudaFree(d_q); cudaFree(d_scr);
}

// ------------------------------------------------------------------------------------------------------
// helpers on results
// ------------------------------------------------------------------------------------------------------
static const Cell& cell_of(const SimResult& r, int scn, int side, int cls) { return r.cells[((size_t)scn * 2 + side) * NCLASS + cls]; }

// Raw equity of class c against a range from the exact matrix, with the same card removal as the simulation.
static std::vector<double> exact_raw(const std::vector<double>& matrix, const std::vector<double>& opp) {
	std::vector<double> out(NCLASS);
	for (int c = 0; c < NCLASS; c++) {
		double num = 0, den = 0;
		for (int o = 0; o < NCLASS; o++) {
			long pairs = 0;
			for (int a = 0; a < g_ncombo[c]; a++)
				for (int b = 0; b < g_ncombo[o]; b++) {
					const int a0 = g_combos[c][a][0], a1 = g_combos[c][a][1], b0 = g_combos[o][b][0], b1 = g_combos[o][b][1];
					if (a0 != b0 && a0 != b1 && a1 != b0 && a1 != b1) pairs++;
				}
			num += opp[o] * pairs * matrix[(size_t)c * NCLASS + o];
			den += opp[o] * pairs;
		}
		out[c] = den > 0 ? num / den : 0.5;
	}
	return out;
}

static bool is_suited(int c) { return g_names[c].size() == 3 && g_names[c][2] == 's'; }
static bool is_offsuit(int c) { return g_names[c].size() == 3 && g_names[c][2] == 'o'; }
static int rank_hi(int c) { return (int)(strchr(RANKS, g_names[c][0]) - RANKS); }
static int rank_lo(int c) { return g_names[c].size() == 2 ? rank_hi(c) : (int)(strchr(RANKS, g_names[c][1]) - RANKS); }

int main(int argc, char** argv) {
	const long long boards = argc > 1 ? atoll(argv[1]) : 8000;
	try {
		setup_classes();
		setup_combos();
		std::vector<PotDef> pots = make_pots(data_dir());

		test_ranges();
		test_scan();

		// simulate every pot type at all its depths
		std::vector<PotRun> runs;
		for (const PotDef& d : pots) {
			printf("simulating %s: %lld boards\n", d.key.c_str(), boards);
			runs.push_back(run_pot(d, boards, 777, false));
			printf("    %.0fs\n", runs.back().res.secs);
		}

		printf("zero-sum and bookkeeping (worst relative error over all boards, pot types, depths)\n");
		double e1 = 0, e2 = 0, e3 = 0, e4 = 0;
		for (const PotRun& r : runs) {
			e1 = std::max(e1, r.res.err_prob); e2 = std::max(e2, r.res.err_chips); e3 = std::max(e3, r.res.err_inv); e4 = std::max(e4, r.res.err_raw);
		}
		printf("    path probabilities %.1e, chips won %.1e, chips put in %.1e, raw equity %.1e\n", e1, e2, e3, e4);
		check(e1 < 1e-5, "the action probabilities of every combo pair add up to 1 (all terminals together)");
		check(e2 < 1e-5, "chips won by IP + chips won by OOP = final pot, pair by pair (weighted by pair weight)");
		check(e3 < 1e-5, "chips put in by both sides = final pot - flop pot");
		check(e4 < 1e-5, "raw equities of the two sides add up to 1");

		printf("raw equity against the exact 169x169 matrix\n");
		{
			std::vector<double> matrix(NCLASS * NCLASS);
			FILE* f = fopen("data/equity-exact.bin", "rb");
			if (!f || fread(matrix.data(), 8, matrix.size(), f) != matrix.size()) { printf("  BAD  cannot read data/equity-exact.bin\n"); failures++; }
			else {
				double zmax = 0, z2 = 0;
				int cells = 0;
				for (size_t p = 0; p < runs.size(); p++)
					for (int side = 0; side < 2; side++) {
						std::vector<double> exp = exact_raw(matrix, side ? runs[p].def.ip : runs[p].def.oop);
						for (int c = 0; c < NCLASS; c++) {
							const Cell& x = cell_of(runs[p].res, 0, side, c);
							if (x.se_raw <= 0) continue;
							const double z = (x.raw - exp[c]) / x.se_raw;
							zmax = std::max(zmax, fabs(z));
							z2 += z * z;
							cells++;
						}
					}
				printf("    %d cells: largest |z| %.2f, mean z^2 %.2f\n", cells, zmax, z2 / cells);
				check(zmax < 5 && z2 / cells < 1.5, "simulated raw equity of every class matches the exact matrix within sampling error");
			}
			if (f) fclose(f);
		}

		printf("determinism\n");
		{
			const std::vector<Scn> sc = {{5.f, 97.5f}, {5.f, 17.5f}};
			SimResult a = simulate(pots[0].ip, pots[0].oop, sc, 600, 4242, false);
			SimResult b = simulate(pots[0].ip, pots[0].oop, sc, 600, 4242, false);
			SimResult c = simulate(pots[0].ip, pots[0].oop, sc, 600, 4243, false);
			check(a.acc == b.acc, "same seed: bit-identical results");
			check(a.acc != c.acc, "different seed: different boards");
		}

		printf("sanity of the table\n");
		for (size_t p = 0; p < runs.size(); p++) {
			const PotRun& r = runs[p];
			for (size_t d = 0; d < r.depths.size(); d++) {
				Summary ip = range_summary(r.res, (int)d, 0, r.def.ip), oop = range_summary(r.res, (int)d, 1, r.def.oop);
				char what[160];
				snprintf(what, sizeof(what), "%s %3dbb: IP range factor %.3f >= OOP %.3f", r.def.key.c_str(), r.depths[d], ip.factor, oop.factor);
				check(ip.factor >= oop.factor, what);
			}
		}
		{
			int out_of_band = 0, in_range = 0;
			double lo = 9, hi = 0, mean_dev = 0, mean_all_lo = 9, mean_all_hi = 0;
			for (const PotRun& r : runs)
				for (size_t d = 0; d < r.depths.size(); d++)
					for (int side = 0; side < 2; side++)
						for (int c = 0; c < NCLASS; c++) {
							const double w = side ? r.def.oop[c] : r.def.ip[c];
							const Cell& x = cell_of(r.res, (int)d, side, c);
							if (w > 0) {
								in_range++;
								lo = std::min(lo, x.factor); hi = std::max(hi, x.factor);
								if (x.factor < 0.5 || x.factor > 1.6) out_of_band++;
								mean_dev += fabs(x.factor - 1);
							}
							mean_all_lo = std::min(mean_all_lo, x.factor); mean_all_hi = std::max(mean_all_hi, x.factor);
						}
			printf("    %d in-range cells: factor from %.3f to %.3f, mean |factor - 1| %.3f; all 169 classes: %.3f to %.3f\n", in_range, lo, hi, mean_dev / in_range, mean_all_lo, mean_all_hi);
			check(out_of_band == 0, "factors of hands in the range are in 0.5 .. 1.6");
			check(mean_all_lo > 0.3 && mean_all_hi < 2.0, "factors of all 169 classes are in 0.3 .. 2.0");
		}
		{
			// suited hands against offsuit hands of the same raw equity
			const PotRun& r = runs[0];
			size_t d = r.depths.size() - 1;  // deepest
			for (int side = 0; side < 2; side++) {
				double sum = 0, var = 0;
				int n = 0;
				for (int c = 0; c < NCLASS; c++) {
					if (!is_suited(c) || rank_hi(c) - rank_lo(c) > 2 || rank_hi(c) > 10 || rank_lo(c) < 2) continue;  // suited connectors and gappers, 53s .. QTs
					const Cell& s = cell_of(r.res, (int)d, side, c);
					int best = -1;
					for (int o = 0; o < NCLASS; o++) {
						if (!is_offsuit(o) || rank_hi(o) > 10 || rank_hi(o) == 12 || rank_hi(o) - rank_lo(o) < 3) continue;  // offsuit, unconnected, no ace
						const Cell& x = cell_of(r.res, (int)d, side, o);
						if (fabs(x.raw - s.raw) < 0.012 && (best < 0 || fabs(x.raw - s.raw) < fabs(cell_of(r.res, (int)d, side, best).raw - s.raw))) best = o;
					}
					if (best < 0) continue;
					const Cell& x = cell_of(r.res, (int)d, side, best);
					sum += s.factor - x.factor;
					var += s.se * s.se + x.se * x.se;
					n++;
				}
				char what[200];
				snprintf(what, sizeof(what), "%s side: %d suited connectors/gappers realise %.3f more than offsuit unconnected hands of the same raw equity (se %.3f)",
				         side ? "OOP" : "IP", n, n ? sum / n : 0, n ? sqrt(var) / n : 0);
				check(n >= 5 && sum / n > 2 * sqrt(var) / n, what);
			}
		}

		printf("limiting cases\n");
		{
			const PotDef& d = pots[0];
			// stacks of 0.02, 4 and 19.5 pots: realisation should move away from 1 as the stack grows
			const std::vector<Scn> sc = {{5.f, 0.1f}, {5.f, 20.f}, {5.f, 97.5f}};
			SimResult r = simulate(d.ip, d.oop, sc, boards, 31337, false);
			double dev[3];
			for (int k = 0; k < 3; k++) {
				double s = 0, sw = 0;
				for (int side = 0; side < 2; side++)
					for (int c = 0; c < NCLASS; c++) {
						const double w = (side ? d.oop[c] : d.ip[c]) * combos_of_class(c);
						s += w * fabs(cell_of(r, k, side, c).factor - 1);
						sw += w;
					}
				dev[k] = s / sw;
			}
			Summary ip = range_summary(r, 0, 0, d.ip), oop = range_summary(r, 0, 1, d.oop);
			printf("    mean |factor - 1| over both ranges: SPR 0.02: %.4f, SPR 4: %.4f, SPR 19.5: %.4f; range factors at SPR 0.02: IP %.4f OOP %.4f\n", dev[0], dev[1], dev[2], ip.factor, oop.factor);
			check(dev[0] < 0.02 && dev[0] < dev[1] && dev[1] < dev[2], "SPR 0.02: realised = raw equity (factors within 2% of 1), and the gap grows with the stack");
			check(fabs(ip.factor - 1) < 0.01 && fabs(oop.factor - 1) < 0.01, "SPR 0.02: the factor of each whole range is within 1% of 1");
		}
		{
			// the same range on both sides: equal raw equity (1/2), and the player in position still gets more
			const PotDef& d = pots[0];
			SimResult r = simulate(d.oop, d.oop, {{5.f, 97.5f}}, boards, 99, false);
			Summary ip = range_summary(r, 0, 0, d.oop), oop = range_summary(r, 0, 1, d.oop);
			printf("    identical ranges: raw IP %.4f OOP %.4f, factor IP %.4f OOP %.4f\n", ip.raw, oop.raw, ip.factor, oop.factor);
			check(fabs(ip.raw - 0.5) < 0.01 && fabs(oop.raw - 0.5) < 0.01, "identical ranges: each side's raw equity is 1/2");
			check(ip.factor > oop.factor + 0.03, "identical ranges: position alone gives IP the higher factor");
		}

		printf("against the old formula (summary, not a pass/fail)\n");
		for (const PotRun& r : runs)
			for (size_t d = 0; d < r.depths.size(); d++)
				for (int side = 0; side < 2; side++) {
					const auto& w = side ? r.def.oop : r.def.ip;
					std::vector<double> raw(NCLASS), real(NCLASS);
					for (int c = 0; c < NCLASS; c++) { raw[c] = cell_of(r.res, (int)d, side, c).raw; real[c] = cell_of(r.res, (int)d, side, c).real; }
					OldDiff od = diff_old(side, raw, real, w);
					printf("    %-4s %3dbb %-3s: range factor new %.3f old %.3f; realised share new - old: mean %+.4f, rms %.4f, max %.4f; rms over all 169 %.4f\n",
					       r.def.key.c_str(), r.depths[d], side ? "OOP" : "IP", od.new_factor, od.old_factor, od.mean, od.rms, od.maxabs, od.rms_all);
				}

		// the results file, if present
		FILE* jf = fopen("data/realization.json", "rb");
		if (jf) {
			fclose(jf);
			printf("data/realization.json\n");
			Json j = json_load("data/realization.json");
			bool ok = j.at("schema").str == "realization/v1" && j.at("classes").arr.size() == NCLASS;
			for (int c = 0; c < NCLASS && ok; c++) ok = j.at("classes").arr[c].str == g_names[c];
			check(ok, "schema realization/v1 and the 169 class names in ALL_HANDS order");
			bool shape = true, consistent = true, ranges_ok = true, ids = true;
			double worst_se = 0, worst_id = 0;
			int cells = 0;
			for (const Json& pot : j.at("pots").arr) {
				const PotDef* def = nullptr;
				for (const PotDef& d : pots) if (d.key == pot.at("pot").str) def = &d;
				if (!def) { shape = false; continue; }
				for (int c = 0; c < NCLASS; c++) {
					ranges_ok = ranges_ok && fabs(pot.at("ip_range").arr[c].num - def->ip[c]) < 1e-4 && fabs(pot.at("oop_range").arr[c].num - def->oop[c]) < 1e-4;
				}
				for (const auto& kv : pot.at("identity_errors").obj) { worst_id = std::max(worst_id, kv.second.num); ids = ids && kv.second.num < 1e-5; }
				for (const Json& dep : pot.at("depths").arr)
					for (int side = 0; side < 2; side++) {
						const Json& s = dep.at(side ? "oop" : "ip");
						const char* cols[] = {"raw", "real", "factor", "se", "ev_bb", "final_pot_bb"};
						for (const char* k : cols) shape = shape && s.at(k).arr.size() == NCLASS;
						if (!shape) continue;
						for (int c = 0; c < NCLASS; c++) {
							const double raw = s.at("raw").arr[c].num, real = s.at("real").arr[c].num, f = s.at("factor").arr[c].num;
							if (raw > 0.05) consistent = consistent && fabs(real / raw - f) < 2e-3 * std::max(1.0, f);
							const double w = side ? def->oop[c] : def->ip[c];
							if (w > 0) { worst_se = std::max(worst_se, s.at("se").arr[c].num); cells++; }
						}
					}
			}
			printf("    %d in-range cells, largest standard error of a factor %.4f, %d boards per pot type\n", cells, worst_se, (int)j.at("boards_per_pot_type").num);
			check(shape, "every column has 169 numbers");
			check(consistent, "factor = real / raw in every cell");
			check(ranges_ok, "the ranges stored in the file are the ones parsed from charts.json / spots.json today");
			check(ids, "identity errors stored in the file are below 1e-5");
			check(worst_se < 0.03 && j.at("boards_per_pot_type").num >= 50000, "at least 50,000 boards per pot type, standard error of every in-range factor below 0.03");
		}
	} catch (const std::exception& e) {
		fprintf(stderr, "error: %s\n", e.what());
		return 1;
	}
	printf(failures ? "FAIL\n" : "PASS\n");
	return failures ? 1 : 0;
}

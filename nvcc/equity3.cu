// Command line for the three-way equity table. Checked by test_equity3.cu.
//
//   nvcc -O3 -arch=sm_89 -o equity3 equity3.cu
//   ./equity3 gen dir [target_se=0.0003] [n0=100000] [nmax=16000000]
//                                           the whole table into dir/equity3.bin (+ equity3-se.bin); resumable
//   ./equity3 triple A B C [samples]         one class triple by sampling (player order as given)
//   ./equity3 exact A B C                    one class triple, every combo triple on every board
//   ./equity3 query table.bin A B C          read a cell of the table
//   ./equity3 ranges table.bin R1 R2 R3      equity of every class of player 1 (and 2, 3) against the other two
//                                           ranges, e.g. "77+,AJs+,AQo+" "22+,A2s+,KTs+" "55+,A9s+,KJs+"

#include "equity3.cuh"

int main(int argc, char** argv) {
	if (argc < 2) { fprintf(stderr, "usage: equity3 gen|triple|exact|query|ranges ...\n"); return 1; }
	setup_classes();
	const std::string mode = argv[1];
	if (mode == "gen" && argc >= 3) {
		e3_generate(argv[2], argc >= 4 ? atof(argv[3]) : 0.0003, argc >= 6 ? strtoull(argv[5], 0, 10) : 16000000ull,
		            argc >= 5 ? strtoull(argv[4], 0, 10) : 100000ull);
	} else if (mode == "triple" && argc >= 5) {
		int3 t = make_int3(class_index(argv[2]), class_index(argv[3]), class_index(argv[4]));
		uint64_t n = argc >= 6 ? strtoull(argv[5], 0, 10) : 8000000ull;
		auto t0 = std::chrono::steady_clock::now();
		E3Est e = e3_sample_ordered({t}, n, 12345)[0];
		double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		if (!e.n) { printf("%s %s %s: impossible (not enough cards)\n", argv[2], argv[3], argv[4]); return 0; }
		printf("%s %s %s: %.5f %.5f %.5f  (se %.5f %.5f %.5f, %llu samples, %.2fs)\n", argv[2], argv[3], argv[4], e.eq[0],
		       e.eq[1], e.eq[2], e.se[0], e.se[1], e.se[2], (unsigned long long)e.n, s);
	} else if (mode == "exact" && argc >= 5) {
		std::vector<std::array<double, 3>> eq;
		std::vector<uint64_t> nj;
		e3_exact({make_int3(class_index(argv[2]), class_index(argv[3]), class_index(argv[4]))}, eq, nj);
		if (!nj[0]) printf("impossible\n");
		else printf("%s %s %s: %.6f %.6f %.6f exact (%llu combo triples)\n", argv[2], argv[3], argv[4], eq[0][0], eq[0][1], eq[0][2],
		            (unsigned long long)nj[0]);
	} else if (mode == "query" && argc >= 6) {
		std::vector<uint16_t> t = e3_load_table(argv[2]);
		float e[3];
		uint32_t n = e3_lookup(t.data(), class_index(argv[3]), class_index(argv[4]), class_index(argv[5]), e);
		if (!n) printf("impossible\n");
		else printf("%s %s %s: %.5f %.5f %.5f  (%u combo triples)\n", argv[3], argv[4], argv[5], e[0], e[1], e[2], n);
	} else if (mode == "ranges" && argc >= 6) {
		std::vector<uint16_t> t = e3_load_table(argv[2]);
		std::vector<double> w;
		for (int p = 0; p < 3; p++) {
			std::vector<double> r = e3_parse_range(argv[3 + p]);
			w.insert(w.end(), r.begin(), r.end());
		}
		std::vector<double> eq(3 * NCLASS), mass(3 * NCLASS);
		e3_ranges_cpu(t, w.data(), eq.data(), mass.data());
		for (int p = 0; p < 3; p++) {
			double num = 0, den = 0;
			printf("player %d\n", p + 1);
			for (int i = 0; i < NCLASS; i++) {
				if (w[p * NCLASS + i] == 0) continue;
				printf("  %-4s %.4f\n", g_names[i].c_str(), eq[p * NCLASS + i]);
				num += w[p * NCLASS + i] * mass[p * NCLASS + i] * eq[p * NCLASS + i];
				den += w[p * NCLASS + i] * mass[p * NCLASS + i];
			}
			printf("  whole range %.4f\n", den > 0 ? num / den : 0);
		}
	} else {
		fprintf(stderr, "bad arguments\n");
		return 1;
	}
	return 0;
}

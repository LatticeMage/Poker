// Command line for the equity matrix. Checked by test_equity.cu.
//
//   nvcc -O3 -arch=sm_89 -o equity equity.cu
//   ./equity pair AKs QQ 8000000          equity of one class pair
//   ./equity matrix 1000000 out.bin       all 14,196 pairs by sampling; out.bin = 169*169 float64, row-major
//   ./equity exact out.bin [A B]          every combo pair on every board: no sampling noise

#include "equity.cuh"

static void write_matrix(const char* path, const std::vector<int2>& pairs, const std::vector<double>& eq) {
	std::vector<double> m(NCLASS * NCLASS, 0.5);
	for (size_t k = 0; k < pairs.size(); k++) {
		m[pairs[k].x * NCLASS + pairs[k].y] = eq[k];
		m[pairs[k].y * NCLASS + pairs[k].x] = 1.0 - eq[k];
	}
	FILE* f = fopen(path, "wb");
	if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
	fwrite(m.data(), sizeof(double), m.size(), f);
	fclose(f);
	printf("wrote %s\n", path);
}

int main(int argc, char** argv) {
	if (argc < 2) { fprintf(stderr, "usage: equity pair|matrix|exact ...\n"); return 1; }
	setup_classes();
	std::string mode = argv[1];
	if (mode == "pair" && argc >= 5) {
		int a = class_index(argv[2]), b = class_index(argv[3]);
		std::vector<double> eq;
		std::vector<unsigned long long> v;
		auto t0 = std::chrono::steady_clock::now();
		run_pairs({make_int2(a, b)}, atoll(argv[4]), 12345, eq, v);
		double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		printf("%s vs %s: %.5f  (%llu valid samples, %.2fs)\n", argv[2], argv[3], eq[0], v[0], secs);
	} else if (mode == "matrix" && argc >= 4) {
		long long samples = atoll(argv[2]);
		std::vector<int2> all = all_pairs();
		std::vector<double> result;
		std::vector<unsigned long long> v;
		run_pairs(all, samples, 20261124, result, v, true);
		write_matrix(argv[3], all, result);
	} else if (mode == "exact" && argc >= 3) {
		std::vector<int2> pairs = argc >= 5 ? std::vector<int2>{make_int2(class_index(argv[3]), class_index(argv[4]))} : all_pairs();
		std::vector<double> eq;
		std::vector<unsigned long long> njob;
		exact_pairs(pairs, eq, njob, true);
		if (pairs.size() == 1) printf("%s vs %s: %.6f exact (%llu combo pairs)\n", argv[3], argv[4], eq[0], njob[0]);
		write_matrix(argv[2], pairs, eq);
	} else {
		fprintf(stderr, "bad arguments\n");
		return 1;
	}
	return 0;
}

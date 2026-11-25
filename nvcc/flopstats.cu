// Command line for flopstats.cuh: range against range on all 1,755 flops, for every heads-up matchup of the
// preflop charts, and the same numbers per flop group.
//
//   nvcc -O3 -arch=sm_89 -o flopstats flopstats.cu
//   ./flopstats run data/matchups.json data/flopstats.bin        compute (a relay: run it again to carry on)
//   ./flopstats groups data/matchups.json data/flopstats.bin > ../src/lib/poker/data/flop-group-stats.json
//                                                                 per matchup and group (level 7, 383 groups),
//                                                                 packed: what the app reads
//   ./flopstats spread data/matchups.json data/flopstats.bin     how much of the differences between flops each
//                                                                 level of the grouping explains
//   ./flopstats flops data/matchups.json data/flopstats.bin <key> one matchup, flop by flop (CSV)
//
// data/matchups.json comes from `npm run matchups` (tools/make-matchups.ts): the ranges the app itself uses.
// The numbers: eq = all-in equity of the preflop aggressor's range; nut = share of a range with at least 80%
// equity; weak = share below 30% (only in the flop-by-flop CSV).
#include "flopstats.cuh"

static const int FS_TREE_LEVEL = 7;  // the level of flop_shape the decision tree stands on: 383 groups
static const int NUT_BIN = 8;   // equity >= 80%
static const int WEAK_BINS = 3;  // equity < 30%

struct Row { double eq, nut_a, nut_c, weak_a, weak_c; };
static Row row_of(const double* r) {
	Row x;
	x.eq = r[0];
	x.nut_a = fs_share_from(r + 2, NUT_BIN);
	x.nut_c = fs_share_from(r + 2 + FS_BINS, NUT_BIN);
	x.weak_a = 1 - fs_share_from(r + 2, WEAK_BINS);
	x.weak_c = 1 - fs_share_from(r + 2 + FS_BINS, WEAK_BINS);
	return x;
}

static std::vector<double> load_results(const std::string& path, size_t want) {
	FILE* f = fopen(path.c_str(), "rb");
	if (!f) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
	std::vector<double> r(want);
	const size_t got = fread(r.data(), 8, want, f);
	fclose(f);
	if (got != want) { fprintf(stderr, "%s: expected %zu numbers, found %zu (other matchups?)\n", path.c_str(), want, got); exit(1); }
	return r;
}

int main(int argc, char** argv) {
	if (argc < 4) {
		fprintf(stderr, "usage: flopstats run|groups|spread|flops <matchups.json> <results.bin> [key]\n");
		return 1;
	}
	const std::string cmd = argv[1], bin = argv[3];
	setup_classes();
	const std::vector<FsMatchup> ms = fs_load_matchups(argv[2]);
	const std::vector<Flop>& flops = all_flops();
	const size_t NF = flops.size();

	if (cmd == "run") {
		const std::vector<double> res = fs_run(ms, bin + ".relay", true);
		const std::string tmp = bin + ".tmp";
		FILE* f = fopen(tmp.c_str(), "wb");
		if (!f || fwrite(res.data(), 8, res.size(), f) != res.size()) { fprintf(stderr, "cannot write %s\n", tmp.c_str()); return 1; }
		fclose(f);
		rename(tmp.c_str(), bin.c_str());
		// the two sides are computed separately: their equities must add up to 1 on every flop
		double worst = 0;
		for (size_t i = 0; i < ms.size() * NF; i++) worst = std::max(worst, fabs(res[i * FS_K] + res[i * FS_K + 1] - 1));
		printf("%zu matchups x %zu flops written to %s; largest |eq_aggressor + eq_caller - 1| = %.2e\n", ms.size(), NF, bin.c_str(), worst);
		return worst < 1e-9 ? 0 : 1;
	}

	const std::vector<double> res = load_results(bin, ms.size() * NF * FS_K);

	if (cmd == "groups") {
		// Per matchup and group: the mean over the group's flops (weighted by flop weight) of the aggressor's range
		// equity and of the two nut shares, and the lowest and highest range equity among the group's flops, so a
		// reader sees how far one flop of the group can be from the group's number.
		// Packed: five values per group (eq, eq_min, eq_max, nut_aggressor, nut_caller), each as two base-32 digits of
		// round(value * 1023) (steps of 0.1%), ten characters per group, groups in order.
		std::vector<int> group;
		const std::vector<std::string> names = fs_groups(FS_TREE_LEVEL, group);
		const int G = (int)names.size();
		const char* digits = "0123456789abcdefghijklmnopqrstuv";
		printf("{\n\"groups\": %d,\n\"columns\": [\"eq\", \"eq_min\", \"eq_max\", \"nut_aggressor\", \"nut_caller\"],\n\"matchups\": {\n", G);
		for (size_t m = 0; m < ms.size(); m++) {
			std::vector<Row> sum(G, Row{0, 0, 0, 0, 0});
			std::vector<double> w(G, 0), lo(G, 9), hi(G, -9);
			for (size_t f = 0; f < NF; f++) {
				const Row r = row_of(&res[(m * NF + f) * FS_K]);
				const int g = group[f];
				const double wt = flops[f].weight;
				sum[g].eq += wt * r.eq; sum[g].nut_a += wt * r.nut_a; sum[g].nut_c += wt * r.nut_c;
				lo[g] = std::min(lo[g], r.eq); hi[g] = std::max(hi[g], r.eq);
				w[g] += wt;
			}
			std::string text;
			for (int g = 0; g < G; g++)
				for (double v : {sum[g].eq / w[g], lo[g], hi[g], sum[g].nut_a / w[g], sum[g].nut_c / w[g]}) {
					const int n = std::max(0, std::min(1023, (int)llround(v * 1023)));
					text += digits[n >> 5];
					text += digits[n & 31];
				}
			printf("\"%s\": \"%s\"%s\n", ms[m].key.c_str(), text.c_str(), m + 1 < ms.size() ? "," : "");
		}
		printf("}\n}\n");
		return 0;
	}

	if (cmd == "spread") {
		// For every level of the grouping: the share of the flop-to-flop variance (flops weighted by their weight)
		// that the groups explain, averaged over the matchups, and the largest difference inside one group.
		// Two measures: the aggressor's range equity, and the nut advantage (nut share aggressor - caller).
		// One line per level and pot kind (srp / 3bet / 4bet: the matchups whose key starts with it), averages over
		// the matchups of that kind.
		printf("level,groups,pot,eq_explained,eq_sd_within,eq_max_spread,nut_explained,nut_sd_within,nut_max_spread\n");
		// Levels 1..9 are the levels of flop_shape; 10 = the three ranks alone (suits ignored), 11 = the three ranks and
		// the number of suits: reference lines that say what the range numbers actually depend on.
		for (int depth = 0; depth <= 11; depth++) {
			std::vector<int> group(NF, 0);
			int G = 1;
			if (depth >= 1 && depth <= 9) G = (int)fs_groups(depth, group).size();
			if (depth >= 10) {
				std::map<int, int> ids;
				for (size_t f = 0; f < NF; f++) {
					int r[3] = {flops[f].cards[0] >> 2, flops[f].cards[1] >> 2, flops[f].cards[2] >> 2};
					std::sort(r, r + 3);
					const int key = ((r[0] * 13 + r[1]) * 13 + r[2]) * 3 + (depth == 11 ? flops[f].suit_kind : 0);
					group[f] = ids.emplace(key, (int)ids.size()).first->second;
				}
				G = (int)ids.size();
			}
			for (const char* kind : {"srp", "3bet", "4bet"}) {
				double ex[2] = {0, 0}, sdw[2] = {0, 0}, spread[2] = {0, 0};
				int n = 0;
				for (size_t m = 0; m < ms.size(); m++) {
					if (ms[m].key.compare(0, strlen(kind) + 1, std::string(kind) + ":") != 0) continue;
					n++;
					for (int q = 0; q < 2; q++) {
						std::vector<double> s(G, 0), s2(G, 0), w(G, 0), lo(G, 9), hi(G, -9);
						double S = 0, S2 = 0, W = 0;
						for (size_t f = 0; f < NF; f++) {
							const Row r = row_of(&res[(m * NF + f) * FS_K]);
							const double v = q == 0 ? r.eq : r.nut_a - r.nut_c, wt = flops[f].weight;
							const int g = group[f];
							s[g] += wt * v; s2[g] += wt * v * v; w[g] += wt;
							lo[g] = std::min(lo[g], v); hi[g] = std::max(hi[g], v);
							S += wt * v; S2 += wt * v * v; W += wt;
						}
						const double var = S2 / W - (S / W) * (S / W);
						double within = 0;
						for (int g = 0; g < G; g++) {
							within += s2[g] - s[g] * s[g] / w[g];
							spread[q] = std::max(spread[q], hi[g] - lo[g]);
						}
						within /= W;
						ex[q] += var > 1e-12 ? 1 - within / var : 1;
						sdw[q] += sqrt(std::max(0.0, within));
					}
				}
				printf("%d,%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", depth, G, kind, ex[0] / n, sdw[0] / n, spread[0], ex[1] / n, sdw[1] / n, spread[1]);
			}
		}
		return 0;
	}

	if (cmd == "flops" && argc >= 5) {
		size_t m = 0;
		while (m < ms.size() && ms[m].key != argv[4]) m++;
		if (m == ms.size()) { fprintf(stderr, "no matchup %s\n", argv[4]); return 1; }
		std::vector<int> group;
		const std::vector<std::string> names = fs_groups(FS_TREE_LEVEL, group);
		printf("flop,weight,group,group_name,eq,nut_aggressor,nut_caller,weak_aggressor,weak_caller\n");
		for (int f : flop_shape_order()) {
			const Row r = row_of(&res[(m * NF + f) * FS_K]);
			printf("%s,%d,%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f\n", flop_name(flops[f]).c_str(), flops[f].weight, group[f] + 1, names[group[f]].c_str(), r.eq,
			       r.nut_a, r.nut_c, r.weak_a, r.weak_c);
		}
		return 0;
	}
	fprintf(stderr, "unknown command %s\n", cmd.c_str());
	return 1;
}

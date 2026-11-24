// Command line for flops.cuh: list the strategically distinct flops.
//
//   nvcc -O2 -o flops flops.cu
//   ./flops list > data/flops-1755.csv        all 1,755 classes, one per line
//   ./flops partition 184 > data/flops-184.csv  every one of the 1,755 with its group (statistical k-medoids)
//   ./flops taxonomy 184 > data/flops-taxonomy-184.csv  the same with NAMED groups (broadway count, pairing, suits, ...)
//
// Columns of list: index, flop, weight, suits (rainbow / two-tone / mono), pairing (unpaired / paired / trips),
// broadway (cards T J Q K A), high (rank of the highest card), span (highest - lowest rank, ace also low).
#include "flops.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const char* RANKS = "23456789TJQKA";
static const char* SUIT_KIND[3] = {"rainbow", "two-tone", "mono"};

static const char* pairing(const Flop& f) { return f.trips ? "trips" : (f.paired ? "paired" : "unpaired"); }

int main(int argc, char** argv) {
	const std::vector<Flop>& all = all_flops();
	if (argc >= 2 && strcmp(argv[1], "list") == 0) {
		printf("index,flop,weight,suits,pairing,broadway,high,span\n");
		for (const Flop& f : all)
			printf("%d,%s,%d,%s,%s,%d,%c,%d\n", f.index, flop_name(f).c_str(), f.weight, SUIT_KIND[f.suit_kind], pairing(f),
			       flop_broadway(f), RANKS[f.high], flop_span_ace_low(f));
		return 0;
	}
	if (argc >= 3 && strcmp(argv[1], "partition") == 0) {
		FlopPartition p = subset_partition(atoi(argv[2]));
		printf("index,flop,weight,group,group_flop,group_weight\n");
		for (size_t i = 0; i < all.size(); i++) {
			const Flop& r = p.rep[p.group_of[i]];
			printf("%d,%s,%d,%d,%s,%d\n", all[i].index, flop_name(all[i]).c_str(), all[i].weight, p.group_of[i], flop_name(r).c_str(), r.weight);
		}
		return 0;
	}
	if (argc >= 2 && strcmp(argv[1], "levels") == 0) {
		// the refinement: every level partitions all 1,755 classes (22,100 flops); each level splits the one before
		printf("level  groups  biggest(classes)  smallest(classes)  total classes  total flops\n");
		for (int depth = 1; depth <= 6; depth++) {
			std::vector<std::string> names;
			std::vector<int> cls, wt;
			for (const Flop& f : all) {
				const std::string n = flop_group_name(f, depth);
				size_t g = std::find(names.begin(), names.end(), n) - names.begin();
				if (g == names.size()) { names.push_back(n); cls.push_back(0); wt.push_back(0); }
				cls[g]++;
				wt[g] += f.weight;
			}
			int big = 0, small = 1 << 30, tc = 0, tw = 0;
			for (size_t g = 0; g < names.size(); g++) { big = std::max(big, cls[g]); small = std::min(small, cls[g]); tc += cls[g]; tw += wt[g]; }
			printf("%5d  %6zu  %16d  %17d  %13d  %11d\n", depth, names.size(), big, small, tc, tw);
		}
		FlopTaxonomy t = flop_taxonomy(184);
		std::vector<int> size(t.name.size(), 0);
		for (size_t i = 0; i < all.size(); i++) size[t.group_of[i]]++;
		int big = 0, small = 1 << 30, tc = 0, tw = 0;
		for (size_t g = 0; g < t.name.size(); g++) { big = std::max(big, size[g]); small = std::min(small, size[g]); tc += size[g]; tw += t.rep[g].weight; }
		printf("%5s  %6zu  %16d  %17d  %13d  %11d   (refined to 184)\n", "184", t.name.size(), big, small, tc, tw);
		return 0;
	}
	if (argc >= 3 && strcmp(argv[1], "taxonomy") == 0) {
		FlopTaxonomy t = flop_taxonomy(atoi(argv[2]));
		printf("index,flop,weight,group,group_name,group_flop,group_weight,group_classes\n");
		std::vector<int> size(t.name.size(), 0);
		for (size_t i = 0; i < all.size(); i++) size[t.group_of[i]]++;
		for (size_t i = 0; i < all.size(); i++) {
			const int g = t.group_of[i];
			printf("%d,%s,%d,%d,%s,%s,%d,%d\n", all[i].index, flop_name(all[i]).c_str(), all[i].weight, g, t.name[g].c_str(),
			       flop_name(t.rep[g]).c_str(), t.rep[g].weight, size[g]);
		}
		return 0;
	}
	fprintf(stderr, "usage: flops list | flops partition <k> | flops taxonomy <k> | flops levels\n");
	return 1;
}

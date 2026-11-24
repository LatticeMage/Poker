// Command line for flops.cuh: list the strategically distinct flops.
//
//   nvcc -O2 -o flops flops.cu
//   ./flops list > data/flops-1755.csv        all 1,755 classes, one per line, in the order of their characteristics
//   ./flops shape 4 > data/flops-shape-groups.csv   the groups down to a level, with size and wetness
//   ./flops partition 184 > data/flops-184.csv  every one of the 1,755 with its group (statistical k-medoids)
//   ./flops taxonomy 184 > data/flops-taxonomy-184.csv  the same with NAMED groups (broadway count, pairing, suits, ...)
//
// Columns of list: order (suits -> high cards -> connectedness of the high cards -> of the low cards -> high to low ->
// suited pair, see flop_shape in flops.cuh), index (the class id in all_flops(), unchanged), flop, weight, the levels,
// gap_top / gap_low (rank differences), group (number and name down to level 4), then the wetness as counted: band,
// share of the 1,176 hole-card hands that make or draw a flush or a straight, number of types, and the counts.
#include "flops.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const char* SUITS_ZH[3] = {"單色", "雙花", "三花"};

static const char* pairing(const Flop& f) { return f.trips ? "trips" : (f.paired ? "paired" : "unpaired"); }

int main(int argc, char** argv) {
	const std::vector<Flop>& all = all_flops();
	if (argc >= 2 && strcmp(argv[1], "list") == 0) {
		// all 1,755 classes in the order of the characteristics (flop_shape_order), each with its counted wetness
		const std::vector<int> order = flop_shape_order();
		printf("order,index,flop,weight,suits,high_cards,high_conn,low_conn,high_low_conn,suited,gap_top,gap_low,group,group_name,"
		       "wetness,share,types,flush,straight,flush_draw,open_ended,gutshot,pairing\n");
		std::string last;
		int group = 0;
		for (size_t o = 0; o < order.size(); o++) {
			const Flop& f = all[order[o]];
			const FlopShape s = flop_shape(f);
			const FlopWetness w = flop_wetness_detail(f);
			const std::string g = flop_shape_name(f, 4);
			if (g != last) { group++; last = g; }
			printf("%zu,%d,%s,%d,%s,%d,%s,%s,%s,%s,%d,%d,%d,%s,%s,%.4f,%d,%d,%d,%d,%d,%d,%s\n", o + 1, f.index, flop_name(f).c_str(), f.weight,
			       SUITS_ZH[s.suits - 1], s.n_high, flop_conn_name(s.high_conn), flop_conn_name(s.low_conn),
			       s.hl_conn == 0 ? "-" : (s.hl_conn == 1 ? "連" : (s.hl_conn == 2 ? "跳" : "斷")),
			       s.suited == 0 ? "-" : (s.suited == 1 ? "雙高" : (s.suited == 2 ? "高低" : "雙低")), s.gap_top, s.gap_low, group, g.c_str(),
			       flop_wetness_name(flop_wetness(f)), w.share, w.types, w.count[0], w.count[1], w.count[2], w.count[3], w.count[4], pairing(f));
		}
		return 0;
	}
	if (argc >= 3 && strcmp(argv[1], "shape") == 0) {
		// the groups down to a level (1..6), in order, each with its size and its wetness (the weighted mean share of
		// hole-card hands that make or draw a flush or a straight, and the driest and wettest member)
		const int depth = atoi(argv[2]);
		const std::vector<int> order = flop_shape_order();
		struct G { std::string name; int classes = 0, flops = 0; double sum = 0, lo = 9, hi = -1; std::string ex; };
		std::vector<G> gs;
		for (int i : order) {
			const std::string n = flop_shape_name(all[i], depth);
			if (gs.empty() || gs.back().name != n) { gs.push_back(G()); gs.back().name = n; gs.back().ex = flop_name(all[i]); }
			G& g = gs.back();
			const double sh = flop_wetness_share(all[i]);
			g.classes++; g.flops += all[i].weight; g.sum += sh * all[i].weight;
			g.lo = std::min(g.lo, sh); g.hi = std::max(g.hi, sh);
		}
		printf("group,group_name,classes,flops,wetness_mean,wetness_min,wetness_max,first_flop\n");
		for (size_t g = 0; g < gs.size(); g++)
			printf("%zu,%s,%d,%d,%.4f,%.4f,%.4f,%s\n", g + 1, gs[g].name.c_str(), gs[g].classes, gs[g].flops, gs[g].sum / gs[g].flops, gs[g].lo,
			       gs[g].hi, gs[g].ex.c_str());
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
	if (argc >= 3 && strcmp(argv[1], "spectrum") == 0) {
		// the groups from driest to wettest: the dry-wet spectrum
		FlopTaxonomy t = flop_taxonomy(atoi(argv[2]));
		std::vector<int> size(t.name.size(), 0), order(t.name.size());
		for (size_t i = 0; i < all.size(); i++) size[t.group_of[i]]++;
		for (size_t g = 0; g < order.size(); g++) order[g] = (int)g;
		std::sort(order.begin(), order.end(), [&](int a, int b) { return t.wetness[a] != t.wetness[b] ? t.wetness[a] < t.wetness[b] : a < b; });
		printf("rank,wetness,flops,classes,group\n");
		for (size_t r = 0; r < order.size(); r++) {
			const int g = order[r];
			printf("%zu,%.3f,%d,%d,%s\n", r + 1, t.wetness[g], t.rep[g].weight, size[g], t.name[g].c_str());
		}
		return 0;
	}
	if (argc >= 3 && strcmp(argv[1], "taxonomy") == 0) {
		FlopTaxonomy t = flop_taxonomy(atoi(argv[2]));
		printf("index,flop,weight,group,group_name,group_flop,group_weight,group_classes,wetness,group_wetness\n");
		std::vector<int> size(t.name.size(), 0);
		for (size_t i = 0; i < all.size(); i++) size[t.group_of[i]]++;
		for (size_t i = 0; i < all.size(); i++) {
			const int g = t.group_of[i];
			printf("%d,%s,%d,%d,%s,%s,%d,%d,%s,%.3f\n", all[i].index, flop_name(all[i]).c_str(), all[i].weight, g, t.name[g].c_str(),
			       flop_name(t.rep[g]).c_str(), t.rep[g].weight, size[g], flop_wetness_name(flop_wetness(all[i])), t.wetness[g]);
		}
		return 0;
	}
	fprintf(stderr, "usage: flops list | flops shape <depth 1-6> | flops partition <k> | flops taxonomy <k> | flops spectrum <k> | flops levels\n");
	return 1;
}

// Command line for flops.cuh: list the strategically distinct flops.
//
//   nvcc -O2 -o flops flops.cu
//   ./flops list > data/flops-1755.csv        all 1,755 classes, one per line, in the order of their characteristics
//   ./flops shape 5 > data/flops-shape-groups.csv   the groups down to a level, with size and wetness
//
// Columns of list: order (suits -> high cards -> connectedness of the high cards -> of the low cards -> all three in one
// straight -> high to low -> suited pair, see flop_shape in flops.cuh), index (the class id in all_flops(), unchanged), flop, weight, the levels,
// gap_top / gap_low (rank differences), group (number and name down to level 5: 73 groups), then the wetness as counted: band,
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
		printf("order,index,flop,weight,suits,high_cards,high_conn,low_conn,same_straight,high_low_conn,suited,gap_top,gap_low,group,group_name,"
		       "wetness,share,types,flush,straight,flush_draw,open_ended,gutshot,pairing\n");
		std::string last;
		int group = 0;
		for (size_t o = 0; o < order.size(); o++) {
			const Flop& f = all[order[o]];
			const FlopShape s = flop_shape(f);
			const FlopWetness w = flop_wetness_detail(f);
			const std::string g = flop_shape_name(f, 5);
			if (g != last) { group++; last = g; }
			printf("%zu,%d,%s,%d,%s,%d,%s,%s,%s,%s,%s,%d,%d,%d,%s,%s,%.4f,%d,%d,%d,%d,%d,%d,%s\n", o + 1, f.index, flop_name(f).c_str(), f.weight,
			       SUITS_ZH[s.suits - 1], s.n_high, flop_conn_name(s.high_conn), flop_conn_name(s.low_conn),
			       s.same_straight == 0 ? "-" : (s.same_straight == 1 ? "同順" : "不同順"),
			       s.hl_conn == 0 ? "-" : (s.hl_conn == 1 ? "連" : (s.hl_conn == 2 ? "跳" : "斷")),
			       s.suited == 0 ? "-" : (s.suited == 1 ? "雙高" : (s.suited == 2 ? "高低" : "雙低")), s.gap_top, s.gap_low, group, g.c_str(),
			       flop_wetness_name(flop_wetness(f)), w.share, w.types, w.count[0], w.count[1], w.count[2], w.count[3], w.count[4], pairing(f));
		}
		return 0;
	}
	if (argc >= 3 && strcmp(argv[1], "shape") == 0) {
		// the groups down to a level (1..7), in order, each with its size and its wetness (the weighted mean share of
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
	fprintf(stderr, "usage: flops list | flops shape <depth 1-7>\n");
	return 1;
}

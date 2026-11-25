// Command line for flopsolve.cuh: the flop betting game solved for the heads-up matchups of the preflop charts.
//
//   nvcc -O3 -arch=sm_89 -o flopsolve flopsolve.cu
//   ./flopsolve run data/matchups.json data/flopsolve rep|all [iterations=200] [rounds=1] [key-prefix]
//                        solve every matchup (or those whose key starts with the prefix) on one flop per group of
//                        the tree (rep: 383 flops) or on all 1,755; one result file and one relay file per matchup
//                        in the folder; run it again to carry on
//   ./flopsolve show data/matchups.json <key> <flop, e.g. As7d2c> [iterations=200] [rounds=1]
//                        one flop, readable: every decision by equity bin
//   ./flopsolve report data/matchups.json data/flopsolve
//                        one line per solved matchup: how well it is solved and the main frequencies (CSV)
//   ./flopsolve flops data/matchups.json data/flopsolve <key>
//                        one matchup, flop by flop (CSV)
//   ./flopsolve groups data/matchups.json data/flopsolve > ../src/lib/poker/data/flop-solved.json
//                        every solved matchup per group of the tree: the main frequencies, packed (what the app reads)
//   ./flopsolve pack data/matchups.json data/flopsolve <key>
//                        one matchup per group in full (every decision by equity bin), packed
//   ./flopsolve spread data/matchups.json data/flopsolve
//                        how much of the differences between flops each level of the grouping explains (needs "all")
//
// rounds = rounds of betting in the model: 1 to 3 without a card between them, "turn" = the flop and a real turn
// card, or "turn3" = the same with a third round of betting on that turn (see flopsolve.cuh).
// A result file: "FVS2", count, FV_K, iterations, rounds (int32), the flop indices (int32 each), then FV_K floats
// per flop.
// The result folder is local (raw results are not in the repository).
#include "flopsolve.cuh"

#include <map>
#include <sys/stat.h>

static int rounds_of(const char* text) {
	const int r = strcmp(text, "turn") == 0 ? FV_TURN_MODE : strcmp(text, "turn3") == 0 ? FV_TURN3_MODE : atoi(text);
	if (r != FV_TURN_MODE && r != FV_TURN3_MODE && (r < 1 || r > 3)) { fprintf(stderr, "rounds: 1, 2, 3, turn or turn3\n"); exit(1); }
	return r;
}
static std::string file_of(const std::string& dir, const std::string& key, const char* ext) {
	std::string name = key;
	for (char& c : name)
		if (c == ':') c = '-';
	return dir + "/" + name + ext;
}

struct Solved {
	std::vector<int> ids;
	std::vector<float> rec;
	int iters = 0, streets = 0;
};
static bool load_solved(const std::string& path, Solved& s) {
	FILE* f = fopen(path.c_str(), "rb");
	if (!f) return false;
	char magic[4];
	int32_t head[4];
	bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, "FVS2", 4) == 0 && fread(head, 4, 4, f) == 4 && head[1] == FV_K;
	if (ok) {
		s.ids.resize(head[0]);
		s.rec.resize((size_t)head[0] * FV_K);
		s.iters = head[2];
		s.streets = head[3];
		ok = fread(s.ids.data(), 4, s.ids.size(), f) == s.ids.size() && fread(s.rec.data(), 4, s.rec.size(), f) == s.rec.size();
	}
	fclose(f);
	return ok;
}
static bool save_solved(const std::string& path, const std::vector<int>& ids, const std::vector<float>& rec, int iters, int streets) {
	const std::string tmp = path + ".tmp";
	FILE* f = fopen(tmp.c_str(), "wb");
	if (!f) return false;
	const int32_t head[4] = {(int32_t)ids.size(), FV_K, iters, streets};
	const bool ok = fwrite("FVS2", 1, 4, f) == 4 && fwrite(head, 4, 4, f) == 4 && fwrite(ids.data(), 4, ids.size(), f) == ids.size() &&
	                fwrite(rec.data(), 4, rec.size(), f) == rec.size();
	fclose(f);
	return ok && rename(tmp.c_str(), path.c_str()) == 0;
}

static void print_record(const float* rec, const FvSpot& spot) {
	printf("ev of the player out of position %.4f pots (all-in equity %.4f), exploitability %.5f pots, %d iterations\n", rec[0], rec[2], rec[1], (int)rec[3]);
	const char* acts3[3] = {"check", "bet33", "bet66"};
	const char* resp[3] = {"fold", "call", "raise"};
	for (int s = 0; s < 2; s++)
		for (int nd = 0; nd < 5; nd++) {
			const int node = s * 5 + nd;
			const bool agg = (s == 1) == spot.aggressor_ip;
			printf("\n%s (%s), %s: %.1f%% of the range gets here by its own actions\n", s == 0 ? "out of position" : "in position", agg ? "aggressor" : "caller",
			       FV_NODE_NAME[nd], 100 * fv_reach(rec, node));
			printf("  equity      share");
			for (int a = 0; a < FV_NACT[nd]; a++) printf("  %6s", nd == 0 ? acts3[a] : resp[a]);
			printf("\n");
			for (int b = FS_BINS - 1; b >= 0; b--) {
				double share = 0;
				for (int a = 0; a < 3; a++) share += rec[FV_HEAD + (node * FS_BINS + b) * 3 + a];
				if (share < 1e-6) continue;
				printf("  %3d-%3d%%  %6.1f%%", b * 10, b * 10 + 10, 100 * share);
				for (int a = 0; a < FV_NACT[nd]; a++) printf("  %5.1f%%", 100 * fv_freq(rec, node, a, b, b + 1));
				printf("\n");
			}
			printf("  all               ");
			for (int a = 0; a < FV_NACT[nd]; a++) printf("  %5.1f%%", 100 * fv_freq(rec, node, a));
			printf("\n");
		}
}

int main(int argc, char** argv) {
	if (argc < 4) {
		fprintf(stderr, "usage: flopsolve run <matchups.json> <dir> rep|all [iterations] [rounds] [key-prefix]\n"
		                "       flopsolve show <matchups.json> <key> <flop> [iterations] [rounds]\n"
		                "       flopsolve report <matchups.json> <dir>\n"
		                "       flopsolve flops <matchups.json> <dir> <key>\n"
		                "       flopsolve groups <matchups.json> <dir>\n"
		                "       flopsolve pack <matchups.json> <dir> <key>\n"
		                "       flopsolve spread <matchups.json> <dir>\n");
		return 1;
	}
	const std::string cmd = argv[1];
	setup_classes();
	const std::vector<FsMatchup> ms = fs_load_matchups(argv[2]);
	const std::vector<Flop>& flops = all_flops();

	if (cmd == "run" && argc >= 5) {
		const std::string dir = argv[3], which = argv[4], prefix = argc > 7 ? argv[7] : "";
		const int iters = argc > 5 ? atoi(argv[5]) : 200, streets = argc > 6 ? rounds_of(argv[6]) : 1;
		if (which != "rep" && which != "all") { fprintf(stderr, "rep or all\n"); return 1; }
		mkdir(dir.c_str(), 0775);
		const std::vector<int> ids = fv_flop_ids(which == "all");
		FvDevice dev;
		for (const FsMatchup& m : ms) {
			if (m.key.compare(0, prefix.size(), prefix) != 0) continue;
			const std::string out = file_of(dir, m.key, ".bin");
			Solved have;
			if (load_solved(out, have) && have.ids == ids && have.iters == iters && have.streets == streets) continue;  // finished earlier
			const std::vector<float> rec = fv_run(dev, m, ids, iters, streets, file_of(dir, m.key, ".relay"), true);
			if (!save_solved(out, ids, rec, iters, streets)) { fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
			double worst = 0, mean = 0;
			for (size_t f = 0; f < ids.size(); f++) { worst = std::max(worst, (double)rec[f * FV_K + 1]); mean += rec[f * FV_K + 1] / ids.size(); }
			printf("%s: %zu flops written to %s; exploitability mean %.5f, largest %.5f pots\n", m.key.c_str(), ids.size(), out.c_str(), mean, worst);
			fflush(stdout);
			remove(file_of(dir, m.key, ".relay").c_str());  // the result file now holds everything
		}
		return 0;
	}

	if (cmd == "show" && argc >= 5) {
		size_t m = 0;
		while (m < ms.size() && ms[m].key != argv[3]) m++;
		if (m == ms.size()) { fprintf(stderr, "no matchup %s\n", argv[3]); return 1; }
		const std::string text = argv[4];
		if (text.size() != 6) { fprintf(stderr, "a flop is three cards, e.g. As7d2c\n"); return 1; }
		int c[3];
		for (int x = 0; x < 3; x++) {
			const char* suit = strchr("shdc", text[x * 2 + 1]);
			if (!suit) { fprintf(stderr, "bad card in %s\n", text.c_str()); return 1; }
			c[x] = rank_of(text[x * 2]) * 4 + (int)(suit - "shdc");
		}
		std::sort(c, c + 3);
		const uint8_t cards[3] = {(uint8_t)c[0], (uint8_t)c[1], (uint8_t)c[2]};
		const int iters = argc > 5 ? atoi(argv[5]) : 200, streets = argc > 6 ? rounds_of(argv[6]) : 1;
		const FvSpot spot = fv_spot(ms[m].key);
		const FvGame game = fv_game_of(spot.spr, streets);
		const FsSide A = fs_side(ms[m].aggressor), C = fs_side(ms[m].caller);
		FvDevice dev;
		float rec[FV_K];
		thermal_gate(-1, -1, -1, true);
		dev.solve(game, spot.aggressor_ip ? C : A, spot.aggressor_ip ? A : C, cards, 1, iters, rec);
		printf("%s on %s: pot %.1fbb, %.1fbb behind (spr %.2f), the aggressor is %s position; %s, %d showdown lines\n", ms[m].key.c_str(), text.c_str(),
		       spot.pot, spot.stack, spot.spr, spot.aggressor_ip ? "in" : "out of",
		       streets == FV_TURN_MODE ? "flop and a real turn card" : streets == FV_TURN3_MODE ? "flop, a real turn card, one more round on it" : streets == 1 ? "one round of betting" : streets == 2 ? "two rounds, no card between" : "three rounds, no card between",
		       game.lines + FV_TURNS * game.tlines);
		print_record(rec, spot);
		return 0;
	}

	if (cmd == "report") {
		const std::string dir = argv[3];
		printf("key,flops,iterations,aggressor,spr,expl_mean,expl_max,eq_aggressor,ev_aggressor,aggr_bet33,aggr_bet66,caller_bet33,caller_bet66,"
		       "caller_fold_vs33,caller_raise_vs33,caller_fold_vs66,caller_raise_vs66\n");
		for (const FsMatchup& m : ms) {
			Solved s;
			if (!load_solved(file_of(dir, m.key, ".bin"), s)) continue;
			const FvSpot spot = fv_spot(m.key);
			const int sa = spot.aggressor_ip ? 1 : 0, sc = 1 - sa;
			double W = 0, v[12] = {0}, worst = 0;
			for (size_t f = 0; f < s.ids.size(); f++) {
				const float* r = &s.rec[f * FV_K];
				const double wt = flops[s.ids[f]].weight;
				// the caller faces the aggressor's bet at its decisions 1 and 2; weight those by how often they happen
				const double x[12] = {r[1], sa == 0 ? r[2] : 1 - r[2], sa == 0 ? r[0] : 1 - r[0], fv_freq(r, sa * 5, 1), fv_freq(r, sa * 5, 2),
				                      fv_freq(r, sc * 5, 1), fv_freq(r, sc * 5, 2), fv_freq(r, sc * 5 + 1, 0), fv_freq(r, sc * 5 + 1, 2),
				                      fv_freq(r, sc * 5 + 2, 0), fv_freq(r, sc * 5 + 2, 2), 0};
				for (int k = 0; k < 11; k++) v[k] += wt * x[k];
				W += wt;
				worst = std::max(worst, (double)r[1]);
			}
			printf("%s,%zu,%d,%s,%.2f,%.5f,%.5f", m.key.c_str(), s.ids.size(), s.iters, sa ? "ip" : "oop", spot.spr, v[0] / W, worst);
			for (int k = 1; k < 11; k++) printf(",%.4f", v[k] / W);
			printf("\n");
		}
		return 0;
	}

	if (cmd == "flops" && argc >= 5) {
		Solved s;
		if (!load_solved(file_of(argv[3], argv[4], ".bin"), s)) { fprintf(stderr, "no result for %s\n", argv[4]); return 1; }
		const FvSpot spot = fv_spot(argv[4]);
		const int sa = spot.aggressor_ip ? 1 : 0, sc = 1 - sa;
		std::vector<int> group;
		const std::vector<std::string> names = fs_groups(7, group);
		printf("flop,weight,group,group_name,expl,eq_aggressor,ev_aggressor,aggr_check,aggr_bet33,aggr_bet66,caller_check,caller_bet33,caller_bet66,"
		       "caller_fold_vs33,caller_fold_vs66\n");
		for (size_t f = 0; f < s.ids.size(); f++) {
			const float* r = &s.rec[f * FV_K];
			const int id = s.ids[f];
			printf("%s,%d,%d,%s,%.5f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", flop_name(flops[id]).c_str(), flops[id].weight, group[id] + 1,
			       names[group[id]].c_str(), r[1], sa == 0 ? r[2] : 1 - r[2], sa == 0 ? r[0] : 1 - r[0], fv_freq(r, sa * 5, 0), fv_freq(r, sa * 5, 1),
			       fv_freq(r, sa * 5, 2), fv_freq(r, sc * 5, 0), fv_freq(r, sc * 5, 1), fv_freq(r, sc * 5, 2), fv_freq(r, sc * 5 + 1, 0), fv_freq(r, sc * 5 + 2, 0));
		}
		return 0;
	}
	if (cmd == "groups") {
		// The table the app's flop plan reads: for every solved matchup and every group of the tree (level 7) seven
		// numbers for the aggressor and seven for the caller, from the group's solved flops (weighted by flop weight):
		//   bet33, bet66   share of the range that bets each size at its first decision
		//   value          share of those bets made with at least 50% equity
		//   fold33, raise33, fold66, raise66   what the range does facing a bet of each size
		// Packed like flop-group-stats.json: two base-32 digits of round(value * 1023), 28 characters per group.
		const std::string dir = argv[3];
		std::vector<int> group;
		const int G = (int)fs_groups(7, group).size();
		const char* d32 = "0123456789abcdefghijklmnopqrstuv";
		std::string body, model;
		int iters = 0;
		for (const FsMatchup& m : ms) {
			Solved sv;
			if (!load_solved(file_of(dir, m.key, ".bin"), sv)) continue;
			const int sa = fv_spot(m.key).aggressor_ip ? 1 : 0;
			std::vector<double> mean((size_t)G * FV_K, 0.0), w(G, 0.0);
			for (size_t f = 0; f < sv.ids.size(); f++) {
				const int g = group[sv.ids[f]];
				const double wt = flops[sv.ids[f]].weight;
				for (int k = 0; k < FV_K; k++) mean[(size_t)g * FV_K + k] += wt * sv.rec[f * FV_K + k];
				w[g] += wt;
			}
			std::string text;
			for (int g = 0; g < G; g++) {
				if (w[g] <= 0) { fprintf(stderr, "%s: group %d has no solved flop\n", m.key.c_str(), g + 1); return 1; }
				std::vector<float> r(FV_K);
				for (int k = 0; k < FV_K; k++) r[k] = (float)(mean[(size_t)g * FV_K + k] / w[g]);
				for (int s : {sa, 1 - sa}) {
					double bets = 0, value = 0;
					for (int b = 0; b < FS_BINS; b++)
						for (int a = 1; a < 3; a++) {
							const double x = r[FV_HEAD + ((s * 5) * FS_BINS + b) * 3 + a];
							bets += x;
							if (b >= FS_BINS / 2) value += x;
						}
					const double v[7] = {fv_freq(r.data(), s * 5, 1), fv_freq(r.data(), s * 5, 2), bets > 1e-9 ? value / bets : 0,
					                     fv_freq(r.data(), s * 5 + 1, 0), fv_freq(r.data(), s * 5 + 1, 2), fv_freq(r.data(), s * 5 + 2, 0), fv_freq(r.data(), s * 5 + 2, 2)};
					for (double x : v) {
						const int n = std::max(0, std::min(1023, (int)llround(std::max(0.0, x) * 1023)));
						text += d32[n >> 5];
						text += d32[n & 31];
					}
				}
			}
			body += (body.empty() ? "" : ",\n") + ("\"" + m.key + "\": \"" + text + "\"");
			model = sv.streets == FV_TURN_MODE ? "turn" : sv.streets == FV_TURN3_MODE ? "turn3" : std::to_string(sv.streets);
			iters = sv.iters;
		}
		printf("{\n\"groups\": %d,\n\"model\": \"%s\",\n\"iterations\": %d,\n\"columns\": [\"bet33\", \"bet66\", \"value\", \"fold33\", \"raise33\", \"fold66\", \"raise66\"],\n"
		       "\"matchups\": {\n%s\n}\n}\n", G, model.c_str(), iters, body.c_str());
		return 0;
	}

	if (cmd == "pack" && argc >= 5) {
		// One matchup for the app, per group of the tree (level 7): the mean record of the group's solved flops
		// (weighted by flop weight), packed. Per group FV_PACK characters:
		//   2   ev of the player out of position, two base-32 digits of round(v * 1023)
		//   then for O and for I, one character each (64 levels, see `digits`):
		//   10  share of the range in each equity bin (0-10% ... 90-100%)
		//   20  first decision, per bin: check, bet 33% (bet 66% is the rest)
		//   20  facing 33%, per bin: fold, call (raise is the rest)      20  facing 66%: the same
		//   10  33% bet was raised, per bin: fold (call is the rest)     10  66% bet was raised: the same
		Solved sv;
		if (!load_solved(file_of(argv[3], argv[4], ".bin"), sv)) { fprintf(stderr, "no result for %s\n", argv[4]); return 1; }
		const FvSpot spot = fv_spot(argv[4]);
		std::vector<int> group;
		const int G = (int)fs_groups(7, group).size();
		std::vector<double> mean((size_t)G * FV_K, 0.0), w(G, 0.0);
		for (size_t f = 0; f < sv.ids.size(); f++) {
			const int g = group[sv.ids[f]];
			const double wt = flops[sv.ids[f]].weight;
			for (int k = 0; k < FV_K; k++) mean[(size_t)g * FV_K + k] += wt * sv.rec[f * FV_K + k];
			w[g] += wt;
		}
		const char* d32 = "0123456789abcdefghijklmnopqrstuv";
		const char* d64 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_";
		std::string text;
		int missing = 0;
		for (int g = 0; g < G; g++) {
			if (w[g] <= 0) { missing++; text += std::string(182, '0'); continue; }
			const double* r = &mean[(size_t)g * FV_K];
			auto at = [&](int node, int b, int a) { return r[FV_HEAD + (node * FS_BINS + b) * 3 + a] / w[g]; };
			auto put = [&](double v) { text += d64[std::max(0, std::min(63, (int)llround(v * 63)))]; };
			const int ev = std::max(0, std::min(1023, (int)llround(r[0] / w[g] * 1023)));
			text += d32[ev >> 5];
			text += d32[ev & 31];
			for (int s = 0; s < 2; s++) {
				for (int b = 0; b < FS_BINS; b++) put(at(s * 5, b, 0) + at(s * 5, b, 1) + at(s * 5, b, 2));
				for (int nd = 0; nd < 5; nd++)
					for (int b = 0; b < FS_BINS; b++) {
						const double all = at(s * 5 + nd, b, 0) + at(s * 5 + nd, b, 1) + at(s * 5 + nd, b, 2);
						for (int a = 0; a < FV_NACT[nd] - 1; a++) put(all > 1e-9 ? at(s * 5 + nd, b, a) / all : 0);
					}
			}
		}
		printf("{\"key\": \"%s\", \"model\": \"%s\", \"iterations\": %d, \"flops\": %zu, \"aggressor_ip\": %s, \"pot\": %.2f, \"spr\": %.3f, "
		       "\"groups\": %d, \"missing\": %d, \"per\": 182, \"data\": \"%s\"}\n",
		       argv[4], sv.streets == FV_TURN_MODE ? "turn" : sv.streets == FV_TURN3_MODE ? "turn3" : sv.streets == 1 ? "1" : sv.streets == 2 ? "2" : "3", sv.iters,
		       sv.ids.size(), spot.aggressor_ip ? "true" : "false", spot.pot, spot.spr, G, missing, text.c_str());
		return 0;
	}

	if (cmd == "spread") {
		// How much of the flop-to-flop differences of the solved numbers the flop groups explain (flops weighted by
		// their weight, averaged over the matchups of a pot kind), as flopstats does for the range equity. Needs the
		// run on all 1,755 flops. Levels as in flops.cuh; 10 = the three ranks alone.
		const std::string dir = argv[3];
		const char* what[5] = {"ev_aggressor", "aggressor_bets", "aggressor_bets_big", "caller_folds_vs33", "caller_folds_vs66"};
		printf("level,groups,pot,quantity,mean,explained,sd_within,max_spread\n");
		std::vector<Solved> all(ms.size());
		std::vector<bool> have(ms.size());
		for (size_t m = 0; m < ms.size(); m++) have[m] = load_solved(file_of(dir, ms[m].key, ".bin"), all[m]) && all[m].ids.size() == flops.size();
		for (int depth : {0, 5, 6, 7, 8, 9, 10}) {
			std::vector<int> group(flops.size(), 0);
			int G = 1;
			if (depth >= 1 && depth <= 9) G = (int)fs_groups(depth, group).size();
			if (depth == 10) {
				std::map<int, int> ids;
				for (size_t f = 0; f < flops.size(); f++) {
					int r[3] = {flops[f].cards[0] >> 2, flops[f].cards[1] >> 2, flops[f].cards[2] >> 2};
					std::sort(r, r + 3);
					group[f] = ids.emplace((r[0] * 13 + r[1]) * 13 + r[2], (int)ids.size()).first->second;
				}
				G = (int)ids.size();
			}
			for (const char* kind : {"srp", "3bet", "4bet"})
				for (int q = 0; q < 5; q++) {
					double ex = 0, sdw = 0, spread = 0, mean = 0;
					int n = 0;
					for (size_t m = 0; m < ms.size(); m++) {
						if (!have[m] || ms[m].key.compare(0, strlen(kind) + 1, std::string(kind) + ":") != 0) continue;
						const int sa = fv_spot(ms[m].key).aggressor_ip ? 1 : 0, sc = 1 - sa;
						std::vector<double> s(G, 0), s2(G, 0), w(G, 0), lo(G, 9), hi(G, -9);
						double S = 0, S2 = 0, W = 0;
						for (size_t f = 0; f < flops.size(); f++) {
							const float* r = &all[m].rec[f * FV_K];  // the ids of a full run are 0..1754 in order
							double v = q == 0 ? (sa == 0 ? r[0] : 1 - r[0]) : q == 1 ? 1 - fv_freq(r, sa * 5, 0) : q == 2 ? fv_freq(r, sa * 5, 2)
							         : q == 3 ? fv_freq(r, sc * 5 + 1, 0) : fv_freq(r, sc * 5 + 2, 0);
							if (v < 0) v = 0;
							const double wt = flops[f].weight;
							const int g = group[all[m].ids[f]];
							s[g] += wt * v; s2[g] += wt * v * v; w[g] += wt;
							lo[g] = std::min(lo[g], v); hi[g] = std::max(hi[g], v);
							S += wt * v; S2 += wt * v * v; W += wt;
						}
						const double var = S2 / W - (S / W) * (S / W);
						double within = 0;
						for (int g = 0; g < G; g++) {
							if (w[g] <= 0) continue;
							within += s2[g] - s[g] * s[g] / w[g];
							spread = std::max(spread, hi[g] - lo[g]);
						}
						within /= W;
						ex += var > 1e-12 ? 1 - within / var : 1;
						sdw += sqrt(std::max(0.0, within));
						mean += S / W;
						n++;
					}
					if (n) printf("%d,%d,%s,%s,%.4f,%.4f,%.4f,%.4f\n", depth, G, kind, what[q], mean / n, ex / n, sdw / n, spread);
				}
		}
		return 0;
	}
	fprintf(stderr, "unknown command %s\n", cmd.c_str());
	return 1;
}

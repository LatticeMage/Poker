// Command line for the 6-max preflop solver (sixmax.cuh). Checked by test_sixmax.cu.
//
//   nvcc -O3 -arch=sm_89 -o sixmax sixmax.cu
//   ./sixmax tree [stack]                        print the tree size
//   ./sixmax solve --stack 100 --iters1 3000 --iters 20000 [--out data/out] [--ckpt PREFIX] [--batch N] [--report N] [--seed-k K]
//
// Stage 1 trains only the first-in decisions, every reply fixed to the source charts; stage 2 trains everything,
// starting from stage 1. The ranges are read from src/lib/poker/data (POKER_DATA overrides that folder).

#include "sixmax_run.cuh"

int main(int argc, char** argv) {
	if (argc < 2) {
		fprintf(stderr, "usage: sixmax tree|solve ...\n");
		return 1;
	}
	setup_classes();
	const std::string cmd = argv[1];
	if (cmd == "tree") {
		Rules r;
		r.stack = argc > 2 ? atof(argv[2]) : 100;
		std::vector<Node6> t = build_tree(r);
		int dec = 0, term = 0, t3 = 0;
		for (const Node6& n : t) {
			if (n.is_term()) {
				term++;
				if (__builtin_popcount(n.live) == 3) t3++;
			} else {
				dec++;
			}
		}
		printf("stack %g: %zu nodes, %d decisions, %d terminals (%d three-way)\n", r.stack, t.size(), dec, term, t3);
		return 0;
	}
	if (cmd == "solve") {
		RunOpts o;
		for (int i = 2; i + 1 < argc; i += 2) {
			const std::string a = argv[i];
			const char* v = argv[i + 1];
			if (a == "--stack") o.stack = atof(v);
			else if (a == "--iters1") o.iters1 = atoll(v);
			else if (a == "--iters") o.iters2 = atoll(v);
			else if (a == "--out") o.out = v;
			else if (a == "--ckpt") o.ckpt = v;
			else if (a == "--batch") o.batch = atoll(v);
			else if (a == "--report") o.report = atoll(v);
			else if (a == "--seed-k") o.seed_k = atof(v);
			else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
		}
		return run_solve(o);
	}
	fprintf(stderr, "unknown command %s\n", cmd.c_str());
	return 1;
}

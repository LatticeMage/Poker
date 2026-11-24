// The heads-up preflop CFR+ solver: game trees, kernels and host drivers. Used by solve.cu (the command line)
// and by test_solve.cu / test_solve_external.cu (the checks). A port of tools/solve-preflop.ts.
//
// Game: blinds 0.5 / 1, small blind (player 0, the button) against big blind (player 1). A tree is a list of
// nodes: decisions (who acts, which actions), folds (what each player has put in) and showdowns (how much
// each has put in; an all-in showdown uses the raw equity, any other call uses the realised equity).
//
// CFR+ runs over the 169 hand classes at once: every node holds one value per class, and the showdown
// values are 169x169 matrix-vector products with the conflict-free combo counts as weights (same maths as
// the TypeScript tool: regrets floored at zero, strategy sums weighted by the iteration number).
//
// GPU layout: all trees of one run (one per stack depth) are packed into one node list. One CFR iteration is
//   top-down   one kernel per tree depth: regret matching, reach of the children
//   showdowns  one kernel, one block per terminal: the matrix-vector products, one thread per hand class
//   bottom-up  one kernel per tree depth, deepest first: values, regret and strategy-sum updates
// The kernels of one iteration are recorded in a CUDA graph and replayed, so a small tree is not dominated
// by launch cost. Exploitability and the value of the averaged profile use the same kernels with a
// different mode. Build with -fmad=false: every sum then runs in the same order and with the same rounding
// as the TypeScript tool (plain double multiply, then add), so the two agree to the last bit.
#pragma once

#include <functional>
#include <set>

#include "poker.cuh"

static const int NP = 192;  // hand slots per vector (169 classes padded to 6 warps; thread h = class h)
static const int MAXA = 8;  // most actions at one decision

// Realised share of the pot for the player in position (the small blind): the heuristic of solve-preflop.ts.
static const double REAL_SLOPE = 1.08, REAL_BONUS = 0.025;
static double realised_ip(double eq) { return std::min(1.0, std::max(0.0, 0.5 + REAL_SLOPE * (eq - 0.5) + REAL_BONUS)); }

// -- game trees (host) ---------------------------------------------------------------------------
enum { DECISION = 0, FOLD = 1, SHOW = 2 };

struct GNode {
	int kind = DECISION;
	int player = 0;  // decision: who acts; fold: who folds
	std::string id;  // path used as the JSON key ("root", "open/2.5", ...)
	std::vector<std::string> labels;
	std::vector<int> child;
	std::vector<uint8_t> path;  // action index taken at each step from the root
	double c0 = 0, c1 = 0;      // fold: what player 0 / 1 has put in; showdown: c0 = what each has put in
	bool allin = false;         // showdown: both all-in (raw equity), else realised equity
};

struct GameTree {
	std::vector<GNode> nodes;  // nodes[0] is the root decision; a parent always comes before its children
	int add(const GNode& n) { nodes.push_back(n); return (int)nodes.size() - 1; }
};

static std::string num(double x) {
	char b[32];
	snprintf(b, sizeof b, "%g", x);
	return b;
}

// The tree of solve-preflop.ts: every raise ladder is a short list of "to" sizes (bb); a raise at or above
// 70% of the stack is an all-in. open_levels[k] holds the sizes allowed for the k-th raise after an open,
// limp_levels likewise after a limp.
typedef std::vector<std::vector<double> > Levels;

static int build_facing(GameTree& t, const std::string& id, std::vector<uint8_t> path, int p, double own,
                        double bet, const Levels& levels, size_t from, double stack) {
	const int d = t.add(GNode());
	t.nodes[d].id = id;
	t.nodes[d].player = p;
	t.nodes[d].path = path;
	GNode f;
	f.kind = FOLD;
	f.player = p;
	f.c0 = p == 0 ? own : bet;
	f.c1 = p == 0 ? bet : own;
	GNode s;
	s.kind = SHOW;
	s.c0 = bet;
	s.allin = bet >= stack;
	std::vector<std::string> labels = {"fold", "call"};
	std::vector<int> child = {t.add(f), t.add(s)};
	if (bet < stack) {
		std::set<double> sizes;
		if (from < levels.size())
			for (double to : levels[from]) sizes.insert(to >= 0.7 * stack ? stack : to);
		sizes.insert(stack);
		for (double to : sizes) {
			if (to <= bet) continue;
			labels.push_back(to >= stack ? "allin" : "raise " + num(to));
			std::vector<uint8_t> np = path;
			np.push_back((uint8_t)labels.size() - 1);
			const int c = build_facing(t, id + "/" + (to >= stack ? "allin" : num(to)), np, 1 - p, bet, to,
			                           levels, from + 1, stack);
			child.push_back(c);
		}
	}
	t.nodes[d].labels = labels;
	t.nodes[d].child = child;
	return d;
}

static GameTree build_ladder_tree(double stack, const Levels& open_levels, const Levels& limp_levels) {
	GameTree t;
	const int root = t.add(GNode());
	t.nodes[root].id = "root";
	t.nodes[root].player = 0;
	GNode f;
	f.kind = FOLD;
	f.player = 0;
	f.c0 = 0.5;
	f.c1 = 1;
	std::vector<std::string> labels = {"fold", "limp"};
	std::vector<int> child = {t.add(f)};
	{  // the big blind answers a limp
		const int limp = t.add(GNode());
		t.nodes[limp].id = "limp";
		t.nodes[limp].player = 1;
		t.nodes[limp].path = {1};
		GNode s;
		s.kind = SHOW;
		s.c0 = 1;
		std::vector<std::string> bl = {"check"};
		std::vector<int> bc = {t.add(s)};
		std::set<double> sizes;
		if (!limp_levels.empty())
			for (double to : limp_levels[0]) sizes.insert(to >= 0.7 * stack ? stack : to);
		sizes.insert(stack);
		for (double to : sizes) {
			bl.push_back(to >= stack ? "allin" : "raise " + num(to));
			const int c = build_facing(t, "limp/" + (to >= stack ? std::string("allin") : num(to)), {1, (uint8_t)(bl.size() - 1)}, 0, 1,
			                           to, limp_levels, 1, stack);
			bc.push_back(c);
		}
		t.nodes[limp].labels = bl;
		t.nodes[limp].child = bc;
		child.push_back(limp);
	}
	std::set<double> sizes;
	if (!open_levels.empty())
		for (double to : open_levels[0]) sizes.insert(to >= 0.7 * stack ? stack : to);
	sizes.insert(stack);
	for (double to : sizes) {
		labels.push_back(to >= stack ? "allin" : "raise " + num(to));
		const int c = build_facing(t, "open/" + (to >= stack ? std::string("allin") : num(to)), {(uint8_t)(labels.size() - 1)}, 1, 1, to,
		                           open_levels, 1, stack);
		child.push_back(c);
	}
	t.nodes[root].labels = labels;
	t.nodes[root].child = child;
	return t;
}

// The tree of the external solver poker-cfr (BSD-2, game_preflop.rs): every raise is a multiple of the bet
// faced (2.5x, 3x, 3.5x, 4x, capped at the stack) plus an all-in; the action numbering is the same
// (0 fold, 1 call or check, 2.. raises), so strategies can be compared node by node. The big blind may also
// fold after a limp (never right, but it is in that tree).
static int build_mult_node(GameTree& t, std::vector<uint8_t> path, int p, double prev, double cur, double stack) {
	static const double MULT[4] = {2.5, 3.0, 3.5, 4.0}, THR[4] = {1.0, 2.5, 3.0, 3.5};
	const int d = t.add(GNode());
	std::string id = "r";
	for (uint8_t a : path) id += "/" + std::to_string(a);
	t.nodes[d].id = id;
	t.nodes[d].player = p;
	t.nodes[d].path = path;
	GNode f;
	f.kind = FOLD;
	f.player = p;
	f.c0 = p == 0 ? prev : cur;
	f.c1 = p == 0 ? cur : prev;
	GNode s;
	s.kind = SHOW;
	s.c0 = cur;
	s.allin = cur >= stack;
	const bool limp = path.size() == 1 && path[0] == 1;
	std::vector<std::string> labels = {"fold", limp ? "check" : "call"};
	std::vector<int> child = {t.add(f), -1};
	// At the root "call" is a limp and the big blind still has to act (only later calls end the hand).
	child[1] = path.empty() ? build_mult_node(t, {1}, 1, 1.0, 1.0, stack) : t.add(s);
	const double ratio = stack / cur;
	std::vector<double> to;
	for (int k = 0; k < 4; k++)
		if (ratio > THR[k]) to.push_back(std::min(cur * MULT[k], stack));
	if (ratio > 4.0) to.push_back(stack);
	for (size_t k = 0; k < to.size(); k++) {
		labels.push_back(to[k] >= stack ? "allin" : "raise " + num(to[k]));
		std::vector<uint8_t> np = path;
		np.push_back((uint8_t)(2 + k));
		child.push_back(build_mult_node(t, np, 1 - p, cur, to[k], stack));
	}
	t.nodes[d].labels = labels;
	t.nodes[d].child = child;
	return d;
}

static GameTree build_mult_tree(double stack) {
	GameTree t;
	build_mult_node(t, {}, 0, 0.5, 1.0, stack);
	return t;
}

// -- the problem: matrices and priors ------------------------------------------------------------
// m[real][player][i*n+j] = (combos of i that do not clash with j) x (player's share of the pot when hand i
// meets hand j); real = 0 raw equity, 1 realised.
struct Problem {
	int n = 0;
	std::vector<double> W;
	std::vector<double> M[2][2];
	std::vector<double> prior;  // share of all combos, sums to 1
	std::vector<double> wsum;   // row sums of W
};

// equity: n*n row-major, equity[i*n+j] = share of the pot class i wins against class j.
static Problem make_problem(const std::vector<double>& equity, bool realisation) {
	Problem pb;
	const int n = NCLASS;
	pb.n = n;
	pb.W.assign(n * n, 0);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			int c = 0;
			for (int a = 0; a < g_ncombo[i]; a++)
				for (int b = 0; b < g_ncombo[j]; b++) {
					const uint8_t *x = g_combos[i][a], *y = g_combos[j][b];
					if (x[0] != y[0] && x[0] != y[1] && x[1] != y[0] && x[1] != y[1]) c++;
				}
			pb.W[i * n + j] = c;
		}
	for (int r = 0; r < 2; r++)
		for (int p = 0; p < 2; p++) pb.M[r][p].assign(n * n, 0);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			const double w = pb.W[i * n + j], eij = equity[i * n + j], eji = equity[j * n + i];
			pb.M[0][0][i * n + j] = w * eij;
			pb.M[0][1][i * n + j] = w * (1 - eji);
			pb.M[1][0][i * n + j] = realisation ? w * realised_ip(eij) : w * eij;
			pb.M[1][1][i * n + j] = realisation ? w * (1 - realised_ip(eji)) : w * (1 - eji);
		}
	pb.prior.assign(n, 0);
	double total = 0;
	for (int i = 0; i < n; i++) {
		pb.prior[i] = g_ncombo[i];
		total += pb.prior[i];
	}
	for (int i = 0; i < n; i++) pb.prior[i] /= total;
	pb.wsum.assign(n, 0);
	for (int i = 0; i < n; i++) {
		double s = 0;
		for (int j = 0; j < n; j++) s += pb.W[i * n + j];
		pb.wsum[i] = s;
	}
	return pb;
}

static std::vector<double> read_matrix(const char* path) {
	std::vector<double> m(NCLASS * NCLASS);
	FILE* f = fopen(path, "rb");
	if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
	if (fread(m.data(), sizeof(double), m.size(), f) != m.size()) { fprintf(stderr, "%s: wrong size\n", path); exit(1); }
	fclose(f);
	return m;
}

// -- device side ---------------------------------------------------------------------------------
struct DevNode {
	int kind, player, nact, aoff;  // aoff: first of this decision's per-action vectors in sigma/regret/sum
	int child[MAXA];
	int allin;
	double c0, c1;
};

struct Mats {
	const double *W, *M[2][2];  // transposed, padded: [j * NP + i]
};

struct Ctx {
	const DevNode* nodes;
	double *reach, *val, *sigma, *regret, *sum;
	const double* t;  // iteration number (device copy, advanced by k_tick)
	Mats m;
	int n;
};

// Regret matching, then the reach of the children. regmatch: take sigma from the regrets (training);
// otherwise sigma is already in the slots (average strategy). skip: the player whose reach is not
// multiplied by his own strategy (best response: the hero keeps reach 1); -1 for none.
__global__ void k_topdown(Ctx c, const int* ids, int regmatch, int skip) {
	const int g = ids[blockIdx.x], h = threadIdx.x;
	if (h >= c.n) return;
	const DevNode& nd = c.nodes[g];
	const int k = nd.nact, p = nd.player;
	if (regmatch) {
		double pos = 0;
		for (int a = 0; a < k; a++) pos += c.regret[(size_t)(nd.aoff + a) * NP + h];
		for (int a = 0; a < k; a++)
			c.sigma[(size_t)(nd.aoff + a) * NP + h] = pos > 0 ? c.regret[(size_t)(nd.aoff + a) * NP + h] / pos : 1.0 / k;
	}
	const double r0 = c.reach[((size_t)g * 2 + 0) * NP + h], r1 = c.reach[((size_t)g * 2 + 1) * NP + h];
	for (int a = 0; a < k; a++) {
		const double s = c.sigma[(size_t)(nd.aoff + a) * NP + h];
		const size_t ch = (size_t)nd.child[a] * 2;
		c.reach[(ch + 0) * NP + h] = (p == 0 && skip != 0) ? r0 * s : r0;
		c.reach[(ch + 1) * NP + h] = (p == 1 && skip != 1) ? r1 * s : r1;
	}
}

// The terminals: values of both players for every hand class, from the reach of both.
__global__ void k_show(Ctx c, const int* ids) {
	__shared__ double r0[NP], r1[NP];
	const int g = ids[blockIdx.x], h = threadIdx.x;
	r0[h] = c.reach[((size_t)g * 2 + 0) * NP + h];
	r1[h] = c.reach[((size_t)g * 2 + 1) * NP + h];
	__syncthreads();
	if (h >= c.n) return;
	const DevNode& nd = c.nodes[g];
	double v0, v1;
	if (nd.kind == FOLD) {
		double s0 = 0, s1 = 0;  // what each player sees: combos of the other side still in the hand
		for (int j = 0; j < c.n; j++) {
			const double w = c.m.W[j * NP + h];
			s0 += w * r1[j];
			s1 += w * r0[j];
		}
		const double loss = nd.player == 0 ? nd.c0 : nd.c1;  // what the folder gives up is what the other wins
		v0 = s0 * (nd.player == 0 ? -loss : loss);
		v1 = s1 * (nd.player == 1 ? -loss : loss);
	} else {
		const double *m0 = c.m.M[nd.allin ? 0 : 1][0], *m1 = c.m.M[nd.allin ? 0 : 1][1];
		double g0 = 0, p0 = 0, g1 = 0, p1 = 0;
		for (int j = 0; j < c.n; j++) {
			const double w = c.m.W[j * NP + h];
			g0 += m0[j * NP + h] * r1[j];
			p0 += w * r1[j];
			g1 += m1[j * NP + h] * r0[j];
			p1 += w * r0[j];
		}
		const double pot = 2 * nd.c0;
		v0 = g0 * pot - nd.c0 * p0;
		v1 = g1 * pot - nd.c0 * p1;
	}
	c.val[((size_t)g * 2 + 0) * NP + h] = v0;
	c.val[((size_t)g * 2 + 1) * NP + h] = v1;
}

// mode 0: training (values, regret and strategy-sum updates); 1: value of the profile in the sigma slots;
// 2: best-response value for player `hero` against the strategy in the sigma slots.
__global__ void k_bottomup(Ctx c, const int* ids, int mode, int hero) {
	const int g = ids[blockIdx.x], h = threadIdx.x;
	if (h >= c.n) return;
	const DevNode& nd = c.nodes[g];
	const int k = nd.nact, p = nd.player;
	double* val = c.val;
	if (mode == 2) {
		double out;
		if (p == hero) {
			out = val[((size_t)nd.child[0] * 2 + hero) * NP + h];
			for (int a = 1; a < k; a++) out = fmax(out, val[((size_t)nd.child[a] * 2 + hero) * NP + h]);
		} else {
			out = 0;
			for (int a = 0; a < k; a++) out += val[((size_t)nd.child[a] * 2 + hero) * NP + h];
		}
		val[((size_t)g * 2 + hero) * NP + h] = out;
		return;
	}
	double vp = 0, vo = 0;
	for (int a = 0; a < k; a++) {
		const size_t ch = (size_t)nd.child[a] * 2;
		vp += c.sigma[(size_t)(nd.aoff + a) * NP + h] * val[(ch + p) * NP + h];
		vo += val[(ch + 1 - p) * NP + h];
	}
	val[((size_t)g * 2 + p) * NP + h] = vp;
	val[((size_t)g * 2 + 1 - p) * NP + h] = vo;
	if (mode == 1) return;
	const double t = *c.t, reach = c.reach[((size_t)g * 2 + p) * NP + h];
	for (int a = 0; a < k; a++) {
		const size_t s = (size_t)(nd.aoff + a) * NP + h;
		const double av = val[((size_t)nd.child[a] * 2 + p) * NP + h];
		c.regret[s] = fmax(0.0, c.regret[s] + av - vp);
		c.sum[s] += t * reach * c.sigma[s];  // linear averaging
	}
}

// sigma slots := the averaged strategy (uniform where a hand never gets there).
__global__ void k_average(Ctx c, const int* ids) {
	const int g = ids[blockIdx.x], h = threadIdx.x;
	if (h >= c.n) return;
	const DevNode& nd = c.nodes[g];
	double s = 0;
	for (int a = 0; a < nd.nact; a++) s += c.sum[(size_t)(nd.aoff + a) * NP + h];
	for (int a = 0; a < nd.nact; a++)
		c.sigma[(size_t)(nd.aoff + a) * NP + h] = s > 0 ? c.sum[(size_t)(nd.aoff + a) * NP + h] / s : 1.0 / nd.nact;
}

__global__ void k_tick(double* t) { *t += 1; }

// -- the solver (host) ---------------------------------------------------------------------------
struct Solver {
	Problem pb;
	std::vector<GameTree> trees;
	std::vector<int> base;  // global index of each tree's root
	std::vector<DevNode> hnodes;
	std::vector<int> depth, dec_ids, term_ids;
	std::vector<std::vector<int> > levels;  // decision nodes by depth (global ids)
	std::vector<int> lv_off;                // where each level starts in d_lv
	int G = 0, S = 0;                       // nodes, action slots
	long long done = 0;                     // iterations finished
	Ctx ctx;
	int *d_lv = nullptr, *d_dec = nullptr, *d_term = nullptr;
	double *d_reach = nullptr, *d_val = nullptr, *d_sigma = nullptr, *d_regret = nullptr, *d_sum = nullptr, *d_t = nullptr;
	DevNode* d_nodes = nullptr;
	double* d_mats[5] = {};
	cudaGraphExec_t exec = nullptr;
	cudaStream_t stream = nullptr;

	Solver(const Problem& problem, const std::vector<GameTree>& ts) : pb(problem), trees(ts) {
		if (pb.n > NP) { fprintf(stderr, "too many hands\n"); exit(1); }
		int aoff = 0;
		for (const GameTree& t : trees) {
			base.push_back(G);
			std::vector<int> dep(t.nodes.size(), 0);
			for (size_t i = 0; i < t.nodes.size(); i++)
				for (int c : t.nodes[i].child) dep[c] = dep[i] + 1;
			for (size_t i = 0; i < t.nodes.size(); i++) {
				const GNode& s = t.nodes[i];
				DevNode d = {};
				d.kind = s.kind;
				d.player = s.player;
				d.allin = s.allin;
				d.c0 = s.c0;
				d.c1 = s.c1;
				if (s.kind == DECISION) {
					if ((int)s.child.size() > MAXA) { fprintf(stderr, "too many actions\n"); exit(1); }
					d.nact = (int)s.child.size();
					d.aoff = aoff;
					aoff += d.nact;
					for (int a = 0; a < d.nact; a++) d.child[a] = base.back() + s.child[a];
					dec_ids.push_back(G + (int)i);
					if (levels.size() <= (size_t)dep[i]) levels.resize(dep[i] + 1);
					levels[dep[i]].push_back(G + (int)i);
				} else {
					term_ids.push_back(G + (int)i);
				}
				hnodes.push_back(d);
				depth.push_back(dep[i]);
			}
			G += (int)t.nodes.size();
		}
		S = std::max(aoff, 1);
		upload();
	}

	~Solver() {
		if (exec) cudaGraphExecDestroy(exec);
		if (stream) cudaStreamDestroy(stream);
		cudaFree(d_lv); cudaFree(d_dec); cudaFree(d_term); cudaFree(d_reach); cudaFree(d_val);
		cudaFree(d_sigma); cudaFree(d_regret); cudaFree(d_sum); cudaFree(d_t); cudaFree(d_nodes);
		for (double* m : d_mats) cudaFree(m);
	}
	Solver(const Solver&) = delete;

	template <class T> T* put(const std::vector<T>& v) {
		T* d;
		CK(cudaMalloc(&d, std::max<size_t>(v.size(), 1) * sizeof(T)));
		CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
		return d;
	}
	double* zeros(size_t count) {
		double* d;
		CK(cudaMalloc(&d, count * sizeof(double)));
		CK(cudaMemset(d, 0, count * sizeof(double)));
		return d;
	}

	void upload() {
		d_nodes = put(hnodes);
		std::vector<int> flat;
		for (auto& l : levels) {
			lv_off.push_back((int)flat.size());
			flat.insert(flat.end(), l.begin(), l.end());
		}
		lv_off.push_back((int)flat.size());
		d_lv = put(flat);
		d_dec = put(dec_ids);
		d_term = put(term_ids);
		// matrices, transposed so that thread i reads column entries j = 0.. with stride-1 across the warp
		const int n = pb.n;
		const std::vector<double>* src[5] = {&pb.W, &pb.M[0][0], &pb.M[0][1], &pb.M[1][0], &pb.M[1][1]};
		for (int m = 0; m < 5; m++) {
			std::vector<double> t((size_t)NP * NP, 0.0);
			for (int i = 0; i < n; i++)
				for (int j = 0; j < n; j++) t[(size_t)j * NP + i] = (*src[m])[i * n + j];
			d_mats[m] = put(t);
		}
		d_reach = zeros((size_t)G * 2 * NP);
		d_val = zeros((size_t)G * 2 * NP);
		d_sigma = zeros((size_t)S * NP);
		d_regret = zeros((size_t)S * NP);
		d_sum = zeros((size_t)S * NP);
		d_t = zeros(1);
		set_t(1);
		std::vector<double> ones((size_t)G * 2 * NP, 0.0);  // root reach: 1 for every real hand class
		for (int b : base)
			for (int p = 0; p < 2; p++)
				for (int h = 0; h < n; h++) ones[((size_t)b * 2 + p) * NP + h] = 1;
		CK(cudaMemcpy(d_reach, ones.data(), ones.size() * sizeof(double), cudaMemcpyHostToDevice));
		ctx.nodes = d_nodes;
		ctx.reach = d_reach; ctx.val = d_val; ctx.sigma = d_sigma; ctx.regret = d_regret; ctx.sum = d_sum;
		ctx.t = d_t;
		ctx.m.W = d_mats[0];
		ctx.m.M[0][0] = d_mats[1]; ctx.m.M[0][1] = d_mats[2]; ctx.m.M[1][0] = d_mats[3]; ctx.m.M[1][1] = d_mats[4];
		ctx.n = n;
		CK(cudaStreamCreate(&stream));
	}

	void set_t(double t) { CK(cudaMemcpy(d_t, &t, sizeof t, cudaMemcpyHostToDevice)); }

	int nlevels() const { return (int)levels.size(); }
	int lv_count(int l) const { return lv_off[l + 1] - lv_off[l]; }

	void sweep_down(cudaStream_t s, int regmatch, int skip) {
		for (int l = 0; l < nlevels(); l++)
			if (lv_count(l)) k_topdown<<<lv_count(l), NP, 0, s>>>(ctx, d_lv + lv_off[l], regmatch, skip);
		if (!term_ids.empty()) k_show<<<(int)term_ids.size(), NP, 0, s>>>(ctx, d_term);
	}
	void sweep_up(cudaStream_t s, int mode, int hero) {
		for (int l = nlevels() - 1; l >= 0; l--)
			if (lv_count(l)) k_bottomup<<<lv_count(l), NP, 0, s>>>(ctx, d_lv + lv_off[l], mode, hero);
	}
	void enqueue_iteration(cudaStream_t s) {
		sweep_down(s, 1, -1);
		sweep_up(s, 0, 0);
		k_tick<<<1, 1, 0, s>>>(d_t);
	}

	// Run `count` more iterations.
	void iterate(long long count) {
		if (!exec) {
			cudaGraph_t graph;
			CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
			enqueue_iteration(stream);
			CK(cudaStreamEndCapture(stream, &graph));
			CK(cudaGraphInstantiate(&exec, graph, 0));
			CK(cudaGraphDestroy(graph));
		}
		for (long long i = 0; i < count; i++) CK(cudaGraphLaunch(exec, stream));
		CK(cudaStreamSynchronize(stream));
		CK(cudaGetLastError());
		done += count;
	}

	void average() {
		if (!dec_ids.empty()) k_average<<<(int)dec_ids.size(), NP, 0, stream>>>(ctx, d_dec);
		CK(cudaStreamSynchronize(stream));
	}

	double root_gain(const std::vector<double>& v, int inst) const {  // one player's expected value in bb
		double s = 0;
		for (int h = 0; h < pb.n; h++) s += (pb.prior[h] * v[h]) / pb.wsum[h];
		(void)inst;
		return s;
	}
	std::vector<double> root_vec(int inst, int p) {
		std::vector<double> v(NP);
		CK(cudaMemcpy(v.data(), d_val + ((size_t)base[inst] * 2 + p) * NP, NP * sizeof(double), cudaMemcpyDeviceToHost));
		return v;
	}

	// What a best response wins against the averaged strategy, per tree, from both seats: {gain0, gain1}.
	std::vector<std::pair<double, double> > exploitability() {
		average();
		std::vector<std::pair<double, double> > out(trees.size());
		for (int hero = 0; hero < 2; hero++) {
			sweep_down(stream, 0, hero);
			sweep_up(stream, 2, hero);
			CK(cudaStreamSynchronize(stream));
			for (size_t i = 0; i < trees.size(); i++) {
				const double g = root_gain(root_vec((int)i, hero), (int)i);
				(hero == 0 ? out[i].first : out[i].second) = g;
			}
		}
		return out;
	}

	// Expected value (bb) of the averaged profile for {small blind, big blind}, per tree.
	std::vector<std::pair<double, double> > value() {
		average();
		sweep_down(stream, 0, -1);
		sweep_up(stream, 1, 0);
		CK(cudaStreamSynchronize(stream));
		std::vector<std::pair<double, double> > out(trees.size());
		for (size_t i = 0; i < trees.size(); i++) {
			out[i].first = root_gain(root_vec((int)i, 0), (int)i);
			out[i].second = root_gain(root_vec((int)i, 1), (int)i);
		}
		return out;
	}

	std::vector<double> download(const double* d) {
		std::vector<double> v((size_t)S * NP);
		CK(cudaMemcpy(v.data(), d, v.size() * sizeof(double), cudaMemcpyDeviceToHost));
		return v;
	}
	// Averaged strategy of every action slot: [slot * NP + hand].
	std::vector<double> averaged() {
		average();
		return download(d_sigma);
	}
	std::vector<double> regrets() { return download(d_regret); }
	std::vector<double> sums() { return download(d_sum); }
	void set_sums(const std::vector<double>& v) {
		CK(cudaMemcpy(d_sum, v.data(), v.size() * sizeof(double), cudaMemcpyHostToDevice));
	}
	int slot_of(int inst, int node) const { return hnodes[base[inst] + node].aoff; }

	// -- checkpoints: regrets, strategy sums and the iteration count ---------------------------------
	bool save(const char* path) {
		std::vector<double> r = regrets(), s = sums();
		FILE* f = fopen((std::string(path) + ".tmp").c_str(), "wb");
		if (!f) return false;
		long long hdr[4] = {0x534f4c5645303031ll, G, S, done};
		fwrite(hdr, sizeof hdr, 1, f);
		fwrite(r.data(), sizeof(double), r.size(), f);
		fwrite(s.data(), sizeof(double), s.size(), f);
		fclose(f);
		return rename((std::string(path) + ".tmp").c_str(), path) == 0;
	}
	bool load(const char* path) {
		FILE* f = fopen(path, "rb");
		if (!f) return false;
		long long hdr[4];
		std::vector<double> r((size_t)S * NP), s((size_t)S * NP);
		bool ok = fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == 0x534f4c5645303031ll && hdr[1] == G && hdr[2] == S &&
		          fread(r.data(), sizeof(double), r.size(), f) == r.size() &&
		          fread(s.data(), sizeof(double), s.size(), f) == s.size();
		fclose(f);
		if (!ok) return false;
		CK(cudaMemcpy(d_regret, r.data(), r.size() * sizeof(double), cudaMemcpyHostToDevice));
		CK(cudaMemcpy(d_sum, s.data(), s.size() * sizeof(double), cudaMemcpyHostToDevice));
		done = hdr[3];
		set_t((double)done + 1);
		return true;
	}

	// -- output: the JSON of solve-preflop.ts -------------------------------------------------------
	static std::string fixed(double x, int digits) {  // Math.round(x * 10^d) / 10^d, printed shortest
		long long scale = digits == 2 ? 100 : 1000;
		long long k = (long long)std::floor(x * scale + 0.5);
		char b[48];
		snprintf(b, sizeof b, "%lld.%0*lld", k / scale, digits, k % scale);
		std::string s = b;
		while (s.back() == '0') s.pop_back();
		if (s.back() == '.') s.pop_back();
		return s;
	}

	std::string json(int inst, double stack, long long iterations, long long samples, double exploitability_bb,
	                 const std::vector<double>& avg, const std::vector<std::string>& hands) {
		const GameTree& t = trees[inst];
		const int n = pb.n;
		std::string out = "{\"stack_bb\":" + num(stack) + ",\"iterations\":" + std::to_string(iterations) +
		                  ",\"samples_per_matchup\":" + std::to_string(samples) + ",\"exploitability_bb\":" +
		                  fixed(exploitability_bb, 3) + ",\"nodes\":{";
		bool first = true;
		std::vector<std::vector<double> > ones(2, std::vector<double>(n, 1.0));
		std::function<void(int, std::vector<std::vector<double> >)> emit = [&](int node, std::vector<std::vector<double> > reach) {
			const GNode& nd = t.nodes[node];
			if (nd.kind != DECISION) return;
			const DevNode& dn = hnodes[base[inst] + node];
			const std::vector<double>& mine = reach[nd.player];
			double played = 0;
			for (int h = 0; h < n; h++) played += pb.prior[h] * mine[h];
			if (played < 0.0005) return;
			if (!first) out += ",";
			first = false;
			out += "\"" + nd.id + "\":{\"player\":\"" + (nd.player == 0 ? "SB" : "BB") + "\",\"actions\":[";
			for (size_t a = 0; a < nd.labels.size(); a++) out += (a ? ",\"" : "\"") + nd.labels[a] + "\"";
			out += "],\"reach\":" + fixed(played, 3) + ",\"freq\":{";
			bool f2 = true;
			for (int h = 0; h < n; h++) {
				if (mine[h] < 0.01) continue;
				if (!f2) out += ",";
				f2 = false;
				out += "\"" + hands[h] + "\":[";
				for (int a = 0; a < dn.nact; a++)
					out += (a ? "," : "") + fixed(avg[(size_t)(dn.aoff + a) * NP + h], 2);
				out += "]";
			}
			out += "}}";
			for (int a = 0; a < dn.nact; a++) {
				std::vector<std::vector<double> > r = reach;
				for (int h = 0; h < n; h++) r[nd.player][h] = mine[h] * avg[(size_t)(dn.aoff + a) * NP + h];
				emit(nd.child[a], r);
			}
		};
		emit(0, ones);
		out += "}}";
		return out;
	}
};

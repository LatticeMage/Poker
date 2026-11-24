// Inputs of the 6-max solver (host side): the matrices, the realisation table turned into pot-share matrices, the
// source charts and the starting strategy made from them. Included by sixmax.cuh (it needs the types defined there).
#pragma once

static std::vector<double> sx_read_matrix(const std::string& path) {
	std::vector<double> m((size_t)NCLASS * NCLASS);
	FILE* f = fopen(path.c_str(), "rb");
	if (!f) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
	if (fread(m.data(), sizeof(double), m.size(), f) != m.size()) { fprintf(stderr, "%s: wrong size\n", path.c_str()); exit(1); }
	fclose(f);
	return m;
}

static std::vector<double> json_vec(const Json& a) {
	std::vector<double> v;
	for (const Json& x : a.arr) v.push_back(x.num);
	return v;
}

// The old heads-up heuristic of solve-preflop.ts, used for limped pots only (no table exists for them).
static double sx_realised_ip(double eq) { return std::min(1.0, std::max(0.0, 0.5 + 1.08 * (eq - 0.5) + 0.025)); }

// The share of the flop pot the player in position takes, per (class in position, class out of position), for one
// pot type of data/realization.json at one depth (linear between the table's depths, the end values outside).
// X(i,j) is the average of what the in-position table says IP wins with class i against class j and what the
// out-of-position table says OOP does NOT win with class j against i, so that the two take the whole pot between
// them (zero-sum). Each side's number is its table's `ev_bb` moved along the line through (raw equity, ev) with the
// slope factor * final_pot (README, "Use in the solver"). Empty result: the pot type does not exist at that depth.
struct TableS {
	std::vector<double> S;
	std::vector<double> f_ip, f_oop;  // the classes' realisation factors, same interpolation
};

static TableS table_S(const Json& real, const std::string& pot, double depth, const std::vector<double>& eq) {
	TableS out;
	const int n = NCLASS;
	const Json* p = nullptr;
	for (const Json& x : real.at("pots").arr)
		if (x.at("pot").str == pot) p = &x;
	if (!p) { fprintf(stderr, "no pot type %s in the realisation file\n", pot.c_str()); exit(1); }
	std::vector<std::pair<double, const Json*> > ds;
	for (const Json& d : p->at("depths").arr) ds.push_back({d.at("depth_bb").num, &d});
	if (depth < ds.front().first - 1e-9 && pot == "4bet") return out;  // a 4-bet pot at 20bb is an all-in
	std::vector<std::pair<const Json*, double> > use;
	if (depth <= ds.front().first) use.push_back({ds.front().second, 1.0});
	else if (depth >= ds.back().first) use.push_back({ds.back().second, 1.0});
	else
		for (size_t k = 0; k + 1 < ds.size(); k++)
			if (depth >= ds[k].first && depth <= ds[k + 1].first) {
				const double w = (depth - ds[k].first) / (ds[k + 1].first - ds[k].first);
				if (w < 1) use.push_back({ds[k].second, 1 - w});
				if (w > 0) use.push_back({ds[k + 1].second, w});
				break;
			}
	out.S.assign((size_t)n * n, 0.0);
	out.f_ip.assign(n, 0.0);
	out.f_oop.assign(n, 0.0);
	for (const auto& u : use) {
		const Json& d = *u.first;
		const double pot_bb = d.at("pot_bb").num;
		const Json &ip = d.at("ip"), &oop = d.at("oop");
		const std::vector<double> rawi = json_vec(ip.at("raw")), fi = json_vec(ip.at("factor")), evi = json_vec(ip.at("ev_bb")),
		                          fpi = json_vec(ip.at("final_pot_bb")), rawo = json_vec(oop.at("raw")), fo = json_vec(oop.at("factor")),
		                          evo = json_vec(oop.at("ev_bb")), fpo = json_vec(oop.at("final_pot_bb"));
		for (int i = 0; i < n; i++) {
			out.f_ip[i] += u.second * fi[i];
			out.f_oop[i] += u.second * fo[i];
			for (int j = 0; j < n; j++) {
				const double e_ip = evi[i] + fi[i] * fpi[i] * (eq[i * n + j] - rawi[i]);
				const double e_oop = evo[j] + fo[j] * fpo[j] * (eq[j * n + i] - rawo[j]);
				const double X = 0.5 * (e_ip + pot_bb - e_oop);
				out.S[(size_t)i * n + j] += u.second * std::min(1.0, std::max(0.0, X / pot_bb));
			}
		}
	}
	return out;
}

// Everything the engine needs for the 169 classes. `real` and `e3` may be null for tests that need neither.
static Problem6 make_problem6(const std::vector<double>& eq, const Json* real, const std::vector<uint16_t>* e3, double depth,
                              bool with3) {
	Problem6 pb;
	const int n = NCLASS;
	pb.n = n;
	std::vector<double> W((size_t)n * n);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			int c = 0;
			for (int a = 0; a < g_ncombo[i]; a++)
				for (int b = 0; b < g_ncombo[j]; b++) {
					const uint8_t *x = g_combos[i][a], *y = g_combos[j][b];
					if (x[0] != y[0] && x[0] != y[1] && x[1] != y[0] && x[1] != y[1]) c++;
				}
			W[(size_t)i * n + j] = c;
		}
	pb.Wn.assign((size_t)n * n, 0.0);
	pb.prior.assign(n, 0.0);
	double total = 0;
	for (int i = 0; i < n; i++) total += g_ncombo[i];
	for (int i = 0; i < n; i++) {
		double ws = 0;
		for (int j = 0; j < n; j++) ws += W[(size_t)i * n + j];
		for (int j = 0; j < n; j++) pb.Wn[(size_t)i * n + j] = W[(size_t)i * n + j] / ws;
		pb.prior[i] = g_ncombo[i] / total;
	}
	std::vector<double> S[NCAT];
	S[CAT_LIMP].assign((size_t)n * n, 0.0);
	S[CAT_ALLIN].assign((size_t)n * n, 0.0);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			S[CAT_LIMP][(size_t)i * n + j] = sx_realised_ip(eq[i * n + j]);
			S[CAT_ALLIN][(size_t)i * n + j] = eq[i * n + j];
		}
	if (real) {
		const char* names[3] = {"srp", "3bet", "4bet"};
		for (int k = 0; k < 3; k++) {
			TableS t = table_S(*real, names[k], depth, eq);
			S[CAT_SRP + k] = t.S;
			if (k == 0) {
				pb.grole[0] = t.f_oop;
				pb.grole[2] = t.f_ip;
				pb.grole[1].resize(n);
				for (int i = 0; i < n; i++) pb.grole[1][i] = 0.5 * (t.f_ip[i] + t.f_oop[i]);
			}
		}
	}
	for (int c = 0; c < NCAT; c++) {
		pb.S[c] = S[c];
		if (S[c].empty()) continue;
		pb.A_ip[c].assign((size_t)n * n, 0.0);
		pb.A_oop[c].assign((size_t)n * n, 0.0);
		for (int h = 0; h < n; h++)
			for (int j = 0; j < n; j++) {
				pb.A_ip[c][(size_t)h * n + j] = pb.Wn[(size_t)h * n + j] * S[c][(size_t)h * n + j];
				pb.A_oop[c][(size_t)h * n + j] = pb.Wn[(size_t)h * n + j] * S[c][(size_t)j * n + h];
			}
	}
	if (with3 && e3) {
		for (int s = 0; s < 3; s++) pb.T[s].assign((size_t)n * n * n, make_float4(0, 0, 0, 0));
		pb.NS.assign(n, 0.0);
		for (int a = 0; a < n; a++)
			for (int b = 0; b < n; b++)
				for (int c = 0; c < n; c++) {
					float e[3];
					const uint32_t N = e3_lookup(e3->data(), a, b, c, e);
					const float4 v = make_float4(e[0], e[1], e[2], (float)N);
					pb.T[0][((size_t)a * n + b) * n + c] = v;
					pb.T[1][((size_t)b * n + a) * n + c] = v;
					pb.T[2][((size_t)c * n + a) * n + b] = v;
					pb.NS[a] += N;
				}
	}
	return pb;
}

// -- the source charts (host) ------------------------------------------------------------------------
struct Source {
	std::vector<double> rfi[NPLAYER];
	double stated[NPLAYER] = {};
	std::string key[NPLAYER];
	// [hero][opener]: facing an open; [opener][3-bettor]: facing a 3-bet
	std::vector<double> vo_call[NPLAYER][NPLAYER], vo_bluff[NPLAYER][NPLAYER], vo_value[NPLAYER][NPLAYER];
	std::vector<double> v3_call[NPLAYER][NPLAYER], v3_bluff[NPLAYER][NPLAYER], v3_value[NPLAYER][NPLAYER];
	std::vector<double> iso, limp_call;  // not in the sources: made up, as a starting point only
};

static int seat_of(const std::string& s) {
	for (int p = 0; p < NPLAYER; p++)
		if (s == SEAT[p]) return p;
	fprintf(stderr, "unknown seat %s\n", s.c_str());
	exit(1);
}

static Source load_source(const std::string& dir) {
	Source src;
	const Json charts = json_load(dir + "/charts.json"), spots = json_load(dir + "/spots.json");
	const char* keys[5] = {"UTG6", "HJ6", "CO6", "BTN6", "SB6"};
	for (int p = 0; p < 5; p++) {
		bool found = false;
		for (const Json& c : charts.at("charts").arr)
			if (c.at("key").str == keys[p]) {
				src.rfi[p] = parse_weighted(c.at("text").str);
				src.stated[p] = c.at("stated_pct").num;
				src.key[p] = keys[p];
				found = true;
			}
		if (!found) { fprintf(stderr, "chart %s missing\n", keys[p]); exit(1); }
	}
	for (const auto& kv : spots.at("vs_open").obj) {
		const size_t bar = kv.first.find('|');
		const int hero = seat_of(kv.first.substr(0, bar)), op = seat_of(kv.first.substr(bar + 1));
		src.vo_call[hero][op] = parse_weighted(kv.second.at("call").str);
		src.vo_bluff[hero][op] = parse_weighted(kv.second.at("bluff").str);
		src.vo_value[hero][op] = parse_weighted(kv.second.at("value").str);
	}
	for (const auto& kv : spots.at("vs_3bet").obj) {
		const size_t bar = kv.first.find('|');
		const int hero = seat_of(kv.first.substr(0, bar)), r = seat_of(kv.first.substr(bar + 1));
		src.v3_call[hero][r] = parse_weighted(kv.second.at("call").str);
		src.v3_bluff[hero][r] = parse_weighted(kv.second.at("bluff").str);
		src.v3_value[hero][r] = parse_weighted(kv.second.at("value").str);
	}
	src.iso = parse_weighted("TT+, AJs+, KQs, AQo+, A5s-A4s");
	src.limp_call = parse_weighted("22+, A2s+, K8s+, Q9s+, J9s+, T9s, 98s, 87s, 76s, ATo+, KJo+, QJo");
	return src;
}

// -- the starting strategy: the source charts ---------------------------------------------------------
// Per node and class, the probability of each kind of action (fold, call, aggressive, limp), then spread over
// the node's actions. A kind the node does not have gives its mass to calling (or folding if there is no call).
static void init_probs(const Node6& nd, const Source& src, int h, double* kp /*[4]: fold, call, aggr, limp*/) {
	kp[0] = kp[1] = kp[2] = kp[3] = 0;
	auto at = [&](const std::vector<double>& v) { return v.empty() ? 0.0 : v[h]; };
	const std::string& nm = g_names[h];
	const double shove = (nm == "AA" || nm == "KK") ? 1.0 : (nm == "QQ" || nm == "AKs" || nm == "AKo") ? 0.3 : 0.0;
	switch (nd.type) {
		case NT_RFI: kp[2] = at(src.rfi[nd.player]); kp[0] = 1 - kp[2]; break;
		case NT_LIMP_BB: kp[2] = at(src.iso); kp[1] = 1 - kp[2]; break;
		case NT_LIMP_SB: kp[1] = at(src.limp_call); kp[0] = 1 - kp[1]; break;
		case NT_RESP: {
			const int r = nd.player, o = nd.opener;
			kp[2] = std::min(1.0, at(src.vo_bluff[r][o]) + at(src.vo_value[r][o]));
			kp[1] = std::min(1 - kp[2], at(src.vo_call[r][o]));
			kp[0] = 1 - kp[1] - kp[2];
			break;
		}
		case NT_FACE3: {
			const int o = nd.player, r = nd.raiser;
			kp[2] = std::min(1.0, at(src.v3_bluff[o][r]) + at(src.v3_value[o][r]));
			kp[1] = std::min(1 - kp[2], at(src.v3_call[o][r]));
			kp[0] = 1 - kp[1] - kp[2];
			break;
		}
		case NT_FACE4: {  // the 3-bettor meets a 4-bet: the hands that 3-bet for value continue
			const int r = nd.player, o = nd.opener;
			const double b = at(src.vo_bluff[r][o]), v = at(src.vo_value[r][o]);
			const double cont = b + v > 0 ? v / (b + v) : 0.0;
			kp[2] = cont * shove;
			kp[1] = cont - kp[2];
			kp[0] = 1 - cont;
			break;
		}
		case NT_FACE5: {  // the opener meets a 5-bet: the hands that 4-bet for value call
			const int o = nd.player, r = nd.raiser;
			const double b = at(src.v3_bluff[o][r]), v = at(src.v3_value[o][r]);
			kp[1] = b + v > 0 ? v / (b + v) : 0.0;
			kp[0] = 1 - kp[1];
			break;
		}
		default: break;
	}
}

// sigma[(aoff + a) * NPAD + h] for every decision; frozen nodes get the exact source, the others a little noise.
static std::vector<double> init_sigma(const std::vector<Node6>& nodes, const std::vector<DNode>& hn, int S, const Source& src,
                                      const std::vector<bool>& frozen, double eps) {
	std::vector<double> sigma((size_t)S * NPAD, 0.0);
	for (size_t i = 0; i < nodes.size(); i++) {
		const Node6& nd = nodes[i];
		if (nd.is_term()) continue;
		const int k = (int)nd.child.size();
		for (int h = 0; h < NCLASS; h++) {
			double kp[4];
			init_probs(nd, src, h, kp);
			bool has[4] = {false, false, false, false};
			int slot[MAXACT];
			for (int a = 0; a < k; a++) {
				const int kind = nd.akind[a];
				slot[a] = kind == A_FOLD ? 0 : kind == A_CALL ? 1 : kind == A_LIMP ? 3 : 2;
				has[slot[a]] = true;
			}
			for (int s2 = 0; s2 < 4; s2++)
				if (!has[s2] && kp[s2] > 0) {
					kp[has[1] ? 1 : 0] += kp[s2];
					kp[s2] = 0;
				}
			double tot = 0;
			for (int s2 = 0; s2 < 4; s2++) tot += kp[s2];
			const bool fz = i < frozen.size() && frozen[i];
			for (int a = 0; a < k; a++) {
				double p = tot > 0 ? kp[slot[a]] / tot : 1.0 / k;
				if (!fz) p = (1 - eps) * p + eps / k;
				sigma[(size_t)(hn[i].aoff + a) * NPAD + h] = p;
			}
		}
	}
	return sigma;
}

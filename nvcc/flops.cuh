// The 1,755 strategically distinct flops: the 22,100 three-card flops grouped by suit permutation (24 of them
// map a flop to another flop with the same strategic meaning). Each class has a representative flop and a weight
// = how many of the 22,100 flops it stands for (4, 12 or 24). Host code, no GPU needed.
//
//   const std::vector<Flop>& flops = all_flops();   // 1,755 entries, canonical order
//   Flop f: cards[3] (0..51, rank index * 4 + suit, sorted), weight, kind (rainbow/two-tone/monotone),
//           texture bits (paired, high card rank, connectedness) for grouping by board type.
//
// Used to weight a flop-by-flop computation so that it equals the average over all 22,100 flops, with 1,755
// computations instead of 22,100.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

struct Flop {
	int index;           // position in all_flops(): the id of the class, the same in every list
	int cards[3];        // sorted ascending; the representative of the class
	int weight;          // number of the 22,100 flops this class stands for
	int suit_kind;       // 0 rainbow (3 suits), 1 two-tone (2 suits), 2 monotone (1 suit)
	bool paired;         // two cards of the same rank
	bool trips;          // three cards of the same rank
	int high;            // highest rank index (0 = deuce .. 12 = ace)
	int span;            // highest rank - lowest rank (connectedness: 2 = three in a row if all distinct)
};

// Canonical form of a card triple under the 24 suit permutations: the smallest sorted triple.
static inline void flop_canon(const int in[3], int out[3]) {
	static int perms[24][4];
	static bool built = false;
	if (!built) {
		int p[4] = {0, 1, 2, 3}, n = 0;
		do {
			for (int i = 0; i < 4; i++) perms[n][i] = p[i];
			n++;
		} while (std::next_permutation(p, p + 4));
		built = true;
	}
	bool have = false;
	for (int k = 0; k < 24; k++) {
		int t[3];
		for (int i = 0; i < 3; i++) t[i] = (in[i] & ~3) | perms[k][in[i] & 3];
		std::sort(t, t + 3);
		if (!have || t[0] < out[0] || (t[0] == out[0] && (t[1] < out[1] || (t[1] == out[1] && t[2] < out[2])))) {
			out[0] = t[0]; out[1] = t[1]; out[2] = t[2];
			have = true;
		}
	}
}

static inline const std::vector<Flop>& all_flops() {
	static std::vector<Flop> flops;
	if (!flops.empty()) return flops;
	// every canonical triple appears as its own canon: collect the canon triples and count the flops mapping to each
	for (int a = 0; a < 52; a++)
		for (int b = a + 1; b < 52; b++)
			for (int c = b + 1; c < 52; c++) {
				const int in[3] = {a, b, c};
				int canon[3];
				flop_canon(in, canon);
				if (canon[0] == a && canon[1] == b && canon[2] == c) {
					Flop f;
					f.index = (int)flops.size();
					f.cards[0] = a; f.cards[1] = b; f.cards[2] = c;
					f.weight = 0;
					int suits = 0;
					for (int i = 0; i < 3; i++) suits |= 1 << (in[i] & 3);
					const int ns = __builtin_popcount(suits);
					f.suit_kind = ns == 3 ? 0 : (ns == 2 ? 1 : 2);
					const int r0 = a >> 2, r1 = b >> 2, r2 = c >> 2;
					f.paired = (r0 == r1) || (r1 == r2);
					f.trips = (r0 == r1) && (r1 == r2);
					f.high = std::max(r0, std::max(r1, r2));
					f.span = f.high - std::min(r0, std::min(r1, r2));
					flops.push_back(f);
				}
			}
	// weights: count every flop under its canonical class
	for (int a = 0; a < 52; a++)
		for (int b = a + 1; b < 52; b++)
			for (int c = b + 1; c < 52; c++) {
				const int in[3] = {a, b, c};
				int canon[3];
				flop_canon(in, canon);
				for (Flop& f : flops)
					if (f.cards[0] == canon[0] && f.cards[1] == canon[1] && f.cards[2] == canon[2]) { f.weight++; break; }
			}
	return flops;
}

// -- names ------------------------------------------------------------------------------------------------------
// "AsKsQd": ranks high to low, suits as s h d c (the canonical representative's suits, nothing more).
static inline std::string flop_name(const Flop& f) {
	const char* ranks = "23456789TJQKA";
	const char* suits = "shdc";
	int c[3] = {f.cards[0], f.cards[1], f.cards[2]};
	std::sort(c, c + 3, [](int a, int b) { return (a >> 2) != (b >> 2) ? (a >> 2) > (b >> 2) : (a & 3) < (b & 3); });
	std::string out;
	for (int i = 0; i < 3; i++) {
		out += ranks[c[i] >> 2];
		out += suits[c[i] & 3];
	}
	return out;
}

// Number of broadway cards (T J Q K A) on the flop.
static inline int flop_broadway(const Flop& f) {
	int n = 0;
	for (int i = 0; i < 3; i++) n += (f.cards[i] >> 2) >= 8;
	return n;
}

// -- wetness: a RESULT, not a category ------------------------------------------------------------------------------
// Wet = different hole cards easily make different TYPES of hand on this flop (a made flush or straight, a flush draw, an
// open-ended or a gutshot straight draw, ...); dry = different hole cards can hardly make any new type (no flush, no
// straight, no draws). It is counted, not guessed: every one of the 1,176 two-card hands from the other 49 cards is
// classified on the flop; the wetness is the share of hands that land in a drawing or made-straight/flush type, and the
// number of different such types that occur says how varied it is.
//   types: FLUSH (5 of a suit), STRAIGHT (five ranks in a row, ace high or low), FD (four of a suit, at least one in hand),
//          OESD (open-ended straight draw: two different cards complete a straight), GUT (one rank completes it),
//          BDFD (exactly three of a suit with a card of it in hand: a backdoor flush draw; reported in the counts only, it is not
//          one of the types and not in the share)
enum FlopType { T_FLUSH = 1, T_STRAIGHT = 2, T_FD = 4, T_OESD = 8, T_GUT = 16, T_BDFD = 32 };

struct FlopWetness {
	double share;   // share of the 1,176 hands in FLUSH | STRAIGHT | FD | OESD | GUT (0 .. 1)
	int types;      // how many of the five main types occur at all (0 .. 5); BDFD is only in count[5]
	int count[6];   // hands per type, in the order FLUSH STRAIGHT FD OESD GUT BDFD (a hand can be in several)
};

static inline int hand_types(const int board[3], int h0, int h1) {
	const int cards[5] = {board[0], board[1], board[2], h0, h1};
	int suits[4] = {0, 0, 0, 0}, hole_suits[4] = {0, 0, 0, 0};
	unsigned ranks = 0;  // bit r = rank r present (0 = deuce .. 12 = ace); the ace is also looked at as rank -1
	for (int i = 0; i < 5; i++) {
		suits[cards[i] & 3]++;
		if (i >= 3) hole_suits[cards[i] & 3]++;
		ranks |= 1u << (cards[i] >> 2);
	}
	int t = 0;
	for (int s = 0; s < 4; s++) {
		if (suits[s] >= 5) t |= T_FLUSH;
		else if (suits[s] == 4 && hole_suits[s] >= 1) t |= T_FD;
		else if (suits[s] == 3 && hole_suits[s] >= 1) t |= T_BDFD;
	}
	// windows of five consecutive ranks: A-5 (ace low), 2-6, ..., T-A
	int made = 0, four = 0;
	for (int w = -1; w <= 8; w++) {
		int present = 0;
		for (int k = 0; k < 5; k++) {
			const int r = w + k;
			present += r < 0 ? ((ranks >> 12) & 1) : ((ranks >> r) & 1);
		}
		if (present == 5) made = 1;
		else if (present == 4) four++;
	}
	if (made) t |= T_STRAIGHT;
	else if (four >= 2) t |= T_OESD;
	else if (four == 1) t |= T_GUT;
	return t;
}

static inline FlopWetness flop_wetness_detail(const Flop& f) {
	FlopWetness w;
	w.share = 0;
	w.types = 0;
	for (int i = 0; i < 6; i++) w.count[i] = 0;
	int hands = 0, in_share = 0, seen = 0;
	for (int a = 0; a < 52; a++) {
		if (a == f.cards[0] || a == f.cards[1] || a == f.cards[2]) continue;
		for (int b = a + 1; b < 52; b++) {
			if (b == f.cards[0] || b == f.cards[1] || b == f.cards[2]) continue;
			const int t = hand_types(f.cards, a, b);
			hands++;
			if (t & (T_FLUSH | T_STRAIGHT | T_FD | T_OESD | T_GUT)) in_share++;
			for (int i = 0; i < 6; i++)
				if (t & (1 << i)) { w.count[i]++; seen |= 1 << i; }
		}
	}
	for (int i = 0; i < 5; i++) w.types += (seen >> i) & 1;  // the five main types; a backdoor flush draw is too weak to count as a type
	w.share = (double)in_share / hands;
	return w;
}

// 0 = 乾 (hardly any hand makes a new type), 1 = 偏乾, 2 = 中, 3 = 濕, 4 = 很濕 (about 4 hands in 10 or more draw or make).
static inline int flop_wetness(const Flop& f) {
	const double sh = flop_wetness_detail(f).share;
	return sh < 0.03 ? 0 : (sh < 0.12 ? 1 : (sh < 0.25 ? 2 : (sh < 0.40 ? 3 : 4)));
}
static inline double flop_wetness_share(const Flop& f) { return flop_wetness_detail(f).share; }
static inline const char* flop_wetness_name(int w) {
	static const char* n[5] = {"乾", "偏乾", "中", "濕", "很濕"};
	return n[w];
}

// -- the order by characteristics -------------------------------------------------------------------------------
// The order the 1,755 classes are listed in, level by level (the user's scheme, 2026-11-24):
//   1. 色數: 單色 (one suit) / 雙花 (two suits) / 三花 (three suits)
//   2. 高數: how many of the three cards are high (T J Q K A): 3 / 2 / 1 / 0
//   3. 高連度: how connected the high cards are    4. 低連度: the same for the low cards (2..9)
//   5. 三張同順: can all three cards sit in one straight (three different ranks within five, the ace also low)? Then two
//      hole cards make a straight at once. Chosen by measurement: with the four levels above it explains 94.6% of the
//      variance of the wetness (73 groups); 高低連度 in its place explained 77% (93 groups).
//   6. 哪幾張高牌: which high cards they are (A, K, Q, J, T each on its own), e.g. AK, QQ, T
//   7. 低牌中小: each low card is 中 (9..6) or 小 (5..2)
//   8. 高低連度: how close the lowest high card is to the highest low card
//   9. 同花的兩張 (two-tone only): both high / one high one low / both low
// Levels 1 to 5 (73 groups) describe the STRUCTURE of the flop and explain its wetness. They do not say whose range
// the flop is good for: measured on all 45 heads-up matchups of the charts (flopstats.cu), the 73 groups explain only
// 24% of the flop-to-flop variance of the range equity in single-raised pots, because that depends on WHICH ranks are
// out, not on their structure. Levels 6 and 7 add exactly that: 237 groups explain 78%, 383 groups 91% (3-bet pots
// 94%, 4-bet pots 98%), and the wetness explained goes up to 97.6%. The 383 groups of level 7 are the base of the
// postflop decision tree. Merging high ranks (J with T, or Q J T) loses most of it, so they stay separate.
// Connectedness of two cards: 對 same rank / 連 adjacent / 跳 two to four apart (one straight still holds both) /
// 斷 five or more apart (no straight holds both). Of three cards: 三條, 對＋連, 對＋跳, 對＋斷, 三連, 一連一跳, 兩跳,
// 一連一斷, 一跳一斷, 全斷. Wet and dry are NOT a level: they are a result (flop_wetness_detail), listed beside each flop.
struct FlopShape {
	int suits;                 // 1 單色, 2 雙花, 3 三花
	int n_high;                // cards T..A
	int high_conn, high_fine;  // connectedness code of the high cards (0: fewer than two of them), exact gaps for the order inside it
	int low_conn, low_fine;    // the same for the low cards
	int same_straight;         // 0 not three different ranks, 1 同順 (all three in one straight), 2 不同順
	int high_set;              // which high cards: digits (12 - rank) + 1 in base 6, high to low; smaller = higher cards
	int low_band;              // the low cards high to low, 1 中 (9..6) or 2 小 (5..2) each, in base 3
	int hl_conn, hl_dist;      // 0 none (all high or all low), 1 連, 2 跳, 3 斷; and the distance itself
	int suited, suited_fine;   // two-tone: 1 雙高, 2 一高一低, 3 雙低 (else 0); the suited ranks for the order inside it
	int gap_top, gap_low;      // rank differences: highest - middle, middle - lowest
};

static inline int flop_conn2(int d) { return d == 0 ? 1 : (d == 1 ? 2 : (d <= 4 ? 3 : 4)); }  // 對 連 跳 斷

// r: n ranks, high to low
static inline void flop_group_conn(const int* r, int n, int& code, int& fine) {
	code = 0; fine = 0;
	if (n == 2) { code = flop_conn2(r[0] - r[1]); fine = r[0] - r[1]; }
	if (n == 3) {
		const int g1 = r[0] - r[1], g2 = r[1] - r[2], a = flop_conn2(g1), b = flop_conn2(g2);
		fine = (g1 + g2) * 64 + g1;
		if (a == 1 && b == 1) code = 10;                       // 三條
		else if (a == 1 || b == 1) code = 9 + (a == 1 ? b : a);  // 對＋連 11, 對＋跳 12, 對＋斷 13
		else {
			const int lo = std::min(a, b), hi = std::max(a, b);  // 2 連, 3 跳, 4 斷
			code = lo == 2 ? (hi == 2 ? 14 : (hi == 3 ? 15 : 17)) : (lo == 3 ? (hi == 3 ? 16 : 18) : 19);
		}
	}
}

static inline const char* flop_conn_name(int code) {
	switch (code) {
		case 1: return "對"; case 2: return "連"; case 3: return "跳"; case 4: return "斷";
		case 10: return "三條"; case 11: return "對＋連"; case 12: return "對＋跳"; case 13: return "對＋斷";
		case 14: return "三連"; case 15: return "一連一跳"; case 16: return "兩跳";
		case 17: return "一連一斷"; case 18: return "一跳一斷"; case 19: return "全斷";
	}
	return "-";
}

static inline FlopShape flop_shape(const Flop& f) {
	FlopShape s;
	int c[3] = {f.cards[0], f.cards[1], f.cards[2]};
	std::sort(c, c + 3, [](int a, int b) { return a > b; });
	int hi[3], lo[3], nh = 0, nl = 0;
	for (int i = 0; i < 3; i++) {
		if ((c[i] >> 2) >= 8) hi[nh++] = c[i] >> 2;
		else lo[nl++] = c[i] >> 2;
	}
	s.suits = f.suit_kind == 2 ? 1 : (f.suit_kind == 1 ? 2 : 3);
	s.n_high = nh;
	flop_group_conn(hi, nh, s.high_conn, s.high_fine);
	flop_group_conn(lo, nl, s.low_conn, s.low_fine);
	s.high_set = 0; s.low_band = 0;
	for (int i = 0; i < nh; i++) s.high_set = s.high_set * 6 + (12 - hi[i]) + 1;
	for (int i = 0; i < nl; i++) s.low_band = s.low_band * 3 + (lo[i] >= 4 ? 1 : 2);
	s.hl_conn = 0; s.hl_dist = 0;
	if (nh && nl) {
		const int d = hi[nh - 1] - lo[0];   // the lowest high card to the highest low card
		s.hl_dist = d;
		s.hl_conn = d == 1 ? 1 : (d <= 4 ? 2 : 3);
	}
	s.same_straight = 0;
	if (!f.paired) {
		const int r0 = c[0] >> 2, r1 = c[1] >> 2, r2 = c[2] >> 2;
		int span = r0 - r2;
		if (r0 == 12) span = std::min(span, r1 + 1);   // the ace as the card under the deuce: A, r2, r1
		s.same_straight = span <= 4 ? 1 : 2;
	}
	s.suited = 0; s.suited_fine = 0;
	if (s.suits == 2) {
		int cnt[4] = {0, 0, 0, 0};
		for (int i = 0; i < 3; i++) cnt[c[i] & 3]++;
		int nsh = 0, key = 0;
		for (int i = 0; i < 3; i++)
			if (cnt[c[i] & 3] == 2) { nsh += (c[i] >> 2) >= 8; key = key * 13 + (c[i] >> 2); }
		s.suited = nsh == 2 ? 1 : (nsh == 1 ? 2 : 3);
		s.suited_fine = 168 - key;   // the higher suited cards first
	}
	s.gap_top = (c[0] >> 2) - (c[1] >> 2);
	s.gap_low = (c[1] >> 2) - (c[2] >> 2);
	return s;
}

// The name of a flop's group down to a level (1..9), e.g. "單色｜2高｜高跳｜不同順" (5), "三花｜1高｜低斷｜不同順｜K｜中小" (7).
static inline std::string flop_shape_name(const Flop& f, int depth = 5) {
	const FlopShape s = flop_shape(f);
	std::string n = s.suits == 1 ? "單色" : (s.suits == 2 ? "雙花" : "三花");
	if (depth >= 2) n += std::string("｜") + (char)('0' + s.n_high) + "高";
	if (depth >= 3 && s.high_conn) n += std::string("｜高") + flop_conn_name(s.high_conn);
	if (depth >= 4 && s.low_conn) n += std::string("｜低") + flop_conn_name(s.low_conn);
	if (depth >= 5 && s.same_straight) n += s.same_straight == 1 ? "｜同順" : "｜不同順";
	if (depth >= 6 && s.high_set) {
		n += "｜";
		std::string cards;
		for (int v = s.high_set; v; v /= 6) cards.insert(cards.begin(), "AKQJT"[v % 6 - 1]);
		n += cards;
	}
	if (depth >= 7 && s.low_band) {
		n += "｜";
		std::string bands;
		for (int v = s.low_band; v; v /= 3) bands.insert(0, v % 3 == 1 ? "中" : "小");
		n += bands;
	}
	if (depth >= 8 && s.hl_conn) n += std::string("｜高低") + (s.hl_conn == 1 ? "連" : (s.hl_conn == 2 ? "跳" : "斷"));
	if (depth >= 9 && s.suited) n += std::string("｜同花") + (s.suited == 1 ? "雙高" : (s.suited == 2 ? "高低" : "雙低"));
	return n;
}

// The classes of all_flops() in the order of the levels above (indices into all_flops()); inside a group of all nine
// levels: tighter gaps first, then higher cards first.
static inline std::vector<int> flop_shape_order() {
	const std::vector<Flop>& all = all_flops();
	std::vector<FlopShape> s;
	for (const Flop& f : all) s.push_back(flop_shape(f));
	std::vector<int> order(all.size());
	for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
	auto key = [&](int i, int k[18]) {
		const FlopShape& x = s[i];
		int c[3] = {all[i].cards[0] >> 2, all[i].cards[1] >> 2, all[i].cards[2] >> 2};
		std::sort(c, c + 3);
		const int v[18] = {x.suits, -x.n_high, x.high_conn, x.low_conn, x.same_straight, x.high_set, x.low_band, x.hl_conn, x.suited,
		                   x.high_fine, x.low_fine, x.hl_dist, x.suited_fine, -c[2], -c[1], -c[0], i, 0};
		for (int j = 0; j < 18; j++) k[j] = v[j];
	};
	std::sort(order.begin(), order.end(), [&](int a, int b) {
		int ka[18], kb[18];
		key(a, ka); key(b, kb);
		for (int j = 0; j < 18; j++)
			if (ka[j] != kb[j]) return ka[j] < kb[j];
		return false;
	});
	return order;
}

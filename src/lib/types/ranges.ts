// Shapes of the range API (backend/src/models/ranges.py).

export interface SeatRange {
	/** Seat in preflop order: UTG ... CO, BTN, SB, BB (heads-up: SB/BTN, BB). */
	name: string;
	/** Players still to act behind this seat. */
	behind: number;
	/** Share of all 1326 two-card combos opened, in percent. */
	pct: number;
	/** table: copied from a source · computed: top X% of the strength order ·
	 *  fallback: nothing found, widest sourced chart used · none: never opens. */
	source: 'table' | 'computed' | 'fallback' | 'none';
	label: string;
	url: string;
	/** Hand classes such as "AKs", "T9o", "77", strongest first. */
	hands: string[];
	/** Share of each hand's combos that open (0-1); hands near the edge of the range are mixed. */
	mix: Record<string, number>;
}

export interface RangeChart {
	players: number;
	stack_bb: number;
	open_size_bb: number;
	seats: SeatRange[];
	/** All 169 hands, strongest first. */
	order: string[];
	/** Each hand's all-in equity against one random hand (empty if unavailable). */
	equity_vs_random: Record<string, number>;
}

export interface PushFoldChart {
	/** Effective stack the ranges are for (2-20bb). */
	stack_bb: number;
	/** Hands the small blind (button) shoves, strongest first. */
	push: string[];
	/** Hands the big blind calls a shove with, strongest first. */
	call: string[];
	/** Share of each hand's combos that shove (0-1), from the solver's mixed strategy. */
	push_mix: Record<string, number>;
	/** Share of each hand's combos that call (0-1). */
	call_mix: Record<string, number>;
	push_pct: number;
	call_pct: number;
	/** What a best response still gains per hand, in big blinds. */
	exploitability_bb: number;
	samples_per_matchup: number;
	note: string;
}

/** What to do with every hand in one preflop spot; a hand's combos can be split between actions. */
export interface SpotChart {
	/** vs_open: facing an open raise. vs_3bet: you opened and face a 3-bet. */
	kind: 'vs_open' | 'vs_3bet';
	hero: string;
	/** The opener (vs_open) or the 3-bettor (vs_3bet). */
	villain: string;
	open_size_bb: number | null;
	labels: { call: string; bluff: string; value: string };
	/** Share of each hand's combos that call. */
	call: Record<string, number>;
	/** Share that raise and fold to the next raise. */
	bluff: Record<string, number>;
	/** Share that raise and continue. */
	value: Record<string, number>;
	/** Combos per action, as printed by the source. */
	combos: { call: number; bluff: number; value: number };
	url: string;
	note: string;
}

export interface SpotList {
	positions: string[];
	/** Hero position -> openers it can face. */
	vs_open: Record<string, string[]>;
	/** Hero position (who opened) -> 3-bettors it can face. */
	vs_3bet: Record<string, string[]>;
}

/** Equity when neither player's cards are known: range against range. */
export interface UnknownHands {
	my_range_eq: number;
	their_range_eq: number;
	advantage: 'you' | 'them' | 'even';
	my_combos: number;
	their_combos: number;
	/** Equity distribution: hands spread over the range, weakest to strongest; point i is at (i + 0.5) / n. */
	my_curve: number[];
	their_curve: number[];
	view: 'mine' | 'theirs';
	/** Equity of each hand class of the viewed range against the other range. */
	grid: Record<string, number>;
}

/** Equity when your own two cards are known (theirs are not). */
export interface KnownHand {
	cards: number[];
	hand: string;
	/** Against their whole range, with your cards dead for them. */
	equity: number;
	their_equity: number;
	vs_range: number;
	/** Share of your range with more equity than this hand (0 = best). */
	rank: number;
	made_hand: string;
	has_made_hand: boolean;
	action: string;
	reason: string;
}

export interface BoardEquity {
	street: 'flop' | 'turn' | 'river';
	pot: 'srp' | '3bet' | '4bet';
	opener: string;
	other: string;
	side: 'opener' | 'other';
	aggressor: boolean;
	board: number[];
	texture: string;
	unknown: UnknownHands;
	known: KnownHand | null;
	plan: {
		role: 'aggressor' | 'caller';
		bet_size: number;
		bet_frequency: number;
		value_cut: number;
		bluff_quota: number;
		bluff_share_of_bets: number;
	};
	facing: { bet: number; required_equity: number; defend_frequency: number } | null;
	note: string;
}

export interface BoardQuery {
	pot: BoardEquity['pot'];
	opener: string;
	other: string;
	side: 'opener' | 'other';
	/** The flop, then optionally the turn and river: 3 to 5 cards. */
	board: number[];
	/** Your two cards, if known. */
	hole: number[] | null;
	/** A bet you face, as a fraction of the pot. */
	bet: number | null;
	view: 'mine' | 'theirs';
}

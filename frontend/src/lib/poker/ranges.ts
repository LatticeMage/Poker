// Preflop opening ranges for 2–9 players: looked up where a source has them, computed elsewhere.
//
// Lookup
//   Hand lists copied from public charts (100bb cash, raise-first-in; data/charts.json). A seat is
//   matched by the number of players still to act behind it, not by its name, so a 9-max lojack and
//   a 6-max UTG — both with 5 behind — share a chart.
//
// Computed
//   · Strength order: every one of the 169 hands is ranked by the first (tightest) chart that
//     contains it, ties broken by its equity against a random hand (data/tables.json), or by the
//     Chen formula if that table is missing. A range is then "the top X% of this order".
//   · Seats with only a percentage in the sources (9-max UTG+1, UTG+2) take the top X% of it.
//   · Persona / style adjustments grow or shrink any range along the same order.
//   · Heads-up has no source; it falls back to the widest chart found (62%). Real heads-up opens
//     are wider.
//   · Heads-up push/fold ranges for 2–20bb stacks come from an offline fictitious-play
//     approximation of the equilibrium (tools/make_tables.py), stored in data/tables.json.
//
// Percentages are shares of the 1326 two-card combos: pairs 6, suited 4, offsuit 12.

import chartData from './data/charts.json';
import tables from './data/tables.json';
import type { Card } from './cards';

export const RANK_CHARS = '23456789TJQKA'; // index + 2 is the rank value
export const TOTAL_COMBOS = 1326;

const rankOf = (char: string) => RANK_CHARS.indexOf(char) + 2;

/** Class name of two ranks: "AA", "AKs", "T9o". */
export function handName(a: number, b: number, suited: boolean): string {
	const hi = Math.max(a, b);
	const lo = Math.min(a, b);
	if (hi === lo) return RANK_CHARS[hi - 2].repeat(2);
	return `${RANK_CHARS[hi - 2]}${RANK_CHARS[lo - 2]}${suited ? 's' : 'o'}`;
}

/** Class name of two cards (0..51 encoding). */
export function handOf(hole: ArrayLike<Card>): string {
	const a = hole[0];
	const b = hole[1];
	return handName((a >> 2) + 2, (b >> 2) + 2, (a & 3) === (b & 3));
}

export function combos(name: string): number {
	if (name.length === 2) return 6;
	return name[2] === 's' ? 4 : 12;
}

function buildAllHands(): string[] {
	const out: string[] = [];
	for (let hi = 14; hi >= 2; hi--) {
		for (let lo = hi; lo >= 2; lo--) {
			if (hi === lo) out.push(handName(hi, lo, false));
			else out.push(handName(hi, lo, true), handName(hi, lo, false));
		}
	}
	return out;
}

export const ALL_HANDS: readonly string[] = buildAllHands();

const comboCache = new Map<string, readonly (readonly [Card, Card])[]>();

/** Every two-card combination of a hand class, as card ints (rank-2)*4+suit. */
export function classCombos(name: string): readonly (readonly [Card, Card])[] {
	const cached = comboCache.get(name);
	if (cached) return cached;
	const hi = rankOf(name[0]);
	const lo = rankOf(name[1]);
	const out: [Card, Card][] = [];
	if (hi === lo) {
		for (let a = 0; a < 4; a++) {
			for (let b = a + 1; b < 4; b++) out.push([(hi - 2) * 4 + a, (hi - 2) * 4 + b]);
		}
	} else if (name[2] === 's') {
		for (let s = 0; s < 4; s++) out.push([(hi - 2) * 4 + s, (lo - 2) * 4 + s]);
	} else {
		for (let a = 0; a < 4; a++) {
			for (let b = 0; b < 4; b++) if (a !== b) out.push([(hi - 2) * 4 + a, (lo - 2) * 4 + b]);
		}
	}
	comboCache.set(name, out);
	return out;
}

const R = '[2-9TJQKA]';
const PAIR = new RegExp(`^(${R})\\1(\\+)?$`);
const PAIR_SPAN = new RegExp(`^(${R})\\1-(${R})\\2$`);
const HAND = new RegExp(`^(${R})(${R})([so])(\\+)?$`);
const HAND_SPAN = new RegExp(`^(${R})(${R})([so])-(${R})(${R})([so])$`);

/** Parse "22+, ATs+, A5s-A2s, T8s+, AJo+, 87s" into a set of hand class names. */
export function parseRange(text: string): Set<string> {
	const hands = new Set<string>();
	for (const raw of text.split(',')) {
		const token = raw.trim();
		if (!token) continue;
		let m: RegExpExecArray | null;
		if ((m = PAIR.exec(token))) {
			const low = rankOf(m[1]);
			const top = m[2] ? 14 : low;
			for (let r = low; r <= top; r++) hands.add(handName(r, r, false));
		} else if ((m = PAIR_SPAN.exec(token))) {
			const a = rankOf(m[1]);
			const b = rankOf(m[2]);
			for (let r = Math.min(a, b); r <= Math.max(a, b); r++) hands.add(handName(r, r, false));
		} else if ((m = HAND.exec(token))) {
			const hi = rankOf(m[1]);
			const lo = rankOf(m[2]);
			const suited = m[3] === 's';
			if (hi <= lo) throw new Error(`Bad range token: ${token}`);
			const top = m[4] ? hi - 1 : lo; // "T8s+" -> T8s, T9s
			for (let r = lo; r <= top; r++) hands.add(handName(hi, r, suited));
		} else if ((m = HAND_SPAN.exec(token))) {
			const hi = rankOf(m[1]);
			const lo = rankOf(m[2]);
			const suited = m[3] === 's';
			const hi2 = rankOf(m[4]);
			const lo2 = rankOf(m[5]);
			if (hi !== hi2 || hi <= Math.max(lo, lo2) || suited !== (m[6] === 's')) {
				throw new Error(`Bad range token: ${token}`);
			}
			for (let r = Math.min(lo, lo2); r <= Math.max(lo, lo2); r++)
				hands.add(handName(hi, r, suited));
		} else {
			throw new Error(`Bad range token: ${token}`);
		}
	}
	return hands;
}

export function percent(hands: Iterable<string>): number {
	let total = 0;
	for (const h of hands) total += combos(h);
	return (100 * total) / TOTAL_COMBOS;
}

// -- lookup: charts copied from public sources -------------------------------

export interface Chart {
	key: string;
	label: string;
	stated_pct: number;
	url: string;
	text: string;
}

export const CHARTS: Record<string, Chart> = Object.fromEntries(
	(chartData.charts as Chart[]).map((c) => [c.key, c])
);

/** Charts from tightest to widest; they define the strength order below. */
const LAYERS: readonly string[] = chartData.layers;
const PARSED: Record<string, Set<string>> = Object.fromEntries(
	Object.keys(CHARTS).map((key) => [key, parseRange(CHARTS[key].text)])
);

/** The hands of a chart, by its key. */
export const chartHands = (key: string): ReadonlySet<string> => PARSED[key];

/** Seat by number of players still to act behind it; seats not listed here are computed. */
const CHART_BY_BEHIND: Record<number, string> = Object.fromEntries(
	Object.entries(chartData.chart_by_behind).map(([k, v]) => [Number(k), v as string])
);
/** Open shares for seats without a usable list (see the module header). */
const COMPUTED_PCT: Record<number, number> = Object.fromEntries(
	Object.entries(chartData.computed_pct).map(([k, v]) => [Number(k), v as number])
);

// -- computed: strength order, percentiles, adjustments ----------------------

/** Chen formula: a quick rating of a starting hand (AA = 20, 72o = -1). */
export function chenValue(hi: number, lo: number, suited: boolean): number {
	const points: Record<number, number> = { 14: 10, 13: 8, 12: 7, 11: 6 };
	let score = points[hi] ?? hi / 2;
	if (hi === lo) return Math.trunc(Math.max(5, score * 2));
	if (suited) score += 2;
	const gap = hi - lo - 1;
	score -= [0, 1, 2, 4, 5][Math.min(gap, 4)];
	if (gap <= 1 && hi < 12) score += 1;
	return Math.ceil(score); // round half points up
}

/** Equity against a random hand per class (empty if the tables file is missing). */
export const EQUITY_VS_RANDOM: Readonly<Record<string, number>> =
	(tables as { equity_vs_random?: Record<string, number> }).equity_vs_random ?? {};

function sortKey(name: string): number[] {
	const hi = rankOf(name[0]);
	const lo = rankOf(name[1]);
	const suited = name.length === 3 && name[2] === 's';
	let layer = LAYERS.findIndex((key) => PARSED[key].has(name));
	if (layer < 0) layer = LAYERS.length;
	const strength =
		name in EQUITY_VS_RANDOM
			? -Math.round(EQUITY_VS_RANDOM[name] * 100_000)
			: -chenValue(hi, lo, suited);
	return [layer, strength, -hi, -lo, suited ? 0 : 1];
}

function compareKeys(a: number[], b: number[]): number {
	for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] - b[i];
	return 0;
}

/** All 169 hands, strongest first. */
export const ORDER: readonly string[] = [...ALL_HANDS]
	.map((name) => ({ name, key: sortKey(name) }))
	.sort((x, y) => compareKeys(x.key, y.key))
	.map((x) => x.name);

const POSITION: Map<string, number> = new Map(ORDER.map((name, i) => [name, i]));

const PERCENTILE: Map<string, number> = (() => {
	const out = new Map<string, number>();
	let running = 0;
	for (const name of ORDER) {
		running += combos(name);
		out.set(name, (100 * running) / TOTAL_COMBOS);
	}
	return out;
})();

/** Top-X% rank of a hand: AA is ~0.5, the weakest hands approach 100. */
export const percentile = (name: string): number => PERCENTILE.get(name)!;

/** The strongest hands making up about `target`% of all combos. */
export const topPercent = (target: number): Set<string> => adjust(new Set(), target);

/** Grow (strongest missing first) or shrink (weakest first) a range to about `target`%. */
export function adjust(hands: Iterable<string>, target: number): Set<string> {
	const current = new Set(hands);
	let pct = percent(current);
	if (pct > target) {
		const weakestFirst = [...current].sort((a, b) => POSITION.get(b)! - POSITION.get(a)!);
		for (const name of weakestFirst) {
			const worth = (100 * combos(name)) / TOTAL_COMBOS;
			if (Math.abs(pct - worth - target) >= Math.abs(pct - target)) break;
			current.delete(name);
			pct -= worth;
		}
	} else {
		for (const name of ORDER) {
			if (current.has(name)) continue;
			const worth = (100 * combos(name)) / TOTAL_COMBOS;
			if (Math.abs(pct + worth - target) >= Math.abs(pct - target)) break;
			current.add(name);
			pct += worth;
		}
	}
	return current;
}

// -- seats -------------------------------------------------------------------

// Names for the seats before the button, first to act first.
const EARLY_NAMES: Record<number, readonly string[]> = {
	0: [],
	1: ['CO'],
	2: ['HJ', 'CO'],
	3: ['UTG', 'HJ', 'CO'],
	4: ['UTG', 'LJ', 'HJ', 'CO'],
	5: ['UTG', 'UTG+1', 'LJ', 'HJ', 'CO'],
	6: ['UTG', 'UTG+1', 'UTG+2', 'LJ', 'HJ', 'CO']
};

/** Seat names in preflop acting order: UTG … CO, BTN, SB, BB. Heads-up: SB/BTN, BB. */
export function seatNames(players: number): string[] {
	if (players < 2 || players > 9) throw new Error('2 to 9 players');
	if (players === 2) return ['SB/BTN', 'BB'];
	return [...EARLY_NAMES[players - 3], 'BTN', 'SB', 'BB'];
}

export interface OpenRange {
	hands: ReadonlySet<string>;
	pct: number;
	/** table = a chart copied from a source; computed = built from the strength order;
	 *  fallback = nothing found for this seat, widest sourced chart used; none = never opens. */
	source: 'table' | 'computed' | 'fallback' | 'none';
	label: string;
	url: string;
}

/** Raise-first-in range for the seat at `index` in preflop acting order (0 = first to act). */
export function seatRange(players: number, index: number): OpenRange {
	if (index < 0 || index >= players || players < 2 || players > 9) {
		throw new Error('seat out of range');
	}
	const behind = players - 1 - index;
	if (behind === 0) {
		return {
			hands: new Set(),
			pct: 0,
			source: 'none',
			label: 'Big blind: never opens (checks or raises limpers)',
			url: ''
		};
	}
	const fromChart = (key: string, source: 'table' | 'fallback', label: string): OpenRange => ({
		hands: PARSED[key],
		pct: percent(PARSED[key]),
		source,
		label,
		url: CHARTS[key].url
	});
	if (behind === 1) {
		if (players === 2) {
			return fromChart(
				'PSBTN',
				'fallback',
				'Heads-up: no source found; widest chart used - real heads-up opens are wider'
			);
		}
		return fromChart('SB6', 'table', CHARTS.SB6.label);
	}
	const key = CHART_BY_BEHIND[behind];
	if (key) return fromChart(key, 'table', CHARTS[key].label);
	const hands = topPercent(COMPUTED_PCT[behind]);
	return {
		hands,
		pct: percent(hands),
		source: 'computed',
		label: `Top ${COMPUTED_PCT[behind]}% of the strength order`,
		url: ''
	};
}

/** (seat name, players behind, range) for every seat at a table of `players`. */
export function chartFor(players: number): { name: string; behind: number; range: OpenRange }[] {
	return seatNames(players).map((name, i) => ({
		name,
		behind: players - 1 - i,
		range: seatRange(players, i)
	}));
}

// -- heads-up push/fold (computed offline) -----------------------------------

export interface PushFold {
	stack_bb: number;
	/** Small blind (button) shoves these hands, folds the rest. */
	push: ReadonlySet<string>;
	/** Big blind calls a shove with these hands, folds the rest. */
	call: ReadonlySet<string>;
	push_pct: number;
	call_pct: number;
	/** What a best response still gains against the pure ranges, in big blinds per hand. */
	exploitability_bb: number;
}

export const MIN_PUSHFOLD_BB = 2;
export const MAX_PUSHFOLD_BB = 20;

interface PushFoldRow {
	push: string[];
	call: string[];
	push_pct: number;
	call_pct: number;
	exploitability_bb: number;
}

/** Round half to even, like Python's round(): 2.5 → 2, 3.5 → 4. */
export function roundHalfEven(x: number): number {
	const floor = Math.floor(x);
	const diff = x - floor;
	if (diff < 0.5) return floor;
	if (diff > 0.5) return floor + 1;
	return floor % 2 === 0 ? floor : floor + 1;
}

/** Heads-up push/fold ranges for an effective stack (rounded, clamped to 2–20bb), if computed. */
export function pushfold(stackBb: number): PushFold | null {
	const rows = (tables as { pushfold?: Record<string, PushFoldRow> }).pushfold;
	if (!rows) return null;
	const depth = Math.min(MAX_PUSHFOLD_BB, Math.max(MIN_PUSHFOLD_BB, roundHalfEven(stackBb)));
	const row = rows[String(depth)];
	return {
		stack_bb: depth,
		push: new Set(row.push),
		call: new Set(row.call),
		push_pct: row.push_pct,
		call_pct: row.call_pct,
		exploitability_bb: row.exploitability_bb
	};
}

export const samplesPerMatchup = (): number =>
	(tables as { samples_per_matchup?: number }).samples_per_matchup ?? 0;

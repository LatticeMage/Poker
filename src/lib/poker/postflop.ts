// Approximate-GTO postflop play for heads-up pots: single-raised, 3-bet and 4-bet pots.
//
// Not a solver. The idea is the standard simplification of one:
//
// 1. Both players' preflop ranges come from the charts (ranges.ts / spots.ts).
// 2. On the actual board (flop, turn or river), Monte Carlo gives the equity of your hand and of
//    sampled hands from your whole range against the opponent's range (to the river: the unseen
//    cards are drawn at random; on the river there is nothing left to draw).
//    That yields your hand's rank inside your range and the range advantage (average equity above
//    or below 50%).
// 3. Betting: the more range advantage, the more you bet. The flop's group (flopGroups.ts: 383 groups by
//    suits, high cards, connectedness, which high cards, low cards middle or small) is the base of the
//    tree: it gives the texture (dry, paired, wet, monotone, from the counted wetness) and, on the flop
//    of a 6-max pot, the range advantage and the nut advantage of the matchup, computed exactly offline
//    (nvcc/flopstats.cu) instead of sampled. Static boards and clear advantages get the small bet; the
//    big bet needs a wet board and no fewer nut hands than the opponent. The strongest hands bet for value; bluffs are drawn from the no-pair hands with the
//    most equity, in the ratio that makes a caller indifferent: bluffs / bets = s / (1 + 2s) for a
//    bet of s times the pot.
// 4. Facing a bet: defend the minimum defence frequency, pot / (pot + bet), taking the strongest
//    hands of the range first. The very strongest raise; a few strong draws raise as bluffs.
//
// Approximations: equity is raw all-in equity (no position / realisation), the opponent is assumed to
// hold their whole preflop range - not narrowed by what they did on earlier streets - bet sizes are
// one of two fractions of the pot, the board texture is read from the flop, and sample sizes are
// small, so the numbers are noisy.

import type { GameState } from '$lib/types/game';
import type { Card } from './cards';
import { CATEGORY_UNIT, handScore } from './evaluate';
import { type FlopGroup, flopGroup, groupLevels, groupStats, matchupKey } from './flopGroups';
import { classCombos, seatRange } from './ranges';
import { type Rng, crc32, mulberry32, sample } from './rng';
import * as spots from './spots';

export type Weighted = Record<string, number>;

/**
 * The range statistics are computed once per board and pair of ranges and cached, so they can use a
 * big sample: N_SCENARIOS opponent hands with a turn and river, and N_REF hands from your range.
 * Your own hand is scored against the *same* scenarios, so its equity and the range's equities share
 * their sampling noise and compare fairly.
 */
export const N_SCENARIOS = 300;
export const N_REF = 60;

export const SMALL_BET = 0.33;
export const BIG_BET = 0.66;

export interface Profile {
	/** Equity of your hand against the opponent's range (your cards are dead for them). */
	eq: number;
	/** Sampled hands from your whole range, strongest first: [equity, has no made hand]. */
	refs: readonly (readonly [number, boolean])[];
	/** Average equity of your whole range against theirs; > 0.5 is a range advantage. It describes the
	 *  ranges on this board, so it does not depend on which hand you hold. */
	range_eq: number;
	/** Your hole cards make nothing better than the board already has. */
	nopair: boolean;
}

export interface Decision {
	/** check | bet | call | raise | fold */
	kind: 'check' | 'bet' | 'call' | 'raise' | 'fold';
	/** Bet size as a fraction of the pot (bet), or a multiple of the bet faced (raise). */
	size: number;
	/** Why, for tests and debugging: value | bluff | defend | draw-bluff | ... */
	reason: string;
}

/** Share of your range with strictly higher equity than your hand (0 = the top of the range). */
export function rankIn(prof: Profile): number {
	return prof.refs.filter(([e]) => e > prof.eq + 1e-9).length / prof.refs.length;
}

type Pool = { combos: [Card, Card][]; cumulative: number[] };

function byName([a]: [string, number], [b]: [string, number]): number {
	return a < b ? -1 : a > b ? 1 : 0;
}

/** A range's combos with running weights (sorted by name so the order never depends on the input). */
function poolOf(range: Weighted, blocked: ReadonlySet<Card>): Pool {
	const combos: [Card, Card][] = [];
	const cumulative: number[] = [];
	let total = 0;
	for (const [name, weight] of Object.entries(range).sort(byName)) {
		if (weight <= 0) continue;
		for (const combo of classCombos(name)) {
			if (blocked.has(combo[0]) || blocked.has(combo[1])) continue;
			total += weight;
			combos.push([combo[0], combo[1]]);
			cumulative.push(total);
		}
	}
	return { combos, cumulative };
}

/** First index whose running weight is at least x (Python's bisect_left). */
function bisectLeft(values: number[], x: number): number {
	let lo = 0;
	let hi = values.length;
	while (lo < hi) {
		const mid = (lo + hi) >> 1;
		if (values[mid] < x) lo = mid + 1;
		else hi = mid;
	}
	return lo;
}

function draw(pool: Pool, rng: Rng): [Card, Card] {
	const { combos, cumulative } = pool;
	const i = Math.min(
		bisectLeft(cumulative, rng() * cumulative[cumulative.length - 1]),
		combos.length - 1
	);
	return combos[i];
}

/**
 * n hands spread evenly over the range by weight: every kind of hand shows up in proportion. Much
 * less noisy than n independent draws, which matters because the range advantage and a hand's rank
 * are read off a small sample.
 */
function systematic(pool: Pool, n: number, rng: Rng): [Card, Card][] {
	const { combos, cumulative } = pool;
	const step = cumulative[cumulative.length - 1] / n;
	const start = rng() * step;
	return Array.from(
		{ length: n },
		(_, i) => combos[Math.min(bisectLeft(cumulative, start + i * step), combos.length - 1)]
	);
}

const categoryOf = (cards: readonly Card[]) => Math.floor(handScore(cards) / CATEGORY_UNIT);

/**
 * Do the hole cards make something better than the board alone (a pair, two pair, ...)? On a paired
 * board every hand holds that pair, so the board's own category does not count.
 */
export function hasMadeHand(hole: readonly Card[], board: readonly Card[]): boolean {
	return categoryOf([...hole, ...board]) > categoryOf(board);
}

/** A random two-card hand from a weighted range that avoids the blocked cards. */
export function sampleHand(
	range: Weighted,
	blocked: readonly Card[],
	rng: Rng
): [Card, Card] | null {
	const pool = poolOf(range, new Set(blocked));
	return pool.combos.length ? draw(pool, rng) : null;
}

/**
 * One sampled situation: the opponent's two cards, the cards still to come (turn and river on the
 * flop, the river on the turn, none on the river) and the opponent's hand score at showdown.
 */
interface Scenario {
	a: Card;
	b: Card;
	rest: Card[];
	theirs: number;
}

/** Opponent hands spread over their range, each with the remaining board cards drawn at random. */
function scenariosFor(opponents: Pool, board: readonly Card[], rng: Rng): Scenario[] {
	const deck = Array.from({ length: 52 }, (_, i) => i).filter((c) => !board.includes(c));
	const missing = 5 - board.length;
	return systematic(opponents, N_SCENARIOS, rng).map(([a, b]) => {
		const rest = missing
			? sample(
					deck.filter((c) => c !== a && c !== b),
					missing,
					rng
				)
			: [];
		return { a, b, rest, theirs: handScore([a, b, ...board, ...rest]) };
	});
}

/**
 * Share of the pot this hand wins on the scenarios. A scenario that uses one of the hand's own cards
 * is impossible (the card is already in your hand) and is skipped.
 */
function equityOn(
	hand: readonly Card[],
	scenarios: readonly Scenario[],
	board: readonly Card[]
): number | null {
	const [h0, h1] = hand;
	let won = 0;
	let seen = 0;
	for (const { a, b, rest, theirs } of scenarios) {
		if (h0 === a || h0 === b || h1 === a || h1 === b || rest.includes(h0) || rest.includes(h1)) {
			continue;
		}
		const mine = handScore([h0, h1, ...board, ...rest]);
		won += mine > theirs ? 1 : mine === theirs ? 0.5 : 0;
		seen++;
	}
	return seen ? won / seen : null;
}

interface RangeStats {
	refs: [number, boolean][];
	rangeEq: number;
	scenarios: Scenario[];
}

/** A small least-recently-used cache. */
class Lru<V> {
	#map = new Map<string, V>();
	constructor(readonly max: number) {}
	get(key: string): V | undefined {
		const value = this.#map.get(key);
		if (value !== undefined) {
			this.#map.delete(key);
			this.#map.set(key, value); // most recent last
		}
		return value;
	}
	set(key: string, value: V) {
		this.#map.set(key, value);
		if (this.#map.size > this.max) this.#map.delete(this.#map.keys().next().value as string);
	}
}

const itemsOf = (range: Weighted): [string, number][] => Object.entries(range).sort(byName);
const keyOf = (board: readonly Card[], mine: [string, number][], theirs: [string, number][]) =>
	`${board.join(',')}|${mine.map(([n, w]) => `${n}:${w}`).join(',')}|${theirs.map(([n, w]) => `${n}:${w}`).join(',')}`;

const statsCache = new Lru<RangeStats | null>(4096);

/**
 * Sampled hands of your range with their equity against the opponent's range, the average, and the
 * scenarios they were scored on. Depends only on the board and the two ranges, so it is cached; the
 * sample is seeded from them, which also makes it reproducible.
 */
function rangeStats(board: readonly Card[], mine: Weighted, theirs: Weighted): RangeStats | null {
	const mineItems = itemsOf(mine);
	const theirItems = itemsOf(theirs);
	const key = keyOf(board, mineItems, theirItems);
	const hit = statsCache.get(key);
	if (hit !== undefined) return hit;

	const blocked = new Set(board);
	const references = poolOf(mine, blocked);
	const opponents = poolOf(theirs, blocked);
	let result: RangeStats | null = null;
	if (references.combos.length && opponents.combos.length) {
		const rng = mulberry32(crc32(key));
		const refs = systematic(references, N_REF, rng);
		const scenarios = scenariosFor(opponents, board, rng);
		const valid: [number, boolean][] = [];
		for (const r of refs) {
			const e = equityOn(r, scenarios, board);
			if (e !== null) valid.push([e, !hasMadeHand(r, board)]);
		}
		valid.sort((x, y) => y[0] - x[0]);
		if (valid.length) {
			result = {
				refs: valid,
				rangeEq: valid.reduce((s, [e]) => s + e, 0) / valid.length,
				scenarios
			};
		}
	}
	statsCache.set(key, result);
	return result;
}

/** Equity of your hand, and of hands sampled from your range, against the opponent's range. */
export function profile(
	board: readonly Card[],
	hole: readonly Card[],
	mine: Weighted,
	theirs: Weighted
): Profile | null {
	const stats = rangeStats(board, mine, theirs);
	if (!stats) return null;
	const eq = equityOn(hole, stats.scenarios, board);
	if (eq === null) return null;
	return { eq, refs: stats.refs, range_eq: stats.rangeEq, nopair: !hasMadeHand(hole, board) };
}

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

export type Texture = 'monotone' | 'paired' | 'wet' | 'dry';

/** A flop group is wet from here: the counted share of hands that make or draw a flush or straight. */
export const WET_FROM = 0.25;

/**
 * Flop category, read from the flop's group (flopGroups.ts), so the turn and river keep the texture
 * the flop had. Monotone and paired are what the group name says; wet is counted, not guessed from the
 * gaps: the groups fall into two clusters (at most 0.162 and at least 0.30), so the cut is clean.
 */
export function boardTexture(board: readonly Card[]): Texture {
	const group = flopGroup(board);
	const [suits] = groupLevels(group);
	if (suits === '單色') return 'monotone';
	if (/對|三條/.test(group.name)) return 'paired';
	return group.wetness >= WET_FROM ? 'wet' : 'dry';
}

/**
 * What the flop's group says about a matchup, from the side of the player to act: the leaf of the tree.
 * Exact offline numbers (every flop of the group, every turn and river), so nothing here is sampled.
 */
export interface FlopRead {
	group: FlopGroup;
	/** Your range's equity on flops of this group, and the lowest and highest among its flops. */
	range_eq: number;
	range_eq_min: number;
	range_eq_max: number;
	/** Share of your range and of theirs with at least 80% equity: the nut hands. */
	nut_mine: number;
	nut_theirs: number;
}

/**
 * The read of a flop for a matchup (see matchupKey), or null when there is none: off the flop (the
 * turn and river change the ranges' standing, the flop's group no longer describes it) or for a
 * matchup the table does not have.
 */
export function flopRead(
	key: string | null,
	board: readonly Card[],
	aggressor: boolean
): FlopRead | null {
	if (!key || board.length !== 3) return null;
	const group = flopGroup(board);
	const stats = groupStats(key, group);
	if (!stats) return null;
	return aggressor
		? {
				group,
				range_eq: stats.eq,
				range_eq_min: stats.eq_min,
				range_eq_max: stats.eq_max,
				nut_mine: stats.nut_aggressor,
				nut_theirs: stats.nut_caller
			}
		: {
				group,
				range_eq: 1 - stats.eq,
				range_eq_min: 1 - stats.eq_max,
				range_eq_max: 1 - stats.eq_min,
				nut_mine: stats.nut_caller,
				nut_theirs: stats.nut_aggressor
			};
}

/** How much of the range bets on each kind of flop, relative to a dry one. */
const TEXTURE_FREQUENCY: Record<Texture, number> = {
	dry: 1,
	paired: 1.05,
	wet: 0.85,
	monotone: 0.7
};

/** A nut edge (your nut share minus theirs) from here on counts as holding more of the nuts. */
export const NUT_EDGE = 0.03;
/** How much one point of nut edge moves the betting frequency (a point of range equity moves it 4.5). */
const NUT_WEIGHT = 1;

/**
 * Small bets on static boards or with a clear range advantage, bigger ones on wet boards. With the
 * nut edge known (the flop read), a big bet also needs the nuts on your side: without at least as many
 * nut hands as the opponent the bet stays small, and on a static board where your range is behind, a
 * big bet is only for a real nut edge.
 *
 * A monotone board always takes the small bet. Measured over the 15 single-raised matchups, the two
 * ranges hold the same share of nut hands there (10.8% against 10.9%), so nobody has the nut edge a big
 * bet stands on, and "at least as many nut hands" was a coin flip that sent half of the monotone flops
 * to the big bet.
 */
export function betSize(rangeEq: number, texture: Texture = 'dry', nutEdge?: number): number {
	if (texture === 'monotone') return SMALL_BET;
	const advantage = rangeEq - 0.5;
	const wet = texture === 'wet';
	if (advantage >= 0.06 || (!wet && advantage >= 0)) return SMALL_BET;
	if (nutEdge === undefined) return BIG_BET;
	return nutEdge >= (wet ? 0 : NUT_EDGE) ? BIG_BET : SMALL_BET;
}

/**
 * How much of the range bets: more with a range advantage and with a nut edge, less for the player
 * who checked to.
 */
export function betFrequency(
	rangeEq: number,
	aggressor: boolean,
	texture: Texture = 'dry',
	nutEdge = 0
): number {
	const freq =
		clamp(0.45 + 4.5 * (rangeEq - 0.5) + NUT_WEIGHT * nutEdge, 0.2, 0.9) *
		TEXTURE_FREQUENCY[texture];
	return aggressor ? freq : freq * 0.75;
}

/** Bluffs per bet that make a caller indifferent: s / (1 + 2s). */
export const bluffShare = (size: number): number => size / (1 + 2 * size);

/** Minimum defence frequency facing a bet: the share of the range that must continue. */
export const defendFrequency = (pot: number, toCall: number): number => (pot - toCall) / pot;

/** The strategy numbers for a spot, before any hand is looked at. */
export interface Plan {
	/** Bet size as a fraction of the pot, and the share of the range that bets. */
	size: number;
	frequency: number;
	/** The strongest share of the range that bets for value, and the share that bluffs. */
	value_cut: number;
	bluff_quota: number;
	/** Share of the range that continues facing a bet (0 when nobody has bet). */
	defend: number;
	bluff_factor: number;
}

export interface PlanInput {
	texture: Texture;
	/** The flop's group read for this matchup; without it the sampled range equity is used. */
	flop?: FlopRead | null;
	pot: number;
	to_call: number;
	aggressor: boolean;
	facing_raise: boolean;
	persona_aggro?: number;
	persona_bluff?: number;
	persona_tight?: number;
}

/** Bet size and frequency, the value / bluff split, and the defence frequency for this spot. */
export function makePlan(prof: Profile, input: PlanInput): Plan {
	const {
		aggro = 0.5,
		bluff = 0.25,
		tight = 0.5
	} = {
		aggro: input.persona_aggro,
		bluff: input.persona_bluff,
		tight: input.persona_tight
	};
	const rangeEq = input.flop?.range_eq ?? prof.range_eq;
	const nutEdge = input.flop ? input.flop.nut_mine - input.flop.nut_theirs : undefined;
	const size = betSize(rangeEq, input.texture, nutEdge);
	const frequency = clamp(
		betFrequency(rangeEq, input.aggressor, input.texture, nutEdge) * (0.8 + 0.4 * aggro),
		0.05,
		0.95
	);
	const bluffFactor = 1 + (bluff - 0.25) * 1.5;
	const valueShare = 1 - bluffShare(size);
	let defend = 0;
	if (input.to_call > 0) {
		const width = 1 + (0.5 - tight) * 0.6;
		const realisation = input.facing_raise ? 0.75 : 0.92;
		defend = clamp(defendFrequency(input.pot, input.to_call) * realisation * width, 0.08, 0.95);
	}
	return {
		size,
		frequency,
		value_cut: frequency * valueShare,
		bluff_quota: frequency * (1 - valueShare) * bluffFactor,
		defend,
		bluff_factor: bluffFactor
	};
}

/**
 * Share of the range that has no made hand, sits outside the value region and has more equity than
 * this hand: how deep in the bluff queue it is.
 */
export function bluffsAhead(prof: Profile, valueCut: number): number {
	const n = prof.refs.length;
	const firstOutside = Math.trunc(valueCut * n + 0.5);
	let count = 0;
	prof.refs.forEach(([e, none], i) => {
		if (none && i >= firstOutside && e > prof.eq + 1e-9) count++;
	});
	return count / n;
}

export interface DecideInput extends PlanInput {
	can_raise: boolean;
	persona_aggro: number;
	persona_bluff: number;
	persona_tight: number;
}

/** The decision on any street for a hand with this profile. `to_call == 0` means nobody has bet. */
export function decide(prof: Profile, input: DecideInput, rng: Rng): Decision {
	const jitter = (rng() - 0.5) * 0.06; // keeps hands at a threshold mixed
	const rank = rankIn(prof) + jitter;
	const plan = makePlan(prof, input);

	if (input.to_call <= 0) {
		if (rank <= plan.value_cut) return { kind: 'bet', size: plan.size, reason: 'value' };
		// Bluffs are the best no-made-hand hands outside the value region (draws and overcards).
		if (prof.nopair && bluffsAhead(prof, plan.value_cut) + jitter <= plan.bluff_quota) {
			return { kind: 'bet', size: plan.size, reason: 'bluff' };
		}
		return { kind: 'check', size: 0, reason: 'check' };
	}

	if (rank > plan.defend) return { kind: 'fold', size: 0, reason: 'below the defence frequency' };
	if (input.can_raise && rank <= 0.1) return { kind: 'raise', size: 2.8, reason: 'value' };
	if (input.can_raise && prof.nopair && rng() < 0.25 * plan.bluff_factor) {
		const better =
			prof.refs.filter(([e, none]) => none && e > prof.eq + 1e-9).length / prof.refs.length;
		if (better <= 0.06) return { kind: 'raise', size: 2.8, reason: 'draw-bluff' };
	}
	return { kind: 'call', size: 0, reason: 'defend' };
}

/**
 * (aggressor's range, caller's range) of a heads-up pot from the preflop charts.
 *
 * kind "srp": a single-raised pot, `other` called the opener's raise (`openerHands` is the opener's
 * opening range). kind "3bet": `other` 3-bet and the opener called. kind "4bet": the opener 4-bet the
 * 3-bet from `other`, who continued (the hands the chart 3-bets for value).
 */
export function potRanges(
	kind: 'srp' | '3bet' | '4bet',
	openerClass: string,
	otherClass: string,
	openerHands: Iterable<string>
): [Weighted, Weighted] | null {
	let aggressor: Weighted;
	let caller: Weighted;
	const merge = (a: Weighted, b: Weighted): Weighted => {
		const out: Weighted = {};
		for (const h of new Set([...Object.keys(a), ...Object.keys(b)]))
			out[h] = Math.min(1, (a[h] ?? 0) + (b[h] ?? 0));
		return out;
	};
	if (kind === 'srp') {
		const spot = spots.vsOpen(otherClass, openerClass);
		const hands = [...openerHands];
		if (!spot || hands.length === 0) return null;
		aggressor = Object.fromEntries(hands.map((h) => [h, 1]));
		caller = { ...spot.call };
	} else if (kind === '3bet') {
		const raised = spots.vsOpen(otherClass, openerClass);
		const defended = spots.vs3bet(openerClass, otherClass);
		if (!raised || !defended) return null;
		aggressor = merge(raised.value3, raised.bluff3);
		caller = { ...defended.call };
	} else {
		const raised = spots.vsOpen(otherClass, openerClass);
		const faced = spots.vs3bet(openerClass, otherClass);
		if (!raised || !faced) return null;
		aggressor = merge(faced.value4, faced.bluff4);
		caller = { ...raised.value3 };
	}
	if (!Object.keys(aggressor).length || !Object.keys(caller).length) return null;
	return [aggressor, caller];
}

/** Combos a weighted range holds on this board (combos that use a board card are impossible). */
export function rangeCombos(range: Weighted, board: readonly Card[]): number {
	let total = 0;
	for (const [name, weight] of Object.entries(range)) {
		for (const combo of classCombos(name)) {
			if (!board.includes(combo[0]) && !board.includes(combo[1])) total += weight;
		}
	}
	return total;
}

const classCache = new Lru<Record<string, number> | null>(512);

/** Equity of every hand class of your range against the opponent's range on this board. */
export function classEquities(
	board: readonly Card[],
	mine: Weighted,
	theirs: Weighted
): Record<string, number> | null {
	const mineItems = itemsOf(mine);
	const key = keyOf(board, mineItems, itemsOf(theirs));
	const hit = classCache.get(key);
	if (hit !== undefined) return hit;
	const stats = rangeStats(board, mine, theirs);
	let out: Record<string, number> | null = null;
	if (stats) {
		out = {};
		for (const [name, weight] of mineItems) {
			if (weight <= 0) continue;
			const live = classCombos(name).filter((c) => !board.includes(c[0]) && !board.includes(c[1]));
			if (!live.length) continue;
			const step = Math.max(1, Math.floor(live.length / 3)); // up to three combos spread across the suits
			const values: number[] = [];
			for (let i = 0; i < live.length && values.length < 3; i += step) {
				const e = equityOn(live[i], stats.scenarios, board);
				if (e !== null) values.push(e);
			}
			if (values.length) out[name] = values.reduce((s, v) => s + v, 0) / values.length;
		}
	}
	classCache.set(key, out);
	return out;
}

/**
 * What the model does with every hand class of your range on this board: the share of its combos
 * that [bet / raise for value, bluff, call] (the rest checks or folds). Every live combo of the
 * class is judged on its own, then averaged, so suits that matter (a flush draw against none) show
 * as a part of the cell.
 */
export function classActions(
	board: readonly Card[],
	mine: Weighted,
	theirs: Weighted,
	pick: (prof: Profile) => [value: number, bluff: number, call: number]
): Record<string, [number, number, number]> | null {
	const stats = rangeStats(board, mine, theirs);
	if (!stats) return null;
	const out: Record<string, [number, number, number]> = {};
	for (const [name, weight] of itemsOf(mine)) {
		if (weight <= 0) continue;
		const total: [number, number, number] = [0, 0, 0];
		let n = 0;
		for (const combo of classCombos(name)) {
			if (board.includes(combo[0]) || board.includes(combo[1])) continue;
			const prof = profile(board, combo, mine, theirs);
			if (!prof) continue;
			const shares = pick(prof);
			for (let i = 0; i < 3; i++) total[i] += shares[i];
			n++;
		}
		if (n) out[name] = [total[0] / n, total[1] / n, total[2] / n];
	}
	return out;
}

/** The chance that `rank + jitter <= limit`, the jitter being uniform on ±JITTER (as in `decide`). */
export const JITTER = 0.03;
export const below = (rank: number, limit: number): number =>
	clamp((limit - rank + JITTER) / (2 * JITTER), 0, 1);

/** One hand class of a range on the equity chart. */
export interface ClassPoint {
	hand: string;
	equity: number;
	/** Share of the range's live combos this class makes up (the shares sum to 1). */
	weight: number;
	/** Where the class sits in the range: 0 weakest .. 1 strongest (the middle of its slice). */
	percentile: number;
}

/** Every hand class of your range with its equity, weakest first, for labelling the curve. */
export function classPoints(
	board: readonly Card[],
	mine: Weighted,
	theirs: Weighted
): ClassPoint[] | null {
	const eqs = classEquities(board, mine, theirs);
	if (!eqs) return null;
	const rows = Object.entries(eqs)
		.map(([hand, equity]) => {
			const live = classCombos(hand).filter(
				(c) => !board.includes(c[0]) && !board.includes(c[1])
			).length;
			return { hand, equity, mass: live * (mine[hand] ?? 0) };
		})
		.filter((r) => r.mass > 0)
		.sort((a, b) => a.equity - b.equity || (a.hand < b.hand ? -1 : 1));
	const total = rows.reduce((s, r) => s + r.mass, 0);
	if (total <= 0) return null;
	let below = 0;
	return rows.map((r) => {
		const point = {
			hand: r.hand,
			equity: r.equity,
			weight: r.mass / total,
			percentile: (below + r.mass / 2) / total
		};
		below += r.mass;
		return point;
	});
}

/**
 * Equity of hands spread evenly over your range against theirs, weakest first. Plotted against the
 * percentile of the range (0 = weakest, 1 = strongest) this is the usual equity distribution chart;
 * drawn for both players it shows who has the stronger top and bottom.
 */
export function equityCurve(
	board: readonly Card[],
	mine: Weighted,
	theirs: Weighted
): number[] | null {
	const stats = rangeStats(board, mine, theirs);
	return stats ? stats.refs.map(([e]) => e).sort((a, b) => a - b) : null;
}

/**
 * (your range, their range, you were the preflop aggressor, the matchup key of the flop table) for a
 * heads-up postflop spot, or null. The key is null away from a 6-player table: the table was computed
 * for the 6-max charts, and other table sizes open other ranges.
 * Covers single-raised pots (the opener vs the caller), 3-bet pots (the 3-bettor vs the opener who
 * called) and 4-bet pots (the opener vs the 3-bettor who continued). Anything else - limped pots, a
 * cold caller, 5-bet pots - has no range model here.
 */
export function headsUpRanges(
	g: GameState,
	seat: number,
	order: number[]
): [Weighted, Weighted, boolean, string | null] | null {
	const contenders = g.players.filter((p) => p.in_hand && !p.folded).map((p) => p.seat);
	if (contenders.length !== 2 || !contenders.includes(seat)) return null;
	const n = order.length;
	const behind = (s: number) => n - 1 - order.indexOf(s);
	const raisers = g.raisers;
	let built: [Weighted, Weighted] | null;
	let aggressor: number;
	let kind: 'srp' | '3bet' | '4bet';
	let seats: [opener: number, other: number];
	if (raisers.length === 1) {
		const opener = raisers[0];
		const callers = contenders.filter((c) => c !== opener);
		if (!contenders.includes(opener) || callers.length !== 1) return null;
		built = potRanges(
			'srp',
			spots.positionClass(behind(opener), true),
			spots.positionClass(behind(callers[0])),
			seatRange(n, order.indexOf(opener)).hands
		);
		aggressor = opener;
		kind = 'srp';
		seats = [opener, callers[0]];
	} else if (raisers.length === 2) {
		const [opener, bettor] = raisers;
		if (contenders.length !== 2 || !contenders.includes(opener) || !contenders.includes(bettor))
			return null;
		built = potRanges(
			'3bet',
			spots.positionClass(behind(opener), true),
			spots.positionClass(behind(bettor)),
			[]
		);
		aggressor = bettor;
		kind = '3bet';
		seats = [opener, bettor];
	} else if (raisers.length === 3 && raisers[0] === raisers[2]) {
		const [opener, bettor] = raisers;
		if (!contenders.includes(opener) || !contenders.includes(bettor)) return null;
		built = potRanges(
			'4bet',
			spots.positionClass(behind(opener), true),
			spots.positionClass(behind(bettor)),
			[]
		);
		aggressor = opener;
		kind = '4bet';
		seats = [opener, bettor];
	} else {
		return null;
	}
	if (!built) return null;
	const [aggressorRange, callerRange] = built;
	const key =
		n === 6
			? matchupKey(
					kind,
					spots.positionClass(behind(seats[0]), true),
					spots.positionClass(behind(seats[1]))
				)
			: null;
	return seat === aggressor
		? [aggressorRange, callerRange, true, key]
		: [callerRange, aggressorRange, false, key];
}

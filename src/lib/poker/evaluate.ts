// Poker hand evaluation for 5–7 cards (the best five are chosen automatically).

import { type Card, rankName, rankPlural } from './cards';

export const HIGH_CARD = 0;
export const PAIR = 1;
export const TWO_PAIR = 2;
export const TRIPS = 3;
export const STRAIGHT = 4;
export const FLUSH = 5;
export const FULL_HOUSE = 6;
export const QUADS = 7;
export const STRAIGHT_FLUSH = 8;

export interface HandValue {
	/** Comparable strength: a higher score beats a lower one, equal scores tie. */
	score: number;
	category: number;
	/** Human readable, e.g. "Two Pair, Kings and Sixes". */
	name: string;
	/** The five cards (fewer if fewer were supplied) that make the hand. */
	cards: Card[];
}

interface Ranked {
	category: number;
	/** Ranks that break ties within the category, most significant first. */
	tiebreak: number[];
	flushSuit: number;
}

/** Highest card of a straight in a rank bitmask (bit r set = rank r present), or 0. */
function straightHigh(mask: number): number {
	// The ace also plays low: mirror bit 14 onto bit 1.
	const m = mask | (((mask >> 14) & 1) << 1);
	for (let high = 14; high >= 5; high--) {
		const need = 0b11111 << (high - 4);
		if ((m & need) === need) return high;
	}
	return 0;
}

function topRanks(mask: number, n: number): number[] {
	const out: number[] = [];
	for (let r = 14; r >= 2 && out.length < n; r--) if (mask & (1 << r)) out.push(r);
	return out;
}

function rank(cards: ArrayLike<Card>): Ranked {
	const counts = new Array<number>(15).fill(0);
	const suitCounts = [0, 0, 0, 0];
	const suitMasks = [0, 0, 0, 0];
	let mask = 0;
	for (let i = 0; i < cards.length; i++) {
		const c = cards[i];
		const r = (c >> 2) + 2;
		const s = c & 3;
		counts[r]++;
		suitCounts[s]++;
		suitMasks[s] |= 1 << r;
		mask |= 1 << r;
	}

	let flushSuit = -1;
	for (let s = 0; s < 4; s++) {
		if (suitCounts[s] >= 5) {
			flushSuit = s;
			const high = straightHigh(suitMasks[s]);
			if (high) return { category: STRAIGHT_FLUSH, tiebreak: [high], flushSuit: s };
		}
	}

	let quads = 0;
	const trips: number[] = [];
	const pairs: number[] = [];
	const singles: number[] = [];
	for (let r = 14; r >= 2; r--) {
		const n = counts[r];
		if (n === 1) singles.push(r);
		else if (n === 2) pairs.push(r);
		else if (n === 3) trips.push(r);
		else if (n === 4 && !quads) quads = r;
	}

	if (quads) {
		return {
			category: QUADS,
			tiebreak: [quads, ...topRanks(mask & ~(1 << quads), 1)],
			flushSuit
		};
	}
	if (trips.length > 0 && (trips.length > 1 || pairs.length > 0)) {
		const pair = Math.max(trips[1] ?? 0, pairs[0] ?? 0);
		return { category: FULL_HOUSE, tiebreak: [trips[0], pair], flushSuit };
	}
	if (flushSuit >= 0) {
		return { category: FLUSH, tiebreak: topRanks(suitMasks[flushSuit], 5), flushSuit };
	}
	const high = straightHigh(mask);
	if (high) return { category: STRAIGHT, tiebreak: [high], flushSuit };
	if (trips.length > 0) {
		return { category: TRIPS, tiebreak: [trips[0], ...singles.slice(0, 2)], flushSuit };
	}
	if (pairs.length >= 2) {
		const kicker = Math.max(pairs[2] ?? 0, singles[0] ?? 0);
		return {
			category: TWO_PAIR,
			tiebreak: kicker ? [pairs[0], pairs[1], kicker] : [pairs[0], pairs[1]],
			flushSuit
		};
	}
	if (pairs.length === 1) {
		return { category: PAIR, tiebreak: [pairs[0], ...singles.slice(0, 3)], flushSuit };
	}
	return { category: HIGH_CARD, tiebreak: singles.slice(0, 5), flushSuit };
}

/** hand_score() packs the category above five tie-break ranks of base 15. */
export const CATEGORY_UNIT = 15 ** 5;

function pack(r: Ranked): number {
	let score = r.category;
	for (let i = 0; i < 5; i++) score = score * 15 + (r.tiebreak[i] ?? 0);
	return score;
}

/** Strength only — the hot path for equity simulation. */
export function handScore(cards: ArrayLike<Card>): number {
	return pack(rank(cards));
}

/** The hand category (0 high card .. 8 straight flush) of a score. */
export const categoryOf = (score: number): number => Math.floor(score / CATEGORY_UNIT);

function straightRanks(high: number): number[] {
	return [0, 1, 2, 3, 4].map((i) => (high - i === 1 ? 14 : high - i));
}

function pickCards(cards: readonly Card[], r: Ranked): Card[] {
	const pool = [...cards].sort((a, b) => (b >> 2) - (a >> 2));
	const used: Card[] = [];
	const take = (wanted: number, n: number, suit = -1) => {
		for (const c of pool) {
			if (n === 0) break;
			if ((c >> 2) + 2 !== wanted || used.includes(c)) continue;
			if (suit >= 0 && (c & 3) !== suit) continue;
			used.push(c);
			n--;
		}
	};
	const t = r.tiebreak;
	switch (r.category) {
		case STRAIGHT_FLUSH:
			straightRanks(t[0]).forEach((x) => take(x, 1, r.flushSuit));
			break;
		case QUADS:
			take(t[0], 4);
			t.slice(1).forEach((x) => take(x, 1));
			break;
		case FULL_HOUSE:
			take(t[0], 3);
			take(t[1], 2);
			break;
		case FLUSH:
			t.forEach((x) => take(x, 1, r.flushSuit));
			break;
		case STRAIGHT:
			straightRanks(t[0]).forEach((x) => take(x, 1));
			break;
		case TRIPS:
			take(t[0], 3);
			t.slice(1).forEach((x) => take(x, 1));
			break;
		case TWO_PAIR:
			take(t[0], 2);
			take(t[1], 2);
			t.slice(2).forEach((x) => take(x, 1));
			break;
		case PAIR:
			take(t[0], 2);
			t.slice(1).forEach((x) => take(x, 1));
			break;
		default:
			t.forEach((x) => take(x, 1));
	}
	return used;
}

function describe(category: number, t: number[]): string {
	switch (category) {
		case STRAIGHT_FLUSH:
			return t[0] === 14 ? 'Royal Flush' : `Straight Flush, ${rankName(t[0])} high`;
		case QUADS:
			return `Four of a Kind, ${rankPlural(t[0])}`;
		case FULL_HOUSE:
			return `Full House, ${rankPlural(t[0])} over ${rankPlural(t[1])}`;
		case FLUSH:
			return `Flush, ${rankName(t[0])} high`;
		case STRAIGHT:
			return `Straight, ${rankName(t[0])} high`;
		case TRIPS:
			return `Three of a Kind, ${rankPlural(t[0])}`;
		case TWO_PAIR:
			return `Two Pair, ${rankPlural(t[0])} and ${rankPlural(t[1])}`;
		case PAIR:
			return `Pair of ${rankPlural(t[0])}`;
		default:
			return `${rankName(t[0])} high`;
	}
}

/** Full evaluation: strength plus the name and the cards that make the hand. */
export function evaluate(cards: readonly Card[]): HandValue {
	const r = rank(cards);
	return {
		score: pack(r),
		category: r.category,
		name: describe(r.category, r.tiebreak),
		cards: pickCards(cards, r)
	};
}

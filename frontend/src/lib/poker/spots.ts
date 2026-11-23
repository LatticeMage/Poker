// Common preflop spots beyond opening: facing an open raise, and facing a 3-bet after opening.
//
// Hand lists copied from the PokerStrategy 6-max lessons (cash, ~100bb), one lesson per hero
// position (data/spots.json). Each action's range is a *weighted* hand list: the source splits a
// hand's combos between actions (e.g. AA: one combo calls, five 3-bet), so a hand maps to action
// frequencies, not a single action. Every list is checked against the combo count (#N) the page
// prints (see spots.test.ts).
//
//   facing an open raise (hero, opener):
//     call      cold-call
//     bluff3    3-bet, then fold to a 4-bet   (page label "3-Bet/Fold")
//     value3    3-bet, then continue          (page label "3-Bet/5-Bet", BB vs SB: "3-Bet/Call")
//   facing a 3-bet after opening (hero, 3-bettor):
//     call      call the 3-bet
//     bluff4    4-bet, then fold to a 5-bet   ("4-Bet/Fold")
//     value4    4-bet, then continue          ("4-Bet/Call")
//
// Computed: seats of other table sizes map to the nearest 6-max seat by players behind, and calling
// ranges shrink when the open is bigger than the chart's size (minimum defence frequency).

import spotData from './data/spots.json';
import { RANK_CHARS, combos, handName, roundHalfEven } from './ranges';

const URL = (lesson: number) => `https://www.pokerstrategy.com/strategy/bss/${lesson}/`;
const LESSON = spotData.lesson as Record<string, number>;

/** Open size (in big blinds) each position uses in the source charts. */
export const OPEN_SIZE = spotData.open_size as Record<string, number>;

export type Weights = Record<string, number>;

const rankOf = (char: string) => RANK_CHARS.indexOf(char) + 2;
const R = '[2-9TJQKA]';
const COMBO = new RegExp(`^(${R})([shdc])(${R})([shdc])$`);
const PAIR = new RegExp(`^(${R})\\1(\\+)?$`);
const PAIR_SPAN = new RegExp(`^(${R})\\1-(${R})\\2$`);
const HAND = new RegExp(`^(${R})(${R})([so]?)(\\+)?$`);
const SPAN = new RegExp(`^(${R})(${R})([so]?)-(${R})(${R})([so]?)$`);

/** Suitedness covered by a suffix: "s", "o", or none (both). */
const suffixes = (flag: string): boolean[] => (flag === '' ? [true, false] : [flag === 's']);

/**
 * Parse a hand list into {hand class: share of that class's combos in the list}. Understands
 * "QQ-88", "TT+", "AQs+", "A5s-A2s", "JTs-98s", "AK" (suited and offsuit), "KJo+", and single combos
 * such as "AsAc" or "AdKh".
 */
export function parseWeighted(text: string): Weights {
	const counts: Record<string, number> = {};
	const add = (name: string, n?: number) => {
		counts[name] = (counts[name] ?? 0) + (n ?? combos(name));
	};

	for (const raw of text.split(',')) {
		const token = raw.trim();
		if (!token) continue;
		let m: RegExpExecArray | null;
		if ((m = COMBO.exec(token))) {
			const r1 = rankOf(m[1]);
			const r2 = rankOf(m[3]);
			if (r1 === r2 && m[2] === m[4]) throw new Error(`Bad combo: ${token}`);
			add(handName(r1, r2, m[2] === m[4]), 1);
		} else if ((m = PAIR.exec(token))) {
			const low = rankOf(m[1]);
			for (let r = low; r <= (m[2] ? 14 : low); r++) add(handName(r, r, false));
		} else if ((m = PAIR_SPAN.exec(token))) {
			const a = rankOf(m[1]);
			const b = rankOf(m[2]);
			for (let r = Math.min(a, b); r <= Math.max(a, b); r++) add(handName(r, r, false));
		} else if ((m = HAND.exec(token))) {
			const hi = rankOf(m[1]);
			const lo = rankOf(m[2]);
			if (hi <= lo) throw new Error(`Bad token: ${token}`);
			const top = m[4] ? hi - 1 : lo;
			for (const suited of suffixes(m[3])) {
				for (let r = lo; r <= top; r++) add(handName(hi, r, suited));
			}
		} else if ((m = SPAN.exec(token))) {
			const [hi1, lo1, hi2, lo2] = [1, 2, 4, 5].map((i) => rankOf(m![i]));
			if (m[3] !== m[6]) throw new Error(`Bad span: ${token}`);
			for (const suited of suffixes(m[3])) {
				if (hi1 === hi2) {
					// A5s-A2s: same high card, the low card walks down
					for (let r = Math.min(lo1, lo2); r <= Math.max(lo1, lo2); r++)
						add(handName(hi1, r, suited));
				} else if (hi1 - lo1 === hi2 - lo2) {
					// JTs-98s: connectors walk down together
					for (let h = Math.min(hi1, hi2); h <= Math.max(hi1, hi2); h++) {
						add(handName(h, h - (hi1 - lo1), suited));
					}
				} else {
					throw new Error(`Bad span: ${token}`);
				}
			}
		} else {
			throw new Error(`Bad token: ${token}`);
		}
	}
	return Object.fromEntries(
		Object.entries(counts).map(([name, n]) => [name, Math.min(1, n / combos(name))])
	);
}

/** Number of two-card combos a weighted range holds. */
export function comboCount(weights: Weights): number {
	let total = 0;
	for (const [name, share] of Object.entries(weights)) total += share * combos(name);
	return roundHalfEven(total);
}

export interface VsOpen {
	call: Weights;
	bluff3: Weights;
	value3: Weights;
	url: string;
}

export interface Vs3Bet {
	call: Weights;
	bluff4: Weights;
	value4: Weights;
	url: string;
}

interface RawSpot {
	call: string;
	bluff: string;
	value: string;
	stated: [number, number, number];
}

/** Stated combo counts per spot (call, bluff, value), kept for the tests of the transcription. */
export const STATED_OPEN: Record<string, [number, number, number]> = {};
export const STATED_3BET: Record<string, [number, number, number]> = {};

const VS_OPEN = new Map<string, VsOpen>();
for (const [key, raw] of Object.entries(spotData.vs_open as unknown as Record<string, RawSpot>)) {
	STATED_OPEN[key] = raw.stated;
	VS_OPEN.set(key, {
		call: parseWeighted(raw.call),
		bluff3: parseWeighted(raw.bluff),
		value3: parseWeighted(raw.value),
		url: URL(LESSON[key.split('|')[0]])
	});
}
const VS_3BET = new Map<string, Vs3Bet>();
for (const [key, raw] of Object.entries(spotData.vs_3bet as unknown as Record<string, RawSpot>)) {
	STATED_3BET[key] = raw.stated;
	VS_3BET.set(key, {
		call: parseWeighted(raw.call),
		bluff4: parseWeighted(raw.bluff),
		value4: parseWeighted(raw.value),
		url: URL(LESSON[key.split('|')[0]])
	});
}

export const POSITIONS = ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB'] as const;
const BY_BEHIND = ['BB', 'SB', 'BTN', 'CO', 'MP', 'UTG'];

/**
 * Nearest 6-max seat for a seat with `behind` players still to act. Early seats of bigger tables are
 * all "UTG" as an opener; as a player facing a raise they are "MP" (there is a raiser in front of
 * them, so they cannot be the first seat).
 */
export function positionClass(behind: number, opener = false): string {
	if (behind >= 5) return opener ? 'UTG' : 'MP';
	return BY_BEHIND[Math.max(0, behind)];
}

/** Hero's strategy facing an open raise, or undefined if hero cannot act after that opener. */
export const vsOpen = (hero: string, opener: string): VsOpen | undefined =>
	VS_OPEN.get(`${hero}|${opener}`);

/** Strategy of the player who opened, facing a 3-bet from `threeBettor`. */
export const vs3bet = (hero: string, threeBettor: string): Vs3Bet | undefined =>
	VS_3BET.get(`${hero}|${threeBettor}`);

/** Minimum defence frequency against an open to `sizeBb` into the 1.5bb of blinds. */
export const mdf = (sizeBb: number): number => 1.5 / (1.5 + sizeBb);

/** How much to scale calling ranges when the open is bigger or smaller than the chart's size. */
export function sizeFactor(opener: string, sizeBb: number): number {
	const base = OPEN_SIZE[opener] ?? 2.5;
	return Math.max(0.4, Math.min(1.3, mdf(sizeBb) / mdf(base)));
}

/** Who can face whom, for the learning pages. */
export function spotPairs(): { vsOpen: [string, string][]; vs3bet: [string, string][] } {
	const pairs = (map: Map<string, unknown>) =>
		[...map.keys()].map((key) => key.split('|') as [string, string]);
	return { vsOpen: pairs(VS_OPEN), vs3bet: pairs(VS_3BET) };
}

// A range as one weight per two-card combo (1,326): how often the player holds exactly those cards.
// Suits matter after the flop (A♠K♠ on two spades is not A♥K♥), so the hand classes of the preflop
// charts are only how a range starts out and how it is drawn on a 13×13 grid.

import type { Card } from './cards';
import { ALL_HANDS, TOTAL_COMBOS, classCombos, handOf } from './ranges';

/** Plain numbers, so a range goes into JSON and comes back unchanged. */
export type Range = number[];

/** The combos in one fixed order: classes by name, each class's combos as classCombos lists them. */
export const COMBOS: readonly (readonly [Card, Card])[] = [...ALL_HANDS]
	.sort()
	.flatMap((name) => classCombos(name));
export const COMBO_CLASS: readonly string[] = COMBOS.map((c) => handOf(c));

const key = (a: Card, b: Card) => (a < b ? a * 52 + b : b * 52 + a);
const INDEX = new Map(COMBOS.map(([a, b], i) => [key(a, b), i]));
export const comboIndex = (a: Card, b: Card): number => INDEX.get(key(a, b))!;

export const isLive = (i: number, board: readonly Card[]): boolean =>
	!board.includes(COMBOS[i][0]) && !board.includes(COMBOS[i][1]);

/** A range given per hand class (as the charts are): every combo of a class gets its weight. */
export function fromClasses(classes: Readonly<Record<string, number>>): Range {
	return COMBO_CLASS.map((name) => classes[name] ?? 0);
}

/** The weighted number of combos the board still allows. */
export function comboCount(range: Range, board: readonly Card[]): number {
	let total = 0;
	for (let i = 0; i < range.length; i++) if (range[i] > 0 && isLive(i, board)) total += range[i];
	return total;
}

/**
 * Per hand class, the average over its combos the board allows of `value` (weights, action shares):
 * what a cell of the 13×13 grid shows. `weight` says how much each combo counts in the average.
 */
export function byClass(
	board: readonly Card[],
	value: (i: number) => number,
	weight: (i: number) => number = () => 1
): Record<string, number> {
	const sum: Record<string, number> = {};
	const total: Record<string, number> = {};
	for (let i = 0; i < TOTAL_COMBOS; i++) {
		const w = weight(i);
		if (w <= 0 || !isLive(i, board)) continue;
		const name = COMBO_CLASS[i];
		sum[name] = (sum[name] ?? 0) + w * value(i);
		total[name] = (total[name] ?? 0) + w;
	}
	for (const name of Object.keys(sum)) sum[name] /= total[name];
	return sum;
}

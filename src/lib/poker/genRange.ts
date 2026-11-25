// genRange: both ranges at any table status. A pure function of the status alone: its setup says where
// the ranges start, its history what was done since.
//
//   weight after an action = weight before × P(action | combo)      (the acting player only)
//
// The probabilities are genStrategy's, so a range is as good as the strategy that narrowed it. The
// product over the history is how often a combo plays the whole line.

import type { Card } from './cards';
import { type Strategy, genStrategy } from './genStrategy';
import * as postflop from './postflop';
import { type Range, fromClasses, isLive } from './range';
import { seatRange } from './ranges';
import {
	LineError,
	type LineStep,
	type Setup,
	type TableStatus,
	nextStatus,
	sideIndex,
	startStatus,
	toCallOf
} from './table';

export type Ranges = [opener: Range, other: Range];

/** An action the model never takes still leaves this much of a hand: the line happened, the model is not sure. */
export const LINE_FLOOR = 0.02;

const OPEN_INDEX: Record<string, number> = { UTG: 0, MP: 1, CO: 2, BTN: 3, SB: 4 };

/**
 * The ranges that come to the flop, from the preflop charts; with more than two players each is
 * tightened to its strongest part (postflop.potTightness).
 */
export function startRanges({ kind, opener, other, players }: Setup): Ranges {
	if (!(opener in OPEN_INDEX)) throw new LineError('That position cannot open');
	const built = postflop.potRanges(kind, opener, other, seatRange(6, OPEN_INDEX[opener]).hands);
	if (!built) throw new LineError('No chart for this matchup');
	// built = [aggressor, caller]; the opener is the aggressor except in a 3-bet pot.
	const [mine, theirs] = kind === '3bet' ? [built[1], built[0]] : built;
	const share = postflop.potTightness(players);
	return [fromClasses(postflop.tighten(mine, share)), fromClasses(postflop.tighten(theirs, share))];
}

/** Which of the strategy's options an action is, facing `toCall`. */
export function optionOf(step: LineStep, toCall: number): number {
	if (toCall <= 0) return step.kind === 'check' ? 0 : 1;
	return step.kind === 'fold' ? 0 : step.kind === 'call' ? 1 : 2;
}

/** The ranges after one step, from the status and the ranges before it. */
function after(table: TableStatus, ranges: Ranges, strategy: Strategy | null, step: LineStep) {
	if (step.kind === 'deal') {
		const board = [...table.board, ...step.cards];
		return [without(ranges[0], board), without(ranges[1], board)] as Ranges;
	}
	if (!strategy) return ranges;
	const me = sideIndex(step.by);
	const option = optionOf(step, toCallOf(table, step.by));
	const next = ranges[me].map((w, i) =>
		w > 0 ? w * Math.max(LINE_FLOOR, strategy.combos[i]?.[option] ?? 0) : 0
	);
	if (!next.some((w) => w > 0)) return ranges;
	return (me === 0 ? [next, ranges[1]] : [ranges[0], next]) as Ranges;
}

/** Answers already worked out, so walking a game does not start over at every status. */
const seen = new Map<string, Ranges>();
const LIMIT = 200;

/** Both ranges at this status. Throws LineError on a history that cannot be played. */
export function genRange(status: TableStatus): Ranges {
	const { kind, opener, other, players, history } = status;
	const key = JSON.stringify([kind, opener, other, players, history]);
	let ranges = seen.get(key);
	if (!ranges) {
		if (!history.length) {
			ranges = startRanges(status);
		} else {
			const before = history.slice(0, -1).reduce((t, s) => nextStatus(t, s), startStatus(status));
			const step = history[history.length - 1];
			nextStatus(before, step); // refuses a step that cannot be played
			const was = genRange(before);
			ranges = after(before, was, genStrategy(before, was), step);
		}
		seen.set(key, ranges);
		if (seen.size > LIMIT) seen.delete(seen.keys().next().value as string);
	}
	return [[...ranges[0]], [...ranges[1]]];
}

/** The range without the combos that use a board card. */
export function without(range: Range, board: readonly Card[]): Range {
	return range.map((w, i) => (isLive(i, board) ? w : 0));
}

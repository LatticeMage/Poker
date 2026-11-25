// The decision chain of a heads-up hand: what both players did, in order, from the flop to the end.
//
// A game is an array of table statuses (table.ts), plain JSON, one before any step and one after each.
// Everything else comes from two pure functions:
//
//   genRange(status)                    both ranges at that point (genRange.ts)
//   genStrategy(status, ranges, hole?)  what the player to act does with every combo (genStrategy.ts)
//
// Any status can be taken out and asked on its own. Equity at a stage is the equity of the ranges as
// they are then, which is why it cannot be computed without what came before. A bet's size does not
// change a range yet: a hand "bets" or not; the per-size tables exist offline and are the next
// refinement.

import type { Card } from './cards';
import { type Ranges, genRange, optionOf, without } from './genRange';
import { type Strategy, genStrategy } from './genStrategy';
import * as postflop from './postflop';
import { comboCount } from './range';
import {
	type LineAction,
	type LineStep,
	type PotKind,
	type Side,
	type TableStatus,
	nextStatus,
	sideIndex,
	startStatus,
	toCallOf
} from './table';

export { LINE_FLOOR } from './genRange';
export type { Strategy as LineStrategy } from './genStrategy';
export {
	LineError,
	type LineAction,
	type LineStep,
	type PotKind,
	type Side,
	type TableStatus,
	potOnFlop
} from './table';

/**
 * The steps of a hand from its board and its actions: the flop first, then the actions in order, and
 * the turn and river as soon as the betting before them is over. `used` is how many actions went in:
 * the rest wait for a card that is not on the board yet, or come after a fold.
 */
export function lineSteps(
	board: readonly Card[],
	actions: readonly LineAction[]
): { steps: LineStep[]; used: number } {
	const steps: LineStep[] = [];
	if (board.length < 3) return { steps, used: 0 };
	steps.push({ kind: 'deal', cards: board.slice(0, 3) });
	let dealt = 3;
	let acted = 0;
	let used = 0;
	let waiting = false;
	for (const action of actions) {
		if (waiting) break;
		steps.push(action);
		used++;
		acted++;
		if (action.kind === 'fold') break;
		if (action.kind === 'call' || (action.kind === 'check' && acted >= 2)) {
			if (dealt < board.length) {
				steps.push({ kind: 'deal', cards: [board[dealt++]] });
				acted = 0;
			} else {
				waiting = true;
			}
		}
	}
	return { steps, used };
}

export interface HandLine {
	/** How the hand got to the flop: the preflop part of the chain. */
	pot: PotKind;
	opener: string;
	other: string;
	/** Hole cards that are known, if any. */
	hole?: { opener?: Card[]; other?: Card[] };
	/**
	 * Players who came into the pot, two by default. With more, the hand is still followed as these two
	 * players, on ranges tightened for the number of players (postflop.potTightness), and without the
	 * flop tables, which are for two.
	 */
	players?: number;
	steps: LineStep[];
}

export interface LineState {
	/** How many steps have been played. */
	index: number;
	street: 'preflop' | 'flop' | 'turn' | 'river';
	board: Card[];
	/** The pot in big blinds, bets of this street included, and what each player put in on this street. */
	pot: number;
	in: [opener: number, other: number];
	/** Who acts now; null while a card is awaited or when the hand is over. */
	to_act: Side | null;
	over: null | 'fold' | 'showdown';
	/** The ranges as the line has left them, and how many combos they hold on this board. */
	ranges: Ranges;
	combos: [number, number];
	/** Share of each player's starting range that plays the line so far. */
	reach: [number, number];
	/** Equity of each range against the other on this board; null before the flop. */
	equity: [number, number] | null;
	/** Equity of each known hand against the other player's range; null if unknown or before the flop. */
	hand_equity: [number | null, number | null];
	/** The model's strategy of the player to act; null when nobody acts. */
	strategy: Strategy | null;
	/** The step that led here, how much of the acting range takes it, and how often the known hand does. */
	step: LineStep | null;
	taken: number | null;
	hand_taken: number | null;
}

/** The hand as an array of table statuses: before any step, then after each one. */
export function genGame(line: HandLine): TableStatus[] {
	const dead = [...(line.hole?.opener ?? []), ...(line.hole?.other ?? [])];
	const start = startStatus({
		kind: line.pot,
		opener: line.opener,
		other: line.other,
		players: line.players ?? 2
	});
	const game = [start];
	for (const step of line.steps) game.push(nextStatus(game[game.length - 1], step, dead));
	return game;
}

/**
 * The game with what a page shows at every status: ranges, strategy, equities, combos, reach. Throws
 * LineError on a step that cannot be played.
 */
export function replay(line: HandLine): LineState[] {
	const game = genGame(line);
	const holes = [line.hole?.opener ?? null, line.hole?.other ?? null];
	const start = genRange(game[0]);
	const states: LineState[] = [];
	let narrowed = false; // an action has changed a range since the flop
	game.forEach((table, index) => {
		const { board, to_act: by } = table;
		const ranges = genRange(table);
		const strategy = genStrategy(table, ranges, by ? holes[sideIndex(by)] : null);

		// What led here: how much of the acting range, and of the known hand, takes that step.
		const step = line.steps[index - 1] ?? null;
		let taken: number | null = null;
		let handTaken: number | null = null;
		if (step && step.kind !== 'deal') {
			narrowed = true;
			const before = states[index - 1].strategy;
			if (before) {
				const option = optionOf(step, toCallOf(game[index - 1], step.by));
				taken = before.range[option];
				handTaken = before.hand ? before.hand[option] : null;
			}
		}

		let equity: LineState['equity'] = null;
		const handEquity: LineState['hand_equity'] = [null, null];
		const combos: [number, number] = [comboCount(ranges[0], board), comboCount(ranges[1], board)];
		if (board.length >= 3 && combos[0] > 0 && combos[1] > 0) {
			// Untouched ranges on the flop: the group's exact number. Otherwise sampled on the ranges left.
			const read =
				!narrowed && board.length === 3
					? postflop.flopRead(table.flop_key, board, table.aggressor === 'opener')
					: null;
			const eq = read?.range_eq ?? postflop.matchupOf(board, ranges[0], ranges[1])?.range_eq;
			if (eq !== undefined) equity = [eq, 1 - eq];
			holes.forEach((hole, s) => {
				if (hole)
					handEquity[s] =
						postflop.matchupOf(board, ranges[s], ranges[1 - s])?.profile(hole)?.eq ?? null;
			});
		}
		const reach = (s: 0 | 1) => {
			const all = comboCount(without(start[s], board), board);
			return all > 0 ? combos[s] / all : 0;
		};
		states.push({
			index,
			street:
				board.length < 3
					? 'preflop'
					: board.length === 3
						? 'flop'
						: board.length === 4
							? 'turn'
							: 'river',
			board: [...board],
			pot: table.pot,
			in: [table.in[0], table.in[1]],
			to_act: table.over ? null : by,
			over: table.over,
			ranges,
			combos,
			reach: [reach(0), reach(1)],
			equity,
			hand_equity: handEquity,
			strategy,
			step,
			taken,
			hand_taken: handTaken
		});
	});
	return states;
}

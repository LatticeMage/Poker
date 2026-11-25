// The decision chain of a heads-up hand: what both players did, in order, from the flop to the end.
//
// A hand is a list of steps (a card is dealt, a player checks, bets, calls, raises or folds). The state
// after k steps is computed from the first k steps and from nothing else, so every stage can be asked
// for and going back is just a shorter list: `replay` always starts at the beginning.
//
// What moves along the chain is each player's range. A range starts as the preflop charts give it
// for the pot (single-raised, 3-bet, 4-bet), and every action multiplies each hand's weight by how
// often the model takes that action with that hand at that point:
//
//   weight after the step = weight before × P(action | hand, everything before)
//
// The product over the steps is how often a hand plays the whole line. Equity at a stage is the
// equity of the ranges as they are then, which is why it cannot be computed without what came before.
//
// P(action | hand) is the postflop model's: on the flop the solved table of the group where the
// matchup is solved (nvcc/flopsolve.cu), the rules on the range equity elsewhere, and on the turn and
// river always the rules, on the ranges the line has left. A bet's size does not change it yet: a hand
// "bets" or not; the per-size tables exist offline and are the next refinement.

import type { Card } from './cards';
import { matchupKey } from './flopGroups';
import * as postflop from './postflop';
import { seatRange } from './ranges';
import * as spots from './spots';

export type Side = 'opener' | 'other';
export type PotKind = 'srp' | '3bet' | '4bet';

/** One step of a hand. Amounts are in big blinds: `to` is what the player has put in on this street after acting. */
export type LineStep =
	| { kind: 'deal'; cards: Card[] }
	| { kind: 'check' | 'call' | 'fold'; by: Side }
	| { kind: 'bet' | 'raise'; by: Side; to: number };

/** A step a player takes (everything but a card being dealt). */
export type LineAction = Exclude<LineStep, { kind: 'deal' }>;

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

export class LineError extends Error {}

/** What the model does at a decision: shares of the acting player's range, and of a known hand. */
export interface LineStrategy {
	by: Side;
	/** Where the numbers come from: the solved table of the flop group, or the rules. */
	source: 'solved' | 'rules';
	/** Nobody has bet: [check, bet]. Facing a bet: [fold, call, raise]. */
	options: readonly string[];
	range: number[];
	/** The same for the player's own cards when they are known. */
	hand: number[] | null;
	bet_size: number;
	/** The same shares for every hand class of the acting range, in the order of `options`. */
	classes: Record<string, number[]>;
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
	ranges: [opener: postflop.Weighted, other: postflop.Weighted];
	combos: [number, number];
	/** Share of each player's starting range that plays the line so far. */
	reach: [number, number];
	/** Equity of each range against the other on this board; null before the flop. */
	equity: [number, number] | null;
	/** Equity of each known hand against the other player's range; null if unknown or before the flop. */
	hand_equity: [number | null, number | null];
	/** The model's strategy of the player to act; null when nobody acts. */
	strategy: LineStrategy | null;
	/** The step that led here, how much of the acting range takes it, and how often the known hand does. */
	step: LineStep | null;
	taken: number | null;
	hand_taken: number | null;
}

const OPEN_INDEX: Record<string, number> = { UTG: 0, MP: 1, CO: 2, BTN: 3, SB: 4 };
const ORDER = ['SB', 'BB', 'UTG', 'MP', 'CO', 'BTN']; // who acts first after the flop

/** An action the model never takes still leaves this much of a hand: the line happened, the model is not sure. */
export const LINE_FLOOR = 0.02;

/**
 * The pot on the flop and what each player has behind, in big blinds. 100bb stacks; the open is the
 * chart's; a 3-bet is three opens in position and four out of position; a 4-bet is 2.25 times the
 * 3-bet; the blinds of players who folded stay in the pot. The same sizes as nvcc/flopsolve.cuh.
 */
export function potOnFlop(
	kind: PotKind,
	opener: string,
	other: string
): { pot: number; stack: number } {
	const open = spots.OPEN_SIZE[opener];
	if (open === undefined || !ORDER.includes(other)) throw new LineError('No such matchup');
	const three = open * (ORDER.indexOf(other) > ORDER.indexOf(opener) ? 3 : 4);
	const put = kind === 'srp' ? open : kind === '3bet' ? three : three * 2.25;
	const dead =
		(opener !== 'SB' && other !== 'SB' ? 0.5 : 0) + (opener !== 'BB' && other !== 'BB' ? 1 : 0);
	return { pot: 2 * put + dead, stack: 100 - put };
}

const sideIndex = (side: Side): 0 | 1 => (side === 'opener' ? 0 : 1);
const SIDES: readonly Side[] = ['opener', 'other'];

/** The model's shares of [check, bet] or [fold, call, raise] for every hand class of the acting range. */
function classShares(
	board: readonly Card[],
	mine: postflop.Weighted,
	theirs: postflop.Weighted,
	plan: postflop.Plan,
	toCall: number
): Record<string, number[]> | null {
	const raw = postflop.classActions(board, mine, theirs, (p) =>
		postflop.actionShares(p, plan, toCall)
	);
	if (!raw) return null;
	const out: Record<string, number[]> = {};
	for (const [name, [value, bluff, call]] of Object.entries(raw)) {
		out[name] = toCall <= 0 ? [1 - value - bluff, value + bluff] : [1 - value - call, call, value];
	}
	return out;
}

const handShares = (prof: postflop.Profile, plan: postflop.Plan, toCall: number): number[] => {
	const [value, bluff, call] = postflop.actionShares(prof, plan, toCall);
	return toCall <= 0 ? [1 - value - bluff, value + bluff] : [1 - value - call, call, value];
};

/** Which of the model's options a step is. */
function optionOf(step: LineStep, toCall: number): number {
	if (toCall <= 0) return step.kind === 'check' ? 0 : 1;
	return step.kind === 'fold' ? 0 : step.kind === 'call' ? 1 : 2;
}

/**
 * The hand, stage by stage: the state before any step, then after each one. The ranges, the equities
 * and the strategy of a state depend on every step before it, so each call starts from the preflop
 * ranges. Throws LineError on a step that cannot be played.
 */
export function replay(line: HandLine): LineState[] {
	if (!(line.opener in OPEN_INDEX)) throw new LineError('That position cannot open');
	const built = postflop.potRanges(
		line.pot,
		line.opener,
		line.other,
		seatRange(6, OPEN_INDEX[line.opener]).hands
	);
	if (!built) throw new LineError('No chart for this matchup');
	// built = [aggressor, caller]; the opener is the aggressor except in a 3-bet pot.
	const aggressorSide: Side = line.pot === '3bet' ? 'other' : 'opener';
	const players = line.players ?? 2;
	const share = postflop.potTightness(players);
	const narrow = (r: postflop.Weighted) => postflop.tighten(r, share);
	const start: [postflop.Weighted, postflop.Weighted] =
		aggressorSide === 'opener'
			? [narrow(built[0]), narrow(built[1])]
			: [narrow(built[1]), narrow(built[0])];
	const key = players > 2 ? null : matchupKey(line.pot, line.opener, line.other);
	const first: Side = ORDER.indexOf(line.opener) < ORDER.indexOf(line.other) ? 'opener' : 'other';
	const holes = [line.hole?.opener ?? null, line.hole?.other ?? null];

	let ranges: [postflop.Weighted, postflop.Weighted] = start;
	let board: Card[] = [];
	let pot = potOnFlop(line.pot, line.opener, line.other).pot;
	let put: [number, number] = [0, 0];
	let toAct: Side | null = null;
	let over: LineState['over'] = null;
	let initiative: Side = aggressorSide; // who bet or raised last; the preflop aggressor until someone does
	let acted = 0; // actions on this street
	let narrowed = false; // an action has changed a range since the flop

	const startCombos = (b: readonly Card[]): [number, number] => [
		postflop.rangeCombos(start[0], b),
		postflop.rangeCombos(start[1], b)
	];

	/** The strategy of the player to act and the per-class shares behind it. */
	const strategyNow = (): { view: LineStrategy; classes: Record<string, number[]> } | null => {
		if (!toAct || board.length < 3) return null;
		const me = sideIndex(toAct);
		const mine = ranges[me];
		const theirs = ranges[1 - me];
		const toCall = Math.max(...put) - put[me];
		const hole = holes[me];
		const prof = postflop.profile(board, hole ?? postflop.anyTwo(board, mine), mine, theirs);
		if (!prof) return null;
		// The flop table describes the preflop ranges on the flop: it answers the flop's decisions.
		const read = board.length === 3 ? postflop.flopRead(key, board, toAct === aggressorSide) : null;
		const facingRaise = toCall > 0 && put[me] > 0;
		const plan = postflop.makePlan(prof, {
			texture: postflop.boardTexture(board),
			flop: read,
			pot,
			to_call: toCall,
			aggressor: toAct === initiative,
			facing_raise: facingRaise
		});
		const classes = classShares(board, mine, theirs, plan, toCall);
		if (!classes) return null;
		const n = toCall <= 0 ? 2 : 3;
		const total = new Array<number>(n).fill(0);
		let weight = 0;
		for (const [name, shares] of Object.entries(classes)) {
			const w = (mine[name] ?? 0) * postflop.rangeCombos({ [name]: 1 }, board);
			weight += w;
			shares.forEach((x, i) => (total[i] += w * x));
		}
		return {
			classes,
			view: {
				by: toAct,
				source: read?.solved && !facingRaise ? 'solved' : 'rules',
				options: toCall <= 0 ? ['check', 'bet'] : ['fold', 'call', 'raise'],
				range: total.map((x) => (weight > 0 ? x / weight : 0)),
				hand: hole ? handShares(prof, plan, toCall) : null,
				bet_size: plan.size,
				classes
			}
		};
	};

	const snapshot = (
		index: number,
		step: LineStep | null,
		taken: number | null,
		handTaken: number | null,
		strategy: LineStrategy | null
	): LineState => {
		let equity: LineState['equity'] = null;
		const handEquity: LineState['hand_equity'] = [null, null];
		const combos: [number, number] = [
			postflop.rangeCombos(ranges[0], board),
			postflop.rangeCombos(ranges[1], board)
		];
		if (board.length >= 3 && combos[0] > 0 && combos[1] > 0) {
			// Untouched ranges on the flop: the group's exact number. Otherwise sampled on the ranges left.
			const read =
				!narrowed && board.length === 3
					? postflop.flopRead(key, board, aggressorSide === 'opener')
					: null;
			const eq = read?.range_eq ?? postflop.rangeEquity(board, ranges[0], ranges[1]);
			if (eq !== null) equity = [eq, 1 - eq];
			holes.forEach((hole, s) => {
				if (hole)
					handEquity[s] = postflop.profile(board, hole, ranges[s], ranges[1 - s])?.eq ?? null;
			});
		}
		const base = startCombos(board);
		return {
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
			pot,
			in: [put[0], put[1]],
			to_act: over ? null : toAct,
			over,
			ranges: [ranges[0], ranges[1]],
			combos,
			reach: [base[0] > 0 ? combos[0] / base[0] : 0, base[1] > 0 ? combos[1] / base[1] : 0],
			equity,
			hand_equity: handEquity,
			strategy,
			step,
			taken,
			hand_taken: handTaken
		};
	};

	const states: LineState[] = [];
	let now = strategyNow();
	states.push(snapshot(0, null, null, null, now?.view ?? null));

	line.steps.forEach((step, k) => {
		if (over) throw new LineError('The hand is over');
		let taken: number | null = null;
		let handTaken: number | null = null;
		if (step.kind === 'deal') {
			if (toAct) throw new LineError('A player has to act before the next card');
			const want = board.length === 0 ? 3 : board.length < 5 ? 1 : 0;
			if (step.cards.length !== want) throw new LineError(`Expected ${want} card(s) here`);
			const seen = [...board, ...(holes[0] ?? []), ...(holes[1] ?? [])];
			for (const c of step.cards) {
				if (seen.includes(c) || step.cards.indexOf(c) !== step.cards.lastIndexOf(c))
					throw new LineError('That card is already out');
			}
			board = [...board, ...step.cards];
			put = [0, 0];
			acted = 0;
			toAct = first;
		} else {
			if (step.by !== toAct) throw new LineError(`It is not ${step.by}'s turn`);
			const me = sideIndex(step.by);
			const high = Math.max(...put);
			const toCall = high - put[me];
			if ((step.kind === 'check' || step.kind === 'bet') && toCall > 0)
				throw new LineError('There is a bet to answer');
			if ((step.kind === 'call' || step.kind === 'fold' || step.kind === 'raise') && toCall <= 0)
				throw new LineError('There is no bet to answer');
			if ((step.kind === 'bet' || step.kind === 'raise') && !(step.to > high))
				throw new LineError('A bet or raise has to put in more than the other player has');

			// The range of the player who acts: every hand keeps the share that takes this action.
			if (now) {
				const option = optionOf(step, toCall);
				const mine = ranges[me];
				const next: postflop.Weighted = {};
				let before = 0;
				let after = 0;
				for (const [name, w] of Object.entries(mine)) {
					const live = postflop.rangeCombos({ [name]: 1 }, board);
					const p = Math.max(LINE_FLOOR, now.classes[name]?.[option] ?? 0);
					before += w * live;
					after += w * live * p;
					if (w * p > 0) next[name] = w * p;
				}
				taken = now.view.range[option];
				handTaken = now.view.hand ? now.view.hand[option] : null;
				if (after > 0 && before > 0) {
					ranges = me === 0 ? [next, ranges[1]] : [ranges[0], next];
					narrowed = true;
				}
			}

			acted++;
			if (step.kind === 'fold') {
				over = 'fold';
				toAct = null;
			} else if (step.kind === 'check') {
				if (acted >= 2)
					toAct = null; // checked through
				else toAct = SIDES[1 - me];
			} else if (step.kind === 'call') {
				pot += toCall;
				put[me] = high;
				toAct = null;
			} else if (step.kind === 'bet' || step.kind === 'raise') {
				pot += step.to - put[me];
				put[me] = step.to;
				initiative = step.by;
				toAct = SIDES[1 - me];
			}
			if (!toAct && !over && board.length === 5) over = 'showdown';
		}
		now = over ? null : strategyNow();
		states.push(snapshot(k + 1, step, taken, handTaken, now?.view ?? null));
	});
	return states;
}

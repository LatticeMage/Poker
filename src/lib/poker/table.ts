// The table status of a heads-up hand and its bookkeeping: pot, bets, whose turn it is. A game is an
// array of these. Nothing here looks at a range: ranges are genRange.ts, strategies genStrategy.ts.

import type { Card } from './cards';
import { matchupKey } from './flopGroups';
import * as spots from './spots';

export type Side = 'opener' | 'other';

/** One step of a hand. Amounts are in big blinds: `to` is what the player has put in on this street after acting. */
export type LineStep =
	| { kind: 'deal'; cards: Card[] }
	| { kind: 'check' | 'call' | 'fold'; by: Side }
	| { kind: 'bet' | 'raise'; by: Side; to: number };

/** A step a player takes (everything but a card being dealt). */
export type LineAction = Exclude<LineStep, { kind: 'deal' }>;

export type PotKind = 'srp' | '3bet' | '4bet';

/** How the hand got to the flop. */
export interface Setup {
	kind: PotKind;
	opener: string;
	other: string;
	/** Players who came into the pot; with more than two the hand is still followed as these two. */
	players: number;
}

/**
 * Everything public about a hand at one moment, as plain JSON. It stands on its own: the setup and the
 * history are what genRange needs, the rest is what genStrategy looks at.
 */
export interface TableStatus extends Setup {
	/** Every step played so far, in order. */
	history: LineStep[];
	board: Card[];
	/** The pot in big blinds, bets of this street included, and what each player put in on this street. */
	pot: number;
	in: [opener: number, other: number];
	/** Who acts now; null while a card is awaited or when the hand is over. */
	to_act: Side | null;
	over: null | 'fold' | 'showdown';
	/** Who acts first after the flop (out of position). */
	first: Side;
	/** Who raised last before the flop, and who bet or raised last since. */
	aggressor: Side;
	initiative: Side;
	/** Actions on this street. */
	acted: number;
	/** The matchup of the offline flop tables; null where they do not apply (more than two players). */
	flop_key: string | null;
}

export class LineError extends Error {}

export const SIDES: readonly Side[] = ['opener', 'other'];
export const sideIndex = (side: Side): 0 | 1 => (side === 'opener' ? 0 : 1);

export const toCallOf = (table: TableStatus, side: Side): number =>
	Math.max(...table.in) - table.in[sideIndex(side)];

const ORDER = ['SB', 'BB', 'UTG', 'MP', 'CO', 'BTN']; // who acts first after the flop

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

/** The status before the flop is dealt. */
export function startStatus({ kind, opener, other, players }: Setup): TableStatus {
	const aggressor: Side = kind === '3bet' ? 'other' : 'opener';
	return {
		kind,
		opener,
		other,
		players,
		history: [],
		board: [],
		pot: potOnFlop(kind, opener, other).pot,
		in: [0, 0],
		to_act: null,
		over: null,
		first: ORDER.indexOf(opener) < ORDER.indexOf(other) ? 'opener' : 'other',
		aggressor,
		initiative: aggressor,
		acted: 0,
		flop_key: players > 2 ? null : matchupKey(kind, opener, other)
	};
}

/** The status after a step. `dead` are cards that cannot come (known hole cards). */
export function nextStatus(
	table: TableStatus,
	step: LineStep,
	dead: readonly Card[] = []
): TableStatus {
	if (table.over) throw new LineError('The hand is over');
	if (step.kind === 'deal') {
		if (table.to_act) throw new LineError('A player has to act before the next card');
		const want = table.board.length === 0 ? 3 : table.board.length < 5 ? 1 : 0;
		if (step.cards.length !== want) throw new LineError(`Expected ${want} card(s) here`);
		const seen = [...table.board, ...dead];
		for (const c of step.cards) {
			if (seen.includes(c) || step.cards.indexOf(c) !== step.cards.lastIndexOf(c))
				throw new LineError('That card is already out');
		}
		return {
			...table,
			history: [...table.history, step],
			board: [...table.board, ...step.cards],
			in: [0, 0],
			acted: 0,
			to_act: table.first
		};
	}

	if (step.by !== table.to_act) throw new LineError(`It is not ${step.by}'s turn`);
	const me = sideIndex(step.by);
	const high = Math.max(...table.in);
	const toCall = toCallOf(table, step.by);
	if ((step.kind === 'check' || step.kind === 'bet') && toCall > 0)
		throw new LineError('There is a bet to answer');
	if ((step.kind === 'call' || step.kind === 'fold' || step.kind === 'raise') && toCall <= 0)
		throw new LineError('There is no bet to answer');
	if ((step.kind === 'bet' || step.kind === 'raise') && !(step.to > high))
		throw new LineError('A bet or raise has to put in more than the other player has');

	const next: TableStatus = {
		...table,
		history: [...table.history, step],
		in: [table.in[0], table.in[1]],
		acted: table.acted + 1
	};
	const other = SIDES[1 - me];
	if (step.kind === 'fold') {
		next.over = 'fold';
		next.to_act = null;
	} else if (step.kind === 'check') {
		next.to_act = next.acted >= 2 ? null : other; // null: checked through
	} else if (step.kind === 'call') {
		next.pot += toCall;
		next.in[me] = high;
		next.to_act = null;
	} else if (step.kind === 'bet' || step.kind === 'raise') {
		next.pot += step.to - next.in[me];
		next.in[me] = step.to;
		next.initiative = step.by;
		next.to_act = other;
	}
	if (!next.to_act && !next.over && next.board.length === 5) next.over = 'showdown';
	return next;
}

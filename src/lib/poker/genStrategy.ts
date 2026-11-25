// genStrategy: from the table status and the two ranges, what the player to act does with every combo.
//
// It looks only at what it is given. How the hand got here matters through the ranges, and through
// two things the rules still read from the table: who has the initiative and whether this is a raise.
// Today the numbers are the postflop model's (the solved flop table for range totals where there is
// one, rules elsewhere); a solver for a street replaces them here, behind the same signature.

import type { Card } from './cards';
import * as postflop from './postflop';
import { COMBOS, type Range, isLive } from './range';
import { type Side, type TableStatus, sideIndex, toCallOf } from './table';

export interface Strategy {
	by: Side;
	/** Where the numbers come from: the solved table of the flop group, or the rules. */
	source: 'solved' | 'rules';
	/** Nobody has bet: [check, bet]. Facing a bet: [fold, call, raise]. */
	options: readonly string[];
	/** Shares of the options for every combo of the acting range (1,326); null where it holds none. */
	combos: (number[] | null)[];
	/** The same over the whole range. */
	range: number[];
	/** And for the player's own cards when they are known. */
	hand: number[] | null;
	bet_size: number;
}

/**
 * Pure: the same status, ranges and cards give the same strategy, and nothing given is changed.
 * `hole` are the cards of the player to act, when known: only `hand` in the result depends on them.
 */
export function genStrategy(
	table: TableStatus,
	ranges: readonly [opener: Range, other: Range],
	hole: readonly Card[] | null = null
): Strategy | null {
	const { board, to_act: by } = table;
	if (!by || table.over || board.length < 3) return null;
	const me = sideIndex(by);
	const mine = ranges[me];
	const matchup = postflop.matchupOf(board, mine, ranges[1 - me]);
	if (!matchup) return null;
	const first = COMBOS.findIndex((_, i) => mine[i] > 0 && isLive(i, board));
	const prof = matchup.profile(hole ?? COMBOS[first]);
	if (!prof) return null;

	const toCall = toCallOf(table, by);
	// The flop table describes the preflop ranges on the flop: it answers the flop's decisions.
	const read =
		board.length === 3 ? postflop.flopRead(table.flop_key, board, by === table.aggressor) : null;
	const facingRaise = toCall > 0 && table.in[me] > 0;
	const plan = postflop.makePlan(prof, {
		texture: postflop.boardTexture(board),
		flop: read,
		pot: table.pot,
		to_call: toCall,
		aggressor: by === table.initiative,
		facing_raise: facingRaise
	});
	const shares = (p: postflop.Profile): number[] => {
		const [value, bluff, call] = postflop.actionShares(p, plan, toCall);
		return toCall <= 0 ? [1 - value - bluff, value + bluff] : [1 - value - call, call, value];
	};

	const options = toCall <= 0 ? ['check', 'bet'] : ['fold', 'call', 'raise'];
	const combos: (number[] | null)[] = new Array(COMBOS.length).fill(null);
	const total = new Array<number>(options.length).fill(0);
	let weight = 0;
	for (let i = 0; i < COMBOS.length; i++) {
		if (mine[i] <= 0 || !isLive(i, board)) continue;
		const p = matchup.profile(COMBOS[i]);
		if (!p) continue;
		const x = shares(p);
		combos[i] = x;
		weight += mine[i];
		x.forEach((v, k) => (total[k] += mine[i] * v));
	}
	if (weight <= 0) return null;
	return {
		by,
		source: read?.solved && !facingRaise ? 'solved' : 'rules',
		options,
		combos,
		range: total.map((x) => x / weight),
		hand: hole ? shares(prof) : null,
		bet_size: plan.size
	};
}

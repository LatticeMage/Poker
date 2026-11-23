// The charts and equity views behind the Learn page, computed locally. Each function returns the same
// shapes the pages used to get from a server ($lib/types/ranges), so the components do not care.

import type {
	BoardEquity,
	BoardQuery,
	PushFoldChart,
	RangeChart,
	SpotChart,
	SpotList
} from '$lib/types/ranges';
import type { Card } from './cards';
import { evaluate } from './evaluate';
import * as postflop from './postflop';
import {
	EQUITY_VS_RANDOM,
	ORDER,
	chartFor,
	classCombos,
	handOf,
	percentile,
	pushfold,
	roundHalfEven,
	samplesPerMatchup,
	seatRange
} from './ranges';
import * as spots from './spots';

/** The request cannot be answered (a position that cannot open, a board with repeated cards, …). */
export class ChartError extends Error {}

const round1 = (x: number) => Math.round(x * 10) / 10;
const byStrength = (hands: Iterable<string>) =>
	[...hands].sort((a, b) => percentile(a) - percentile(b));

/** Raise-first-in starting-hand ranges for every seat at a table of this size. */
export function rangeChart(players: number): RangeChart {
	if (!Number.isInteger(players) || players < 2 || players > 9) {
		throw new ChartError('players must be between 2 and 9');
	}
	return {
		players,
		stack_bb: 100,
		open_size_bb: 2.5,
		seats: chartFor(players).map(({ name, behind, range }) => ({
			name,
			behind,
			pct: round1(range.pct),
			source: range.source,
			label: range.label,
			url: range.url,
			hands: byStrength(range.hands)
		})),
		order: [...ORDER],
		equity_vs_random: { ...EQUITY_VS_RANDOM }
	};
}

/** Heads-up push/fold ranges: the small blind shoves or folds, the big blind calls or folds. */
export function pushFoldChart(stackBb: number): PushFoldChart {
	const table = pushfold(stackBb);
	if (!table) throw new ChartError('Push/fold tables have not been generated');
	return {
		stack_bb: table.stack_bb,
		push: byStrength(table.push),
		call: byStrength(table.call),
		push_pct: table.push_pct,
		call_pct: table.call_pct,
		exploitability_bb: table.exploitability_bb,
		samples_per_matchup: samplesPerMatchup(),
		note:
			'Approximation by fictitious play on a Monte Carlo equity matrix. Push/fold is a good model ' +
			'only up to about 10bb; deeper, players also raise or limp.'
	};
}

/** Which spots have a chart: who can face whom. */
export function spotList(): SpotList {
	const order = spots.POSITIONS;
	const vsOpen: Record<string, string[]> = {};
	const vs3bet: Record<string, string[]> = {};
	order.forEach((hero, i) => {
		const openers = order.slice(0, i).filter((o) => spots.vsOpen(hero, o));
		const bettors = order.slice(i + 1).filter((t) => spots.vs3bet(hero, t));
		if (openers.length) vsOpen[hero] = openers;
		if (bettors.length) vs3bet[hero] = bettors;
	});
	return { positions: [...order], vs_open: vsOpen, vs_3bet: vs3bet };
}

/** Strategy for one spot: facing an open raise, or facing a 3-bet after opening (6-max, ~100bb). */
export function spotChart(kind: SpotChart['kind'], hero: string, villain: string): SpotChart {
	const key = `${hero}|${villain}`;
	let call: spots.Weights;
	let bluff: spots.Weights;
	let value: spots.Weights;
	let stated: [number, number, number] | undefined;
	let url: string | undefined;
	let labels: SpotChart['labels'];
	let note: string;
	if (kind === 'vs_open') {
		const spot = spots.vsOpen(hero, villain);
		stated = spots.STATED_OPEN[key];
		if (!spot) throw new ChartError('No chart for this spot');
		[call, bluff, value, url] = [spot.call, spot.bluff3, spot.value3, spot.url];
		labels = {
			call: 'Call',
			bluff: '3-bet (folds to a 4-bet)',
			value: '3-bet (continues vs a 4-bet)'
		};
		note = `${hero} facing an open from ${villain}, which the chart sizes at ${spots.OPEN_SIZE[villain]}bb.`;
	} else {
		const spot = spots.vs3bet(hero, villain);
		stated = spots.STATED_3BET[key];
		if (!spot) throw new ChartError('No chart for this spot');
		[call, bluff, value, url] = [spot.call, spot.bluff4, spot.value4, spot.url];
		labels = {
			call: 'Call',
			bluff: '4-bet (folds to a 5-bet)',
			value: '4-bet (continues vs a 5-bet)'
		};
		note = `${hero} opened and ${villain} 3-bet.`;
	}
	if (!stated) throw new ChartError('No chart for this spot');
	return {
		kind,
		hero,
		villain,
		open_size_bb: kind === 'vs_open' ? spots.OPEN_SIZE[villain] : null,
		labels,
		call,
		bluff,
		value,
		combos: { call: stated[0], bluff: stated[1], value: stated[2] },
		url,
		note: `${note} Seats at other table sizes map to the nearest 6-max seat.`
	};
}

const STREETS = { 3: 'flop', 4: 'turn', 5: 'river' } as const;
const OPEN_INDEX: Record<string, number> = { UTG: 0, MP: 1, CO: 2, BTN: 3, SB: 4 }; // seats of a 6-player table

/** Two cards from the range that avoid the board: the profile needs a hand even when none is given. */
function anyTwo(board: readonly Card[], mine: postflop.Weighted): [Card, Card] {
	for (const name of Object.keys(mine).sort()) {
		for (const combo of classCombos(name)) {
			if (!board.includes(combo[0]) && !board.includes(combo[1])) return [combo[0], combo[1]];
		}
	}
	const free = Array.from({ length: 52 }, (_, i) => i).filter((c) => !board.includes(c));
	return [free[0], free[1]];
}

const percentText = (x: number) => `${Math.round(x * 100)}%`;

/** What the postflop model does with this hand (without the random mixing at the thresholds). */
function recommend(
	prof: postflop.Profile,
	plan: postflop.Plan,
	toCall: number
): [action: string, reason: string] {
	const rank = postflop.rankIn(prof);
	if (toCall <= 0) {
		if (rank <= plan.value_cut) {
			return [
				'bet for value',
				`it is in the top ${percentText(plan.value_cut)} of your range, which bets for value`
			];
		}
		if (prof.nopair && postflop.bluffsAhead(prof, plan.value_cut) <= plan.bluff_quota) {
			return [
				'bluff',
				'no made hand but one of the best draws / overcards, so it fits the bluff quota'
			];
		}
		return [
			'check',
			`it is outside the top ${percentText(plan.value_cut)} and not one of the best bluffs`
		];
	}
	if (rank > plan.defend) {
		return [
			'fold',
			`it is outside the top ${percentText(plan.defend)} of your range that must continue`
		];
	}
	if (rank <= 0.1) return ['raise', 'it is in the top 10% of your range'];
	return [
		'call',
		`it is inside the top ${percentText(plan.defend)} of your range that must continue`
	];
}

/**
 * Equity on a heads-up flop, turn or river, with and without knowing your own hand.
 *
 * Unknown hands: neither player's cards are known, so it is range against range: the average equity
 * of each range, and the equity of every hand class in a range. Known hand: your two cards are given,
 * theirs are not: your hand against their whole range, where it ranks inside your range, and what the
 * postflop model would do with it. Ranges come from the 6-max preflop charts.
 */
export function boardEquity(q: BoardQuery): BoardEquity {
	const board = q.board;
	const hole = q.hole;
	if (board.length < 3 || board.length > 5) throw new ChartError('board: expected 3 to 5 cards');
	if (new Set(board).size !== board.length) throw new ChartError('board: cards must be different');
	if (hole && (hole.length !== 2 || hole[0] === hole[1]))
		throw new ChartError('hole: expected 2 different cards');
	if (hole && hole.some((c) => board.includes(c)))
		throw new ChartError('hole: cards are already on the board');
	if (!(q.opener in OPEN_INDEX)) throw new ChartError('That position cannot open');
	const built = postflop.potRanges(
		q.pot,
		q.opener,
		q.other,
		seatRange(6, OPEN_INDEX[q.opener]).hands
	);
	if (!built) throw new ChartError('No chart for this matchup');
	const [aggressorRange, callerRange] = built;
	// The opener is the preflop aggressor in single-raised and 4-bet pots; the 3-bettor in a 3-bet pot.
	const isAggressor = (q.pot !== '3bet') === (q.side === 'opener');
	const [mine, theirs] = isAggressor
		? [aggressorRange, callerRange]
		: [callerRange, aggressorRange];

	const texture = postflop.boardTexture(board);
	const grid =
		q.view === 'mine'
			? postflop.classEquities(board, mine, theirs)
			: postflop.classEquities(board, theirs, mine);
	const prof = postflop.profile(board, hole ?? anyTwo(board, mine), mine, theirs);
	const myCurve = postflop.equityCurve(board, mine, theirs);
	const theirCurve = postflop.equityCurve(board, theirs, mine);
	if (!grid || !prof || !myCurve || !theirCurve)
		throw new ChartError('No hands left in a range on this board');
	const myEq = prof.range_eq;
	const advantage = myEq > 0.52 ? 'you' : myEq < 0.48 ? 'them' : 'even';

	const toCall = q.bet ?? 0;
	const plan = postflop.makePlan(prof, {
		texture,
		pot: 1 + toCall,
		to_call: toCall,
		aggressor: isAggressor,
		facing_raise: false
	});
	const facing = q.bet
		? {
				bet: toCall,
				required_equity: toCall / (1 + 2 * toCall),
				defend_frequency: postflop.defendFrequency(1 + toCall, toCall)
			}
		: null;

	let known: BoardEquity['known'] = null;
	if (hole) {
		const made = evaluate([...hole, ...board]);
		const [action, reason] = recommend(prof, plan, toCall);
		known = {
			cards: [...hole],
			hand: handOf(hole),
			equity: prof.eq,
			their_equity: 1 - prof.eq,
			vs_range: prof.eq - myEq,
			rank: postflop.rankIn(prof),
			made_hand: made.name,
			has_made_hand: !prof.nopair,
			action,
			reason
		};
	}
	return {
		street: STREETS[board.length as 3 | 4 | 5],
		pot: q.pot,
		opener: q.opener,
		other: q.other,
		side: q.side,
		aggressor: isAggressor,
		board: [...board],
		texture,
		unknown: {
			my_range_eq: myEq,
			their_range_eq: 1 - myEq,
			advantage,
			my_combos: roundHalfEven(postflop.rangeCombos(mine, board)),
			their_combos: roundHalfEven(postflop.rangeCombos(theirs, board)),
			my_curve: myCurve,
			their_curve: theirCurve,
			view: q.view,
			grid
		},
		known,
		plan: {
			role: isAggressor ? 'aggressor' : 'caller',
			bet_size: plan.size,
			bet_frequency: plan.frequency,
			value_cut: plan.value_cut,
			bluff_quota: plan.bluff_quota,
			bluff_share_of_bets: postflop.bluffShare(plan.size)
		},
		facing,
		note:
			'6-max preflop charts, ~100bb. Equity is all-in equity to the river; the opponent is assumed to ' +
			'hold their whole preflop range.'
	};
}

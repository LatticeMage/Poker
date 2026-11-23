import { describe, expect, it } from 'vitest';
import type { BoardQuery } from '$lib/types/ranges';
import { ChartError, boardEquity, pushFoldChart, rangeChart, spotChart, spotList } from './charts';
import { parseCompact } from './cards';
import { combos } from './ranges';

const base: BoardQuery = {
	pot: 'srp',
	opener: 'BTN',
	other: 'BB',
	side: 'opener',
	board: parseCompact('As7d2c'),
	hole: null,
	bet: null,
	view: 'mine'
};
type QueryPatch = Omit<Partial<BoardQuery>, 'board' | 'hole'> & {
	board?: string;
	hole?: string | null;
};
const query = ({ board, hole, ...rest }: QueryPatch): BoardQuery => ({
	...base,
	...rest,
	board: board ? parseCompact(board) : base.board,
	hole: hole ? parseCompact(hole) : null
});

describe('range charts', () => {
	it('answers every table size', () => {
		for (let players = 2; players <= 9; players++) {
			const chart = rangeChart(players);
			expect(chart.seats).toHaveLength(players);
			expect(chart.seats[players - 1].name).toBe('BB');
			expect(chart.order).toHaveLength(169);
			expect(Object.keys(chart.equity_vs_random)).toHaveLength(169);
			for (const seat of chart.seats) {
				expect(seat.source === 'none').toBe(seat.name === 'BB');
				expect(seat.hands.every((h) => chart.order.includes(h))).toBe(true);
			}
		}
		const six = Object.fromEntries(rangeChart(6).seats.map((s) => [s.name, s]));
		expect(six.UTG.pct).toBe(14.3);
		expect(six.UTG.source).toBe('table');
		expect(six.UTG.hands).toContain('AKs');
		expect(six.BTN.hands).not.toContain('72o');
		expect(() => rangeChart(1)).toThrow(ChartError);
		expect(() => rangeChart(10)).toThrow(ChartError);
	});

	it('mixes the edge of an opening range and keeps its total share', () => {
		for (const seat of rangeChart(6).seats.filter((x) => x.source !== 'none')) {
			const freqs = Object.values(seat.mix);
			expect(
				freqs.some((f) => f > 0 && f < 1),
				seat.name
			).toBe(true);
			expect(seat.hands.every((h) => seat.mix[h] > 0)).toBe(true);
			const share = Object.entries(seat.mix).reduce((n, [h, f]) => n + f * combos(h), 0) / 13.26;
			expect(Math.abs(share - seat.pct)).toBeLessThan(0.6);
		}
		const solved = pushFoldChart(10);
		expect(Object.values(solved.push_mix).some((f) => f > 0 && f < 1)).toBe(true);
		expect(solved.push_mix.AA).toBe(1);
	});

	it('answers the push/fold chart', () => {
		for (const stack of [2, 10, 20]) {
			const chart = pushFoldChart(stack);
			expect(chart.stack_bb).toBe(stack);
			expect(chart.push).toEqual(expect.arrayContaining(['AA', 'KK', 'AKs']));
			expect(chart.samples_per_matchup).toBeGreaterThan(0);
			expect(chart.exploitability_bb).toBeLessThan(0.05);
		}
	});

	it('lists the spots and answers each chart', () => {
		const list = spotList();
		expect(list.positions).toEqual(['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']);
		expect(list.vs_open.BB).toEqual(['UTG', 'MP', 'CO', 'BTN', 'SB']);
		expect(list.vs_open.UTG).toBeUndefined();
		expect(list.vs_3bet.UTG).toEqual(['MP', 'CO', 'BTN', 'SB', 'BB']);
		expect(list.vs_3bet.BB).toBeUndefined();
		const open = spotChart('vs_open', 'BTN', 'CO');
		expect(open.combos).toEqual({ call: 154, bluff: 64, value: 54 });
		expect(open.open_size_bb).toBe(2.5);
		expect(open.call.AA).toBeCloseTo(1 / 6);
		expect(open.value.AA).toBeCloseTo(5 / 6);
		expect(open.url.endsWith('/2256/')).toBe(true);
		expect(spotChart('vs_3bet', 'CO', 'BTN').combos).toEqual({ call: 50, bluff: 44, value: 32 });
		expect(() => spotChart('vs_open', 'UTG', 'MP')).toThrow(ChartError);
	});
});

describe('board equity', () => {
	it('without and with my hand', () => {
		const out = boardEquity(base);
		const u = out.unknown;
		expect(u.my_range_eq + u.their_range_eq).toBeCloseTo(1);
		expect(u.advantage).toBe('you'); // the opener is ahead on an ace-high board
		expect(out.aggressor).toBe(true);
		expect(out.plan.role).toBe('aggressor');
		expect(out.texture).toBe('dry');
		expect(out.known).toBeNull();
		expect(out.facing).toBeNull();
		expect(u.grid.AA).toBeGreaterThan(0.8);
		expect(u.grid.AA).toBeGreaterThan(u.grid.KQs);
		expect(u.my_combos).toBeGreaterThan(100);
		expect(u.their_combos).toBeGreaterThan(100);
		// The equity distribution: two curves, weakest hand first.
		expect(u.my_curve.length).toBeGreaterThanOrEqual(40);
		expect(u.my_curve).toEqual([...u.my_curve].sort((a, b) => a - b));
		expect(u.their_curve).toEqual([...u.their_curve].sort((a, b) => a - b));
		expect(u.my_curve.reduce((s, v) => s + v, 0) / u.my_curve.length).toBeCloseTo(u.my_range_eq);
		expect(
			Math.abs(u.their_curve.reduce((s, v) => s + v, 0) / u.their_curve.length - u.their_range_eq)
		).toBeLessThan(0.04);
		expect(u.my_curve[u.my_curve.length - 1]).toBeGreaterThan(0.9);
		expect(u.my_curve[0]).toBeLessThan(0.25);

		const set = boardEquity(query({ hole: 'AhAd' })).known!;
		expect(set.hand).toBe('AA');
		expect(set.has_made_hand).toBe(true);
		expect(set.made_hand).toBe('Three of a Kind, Aces');
		expect(set.equity).toBeGreaterThan(0.9);
		expect(set.rank).toBeLessThan(0.1);
		expect(set.their_equity).toBeCloseTo(1 - set.equity);
		expect(set.vs_range).toBeCloseTo(set.equity - u.my_range_eq);
		expect(set.action).toBe('bet for value');

		const air = boardEquity(query({ hole: '9s3d' })).known!;
		expect(air.has_made_hand).toBe(false);
		expect(air.equity).toBeLessThan(0.4);
		expect(['check', 'bluff']).toContain(air.action);
	});

	it('for the caller facing a bet', () => {
		const out = boardEquity(query({ side: 'other', hole: '7h7c', bet: 0.5 }));
		expect(out.aggressor).toBe(false);
		expect(out.plan.role).toBe('caller');
		expect(out.facing!.defend_frequency).toBeCloseTo(2 / 3);
		expect(out.facing!.required_equity).toBeCloseTo(0.25); // call 0.5 to win 1 + 0.5 + 0.5
		expect(['raise', 'call']).toContain(out.known!.action);
		expect(boardEquity(query({ side: 'other', hole: '9s3d', bet: 0.5 })).known!.action).toBe(
			'fold'
		);
		const theirs = boardEquity(query({ side: 'other', view: 'theirs' }));
		const mine = boardEquity(query({ side: 'other' }));
		expect(theirs.unknown.view).toBe('theirs');
		expect(new Set(Object.keys(theirs.unknown.grid))).not.toEqual(
			new Set(Object.keys(mine.unknown.grid))
		);
	});

	it('in 3-bet and 4-bet pots', () => {
		const board = 'Kh9d4c';
		const three = boardEquity(
			query({ pot: '3bet', opener: 'CO', other: 'BTN', side: 'other', board })
		);
		expect(three.aggressor).toBe(true); // the 3-bettor
		const opener = boardEquity(
			query({ pot: '3bet', opener: 'CO', other: 'BTN', side: 'opener', board })
		);
		expect(opener.aggressor).toBe(false);
		expect(opener.plan.role).toBe('caller');
		expect(three.unknown.my_combos).toBeGreaterThan(opener.unknown.my_combos); // value + bluff 3-bets vs a call
		const four = boardEquity(
			query({ pot: '4bet', opener: 'CO', other: 'BTN', side: 'opener', board })
		);
		expect(four.aggressor).toBe(true);
		expect(four.unknown.my_combos).toBeLessThan(three.unknown.my_combos); // a 4-betting range is tighter still
	});

	it('on the turn and river', () => {
		const turn = boardEquity(query({ board: 'As7d2cKh', hole: 'AhAd' }));
		const river = boardEquity(query({ board: 'As7d2cKh9s', hole: 'AhAd' }));
		expect([turn.street, turn.board.length]).toEqual(['turn', 4]);
		expect([river.street, river.board.length]).toEqual(['river', 5]);
		for (const out of [turn, river]) {
			expect(out.unknown.my_range_eq + out.unknown.their_range_eq).toBeCloseTo(1, 1);
			expect(out.texture).toBe('dry'); // read from the flop
			expect(out.known!.equity).toBeGreaterThan(0.9);
			expect(out.known!.made_hand).toBe('Three of a Kind, Aces');
		}
		expect(river.unknown.my_range_eq).not.toBe(turn.unknown.my_range_eq);
	});

	it('a missed draw loses its equity by the river', () => {
		const q = { side: 'other' as const, hole: 'Qh5h', pot: 'srp' as const };
		const flop = boardEquity(query({ ...q, board: 'As7h2h' })).known!;
		const river = boardEquity(query({ ...q, board: 'As7h2hKc3d' })).known!;
		expect(flop.equity).toBeGreaterThan(river.equity + 0.1);
		expect(river.has_made_hand).toBe(false);
		const hit = boardEquity(query({ ...q, board: 'As7h2hKh3d' })).known!;
		expect(hit.equity).toBeGreaterThan(0.9);
		expect(hit.made_hand.startsWith('Flush')).toBe(true);
	});

	it('rejects bad requests', () => {
		const bad: Partial<BoardQuery>[] = [
			{ board: [48, 22] }, // too few cards
			{ board: [48, 22, 48] }, // repeated card
			{ board: parseCompact('As7d2cKh9s3c') }, // too many
			{ hole: parseCompact('As2d') }, // on the board
			{ opener: 'BB' }, // cannot open
			{ other: 'UTG' } // cannot face an open from later seats
		];
		for (const patch of bad)
			expect(() => boardEquity({ ...base, ...patch }), JSON.stringify(patch)).toThrow(ChartError);
		expect(() => boardEquity(query({ board: 'As7d2cKh', hole: 'KhQd' }))).toThrow(ChartError); // the turn is in your hand
	});

	it('is quick enough to answer on every click', () => {
		const started = performance.now();
		boardEquity(query({ board: 'Kc9h4d', hole: 'KhQh' })); // a board it has not seen
		const firstMs = performance.now() - started;
		const again = performance.now();
		boardEquity(query({ board: 'Kc9h4d', hole: 'AhQh' })); // same board, other hand: cached statistics
		const secondMs = performance.now() - again;
		console.log(
			`board equity: new board ${firstMs.toFixed(0)} ms, same board ${secondMs.toFixed(0)} ms`
		);
		expect(firstMs).toBeLessThan(1500);
		expect(secondMs).toBeLessThan(firstMs);
	});
});

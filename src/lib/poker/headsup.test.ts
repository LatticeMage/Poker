import { describe, expect, it } from 'vitest';
import { ChartError, HEADS_UP_STACKS, headsUpChart, rangeChart } from './charts';
import { DEFAULT_HEADS_UP_BB, firstIn, nearestStack } from './headsup';
import { combos, seatRange } from './ranges';

describe('solved heads-up trees', () => {
	it('loads every solved depth from the data files', () => {
		expect([...HEADS_UP_STACKS]).toEqual([5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150]);
		for (const depth of HEADS_UP_STACKS) {
			const chart = headsUpChart(depth);
			expect(chart.stack_bb).toBe(depth);
			expect(chart.nodes.root.player).toBe('SB');
			expect(Object.keys(chart.nodes.root.freq)).toHaveLength(169);
		}
		expect(() => headsUpChart(33)).toThrow(ChartError);
	});

	it('finds the nearest solved depth', () => {
		expect(nearestStack(100)).toBe(100);
		expect(nearestStack(120)).toBe(100);
		expect(nearestStack(130)).toBe(150);
		expect(nearestStack(1)).toBe(5);
		expect(nearestStack(1000)).toBe(150);
	});

	it('splits the first action of every hand into raise, limp and fold shares', () => {
		const { stack_bb, shares } = firstIn();
		expect(stack_bb).toBe(DEFAULT_HEADS_UP_BB);
		for (const [raise, limp, fold] of Object.values(shares)) {
			expect(raise + limp + fold).toBeCloseTo(1, 3);
		}
	});
});

describe('heads-up first-in reads the solved tree', () => {
	const range = seatRange(2, 0);

	it('is labelled as a solver approximation, not a chart', () => {
		expect(range.source).toBe('solver');
		expect(range.url).toBe('');
		expect(range.label).toMatch(/Solver approximation at 100bb/);
	});

	it('mixes raise and limp shares and keeps pct as the weighted raise total', () => {
		const mix = range.mix!;
		const limp = range.limp!;
		expect(Object.values(mix).some((f) => f > 0 && f < 1)).toBe(true);
		expect(Object.keys(limp).length).toBeGreaterThan(0);
		const share = Object.entries(mix).reduce((n, [h, f]) => n + f * combos(h), 0) / 13.26;
		expect(Math.abs(share - range.pct)).toBeLessThan(0.6);
		expect([...range.hands].every((h) => mix[h] > 0)).toBe(true);
		expect((mix.AA ?? 0) + (limp.AA ?? 0)).toBeLessThanOrEqual(1.01);
		expect(range.hands.has('72o') && mix['72o'] > 0.5).toBe(false);
	});

	it('shows up in the chart for 2 players, other table sizes are unchanged', () => {
		const seat = rangeChart(2).seats[0];
		expect(seat.name).toBe('SB/BTN');
		expect(seat.source).toBe('solver');
		expect(seat.mix.AA).toBe(range.mix!.AA);
		expect(seat.limp).toEqual(range.limp);
		for (const seat of rangeChart(3).seats) {
			expect(seat.source).not.toBe('solver');
			expect(seat.limp).toBeUndefined();
		}
	});
});

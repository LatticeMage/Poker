import { describe, expect, it } from 'vitest';
import { parseCard, parseCards } from './cards';
import {
	ALL_HANDS,
	CHARTS,
	EQUITY_VS_RANDOM,
	MAX_PUSHFOLD_BB,
	MIN_PUSHFOLD_BB,
	ORDER,
	adjust,
	chartFor,
	chartHands,
	chenValue,
	combos,
	handOf,
	parseRange,
	percent,
	percentile,
	pushfold,
	roundHalfEven,
	seatNames,
	seatRange,
	topPercent
} from './ranges';

describe('ranges', () => {
	it('has 169 hands and 1326 combos', () => {
		expect(ALL_HANDS).toHaveLength(169);
		expect(ALL_HANDS.reduce((sum, h) => sum + combos(h), 0)).toBe(1326);
	});

	it('parses the notation', () => {
		expect(parseRange('77-55')).toEqual(new Set(['77', '66', '55']));
		expect(parseRange('KTs+')).toEqual(new Set(['KTs', 'KJs', 'KQs']));
		expect(parseRange('T8s+')).toEqual(new Set(['T8s', 'T9s']));
		expect(parseRange('A5s-A2s')).toEqual(new Set(['A5s', 'A4s', 'A3s', 'A2s']));
		expect(parseRange('AJo+')).toEqual(new Set(['AJo', 'AQo', 'AKo']));
		expect(parseRange('87s, 65s')).toEqual(new Set(['87s', '65s']));
		for (const bad of ['AKx', 'KA+', 'A5s-K5s', 'ATs-A9o', '1+']) {
			expect(() => parseRange(bad), bad).toThrow();
		}
	});

	it('names two cards', () => {
		expect(handOf(parseCards('As Ah'))).toBe('AA');
		expect(handOf(parseCards('Ks As'))).toBe('AKs');
		expect(handOf([parseCard('9d'), parseCard('Th')])).toBe('T9o');
	});

	it('every chart list matches the percentage its source states', () => {
		for (const chart of Object.values(CHARTS)) {
			const tolerance = Number.isInteger(chart.stated_pct) ? 0.5 : 0.15;
			expect(
				Math.abs(percent(chartHands(chart.key)) - chart.stated_pct),
				chart.key
			).toBeLessThanOrEqual(tolerance);
		}
	});

	it('the charts nest from tight to wide', () => {
		const pairs = [
			['UTG9', 'UTG6'],
			['UTG9', 'MP9'],
			['UTG6', 'HJ6'],
			['HJ6', 'CO6'],
			['CO6', 'SB6'],
			['SB6', 'BTN6']
		];
		for (const [tight, wide] of pairs) {
			const missing = [...chartHands(tight)].filter((h) => !chartHands(wide).has(h));
			expect(missing, `${tight} in ${wide}`).toEqual([]);
		}
	});

	it('strength order puts premiums first', () => {
		expect(ORDER.slice(0, 3)).toEqual(['AA', 'KK', 'QQ']);
		expect(ORDER.slice(0, 12)).toEqual(expect.arrayContaining(['JJ', 'TT', 'AKs', 'AKo', 'AQs']));
		expect(percentile('AA')).toBeLessThan(percentile('72o'));
		expect(ORDER.indexOf('AKs')).toBeLessThan(ORDER.indexOf('A9s'));
		expect(ORDER.indexOf('A9s')).toBeLessThan(ORDER.indexOf('72o'));
		expect(percentile(ORDER[ORDER.length - 1])).toBeCloseTo(100);
	});

	it('topPercent and adjust', () => {
		expect(Math.abs(percent(topPercent(10)) - 10)).toBeLessThan(1);
		const base = chartHands('UTG6');
		const tighter = adjust(base, 8);
		const looser = adjust(base, 25);
		expect([...tighter].every((h) => base.has(h))).toBe(true);
		expect([...base].every((h) => looser.has(h))).toBe(true);
		expect(Math.abs(percent(tighter) - 8)).toBeLessThan(1);
		expect(Math.abs(percent(looser) - 25)).toBeLessThan(1);
		const weakestKept = Math.max(...[...tighter].map((h) => ORDER.indexOf(h)));
		for (const h of base)
			if (!tighter.has(h)) expect(ORDER.indexOf(h)).toBeGreaterThan(weakestKept);
	});

	it('Chen formula', () => {
		expect(chenValue(14, 14, false)).toBe(20);
		expect(chenValue(14, 13, true)).toBe(12);
		expect(chenValue(10, 9, true)).toBe(8);
		expect(chenValue(7, 2, false)).toBe(-1);
	});

	it('names seats', () => {
		expect(seatNames(2)).toEqual(['SB/BTN', 'BB']);
		expect(seatNames(3)).toEqual(['BTN', 'SB', 'BB']);
		expect(seatNames(6)).toEqual(['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']);
		expect(seatNames(9)).toEqual(['UTG', 'UTG+1', 'UTG+2', 'LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']);
	});

	it('six-max uses the six-max charts verbatim', () => {
		const got = Object.fromEntries(chartFor(6).map((s) => [s.name, s.range]));
		expect(got.UTG.hands).toEqual(chartHands('UTG6'));
		expect(got.CO.hands).toEqual(chartHands('CO6'));
		expect(got.BTN.hands).toEqual(chartHands('BTN6'));
		expect(got.SB.hands).toEqual(chartHands('SB6'));
		expect(got.BB.source).toBe('none');
	});

	it('nine-max mixes lookup and computed seats, widening toward the button', () => {
		const rows = chartFor(9);
		const sources = Object.fromEntries(rows.map((r) => [r.name, r.range.source]));
		expect(sources.UTG).toBe('table');
		expect(sources['UTG+1']).toBe('computed');
		expect(sources['UTG+2']).toBe('computed');
		const pct = Object.fromEntries(rows.map((r) => [r.name, r.range.pct]));
		expect(pct.UTG).toBeLessThan(pct['UTG+1']);
		expect(pct['UTG+1']).toBeLessThan(pct['UTG+2']);
		expect(pct['UTG+2']).toBeLessThan(pct.HJ);
		expect(pct.HJ).toBeLessThan(pct.CO);
		expect(pct.CO).toBeLessThan(pct.BTN);
	});

	it('every table size has a chart; fewer players open wider', () => {
		for (let n = 2; n <= 9; n++) {
			const rows = chartFor(n);
			expect(rows.map((r) => r.name)).toEqual(seatNames(n));
			const pcts = rows
				.filter((r) => !['SB', 'BB', 'SB/BTN'].includes(r.name))
				.map((r) => r.range.pct);
			expect(pcts).toEqual([...pcts].sort((a, b) => a - b));
		}
		const utg = [3, 4, 5, 6, 7, 8, 9].map((n) => seatRange(n, 0).pct);
		expect(utg).toEqual([...utg].sort((a, b) => b - a));
		expect(seatRange(2, 0).source).toBe('fallback');
	});
});

describe('computed tables', () => {
	it('equity against a random hand matches the known figures', () => {
		expect(Object.keys(EQUITY_VS_RANDOM)).toHaveLength(169);
		expect(Math.abs(EQUITY_VS_RANDOM.AA - 0.852)).toBeLessThan(0.015);
		expect(Math.abs(EQUITY_VS_RANDOM.KK - 0.824)).toBeLessThan(0.015);
		expect(Math.abs(EQUITY_VS_RANDOM.AKs - 0.67)).toBeLessThan(0.015);
		expect(Math.abs(EQUITY_VS_RANDOM['72o'] - 0.346)).toBeLessThan(0.015);
	});

	it('push/fold covers every depth, clamps, and tightens as stacks deepen', () => {
		for (let d = MIN_PUSHFOLD_BB; d <= MAX_PUSHFOLD_BB; d++) {
			const t = pushfold(d)!;
			expect(t.stack_bb).toBe(d);
			expect(Math.abs(percent(t.push) - t.push_pct)).toBeLessThan(0.1);
			expect(Math.abs(percent(t.call) - t.call_pct)).toBeLessThan(0.1);
			for (const h of ['AA', 'KK', 'QQ', 'AKs', 'AKo']) expect(t.push.has(h)).toBe(true);
			expect(t.exploitability_bb).toBeLessThan(0.05);
		}
		expect(pushfold(1)!.stack_bb).toBe(MIN_PUSHFOLD_BB);
		expect(pushfold(50)!.stack_bb).toBe(MAX_PUSHFOLD_BB);
		expect(pushfold(9.6)!.stack_bb).toBe(10);
		expect(pushfold(10)!.push_pct).toBeGreaterThan(50);
		expect(pushfold(10)!.call_pct).toBeLessThan(45);
		expect(pushfold(2)!.call_pct).toBeGreaterThan(95);
	});

	it('rounds half to even like Python', () => {
		expect([0.5, 1.5, 2.5, 3.5, 10.5].map(roundHalfEven)).toEqual([0, 2, 2, 4, 10]);
	});
});

// The spot charts are transcribed from PokerStrategy lists; each prints its combo count (#N), so a
// typo or a parser bug shows up as a count that does not match. (Same checks as the Python tests.)

import { describe, expect, it } from 'vitest';
import {
	POSITIONS,
	STATED_3BET,
	STATED_OPEN,
	comboCount,
	mdf,
	parseWeighted,
	positionClass,
	sizeFactor,
	vs3bet,
	vsOpen
} from './spots';

describe('weighted hand lists', () => {
	it('parses the notation', () => {
		expect(parseWeighted('QQ-88')).toEqual(
			Object.fromEntries(['QQ', 'JJ', 'TT', '99', '88'].map((h) => [h, 1]))
		);
		expect(parseWeighted('AsAc')).toEqual({ AA: 1 / 6 });
		expect(parseWeighted('AsAc, AdAh')).toEqual({ AA: 2 / 6 });
		expect(parseWeighted('AdKh')).toEqual({ AKo: 1 / 12 });
		expect(parseWeighted('AsKs')).toEqual({ AKs: 1 / 4 });
		expect(parseWeighted('AK')).toEqual({ AKs: 1, AKo: 1 });
		expect(parseWeighted('AQ+')).toEqual({ AQs: 1, AQo: 1, AKs: 1, AKo: 1 });
		expect(parseWeighted('KJo+')).toEqual({ KJo: 1, KQo: 1 });
		expect(Object.keys(parseWeighted('A5s-A2s')).sort()).toEqual(['A2s', 'A3s', 'A4s', 'A5s']);
		expect(Object.keys(parseWeighted('JTs-98s')).sort()).toEqual(['98s', 'JTs', 'T9s']);
		expect(Object.keys(parseWeighted('86s+')).sort()).toEqual(['86s', '87s']);
		expect(Object.keys(parseWeighted('K7s-K6s')).sort()).toEqual(['K6s', 'K7s']);
		expect(Object.keys(parseWeighted('TT+')).sort()).toEqual(['AA', 'JJ', 'KK', 'QQ', 'TT']);
		expect(comboCount(parseWeighted('AsAc, KdKh, KdKs, KdKc, QQ-JJ, AQs+, KQs'))).toBe(
			1 + 3 + 12 + 8 + 4
		);
		for (const bad of ['AKx', 'KA+', 'A5s-K5s', 'JTs-97s', 'AsAs', '1+', 'AQs-AQo']) {
			expect(() => parseWeighted(bad), bad).toThrow();
		}
	});
});

describe('the transcribed spots', () => {
	it('facing an open: every list matches the printed combo counts', () => {
		for (const [key, stated] of Object.entries(STATED_OPEN)) {
			const [hero, opener] = key.split('|');
			const spot = vsOpen(hero, opener)!;
			expect(
				[comboCount(spot.call), comboCount(spot.bluff3), comboCount(spot.value3)],
				key
			).toEqual(stated);
		}
	});

	it('facing a 3-bet: every list matches the printed combo counts', () => {
		for (const [key, stated] of Object.entries(STATED_3BET)) {
			const [hero, bettor] = key.split('|');
			const spot = vs3bet(hero, bettor)!;
			expect(
				[comboCount(spot.call), comboCount(spot.bluff4), comboCount(spot.value4)],
				key
			).toEqual(stated);
		}
	});

	it('never gives a hand more than all of its combos', () => {
		for (const key of Object.keys(STATED_OPEN)) {
			const [hero, opener] = key.split('|');
			const s = vsOpen(hero, opener)!;
			for (const h of new Set([
				...Object.keys(s.call),
				...Object.keys(s.bluff3),
				...Object.keys(s.value3)
			])) {
				expect(
					(s.call[h] ?? 0) + (s.bluff3[h] ?? 0) + (s.value3[h] ?? 0),
					`${key} ${h}`
				).toBeLessThanOrEqual(1 + 1e-9);
			}
		}
		for (const key of Object.keys(STATED_3BET)) {
			const [hero, bettor] = key.split('|');
			const s = vs3bet(hero, bettor)!;
			for (const h of new Set([
				...Object.keys(s.call),
				...Object.keys(s.bluff4),
				...Object.keys(s.value4)
			])) {
				expect(
					(s.call[h] ?? 0) + (s.bluff4[h] ?? 0) + (s.value4[h] ?? 0),
					`${key} ${h}`
				).toBeLessThanOrEqual(1 + 1e-9);
			}
		}
	});

	it('has a chart for every valid pair of positions', () => {
		POSITIONS.forEach((opener, i) => {
			for (const hero of POSITIONS.slice(i + 1)) {
				expect(vsOpen(hero, opener), `${hero} vs ${opener}`).toBeDefined();
				expect(vs3bet(opener, hero), `${opener} vs ${hero}`).toBeDefined();
			}
		});
		expect(vsOpen('UTG', 'MP')).toBeUndefined(); // cannot act after a later seat's open
	});

	it('looks right on a few hands', () => {
		const btnVsCo = vsOpen('BTN', 'CO')!;
		expect(btnVsCo.value3.QQ).toBe(1);
		expect(btnVsCo.value3.AKs).toBe(1);
		expect(btnVsCo.call['99']).toBe(1);
		expect(btnVsCo.call.AA).toBeCloseTo(1 / 6);
		expect(btnVsCo.value3.AA).toBeCloseTo(5 / 6);
		expect(btnVsCo.bluff3.A5s).toBe(1);
		expect(btnVsCo.call['72o']).toBeUndefined();
		const bbVsBtn = vsOpen('BB', 'BTN')!;
		expect(comboCount(bbVsBtn.call) + comboCount(bbVsBtn.bluff3) + comboCount(bbVsBtn.value3)).toBe(
			766
		);
	});
});

describe('seats and sizes', () => {
	it('maps players behind to a 6-max seat', () => {
		expect([0, 1, 2, 3, 4, 5, 6].map((b) => positionClass(b, true))).toEqual([
			'BB',
			'SB',
			'BTN',
			'CO',
			'MP',
			'UTG',
			'UTG'
		]);
		expect(positionClass(6)).toBe('MP'); // an early seat facing a raise
		expect(positionClass(5)).toBe('MP');
		expect(positionClass(4)).toBe('MP');
	});

	it('shrinks the calling range against bigger opens', () => {
		expect(mdf(2.5)).toBeCloseTo(0.375);
		expect(sizeFactor('CO', 2.5)).toBeCloseTo(1);
		expect(sizeFactor('CO', 5)).toBeLessThan(sizeFactor('CO', 3));
		expect(sizeFactor('CO', 3)).toBeLessThan(1);
		expect(sizeFactor('BTN', 2)).toBeCloseTo(1);
		expect(sizeFactor('UTG', 2)).toBeGreaterThan(1);
		expect(sizeFactor('UTG', 50)).toBeGreaterThanOrEqual(0.4);
		expect(sizeFactor('UTG', 50)).toBeLessThanOrEqual(1.3);
	});
});

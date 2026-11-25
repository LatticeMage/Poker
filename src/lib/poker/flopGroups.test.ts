import { describe, expect, it } from 'vitest';
import { FLOP_GROUPS, flopGroup, flopKey, groupLevels } from './flopGroups';
import { parseCompact } from './cards';

const allFlops = (): number[][] => {
	const out: number[][] = [];
	for (let a = 0; a < 52; a++)
		for (let b = a + 1; b < 52; b++) for (let c = b + 1; c < 52; c++) out.push([a, b, c]);
	return out;
};

describe('flop groups', () => {
	it('has 383 groups covering the 22,100 flops, refining the 73 structure groups', () => {
		expect(FLOP_GROUPS).toHaveLength(383);
		expect(new Set(FLOP_GROUPS.map((g) => g.base)).size).toBe(73);
		expect(FLOP_GROUPS.reduce((s, g) => s + g.flops, 0)).toBe(22100);
	});

	it('puts every one of the 22,100 flops in the group whose size says so', () => {
		const counted = new Array(384).fill(0);
		for (const flop of allFlops()) counted[flopGroup(flop).id]++;
		FLOP_GROUPS.forEach((g) => expect(counted[g.id]).toBe(g.flops));
	});

	it('does not care how the suits are named or in which order the cards come', () => {
		expect(flopKey(parseCompact('AsKsQs'))).toBe(flopKey(parseCompact('QhAhKh')));
		expect(flopKey(parseCompact('AsKh7d'))).not.toBe(flopKey(parseCompact('AsKs7d')));
	});

	it('names the groups by their characteristics', () => {
		expect(flopGroup(parseCompact('AsKsQs')).name).toBe('單色｜3高｜高三連｜同順｜AKQ');
		expect(flopGroup(parseCompact('Kh7d2c')).name).toBe('三花｜1高｜低斷｜不同順｜K｜中小');
		// same structure, another high card: another group, same base
		expect(flopGroup(parseCompact('Ah7d2c')).name).toBe('三花｜1高｜低斷｜不同順｜A｜中小');
		expect(flopGroup(parseCompact('Ah7d2c')).base).toBe(flopGroup(parseCompact('Kh7d2c')).base);
		expect(groupLevels(flopGroup(parseCompact('9h9d8s')))[0]).toBe('三花');
	});

	it('counts a monotone connected flop wetter than a dry rainbow one', () => {
		expect(flopGroup(parseCompact('9s8s7s')).wetness).toBeGreaterThan(0.5);
		expect(flopGroup(parseCompact('Kh7d2c')).wetness).toBeLessThan(0.1);
	});
});

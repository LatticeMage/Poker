import { describe, expect, it } from 'vitest';
import { parseCards } from './cards';
import {
	FLUSH,
	FULL_HOUSE,
	HIGH_CARD,
	PAIR,
	QUADS,
	STRAIGHT,
	STRAIGHT_FLUSH,
	TRIPS,
	TWO_PAIR,
	categoryOf,
	evaluate,
	handScore
} from './evaluate';

const hand = (text: string) => evaluate(parseCards(text));

describe('hand evaluation', () => {
	it('recognises every category', () => {
		const cases: [string, number][] = [
			['As Ks Qs Js Ts 2d 3c', STRAIGHT_FLUSH],
			['9s 9h 9d 9c Ks 2d 3c', QUADS],
			['9s 9h 9d Kc Ks 2d 3c', FULL_HOUSE],
			['As 9s 7s 4s 2s Kd Kc', FLUSH],
			['9s 8h 7d 6c 5s Kd Kc', STRAIGHT],
			['9s 9h 9d 6c 5s Kd 2c', TRIPS],
			['9s 9h 6d 6c 5s Kd 2c', TWO_PAIR],
			['9s 9h 7d 6c 4s Kd 2c', PAIR],
			['9s Jh 7d 6c 4s Kd 2c', HIGH_CARD]
		];
		for (const [cards, category] of cases) expect(hand(cards).category, cards).toBe(category);
	});

	it('names hands', () => {
		expect(hand('As Ks Qs Js Ts').name).toBe('Royal Flush');
		expect(hand('9s 9h 9d Kc Ks').name).toBe('Full House, Nines over Kings');
		expect(hand('6s 6h Kd Kc 2s').name).toBe('Two Pair, Kings and Sixes');
		expect(hand('5s 4h 3d 2c As Kd Kc').name).toBe('Straight, Five high');
	});

	it('plays the ace low in a wheel, below a six-high straight', () => {
		expect(hand('As 2h 3d 4c 5s').score).toBeLessThan(hand('2h 3d 4c 5s 6h').score);
		expect(hand('As 2h 3d 4c 5s').score).toBeGreaterThan(hand('As Ah Ad Kc Qs').score);
	});

	it('picks the best five from seven', () => {
		expect(hand('9s 9h 9d 4c 4s 4d Kc').name).toBe('Full House, Nines over Fours');
		expect(hand('9s 8s 7s 6h 5s 2s Kd').category).toBe(FLUSH);
		expect(hand('As Ah Kd 7c 4s 3d 2c').score).toBeGreaterThan(hand('Ad Ac Qd 7h 4h 3s 2h').score);
		// Suits never break ties.
		expect(handScore(parseCards('As Kh Qd Jc 9s'))).toBe(handScore(parseCards('Ad Kc Qh Js 9h')));
	});

	it('returns the five cards that make the hand', () => {
		const sorted = (text: string) => [...parseCards(text)].sort((a, b) => a - b);
		expect([...hand('Ks Kh 7d 7c 2s Ad 3c').cards].sort((a, b) => a - b)).toEqual(
			sorted('Ks Kh 7d 7c Ad')
		);
		expect([...hand('5s 4h 3d 2c As Kd Kc').cards].sort((a, b) => a - b)).toEqual(
			sorted('5s 4h 3d 2c As')
		);
	});

	it('scores boards of any length, and reads the category back', () => {
		expect(categoryOf(handScore(parseCards('Kh Ks')))).toBe(PAIR);
		expect(categoryOf(handScore(parseCards('As 7d 2c Kd')))).toBe(HIGH_CARD);
		expect(categoryOf(handScore(parseCards('As 7d 2c Kd Kc')))).toBe(PAIR);
	});
});

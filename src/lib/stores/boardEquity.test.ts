import { beforeEach, describe, expect, it } from 'vitest';
import { makeCard } from '$lib/utils/cards';
import { boardInputs as inputs } from './boardEquity.svelte';

const As = makeCard(14, 0);
const Kh = makeCard(13, 1);
const Qd = makeCard(12, 2);
const Jc = makeCard(11, 3);
const Ts = makeCard(10, 0);
const NineH = makeCard(9, 1);
const TwoC = makeCard(2, 3);

describe('board-equity inputs', () => {
	beforeEach(() => {
		inputs.pot = 'srp';
		inputs.opener = 'BTN';
		inputs.other = 'BB';
		inputs.side = 'other';
		inputs.bet = null;
		inputs.board = [As, Kh, Qd, Jc, null];
		inputs.hand = [Ts, NineH];
	});

	it('picking the seats, the pot, the side or the bet leaves the board and the hand alone', () => {
		inputs.opener = 'CO';
		inputs.other = 'SB';
		inputs.pot = '3bet';
		inputs.side = 'opener';
		inputs.bet = 0.5;
		expect([...inputs.board]).toEqual([As, Kh, Qd, Jc, null]);
		expect([...inputs.hand]).toEqual([Ts, NineH]);
	});

	it('changing a board card leaves the hand and the seats alone, and the other way round', () => {
		inputs.opener = 'CO';
		inputs.setSlot(4, TwoC); // the river
		expect([...inputs.board]).toEqual([As, Kh, Qd, Jc, TwoC]);
		expect([...inputs.hand]).toEqual([Ts, NineH]);
		inputs.setSlot(5, null); // the first hand card
		expect([...inputs.hand]).toEqual([null, NineH]);
		expect([...inputs.board]).toEqual([As, Kh, Qd, Jc, TwoC]);
		expect(inputs.opener).toBe('CO');
		expect(inputs.other).toBe('BB');
	});

	it('a new flop clears the turn and river, and takes from the hand only a card now on the board', () => {
		inputs.setFlop([TwoC, Qd, Kh]);
		expect([...inputs.board]).toEqual([TwoC, Qd, Kh, null, null]);
		expect([...inputs.hand]).toEqual([Ts, NineH]);
		inputs.setFlop([Ts, Qd, Kh]);
		expect([...inputs.hand]).toEqual([null, NineH]);
	});

	it('a random flop avoids the hand and does not change it', () => {
		for (let i = 0; i < 50; i++) {
			inputs.randomFlop();
			const flop = inputs.board.slice(0, 3);
			expect(new Set(flop).size).toBe(3);
			expect(flop).not.toContain(Ts);
			expect(flop).not.toContain(NineH);
		}
		expect([...inputs.hand]).toEqual([Ts, NineH]);
	});

	it('puts the inputs together only when asked: cards in play, hole cards, picker slots', () => {
		expect(inputs.inPlay).toEqual([As, Kh, Qd, Jc]);
		inputs.setSlot(3, null); // no turn: a river alone would not count
		inputs.setSlot(4, TwoC);
		expect(inputs.inPlay).toEqual([As, Kh, Qd]);
		expect(inputs.holeCards).toEqual([Ts, NineH]);
		expect(inputs.slots).toEqual([As, Kh, Qd, null, TwoC, Ts, NineH]);
		inputs.clearHand();
		expect(inputs.holeCards).toBeNull();
		expect(inputs.hasHand).toBe(false);
		expect([...inputs.board]).toEqual([As, Kh, Qd, null, TwoC]);
	});
});

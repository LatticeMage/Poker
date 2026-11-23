import { describe, expect, it } from 'vitest';
import { parseCard, parseCards } from './cards';
import {
	HANDS_PER_LEVEL,
	advance,
	applyAction,
	blindsForHand,
	computePots,
	legalActions,
	startHand
} from './engine';
import { mulberry32 } from './rng';
import { CALL, CHECK, FOLD, newGame, raise, totalChips } from './testing';

describe('betting rules', () => {
	it('posts blinds and starts left of the big blind', () => {
		const g = newGame(4);
		startHand(g, mulberry32(2));
		const n = g.players.length;
		expect(g.sb_seat).toBe((g.dealer + 1) % n);
		expect(g.bb_seat).toBe((g.dealer + 2) % n);
		expect(g.to_act).toBe((g.dealer + 3) % n);
		expect(g.players[g.sb_seat].bet).toBe(10);
		expect(g.players[g.bb_seat].bet).toBe(20);
		expect(g.pot).toBe(30);
	});

	it('heads-up: the button is the small blind and acts first preflop, last after', () => {
		const g = newGame(2);
		startHand(g, mulberry32(2));
		expect(g.sb_seat).toBe(g.dealer);
		expect(g.to_act).toBe(g.dealer);
		applyAction(g, CALL);
		expect(g.to_act).toBe(g.bb_seat); // the big blind keeps its option
		applyAction(g, CHECK);
		expect(g.phase).toBe('dealing');
		advance(g);
		expect(g.street).toBe('flop');
		expect(g.to_act).toBe(g.bb_seat);
	});

	it('enforces the minimum raise', () => {
		const g = newGame(3);
		startHand(g, mulberry32(2));
		expect(legalActions(g).min_to).toBe(40);
		applyAction(g, raise(100)); // a raise of 80
		const legal = legalActions(g);
		expect(legal.to_call).toBe(90);
		expect(legal.min_to).toBe(180);
		applyAction(g, raise(120)); // too small: bumped to the minimum
		expect(g.current_bet).toBe(180);
	});

	it('does not let a short all-in reopen the betting', () => {
		const g = newGame(3);
		g.players[2].chips = 130;
		g.dealer = 2; // next hand: button seat 0, SB seat 1, BB seat 2
		startHand(g, mulberry32(2));
		expect(g.to_act).toBe(0);
		applyAction(g, raise(100)); // seat 0 raises by 80
		applyAction(g, CALL); // seat 1 calls
		applyAction(g, raise(130)); // seat 2 is all-in for 30 more
		expect(g.players[2].all_in).toBe(true);
		const legal = legalActions(g);
		expect(legal.seat).toBe(0);
		expect(legal.to_call).toBe(30);
		expect(legal.can_raise).toBe(false);
	});

	it('returns an uncalled bet and awards the pot when everyone folds', () => {
		const g = newGame(3);
		startHand(g, mulberry32(2));
		const raiser = g.to_act;
		applyAction(g, raise(200));
		applyAction(g, FOLD);
		applyAction(g, FOLD);
		expect(g.phase).toBe('hand_over');
		expect(g.players[raiser].chips).toBe(1030);
		expect(g.players[raiser].won).toBe(50); // own 20 matched by the big blind, plus the blinds
		expect(totalChips(g)).toBe(3000);
	});

	it('falls back to a legal action when the request is illegal', () => {
		const g = newGame(3);
		startHand(g, mulberry32(2));
		const first = g.to_act;
		applyAction(g, CHECK); // cannot check facing the big blind: folds
		expect(g.players[first].folded).toBe(true);
		applyAction(g, CALL);
		const bb = g.to_act;
		applyAction(g, FOLD); // the big blind can check: never folds for free
		expect(g.players[bb].folded).toBe(false);
	});

	it('builds side pots from all-in levels', () => {
		const g = newGame(4);
		for (const p of g.players) p.in_hand = true;
		[100, 300, 300, 50].forEach((c, i) => (g.players[i].committed = c));
		g.players[3].folded = true;
		expect(computePots(g.players)).toEqual([
			{ amount: 350, eligible: [0, 1, 2] },
			{ amount: 400, eligible: [1, 2] }
		]);
	});

	it('pays main and side pots to the right players', () => {
		const g = newGame(3);
		g.players[1].chips = 200;
		g.dealer = 2; // button seat 0 (you), SB seat 1, BB seat 2
		// You hold the nut flush, the short stack trips, the big stack two pair.
		startHand(g, mulberry32(5), parseCards('As Ks'), parseCards('Qs 7s 2s 9d 9c'));
		g.players[1].hole = parseCards('9h 8h');
		g.players[2].hole = parseCards('Qd Jc');
		applyAction(g, raise(1000));
		applyAction(g, CALL);
		applyAction(g, CALL);
		expect(g.phase).toBe('dealing');
		expect(g.players.every((p) => p.revealed)).toBe(true);
		while (g.phase === 'dealing') advance(g);
		expect(g.phase).toBe('game_over');
		expect(g.players[0].chips).toBe(2200);
		expect(g.players[0].place).toBe(1);
		// The bigger starting stack finishes ahead when two players bust together.
		expect(g.players[2].place).toBe(2);
		expect(g.players[1].place).toBe(3);
	});

	it('gives the odd chip of a split pot to the winner left of the button', () => {
		const g = newGame(3);
		g.dealer = 2; // button seat 0, SB seat 1, BB seat 2
		startHand(g, mulberry32(3), parseCards('2c 3d'), parseCards('As Ks Qs Js Ts'));
		applyAction(g, FOLD); // you fold; the board plays for the two blinds
		applyAction(g, CALL);
		applyAction(g, CHECK);
		g.players[1].committed += 1; // make the pot odd: 41
		g.players[1].chips -= 1;
		while (g.phase === 'dealing') advance(g);
		while ((g.phase as string) === 'betting') {
			applyAction(g, CHECK);
			while ((g.phase as string) === 'dealing') advance(g);
		}
		expect(g.players[1].won + g.players[2].won).toBe(41);
		expect(g.players[1].won).toBe(21); // seat 1 is first to the left of the button
	});

	it('deals the cards that were picked, never twice', () => {
		const g = newGame(5);
		const hole = parseCards('7h 2c');
		const flop = parseCards('Ah Kh Qh');
		startHand(g, mulberry32(9), hole, [...flop, null, null]);
		expect(g.players[0].hole).toEqual(hole);
		expect(g.runout.slice(0, 3)).toEqual(flop);
		const dealt = [...g.players.flatMap((p) => p.hole), ...g.runout];
		expect(new Set(dealt).size).toBe(dealt.length);

		const dupes = newGame(3);
		const ace = parseCard('As');
		startHand(dupes, mulberry32(9), [ace, ace], [ace, null, null, null, null]);
		const all = [...dupes.players.flatMap((p) => p.hole), ...dupes.runout];
		expect(dupes.players[0].hole[0]).toBe(ace);
		expect(new Set(all).size).toBe(all.length);
	});

	it('follows the blind schedule and tells clients the next level', () => {
		expect(blindsForHand(1)).toEqual({ small: 10, big: 20 });
		expect(blindsForHand(HANDS_PER_LEVEL)).toEqual({ small: 10, big: 20 });
		expect(blindsForHand(HANDS_PER_LEVEL + 1)).toEqual({ small: 15, big: 30 });
		const g = newGame(3);
		startHand(g, mulberry32(1));
		expect([g.hands_per_level, g.next_big_blind]).toEqual([HANDS_PER_LEVEL, 20]);
		g.hand_number = HANDS_PER_LEVEL - 1; // the next deal is hand 6, the last at 10/20
		g.phase = 'hand_over';
		startHand(g, mulberry32(1));
		expect([g.big_blind, g.next_big_blind]).toEqual([20, 30]);
	});
});

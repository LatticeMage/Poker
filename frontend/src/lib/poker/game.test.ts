import { describe, expect, it } from 'vitest';
import type { Action, GameReply, GameState } from '$lib/types/game';
import { parseCards } from './cards';
import { InvalidState, NotYourTurn, act, checkGame, newGame, nextHand } from './game';
import { mulberry32 } from './rng';

const passive = (legal: NonNullable<GameReply['legal']>): Action =>
	legal.can_check ? { type: 'check' } : { type: 'call' };

/** Keep answering with `choose(legal)` until the hand (or game) is over. */
function playHand(reply: GameReply, choose = passive, seed = 1): GameReply {
	const rng = mulberry32(seed);
	while (reply.legal) reply = act(reply.state, choose(reply.legal), rng);
	return reply;
}

describe('starting a game', () => {
	it('returns the whole game to keep', () => {
		const reply = newGame({ players: 6, chips: 2000 }, mulberry32(1));
		const state = reply.state;
		expect(state.players).toHaveLength(6);
		expect(state.hand_number).toBe(1);
		expect(state.hands_per_level).toBe(6);
		expect(state.next_big_blind).toBe(20); // the blinds first go up at hand 7
		expect(reply.steps[reply.steps.length - 1].state).toEqual(state);
		expect(state.players.reduce((s, p) => s + p.chips + p.bet, 0)).toBe(12000);
		// The state is self-contained: every hole card and the whole runout (the UI hides them).
		expect(state.players.every((p) => p.hole.length === 2)).toBe(true);
		expect(state.runout).toHaveLength(5);
		expect(state.board).toEqual([]);
		expect(reply.legal !== null).toBe(state.phase === 'betting');
		if (reply.legal) expect(state.to_act).toBe(reply.legal.seat);
	});

	it('is reproducible with a seed', () => {
		expect(newGame({ players: 5, chips: 1000 }, mulberry32(11))).toEqual(
			newGame({ players: 5, chips: 1000 }, mulberry32(11))
		);
		expect(newGame({ players: 5, chips: 1000 }, mulberry32(12)).state).not.toEqual(
			newGame({ players: 5, chips: 1000 }, mulberry32(11)).state
		);
	});

	it('deals picked cards', () => {
		const hole = parseCards('As Ah');
		const flop = parseCards('Ad Ac Kh');
		const reply = newGame({ players: 2, chips: 1000, picks: { hole, board: flop } }, mulberry32(1));
		expect(reply.state.players[0].hole).toEqual(hole);
		expect(reply.state.runout.slice(0, 3)).toEqual(flop);
	});
});

describe('playing', () => {
	it('keeps nothing between calls and never changes its input', () => {
		const reply = newGame({ players: 4, chips: 1000 }, mulberry32(3));
		const before = structuredClone(reply.state);
		const move: Action = reply.legal!.can_call ? { type: 'call' } : { type: 'check' };
		const first = act(reply.state, move, mulberry32(1));
		expect(reply.state).toEqual(before);
		// The same state can be submitted again: nothing was consumed.
		const again = act(reply.state, move, mulberry32(1));
		expect(again).toEqual(first);
		expect(again.state.hand_number).toBe(reply.state.hand_number);
	});

	it('tells the human their hand once there is a board', () => {
		let reply = newGame({ players: 3, chips: 1000 }, mulberry32(5));
		const names: string[] = [];
		const rng = mulberry32(1);
		while (reply.legal) {
			reply = act(reply.state, passive(reply.legal), rng);
			const s = reply.state;
			if (s.board.length >= 3 && !s.players[0].folded) names.push(s.players[0].hand_name);
		}
		expect(names.length).toBeGreaterThan(0);
		expect(names.every((n) => n !== '')).toBe(true);
	});

	it('refuses requests for the wrong moment', () => {
		const reply = newGame({ players: 3, chips: 1000 }, mulberry32(2));
		expect(reply.state.phase).toBe('betting');
		expect(() => nextHand(reply.state)).toThrow(NotYourTurn); // dealing while a hand is on
		const done = playHand(reply).state;
		if (done.phase === 'hand_over') expect(() => act(done, { type: 'check' })).toThrow(NotYourTurn);
	});

	it('rejects tampered or broken states', () => {
		const state = newGame({ players: 4, chips: 1000 }, mulberry32(4)).state;
		const bad = (mutate: (s: GameState) => void) => {
			const broken = structuredClone(state);
			mutate(broken);
			expect(() => act(broken, { type: 'call' })).toThrow(InvalidState);
		};
		bad((s) => (s.runout[0] = s.players[1].hole[0])); // a card dealt twice
		bad((s) => (s.players[1].seat = 3)); // seats out of order
		bad((s) => (s.to_act = 9)); // nobody there
		bad((s) => (s.players[0].is_human = false)); // no human
		bad((s) => (s.board = [s.runout[4]])); // board is not a prefix of the runout
		bad((s) => (s.players[1].hole = [99, 98])); // not cards
		bad((s) => (s.players[1].bet = s.players[1].committed + 5)); // a bet larger than what was put in
		expect(() => checkGame(structuredClone(state))).not.toThrow();
	});
});

describe('a whole game', () => {
	it('conserves chips, deals every hand and ranks everyone', () => {
		const rng = mulberry32(4);
		let reply = newGame({ players: 5, chips: 1000 }, mulberry32(9));
		const bank = 5000;
		const choose = (legal: NonNullable<GameReply['legal']>): Action => {
			const roll = rng();
			if (legal.can_raise && roll < 0.25)
				return { type: 'raise', amount: rng() < 0.5 ? legal.min_to : legal.max_to };
			if (legal.can_call && roll > 0.85) return { type: 'fold' };
			return passive(legal);
		};
		let hands = 0;
		for (;;) {
			reply = playHand(reply, choose, 7 + hands);
			for (const s of reply.steps) {
				expect(s.state.players.reduce((sum, p) => sum + p.chips, 0) + s.state.pot).toBe(bank);
			}
			if (++hands > 2000) throw new Error('the game did not finish');
			if (reply.state.phase === 'game_over') break;
			expect(reply.state.phase).toBe('hand_over');
			reply = nextHand(reply.state, undefined, mulberry32(hands));
		}
		const players = reply.state.players;
		expect(players.map((p) => p.place).sort()).toEqual([1, 2, 3, 4, 5]);
		expect(players.find((p) => p.place === 1)!.chips).toBe(bank);
	});
});

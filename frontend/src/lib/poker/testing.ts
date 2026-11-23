// Helpers shared by the tests.

import type { Action, GameState, Persona } from '$lib/types/game';
import { PERSONAS, preflopOrder } from './bot';
import { parseCards } from './cards';
import { advance, applyAction, createGame, startHand } from './engine';
import { type Rng, mulberry32 } from './rng';

export const BALANCED: Persona[] = Array.from({ length: 9 }, (_, i) => ({
	name: `B${i}`,
	style: 'balanced',
	tight: 0.5,
	aggro: 0.5,
	bluff: 0.25
}));

export const newGame = (
	players: number,
	chips = 1000,
	seed = 1,
	bots: readonly Persona[] = PERSONAS
) => createGame({ players, chips, bots }, mulberry32(seed));

export const totalChips = (g: GameState) =>
	g.players.reduce((sum, p) => sum + p.chips + p.committed, 0);

export const FOLD: Action = { type: 'fold' };
export const CHECK: Action = { type: 'check' };
export const CALL: Action = { type: 'call' };
export const raise = (amount: number): Action => ({ type: 'raise', amount });

/** Positions at a six-player table, in preflop order. */
export const [UTG, HJ, CO, BTN, SB, BB] = [0, 1, 2, 3, 4, 5];

/**
 * A six-player hand played through `script` ([position, action] pairs, in preflop order) up to the
 * flop. `boardText` may name the turn and river too: the game is then moved to that street.
 */
export function toFlop(
	script: [number, Action][],
	boardText = 'As 7d 2c',
	seed = 3,
	personas: readonly Persona[] = BALANCED
): { g: GameState; order: number[]; rng: Rng } {
	const rng = mulberry32(seed);
	const g = createGame({ players: 6, chips: 2000, bots: personas }, rng);
	startHand(g, rng);
	const order = preflopOrder(g);
	for (const [position, action] of script) {
		if (g.to_act !== order[position]) throw new Error(`position ${position} is not next to act`);
		applyAction(g, action);
	}
	advance(g);
	const board = parseCards(boardText);
	const rest = Array.from({ length: 52 }, (_, i) => i).filter((c) => !board.includes(c));
	g.runout = [...board, ...rest.slice(0, 5 - board.length)];
	g.board = board;
	g.street = ({ 3: 'flop', 4: 'turn', 5: 'river' } as const)[board.length as 3 | 4 | 5];
	return { g, order, rng };
}

export const BTN_VS_BB: [number, Action][] = [
	[UTG, FOLD],
	[HJ, FOLD],
	[CO, FOLD],
	[BTN, raise(40)],
	[SB, FOLD],
	[BB, CALL]
];

export const CO_3BET_BTN: [number, Action][] = [
	[UTG, FOLD],
	[HJ, FOLD],
	[CO, raise(50)],
	[BTN, raise(160)],
	[SB, FOLD],
	[BB, FOLD],
	[CO, CALL]
];

// Game service: pure functions from a game state to the next one. Nothing is stored here.
//
// Each call advances the game until the human has to decide (or the hand ends) and returns the
// intermediate *steps*, so the UI can replay what happened at its own pace. The state is the plain
// JSON `GameState` the store keeps; every function works on a copy and never touches its input.

import type { Action, GameEvent, GameReply, GameState, Picks, Step } from '$lib/types/game';
import { isCard } from '$lib/utils/cards';
import { PERSONAS, decide } from './bot';
import { type Game, advance, applyAction, createGame, legalActions, startHand } from './engine';
import { type Rng, shuffle, systemRng } from './rng';

/** The submitted game state is not one the engine could have produced (e.g. a damaged save). */
export class InvalidState extends Error {}

/** The request does not fit the moment (acting when it is not your turn, dealing mid-hand). */
export class NotYourTurn extends Error {}

/** Reject states that would crash or confuse the engine (it trusts its own invariants). */
export function checkGame(g: GameState): void {
	const n = g.players.length;
	if (n < 2 || n > 9) throw new InvalidState('Between 2 and 9 players');
	if (g.players.some((p, i) => p.seat !== i))
		throw new InvalidState('Seats must be numbered 0..n-1 in order');
	if (g.players.filter((p) => p.is_human).length !== 1 || !g.players[0].is_human) {
		throw new InvalidState('Exactly one human, in seat 0');
	}
	if (!(g.dealer >= 0 && g.dealer < n)) throw new InvalidState('Dealer seat out of range');
	const cards = [...g.players.filter((p) => p.in_hand).flatMap((p) => p.hole), ...g.runout];
	if (!cards.every(isCard) || new Set(cards).size !== cards.length) {
		throw new InvalidState('Cards must be distinct');
	}
	if (g.hand_number > 0 && g.phase !== 'idle') {
		const live = g.players.filter((p) => p.in_hand);
		const board = g.board;
		if (
			live.some((p) => p.hole.length !== 2) ||
			g.runout.length !== 5 ||
			board.some((c, i) => c !== g.runout[i])
		) {
			throw new InvalidState('Inconsistent hole cards or board');
		}
		if (!(g.sb_seat >= 0 && g.sb_seat < n) || !(g.bb_seat >= 0 && g.bb_seat < n)) {
			throw new InvalidState('Blind seats out of range');
		}
	}
	if (g.phase === 'betting') {
		const p = g.players[g.to_act];
		if (!p || !p.in_hand || p.folded) throw new InvalidState('No valid player to act');
		if (p.all_in || p.chips <= 0) throw new InvalidState('The player to act cannot bet');
	}
	if (g.raisers.some((seat) => !(seat >= 0 && seat < n)))
		throw new InvalidState('Raiser seat out of range');
	if (g.players.some((p) => p.bet > p.committed))
		throw new InvalidState('A bet cannot exceed what the player put in');
}

/** The phase, read afresh (TypeScript cannot see that advance() and applyAction() change it). */
const phaseOf = (g: Game): Game['phase'] => g.phase;

function step(g: Game, kind: Step['kind'], events: GameEvent[]): Step {
	return { kind, events, state: structuredClone(g) };
}

/** Run the robots and the dealer until the human must act or the hand ends. */
function playUntilHuman(g: Game, rng: Rng): Step[] {
	const steps: Step[] = [];
	while (g.phase === 'betting' || g.phase === 'dealing') {
		let events: GameEvent[];
		let kind: Step['kind'];
		if (g.phase === 'dealing') {
			events = advance(g);
			const after = phaseOf(g); // advance() moved it: the hand may have ended
			kind = after === 'hand_over' || after === 'game_over' ? 'showdown' : 'street';
		} else {
			const player = g.players[g.to_act];
			if (player.is_human) break;
			events = applyAction(g, decide(g, player.seat, rng));
			kind = 'action';
		}
		steps.push(step(g, kind, events));
	}
	return steps;
}

function reply(g: Game, steps: Step[]): GameReply {
	return {
		state: structuredClone(g),
		steps,
		legal: g.phase === 'betting' ? legalActions(g) : null
	};
}

/** Start a game and deal the first hand. */
export function newGame(
	config: { players: number; chips: number; picks?: Picks },
	rng: Rng = systemRng
): GameReply {
	const bots = shuffle([...PERSONAS], rng);
	const g = createGame({ players: config.players, chips: config.chips, bots }, rng);
	const intro: GameEvent = {
		kind: 'info',
		text: `New game — ${config.players} players, ${config.chips.toLocaleString('en-US')} chips each`
	};
	const dealt = startHand(g, rng, config.picks?.hole, config.picks?.board);
	const steps = [step(g, 'deal', [intro, ...dealt])];
	steps.push(...playUntilHuman(g, rng));
	return reply(g, steps);
}

/** Make the human's move; the robots then play until it is the human's turn again or the hand ends. */
export function act(state: GameState, action: Action, rng: Rng = systemRng): GameReply {
	const g = structuredClone(state);
	checkGame(g);
	if (g.phase !== 'betting' || !g.players[g.to_act].is_human)
		throw new NotYourTurn('It is not your turn');
	const steps = [step(g, 'action', applyAction(g, action))];
	steps.push(...playUntilHuman(g, rng));
	return reply(g, steps);
}

/** Deal the next hand once the previous one is over. */
export function nextHand(state: GameState, picks?: Picks, rng: Rng = systemRng): GameReply {
	const g = structuredClone(state);
	checkGame(g);
	if (g.phase !== 'hand_over') throw new NotYourTurn('The hand is not over');
	const steps = [step(g, 'deal', startHand(g, rng, picks?.hole, picks?.board))];
	steps.push(...playUntilHuman(g, rng));
	return reply(g, steps);
}

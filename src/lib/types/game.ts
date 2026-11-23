// Shapes of the game API (backend/src/models/game.py).
//
// The API is stateless: the client keeps the whole `GameState` (see stores/game.store.ts) and
// sends it with every request. It includes every robot's hole cards and the undealt board
// (`runout`) - the UI must only draw what a player at the table could see.

import type { Card } from '$lib/utils/cards';

export type Phase = 'idle' | 'betting' | 'dealing' | 'hand_over' | 'game_over';
export type Street = 'preflop' | 'flop' | 'turn' | 'river';

export interface Persona {
	name: string;
	style: string;
	tight: number;
	aggro: number;
	bluff: number;
}

export interface PlayerState {
	seat: number;
	name: string;
	is_human: boolean;
	persona: Persona | null;
	chips: number;
	start_chips: number;
	/** Always filled while dealt in - show it only for you, or once `revealed`. */
	hole: Card[];
	in_hand: boolean;
	folded: boolean;
	all_in: boolean;
	/** Chips in front of the player on the current street. */
	bet: number;
	/** Chips the player put in during this hand. */
	committed: number;
	acted: boolean;
	/** Last action: SB, BB, Fold, Check, Call, Bet, Raise, All-in or empty. */
	badge: string;
	/** Hole cards are face up for everyone. */
	revealed: boolean;
	won: number;
	hand_name: string;
	best_cards: Card[];
	/** Finishing place once the player is out (or 1 for the champion). */
	place: number | null;
}

export interface PotState {
	amount: number;
	eligible: number[];
}

export interface GameState {
	players: PlayerState[];
	dealer: number;
	hand_number: number;
	sb_seat: number;
	bb_seat: number;
	small_blind: number;
	big_blind: number;
	phase: Phase;
	street: Street;
	/** The community cards dealt so far. */
	board: Card[];
	/** All five community cards, dealt or not - never draw the undealt ones. */
	runout: Card[];
	current_bet: number;
	min_raise: number;
	/** Seat whose turn it is, or -1. */
	to_act: number;
	/** Chips in the middle, including bets on the current street. */
	pot: number;
	pots: PotState[];
	/** One-line result of the last finished hand. */
	headline: string;
	/** Seats that raised before the flop, in order: opener, 3-bettor, 4-bettor ... */
	raisers: number[];
	hands_per_level: number;
	next_big_blind: number;
}

export interface GameEvent {
	kind: 'hand' | 'street' | 'action' | 'win' | 'info';
	text: string;
}

/** One thing that happened, with the game as it stood right after. */
export interface Step {
	kind: 'deal' | 'action' | 'street' | 'showdown';
	events: GameEvent[];
	state: GameState;
}

/** Your options when it is your turn. Amounts are "raise to" totals for the street. */
export interface Legal {
	seat: number;
	to_call: number;
	can_check: boolean;
	can_call: boolean;
	can_raise: boolean;
	min_to: number;
	max_to: number;
	stack_to: number;
	/** Nobody has bet on this street yet (a raise is a "bet"). */
	opening: boolean;
	current_bet: number;
	pot: number;
	chips: number;
}

export interface GameReply {
	/** The game after everything in `steps`: keep it, send it with the next request. */
	state: GameState;
	steps: Step[];
	legal: Legal | null;
}

/** Cards fixed for the next deal (practice mode); null = random. */
export interface Picks {
	hole: (Card | null)[];
	board: (Card | null)[];
}

export type Action =
	| { type: 'fold' }
	| { type: 'check' }
	| { type: 'call' }
	| { type: 'raise'; amount: number };

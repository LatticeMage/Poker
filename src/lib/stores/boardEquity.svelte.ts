// The inputs of the board-equity page. Every input is its own piece of state with its own
// localStorage key: changing one never touches another, and each survives a reload or leaving the
// page on its own. They are only put together where something is computed from them (the cards in
// play, the query): see the getters at the bottom and BoardEquity.svelte.

import { browser } from '$app/environment';
import type { BoardQuery } from '$lib/types/ranges';
import { type Card, isCard, makeCard } from '$lib/utils/cards';

export const BOARD_SLOTS = 5;
export const HAND_SLOTS = 2;

type Pot = BoardQuery['pot'];
type Side = BoardQuery['side'];

const KEY = 'robo-holdem:board-equity:';
const POTS: Pot[] = ['srp', '3bet', '4bet'];
const DEFAULT_BOARD: (Card | null)[] = [
	makeCard(14, 0),
	makeCard(7, 2),
	makeCard(2, 3),
	null,
	null
];

function read<T>(name: string, clean: (value: unknown) => T | undefined, fallback: T): T {
	if (!browser) return fallback;
	try {
		const raw = localStorage.getItem(KEY + name);
		return (raw === null ? undefined : clean(JSON.parse(raw))) ?? fallback;
	} catch {
		return fallback; // corrupt or unavailable storage: the default
	}
}

function write(name: string, value: unknown) {
	try {
		localStorage.setItem(KEY + name, JSON.stringify(value));
	} catch {
		// Storage can be unavailable (private mode); the value then lasts for the session.
	}
}

function cleanSlots(size: number) {
	return (value: unknown): (Card | null)[] | undefined => {
		if (!Array.isArray(value)) return undefined;
		// A card can sit in one slot only: a repeat is dropped.
		return Array.from({ length: size }, (_, i) => {
			const card = value[i];
			return isCard(card) && value.indexOf(card) === i ? card : null;
		});
	};
}

const cleanText = (value: unknown) => (typeof value === 'string' && value ? value : undefined);

class BoardEquityInputs {
	/** Single-raised, 3-bet or 4-bet pot. */
	pot = $state<Pot>(read('pot', (v) => POTS.find((p) => p === v), 'srp'));
	/** The seat picked as opener. Whether a chart exists for it is decided where it is used. */
	opener = $state(read('opener', cleanText, 'BTN'));
	/** The seat picked as caller / 3-bettor. */
	other = $state(read('other', cleanText, 'BB'));
	/** Which of the two you are. */
	side = $state<Side>(
		read('side', (v) => (v === 'opener' || v === 'other' ? v : undefined), 'other')
	);
	/** A bet you face, as a fraction of the pot. */
	bet = $state<number | null>(
		read('bet', (v) => (typeof v === 'number' && v > 0 ? v : undefined), null)
	);
	/** Flop 1-3, turn, river. */
	board = $state<(Card | null)[]>(read('board', cleanSlots(BOARD_SLOTS), DEFAULT_BOARD));
	/** Your two cards. */
	hand = $state<(Card | null)[]>(read('hand', cleanSlots(HAND_SLOTS), [null, null]));

	constructor() {
		if (!browser) return;
		// One effect per input: saving one never reads or rewrites another.
		$effect.root(() => {
			$effect(() => write('pot', this.pot));
			$effect(() => write('opener', this.opener));
			$effect(() => write('other', this.other));
			$effect(() => write('side', this.side));
			$effect(() => write('bet', this.bet));
			$effect(() => write('board', [...this.board]));
			$effect(() => write('hand', [...this.hand]));
		});
	}

	// -- computed from the inputs: the only place where they are put together --------------------

	/** The board cards in play: the flop and, in order, the turn and river if they are set. */
	get inPlay(): Card[] {
		const cards: Card[] = [];
		for (const card of this.board) {
			if (card === null) break;
			cards.push(card);
		}
		return cards;
	}

	/** Both of your cards, or null while either is missing. */
	get holeCards(): Card[] | null {
		return this.hand.every((c) => c !== null) ? (this.hand as Card[]) : null;
	}

	get hasHand(): boolean {
		return this.hand.some((c) => c !== null);
	}

	/** Board then hand, for the one card picker that shows both. */
	get slots(): (Card | null)[] {
		return [...this.board, ...this.hand];
	}

	// -- changes: each writes one input ----------------------------------------------------------

	/** Set one slot of the picker (0-4 board, 5-6 hand); null empties it. */
	setSlot(index: number, card: Card | null) {
		if (index < BOARD_SLOTS) this.board[index] = card;
		else this.hand[index - BOARD_SLOTS] = card;
	}

	/**
	 * Put a new flop down (the turn and river are cleared: they belonged to the old flop). Your hand
	 * stays as it is, except a card that is now on the board, which cannot be in both places.
	 */
	setFlop(cards: Card[]) {
		this.board = [...cards.slice(0, 3), null, null];
		if (this.hand.some((c) => c !== null && cards.includes(c))) {
			this.hand = this.hand.map((c) => (c !== null && cards.includes(c) ? null : c));
		}
	}

	/** A random flop from the cards that are not in your hand. */
	randomFlop() {
		const taken = new Set(this.hand.filter((c): c is Card => c !== null));
		const deck = Array.from({ length: 52 }, (_, i) => i).filter((c) => !taken.has(c));
		const picked: Card[] = [];
		while (picked.length < 3) {
			picked.push(...deck.splice(Math.floor(Math.random() * deck.length), 1));
		}
		this.board = [...picked, null, null];
	}

	clearHand() {
		this.hand = [null, null];
	}
}

export const boardInputs = new BoardEquityInputs();

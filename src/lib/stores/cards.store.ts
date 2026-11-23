// The cards picked on the board-equity page: one pool of seven slots (flop, turn, river, then
// your two cards). Everything else - the board, your hand, the cards in play - is derived from it.

import { type Card, makeCard } from '$lib/utils/cards';
import { derived, get, writable } from 'svelte/store';

export const BOARD_SLOTS = 5;
export const HOLE_SLOTS = 2;

export const slots = writable<(Card | null)[]>([
	makeCard(14, 0),
	makeCard(7, 2),
	makeCard(2, 3),
	null,
	null,
	null,
	null
]);

export const board = derived(slots, ($slots) => $slots.slice(0, BOARD_SLOTS));
export const hole = derived(slots, ($slots) => $slots.slice(BOARD_SLOTS, BOARD_SLOTS + HOLE_SLOTS));
/** The board cards in play: the flop and, in order, the turn and river if they are set. */
export const inPlay = derived(board, ($board) => {
	const cards: Card[] = [];
	for (const card of $board) {
		if (card === null) break;
		cards.push(card);
	}
	return cards;
});
/** Both hole cards, or null while either is missing. */
export const holeCards = derived(hole, ($hole) =>
	$hole.every((c) => c !== null) ? ($hole as Card[]) : null
);
export const hasHole = derived(hole, ($hole) => $hole.some((c) => c !== null));

/** Put a new flop down; a card that is now on the board leaves your hand. */
export function setBoard(cards: Card[]) {
	const $hole = get(hole);
	slots.set([
		...cards,
		null,
		null,
		...$hole.map((c) => (c !== null && cards.includes(c) ? null : c))
	]);
}

export function randomFlop() {
	const taken = new Set(get(hole).filter((c): c is Card => c !== null));
	const deck = Array.from({ length: 52 }, (_, i) => i).filter((c) => !taken.has(c));
	const picked: Card[] = [];
	while (picked.length < 3) {
		picked.push(...deck.splice(Math.floor(Math.random() * deck.length), 1));
	}
	slots.set([...picked, null, null, ...get(hole)]);
}

export function clearHole() {
	slots.update(($slots) => [...$slots.slice(0, BOARD_SLOTS), null, null]);
}

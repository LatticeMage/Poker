// Card logic for the engine: parsing, naming, the deck. Display helpers (symbols, colours) live in
// $lib/utils/cards. A card is an int 0..51: rank = (card >> 2) + 2 (2..14, ace high), suit = card & 3.

import { type Card, SUIT_SYMBOLS, makeCard, rankLabel, rankOf, suitOf } from '$lib/utils/cards';

export type { Card };

const RANK_NAMES = [
	'',
	'',
	'Two',
	'Three',
	'Four',
	'Five',
	'Six',
	'Seven',
	'Eight',
	'Nine',
	'Ten',
	'Jack',
	'Queen',
	'King',
	'Ace'
];

export const rankName = (rank: number): string => RANK_NAMES[rank];
export const rankPlural = (rank: number): string => (rank === 6 ? 'Sixes' : `${RANK_NAMES[rank]}s`);

/** "A♠", "T♦" — the text form used in the hand history. */
export const cardText = (card: Card): string =>
	rankLabel(rankOf(card)) + SUIT_SYMBOLS[suitOf(card)];

const CARD_RE = /^(10|[2-9TJQKA])([shdc])$/i;
const RANK_VALUES: Record<string, number> = { A: 14, K: 13, Q: 12, J: 11, T: 10 };

/** Parse shorthand like "As", "Td", "10h", "9c". */
export function parseCard(text: string): Card {
	const match = CARD_RE.exec(text.trim());
	if (!match) throw new Error(`Bad card: ${text}`);
	const r = match[1].toUpperCase();
	return makeCard(RANK_VALUES[r] ?? parseInt(r, 10), 'shdc'.indexOf(match[2].toLowerCase()));
}

/** "As Kh 7d" → cards. */
export const parseCards = (text: string): Card[] =>
	text.trim() === '' ? [] : text.trim().split(/\s+/).map(parseCard);

/** Cards written without separators: "As7d2c" → three cards (every card is two characters). */
export function parseCompact(text: string): Card[] {
	const compact = text.replace(/[\s,]/g, '');
	if (compact.length % 2 !== 0) throw new Error(`Bad cards: ${text}`);
	const cards: Card[] = [];
	for (let i = 0; i < compact.length; i += 2) cards.push(parseCard(compact.slice(i, i + 2)));
	return cards;
}

export const freshDeck = (): Card[] => Array.from({ length: 52 }, (_, i) => i);

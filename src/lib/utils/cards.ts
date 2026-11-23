// Card display helpers. A card is an integer 0..51 (the backend's encoding):
// rank = (card >> 2) + 2 (2..14, ace high), suit = card & 3 (♠ ♥ ♦ ♣).

export type Card = number;

// U+FE0E forces the text glyph so phones do not swap in emoji suits.
export const SUIT_SYMBOLS = ['♠︎', '♥︎', '♦︎', '♣︎'] as const;
export const SUIT_NAMES = ['spades', 'hearts', 'diamonds', 'clubs'] as const;

const RANK_LABELS: Record<number, string> = { 11: 'J', 12: 'Q', 13: 'K', 14: 'A' };
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

export const makeCard = (rank: number, suit: number): Card => (rank - 2) * 4 + suit;
export const rankOf = (card: Card): number => (card >> 2) + 2;
export const suitOf = (card: Card): number => card & 3;
export const isRed = (card: Card): boolean => suitOf(card) === 1 || suitOf(card) === 2;

export const rankLabel = (rank: number): string => RANK_LABELS[rank] ?? String(rank);
export const cardName = (card: Card): string =>
	`${RANK_NAMES[rankOf(card)]} of ${SUIT_NAMES[suitOf(card)]}`;

export const isCard = (value: unknown): value is Card =>
	typeof value === 'number' && Number.isInteger(value) && value >= 0 && value < 52;

/** Two-character code the API takes: "As", "Td", "7h". */
export const cardCode = (card: Card): string => `${'23456789TJQKA'[card >> 2]}${'shdc'[card & 3]}`;

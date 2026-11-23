// Starting-hand helpers: names ("AKs", "T9o", "77"), the 13×13 grid layout, strength percentiles,
// cards for a hand class, and the preflop seat order.

import type { GameState } from '$lib/types/game';
import { type Card, makeCard } from '$lib/utils/cards';

export const RANKS = 'AKQJT98765432';
export const INDEXES = Array.from({ length: 13 }, (_, i) => i);
export const TOTAL_COMBOS = 1326;

const RANK_WORDS: Record<string, [string, string]> = {
	A: ['Ace', 'Aces'],
	K: ['King', 'Kings'],
	Q: ['Queen', 'Queens'],
	J: ['Jack', 'Jacks'],
	T: ['Ten', 'Tens'],
	'9': ['Nine', 'Nines'],
	'8': ['Eight', 'Eights'],
	'7': ['Seven', 'Sevens'],
	'6': ['Six', 'Sixes'],
	'5': ['Five', 'Fives'],
	'4': ['Four', 'Fours'],
	'3': ['Three', 'Threes'],
	'2': ['Two', 'Twos']
};

/** The hand at a grid cell: pairs on the diagonal, suited above it, offsuit below. */
export function handAt(row: number, col: number): string {
	if (row === col) return RANKS[row] + RANKS[col];
	return row < col ? `${RANKS[row]}${RANKS[col]}s` : `${RANKS[col]}${RANKS[row]}o`;
}

export function handCombos(name: string): number {
	if (name.length === 2) return 6;
	return name[2] === 's' ? 4 : 12;
}

/** "Ace-King suited", "Pair of Eights". */
export function handTitle(name: string): string {
	if (name.length === 2) return `Pair of ${RANK_WORDS[name[0]][1]}`;
	return `${RANK_WORDS[name[0]][0]}-${RANK_WORDS[name[1]][0]} ${name[2] === 's' ? 'suited' : 'offsuit'}`;
}

/** Top-X% rank of every hand given the strength order (strongest first): AA ≈ 0.5, the worst ≈ 100. */
export function percentiles(order: string[]): Record<string, number> {
	const out: Record<string, number> = {};
	let running = 0;
	for (const name of order) {
		running += handCombos(name);
		out[name] = (100 * running) / TOTAL_COMBOS;
	}
	return out;
}

/** Class name of two cards (0..51 encoding). */
export function holeToHand(hole: Card[]): string {
	const [a, b] = hole;
	const ra = (a >> 2) + 2;
	const rb = (b >> 2) + 2;
	const hi = Math.max(ra, rb);
	const lo = Math.min(ra, rb);
	const char = (r: number) => RANKS[14 - r];
	if (hi === lo) return char(hi) + char(lo);
	return `${char(hi)}${char(lo)}${(a & 3) === (b & 3) ? 's' : 'o'}`;
}

/** Two cards of a hand class with random suits. */
export function cardsForHand(name: string): [Card, Card] {
	const hi = 14 - RANKS.indexOf(name[0]);
	const lo = 14 - RANKS.indexOf(name[1]);
	const suit = () => Math.floor(Math.random() * 4);
	const first = suit();
	if (name.length === 3 && name[2] === 's') return [makeCard(hi, first), makeCard(lo, first)];
	let second = suit();
	while (second === first) second = suit();
	return [makeCard(hi, first), makeCard(lo, second)];
}

/** A random hand class, weighted by how often it is dealt. */
export function randomHand(): string {
	let pick = Math.floor(Math.random() * TOTAL_COMBOS);
	for (const row of INDEXES) {
		for (const col of INDEXES) {
			const name = handAt(row, col);
			pick -= handCombos(name);
			if (pick < 0) return name;
		}
	}
	return 'AA';
}

/** Seats of the players dealt in, in preflop acting order: first to act ... small blind, big blind. */
export function preflopOrder(g: GameState): number[] {
	const n = g.players.length;
	const order: number[] = [];
	let seat = g.bb_seat;
	for (let i = 0; i < n; i++) {
		seat = (seat + 1) % n;
		if (g.players[seat].in_hand) order.push(seat);
	}
	return order;
}

/** Nearest 6-max seat for a seat with `behind` players still to act (mirrors the backend). */
export function positionClass(behind: number, opener = false): string {
	if (behind >= 5) return opener ? 'UTG' : 'MP';
	return ['BB', 'SB', 'BTN', 'CO', 'MP', 'UTG'][Math.max(0, behind)];
}

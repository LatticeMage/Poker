// The opening-range quiz: the question being asked, your answer and your score. The score is saved
// by the change that makes it (answering, resetting), not by watching it.

import { browser } from '$app/environment';
import { attempt, rangeChart } from '$lib/poker/lookup';
import type { Card } from '$lib/utils/cards';
import { cardsForHand, percentiles, randomHand } from '$lib/utils/hands';

const STORAGE_KEY = 'robo-holdem:quiz';

export interface Question {
	players: number;
	seatName: string;
	seatPct: number;
	source: string;
	label: string;
	hand: string;
	cards: [Card, Card];
	raise: boolean;
	hands: Set<string>;
	/** Strength order of the table size's chart, for the percentile shown with the answer. */
	order: string[] | undefined;
}

interface Score {
	right: number;
	total: number;
	streak: number;
	best: number;
}

const ZERO: Score = { right: 0, total: 0, streak: 0, best: 0 };

function loadScore(): Score {
	if (!browser) return { ...ZERO };
	try {
		const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null');
		if (saved && [saved.right, saved.total, saved.streak, saved.best].every(Number.isFinite)) {
			return { right: saved.right, total: saved.total, streak: saved.streak, best: saved.best };
		}
	} catch {
		// No saved score: start from zero.
	}
	return { ...ZERO };
}

function saveScore(score: Score) {
	try {
		localStorage.setItem(STORAGE_KEY, JSON.stringify(score));
	} catch {
		// Storage unavailable: the score lasts for this visit only.
	}
}

class Quiz {
	/** Table size of the questions: any, or a number of players. */
	size = $state<'any' | number>('any');
	score = $state<Score>(loadScore());
	answer = $state<'raise' | 'fold' | null>(null);
	/** Why there is no question (no chart), if so. */
	error = $state<string | null>(null);
	question = $state.raw<Question | null>(null);

	correct = $derived(
		this.question !== null &&
			this.answer !== null &&
			(this.answer === 'raise') === this.question.raise
	);
	percentile = $derived(
		this.question?.order ? percentiles(this.question.order)[this.question.hand] : undefined
	);

	constructor() {
		this.next();
	}

	setSize(value: string) {
		this.size = value === 'any' ? 'any' : Number(value);
		this.next();
	}

	/** Ask a new question: a random seat at a random (or the chosen) table size. */
	next() {
		this.answer = null;
		const players = this.size === 'any' ? 2 + Math.floor(Math.random() * 8) : this.size;
		const result = attempt(() => rangeChart(players));
		this.error = result.error ?? null;
		const chart = result.value;
		if (!chart) return;
		const seats = chart.seats.filter((s) => s.source !== 'none');
		const seat = seats[Math.floor(Math.random() * seats.length)];
		const hand = randomHand();
		// A solver range mixes some hands: count a hand as a raise when it raises at least half the time.
		const hands =
			seat.source === 'solver' ? seat.hands.filter((h) => seat.mix[h] >= 0.5) : seat.hands;
		this.question = {
			players,
			seatName: seat.name,
			seatPct: seat.pct,
			source: seat.source,
			label: seat.label,
			hand,
			cards: cardsForHand(hand),
			raise: hands.includes(hand),
			hands: new Set(hands),
			order: chart.order
		};
	}

	choose(choice: 'raise' | 'fold') {
		const question = this.question;
		if (!question || this.answer) return;
		this.answer = choice;
		const right = (choice === 'raise') === question.raise;
		const streak = right ? this.score.streak + 1 : 0;
		this.score = {
			right: this.score.right + (right ? 1 : 0),
			total: this.score.total + 1,
			streak,
			best: Math.max(this.score.best, streak)
		};
		saveScore(this.score);
	}

	resetScore() {
		this.score = { ...ZERO };
		saveScore(this.score);
	}
}

export const quiz = new Quiz();

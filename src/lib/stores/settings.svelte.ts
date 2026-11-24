// User settings, persisted to localStorage.

import { browser } from '$app/environment';
import type { Picks } from '$lib/types/game';
import { type Card, isCard } from '$lib/utils/cards';

const STORAGE_KEY = 'robo-holdem:settings';

export const LIMITS = {
	speed: { min: 0, max: 1500, step: 100, fallback: 600 },
	players: { min: 2, max: 9, step: 1, fallback: 6 },
	chips: { min: 1000, max: 20000, step: 1000, fallback: 1000 }
} as const;

function clampInt(value: unknown, limit: { min: number; max: number; fallback: number }): number {
	const n = typeof value === 'number' ? Math.round(value) : NaN;
	return Number.isNaN(n) || n < limit.min || n > limit.max ? limit.fallback : n;
}

function cleanSlots(value: unknown, size: number): (Card | null)[] {
	const list = Array.isArray(value) ? value : [];
	return Array.from({ length: size }, (_, i) => (isCard(list[i]) ? list[i] : null));
}

class Settings {
	// Each setting is a signal; the setter saves the whole record in the same step, so nothing has to
	// watch the state to save it. Slots are replaced, never edited in place, so the setter always runs.
	#speed = $state<number>(LIMITS.speed.fallback);
	#players = $state<number>(LIMITS.players.fallback);
	#chips = $state<number>(LIMITS.chips.fallback);
	#autoDeal = $state(true);
	#hints = $state(false);
	#pickedHole = $state<(Card | null)[]>([null, null]);
	#pickedBoard = $state<(Card | null)[]>([null, null, null, null, null]);

	constructor() {
		if (browser) this.#load();
	}

	/** Milliseconds a robot takes per action; 0 is instant. */
	get speed() {
		return this.#speed;
	}
	set speed(value: number) {
		this.#speed = value;
		this.#save();
	}
	/** Seats at the table, including you. Applies to the next new game. */
	get players() {
		return this.#players;
	}
	set players(value: number) {
		this.#players = value;
		this.#save();
	}
	/** Starting stack. Applies to the next new game. */
	get chips() {
		return this.#chips;
	}
	set chips(value: number) {
		this.#chips = value;
		this.#save();
	}
	/** Deal the next hand automatically after a short pause. */
	get autoDeal() {
		return this.#autoDeal;
	}
	set autoDeal(value: boolean) {
		this.#autoDeal = value;
		this.#save();
	}
	/** Show what the starting-hand chart says about your hand while you play. */
	get hints() {
		return this.#hints;
	}
	set hints(value: boolean) {
		this.#hints = value;
		this.#save();
	}
	/** Cards picked for upcoming deals; null = random. */
	get pickedHole() {
		return this.#pickedHole;
	}
	get pickedBoard() {
		return this.#pickedBoard;
	}

	/** Pick (or clear) one slot: 0-1 your hole cards, 2-6 the board. */
	setPick(slot: number, card: Card | null) {
		if (slot < 2) this.#pickedHole = this.#pickedHole.map((c, i) => (i === slot ? card : c));
		else this.#pickedBoard = this.#pickedBoard.map((c, i) => (i === slot - 2 ? card : c));
		this.#save();
	}

	#save() {
		if (!browser) return;
		const data = {
			speed: this.#speed,
			players: this.#players,
			chips: this.#chips,
			autoDeal: this.#autoDeal,
			hints: this.#hints,
			pickedHole: this.#pickedHole,
			pickedBoard: this.#pickedBoard
		};
		try {
			localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
		} catch {
			// Storage can be unavailable (private mode); settings then last for the session.
		}
	}

	#load() {
		try {
			const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}');
			this.#speed = clampInt(saved.speed, LIMITS.speed);
			this.#players = clampInt(saved.players, LIMITS.players);
			this.#chips = clampInt(saved.chips, LIMITS.chips);
			this.#autoDeal = saved.autoDeal !== false;
			this.#hints = saved.hints === true;
			this.#pickedHole = cleanSlots(saved.pickedHole, 2);
			this.#pickedBoard = cleanSlots(saved.pickedBoard, 5);
		} catch {
			// Corrupt or missing data: keep the defaults.
		}
	}

	get hasPicks(): boolean {
		return [...this.#pickedHole, ...this.#pickedBoard].some((c) => c !== null);
	}

	/** The picks in the shape the API takes. */
	picks(): Picks {
		return { hole: [...this.#pickedHole], board: [...this.#pickedBoard] };
	}
}

export const settings = new Settings();

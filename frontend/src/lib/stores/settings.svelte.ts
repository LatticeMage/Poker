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
	/** Milliseconds a robot takes per action; 0 is instant. */
	speed = $state<number>(LIMITS.speed.fallback);
	/** Seats at the table, including you. Applies to the next new game. */
	players = $state<number>(LIMITS.players.fallback);
	/** Starting stack. Applies to the next new game. */
	chips = $state<number>(LIMITS.chips.fallback);
	/** Deal the next hand automatically after a short pause. */
	autoDeal = $state(true);
	/** Show what the starting-hand chart says about your hand while you play. */
	hints = $state(false);
	/** Cards picked for upcoming deals; null = random. */
	pickedHole = $state<(Card | null)[]>([null, null]);
	pickedBoard = $state<(Card | null)[]>([null, null, null, null, null]);

	constructor() {
		if (!browser) return;
		this.#load();
		$effect.root(() => {
			$effect(() => {
				const data = {
					speed: this.speed,
					players: this.players,
					chips: this.chips,
					autoDeal: this.autoDeal,
					hints: this.hints,
					pickedHole: [...this.pickedHole],
					pickedBoard: [...this.pickedBoard]
				};
				try {
					localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
				} catch {
					// Storage can be unavailable (private mode); settings then last for the session.
				}
			});
		});
	}

	#load() {
		try {
			const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}');
			this.speed = clampInt(saved.speed, LIMITS.speed);
			this.players = clampInt(saved.players, LIMITS.players);
			this.chips = clampInt(saved.chips, LIMITS.chips);
			this.autoDeal = saved.autoDeal !== false;
			this.hints = saved.hints === true;
			this.pickedHole = cleanSlots(saved.pickedHole, 2);
			this.pickedBoard = cleanSlots(saved.pickedBoard, 5);
		} catch {
			// Corrupt or missing data: keep the defaults.
		}
	}

	get hasPicks(): boolean {
		return [...this.pickedHole, ...this.pickedBoard].some((c) => c !== null);
	}

	/** The picks in the shape the API takes. */
	picks(): Picks {
		return { hole: [...this.pickedHole], board: [...this.pickedBoard] };
	}
}

export const settings = new Settings();

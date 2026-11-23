// The game as the client holds it. The backend is stateless: this store is the only copy of the
// game, it is sent with every request, and it is saved to localStorage so a reload carries on.
//
//   event -> lib/actions/games.ts -> engine -> reply -> this store -> $-subscriptions in the UI
//
// A reply holds the final `state` plus the `steps` that led to it. `showReply` plays the steps
// one at a time, paced by the robot-speed setting, so the table shows the robots acting in turn.
// Nothing here decides game logic.

import { browser } from '$app/environment';
import { settings } from '$lib/stores/settings.svelte';
import type { GameEvent, GameReply, GameState, Legal, Step } from '$lib/types/game';
import { derived, get, writable } from 'svelte/store';

export interface LogLine extends GameEvent {
	id: number;
}

const STORAGE_KEY = 'robo-holdem:game';
const MAX_LOG_LINES = 500;
const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** The table: cards, chips, pots - as of the step currently shown. */
export const game = writable<GameState | null>(null);
/** Your options - set only while it is your turn and the replay has caught up. */
export const legal = writable<Legal | null>(null);
export const log = writable<LogLine[]>([]);
/** A request is in flight. */
export const loading = writable(false);
/** Steps of the last reply are still being shown. */
export const replaying = writable(false);
export const error = writable<string | null>(null);

export const hero = derived(game, ($game) => $game?.players.find((p) => p.is_human) ?? null);
/** You have busted; the robots are playing on. */
export const heroOut = derived(
	hero,
	($hero) => !!$hero && $hero.chips === 0 && $hero.place !== null
);
/** The hand is over and the next deal is up to the player (or the auto-deal timer). */
export const awaitingNext = derived(
	[game, replaying, loading],
	([$game, $replaying, $loading]) => $game?.phase === 'hand_over' && !$replaying && !$loading
);

/** The game after the latest reply - what the next request must carry. Never a mid-replay step. */
let authoritative: GameState | null = null;
let authoritativeLegal: Legal | null = null;
let generation = 0;
let lineId = 0;

export function currentState(): GameState | null {
	return authoritative;
}

export function currentGeneration(): number {
	return generation;
}

function lines(events: GameEvent[]): LogLine[] {
	return events.map((e) => ({ ...e, id: ++lineId }));
}

function persist(state: GameState | null, options: Legal | null, history: LogLine[]) {
	if (!browser) return;
	try {
		if (!state) localStorage.removeItem(STORAGE_KEY);
		else localStorage.setItem(STORAGE_KEY, JSON.stringify({ state, legal: options, log: history }));
	} catch {
		// Storage unavailable or full: the game simply will not survive a reload.
	}
}

/**
 * Forget the current game. Returns the new generation; stale replays and replies stop on it.
 * `forgetSaved: false` keeps the saved copy (used when about to load it).
 */
export function reset(forgetSaved = true): number {
	authoritative = null;
	authoritativeLegal = null;
	game.set(null);
	legal.set(null);
	log.set([]);
	loading.set(false);
	replaying.set(false);
	error.set(null);
	if (forgetSaved) persist(null, null, []);
	return ++generation;
}

function looksLikeAGame(value: unknown): value is GameState {
	const g = value as GameState | null;
	return (
		!!g &&
		Array.isArray(g.players) &&
		g.players.length >= 2 &&
		Array.isArray(g.runout) &&
		typeof g.phase === 'string' &&
		g.players.every((p) => Array.isArray(p.hole) && typeof p.chips === 'number')
	);
}

/** Continue the game saved by a previous visit. Returns false if there is none (or it is unusable). */
export function restore(): boolean {
	if (!browser) return false;
	try {
		const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null');
		if (!saved || !looksLikeAGame(saved.state)) return false;
		reset(false); // the saved copy is the one being loaded: keep it for the next reload
		authoritative = saved.state;
		authoritativeLegal = saved.legal ?? null;
		game.set(saved.state);
		legal.set(authoritativeLegal);
		log.set(Array.isArray(saved.log) ? saved.log.slice(-MAX_LOG_LINES) : []);
		return true;
	} catch {
		return false;
	}
}

/** How long to hold before showing a step. */
function delay(step: Step, index: number, before: GameState | null): number {
	// The first step is the direct result of the player's click: show it at once.
	if (index === 0) return 0;
	// Once you are out of the hand, the rest plays out faster.
	const watching = before?.players.some((p) => p.is_human && p.in_hand && !p.folded) ?? false;
	const factor = step.kind === 'action' ? 0.7 + Math.random() * 0.6 : 1.3;
	return settings.speed * factor * (watching ? 1 : 0.5);
}

/**
 * Take a reply into the store and show its steps one by one. Resolves to false if the table was
 * reset meanwhile (the reply belonged to an abandoned game).
 */
export async function showReply(reply: GameReply, forGeneration: number): Promise<boolean> {
	if (forGeneration !== generation) return false;

	// The authoritative game and the saved history move to the end of the reply at once, so a
	// reload in the middle of a replay resumes at the end of it.
	authoritative = reply.state;
	authoritativeLegal = reply.legal;
	const incoming = lines(reply.steps.flatMap((s) => s.events));
	persist(reply.state, reply.legal, [...get(log), ...incoming].slice(-MAX_LOG_LINES));

	legal.set(null);
	replaying.set(true);
	let shown = 0;
	for (const [index, step] of reply.steps.entries()) {
		const ms = delay(step, index, get(game));
		if (ms > 0) await sleep(ms);
		if (forGeneration !== generation) return false;
		game.set(step.state);
		const mine = incoming.slice(shown, shown + step.events.length);
		shown += step.events.length;
		log.update((all) => [...all, ...mine].slice(-MAX_LOG_LINES));
	}
	game.set(reply.state);
	replaying.set(false);
	legal.set(authoritativeLegal);
	return true;
}

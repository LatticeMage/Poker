// Game controller: every move goes through here. It asks the game engine (lib/poker) for the next
// state, hands the reply to the game store, and schedules the automatic deal. Components call these
// functions; they never run the engine or edit the table themselves.
//
// The engine is a set of pure functions, so each call passes the whole game from the store.

import {
	currentGeneration,
	currentState,
	reset,
	restore,
	showReply,
	table
} from '$lib/stores/game.svelte';
import { settings } from '$lib/stores/settings.svelte';
import * as engine from '$lib/poker/game';
import type { Action, GameReply } from '$lib/types/game';

let autoDealTimer: ReturnType<typeof setTimeout> | undefined;
/** The request that last failed, so "Try again" can repeat it. */
let failed: (() => Promise<void>) | null = null;

/** With auto-deal on, the next hand starts by itself after a pause to read the result. */
function scheduleAutoDeal(generation: number) {
	clearTimeout(autoDealTimer);
	const state = table.game;
	if (!settings.autoDeal || state?.phase !== 'hand_over') return;
	const humanOut = state.players.some((p) => p.is_human && p.chips === 0 && p.place !== null);
	const pause = humanOut ? Math.max(1200, settings.speed * 3) : Math.max(2800, settings.speed * 6);
	autoDealTimer = setTimeout(() => {
		if (generation === currentGeneration()) void nextHand();
	}, pause);
}

/** Compute the next state, then show the reply on the table. */
async function run(generation: number, request: () => GameReply) {
	clearTimeout(autoDealTimer);
	table.loading = true;
	table.error = null;
	failed = null;
	try {
		await Promise.resolve(); // let the "loading" state paint before the engine runs
		const reply = request();
		if (generation !== currentGeneration()) return;
		table.loading = false;
		if (await showReply(reply, generation)) scheduleAutoDeal(generation);
	} catch (cause) {
		if (generation !== currentGeneration()) return;
		table.loading = false;
		failed = () => run(generation, request);
		if (cause instanceof engine.InvalidState) {
			// The saved game is not one the engine could have produced (e.g. it came from an older version).
			table.error = 'The saved game could not be continued.';
			failed = null;
		} else {
			table.error = cause instanceof Error ? cause.message : 'Something went wrong';
		}
	}
}

/** Start a new game with the current settings, abandoning any game in progress. */
export function startGame(): Promise<void> {
	const generation = reset();
	return run(generation, () =>
		engine.newGame({ players: settings.players, chips: settings.chips, picks: settings.picks() })
	);
}

/** On page load: carry on with the game saved by the last visit, else start one. */
export function resumeGame(): Promise<void> {
	if (!restore()) return startGame();
	scheduleAutoDeal(currentGeneration());
	return Promise.resolve();
}

/** Repeat the request that failed. */
export function retry(): Promise<void> {
	return failed ? failed() : Promise.resolve();
}

/** Your decision for the current turn. */
export function act(action: Action): Promise<void> {
	const state = currentState();
	if (!table.legal || !state || table.loading) return Promise.resolve();
	table.legal = null;
	return run(currentGeneration(), () => engine.act(state, action));
}

/** Deal the next hand (also called by the auto-deal timer). */
export function nextHand(): Promise<void> {
	const state = currentState();
	if (!table.awaitingNext || !state) return Promise.resolve();
	return run(currentGeneration(), () => engine.nextHand(state, settings.picks()));
}

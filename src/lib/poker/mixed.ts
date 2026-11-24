// Mixed-frequency preflop strategies for the robots.
//
// The charts give every hand a frequency, not a yes/no: a hand at the edge of an opening range opens
// 30% of the time, a solved heads-up node says "raise 60%, call 40%". This module turns those tables
// into per-hand action distributions the bot samples from, and applies a persona's width to them.
//
// Widening or narrowing a strategy keeps the idea of ranges.adjust (grow the strongest missing hands
// first, shrink the weakest first) but moves *probability mass* instead of whole hands, so the share
// of hands that play stays exactly width times the chart's share.

import type { HeadsUpNode } from '$lib/types/ranges';
import { HEADS_UP_STACKS, headsUpChart, pushFoldChart, rangeChart } from './charts';
import type { Game } from './engine';
import { ORDER, TOTAL_COMBOS, combos, roundHalfEven } from './ranges';
import type { Rng } from './rng';

export type Freq = Record<string, number>;

/** Share of hands (0-1) a frequency table plays, counting every combo. */
export function share(freq: Freq): number {
	let total = 0;
	for (const [name, f] of Object.entries(freq)) total += f * combos(name);
	return total / TOTAL_COMBOS;
}

/**
 * Scale how many hands play by `factor`. Growing tops hands up to 1 from the strongest down (those
 * just outside the range come in first, up to `cap` per hand); shrinking removes the weakest hands
 * first. Growth stops early if the caps leave no room.
 */
export function scaleMass(freq: Freq, factor: number, cap: Freq = {}): Freq {
	const out: Freq = { ...freq };
	if (factor === 1) return out;
	const have = share(freq) * TOTAL_COMBOS;
	const target = Math.min(TOTAL_COMBOS, Math.max(0, have * factor));
	let diff = target - have;
	if (diff > 0) {
		for (const name of ORDER) {
			const f = out[name] ?? 0;
			const room = (cap[name] ?? 1) - f; // a hand that raises cannot also be filled up with calls
			if (room <= 0) continue;
			const take = Math.min(room, diff / combos(name));
			out[name] = f + take;
			diff -= take * combos(name);
			if (diff <= 1e-9) break;
		}
	} else if (diff < 0) {
		for (let i = ORDER.length - 1; i >= 0 && diff < -1e-9; i--) {
			const name = ORDER[i];
			const f = out[name] ?? 0;
			if (f <= 0) continue;
			const take = Math.min(f, -diff / combos(name));
			out[name] = f - take;
			diff += take * combos(name);
		}
	}
	return out;
}

/** The persona's width (1 = the chart): tight players play fewer hands, loose ones more. */
export const widthOf = (tight: number): number => 1 + (0.5 - tight) * 0.6;

// -- first in ----------------------------------------------------------------

const openCache = new Map<number, Freq[]>();

/** How often each hand raises first in, for the seat at `index` in preflop order (the chart's mix). */
export function openFreq(players: number, index: number): Freq {
	let seats = openCache.get(players);
	if (!seats) {
		seats = rangeChart(players).seats.map((s) => s.mix);
		openCache.set(players, seats);
	}
	return seats[index] ?? {};
}

const scaledCache = new Map<string, Freq>();

/** `openFreq` scaled by `factor`, cached. */
export function scaledOpenFreq(players: number, index: number, factor: number): Freq {
	const key = `${players}|${index}|${factor.toFixed(4)}`;
	let out = scaledCache.get(key);
	if (!out) {
		out = scaleMass(openFreq(players, index), factor);
		scaledCache.set(key, out);
	}
	return out;
}

/** Hand-wise sum of two frequency tables. */
export function sum(a: Freq, b: Freq): Freq {
	const out: Freq = { ...a };
	for (const [name, f] of Object.entries(b)) out[name] = (out[name] ?? 0) + f;
	return out;
}

const calledCache = new WeakMap<object, Map<number, Freq>>();

/**
 * The calling range of a spot (`owner`: the spot object) scaled by `factor`. A hand has no room for
 * calls beyond what its raises leave: `raises` (hand to share that raises) caps it.
 */
export function scaledCalls(owner: object, calls: Freq, raises: Freq, factor: number): Freq {
	let byFactor = calledCache.get(owner);
	if (!byFactor) calledCache.set(owner, (byFactor = new Map()));
	const key = Math.round(factor * 10_000);
	let out = byFactor.get(key);
	if (!out) {
		const cap: Freq = {};
		for (const name of ORDER) cap[name] = Math.max(0, 1 - (raises[name] ?? 0));
		out = scaleMass(calls, factor, cap);
		byFactor.set(key, out);
	}
	return out;
}

// -- heads-up solved trees ---------------------------------------------------

/** What a heads-up node tells the bot to do. */
export type Choice =
	| { kind: 'fold' | 'check' | 'call' | 'limp' | 'allin' }
	| { kind: 'raise'; to_bb: number };

/** The solved depth nearest to the stack, in log terms (a 30bb stack is closer to 40 than to 20). */
export function nearestDepth(stackBb: number): number {
	let best = HEADS_UP_STACKS[0];
	for (const d of HEADS_UP_STACKS) {
		if (Math.abs(Math.log(stackBb / d)) < Math.abs(Math.log(stackBb / best))) best = d;
	}
	return best;
}

/** Below this effective stack the heads-up game is played from the push/fold tables. */
export const TREE_MIN_BB = 7;

/** Raise sizes (to, in bb) of the solved trees, by the line that led there. */
const SIZES = {
	open: [2.5, 9, 22],
	limp: [4, 12]
};

/**
 * Path of the node the player to act is facing, in the naming of the solved trees ("root", "limp",
 * "open/2.5/9", "limp/4/allin"). Null where the line is not one the tree knows (an odd raise size).
 */
export function nodePath(g: Game, seat: number): string | null {
	const bb = g.big_blind;
	const k = g.raisers.length;
	if (k === 0) {
		if (seat === g.sb_seat) return 'root';
		return g.players[g.sb_seat].bet <= bb ? 'limp' : null; // the big blind's option
	}
	const kind = g.raisers[0] === g.sb_seat ? 'open' : 'limp';
	const sizes = SIZES[kind];
	if (k > sizes.length) {
		// only an all-in can go beyond the sized raises ("limp/4/12/allin")
		if (k !== sizes.length + 1 || !g.players[g.raisers[k - 1]].all_in) return null;
		return [kind, ...sizes.map(String), 'allin'].join('/');
	}
	const path: string[] = [kind, ...sizes.slice(0, k).map(String)];
	if (g.players[g.raisers[k - 1]].all_in) {
		path[k] = 'allin';
	} else {
		const actual = g.current_bet / bb;
		if (actual < sizes[k - 1] * 0.6 || actual > sizes[k - 1] * 1.7) return null;
	}
	return path.join('/');
}

const treeCache = new Map<string, Record<string, number[]>>();

/** Every hand's action distribution at a node, with the continue share scaled by `factor`. */
function nodeDists(
	depth: number,
	path: string,
	node: HeadsUpNode,
	factor: number
): Record<string, number[]> {
	const key = `${depth}|${path}|${factor.toFixed(4)}`;
	const cached = treeCache.get(key);
	if (cached) return cached;
	const n = node.actions.length;
	const fold = node.actions.indexOf('fold');
	const passive = node.actions.findIndex((a) => a === 'call' || a === 'check' || a === 'limp');
	const out: Record<string, number[]> = {};
	if (fold < 0 || factor === 1) {
		for (const name of ORDER) out[name] = node.freq[name] ?? new Array<number>(n).fill(0);
		treeCache.set(key, out);
		return out;
	}
	const cont: Freq = {};
	for (const name of ORDER) {
		const f = node.freq[name];
		if (f) cont[name] = 1 - f[fold];
	}
	const scaled = scaleMass(cont, factor);
	for (const name of ORDER) {
		const f = node.freq[name];
		const had = cont[name] ?? 0;
		const now = scaled[name] ?? 0;
		const d = new Array<number>(n).fill(0);
		if (f && had > 1e-9) {
			for (let i = 0; i < n; i++) d[i] = i === fold ? 1 - now : (f[i] * now) / had;
		} else {
			d[passive] = now; // hands the tree never plays come in as calls or limps
			d[fold] = 1 - now;
		}
		out[name] = d;
	}
	treeCache.set(key, out);
	return out;
}

function choiceOf(action: string): Choice {
	if (action.startsWith('raise ')) return { kind: 'raise', to_bb: Number(action.slice(6)) };
	return { kind: action as 'fold' | 'check' | 'call' | 'limp' | 'allin' };
}

/**
 * Sample the solved heads-up action for a hand: the tree at the nearest depth, at the node the game
 * is in. Null if no node fits (the caller falls back on its own heuristics).
 */
export function headsUpChoice(
	g: Game,
	seat: number,
	name: string,
	effectiveBb: number,
	factor: number,
	rng: Rng
): Choice | null {
	const path = nodePath(g, seat);
	if (!path) return null;
	const depth = nearestDepth(effectiveBb);
	const node = headsUpChart(depth).nodes[path];
	if (!node) return null;
	const dist = nodeDists(depth, path, node, factor)[name];
	let total = 0;
	for (const x of dist) total += x;
	if (total <= 0) return { kind: 'fold' };
	let u = rng() * total;
	for (let i = 0; i < dist.length; i++) {
		if (u < dist[i]) return choiceOf(node.actions[i]);
		u -= dist[i];
	}
	return choiceOf(node.actions[node.actions.length - 1]);
}

/** The action shares of one hand at a node of the tree at `depth`, or null if there is no such node. */
export function headsUpDist(depth: number, path: string, name: string, factor = 1): Freq | null {
	const node = headsUpChart(depth).nodes[path];
	if (!node) return null;
	const dist = nodeDists(depth, path, node, factor)[name];
	return Object.fromEntries(node.actions.map((a, i) => [a, dist[i]]));
}

// -- push/fold below the trees -----------------------------------------------

/** How often each hand shoves (small blind) or calls a shove (big blind), from the solved mix. */
export function pushFoldFreq(stackBb: number, side: 'push' | 'call'): Freq {
	const chart = pushFoldChart(roundHalfEven(stackBb));
	const out: Freq = { ...(side === 'push' ? chart.push_mix : chart.call_mix) };
	for (const h of side === 'push' ? chart.push : chart.call) out[h] ??= 1;
	return out;
}

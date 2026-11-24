// The solved heads-up preflop trees (nvcc/ CFR+ solver), one JSON file per stack depth.
//
// Files are found by name: data/headsup-<bb>.json. A re-solved file with a better realisation model,
// data/headsup-<bb>-real.json, takes the place of the plain one for that depth. Adding or swapping a
// depth is a data-only change: drop the file in data/ and nothing else needs to be edited.

import type { HeadsUpChart } from '$lib/types/ranges';

const FILES = import.meta.glob('./data/headsup-*.json', {
	eager: true,
	import: 'default'
}) as Record<string, unknown>;

const NAME = /headsup-(\d+)(-real)?\.json$/;

const CHARTS = new Map<number, HeadsUpChart>();
{
	const real = new Set<number>();
	for (const [path, data] of Object.entries(FILES)) {
		const m = NAME.exec(path);
		if (!m) continue;
		const depth = Number(m[1]);
		if (real.has(depth) && !m[2]) continue; // a -real file wins
		if (m[2]) real.add(depth);
		CHARTS.set(depth, data as HeadsUpChart);
	}
}

/** Stack depths (in big blinds) with a solved tree, ascending. */
export const HEADS_UP_STACKS: readonly number[] = [...CHARTS.keys()].sort((a, b) => a - b);

/** The depth the first-in ranges use when no stack is given. */
export const DEFAULT_HEADS_UP_BB = 100;

/** The solved depth closest to `bb` (the shallower one on a tie). */
export function nearestStack(bb: number): number {
	let best = HEADS_UP_STACKS[0];
	for (const depth of HEADS_UP_STACKS) {
		if (Math.abs(depth - bb) < Math.abs(best - bb)) best = depth;
	}
	return best;
}

/** The solved tree for exactly this depth, if there is one. */
export const solvedChart = (stackBb: number): HeadsUpChart | undefined => CHARTS.get(stackBb);

export interface FirstIn {
	stack_bb: number;
	/** Per hand: [raise (incl. all-in), limp, fold] shares of the small blind's first action. */
	shares: Record<string, [number, number, number]>;
}

/** The small blind's first action at the solved depth nearest to `bb`. */
export function firstIn(bb: number = DEFAULT_HEADS_UP_BB): FirstIn {
	const chart = CHARTS.get(nearestStack(bb))!;
	const node = chart.nodes.root;
	const shares: FirstIn['shares'] = {};
	for (const [hand, freq] of Object.entries(node.freq)) {
		const row: [number, number, number] = [0, 0, 0];
		node.actions.forEach((action, i) => {
			if (action === 'fold') row[2] += freq[i];
			else if (action === 'limp' || action === 'call' || action === 'check') row[1] += freq[i];
			else row[0] += freq[i];
		});
		shares[hand] = row;
	}
	return { stack_bb: chart.stack_bb, shares };
}

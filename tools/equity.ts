// Shared by the offline tools: the 169x169 matrix of hand-class equities (Monte Carlo over concrete
// suit combinations and random boards, card removal exact) and the matching combo weights.
// The matrix is cached under tools/.cache so repeated solver runs do not recompute it.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { handScore } from '../src/lib/poker/evaluate';
import { ALL_HANDS, classCombos } from '../src/lib/poker/ranges';
import { mulberry32 } from '../src/lib/poker/rng';

export const N = ALL_HANDS.length;
export const CLASSES = ALL_HANDS.map((name) => classCombos(name));
const SEED = 20261123;

function matchup(i: number, j: number, samples: number): number {
	const rng = mulberry32(SEED + i * 169 + j);
	const aHands = CLASSES[i];
	const bHands = CLASSES[j];
	let wins = 0;
	let done = 0;
	while (done < samples) {
		const a = aHands[Math.floor(rng() * aHands.length)];
		const b = bHands[Math.floor(rng() * bHands.length)];
		if (a[0] === b[0] || a[0] === b[1] || a[1] === b[0] || a[1] === b[1]) continue;
		const used = new Set<number>([a[0], a[1], b[0], b[1]]);
		const board: number[] = [];
		while (board.length < 5) {
			const c = Math.floor(rng() * 52);
			if (!used.has(c)) {
				used.add(c);
				board.push(c);
			}
		}
		const sa = handScore([a[0], a[1], ...board]);
		const sb = handScore([b[0], b[1], ...board]);
		wins += sa > sb ? 1 : sa === sb ? 0.5 : 0;
		done++;
	}
	return wins / samples;
}

/** equity[i][j]: share of the pot class i wins against class j. */
export function equityMatrix(samples: number, log = console.log): Float64Array[] {
	const cache = `tools/.cache/equity-${samples}.json`;
	if (existsSync(cache)) {
		return (JSON.parse(readFileSync(cache, 'utf8')) as number[][]).map((r) => Float64Array.from(r));
	}
	const started = Date.now();
	const equity = Array.from({ length: N }, () => new Float64Array(N).fill(0.5));
	let done = 0;
	const pairs = (N * (N - 1)) / 2;
	for (let i = 0; i < N; i++) {
		for (let j = i + 1; j < N; j++) {
			const e = matchup(i, j, samples);
			equity[i][j] = e;
			equity[j][i] = 1 - e;
			if (++done % 4000 === 0)
				log(`  ${done}/${pairs} matchups (${Math.round((Date.now() - started) / 1000)}s)`);
		}
	}
	mkdirSync('tools/.cache', { recursive: true });
	writeFileSync(cache, JSON.stringify(equity.map((r) => Array.from(r))));
	return equity;
}

/** w[i][j]: number of (combo of i, combo of j) pairs that share no card. */
export function conflictFreeWeights(): Float64Array[] {
	return CLASSES.map((ci) =>
		Float64Array.from(CLASSES, (cj) => {
			let n = 0;
			for (const a of ci)
				for (const b of cj)
					if (a[0] !== b[0] && a[0] !== b[1] && a[1] !== b[0] && a[1] !== b[1]) n++;
			return n;
		})
	);
}

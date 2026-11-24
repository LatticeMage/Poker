// Offline generator for src/lib/poker/data/tables.json (run once; the app only reads the result).
//
//   npm run tables -- [samples-per-matchup] [output-file]
//
// 1. Equity matrix. For every pair of the 169 starting-hand classes, the share of the pot class A
//    wins against class B, by Monte Carlo over concrete suit combinations and random boards. Card
//    removal is exact (combos that share a card are never dealt).
// 2. Equity against a random hand, per class (weights = the non-conflicting combos), used as the
//    strength order inside the range charts.
// 3. Heads-up push/fold Nash approximation for effective stacks of 2-20bb (cEV, no ICM): the small
//    blind shoves or folds, the big blind calls or folds. Solved by fictitious play on the equity
//    matrix, then thresholded to pure ranges; the leftover exploitability is stored with each.
//
// Superseded: the committed tables.json no longer comes from this script. The equity matrix is the exact one
// of nvcc/ (equity exact) and the push/fold part is solved on the GPU (nvcc/pushfold.cu), then copied in by
// tools/apply-pushfold.ts. This script is kept for the sampled, CPU-only version of the same tables.
//
// With the default 4000 samples this takes a few minutes. The output path defaults to a scratch
// file next to the real tables, so regenerating never overwrites the committed data by accident:
// compare, then copy it over src/lib/poker/data/tables.json yourself.

import { writeFileSync } from 'node:fs';
import { handScore } from '../src/lib/poker/evaluate';
import { ALL_HANDS, classCombos, combos } from '../src/lib/poker/ranges';
import { mulberry32 } from '../src/lib/poker/rng';

const samples = Number(process.argv[2] ?? 4000);
const out = process.argv[3] ?? 'src/lib/poker/data/tables.generated.json';
const STACKS = Array.from({ length: 19 }, (_, i) => i + 2);
const ITERATIONS = 800;
const SEED = 20261123;
const N = ALL_HANDS.length;

const CLASSES = ALL_HANDS.map((name) => classCombos(name));

/** Equity of class i against class j: random combos of each (never sharing a card) and a random board. */
function matchup(i: number, j: number): number {
	const rng = mulberry32(SEED + i * 169 + j);
	const aHands = CLASSES[i];
	const bHands = CLASSES[j];
	let wins = 0;
	let done = 0;
	const cards = new Int8Array(7);
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
		cards[0] = a[0];
		cards[1] = a[1];
		const sa = handScore([a[0], a[1], ...board]);
		const sb = handScore([b[0], b[1], ...board]);
		wins += sa > sb ? 1 : sa === sb ? 0.5 : 0;
		done++;
	}
	return wins / samples;
}

/** w[i][j] = number of (combo of i, combo of j) pairs that share no card. */
function conflictFreeWeights(): Float64Array[] {
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

interface Solution {
	/** Averaged fictitious-play frequencies: the mixed strategy before thresholding. */
	pushMix: Float64Array;
	callMix: Float64Array;
	push: Float64Array;
	call: Float64Array;
	exploitability: number;
}

function solve(equity: Float64Array[], w: Float64Array[], stack: number): Solution {
	const wsum = w.map((row) => row.reduce((s, x) => s + x, 0));
	const pHand = Float64Array.from(ALL_HANDS, (name) => combos(name));
	const total = pHand.reduce((s, x) => s + x, 0);
	for (let i = 0; i < N; i++) pHand[i] /= total;

	const sbValues = (call: Float64Array): Float64Array => {
		const values = new Float64Array(N);
		for (let h = 0; h < N; h++) {
			let mass = 0;
			let weighted = 0;
			for (let g = 0; g < N; g++) {
				const m = w[h][g] * call[g];
				mass += m;
				weighted += m * equity[h][g];
			}
			const eq = mass > 0 ? weighted / mass : 0.5;
			values[h] = (1 - mass / wsum[h]) * 1 + (mass / wsum[h]) * stack * (2 * eq - 1);
		}
		return values;
	};
	const bbValues = (push: Float64Array): Float64Array => {
		const values = new Float64Array(N);
		for (let g = 0; g < N; g++) {
			let mass = 0;
			let weighted = 0;
			for (let h = 0; h < N; h++) {
				const m = w[h][g] * push[h];
				mass += m;
				weighted += m * (1 - equity[h][g]);
			}
			const eq = mass > 0 ? weighted / mass : 0.5;
			values[g] = stack * (2 * eq - 1);
		}
		return values;
	};
	// Payoffs relative to the posted blinds: the small blind folds for -0.5, the big blind for -1.
	const bestSb = (call: Float64Array) => sbValues(call).map((v) => (v > -0.5 ? 1 : 0));
	const bestBb = (push: Float64Array) => bbValues(push).map((v) => (v > -1 ? 1 : 0));

	let push = new Float64Array(N).fill(1);
	let call = Float64Array.from(bestBb(push));
	for (let t = 1; t <= ITERATIONS; t++) {
		const sb = bestSb(call);
		push = push.map((p, i) => p + (sb[i] - p) / (t + 1));
		const bb = bestBb(push);
		call = call.map((c, i) => c + (bb[i] - c) / (t + 1));
	}
	const pushPure = push.map((p) => (p > 0.5 ? 1 : 0));
	const callPure = call.map((c) => (c > 0.5 ? 1 : 0));

	// Exploitability of the pure strategies: how much a best response gains, weighted by how often
	// each hand is dealt.
	const sbTotal = (strategy: Float64Array, vsCall: Float64Array) => {
		const v = sbValues(vsCall);
		return strategy.reduce((s, x, i) => s + pHand[i] * (x * v[i] + (1 - x) * -0.5), 0);
	};
	const bbTotal = (strategy: Float64Array, vsPush: Float64Array) => {
		const v = bbValues(vsPush);
		return strategy.reduce((s, x, i) => s + pHand[i] * (x * v[i] + (1 - x) * -1), 0);
	};
	const gapSb =
		sbTotal(Float64Array.from(bestSb(callPure)), callPure) - sbTotal(pushPure, callPure);
	const gapBb =
		bbTotal(Float64Array.from(bestBb(pushPure)), pushPure) - bbTotal(callPure, pushPure);
	return {
		pushMix: push,
		callMix: call,
		push: pushPure,
		call: callPure,
		exploitability: gapSb + gapBb
	};
}

const started = Date.now();
const equity = Array.from({ length: N }, () => new Float64Array(N).fill(0.5));
let done = 0;
const pairs = (N * (N - 1)) / 2;
for (let i = 0; i < N; i++) {
	for (let j = i + 1; j < N; j++) {
		const e = matchup(i, j);
		equity[i][j] = e;
		equity[j][i] = 1 - e;
		if (++done % 2000 === 0) {
			console.log(`  ${done}/${pairs} matchups  (${Math.round((Date.now() - started) / 1000)}s)`);
		}
	}
}
console.log(`equity matrix done in ${Math.round((Date.now() - started) / 1000)}s`);

const w = conflictFreeWeights();
const vsRandom: Record<string, number> = {};
ALL_HANDS.forEach((name, i) => {
	let weighted = 0;
	let total = 0;
	for (let j = 0; j < N; j++) {
		weighted += w[i][j] * equity[i][j];
		total += w[i][j];
	}
	vsRandom[name] = Math.round((weighted / total) * 10000) / 10000;
});

const pct = (flags: Float64Array) => {
	let hit = 0;
	let all = 0;
	flags.forEach((f, i) => {
		const c = combos(ALL_HANDS[i]);
		all += c;
		hit += f * c;
	});
	return Math.round(((100 * hit) / all) * 10) / 10;
};
const pushFold: Record<string, unknown> = {};
const mixes: Record<string, unknown> = {};
const freqs = (f: Float64Array) =>
	Object.fromEntries(
		ALL_HANDS.flatMap((name, i) =>
			f[i] >= 0.02 ? [[name, Math.round(Math.min(1, f[i]) * 100) / 100]] : []
		)
	);
for (const stack of STACKS) {
	const s = solve(equity, w, stack);
	const row = {
		push: ALL_HANDS.filter((_, i) => s.push[i]),
		call: ALL_HANDS.filter((_, i) => s.call[i]),
		push_pct: pct(s.push),
		call_pct: pct(s.call),
		exploitability_bb: Math.round(s.exploitability * 10000) / 10000
	};
	pushFold[String(stack)] = row;
	mixes[String(stack)] = { push: freqs(s.pushMix), call: freqs(s.callMix) };
	console.log(
		`  ${String(stack).padStart(2)}bb  push ${row.push_pct}%  call ${row.call_pct}%  exploitability ${row.exploitability_bb}`
	);
}

writeFileSync(
	out,
	JSON.stringify({
		samples_per_matchup: samples,
		equity_vs_random: vsRandom,
		pushfold: pushFold,
		mix: mixes
	})
);
console.log(`wrote ${out} in ${Math.round((Date.now() - started) / 1000)}s`);

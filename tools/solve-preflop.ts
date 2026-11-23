// Offline heads-up preflop solver: the small blind (button, player 0) against the big blind (player 1).
//
//   npm run solve -- [samples-per-matchup] [iterations]
//
// Game: blinds 0.5 / 1; the small blind can fold, limp or raise; every raise ladder is a short fixed list
// of sizes, then all-in. Full CFR+ over the 169 hand classes, range-vs-range (vectorised over hands).
//
// Showdown values: all-in calls use the plain equity matrix. Calls that see a flop use a *realised*
// equity: the player in position (the small blind) realises a little more than its raw equity and the big
// blind a little less (zero-sum). It is a heuristic, not a postflop solve; the site says so.
//
// Output: src/lib/poker/data/headsup-<stack>.json, one file per stack depth, holding the strategy at every
// reachable node and the exploitability of the averaged strategy.

import { writeFileSync } from 'node:fs';
import { ALL_HANDS, combos } from '../src/lib/poker/ranges';
import { N, conflictFreeWeights, equityMatrix } from './equity';

const STACKS = (process.env.STACKS ?? '10,20,40,80').split(',').map(Number);
const samples = Number(process.argv[2] ?? 4000);
const ITERATIONS = Number(process.argv[3] ?? 4000);

/** Raise ladders, in bb "to". A raise at or above 70% of the stack is an all-in. */
const OPEN_LEVELS = [[2.5], [9], [22]];
const LIMP_LEVELS = [[4], [12]];

type Terminal =
	| { kind: 'fold'; folder: 0 | 1; commit: [number, number] }
	| { kind: 'show'; commit: number; allIn: boolean };
interface Decision {
	kind: 'decision';
	id: string;
	player: 0 | 1;
	labels: string[];
	children: Node[];
}
type Node = Terminal | Decision;

function build(stack: number): Decision {
	const sizeOf = (to: number) => (to >= 0.7 * stack ? stack : to);
	/** Player `p` faces `bet` (their own commit is `own`); `levels` are the remaining raise ladders. */
	function facing(id: string, p: 0 | 1, own: number, bet: number, levels: number[][]): Decision {
		const labels: string[] = ['fold', 'call'];
		const children: Node[] = [
			{ kind: 'fold', folder: p, commit: p === 0 ? [own, bet] : [bet, own] },
			{ kind: 'show', commit: bet, allIn: bet >= stack }
		];
		if (bet < stack) {
			const sizes = new Set<number>();
			for (const to of levels[0] ?? []) sizes.add(sizeOf(to));
			sizes.add(stack);
			for (const to of [...sizes].sort((a, b) => a - b)) {
				if (to <= bet) continue;
				labels.push(to >= stack ? 'allin' : `raise ${to}`);
				children.push(
					facing(`${id}/${to >= stack ? 'allin' : to}`, (1 - p) as 0 | 1, bet, to, levels.slice(1))
				);
			}
		}
		return { kind: 'decision', id, player: p, labels, children };
	}
	// The small blind opens: fold / limp / raise / all-in.
	const labels: string[] = ['fold', 'limp'];
	const children: Node[] = [{ kind: 'fold', folder: 0, commit: [0.5, 1] }];
	const limp: Decision = (() => {
		const bbLabels = ['check'];
		const bbChildren: Node[] = [{ kind: 'show', commit: 1, allIn: false }];
		const sizes = new Set<number>([...LIMP_LEVELS[0].map(sizeOf), stack]);
		for (const to of [...sizes].sort((a, b) => a - b)) {
			bbLabels.push(to >= stack ? 'allin' : `raise ${to}`);
			bbChildren.push(facing(`limp/${to >= stack ? 'allin' : to}`, 0, 1, to, LIMP_LEVELS.slice(1)));
		}
		return { kind: 'decision', id: 'limp', player: 1, labels: bbLabels, children: bbChildren };
	})();
	children.push(limp);
	const sizes = new Set<number>([...OPEN_LEVELS[0].map(sizeOf), stack]);
	for (const to of [...sizes].sort((a, b) => a - b)) {
		labels.push(to >= stack ? 'allin' : `raise ${to}`);
		children.push(facing(`open/${to >= stack ? 'allin' : to}`, 1, 1, to, OPEN_LEVELS.slice(1)));
	}
	return { kind: 'decision', id: 'root', player: 0, labels, children };
}

/** Share of the pot the player in position (the small blind) realises. */
const realisedIp = (eq: number) => Math.min(1, Math.max(0, 0.5 + 1.08 * (eq - 0.5) + 0.025));

function solve(stack: number, equity: Float64Array[], w: Float64Array[]) {
	const root = build(stack);
	const wsum = w.map((row) => row.reduce((s, x) => s + x, 0));
	const prior = Float64Array.from(ALL_HANDS, (h) => combos(h));
	const total = prior.reduce((s, x) => s + x, 0);
	for (let i = 0; i < N; i++) prior[i] /= total;

	// Matrices of (combos x share) for showdowns, indexed [my hand][their hand], per player and mode.
	const mat = (f: (i: number, j: number) => number) =>
		w.map((row, i) => Float64Array.from(row, (x, j) => x * f(i, j)));
	const share = {
		raw: [mat((i, j) => equity[i][j]), mat((i, j) => 1 - equity[j][i])],
		real: [mat((i, j) => realisedIp(equity[i][j])), mat((i, j) => 1 - realisedIp(equity[j][i]))]
	};
	const times = (m: Float64Array[], r: Float64Array) =>
		Float64Array.from(m, (row) => {
			let s = 0;
			for (let j = 0; j < N; j++) s += row[j] * r[j];
			return s;
		});

	interface State {
		regret: Float64Array[];
		sum: Float64Array[];
	}
	const states = new Map<string, State>();
	const decisions: Decision[] = [];
	(function collect(n: Node) {
		if (n.kind !== 'decision') return;
		decisions.push(n);
		states.set(n.id, {
			regret: n.labels.map(() => new Float64Array(N)),
			sum: n.labels.map(() => new Float64Array(N))
		});
		n.children.forEach(collect);
	})(root);

	function strategy(st: State): Float64Array[] {
		const k = st.regret.length;
		const out = st.regret.map(() => new Float64Array(N));
		for (let h = 0; h < N; h++) {
			let pos = 0;
			for (let a = 0; a < k; a++) pos += st.regret[a][h];
			for (let a = 0; a < k; a++) out[a][h] = pos > 0 ? st.regret[a][h] / pos : 1 / k;
		}
		return out;
	}

	function terminal(
		n: Terminal,
		reach: [Float64Array, Float64Array]
	): [Float64Array, Float64Array] {
		const ones = (r: Float64Array) => times(w, r);
		if (n.kind === 'fold') {
			const [c0, c1] = n.commit;
			const loss = n.folder === 0 ? c0 : c1; // what the folder gives up is what the other wins
			const seen = [ones(reach[1]), ones(reach[0])];
			return [0, 1].map((p) => seen[p].map((x) => x * (p === n.folder ? -loss : loss))) as [
				Float64Array,
				Float64Array
			];
		}
		const m = n.allIn ? share.raw : share.real;
		const pot = 2 * n.commit;
		return [0, 1].map((p) => {
			const got = times(m[p], reach[1 - p]);
			const paid = ones(reach[1 - p]);
			return got.map((x, h) => x * pot - n.commit * paid[h]);
		}) as [Float64Array, Float64Array];
	}

	function cfr(
		n: Node,
		reach: [Float64Array, Float64Array],
		t: number
	): [Float64Array, Float64Array] {
		if (n.kind !== 'decision') return terminal(n, reach);
		const st = states.get(n.id)!;
		const sigma = strategy(st);
		const p = n.player;
		const value: [Float64Array, Float64Array] = [new Float64Array(N), new Float64Array(N)];
		const actionValues: Float64Array[] = [];
		n.children.forEach((child, a) => {
			const r: [Float64Array, Float64Array] = [reach[0], reach[1]];
			r[p] = reach[p].map((x, h) => x * sigma[a][h]);
			const v = cfr(child, r, t);
			actionValues.push(v[p]);
			for (let h = 0; h < N; h++) {
				value[p][h] += sigma[a][h] * v[p][h];
				value[1 - p][h] += v[1 - p][h];
			}
		});
		for (let a = 0; a < n.children.length; a++) {
			for (let h = 0; h < N; h++) {
				st.regret[a][h] = Math.max(0, st.regret[a][h] + actionValues[a][h] - value[p][h]);
				st.sum[a][h] += t * reach[p][h] * sigma[a][h]; // linear averaging
			}
		}
		return value;
	}

	function averaged(st: State): Float64Array[] {
		const k = st.sum.length;
		const out = st.sum.map(() => new Float64Array(N));
		for (let h = 0; h < N; h++) {
			let s = 0;
			for (let a = 0; a < k; a++) s += st.sum[a][h];
			for (let a = 0; a < k; a++) out[a][h] = s > 0 ? st.sum[a][h] / s : 1 / k;
		}
		return out;
	}

	/** Value vector of a best response for `hero` against the averaged strategy of the other player. */
	function bestResponse(n: Node, hero: 0 | 1, reachOpp: Float64Array): Float64Array {
		const ones = new Float64Array(N).fill(1);
		const reach = (r: Float64Array): [Float64Array, Float64Array] =>
			hero === 0 ? [ones, r] : [r, ones];
		if (n.kind !== 'decision') return terminal(n, reach(reachOpp))[hero];
		if (n.player === hero) {
			let best: Float64Array | null = null;
			for (const child of n.children) {
				const v = bestResponse(child, hero, reachOpp);
				best = best ? best.map((x, h) => Math.max(x, v[h])) : v;
			}
			return best!;
		}
		const sigma = averaged(states.get(n.id)!);
		const out = new Float64Array(N);
		n.children.forEach((child, a) => {
			const v = bestResponse(
				child,
				hero,
				reachOpp.map((x, h) => x * sigma[a][h])
			);
			for (let h = 0; h < N; h++) out[h] += v[h];
		});
		return out;
	}

	for (let t = 1; t <= ITERATIONS; t++) {
		cfr(root, [new Float64Array(N).fill(1), new Float64Array(N).fill(1)], t);
	}

	// Exploitability: what a best response wins in total from both seats (0 for an exact solution).
	const ones = new Float64Array(N).fill(1);
	const gain = ([0, 1] as const).map((p) => {
		const v = bestResponse(root, p, ones);
		return v.reduce((s, x, h) => s + (prior[h] * x) / wsum[h], 0);
	});

	// Reach of each decision under the averaged strategy, to drop nodes nobody gets to.
	const nodes: Record<string, unknown> = {};
	(function emit(n: Node, reach: [Float64Array, Float64Array]) {
		if (n.kind !== 'decision') return;
		const sigma = averaged(states.get(n.id)!);
		const mine = reach[n.player];
		const played = mine.reduce((s, x, h) => s + prior[h] * x, 0);
		if (played < 0.0005) return;
		const freq: Record<string, number[]> = {};
		ALL_HANDS.forEach((hand, h) => {
			if (mine[h] < 0.01) return;
			freq[hand] = sigma.map((row) => Math.round(row[h] * 100) / 100);
		});
		nodes[n.id] = {
			player: n.player === 0 ? 'SB' : 'BB',
			actions: n.labels,
			reach: Math.round(played * 1000) / 1000,
			freq
		};
		n.children.forEach((child, a) => {
			const r: [Float64Array, Float64Array] = [reach[0], reach[1]];
			r[n.player] = mine.map((x, h) => x * sigma[a][h]);
			emit(child, r);
		});
	})(root, [ones, ones]);
	return { nodes, exploitability: gain[0] + gain[1] };
}

const started = Date.now();
const equity = equityMatrix(samples);
const w = conflictFreeWeights();
for (const stack of STACKS) {
	const t0 = Date.now();
	const { nodes, exploitability } = solve(stack, equity, w);
	const file = `src/lib/poker/data/headsup-${stack}.json`;
	writeFileSync(
		file,
		JSON.stringify({
			stack_bb: stack,
			iterations: ITERATIONS,
			samples_per_matchup: samples,
			exploitability_bb: Math.round(exploitability * 1000) / 1000,
			nodes
		})
	);
	console.log(
		`${stack}bb: ${Object.keys(nodes).length} nodes, exploitability ${exploitability.toFixed(3)} bb, ${Math.round((Date.now() - t0) / 1000)}s -> ${file}`
	);
}
console.log(`done in ${Math.round((Date.now() - started) / 1000)}s`);

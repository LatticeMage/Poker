import { describe, expect, it } from 'vitest';
import type { Action, Persona } from '$lib/types/game';
import { decide } from './bot';
import { parseCards } from './cards';
import { HEADS_UP_STACKS, headsUpChart } from './charts';
import { applyAction, createGame, startHand } from './engine';
import {
	headsUpDist,
	nearestDepth,
	nodePath,
	openFreq,
	pushFoldFreq,
	scaleMass,
	scaledOpenFreq,
	share,
	widthOf
} from './mixed';
import { ALL_HANDS, classCombos, seatRange } from './ranges';
import { mulberry32 } from './rng';
import { vsOpen, vs3bet } from './spots';
import { BALANCED, FOLD, raise } from './testing';

const BB = 20; // the blinds are 10/20

const persona = (tight: number): Persona[] =>
	BALANCED.map((p) => ({ ...p, style: `tight ${tight}`, tight }));

/** The action as a label: fold, call (a check counts), raise, or allin. */
function label(a: Action, bet: number, chips: number): string {
	if (a.type === 'fold') return 'fold';
	if (a.type === 'raise') return a.amount >= bet + chips ? 'allin' : 'raise';
	return 'call';
}

type Counts = Record<string, number>;

/** Run `n` hands of `setup` and count what the player to act does with `hole`. */
function sample(
	players: number,
	chips: number,
	script: (Action | number)[],
	hole: string | null,
	n: number,
	options: { seed?: number; bots?: Persona[] } = {}
): Counts {
	const rng = mulberry32(options.seed ?? 1);
	const counts: Counts = {};
	for (let i = 0; i < n; i++) {
		const g = createGame({ players, chips, bots: options.bots ?? BALANCED }, rng);
		startHand(g, rng);
		for (const step of script) applyAction(g, typeof step === 'number' ? raise(step) : step);
		const seat = g.to_act;
		if (hole) g.players[seat].hole = parseCards(hole);
		const p = g.players[seat];
		const k = label(decide(g, seat, rng), p.bet, p.chips);
		counts[k] = (counts[k] ?? 0) + 1;
	}
	return counts;
}

/** Three standard errors of a frequency `p` seen in `n` samples. */
const tol = (p: number, n: number) => 3 * Math.sqrt((p * (1 - p)) / n) + 0.005;

/** A hand whose largest action share at a node lies between 40% and 80%, so it is truly mixed. */
function mixedHand(depth: number, path: string): string {
	const node = headsUpChart(depth).nodes[path];
	for (const name of ALL_HANDS) {
		const f = node.freq[name];
		if (f && Math.max(...f) < 0.8 && Math.max(...f) > 0.4) return name;
	}
	throw new Error(`no mixed hand at ${depth}/${path}`);
}

/** Two cards of a class, as text. */
const RANKS = '23456789TJQKA';
const SUITS = 'shdc';
function holeText(name: string): string {
	const [a, b] = classCombos(name)[0];
	const text = (c: number) => RANKS[c >> 2] + SUITS[c & 3];
	return `${text(a)} ${text(b)}`;
}

describe('scaling a mixed range', () => {
	const mix = openFreq(6, 3); // the button
	it('keeps the share exact and the order of strength', () => {
		const base = share(mix);
		expect(base).toBeGreaterThan(0.3);
		for (const factor of [0.5, 0.8, 1.2, 1.6]) {
			expect(share(scaleMass(mix, factor))).toBeCloseTo(base * factor, 5);
		}
		const narrow = scaleMass(mix, 0.7);
		const wide = scaleMass(mix, 1.3);
		for (const name of ALL_HANDS) {
			expect(narrow[name] ?? 0).toBeLessThanOrEqual(mix[name] ?? 0);
			expect(wide[name] ?? 0).toBeGreaterThanOrEqual(mix[name] ?? 0);
		}
		expect(share(scaleMass(mix, 100))).toBe(1);
	});

	it('widens by the persona width', () => {
		expect(widthOf(0.5)).toBe(1);
		expect(widthOf(0.95)).toBeLessThan(0.75);
		expect(widthOf(0.2)).toBeGreaterThan(1.15);
		expect(share(scaledOpenFreq(6, 3, 1))).toBeCloseTo(share(mix));
	});
});

describe('the preflop first-in mix', () => {
	it('the share of hands each seat raises matches the chart, hand by hand and overall', () => {
		for (const players of [3, 6, 9]) {
			for (let index = 0; index < players - 1; index++) {
				// the mix keeps the chart's share (the smoothing only moves hands around the edge)
				expect(
					Math.abs(100 * share(openFreq(players, index)) - seatRange(players, index).pct)
				).toBeLessThan(2.5);
			}
		}
	});

	it('an edge hand opens exactly as often as its frequency', () => {
		const mix = openFreq(6, 2); // the cutoff
		const edge = Object.entries(mix).filter(([, f]) => f >= 0.3 && f <= 0.7);
		expect(edge.length).toBeGreaterThan(0);
		for (const [name, f] of edge.slice(0, 3)) {
			const n = 2000;
			const got = sample(6, 2000, [FOLD, FOLD], holeText(name), n);
			expect((got.raise ?? 0) / n, name).toBeCloseTo(f, 1);
			expect(Math.abs((got.raise ?? 0) / n - f), name).toBeLessThan(tol(f, n));
		}
	});

	it('the voluntary rate of every seat matches the chart share, over random hands', () => {
		for (const players of [6, 9]) {
			const rng = mulberry32(100 + players);
			const hands = 4000;
			for (let index = 0; index < players - 1; index++) {
				let opened = 0;
				for (let h = 0; h < hands; h++) {
					const g = createGame({ players, chips: 2000, bots: BALANCED }, rng);
					startHand(g, rng);
					for (let i = 0; i < index; i++) applyAction(g, FOLD);
					opened += decide(g, g.to_act, rng).type === 'raise' ? 1 : 0;
				}
				const want = share(openFreq(players, index));
				expect(Math.abs(opened / hands - want), `${players} players, seat ${index}`).toBeLessThan(
					tol(want, hands)
				);
			}
		}
	});

	it('personas widen and narrow the same range', () => {
		const n = 4000;
		const rate = (tight: number) => {
			const got = sample(6, 2000, [FOLD, FOLD], null, n, { bots: persona(tight), seed: 5 });
			return (got.raise ?? 0) / n;
		};
		const nit = rate(0.95);
		const normal = rate(0.5);
		const loose = rate(0.2);
		expect(nit).toBeLessThan(normal);
		expect(normal).toBeLessThan(loose);
		for (const [got, tight] of [
			[nit, 0.95],
			[normal, 0.5],
			[loose, 0.2]
		] as const) {
			const want = share(scaledOpenFreq(6, 2, widthOf(tight)));
			expect(Math.abs(got - want)).toBeLessThan(tol(want, n));
		}
	});

	it('is reproducible: the same seed gives the same decisions', () => {
		const run = () => {
			const rng = mulberry32(42);
			const out: string[] = [];
			for (let i = 0; i < 300; i++) {
				const g = createGame({ players: 6, chips: 2000, bots: BALANCED }, rng);
				startHand(g, rng);
				out.push(JSON.stringify(decide(g, g.to_act, rng)));
			}
			return out;
		};
		expect(run()).toEqual(run());
	});
});

describe('facing an open, a 3-bet: sampled from the chart', () => {
	it('a mixed hand facing an open calls, 3-bets and folds at its shares', () => {
		const spot = vsOpen('BTN', 'CO')!;
		const name = Object.keys(spot.call).find(
			(h) => (spot.call[h] ?? 0) > 0.2 && (spot.bluff3[h] ?? 0) + (spot.value3[h] ?? 0) > 0.2
		)!;
		expect(name).toBeDefined();
		const n = 3000;
		const got = sample(6, 2000, [FOLD, FOLD, 50], holeText(name), n);
		const three = (spot.bluff3[name] ?? 0) + (spot.value3[name] ?? 0);
		expect(Math.abs((got.raise ?? 0) / n - three)).toBeLessThan(tol(three, n));
		expect(Math.abs((got.call ?? 0) / n - spot.call[name])).toBeLessThan(tol(spot.call[name], n));
	});

	it('an opener facing a 3-bet 4-bets and calls at the chart shares', () => {
		const spot = vs3bet('CO', 'BTN')!;
		const name = Object.keys(spot.call).find(
			(h) => (spot.call[h] ?? 0) > 0.2 && (spot.bluff4[h] ?? 0) + (spot.value4[h] ?? 0) > 0.2
		)!;
		expect(name).toBeDefined();
		const n = 3000;
		const got = sample(6, 2000, [FOLD, FOLD, 50, 160, FOLD, FOLD], holeText(name), n);
		const four = (spot.bluff4[name] ?? 0) + (spot.value4[name] ?? 0);
		expect(Math.abs((got.raise ?? 0) / n - four)).toBeLessThan(tol(four, n));
		expect(Math.abs((got.call ?? 0) / n - spot.call[name])).toBeLessThan(tol(spot.call[name], n));
	});

	it('a loose persona defends more and a tight one less than the chart', () => {
		const n = 4000;
		const calls = (tight: number) => {
			const got = sample(6, 2000, [FOLD, FOLD, 50], null, n, { bots: persona(tight), seed: 9 });
			return (got.call ?? 0) / n;
		};
		expect(calls(0.95)).toBeLessThan(calls(0.5));
		expect(calls(0.5)).toBeLessThan(calls(0.2));
	});
});

describe('heads-up: the solved trees', () => {
	it('picks the nearest solved depth', () => {
		const depths = [...HEADS_UP_STACKS].sort((a, b) => a - b);
		expect(depths).toEqual([5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150]);
		for (const d of depths) expect(nearestDepth(d)).toBe(d);
		// in log terms: 25 is closer to 20 than to 30 (ratios 1.25 and 1.2 the other way round)
		expect(nearestDepth(24)).toBe(20);
		expect(nearestDepth(25)).toBe(30);
		expect(nearestDepth(14)).toBe(15);
		expect(nearestDepth(70)).toBe(80);
		expect(nearestDepth(120)).toBe(100);
		expect(nearestDepth(300)).toBe(150);
	});

	it('names the node from the betting so far', () => {
		const rng = mulberry32(3);
		const make = (script: number[]) => {
			const g = createGame({ players: 2, chips: 800, bots: BALANCED }, rng); // 40bb
			startHand(g, rng);
			for (const amount of script) applyAction(g, raise(amount));
			return nodePath(g, g.to_act);
		};
		expect(make([])).toBe('root');
		expect(make([50])).toBe('open/2.5');
		expect(make([50, 180])).toBe('open/2.5/9');
		expect(make([50, 180, 440])).toBe('open/2.5/9/22');
		expect(make([50, 800])).toBe('open/2.5/allin');
		expect(make([800])).toBe('open/allin');
		expect(make([100])).toBeNull(); // a 5bb open is not a size the tree has
	});

	for (const depth of [10, 20, 40, 80]) {
		it(`${depth}bb: the small blind's first action follows the root node`, () => {
			const chips = depth * BB;
			const node = headsUpChart(depth).nodes.root;
			const n = 4000;
			const hands = ALL_HANDS;
			// expected share of each action over all hands, weighted by combos
			const want: Record<string, number> = {};
			let total = 0;
			for (const h of hands) {
				const w = classCombos(h).length;
				total += w;
				node.actions.forEach((a, i) => (want[a] = (want[a] ?? 0) + w * (node.freq[h]?.[i] ?? 0)));
			}
			const got = sample(2, chips, [], null, n, { seed: depth });
			const folds = (got.fold ?? 0) / n;
			expect(Math.abs(folds - want.fold / total)).toBeLessThan(tol(folds, n));
			const raises = ((got.raise ?? 0) + (got.allin ?? 0)) / n;
			const wantRaises = (want['raise 2.5'] + want.allin) / total;
			expect(Math.abs(raises - wantRaises)).toBeLessThan(tol(wantRaises, n));
			const limps = (got.call ?? 0) / n;
			expect(Math.abs(limps - want.limp / total)).toBeLessThan(tol(limps, n));
		});
	}

	it('a mixed hand facing each kind of node follows the tree', () => {
		const cases: [number, string, number[]][] = [
			[20, 'root', []],
			[20, 'open/2.5', [50]],
			[40, 'open/2.5/9', [50, 180]],
			[40, 'open/2.5/9/22', [50, 180, 440]],
			[20, 'limp', [20]],
			[40, 'limp/4', [20, 80]]
		];
		for (const [depth, path, script] of cases) {
			const name = mixedHand(depth, path);
			const dist = headsUpDist(depth, path, name)!;
			const n = 3000;
			const steps = script.map((x) => (x === 20 ? ({ type: 'call' } as Action) : x));
			const got = sample(2, depth * BB, steps, holeText(name), n);
			const call = (got.call ?? 0) / n;
			const wantCall = (dist.call ?? 0) + (dist.limp ?? 0) + (dist.check ?? 0);
			const fold = (got.fold ?? 0) / n;
			const wantRaise = 1 - wantCall - (dist.fold ?? 0);
			const raised = ((got.raise ?? 0) + (got.allin ?? 0)) / n;
			expect(Math.abs(call - wantCall), `${depth}/${path} ${name} call`).toBeLessThan(
				tol(wantCall, n)
			);
			expect(Math.abs(fold - (dist.fold ?? 0)), `${depth}/${path} ${name} fold`).toBeLessThan(
				tol(dist.fold ?? 0, n)
			);
			expect(Math.abs(raised - wantRaise), `${depth}/${path} ${name} raise`).toBeLessThan(
				tol(wantRaise, n)
			);
		}
	});

	it('a big blind facing a jam calls with the solved share', () => {
		const name = mixedHand(10, 'open/allin');
		const dist = headsUpDist(10, 'open/allin', name)!;
		const n = 3000;
		const got = sample(2, 200, [200], holeText(name), n);
		expect(Math.abs((got.call ?? 0) / n - dist.call)).toBeLessThan(tol(dist.call, n));
	});

	it('personas change how wide the small blind plays', () => {
		const n = 3000;
		const play = (tight: number) => {
			const got = sample(2, 800, [], null, n, { bots: persona(tight), seed: 4 });
			return 1 - (got.fold ?? 0) / n;
		};
		expect(play(0.95)).toBeLessThan(play(0.5) - 0.05);
		expect(play(0.5)).toBeLessThan(play(0.2) - 0.03);
	});

	it('is reproducible with a fixed seed', () => {
		const a = sample(2, 800, [], null, 200, { seed: 77 });
		const b = sample(2, 800, [], null, 200, { seed: 77 });
		expect(a).toEqual(b);
		const c = sample(2, 800, [], null, 200, { seed: 78 });
		expect(c).not.toEqual(a);
	});

	it('plays the push/fold mix below 7bb', () => {
		const n = 3000;
		const pushMix = pushFoldFreq(5, 'push');
		const name = Object.entries(pushMix).find(([, f]) => f > 0.03 && f < 0.99)![0];
		const got = sample(2, 100, [], holeText(name), n); // 5bb
		const f = pushMix[name];
		expect(Math.abs((got.allin ?? 0) / n - f)).toBeLessThan(tol(f, n));
		const callMix = pushFoldFreq(5, 'call');
		const caller = Object.entries(callMix).find(([, f2]) => f2 > 0.03 && f2 < 0.99)![0];
		const c = sample(2, 100, [100], holeText(caller), n);
		expect(Math.abs((c.call ?? 0) / n - callMix[caller])).toBeLessThan(tol(callMix[caller], n));
	});
});

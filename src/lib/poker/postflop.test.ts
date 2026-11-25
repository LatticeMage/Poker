// Heads-up postflop play: ranges from the preflop charts, and the bet / defend rules built on them.

import { describe, expect, it } from 'vitest';
import type { Action } from '$lib/types/game';
import { decide as botDecide } from './bot';
import { parseCards } from './cards';
import { advance, applyAction } from './engine';
import * as postflop from './postflop';
import { chartHands, seatRange } from './ranges';
import { mulberry32 } from './rng';
import * as spots from './spots';
import {
	BB,
	BTN,
	BTN_VS_BB,
	CALL,
	CHECK,
	CO,
	CO_3BET_BTN,
	FOLD,
	HJ,
	SB,
	UTG,
	raise,
	toFlop
} from './testing';

const DRY = 'As 7d 2c';
const WET = '7h 6h 5d';
const N = 120;

function profileFor(
	g: ReturnType<typeof toFlop>['g'],
	order: number[],
	position: number,
	hole: string
) {
	const [mine, theirs] = postflop.headsUpRanges(g, order[position], order)!;
	return { prof: postflop.profile(g.board, parseCards(hole), mine, theirs)!, mine, theirs };
}

interface DecisionOptions {
	pot: number;
	toCall: number;
	aggressor: boolean;
	n?: number;
	seed?: number;
	hole?: string;
	facingRaise?: boolean;
}

/** Decisions for hands drawn from the seat's own range (or one fixed hand). */
function decisions(
	g: ReturnType<typeof toFlop>['g'],
	order: number[],
	position: number,
	o: DecisionOptions
) {
	const rng = mulberry32(o.seed ?? 2);
	const [mine, theirs] = postflop.headsUpRanges(g, order[position], order)!;
	const out: postflop.Decision[] = [];
	for (let i = 0; i < (o.n ?? N); i++) {
		const hand = o.hole ? parseCards(o.hole) : postflop.sampleHand(mine, g.board, rng)!;
		const prof = postflop.profile(g.board, hand, mine, theirs)!;
		out.push(
			postflop.decide(
				prof,
				{
					texture: postflop.boardTexture(g.board),
					pot: o.pot,
					to_call: o.toCall,
					can_raise: true,
					aggressor: o.aggressor,
					facing_raise: o.facingRaise ?? false,
					persona_aggro: 0.5,
					persona_bluff: 0.25,
					persona_tight: 0.5
				},
				rng
			)
		);
	}
	return out;
}

const share = (items: postflop.Decision[], pred: (d: postflop.Decision) => boolean) =>
	items.filter(pred).length / items.length;

describe('formulas', () => {
	it('classifies board textures', () => {
		const cases: Record<string, string> = {
			'As 7d 2c': 'dry',
			'Kc 8d 3h': 'dry',
			'7h 6h 5d': 'wet',
			'Jc Td 9s': 'wet',
			'Qh 8h 3h': 'monotone',
			'Ks Kd 4c': 'paired',
			'9c 9d 9s': 'paired',
			'Ah 8h 2c': 'dry', // a flush draw alone does not make an ace-high board wet
			'Kh 9h 6c': 'dry', // counted 13%: a flush draw and a few gutshots, no straight to speak of
			'Ac 5d 4h': 'wet' // the wheel: ace plays low, three cards in one straight
		};
		for (const [text, want] of Object.entries(cases)) {
			expect(postflop.boardTexture(parseCards(text)), text).toBe(want);
		}
	});

	it('bluff, defence and size formulas', () => {
		expect(postflop.bluffShare(0.33)).toBeCloseTo(0.199, 3);
		expect(postflop.bluffShare(1)).toBeCloseTo(1 / 3);
		expect(postflop.defendFrequency(15, 5)).toBeCloseTo(2 / 3); // a half-pot bet into 10
		expect(postflop.defendFrequency(20, 10)).toBeCloseTo(0.5); // a pot-sized bet
		expect(postflop.betSize(0.57, 'dry')).toBe(postflop.SMALL_BET);
		expect(postflop.betSize(0.5, 'wet')).toBe(postflop.BIG_BET);
		expect(postflop.betSize(0.5, 'dry')).toBe(postflop.SMALL_BET); // static board, even ranges
		expect(postflop.betSize(0.61, 'wet')).toBe(postflop.SMALL_BET); // clear advantage
		// with the nut edge known: a big bet needs the nuts on your side
		expect(postflop.betSize(0.5, 'wet', 0.02)).toBe(postflop.BIG_BET);
		expect(postflop.betSize(0.5, 'wet', -0.05)).toBe(postflop.SMALL_BET);
		expect(postflop.betSize(0.46, 'dry', 0.01)).toBe(postflop.SMALL_BET); // behind, static, no real edge
		expect(postflop.betSize(0.46, 'dry', 0.08)).toBe(postflop.BIG_BET);
		// a monotone board never takes the big bet: neither range holds more of the nuts there
		expect(postflop.betSize(0.5, 'monotone')).toBe(postflop.SMALL_BET);
		expect(postflop.betSize(0.5, 'monotone', 0.05)).toBe(postflop.SMALL_BET);
		expect(postflop.betSize(0.44, 'monotone', 0.05)).toBe(postflop.SMALL_BET);
		expect(postflop.betFrequency(0.52, true, 'dry', 0.1)).toBeGreaterThan(
			postflop.betFrequency(0.52, true, 'dry', -0.1)
		);
		expect(postflop.betFrequency(0.58, true, 'dry')).toBeGreaterThan(
			postflop.betFrequency(0.52, true, 'dry')
		);
		expect(postflop.betFrequency(0.55, true, 'dry')).toBeGreaterThan(
			postflop.betFrequency(0.55, true, 'monotone')
		);
		expect(postflop.betFrequency(0.55, false, 'dry')).toBeLessThan(
			postflop.betFrequency(0.55, true, 'dry')
		);
	});

	it('judges a made hand against the board, on boards of any length', () => {
		const paired = parseCards('Ks Kd 4c');
		expect(postflop.hasMadeHand(parseCards('Qh Jh'), paired)).toBe(false); // the board's pair is not yours
		expect(postflop.hasMadeHand(parseCards('Kh 9c'), paired)).toBe(true);
		expect(postflop.hasMadeHand(parseCards('9h 9c'), paired)).toBe(true);
		const turn = parseCards('As 7d 2c Kd');
		expect(postflop.hasMadeHand(parseCards('Kh 5c'), turn)).toBe(true);
		expect(postflop.hasMadeHand(parseCards('Qh 5c'), turn)).toBe(false);
		const river = parseCards('As 7d 2c Kd Kc');
		expect(postflop.hasMadeHand(parseCards('Qh 5c'), river)).toBe(false);
		expect(postflop.hasMadeHand(parseCards('Ah 5c'), river)).toBe(true);
		expect(postflop.boardTexture(river)).toBe(postflop.boardTexture(parseCards('As 7d 2c')));
	});
});

describe('the flop read', () => {
	it('is keyed by the matchup and only exists on the flop of a 6-max pot', () => {
		const { g, order } = toFlop(BTN_VS_BB);
		const [, , aggressor, key] = postflop.headsUpRanges(g, order[BTN], order)!;
		expect(key).toBe('srp:BTN:BB');
		const read = postflop.flopRead(key, g.board, aggressor)!;
		expect(read.group.name).toBe('三花｜1高｜低斷｜不同順｜A｜中小');
		expect(postflop.flopRead(key, [...g.board, ...parseCards('Kd')], aggressor)).toBeNull();
		expect(postflop.flopRead(null, g.board, aggressor)).toBeNull();
		expect(postflop.flopRead('srp:BB:UTG', g.board, aggressor)).toBeNull();
	});

	it('gives the two players the two sides of the same numbers', () => {
		const board = parseCards(WET);
		const a = postflop.flopRead('srp:CO:BB', board, true)!;
		const c = postflop.flopRead('srp:CO:BB', board, false)!;
		expect(a.range_eq + c.range_eq).toBeCloseTo(1, 6);
		expect(a.nut_mine).toBe(c.nut_theirs);
		expect(a.range_eq_min).toBeLessThanOrEqual(a.range_eq);
		expect(a.range_eq_max).toBeGreaterThanOrEqual(a.range_eq);
		expect(c.range_eq_min).toBeCloseTo(1 - a.range_eq_max, 6);
	});

	it('agrees with the sampled range equity of the actual board', () => {
		// The table is exact for the group (flops of one group differ by about a point), the profile is
		// sampled on this very flop (a few points of noise). If the ranges or the sides of the table and
		// of the app did not match, they would be far apart.
		const diffs: number[] = [];
		for (const seat of [BTN, BB]) {
			for (const text of [DRY, WET, 'Kc 8d 3h', 'Qh Jh 4s', '9c 9d 4s', 'Ts 6s 2s']) {
				const { g, order } = toFlop(BTN_VS_BB, text);
				const [mine, theirs, aggressor, key] = postflop.headsUpRanges(g, order[seat], order)!;
				const read = postflop.flopRead(key, g.board, aggressor)!;
				const hand = postflop.sampleHand(mine, g.board, mulberry32(1))!;
				const prof = postflop.profile(g.board, hand, mine, theirs)!;
				diffs.push(Math.abs(read.range_eq - prof.range_eq));
			}
		}
		expect(Math.max(...diffs)).toBeLessThan(0.07);
		expect(diffs.reduce((s, d) => s + d, 0) / diffs.length).toBeLessThan(0.03);
	});

	it('the plan stands on the read when there is one', () => {
		const { g, order } = toFlop(BTN_VS_BB);
		const { prof } = profileFor(g, order, BTN, 'Ah Kd');
		const base = {
			texture: 'dry' as const,
			pot: 6,
			to_call: 0,
			aggressor: true,
			facing_raise: false
		};
		const read = { ...postflop.flopRead('srp:BTN:BB', g.board, true)!, solved: null };
		const strong = postflop.makePlan(prof, {
			...base,
			flop: { ...read, range_eq: 0.6, nut_mine: 0.3, nut_theirs: 0.1 }
		});
		const weak = postflop.makePlan(prof, {
			...base,
			flop: { ...read, range_eq: 0.45, nut_mine: 0.05, nut_theirs: 0.2 }
		});
		expect(strong.frequency).toBeGreaterThan(weak.frequency + 0.3);
	});

	it('where the flop is solved, the solved numbers make the plan', () => {
		const { g, order } = toFlop(BTN_VS_BB);
		const { prof } = profileFor(g, order, BTN, 'Ah Kd');
		const read = postflop.flopRead('srp:BTN:BB', g.board, true)!;
		const solved = {
			bet33: 0.4,
			bet66: 0.2,
			value: 0.55,
			fold33: 0.3,
			raise33: 0.1,
			fold66: 0.5,
			raise66: 0.05
		};
		const base = { texture: 'dry' as const, aggressor: true, facing_raise: false };
		const first = postflop.makePlan(prof, {
			...base,
			pot: 6,
			to_call: 0,
			flop: { ...read, solved }
		});
		expect(first.frequency).toBeCloseTo(0.6);
		expect(first.size).toBe(postflop.SMALL_BET); // the size the range uses more
		expect(first.value_cut).toBeCloseTo(0.6 * 0.55);
		expect(first.bluff_quota).toBeCloseTo(0.6 * 0.45);
		const big = postflop.makePlan(prof, {
			...base,
			pot: 6,
			to_call: 0,
			flop: { ...read, solved: { ...solved, bet33: 0.1, bet66: 0.3 } }
		});
		expect(big.size).toBe(postflop.BIG_BET);
		// facing a bet: the solved folding share for that size, between the two sizes in proportion
		expect(postflop.solvedFold(solved, 4 + 0.33 * 4, 0.33 * 4)).toBeCloseTo(0.3);
		expect(postflop.solvedFold(solved, 4 + 0.66 * 4, 0.66 * 4)).toBeCloseTo(0.5);
		expect(postflop.solvedFold(solved, 4 + 0.495 * 4, 0.495 * 4)).toBeCloseTo(0.4);
		expect(postflop.solvedFold(solved, 4 + 2 * 4, 2 * 4)).toBeCloseTo(0.5); // beyond: the nearer size
		const facing = postflop.makePlan(prof, {
			...base,
			pot: 4 + 0.66 * 4,
			to_call: 0.66 * 4,
			flop: { ...read, solved }
		});
		expect(facing.defend).toBeCloseTo(0.5);
		// the real table: the button bets a dry ace-high flop often, the big blind folds more to the bigger bet
		const bb = postflop.flopRead('srp:BTN:BB', g.board, false)!;
		expect(read.solved).not.toBeNull();
		expect(read.solved!.bet33 + read.solved!.bet66).toBeGreaterThan(0.4);
		expect(read.solved!.bet33).toBeGreaterThan(read.solved!.bet66); // a third of the pot is the main size there
		expect(bb.solved!.fold66).toBeGreaterThan(bb.solved!.fold33);
		expect(bb.solved!.bet33 + bb.solved!.bet66).toBeLessThan(0.2); // the caller rarely leads
		expect(postflop.flopRead('3bet:BTN:BB', g.board, true)!.solved).toBeNull(); // not solved yet: the rules
	});
});

describe('more than two players: tightened ranges', () => {
	it('the share comes from how the opening charts narrow', () => {
		expect(postflop.potTightness(2)).toBe(1);
		expect(postflop.potTightness(3)).toBeCloseTo(0.79, 1);
		for (let n = 3; n <= 9; n++)
			expect(postflop.potTightness(n)).toBeLessThan(postflop.potTightness(n - 1));
		expect(postflop.potTightness(6)).toBeGreaterThan(0.15);
	});

	it('tightening keeps the strongest hands, by combos', () => {
		const range = spots.vsOpen('BB', 'BTN')!.call;
		const size = (r: postflop.Weighted) => postflop.rangeCombos(r, []);
		expect(postflop.tighten(range, 1)).toBe(range);
		const half = postflop.tighten(range, 0.5);
		expect(size(half)).toBeCloseTo(0.5 * size(range), 6);
		expect(half['99']).toBe(range['99']); // strong hands stay whole
		expect(half['T8o']).toBeUndefined(); // the weakest go
		for (const [h, w] of Object.entries(half)) expect(w).toBeLessThanOrEqual(range[h] + 1e-12);
		const third = postflop.tighten(range, 1 / 3);
		for (const h of Object.keys(third)) expect(half[h]).toBeGreaterThan(0); // nested
	});

	it('a three-way single-raised pot uses tightened ranges for every player', () => {
		// CO opens, BTN and BB call
		const { g, order } = toFlop([
			[UTG, FOLD],
			[HJ, FOLD],
			[CO, raise(50)],
			[BTN, CALL],
			[SB, FOLD],
			[BB, CALL]
		]);
		expect(postflop.headsUpRanges(g, order[CO], order)).toBeNull();
		const share = postflop.potTightness(3);
		const size = (r: postflop.Weighted) => postflop.rangeCombos(r, []);
		const [mine, theirs, opened] = postflop.multiwayRanges(g, order[CO], order)!;
		expect(opened).toBe(true);
		const open = Object.fromEntries([...seatRange(6, CO).hands].map((h) => [h, 1]));
		expect(size(mine)).toBeCloseTo(share * size(open), 6);
		expect(mine['AA']).toBe(1);
		// "they" are the two callers put together, tightened the same way
		const btn = spots.vsOpen('BTN', 'CO')!.call;
		const bb = spots.vsOpen('BB', 'CO')!.call;
		expect(size(theirs)).toBeCloseTo((share * (size(btn) + size(bb))) / 2, 6);
		const [m2, , o2] = postflop.multiwayRanges(g, order[BB], order)!;
		expect(o2).toBe(false);
		expect(size(m2)).toBeCloseTo(share * size(bb), 6);
		// two players: no multiway model
		const hu = toFlop(BTN_VS_BB);
		expect(postflop.multiwayRanges(hu.g, hu.order[BTN], hu.order)).toBeNull();
	});
});

describe('ranges', () => {
	it('single-raised pot ranges come from the charts', () => {
		const { g, order } = toFlop(BTN_VS_BB);
		const [mine, theirs, aggressor] = postflop.headsUpRanges(g, order[BTN], order)!;
		expect(aggressor).toBe(true);
		expect(new Set(Object.keys(mine))).toEqual(new Set(seatRange(6, BTN).hands));
		expect(new Set(Object.values(mine))).toEqual(new Set([1]));
		expect(theirs).toEqual(spots.vsOpen('BB', 'BTN')!.call);
		const [m2, t2, a2] = postflop.headsUpRanges(g, order[BB], order)!;
		expect(a2).toBe(false);
		expect(m2).toEqual(theirs);
		expect(t2).toEqual(mine);
	});

	it('3-bet pot ranges are the 3-bet and the call', () => {
		const { g, order } = toFlop(CO_3BET_BTN);
		const [mine, theirs, aggressor] = postflop.headsUpRanges(g, order[BTN], order)!;
		const raised = spots.vsOpen('BTN', 'CO')!;
		expect(aggressor).toBe(true);
		expect(mine.QQ).toBe(1);
		expect(mine.A5s).toBe(1);
		expect(mine.AA).toBeCloseTo(5 / 6);
		expect(new Set(Object.keys(mine))).toEqual(
			new Set([...Object.keys(raised.value3), ...Object.keys(raised.bluff3)])
		);
		expect(theirs).toEqual(spots.vs3bet('CO', 'BTN')!.call);
	});

	it('4-bet pot ranges are the 4-bet and the continue', () => {
		const { g, order } = toFlop([...CO_3BET_BTN.slice(0, 6), [CO, raise(400)], [BTN, CALL]]);
		const [mine, theirs, aggressor] = postflop.headsUpRanges(g, order[CO], order)!;
		const faced = spots.vs3bet('CO', 'BTN')!;
		expect(aggressor).toBe(true);
		expect(new Set(Object.keys(mine))).toEqual(
			new Set([...Object.keys(faced.value4), ...Object.keys(faced.bluff4)])
		);
		expect(theirs).toEqual(spots.vsOpen('BTN', 'CO')!.value3);
		expect(postflop.headsUpRanges(g, order[BTN], order)![2]).toBe(false);
	});

	it('has no model for limped, three-way or 5-bet pots', () => {
		const limped: [number, Action][] = [
			[UTG, FOLD],
			[HJ, FOLD],
			[CO, FOLD],
			[BTN, CALL],
			[SB, FOLD],
			[BB, CHECK]
		];
		const threeWay: [number, Action][] = [
			[UTG, FOLD],
			[HJ, FOLD],
			[CO, raise(50)],
			[BTN, CALL],
			[SB, FOLD],
			[BB, CALL]
		];
		const fiveBet: [number, Action][] = [
			...CO_3BET_BTN.slice(0, 6),
			[CO, raise(400)],
			[BTN, raise(1000)],
			[CO, CALL]
		];
		for (const script of [limped, threeWay, fiveBet]) {
			const { g, order } = toFlop(script);
			for (const seat of order) expect(postflop.headsUpRanges(g, seat, order)).toBeNull();
		}
	});
});

describe('profiles', () => {
	it('ranks hands and describes the range', () => {
		const { g, order } = toFlop(BTN_VS_BB, DRY);
		const nuts = profileFor(g, order, BTN, 'Ah Ad').prof; // top set
		expect(nuts.eq).toBeGreaterThan(0.9);
		expect(postflop.rankIn(nuts)).toBeLessThan(0.1);
		expect(nuts.nopair).toBe(false);
		const air = profileFor(g, order, BTN, 'Jh 3d').prof;
		expect(air.eq).toBeLessThan(0.45);
		expect(postflop.rankIn(air)).toBeGreaterThan(0.5);
		expect(air.nopair).toBe(true);
		// The range's own equity is a property of the board and the two ranges, not of your hand.
		expect(nuts.range_eq).toBe(air.range_eq);
		expect(nuts.refs).toEqual(air.refs);
		expect(nuts.range_eq).toBeGreaterThan(0.5);
	});

	it('turn and river have their own range statistics', () => {
		const flop = toFlop(BTN_VS_BB, DRY);
		const turn = toFlop(BTN_VS_BB, 'As 7d 2c Kh');
		const river = toFlop(BTN_VS_BB, 'As 7d 2c Kh 9s');
		const eqs = [flop, turn, river].map(({ g, order }) => {
			const [mine, theirs] = postflop.headsUpRanges(g, order[BTN], order)!;
			return postflop.equityCurve(g.board, mine, theirs)!;
		});
		const means = eqs.map((c) => c.reduce((s, v) => s + v, 0) / c.length);
		expect(new Set(means.map((m) => m.toFixed(3))).size).toBe(3); // different boards, different advantages
		expect(means.every((m) => m > 0.35 && m < 0.75)).toBe(true);
		for (const c of eqs) expect(c).toEqual([...c].sort((a, b) => a - b));
	});

	it('river equity is a showdown comparison', () => {
		const { g, order } = toFlop(BTN_VS_BB, 'As 7d 2c Kh 9s');
		const top = profileFor(g, order, BTN, 'Ah Ad').prof;
		const air = profileFor(g, order, BTN, 'Qd 3c').prof;
		expect(top.eq).toBeGreaterThan(0.95);
		expect(top.nopair).toBe(false);
		expect(air.eq).toBeLessThan(0.45);
		expect(air.nopair).toBe(true);
		const ds = decisions(g, order, BTN, { pot: g.pot, toCall: 0, aggressor: true, n: 100 });
		expect(ds.some((d) => d.reason === 'value')).toBe(true);
		expect(share(ds, (d) => d.kind === 'bet')).toBeGreaterThan(0.2);
	});
});

describe('betting', () => {
	it('the aggressor bets more often on a board it hits, and sizes by texture', () => {
		const dry = toFlop(BTN_VS_BB, DRY);
		const wet = toFlop(BTN_VS_BB, WET);
		const ds = decisions(dry.g, dry.order, BTN, { pot: dry.g.pot, toCall: 0, aggressor: true });
		const ws = decisions(wet.g, wet.order, BTN, { pot: dry.g.pot, toCall: 0, aggressor: true });
		const bets = (x: postflop.Decision[]) => share(x, (d) => d.kind === 'bet');
		expect(bets(ds)).toBeGreaterThan(bets(ws) + 0.08);
		expect(bets(ds)).toBeGreaterThan(0.45);
		expect(bets(ds)).toBeLessThan(0.8);
		expect(new Set(ds.filter((d) => d.kind === 'bet').map((d) => d.size))).toEqual(
			new Set([postflop.SMALL_BET])
		);
		expect(new Set(ws.filter((d) => d.kind === 'bet').map((d) => d.size))).toEqual(
			new Set([postflop.BIG_BET])
		);
	});

	it('bets carry the indifference share of bluffs', () => {
		const { g, order } = toFlop(BTN_VS_BB, DRY);
		const bets = decisions(g, order, BTN, {
			pot: g.pot,
			toCall: 0,
			aggressor: true,
			n: 200
		}).filter((d) => d.kind === 'bet');
		const bluffs = share(bets, (d) => d.reason === 'bluff');
		expect(Math.abs(bluffs - postflop.bluffShare(postflop.SMALL_BET))).toBeLessThan(0.09); // about 20%
	});

	it('the best hands bet and air checks', () => {
		const { g, order } = toFlop(BTN_VS_BB, DRY);
		const strong = decisions(g, order, BTN, {
			pot: g.pot,
			toCall: 0,
			aggressor: true,
			n: 40,
			hole: 'Ah Ad'
		});
		const weak = decisions(g, order, BTN, {
			pot: g.pot,
			toCall: 0,
			aggressor: true,
			n: 40,
			hole: '9s 3d'
		});
		expect(share(strong, (d) => d.kind === 'bet')).toBeGreaterThan(0.95);
		expect(share(weak, (d) => d.kind === 'check')).toBeGreaterThan(0.85);
	});

	it('a paired board still gets bluffs', () => {
		const { g, order } = toFlop(BTN_VS_BB, 'Ks Kd 4c');
		const bets = decisions(g, order, BTN, {
			pot: g.pot,
			toCall: 0,
			aggressor: true,
			n: 200
		}).filter((d) => d.kind === 'bet');
		expect(share(bets, (d) => d.reason === 'bluff')).toBeGreaterThan(0.1);
	});
});

describe('defending', () => {
	it('follows the minimum defence frequency, tighter against bigger bets', () => {
		const { g, order } = toFlop(BTN_VS_BB, DRY);
		const p0 = g.pot;
		const cont: Record<string, number> = {};
		for (const [name, bet] of [
			['third', p0 / 3],
			['half', p0 / 2],
			['pot', p0]
		] as const) {
			const ds = decisions(g, order, BB, { pot: p0 + bet, toCall: bet, aggressor: false, n: 150 });
			cont[name] = share(ds, (d) => d.kind !== 'fold');
		}
		expect(Math.abs(cont.half - postflop.defendFrequency(1.5 * p0, 0.5 * p0) * 0.92)).toBeLessThan(
			0.08
		); // ~61%
		expect(cont.third).toBeGreaterThan(cont.half);
		expect(cont.half).toBeGreaterThan(cont.pot);
		expect(Math.abs(cont.pot - 0.5 * 0.92)).toBeLessThan(0.08);
	});

	it('strong hands never fold and the best raise; trash folds', () => {
		const { g, order } = toFlop(BTN_VS_BB, DRY);
		const bet = g.pot / 2;
		const base = { pot: g.pot + bet, toCall: bet, aggressor: false, n: 40 };
		const sets = decisions(g, order, BB, { ...base, hole: '7h 7c' });
		expect(share(sets, (d) => d.kind === 'fold')).toBe(0);
		expect(share(sets, (d) => d.kind === 'raise')).toBeGreaterThan(0.3);
		const trash = decisions(g, order, BB, { ...base, hole: '9s 3d' });
		expect(share(trash, (d) => d.kind === 'fold')).toBeGreaterThan(0.7);
	});

	it('defends tighter against a raise', () => {
		// Hands cluster by strength (e.g. every weak ace), so on one board the realised share moves in
		// steps. The same hands are never looser against a raise, and across boards it is clearly tighter.
		const gaps: number[] = [];
		for (const board of [DRY, WET, 'Kc 9h 4d', 'Qs Qd 6c']) {
			const { g, order } = toFlop(BTN_VS_BB, board);
			const bet = g.pot / 2;
			const base = { pot: g.pot + bet, toCall: bet, aggressor: false, n: 150 };
			const plain = share(decisions(g, order, BB, base), (d) => d.kind !== 'fold');
			const raised = share(
				decisions(g, order, BB, { ...base, facingRaise: true }),
				(d) => d.kind !== 'fold'
			);
			expect(raised, board).toBeLessThanOrEqual(plain);
			gaps.push(plain - raised);
		}
		expect(gaps.reduce((s, x) => s + x, 0) / gaps.length).toBeGreaterThan(0.04);
	});
});

describe('in the bot', () => {
	it('plays the flop, turn and river from the model', () => {
		for (const board of ['As 7d 2c', 'As 7d 2c Kh', 'As 7d 2c Kh 9s']) {
			const kinds = new Set<string>();
			for (let seed = 0; seed < 40; seed++) {
				const { g, order } = toFlop(BTN_VS_BB, board, seed);
				applyAction(g, CHECK); // the big blind checks to the button
				if (g.players[order[BTN]].hole.some((c) => g.board.includes(c))) continue;
				kinds.add(botDecide(g, order[BTN], mulberry32(seed)).type);
			}
			expect(kinds, board).toEqual(new Set(['check', 'raise']));
		}
	});

	it('still acts where there is no range model', () => {
		const { g } = toFlop([
			[UTG, FOLD],
			[HJ, FOLD],
			[CO, raise(50)],
			[BTN, CALL],
			[SB, FOLD],
			[BB, CALL]
		]);
		expect(['check', 'raise', 'call', 'fold']).toContain(
			botDecide(g, g.to_act, mulberry32(4)).type
		);
	});

	it('plays a whole hand through all streets with the chips conserved', () => {
		for (let seed = 0; seed < 6; seed++) {
			const { g } = toFlop(BTN_VS_BB, DRY, seed);
			const bank = g.players.reduce((s, p) => s + p.chips + p.committed, 0);
			const rng = mulberry32(seed);
			while (g.phase === 'betting' || g.phase === 'dealing') {
				if (g.phase === 'dealing') advance(g);
				else applyAction(g, botDecide(g, g.to_act, rng));
				expect(g.players.reduce((s, p) => s + p.chips + p.committed, 0)).toBe(bank);
			}
			expect(['hand_over', 'game_over']).toContain(g.phase);
		}
	});

	it('decides fast enough to play a hand per click', () => {
		const { g, order } = toFlop(BTN_VS_BB, DRY);
		applyAction(g, CHECK);
		const rng = mulberry32(9);
		botDecide(g, order[BTN], rng); // the first call for a board builds the range statistics
		const started = performance.now();
		for (let i = 0; i < 20; i++) botDecide(g, order[BTN], rng);
		expect((performance.now() - started) / 20).toBeLessThan(50);
	});
});

describe('seat ranges used by the model', () => {
	it('the opener chart is the 6-max button chart', () => {
		expect(new Set(seatRange(6, BTN).hands)).toEqual(new Set(chartHands('BTN6')));
	});
});

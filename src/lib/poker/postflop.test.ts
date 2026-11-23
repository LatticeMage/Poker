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
			'Kh 9h 6c': 'wet' // two-tone and loosely connected
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

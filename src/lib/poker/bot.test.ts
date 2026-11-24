import { describe, expect, it } from 'vitest';
import type { Action } from '$lib/types/game';
import { chenScore, decide, equity, preflopOrder } from './bot';
import { parseCards } from './cards';
import { advance, applyAction, createGame, startHand } from './engine';
import { pushFoldFreq, share } from './mixed';
import { handOf, seatRange } from './ranges';
import { mulberry32 } from './rng';
import { comboCount, vs3bet, vsOpen } from './spots';
import { BALANCED, BB, BTN, CO, FOLD, HJ, SB, UTG, newGame, raise, totalChips } from './testing';

describe('bot helpers', () => {
	it('rates starting hands with the Chen formula', () => {
		expect(chenScore(parseCards('As Ah'))).toBe(20);
		expect(chenScore(parseCards('As Ks'))).toBe(12);
		expect(chenScore(parseCards('Ts 9s'))).toBe(8);
		expect(chenScore(parseCards('7s 2h'))).toBe(-1);
	});

	it('estimates equity against random hands', () => {
		const rng = mulberry32(7);
		const aces = equity(parseCards('As Ah'), [], 1, rng, 2000);
		expect(aces).toBeGreaterThan(0.8);
		expect(aces).toBeLessThan(0.9);
		expect(equity(parseCards('As Ks'), parseCards('Qs Js Ts 2d 3c'), 3, rng, 200)).toBe(1);
	});
});

describe('full games between robots', () => {
	it('conserve chips, always terminate and rank everyone', () => {
		for (let seed = 1; seed <= 20; seed++) {
			const rng = mulberry32(seed);
			const players = 2 + (seed % 8);
			const g = newGame(players, 1000 + (seed % 3) * 1000, seed);
			// A human bust ends nothing now, but make every seat a robot so games play out unattended.
			g.players[0].is_human = false;
			const bank = totalChips(g);
			let steps = 0;
			while (g.phase !== 'game_over') {
				if (++steps > 200_000) throw new Error(`seed ${seed}: game did not finish`);
				if (g.phase === 'idle' || g.phase === 'hand_over') startHand(g, rng);
				else if (g.phase === 'betting') applyAction(g, decide(g, g.to_act, rng));
				else advance(g);
				expect(totalChips(g)).toBe(bank);
				expect(g.players.every((p) => p.chips >= 0)).toBe(true);
			}
			const left = g.players.filter((p) => p.chips > 0);
			expect(left).toHaveLength(1);
			expect(left[0].chips).toBe(bank);
			expect(g.players.map((p) => p.place).sort((a, b) => a! - b!)).toEqual(
				Array.from({ length: players }, (_, i) => i + 1)
			);
		}
	});
});

describe('robots follow the charts', () => {
	it('balanced robots open first-in at the chart frequency, seat by seat', () => {
		for (const players of [3, 6, 9]) {
			const rng = mulberry32(players);
			const hands = 1500;
			for (let index = 0; index < players - 1; index++) {
				// the big blind never opens
				let raises = 0;
				for (let h = 0; h < hands; h++) {
					const g = createGame({ players, chips: 2000, bots: BALANCED }, rng);
					startHand(g, rng);
					for (let i = 0; i < index; i++) applyAction(g, FOLD);
					raises += decide(g, g.to_act, rng).type === 'raise' ? 1 : 0;
				}
				const chart = seatRange(players, index);
				expect(
					Math.abs((100 * raises) / hands - chart.pct),
					`${players} players, seat ${index}`
				).toBeLessThan(4.5);
			}
		}
	});

	it('heads-up with 5bb the robots shove and call at the solved push/fold frequencies', () => {
		const rng = mulberry32(8);
		const push = pushFoldFreq(5, 'push');
		const call = pushFoldFreq(5, 'call');
		let shoves = 0;
		const n = 3000;
		for (let i = 0; i < n; i++) {
			const g = createGame({ players: 2, chips: 100, bots: BALANCED }, rng); // 5bb at the 10/20 blinds
			startHand(g, rng);
			const sb = g.to_act;
			expect(sb).toBe(g.sb_seat);
			const action = decide(g, sb, rng);
			const name = handOf(g.players[sb].hole);
			if (action.type === 'raise') {
				shoves++;
				expect(push[name] ?? 0).toBeGreaterThan(0);
				expect(action.amount).toBe(g.players[sb].bet + g.players[sb].chips); // all-in
				applyAction(g, action);
				const reply = decide(g, g.to_act, rng);
				if (reply.type === 'call')
					expect(call[name === 'x' ? 'AA' : handOf(g.players[g.to_act].hole)]).toBeGreaterThan(0);
			} else {
				expect(push[name] ?? 0).toBeLessThan(1);
			}
		}
		const want = share(push);
		expect(Math.abs(shoves / n - want)).toBeLessThan(0.03);
	});
});

// -- the preflop strategy tree -----------------------------------------------

const HANDS = 3000;
const pct = (n: number) => (100 * n) / 1326;

/** Share (%) of hands where the hero folds, calls or raises after the script. */
function measure(
	script: [number, Action][],
	hero: number,
	options: { hole?: string; seed?: number; chips?: number } = {}
): Record<'fold' | 'call' | 'raise', number> {
	const rng = mulberry32(options.seed ?? 1);
	const counts = { fold: 0, call: 0, raise: 0 };
	for (let i = 0; i < HANDS; i++) {
		const g = createGame({ players: 6, chips: options.chips ?? 2000, bots: BALANCED }, rng);
		startHand(g, rng);
		const order = preflopOrder(g);
		for (const [position, action] of script) {
			expect(g.to_act).toBe(order[position]);
			applyAction(g, action);
		}
		const seat = order[hero];
		if (options.hole) g.players[seat].hole = parseCards(options.hole);
		expect(g.to_act).toBe(seat);
		const type = decide(g, seat, rng).type;
		counts[type === 'fold' ? 'fold' : type === 'raise' ? 'raise' : 'call']++; // a check counts as a call
	}
	return {
		fold: (100 * counts.fold) / HANDS,
		call: (100 * counts.call) / HANDS,
		raise: (100 * counts.raise) / HANDS
	};
}

describe('the preflop strategy tree', () => {
	it('BTN facing a CO open calls and 3-bets at the chart frequencies', () => {
		const spot = vsOpen('BTN', 'CO')!;
		const got = measure(
			[
				[UTG, FOLD],
				[HJ, FOLD],
				[CO, raise(50)]
			],
			BTN
		);
		expect(Math.abs(got.call - pct(comboCount(spot.call)))).toBeLessThan(2); // 154 combos
		expect(
			Math.abs(got.raise - pct(comboCount(spot.bluff3) + comboCount(spot.value3)))
		).toBeLessThan(2); // 64 + 54
	});

	it('the big blind defends widely against a button open', () => {
		const spot = vsOpen('BB', 'BTN')!;
		const got = measure(
			[
				[UTG, FOLD],
				[HJ, FOLD],
				[CO, FOLD],
				[BTN, raise(40)],
				[SB, FOLD]
			],
			BB
		);
		expect(Math.abs(got.call - pct(comboCount(spot.call)))).toBeLessThan(2.5); // 592 combos: ~45%
		expect(
			Math.abs(got.raise - pct(comboCount(spot.bluff3) + comboCount(spot.value3)))
		).toBeLessThan(2);
	});

	it('a bigger open makes robots defend less', () => {
		const small = measure(
			[
				[UTG, FOLD],
				[HJ, FOLD],
				[CO, raise(50)]
			],
			BTN
		);
		const big = measure(
			[
				[UTG, FOLD],
				[HJ, FOLD],
				[CO, raise(120)]
			],
			BTN
		); // 6bb
		expect(big.call).toBeLessThan(small.call * 0.75);
		expect(Math.abs(big.raise - small.raise)).toBeLessThan(2); // the 3-bet range does not shrink with the size
	});

	const threeBet: [number, Action][] = [
		[UTG, FOLD],
		[HJ, FOLD],
		[CO, raise(50)],
		[BTN, raise(160)],
		[SB, FOLD],
		[BB, FOLD]
	];

	it('the opener facing a 3-bet calls and 4-bets at the chart frequencies', () => {
		const spot = vs3bet('CO', 'BTN')!;
		const got = measure(threeBet, CO);
		expect(Math.abs(got.call - pct(comboCount(spot.call)))).toBeLessThan(1.5); // 50 combos
		expect(
			Math.abs(got.raise - pct(comboCount(spot.bluff4) + comboCount(spot.value4)))
		).toBeLessThan(1.5); // 76
	});

	it('a 3-bettor facing a 4-bet continues with value hands and folds bluffs', () => {
		const script: [number, Action][] = [...threeBet, [CO, raise(400)]];
		expect(measure(script, BTN, { hole: 'As Ks' }).fold).toBe(0); // AKs is a "3-bet/5-bet" hand
		expect(measure(script, BTN, { hole: 'As 5s' }).fold).toBe(100); // A5s is a "3-bet/fold" bluff
	});

	it('a 4-bettor facing a 5-bet continues with 4-bet/call hands only', () => {
		const script: [number, Action][] = [...threeBet, [CO, raise(400)], [BTN, raise(2000)]];
		expect(measure(script, CO, { hole: 'Ad Kc' }).fold).toBe(0); // AKo is "4-bet/call"
		expect(measure(script, CO, { hole: 'As Ts' }).fold).toBe(100); // ATs is "4-bet/fold"
	});

	it('premium hands never fold to an early open', () => {
		const got = measure([[UTG, raise(60)]], HJ, { hole: 'Ks Kd' });
		expect(got.fold).toBe(0);
		expect(got.raise + got.call).toBeCloseTo(100);
	});

	it('short stacks do not bluff 3-bet', () => {
		const got = measure(
			[
				[UTG, FOLD],
				[HJ, FOLD],
				[CO, raise(50)]
			],
			BTN,
			{ hole: 'As 5s', chips: 400 }
		);
		expect(got.raise).toBe(0);
	});
});

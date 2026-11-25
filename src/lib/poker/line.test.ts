// The decision chain: a hand replayed step by step, ranges narrowed by what each player did.
import { describe, expect, it } from 'vitest';
import { parseCompact } from './cards';
import { genRange } from './genRange';
import { genStrategy } from './genStrategy';
import {
	genGame,
	type HandLine,
	LINE_FLOOR,
	type LineAction,
	LineError,
	type LineStep,
	lineSteps,
	potOnFlop,
	replay
} from './line';

const flop: LineStep = { kind: 'deal', cards: parseCompact('As7d2c') };
// BTN opens, BB calls; BB checks, BTN bets 1.5bb into 4.5, BB calls; the turn is the king of clubs.
const steps: LineStep[] = [
	flop,
	{ kind: 'check', by: 'other' },
	{ kind: 'bet', by: 'opener', to: 1.5 },
	{ kind: 'call', by: 'other' },
	{ kind: 'deal', cards: parseCompact('Kc') },
	{ kind: 'check', by: 'other' },
	{ kind: 'check', by: 'opener' },
	{ kind: 'deal', cards: parseCompact('3h') },
	{ kind: 'bet', by: 'other', to: 5 },
	{ kind: 'fold', by: 'opener' }
];
const hand = (n = steps.length, extra: Partial<HandLine> = {}): HandLine => ({
	pot: 'srp',
	opener: 'BTN',
	other: 'BB',
	steps: steps.slice(0, n),
	...extra
});

describe('the steps of a hand from its board and its actions', () => {
	const acts = steps.filter((s): s is LineAction => s.kind !== 'deal');
	const cards = parseCompact('As7d2cKc3h');

	it('deals the next card as soon as the betting before it is over', () => {
		expect(lineSteps(cards, acts)).toEqual({ steps, used: acts.length });
		expect(lineSteps(cards.slice(0, 2), acts)).toEqual({ steps: [], used: 0 }); // no flop yet
		expect(lineSteps(cards.slice(0, 3), [])).toEqual({ steps: [steps[0]], used: 0 });
	});

	it('leaves out the actions that wait for a card, or come after a fold', () => {
		// only the flop is on the board: the three flop actions go in, the turn actions wait
		const flopOnly = lineSteps(cards.slice(0, 3), acts);
		expect(flopOnly.used).toBe(3);
		expect(flopOnly.steps).toEqual(steps.slice(0, 4));
		const folded = lineSteps(cards, [
			{ kind: 'bet', by: 'other', to: 3 },
			{ kind: 'fold', by: 'opener' },
			{ kind: 'check', by: 'other' }
		]);
		expect(folded.used).toBe(2);
		expect(folded.steps).toHaveLength(3);
	});
});

describe('the pot on the flop', () => {
	it('uses the sizes of the offline solver', () => {
		expect(potOnFlop('srp', 'BTN', 'BB')).toEqual({ pot: 4.5, stack: 98 });
		expect(potOnFlop('3bet', 'BTN', 'BB')).toEqual({ pot: 16.5, stack: 92 });
		expect(potOnFlop('4bet', 'BTN', 'BB')).toEqual({ pot: 36.5, stack: 82 });
		expect(potOnFlop('3bet', 'CO', 'BTN').pot).toBe(16.5); // 3-bet to 7.5 in position, both blinds dead
	});
});

describe('replaying a hand', () => {
	const states = replay(hand());

	it('gives one state before any step and one after each', () => {
		expect(states).toHaveLength(steps.length + 1);
		expect(states[0]).toMatchObject({ index: 0, street: 'preflop', pot: 4.5, to_act: null });
		expect(states[0].equity).toBeNull();
		expect(states.map((s) => s.street)).toEqual([
			'preflop',
			'flop',
			'flop',
			'flop',
			'flop',
			'turn',
			'turn',
			'turn',
			'river',
			'river',
			'river'
		]);
	});

	it('keeps the pot, the bets and whose turn it is', () => {
		expect(states[1]).toMatchObject({ to_act: 'other', pot: 4.5, in: [0, 0] }); // the big blind acts first
		expect(states[2]).toMatchObject({ to_act: 'opener', pot: 4.5 });
		expect(states[3]).toMatchObject({ to_act: 'other', pot: 6, in: [1.5, 0] });
		expect(states[4]).toMatchObject({ to_act: null, pot: 7.5, in: [1.5, 1.5], over: null });
		expect(states[5]).toMatchObject({ to_act: 'other', pot: 7.5, in: [0, 0] });
		expect(states[7]).toMatchObject({ to_act: null, pot: 7.5 }); // checked through
		expect(states[9]).toMatchObject({ to_act: 'opener', pot: 12.5, in: [0, 5] });
		expect(states[10]).toMatchObject({ to_act: null, over: 'fold' });
	});

	it('starts on the flop with the preflop ranges and the group table', () => {
		const s = states[1];
		expect(s.reach).toEqual([1, 1]);
		expect(s.equity![0]).toBeGreaterThan(0.5); // the button's range is ahead on A-7-2
		expect(s.equity![0] + s.equity![1]).toBeCloseTo(1);
		expect(s.strategy).toMatchObject({ by: 'other', options: ['check', 'bet'] });
		expect(s.strategy!.range[0] + s.strategy!.range[1]).toBeCloseTo(1);
		expect(s.strategy!.range[0]).toBeGreaterThan(0.6); // the caller mostly checks to the raiser
	});

	it('narrows the range of the player who acted, by how often each hand takes the action', () => {
		// the check: the big blind keeps the share of its range that checks; the button's range is untouched
		expect(states[2].taken).toBeCloseTo(states[1].strategy!.range[0]);
		expect(states[2].reach[0]).toBe(1);
		expect(states[2].reach[1]).toBeLessThan(1);
		expect(Math.abs(states[2].reach[1] - states[2].taken!)).toBeLessThan(0.03);
		// the bet: now the button's range is the part that bets
		expect(states[3].taken).toBeCloseTo(states[2].strategy!.range[1]);
		expect(Math.abs(states[3].reach[0] - states[3].taken!)).toBeLessThan(0.03);
		expect(states[3].strategy).toMatchObject({ by: 'other', options: ['fold', 'call', 'raise'] });
		// the call: the product of the two steps
		const called = states[1].strategy!.range[0] * states[3].strategy!.range[1];
		expect(Math.abs(states[4].reach[1] - called)).toBeLessThan(0.05);
		// reach never grows along the line
		for (let k = 1; k < states.length; k++) {
			expect(states[k].reach[0]).toBeLessThanOrEqual(states[k - 1].reach[0] + 1e-9);
			expect(states[k].reach[1]).toBeLessThanOrEqual(states[k - 1].reach[1] + 1e-9);
		}
	});

	it('the hands that bet a dry ace-high flop are stronger than the range was', () => {
		// after betting, the button's range holds more aces: its equity against the same checking range goes up
		const before = states[2].equity![0];
		const after = states[3].equity![0];
		expect(after).toBeGreaterThan(before - 0.02);
		// and the big blind's calling range is stronger than its checking range
		expect(states[4].equity![1]).toBeGreaterThan(states[3].equity![1]);
	});

	it('a state depends only on the steps before it: every prefix gives the same states', () => {
		for (const n of [0, 1, 3, 5, 8]) {
			const part = replay(hand(n));
			expect(part).toHaveLength(n + 1);
			expect(JSON.stringify(part[n])).toBe(JSON.stringify(states[n]));
		}
	});

	it('follows known hole cards', () => {
		const known = replay(hand(steps.length, { hole: { opener: parseCompact('AhKd') } }));
		expect(known[0].hand_equity).toEqual([null, null]);
		expect(known[1].hand_equity[0]!).toBeGreaterThan(0.8); // top pair, top kicker
		expect(known[1].hand_equity[1]).toBeNull();
		expect(known[2].strategy!.hand).not.toBeNull();
		expect(known[3].hand_taken!).toBeGreaterThan(0.5); // it bets
		expect(known[5].hand_equity[0]!).toBeGreaterThan(0.8); // two pair on the king
		// a card in a known hand cannot come on the board
		expect(() => replay(hand(1, { hole: { opener: parseCompact('AsKd') } }))).toThrow(LineError);
	});

	it('an action the model never takes still leaves a thin range', () => {
		// the big blind leads for the whole pot twice: unusual, but the hand goes on
		const odd = replay({
			pot: 'srp',
			opener: 'BTN',
			other: 'BB',
			steps: [flop, { kind: 'bet', by: 'other', to: 4.5 }, { kind: 'raise', by: 'opener', to: 15 }]
		});
		expect(odd[2].reach[1]).toBeGreaterThanOrEqual(LINE_FLOOR - 1e-9);
		expect(odd[3].combos[0]).toBeGreaterThan(0);
		expect(odd[3].equity).not.toBeNull();
	});

	it('ends at the showdown when the river is checked through or called', () => {
		const through = replay(hand(8, {}));
		const end = replay({
			...hand(8),
			steps: [...steps.slice(0, 8), { kind: 'check', by: 'other' }, { kind: 'check', by: 'opener' }]
		});
		expect(through[8].over).toBeNull();
		expect(end[10]).toMatchObject({ over: 'showdown', to_act: null });
	});

	it('with more players in the pot the ranges start tighter, and the flop tables are not used', () => {
		const two = replay(hand(2));
		const three = replay(hand(2, { players: 3 }));
		expect(three[0].combos[0]).toBeLessThan(two[0].combos[0] * 0.85);
		expect(three[0].combos[1]).toBeLessThan(two[0].combos[1] * 0.85);
		expect(three[1].reach).toEqual([1, 1]); // reach is counted from the tightened start
		expect(two[1].strategy!.source).toBe('solved'); // BTN against BB is solved
		expect(three[1].strategy!.source).toBe('rules');
		expect(three[1].equity).not.toBeNull();
	});

	it('refuses steps that cannot be played', () => {
		const bad =
			(extra: LineStep[], n = 1) =>
			() =>
				replay({ ...hand(n), steps: [...steps.slice(0, n), ...extra] });
		expect(bad([{ kind: 'check', by: 'opener' }])).toThrow(LineError); // out of turn
		expect(bad([{ kind: 'call', by: 'other' }])).toThrow(LineError); // nothing to call
		expect(bad([{ kind: 'deal', cards: parseCompact('Kc') }])).toThrow(LineError); // a player has to act
		expect(bad([{ kind: 'check', by: 'other' }], 3)).toThrow(LineError); // a bet to answer
		expect(bad([{ kind: 'raise', by: 'other', to: 1.5 }], 3)).toThrow(LineError); // not more than the bet
		expect(bad([{ kind: 'deal', cards: parseCompact('As') }], 4)).toThrow(LineError); // card already out
		expect(bad([{ kind: 'check', by: 'other' }], 10)).toThrow(LineError); // the hand is over
		expect(() => replay({ pot: 'srp', opener: 'BB', other: 'BTN', steps: [] })).toThrow(LineError);
	});
});

describe('a game as table statuses, and the two pure functions', () => {
	const line = hand(6, { hole: { opener: parseCompact('AhKd') } });
	const game = genGame(line);
	const states = replay(line);
	const copy = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

	it('is one status before any step and one after each, and survives JSON', () => {
		expect(game).toHaveLength(7);
		expect(copy(game)).toStrictEqual(game);
		expect(game[3]).toMatchObject({ pot: 6, to_act: 'other', history: line.steps.slice(0, 3) });
		expect(game[3]).not.toHaveProperty('ranges');
	});

	it('any status on its own gives its ranges, and with them its strategy', () => {
		const saved = copy(game);
		for (const k of [5, 0, 3, 6, 1, 4, 2]) {
			const ranges = genRange(saved[k]);
			expect(ranges).toStrictEqual(states[k].ranges);
			const hole = saved[k].to_act === 'opener' ? parseCompact('AhKd') : null;
			expect(genStrategy(saved[k], ranges, hole)).toStrictEqual(states[k].strategy);
		}
	});

	it("does not change what it is given, and what it returns is the caller's own", () => {
		const status = copy(game[3]);
		const ranges = genRange(status);
		const kept = copy(ranges);
		genStrategy(status, ranges);
		expect(status).toStrictEqual(game[3]);
		expect(ranges).toStrictEqual(kept);
		ranges[0].fill(0);
		expect(genRange(status)).toStrictEqual(kept);
	});
});

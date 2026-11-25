// Robot opponents.
//
// Before the flop they play from the charts in ranges.ts / spots.ts (opening range by seat and table
// size, then the facing-a-raise strategy tree, widened or narrowed by persona). After the flop they
// use the range-equity model in postflop.ts in heads-up pots, and an equity-against-random-hands
// heuristic everywhere else. A persona skews every threshold so the bots do not all play alike.

import type { Action, Legal, Persona } from '$lib/types/game';
import type { Card } from './cards';
import { type Game, legalActions } from './engine';
import { handScore } from './evaluate';
import * as postflop from './postflop';
import * as mixed from './mixed';
import { chenValue, handOf, percentile, roundHalfEven, seatRange } from './ranges';
import { type Rng, sample } from './rng';
import * as spots from './spots';

export const PERSONAS: readonly Persona[] = [
	{ name: 'Ada', style: 'tight-aggressive', tight: 0.8, aggro: 0.75, bluff: 0.15 },
	{ name: 'Bolt', style: 'loose-aggressive', tight: 0.2, aggro: 0.9, bluff: 0.5 },
	{ name: 'Cog', style: 'rock', tight: 0.75, aggro: 0.2, bluff: 0.05 },
	{ name: 'Dot', style: 'calling station', tight: 0.25, aggro: 0.25, bluff: 0.1 },
	{ name: 'Echo', style: 'balanced', tight: 0.5, aggro: 0.5, bluff: 0.25 },
	{ name: 'Fuse', style: 'maniac', tight: 0.3, aggro: 0.85, bluff: 0.45 },
	{ name: 'Gizmo', style: 'solid', tight: 0.6, aggro: 0.6, bluff: 0.2 },
	{ name: 'Hex', style: 'nit', tight: 0.95, aggro: 0.5, bluff: 0.08 }
];

const DEFAULT_PERSONA: Persona = {
	name: 'Bot',
	style: 'balanced',
	tight: 0.5,
	aggro: 0.5,
	bluff: 0.2
};

/** Suited wheel aces: good 3-bet bluffs because they block AA and AK. */
const BLUFF_3BET = new Set(['A5s', 'A4s', 'A3s', 'A2s']);

/** Chen formula for two cards: a quick rating of a starting hand (AA = 20, 72o = -1). */
export function chenScore(hole: readonly Card[]): number {
	const a = (hole[0] >> 2) + 2;
	const b = (hole[1] >> 2) + 2;
	return chenValue(Math.max(a, b), Math.min(a, b), (hole[0] & 3) === (hole[1] & 3));
}

/**
 * Monte Carlo equity: the share of the pot this hand wins on average against `opponents` random
 * hands, given the community cards so far.
 */
export function equity(
	hole: readonly Card[],
	board: readonly Card[],
	opponents: number,
	rng: Rng,
	iterations = 200
): number {
	const known = new Set([...hole, ...board]);
	const deck = Array.from({ length: 52 }, (_, i) => i).filter((c) => !known.has(c));
	const missing = 5 - board.length;
	const need = missing + 2 * opponents;

	let total = 0;
	for (let it = 0; it < iterations; it++) {
		const draws = sample(deck, need, rng);
		const full = [...board, ...draws.slice(0, missing)];
		const mine = handScore([hole[0], hole[1], ...full]);
		let ties = 0;
		let beaten = false;
		for (let k = missing; k < need; k += 2) {
			const theirs = handScore([draws[k], draws[k + 1], ...full]);
			if (theirs > mine) {
				beaten = true;
				break;
			}
			if (theirs === mine) ties++;
		}
		if (!beaten) total += 1 / (ties + 1);
	}
	return total / iterations;
}

const check = (): Action => ({ type: 'check' });
const call = (): Action => ({ type: 'call' });
const fold = (): Action => ({ type: 'fold' });
const passive = (legal: Legal): Action => (legal.can_check ? check() : fold());
const flat = (legal: Legal): Action => (legal.can_call ? call() : check());
const jam = (legal: Legal): Action => ({ type: 'raise', amount: legal.max_to });

/** Raise to roughly `target`, snapped to a legal, tidy amount. */
function raiseTo(g: Game, legal: Legal, target: number): Action {
	if (!legal.can_raise) return flat(legal);
	const unit = Math.max(1, g.small_blind);
	let to = roundHalfEven(target / unit) * unit;
	to = Math.min(legal.max_to, Math.max(legal.min_to, to));
	// Do not leave a token stack behind: commit instead.
	if (legal.max_to - to < to * 0.3) to = legal.max_to;
	return { type: 'raise', amount: to };
}

/** Decide the action for the player whose turn it is. Does not modify the game. */
export function decide(g: Game, seat: number, rng: Rng): Action {
	const legal = legalActions(g);
	if (legal.seat !== seat) throw new Error(`It is not seat ${seat}'s turn`);
	const p = g.players[seat];
	const persona = p.persona ?? DEFAULT_PERSONA;
	if (g.street === 'preflop') return preflop(g, legal, persona, rng, p.hole);
	const planned = streetPlan(g, legal, persona, rng, p.hole);
	if (planned) return planned;
	const opponents = g.players.filter((o) => o !== p && o.in_hand && !o.folded).length;
	return postflopHeuristic(g, legal, persona, rng, p.hole, opponents);
}

/** Live seats in preflop acting order: first to act … small blind, big blind. */
export function preflopOrder(g: Game): number[] {
	const n = g.players.length;
	const order: number[] = [];
	let seat = g.bb_seat;
	for (let i = 0; i < n; i++) {
		seat = (seat + 1) % n;
		if (g.players[seat].in_hand) order.push(seat);
	}
	return order;
}

function preflop(g: Game, legal: Legal, persona: Persona, rng: Rng, hole: readonly Card[]): Action {
	const bb = g.big_blind;
	const me = g.players[legal.seat];
	const order = preflopOrder(g);
	const players = order.length;
	const index = order.indexOf(legal.seat);
	const behind = players - 1 - index;
	const name = handOf(hole);

	// How wide this persona plays (tight < 1 < loose), and where the hand ranks (top X% of 169
	// hands). A little noise keeps hands at the edge of a range mixed.
	const width = mixed.widthOf(persona.tight);
	const rank = percentile(name) + (rng() - 0.5) * 2;

	const stackBb = legal.stack_to / bb;
	const raised = g.current_bet > bb;

	// Heads-up: play the solved game, sampling each hand's mixed frequencies. Deep enough, that is the
	// solved preflop tree at the nearest depth; shallower, the push/fold equilibrium (see mixed.ts).
	if (players === 2) {
		const opponent = g.players.find((o) => o !== me && o.in_hand && !o.folded)!;
		const effective = Math.min(legal.stack_to, opponent.bet + opponent.chips) / bb;
		if (effective >= mixed.TREE_MIN_BB) {
			const choice = mixed.headsUpChoice(g, legal.seat, name, effective, width, rng);
			if (choice) return fromChoice(g, legal, choice);
		} else if (effective >= 2) {
			if (raised)
				return rng() < (mixed.pushFoldFreq(effective, 'call')[name] ?? 0) ? call() : passive(legal); // the big blind facing a shove
			if (behind === 1) {
				const shove = rng() < (mixed.pushFoldFreq(effective, 'push')[name] ?? 0);
				return shove && legal.can_raise ? jam(legal) : passive(legal); // the small blind acts first
			}
			return check(); // the big blind, nobody raised
		}
	}

	// Other short stacks: push or fold. Heuristic widths - the sources only have charts as images.
	if (stackBb <= 10) {
		const base = stackBb <= 6 ? 35 : 22;
		const position = behind <= 2 ? 1.4 : behind <= 4 ? 1 : 0.75;
		if (rank <= base * position * width * (raised ? 0.5 : 1)) {
			return legal.can_raise ? jam(legal) : flat(legal);
		}
		return passive(legal);
	}

	if (!raised) {
		const limpers = g.players.filter(
			(o) => o !== me && o.in_hand && !o.folded && o.seat !== g.bb_seat && o.bet >= bb
		).length;
		if (behind === 0) {
			// big blind: nobody raised
			if (limpers && rank <= 14 * width) return raiseTo(g, legal, bb * (3 + limpers));
			return check();
		}
		// The chart's frequency for this hand, stretched by the persona (see mixed.ts). Raising to
		// isolate limpers uses a narrower slice of the same range.
		const play = mixed.scaledOpenFreq(players, index, width)[name] ?? 0;
		const u = rng();
		if (limpers) {
			const isolate = mixed.scaledOpenFreq(players, index, width * 0.7)[name] ?? 0;
			if (u < isolate) return raiseTo(g, legal, bb * (3 + limpers));
			if (u < play || (behind === 1 && legal.to_call <= bb / 2 && rank <= 45)) return call();
			return passive(legal);
		}
		if (u < play) {
			// The small blind plays out of position, so it opens bigger.
			return raiseTo(g, legal, bb * ((behind === 1 ? 3.0 : 2.5) + rng() * 0.5));
		}
		return passive(legal);
	}

	// Facing a raise: walk the strategy tree from the charts; heuristics only if a branch has none.
	const plan = facingRaise(g, legal, persona, rng, order, name, rank, width);
	if (plan) return plan;
	return facingHeuristic(g, legal, persona, rng, order, index, behind, name, rank, width);
}

/** Turn a solved heads-up choice into a legal action. */
function fromChoice(g: Game, legal: Legal, choice: mixed.Choice): Action {
	switch (choice.kind) {
		case 'raise':
			return raiseTo(g, legal, choice.to_bb * g.big_blind);
		case 'allin':
			return legal.can_raise ? jam(legal) : flat(legal);
		case 'call':
		case 'limp':
			return flat(legal);
		case 'check':
			return check();
		default:
			return passive(legal);
	}
}

/** Sample an action from [action, probability] pairs; the rest of the probability is 'fold'. */
function pick<T extends string>(rng: Rng, weights: [T, number][]): T | 'fold' {
	let u = rng();
	for (const [action, p] of weights) {
		if (u < p) return action;
		u -= p;
	}
	return 'fold';
}

/**
 * Strategy tree for a raised pot: facing an open, facing a 3-bet, facing a 4-bet or 5-bet. The charts
 * give each hand a *mix* (e.g. AA: one combo calls, five 3-bet), sampled here. Returns null when no
 * chart covers the spot.
 */
function facingRaise(
	g: Game,
	legal: Legal,
	persona: Persona,
	rng: Rng,
	order: number[],
	name: string,
	rank: number,
	width: number
): Action | null {
	const raisers = g.raisers;
	const me = legal.seat;
	const players = order.length;
	const stackBb = legal.stack_to / g.big_blind;
	const sizeBb = g.current_bet / g.big_blind;

	const behindOf = (seat: number) => players - 1 - order.indexOf(seat);
	const asOpener = (seat: number) => spots.positionClass(behindOf(seat), true);
	const asFacing = (seat: number) => spots.positionClass(behindOf(seat));
	const raiseOver = (multiple: number, allIn: boolean): Action => {
		if (allIn || !legal.can_raise) return legal.can_raise ? jam(legal) : flat(legal);
		return raiseTo(g, legal, g.current_bet * multiple);
	};

	// 1. Facing an open raise.
	if (raisers.length === 1 && !raisers.includes(me)) {
		const opener = raisers[0];
		const openerClass = asOpener(opener);
		const spot = spots.vsOpen(asFacing(me), openerClass);
		if (!spot) return null;
		let bluff = spot.bluff3[name] ?? 0;
		const value = spot.value3[name] ?? 0;
		// Approximations on top of the charts: style, open size (minimum defence frequency), callers
		// already in, and a shallower stack.
		const raises = mixed.sum(spot.value3, spot.bluff3);
		const callers = g.players.filter(
			(o) => o.seat !== me && o.seat !== opener && o.in_hand && !o.folded && o.bet >= g.current_bet
		).length;
		// With callers already in, the calling range is the chart's tightened for that many players
		// (postflop.potTightness: the opening charts narrow the same way).
		let callShare =
			mixed.scaledCalls(
				spot,
				spot.call,
				raises,
				width *
					spots.sizeFactor(openerClass, sizeBb) *
					(stackBb < 40 ? 0.75 : 1) *
					postflop.potTightness(2 + callers)
			)[name] ?? 0;
		bluff *= 0.5 + persona.aggro;
		if (callers) bluff *= 0.25;
		if (stackBb < 25) bluff = 0;
		bluff = Math.min(bluff, 1 - value);
		callShare = Math.max(0, Math.min(callShare, 1 - value - bluff));
		const inPosition = behindOf(opener) > behindOf(me) && behindOf(me) >= 2;
		const allIn = stackBb < 25;
		switch (
			pick(rng, [
				['3bet', value + bluff],
				['call', callShare]
			])
		) {
			case '3bet':
				return raiseOver(inPosition ? 3.0 : 3.7, allIn);
			case 'call':
				return flat(legal);
			default:
				return passive(legal);
		}
	}

	// 2. Opened, now facing a 3-bet.
	if (raisers.length === 2 && raisers[0] === me) {
		const spot = spots.vs3bet(asOpener(me), asFacing(raisers[1]));
		if (!spot) return null;
		let callShare =
			mixed.scaledCalls(spot, spot.call, mixed.sum(spot.value4, spot.bluff4), width)[name] ?? 0;
		const four = (spot.bluff4[name] ?? 0) + (spot.value4[name] ?? 0);
		callShare = Math.max(0, Math.min(callShare, 1 - four));
		switch (
			pick(rng, [
				['4bet', four],
				['call', callShare]
			])
		) {
			case '4bet':
				return raiseOver(2.3, stackBb <= 35);
			case 'call':
				return flat(legal);
			default:
				return passive(legal);
		}
	}

	// 3. 3-bet, then faced a 4-bet: the hands the chart 3-bets for value continue, bluffs fold.
	if (raisers.length === 3 && raisers[1] === me && raisers[0] !== me) {
		const spot = spots.vsOpen(asFacing(me), asOpener(raisers[0]));
		const value = spot?.value3[name] ?? 0;
		const bluff = spot?.bluff3[name] ?? 0;
		if (!spot || value + bluff === 0) return premiumOnly(legal, rank, width);
		if (rng() < value / (value + bluff)) {
			return rank <= 1.5 || stackBb <= 30 ? raiseOver(2.2, true) : flat(legal);
		}
		return passive(legal);
	}

	// 4. 4-bet, then faced a 5-bet: the 4-bet/call hands continue, the 4-bet bluffs fold.
	if (raisers.length === 4 && raisers[2] === me && raisers[0] === me) {
		const spot = spots.vs3bet(asOpener(me), asFacing(raisers[1]));
		const value = spot?.value4[name] ?? 0;
		const bluff = spot?.bluff4[name] ?? 0;
		if (!spot || value + bluff === 0) return premiumOnly(legal, rank, width);
		if (rng() < value / (value + bluff)) {
			return legal.can_raise && rank <= 1.5 ? jam(legal) : flat(legal);
		}
		return passive(legal);
	}

	// Any other raised pot (cold 4-bets, squeezes, five-bets): premium hands only.
	return raisers.length ? premiumOnly(legal, rank, width) : null;
}

/** Continue with the very top of the range; used where no chart exists. */
function premiumOnly(legal: Legal, rank: number, width: number): Action {
	if (rank <= 1.5 * width) return legal.can_raise ? jam(legal) : flat(legal);
	if (rank <= 3.5 * width && legal.to_call <= 0.5 * legal.chips) return flat(legal);
	return passive(legal);
}

/** The pre-chart fallback: the width of the continue range scales with how wide the raiser opens. */
function facingHeuristic(
	g: Game,
	legal: Legal,
	persona: Persona,
	rng: Rng,
	order: number[],
	index: number,
	behind: number,
	name: string,
	rank: number,
	width: number
): Action {
	const bb = g.big_blind;
	const players = order.length;
	const others = g.players.filter((o) => o !== g.players[legal.seat] && o.in_hand && !o.folded);
	const raiser = others.reduce((best, o) => (o.bet > best.bet ? o : best), others[0]);
	const raiserIndex = order.indexOf(raiser.seat);
	const openerPct = seatRange(players, raiserIndex).pct || 12; // a big blind re-raising
	const costShare = legal.to_call / Math.max(1, legal.chips);

	if (g.current_bet > 5 * bb) {
		// a 3-bet or more: premiums only
		if (rank <= 2.5 * width) return legal.can_raise && rng() < 0.6 ? jam(legal) : flat(legal);
		if (rank <= 5 * width && costShare < 0.5) return flat(legal);
		if (rank <= 8 && costShare >= 0.6) return flat(legal); // already committed
		return passive(legal);
	}

	if (costShare >= 0.45) {
		// a big part of the stack
		if (rank <= 3 * width && legal.can_raise) return jam(legal);
		return rank <= 6 * width ? flat(legal) : passive(legal);
	}

	const threeBet = Math.min(6.5, Math.max(2.5, 0.22 * openerPct)) * (0.6 + 0.8 * persona.aggro);
	const inPosition = raiserIndex < index && index < players - 2;
	if (rank <= threeBet && legal.can_raise) {
		if (rank <= 1.2 && rng() < 0.15) return flat(legal); // slow-play the very best hands now and then
		return raiseTo(g, legal, g.current_bet * (inPosition ? 3.0 : 3.7));
	}
	if (BLUFF_3BET.has(name) && openerPct >= 20 && legal.can_raise && rng() < persona.bluff * 0.5) {
		return raiseTo(g, legal, g.current_bet * 3.7);
	}
	let callRange = 0.5 * openerPct * width;
	if (behind === 0)
		callRange *= 1.3; // the big blind is already in and gets a price
	else if (behind === 1) callRange *= 0.8; // the small blind plays out of position
	return rank <= callRange ? call() : passive(legal);
}

/** A flop, turn or river from the range model (postflop.ts); null where it has none. */
function streetPlan(
	g: Game,
	legal: Legal,
	persona: Persona,
	rng: Rng,
	hole: readonly Card[]
): Action | null {
	if (g.street === 'preflop') return null;
	// Two players: the range model with its flop tables. More: the same model on tightened ranges.
	const order = preflopOrder(g);
	const two = postflop.headsUpRanges(g, legal.seat, order);
	const many = two ? null : postflop.multiwayRanges(g, legal.seat, order);
	if (!two && !many) return null;
	const [mine, theirs, aggressor] = two ?? many!;
	const key = two ? two[3] : null;
	const prof = postflop.profile(g.board, hole, mine, theirs);
	if (!prof) return null;
	const me = g.players[legal.seat];
	const decision = postflop.decide(
		prof,
		{
			texture: postflop.boardTexture(g.board),
			flop: postflop.flopRead(key, g.board, aggressor),
			pot: legal.pot,
			to_call: legal.to_call,
			can_raise: legal.can_raise,
			aggressor,
			facing_raise: me.bet > 0 && legal.to_call > 0,
			persona_aggro: persona.aggro,
			persona_bluff: persona.bluff,
			persona_tight: persona.tight
		},
		rng
	);
	switch (decision.kind) {
		case 'bet':
			return raiseTo(g, legal, decision.size * legal.pot);
		case 'raise':
			return raiseTo(g, legal, g.current_bet * decision.size);
		case 'call':
			return flat(legal);
		case 'fold':
			return passive(legal);
		default:
			return legal.can_check ? check() : passive(legal);
	}
}

/** Pots with three or more players, limped pots and the like: equity against random hands. */
function postflopHeuristic(
	g: Game,
	legal: Legal,
	persona: Persona,
	rng: Rng,
	hole: readonly Card[],
	opponents: number
): Action {
	const pot = legal.pot;
	const eq = equity(hole, g.board, opponents, rng);
	// An average hand wins its fair share; `edge` rescales equity so 0 is average and 1 is a lock.
	const fair = 1 / (opponents + 1);
	const river = g.street === 'river';

	if (legal.can_check) {
		const edge = (eq - fair) / (1 - fair);
		if (legal.can_raise && edge > 0.34 - 0.1 * persona.aggro) {
			if (edge > 0.7 && rng() < 0.2) return check(); // slow-play a monster
			const size = edge > 0.6 ? 0.66 + rng() * 0.34 : 0.4 + rng() * 0.3;
			return raiseTo(g, legal, size * pot);
		}
		const bluffChance = (persona.bluff * (river ? 0.2 : 0.35)) / opponents;
		if (legal.can_raise && rng() < bluffChance) return raiseTo(g, legal, 0.5 * pot);
		return check();
	}

	// Facing a bet. Someone betting big rarely has a random hand, so discount.
	const potOdds = legal.to_call / (pot + legal.to_call);
	const pressure = Math.min(1.5, legal.to_call / Math.max(1, pot - legal.to_call));
	const adjusted = eq - 0.09 * pressure - 0.03 * persona.tight;
	const edge = (adjusted - fair) / (1 - fair);

	if (legal.can_raise && edge > 0.55 && rng() < 0.35 + 0.5 * persona.aggro) {
		return raiseTo(g, legal, g.current_bet + (0.6 + rng() * 0.4) * (pot + legal.to_call));
	}
	const bigDecision = legal.to_call >= legal.chips * 0.5;
	const margin = 0.02 + 0.06 * persona.tight + (bigDecision ? 0.05 : 0);
	if (adjusted >= potOdds + margin) return call();
	if (!river && legal.can_raise && rng() < persona.bluff * 0.06) {
		return raiseTo(g, legal, g.current_bet + 0.7 * (pot + legal.to_call));
	}
	return fold();
}

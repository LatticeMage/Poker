// No-limit Texas Hold'em rules engine.
//
// Pure game logic with no UI or timing: every function mutates a `Game` and returns the log events
// it produced. The caller paces the steps:
//
//   startHand()                       deal + blinds
//   while phase is betting | dealing:
//     betting → applyAction()         one player decision
//     dealing → advance()             next street, or showdown after the river
//   phase is hand_over | game_over
//
// `bet` is a player's stake on the current street, `committed` their stake in the whole hand.
// Raise amounts are always "raise to" totals for the street. The game is the plain JSON object
// `GameState` (see $lib/types/game), so it can be stored and cloned as is.

import type {
	Action,
	GameEvent,
	GameState,
	Legal,
	Persona,
	PlayerState,
	PotState
} from '$lib/types/game';
import { isCard } from '$lib/utils/cards';
import { type Card, cardText, freshDeck } from './cards';
import { evaluate } from './evaluate';
import { type Rng, randrange, shuffle } from './rng';

export type Game = GameState;
export type Player = PlayerState;

/** Big blind per level. The small blind is half. */
export const BLIND_LEVELS = [
	20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 640, 960, 1280, 1920, 2560, 3840, 5120, 7680, 10240,
	15360
];
export const HANDS_PER_LEVEL = 6;

export function blindsForHand(handNumber: number): { small: number; big: number } {
	const level = Math.min(
		BLIND_LEVELS.length - 1,
		Math.floor(Math.max(0, handNumber - 1) / HANDS_PER_LEVEL)
	);
	const big = BLIND_LEVELS[level];
	return { small: Math.floor(big / 2), big };
}

export function ordinal(n: number): string {
	const rem100 = n % 100;
	if (rem100 >= 11 && rem100 <= 13) return `${n}th`;
	return `${n}${['th', 'st', 'nd', 'rd'][n % 10] ?? 'th'}`;
}

const ES_VERB = /(s|sh|ch|x|z|o)$/;

/** "You raise" / "Ada raises". */
function says(p: Player, verb: string): string {
	if (p.is_human) return `${p.name} ${verb}`;
	return `${p.name} ${verb}${ES_VERB.test(verb) ? 'es' : 's'}`;
}

function newPlayer(seat: number, name: string, persona: Persona | null, chips: number): Player {
	return {
		seat,
		name,
		is_human: persona === null,
		persona,
		chips,
		start_chips: chips,
		hole: [],
		in_hand: false,
		folded: false,
		all_in: false,
		bet: 0,
		committed: 0,
		acted: false,
		badge: '',
		revealed: false,
		won: 0,
		hand_name: '',
		best_cards: [],
		place: null
	};
}

export function createGame(
	config: { players: number; chips: number; bots: readonly Persona[] },
	rng: Rng
): Game {
	const count = Math.max(2, Math.min(9, Math.floor(config.players)));
	if (config.bots.length < count - 1) throw new Error('Not enough bot personas');
	const players = [newPlayer(0, 'You', null, config.chips)];
	for (let seat = 1; seat < count; seat++) {
		const persona = config.bots[seat - 1];
		players.push(newPlayer(seat, persona.name, persona, config.chips));
	}
	const { small, big } = blindsForHand(1);
	return {
		players,
		dealer: randrange(count, rng),
		hand_number: 0,
		sb_seat: -1,
		bb_seat: -1,
		small_blind: small,
		big_blind: big,
		phase: 'idle',
		street: 'preflop',
		board: [],
		runout: [],
		current_bet: 0,
		min_raise: big,
		to_act: -1,
		pot: 0,
		pots: [],
		headline: '',
		raisers: [],
		hands_per_level: HANDS_PER_LEVEL,
		next_big_blind: big
	};
}

/** First seat clockwise after `start` matching the predicate, or -1. */
function nextSeat(g: Game, start: number, pred: (p: Player) => boolean): number {
	const n = g.players.length;
	for (let i = 1; i <= n; i++) {
		const seat = (start + i) % n;
		if (pred(g.players[seat])) return seat;
	}
	return -1;
}

export const contending = (p: Player) => p.in_hand && !p.folded;
export const canBet = (p: Player) => p.in_hand && !p.folded && !p.all_in;

function put(p: Player, amount: number): number {
	const paid = Math.min(amount, p.chips);
	p.chips -= paid;
	p.bet += paid;
	p.committed += paid;
	if (p.chips === 0) p.all_in = true;
	return paid;
}

/**
 * Chips the player must add to stay in. Normally the gap to the current bet; when every opponent is
 * all-in it is only the gap to the largest of their stakes (a short blind cannot be "called" for
 * more than it is).
 */
function owedBy(g: Game, p: Player): number {
	const others = g.players.filter((o) => o !== p && contending(o));
	const target = others.some(canBet)
		? g.current_bet
		: Math.min(g.current_bet, Math.max(0, ...others.map((o) => o.bet)));
	return Math.max(0, target - p.bet);
}

function needsToAct(g: Game, p: Player): boolean {
	if (!canBet(p)) return false;
	if (owedBy(g, p) > 0) return true;
	if (p.acted) return false;
	// Nothing to call: only worth asking if someone else could respond to a bet.
	return g.players.some((o) => o !== p && canBet(o));
}

/**
 * Split the committed chips into a main pot and side pots. Each all-in level caps what that player
 * can win from every opponent.
 */
export function computePots(players: readonly Player[]): PotState[] {
	const levels = [...new Set(players.filter(contending).map((p) => p.committed))].sort(
		(a, b) => a - b
	);
	const pots: PotState[] = [];
	let prev = 0;
	for (const level of levels) {
		let amount = 0;
		for (const p of players) amount += Math.max(0, Math.min(p.committed, level) - prev);
		if (amount > 0) {
			pots.push({
				amount,
				eligible: players.filter((p) => contending(p) && p.committed >= level).map((p) => p.seat)
			});
		}
		prev = level;
	}
	// Chips a folded player put in above every remaining stake go to the last pot.
	let extra = 0;
	for (const p of players) extra += Math.max(0, p.committed - prev);
	if (extra > 0 && pots.length > 0) pots[pots.length - 1].amount += extra;
	return pots;
}

function sync(g: Game): void {
	g.pot = g.players.reduce((sum, p) => sum + p.committed, 0);
	g.next_big_blind = blindsForHand(g.hand_number + 1).big;
	if (g.phase === 'betting' || g.phase === 'dealing') g.pots = computePots(g.players);
}

/** Return the part of the largest bet that nobody matched. */
function refundUncalled(g: Game, events: GameEvent[] | null): void {
	const byBet = g.players.filter((p) => p.in_hand).sort((a, b) => b.bet - a.bet);
	const top = byBet[0];
	const second = byBet[1]?.bet ?? 0;
	if (!top || top.bet <= second) return;
	const diff = top.bet - second;
	top.bet -= diff;
	top.committed -= diff;
	top.chips += diff;
	top.all_in = false;
	events?.push({ kind: 'info', text: `Uncalled bet of ${diff} returned to ${top.name}` });
}

/** Normalise picked cards to 2 + 5 slots, dropping invalid or repeated cards. */
function cleanForced(
	hole: readonly (Card | null)[] | undefined,
	board: readonly (Card | null)[] | undefined
): { hole: (Card | null)[]; board: (Card | null)[] } {
	const seen = new Set<Card>();
	const clean = (list: readonly (Card | null)[] | undefined, size: number) =>
		Array.from({ length: size }, (_, i) => {
			const c = list?.[i];
			if (c === null || c === undefined || !isCard(c) || seen.has(c)) return null;
			seen.add(c);
			return c;
		});
	return { hole: clean(hole, 2), board: clean(board, 5) };
}

/**
 * Deal a new hand: move the button, post blinds, deal hole cards. `forcedHole` / `forcedBoard` fix
 * the human's hole cards and the community cards for this deal; null slots are dealt at random.
 */
export function startHand(
	g: Game,
	rng: Rng,
	forcedHole?: readonly (Card | null)[],
	forcedBoard?: readonly (Card | null)[]
): GameEvent[] {
	if (g.phase !== 'idle' && g.phase !== 'hand_over') throw new Error(`Cannot deal in ${g.phase}`);
	const events: GameEvent[] = [];

	g.hand_number++;
	const blinds = blindsForHand(g.hand_number);
	g.small_blind = blinds.small;
	g.big_blind = blinds.big;

	for (const p of g.players) {
		p.in_hand = p.chips > 0;
		p.folded = !p.in_hand;
		p.all_in = false;
		p.bet = 0;
		p.committed = 0;
		p.acted = false;
		p.badge = '';
		p.revealed = false;
		p.won = 0;
		p.hand_name = '';
		p.best_cards = [];
		p.hole = [];
		p.start_chips = p.chips;
	}
	const alive = g.players.filter((p) => p.in_hand);
	if (alive.length < 2) throw new Error('Need two players with chips');

	g.dealer = nextSeat(g, g.dealer, (p) => p.in_hand);
	if (alive.length === 2) {
		// Heads-up: the button posts the small blind and acts first before the flop.
		g.sb_seat = g.dealer;
		g.bb_seat = nextSeat(g, g.dealer, (p) => p.in_hand);
	} else {
		g.sb_seat = nextSeat(g, g.dealer, (p) => p.in_hand);
		g.bb_seat = nextSeat(g, g.sb_seat, (p) => p.in_hand);
	}

	const fixed = cleanForced(forcedHole, forcedBoard);
	const taken = new Set<Card>([...fixed.hole, ...fixed.board].filter((c): c is Card => c !== null));
	const deck = shuffle(
		freshDeck().filter((c) => !taken.has(c)),
		rng
	);
	const draw = () => deck.pop() as Card;
	for (const p of alive) {
		p.hole = p.is_human ? fixed.hole.map((c) => c ?? draw()) : [draw(), draw()];
	}
	g.runout = fixed.board.map((c) => c ?? draw());
	g.board = [];
	g.street = 'preflop';
	g.headline = '';
	g.pots = [];
	g.raisers = [];

	const sb = g.players[g.sb_seat];
	const bb = g.players[g.bb_seat];
	put(sb, g.small_blind);
	sb.badge = sb.all_in ? 'All-in' : 'SB';
	put(bb, g.big_blind);
	bb.badge = bb.all_in ? 'All-in' : 'BB';
	g.current_bet = g.big_blind;
	g.min_raise = g.big_blind;

	events.push({
		kind: 'hand',
		text: `Hand #${g.hand_number} · blinds ${g.small_blind}/${g.big_blind} · ${g.players[g.dealer].name} on the button`
	});
	const bbText = bb.is_human ? 'you post' : says(bb, 'post');
	events.push({
		kind: 'action',
		text: `${says(sb, 'post')} small blind ${sb.bet}, ${bbText} big blind ${bb.bet}`
	});
	const human = alive.find((p) => p.is_human);
	if (human)
		events.push({ kind: 'info', text: `Your cards: ${human.hole.map(cardText).join(' ')}` });

	settle(g, g.bb_seat, events);
	sync(g);
	return events;
}

/** After a blind post or an action: pick the next player, or close the round. */
function settle(g: Game, start: number, events: GameEvent[]): void {
	const contenders = g.players.filter(contending);
	if (contenders.length === 1) {
		awardUncontested(g, contenders[0], events);
		return;
	}
	const next = nextSeat(g, start, (p) => needsToAct(g, p));
	if (next !== -1) {
		g.to_act = next;
		g.phase = 'betting';
		return;
	}
	g.to_act = -1;
	g.phase = 'dealing';
	refundUncalled(g, events);
	// With at most one player able to bet there is no more action: table the hands.
	if (contenders.filter((p) => !p.all_in).length < 2 && contenders.some((p) => !p.revealed)) {
		for (const p of contenders) p.revealed = true;
		if (g.street !== 'river')
			events.push({ kind: 'info', text: 'All-in — hands are turned face up' });
	}
}

export function legalActions(g: Game): Legal {
	if (g.phase !== 'betting' || g.to_act < 0) throw new Error('No player to act');
	const p = g.players[g.to_act];
	const owed = owedBy(g, p);
	const stackTo = p.bet + p.chips;
	// No point betting more than the deepest opponent who can still respond.
	const cover = Math.max(
		0,
		...g.players.filter((o) => o !== p && canBet(o)).map((o) => o.bet + o.chips)
	);
	const maxTo = Math.min(stackTo, cover);
	const canRaise = !p.acted && maxTo > g.current_bet;
	return {
		seat: p.seat,
		to_call: Math.min(owed, p.chips),
		can_check: owed === 0,
		can_call: owed > 0,
		can_raise: canRaise,
		min_to: canRaise ? Math.min(maxTo, g.current_bet + g.min_raise) : 0,
		max_to: canRaise ? maxTo : 0,
		stack_to: stackTo,
		opening: g.current_bet === 0,
		current_bet: g.current_bet,
		pot: g.players.reduce((sum, o) => sum + o.committed, 0),
		chips: p.chips
	};
}

/**
 * Apply the decision of the player to act. Illegal requests fall back to the nearest legal one (a
 * raise that is not allowed becomes a call, a check facing a bet becomes a fold, ...).
 */
export function applyAction(g: Game, action: Action): GameEvent[] {
	const legal = legalActions(g);
	const p = g.players[g.to_act];
	const events: GameEvent[] = [];

	let kind: Action['type'] = action.type;
	if (kind === 'raise' && !legal.can_raise) kind = legal.can_call ? 'call' : 'check';
	if (kind === 'call' && !legal.can_call) kind = 'check';
	if (kind === 'check' && !legal.can_check) kind = 'fold';
	if (kind === 'fold' && legal.can_check) kind = 'check';

	if (kind === 'fold') {
		p.folded = true;
		p.badge = 'Fold';
		events.push({ kind: 'action', text: says(p, 'fold') });
	} else if (kind === 'check') {
		p.acted = true;
		p.badge = 'Check';
		events.push({ kind: 'action', text: says(p, 'check') });
	} else if (kind === 'call') {
		const paid = put(p, legal.to_call);
		p.acted = true;
		p.badge = p.all_in ? 'All-in' : 'Call';
		events.push({
			kind: 'action',
			text: `${says(p, 'call')} ${paid}${p.all_in ? ' — all-in' : ''}`
		});
	} else {
		const wanted = action.type === 'raise' ? action.amount : legal.min_to;
		const to = Math.min(legal.max_to, Math.max(legal.min_to, Math.trunc(wanted) || 0));
		const increase = to - g.current_bet;
		put(p, to - p.bet);
		if (increase >= g.min_raise) {
			// A full raise reopens the betting for everyone else. (An all-in for less does not:
			// players who already acted may only call or fold.)
			g.min_raise = increase;
			for (const o of g.players) if (o !== p) o.acted = false;
		}
		g.current_bet = to;
		if (g.street === 'preflop') g.raisers.push(p.seat);
		p.acted = true;
		p.badge = p.all_in ? 'All-in' : legal.opening ? 'Bet' : 'Raise';
		const text = legal.opening ? `${says(p, 'bet')} ${to}` : `${says(p, 'raise')} to ${to}`;
		events.push({ kind: 'action', text: `${text}${p.all_in ? ' — all-in' : ''}` });
	}

	settle(g, p.seat, events);
	sync(g);
	return events;
}

const capitalize = (text: string) => text[0].toUpperCase() + text.slice(1);

/** Close the betting round: deal the next street, or go to showdown after the river. */
export function advance(g: Game): GameEvent[] {
	if (g.phase !== 'dealing') throw new Error(`Cannot advance in ${g.phase}`);
	const events: GameEvent[] = [];

	for (const p of g.players) {
		p.bet = 0;
		p.acted = false;
		if (canBet(p)) p.badge = '';
	}
	g.current_bet = 0;
	g.min_raise = g.big_blind;

	if (g.street === 'river') {
		showdown(g, events);
		sync(g);
		return events;
	}

	if (g.street === 'preflop') {
		g.street = 'flop';
		g.board = g.runout.slice(0, 3);
	} else if (g.street === 'flop') {
		g.street = 'turn';
		g.board = g.runout.slice(0, 4);
	} else {
		g.street = 'river';
		g.board = g.runout.slice(0, 5);
	}
	events.push({
		kind: 'street',
		text: `${capitalize(g.street)}: ${g.board.map(cardText).join(' ')}`
	});
	// Tell the human what they hold now (robots' hands stay unnamed until a showdown).
	for (const p of g.players) {
		if (p.is_human && contending(p)) p.hand_name = evaluate([...p.hole, ...g.board]).name;
	}

	// After the flop the first live player left of the button opens.
	const next = nextSeat(g, g.dealer, (p) => needsToAct(g, p));
	if (next !== -1) {
		g.to_act = next;
		g.phase = 'betting';
	}
	sync(g);
	return events;
}

function listNames(names: string[]): string {
	if (names.length <= 1) return names.join('');
	return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

function awardUncontested(g: Game, winner: Player, events: GameEvent[]): void {
	refundUncalled(g, null);
	const total = g.players.reduce((sum, p) => sum + p.committed, 0);
	winner.chips += total;
	winner.won = total;
	g.pots = [{ amount: total, eligible: [winner.seat] }];
	g.headline = `${says(winner, 'win')} ${total}`;
	events.push({ kind: 'win', text: `${g.headline} — everyone else folded` });
	finishHand(g, events);
}

function showdown(g: Game, events: GameEvent[]): void {
	const pots = computePots(g.players);
	g.pots = pots;

	// Seats in showdown order: clockwise from the left of the button.
	const n = g.players.length;
	const order: Player[] = [];
	for (let i = 1; i <= n; i++) {
		const p = g.players[(g.dealer + i) % n];
		if (contending(p)) order.push(p);
	}

	const scores = new Map<number, number>();
	for (const p of order) {
		const value = evaluate([...p.hole, ...g.board]);
		scores.set(p.seat, value.score);
		p.hand_name = value.name;
		p.best_cards = value.cards;
		p.revealed = true;
		events.push({
			kind: 'info',
			text: `${says(p, 'show')} ${p.hole.map(cardText).join(' ')} — ${value.name}`
		});
	}

	pots.forEach((pot, index) => {
		const eligible = order.filter((p) => pot.eligible.includes(p.seat));
		const best = Math.max(...eligible.map((p) => scores.get(p.seat)!));
		const winners = eligible.filter((p) => scores.get(p.seat) === best);
		const share = Math.floor(pot.amount / winners.length);
		const odd = pot.amount % winners.length;
		winners.forEach((p, i) => {
			// Odd chips go to the winners closest to the left of the button.
			const amount = share + (i < odd ? 1 : 0);
			p.chips += amount;
			p.won += amount;
		});
		const potName =
			pots.length === 1 ? 'the pot' : index === 0 ? 'the main pot' : `side pot ${index}`;
		const who =
			winners.length === 1
				? says(winners[0], 'win')
				: `${listNames(winners.map((p) => p.name))} split`;
		events.push({
			kind: 'win',
			text: `${who} ${potName} (${pot.amount}) with ${winners[0].hand_name}`
		});
	});

	const paid = order.filter((p) => p.won > 0).sort((a, b) => b.won - a.won);
	g.headline = paid.map((p) => `${says(p, 'win')} ${p.won} with ${p.hand_name}`).join(' · ');
	finishHand(g, events);
}

function finishHand(g: Game, events: GameEvent[]): void {
	for (const p of g.players) {
		p.bet = 0;
		p.committed = 0;
	}
	g.to_act = -1;
	g.current_bet = 0;

	const remaining = g.players.filter((p) => p.chips > 0);
	// Players eliminated together rank by the stack they started the hand with.
	const busted = g.players
		.filter((p) => p.in_hand && p.chips === 0)
		.sort((a, b) => b.start_chips - a.start_chips);
	busted.forEach((p, i) => {
		p.place = remaining.length + 1 + i;
		const who = p.is_human ? 'You are' : `${p.name} is`;
		events.push({ kind: 'info', text: `${who} out in ${ordinal(p.place)} place` });
	});

	// The game runs until one player holds every chip - if the human busts, the robots play on for
	// the remaining places.
	if (remaining.length === 1) {
		remaining[0].place = 1;
		g.phase = 'game_over';
		events.push({ kind: 'win', text: `${says(remaining[0], 'win')} the game!` });
	} else {
		g.phase = 'hand_over';
	}
}

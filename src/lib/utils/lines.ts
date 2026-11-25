// Words for the steps of a hand line, shared by the page and its step table.
import type { LineState, Side } from '$lib/poker/line';
import { SUIT_SYMBOLS, rankLabel, rankOf, suitOf } from './cards';

const cardText = (c: number) => `${rankLabel(rankOf(c))}${SUIT_SYMBOLS[suitOf(c)]}`;

export function describeStep(state: LineState, seat: (side: Side) => string): string {
	const step = state.step;
	if (!step) return 'Before the flop';
	if (step.kind === 'deal') {
		const name = state.board.length === 3 ? 'Flop' : state.board.length === 4 ? 'Turn' : 'River';
		return `${name} ${step.cards.map(cardText).join(' ')}`;
	}
	const who = seat(step.by);
	const amount = (v: number) => `${Number(v.toFixed(2))}bb`;
	if (step.kind === 'bet') return `${who} bets ${amount(step.to)}`;
	if (step.kind === 'raise') return `${who} raises to ${amount(step.to)}`;
	return `${who} ${step.kind}s`;
}

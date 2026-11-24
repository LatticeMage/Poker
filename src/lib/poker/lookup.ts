// The charts as plain values: pure, cached functions of their inputs. A component asks for the chart
// it needs inside a $derived, so it follows its inputs by itself - nothing is loaded, nothing is
// stored, and no effect is needed.
//
//   const chart = $derived(attempt(() => rangeChart(players)));   // { value } or { error }

import type {
	BoardEquity,
	BoardQuery,
	HeadsUpChart,
	PushFoldChart,
	RangeChart,
	SpotChart,
	SpotList
} from '$lib/types/ranges';
import { cardCode } from '$lib/utils/cards';
import * as charts from './charts';

export const HEADS_UP_STACKS = charts.HEADS_UP_STACKS;

const LIMIT = 200;

/** A cached version of `fn`: one answer per key; the oldest are dropped past LIMIT. */
function cached<A extends unknown[], T>(key: (...args: A) => string, fn: (...args: A) => T) {
	const seen = new Map<string, T>();
	return (...args: A): T => {
		const k = key(...args);
		if (seen.has(k)) return seen.get(k) as T;
		const value = fn(...args);
		seen.set(k, value);
		if (seen.size > LIMIT) seen.delete(seen.keys().next().value as string);
		return value;
	};
}

export const rangeChart = cached((players: number) => String(players), charts.rangeChart) as (
	players: number
) => RangeChart;
export const pushFoldChart = cached((stack: number) => String(stack), charts.pushFoldChart) as (
	stack: number
) => PushFoldChart;
export const headsUpChart = cached((stack: number) => String(stack), charts.headsUpChart) as (
	stack: number
) => HeadsUpChart;
export const spotList = cached(() => '', charts.spotList) as () => SpotList;
export const spotChart = cached(
	(kind: SpotChart['kind'], hero: string, villain: string) => `${kind}:${hero}:${villain}`,
	charts.spotChart
) as (kind: SpotChart['kind'], hero: string, villain: string) => SpotChart;

export const boardEquity = cached(
	(q: BoardQuery) =>
		[
			q.pot,
			q.opener,
			q.other,
			q.side,
			q.board.map(cardCode).join(''),
			q.hole ? q.hole.map(cardCode).join('') : '-',
			q.bet ?? '-',
			q.view
		].join('|'),
	charts.boardEquity
) as (q: BoardQuery) => BoardEquity;

export type Attempt<T> = { value: T; error?: undefined } | { value?: undefined; error: string };

/** Run a lookup that may be refused (no chart for this spot): the value, or the reason. */
export function attempt<T>(fn: () => T): Attempt<T> {
	try {
		return { value: fn() };
	} catch (cause) {
		return { error: cause instanceof Error ? cause.message : 'Could not load the chart' };
	}
}

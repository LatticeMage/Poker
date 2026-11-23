// Chart controller: computes a chart locally (lib/poker) and hands the answer to the store. The
// functions return promises so the components keep their "load, then render from the store" shape,
// but nothing leaves the browser.

import { ChartError } from '$lib/poker/charts';
import * as charts from '$lib/poker/charts';
import { rangeStore, spotKey } from '$lib/stores/ranges.svelte';
import type { BoardQuery, SpotChart } from '$lib/types/ranges';
import { cardCode } from '$lib/utils/cards';

async function load(compute: () => void): Promise<void> {
	rangeStore.loading = true;
	rangeStore.error = null;
	try {
		compute();
	} catch (error) {
		rangeStore.error =
			error instanceof ChartError || error instanceof Error
				? error.message
				: 'Could not load the chart';
	} finally {
		rangeStore.loading = false;
	}
}

export function loadRangeChart(players: number): Promise<void> {
	if (rangeStore.charts[players]) return Promise.resolve();
	return load(() => {
		rangeStore.charts = { ...rangeStore.charts, [players]: charts.rangeChart(players) };
	});
}

export function loadPushFold(stackBb: number): Promise<void> {
	if (rangeStore.pushFold[stackBb]) return Promise.resolve();
	return load(() => {
		rangeStore.pushFold = { ...rangeStore.pushFold, [stackBb]: charts.pushFoldChart(stackBb) };
	});
}

export function loadSpotList(): Promise<void> {
	if (rangeStore.spotList) return Promise.resolve();
	return load(() => {
		rangeStore.spotList = charts.spotList();
	});
}

export function loadSpot(kind: SpotChart['kind'], hero: string, villain: string): Promise<void> {
	const key = spotKey(kind, hero, villain);
	if (rangeStore.spots[key]) return Promise.resolve();
	return load(() => {
		rangeStore.spots = { ...rangeStore.spots, [key]: charts.spotChart(kind, hero, villain) };
	});
}

/** The cache key of a board-equity query. */
export function boardRequest(q: BoardQuery): string {
	return [
		q.pot,
		q.opener,
		q.other,
		q.side,
		q.board.map(cardCode).join(''),
		q.hole ? q.hole.map(cardCode).join('') : '-',
		q.bet ?? '-',
		q.view
	].join('|');
}

export function loadBoard(q: BoardQuery): Promise<void> {
	const key = boardRequest(q);
	if (rangeStore.boards[key]) return Promise.resolve();
	return load(() => {
		rangeStore.boards = { ...rangeStore.boards, [key]: charts.boardEquity(q) };
	});
}

export function loadHeadsUp(stackBb: number): Promise<void> {
	if (rangeStore.headsUp[stackBb]) return Promise.resolve();
	return load(() => {
		rangeStore.headsUp = { ...rangeStore.headsUp, [stackBb]: charts.headsUpChart(stackBb) };
	});
}

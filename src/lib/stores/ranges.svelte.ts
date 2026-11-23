// Range charts as the server last described them: opening charts by table size, push/fold by stack,
// and the facing-a-raise spots by position.

import type {
	BoardEquity,
	HeadsUpChart,
	PushFoldChart,
	RangeChart,
	SpotChart,
	SpotList
} from '$lib/types/ranges';

class RangeStore {
	charts = $state.raw<Record<number, RangeChart>>({});
	pushFold = $state.raw<Record<number, PushFoldChart>>({});
	headsUp = $state.raw<Record<number, HeadsUpChart>>({});
	/** Keyed `${kind}:${hero}:${villain}`. */
	spots = $state.raw<Record<string, SpotChart>>({});
	spotList = $state.raw<SpotList | null>(null);
	/** Keyed by the query string of the flop-equity request. */
	boards = $state.raw<Record<string, BoardEquity>>({});
	loading = $state(false);
	error = $state<string | null>(null);
}

export const rangeStore = new RangeStore();

export const spotKey = (kind: string, hero: string, villain: string) =>
	`${kind}:${hero}:${villain}`;

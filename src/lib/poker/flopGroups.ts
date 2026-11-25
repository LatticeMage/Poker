// The flop groups as the base of the postflop decision tree. A flop is looked up by its
// suit-isomorphism class (1,755 of them) and falls in one of 383 groups, named by its characteristics
// level by level (nvcc/flops.cuh, flop_shape):
//
//   色數 ｜ 高數 ｜ 高連度 ｜ 低連度 ｜ 同順      the structure: 73 groups, which explain the wetness
//   ｜ 哪幾張高牌 ｜ 低牌中小                    which ranks are out: 383 groups
//
// The structure alone does not say whose range a flop is good for (it explains a quarter of the
// flop-to-flop differences in range equity); which ranks are out does (91% in single-raised pots, no
// group with more than 1.5 points of spread on average). Both were measured on every flop for every
// matchup of the charts by nvcc/flopstats.cu.
//
// Tables: data/flop-groups.json (tools/make-flop-groups.ts, from nvcc/data/flops-1755.csv) and
// data/flop-group-stats.json (`flopstats groups`).

import type { Card } from './cards';
import table from './data/flop-groups.json';
import stats from './data/flop-group-stats.json';

export interface FlopGroup {
	/** 1..383, in the order of the characteristics. */
	id: number;
	/** E.g. "三花｜1高｜低斷｜不同順｜K｜中小": the levels of the tree, see above. */
	name: string;
	/** The structure group (1..73) this group refines: the first five levels. */
	base: number;
	/** Share of the 1,176 hole-card hands that make or draw a flush or a straight, group mean. */
	wetness: number;
	/** How many of the 22,100 flops are in the group. */
	flops: number;
}

const RANK_CHARS = '..23456789TJQKA';
const SUIT_CHARS = 'shdc';

/** All 24 ways to name the four suits. */
const PERMUTATIONS: number[][] = (() => {
	const out: number[][] = [];
	const go = (rest: number[], taken: number[]) => {
		if (rest.length === 0) out.push(taken);
		for (let i = 0; i < rest.length; i++) {
			go([...rest.slice(0, i), ...rest.slice(i + 1)], [...taken, rest[i]]);
		}
	};
	go([0, 1, 2, 3], []);
	return out;
})();

/**
 * The class of a flop under renaming the suits: the smallest text any suit renaming gives, with the
 * cards written high to low. "AsKsQs" and "AhKhQh" have the same key.
 */
export function flopKey(flop: readonly Card[]): string {
	let best = '';
	for (const perm of PERMUTATIONS) {
		const text = flop
			.map((c) => ({ rank: (c >> 2) + 2, suit: perm[c & 3] }))
			.sort((a, b) => b.rank - a.rank || a.suit - b.suit)
			.map((c) => RANK_CHARS[c.rank] + SUIT_CHARS[c.suit])
			.join('');
		if (best === '' || text < best) best = text;
	}
	return best;
}

const groups: FlopGroup[] = (table.groups as [string, number, number, number][]).map(
	([name, base, wetness, flops], i) => ({ id: i + 1, name, base, wetness, flops })
);

/** The group of a flop (the first three cards of the board). */
export function flopGroup(board: readonly Card[]): FlopGroup {
	const id = (table.classes as Record<string, number>)[flopKey(board.slice(0, 3))];
	if (id === undefined) throw new Error(`Not a flop: ${board.slice(0, 3).join(',')}`);
	return groups[id - 1];
}

export const FLOP_GROUPS: readonly FlopGroup[] = groups;

/** The branches of the group name: suits, high cards, connectedness (the levels of the tree). */
export const groupLevels = (group: FlopGroup): string[] => group.name.split('｜');

/**
 * What two preflop ranges look like on the flops of one group (nvcc/flopstats.cu: exact all-in equity
 * of every combo against the other range on every flop of the group, every turn and river counted).
 */
export interface GroupStats {
	/** Equity of the preflop aggressor's range, mean over the group; the caller's is 1 - eq. */
	eq: number;
	/** The lowest and highest range equity among the group's flops: how far one flop can be off. */
	eq_min: number;
	eq_max: number;
	/** Share of each range with at least 80% equity: the nut part. */
	nut_aggressor: number;
	nut_caller: number;
}

/** The matchup key of the table: pot kind, opener, the other player. */
export const matchupKey = (kind: 'srp' | '3bet' | '4bet', opener: string, other: string): string =>
	`${kind}:${opener}:${other}`;

const VALUES = 5; // per group, each packed as two base-32 digits of round(value * 1023)

/** The table row of a matchup on a flop group, or null if the charts have no such matchup. */
export function groupStats(key: string, group: FlopGroup): GroupStats | null {
	const packed = (stats.matchups as Record<string, string>)[key];
	if (!packed) return null;
	const at = (group.id - 1) * VALUES * 2;
	const [eq, eq_min, eq_max, nut_aggressor, nut_caller] = Array.from(
		{ length: VALUES },
		(_, i) => parseInt(packed.slice(at + i * 2, at + i * 2 + 2), 32) / 1023
	);
	return { eq, eq_min, eq_max, nut_aggressor, nut_caller };
}

export const MATCHUP_KEYS: readonly string[] = Object.keys(stats.matchups);

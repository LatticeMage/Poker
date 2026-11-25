// Writes nvcc/data/matchups.json: every heads-up postflop matchup the preflop charts give ranges for
// (single-raised, 3-bet and 4-bet pots), as two ranges of 169 weights in ALL_HANDS order. It is the
// input of nvcc/flopstats.cu, so the GPU tool and the app read the same ranges (postflop.potRanges).
//
//   npm run matchups

import { writeFileSync } from 'node:fs';
import { potRanges } from '../src/lib/poker/postflop';
import { ALL_HANDS, seatRange } from '../src/lib/poker/ranges';
import { spotPairs } from '../src/lib/poker/spots';

const OPEN_INDEX: Record<string, number> = { UTG: 0, MP: 1, CO: 2, BTN: 3, SB: 4 };
const weights = (range: Record<string, number>) => ALL_HANDS.map((h) => range[h] ?? 0);
const combos = (w: number[]) =>
	Math.round(
		w.reduce(
			(s, x, i) => s + x * (ALL_HANDS[i].length === 2 ? 6 : ALL_HANDS[i][2] === 's' ? 4 : 12),
			0
		)
	);

const out: { key: string; aggressor: number[]; caller: number[] }[] = [];
const pairs = spotPairs();
const add = (kind: 'srp' | '3bet' | '4bet', opener: string, other: string) => {
	if (!(opener in OPEN_INDEX)) return;
	const built = potRanges(kind, opener, other, seatRange(6, OPEN_INDEX[opener]).hands);
	if (!built) return;
	const [aggressor, caller] = built.map(weights);
	out.push({ key: `${kind}:${opener}:${other}`, aggressor, caller });
	console.log(`${kind}:${opener}:${other}`.padEnd(16), combos(aggressor), 'vs', combos(caller));
};
// vsOpen pairs are [hero, opener]; vs3bet pairs are [opener, 3-bettor].
for (const [hero, opener] of pairs.vsOpen) add('srp', opener, hero);
for (const [opener, bettor] of pairs.vs3bet) add('3bet', opener, bettor);
for (const [opener, bettor] of pairs.vs3bet) add('4bet', opener, bettor);

writeFileSync('nvcc/data/matchups.json', JSON.stringify(out));
console.log(`${out.length} matchups`);

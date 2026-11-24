// Copies the GPU push/fold solution into the data files the app reads.
//
//   npm run apply-pushfold -- [nvcc/data/out/pushfold.json]
//
// The solution comes from `nvcc/pushfold` (CUDA CFR+ on the exact equity matrix, see nvcc/README.md). This
// script only moves data, it computes nothing:
//   src/lib/poker/data/tables.json        pushfold, equity_vs_random, samples_per_matchup
//   src/lib/poker/data/pushfold-mix.json  the mixed frequencies per hand
// Everything else in tables.json stays as it is. Run `npx prettier --write src/lib/poker/data` afterwards: the
// files are committed in prettier's layout.

import { readFileSync, writeFileSync } from 'node:fs';

/**
 * tables.json counts "samples per matchup". The exact matrix has no samples: every pair of combos is dealt
 * every one of the C(48,5) boards. The field keeps that number so that it stays > 0 and tells the truth.
 */
const BOARDS = 1_712_304;

const source = process.argv[2] ?? 'nvcc/data/out/pushfold.json';
const solved = JSON.parse(readFileSync(source, 'utf8')) as {
	equity_vs_random: Record<string, number>;
	pushfold: Record<string, unknown>;
	mix: Record<string, unknown>;
	samples_per_matchup: number;
};
if (solved.samples_per_matchup !== 0)
	throw new Error('expected the exact matrix (samples_per_matchup 0)');

const tablesPath = 'src/lib/poker/data/tables.json';
const tables = JSON.parse(readFileSync(tablesPath, 'utf8')) as Record<string, unknown>;
tables.samples_per_matchup = BOARDS;
tables.equity_vs_random = solved.equity_vs_random;
tables.pushfold = solved.pushfold;
writeFileSync(tablesPath, JSON.stringify(tables, null, '\t') + '\n');
writeFileSync(
	'src/lib/poker/data/pushfold-mix.json',
	JSON.stringify(solved.mix, null, '\t') + '\n'
);
console.log(`wrote ${tablesPath} and pushfold-mix.json from ${source}`);

// Generates src/lib/poker/data/flop-groups.json from the list of the 1,755 flops
// (nvcc/data/flops-1755.csv, made by `./flops list`): for every flop class its group (1..383, level 7
// of the classification), and for every group its name, the structure group it refines (1..73,
// level 5), its wetness (mean over its flops, weighted by weight) and its flop count.
//
//   npm run flop-groups   (then `npx prettier --write src/lib/poker/data/flop-groups.json`)

import { readFileSync, writeFileSync } from 'node:fs';
import { parseCompact } from '../src/lib/poker/cards';
import { flopKey } from '../src/lib/poker/flopGroups';

const [header, ...rows] = readFileSync('nvcc/data/flops-1755.csv', 'utf8').trim().split('\n');
const column = Object.fromEntries(header.split(',').map((name, i) => [name, i]));
const classes: Record<string, number> = {};
const sums = new Map<number, { name: string; base: number; wet: number; flops: number }>();

for (const row of rows) {
	const c = row.split(',');
	const weight = +c[column.weight];
	const group = +c[column.group];
	classes[flopKey(parseCompact(c[column.flop]))] = group;
	const g = sums.get(group) ?? {
		name: c[column.group_name],
		base: +c[column.base_group],
		wet: 0,
		flops: 0
	};
	g.wet += +c[column.share] * weight;
	g.flops += weight;
	sums.set(group, g);
}

const groups = [...sums.entries()]
	.sort(([a], [b]) => a - b)
	.map(([, g]) => [g.name, g.base, Math.round((g.wet / g.flops) * 1e4) / 1e4, g.flops]);
writeFileSync('src/lib/poker/data/flop-groups.json', JSON.stringify({ groups, classes }));
console.log(`${Object.keys(classes).length} classes, ${groups.length} groups`);

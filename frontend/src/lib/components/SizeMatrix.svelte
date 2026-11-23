<!--
	How much each seat opens at every table size, side by side. Click a cell to study it.
-->

<script lang="ts">
	import { loadRangeChart } from '$lib/actions/ranges';
	import { rangeStore } from '$lib/stores/ranges.svelte';

	let {
		players,
		seatName,
		onselect
	}: { players: number; seatName: string; onselect: (players: number, seat: string) => void } =
		$props();

	const SIZES = [2, 3, 4, 5, 6, 7, 8, 9];
	const SEATS = ['UTG', 'UTG+1', 'UTG+2', 'LJ', 'HJ', 'CO', 'BTN', 'SB', 'SB/BTN'];

	$effect(() => {
		for (const n of SIZES) void loadRangeChart(n);
	});

	function pct(n: number, name: string): number | null {
		const seat = rangeStore.charts[n]?.seats.find((s) => s.name === name && s.source !== 'none');
		return seat ? seat.pct : null;
	}

	// Only list seats that exist at some table size.
	const rows = $derived(SEATS.filter((name) => SIZES.some((n) => pct(n, name) !== null)));
</script>

<div class="overflow-x-auto">
	<table class="w-full border-separate border-spacing-0.5 text-center text-xs tabular-nums">
		<caption class="pb-1 text-left text-xs text-slate-400">
			% of hands opened first-in, by number of players at the table
		</caption>
		<thead>
			<tr>
				<th scope="col" class="text-left font-semibold text-slate-400">Seat</th>
				{#each SIZES as n (n)}
					<th scope="col" class="font-semibold text-slate-400" class:text-amber-400={n === players}
						>{n}</th
					>
				{/each}
			</tr>
		</thead>
		<tbody>
			{#each rows as name (name)}
				<tr>
					<th scope="row" class="pr-1 text-left font-semibold text-slate-300">{name}</th>
					{#each SIZES as n (n)}
						{@const value = pct(n, name)}
						<td class="p-0">
							{#if value === null}
								<span class="block py-1 text-slate-700">–</span>
							{:else}
								<button
									class="cell"
									class:current={n === players && name === seatName}
									style:--heat={Math.min(1, value / 60)}
									aria-label={`${name} at ${n} players: ${value}%`}
									onclick={() => onselect(n, name)}
								>
									{Math.round(value)}
								</button>
							{/if}
						</td>
					{/each}
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<style>
	.cell {
		display: block;
		width: 100%;
		border-radius: 0.25rem;
		padding: 0.25rem 0;
		background: rgb(217 119 6 / calc(0.12 + var(--heat) * 0.7));
		color: #fff7e6;
		font-weight: 600;
	}

	.cell:hover {
		outline: 1px solid #f8fafc;
	}

	.cell.current {
		outline: 2px solid #f8fafc;
	}
</style>

<!--
	Starting-hand charts. Two views:
	  · Preflop: pick a table size and a seat, then the situation (first in, facing an open, or
	    opened and facing a 3-bet). One grid shows fold / call / raise as shares inside each cell.
	  · Short stack: heads-up push/fold for a stack of 2-20bb.
	Click a hand to study it. `players` and `seatName` can be bound, so a page can steer the chart.
-->

<script lang="ts">
	import {
		HEADS_UP_STACKS as headsUpStacks,
		attempt,
		headsUpChart,
		pushFoldChart,
		rangeChart,
		spotChart,
		spotList as getSpotList
	} from '$lib/poker/lookup';
	import { settings } from '$lib/stores/settings.svelte';
	import { handCombos, handTitle, percentiles, positionClass } from '$lib/utils/hands';
	import HandGrid from './HandGrid.svelte';

	let {
		players = $bindable(settings.players),
		seatName = $bindable('')
	}: { players?: number; seatName?: string } = $props();

	let mode = $state<'open' | 'headsup' | 'pushfold'>('open');
	let picked = $state<string | null>(null);

	// Opening view.
	const chartResult = $derived(attempt(() => rangeChart(players)));
	const chart = $derived(chartResult.value);
	// The chosen seat by name, kept when the table size changes if the seat still exists; the big
	// blind never opens, so a missing or blind seat falls back to the first seat.
	const seat = $derived.by(() => {
		if (!chart) return null;
		return (
			chart.seats.find((s) => s.name === seatName) ?? chart.seats.find((s) => s.source !== 'none')
		);
	});

	// The situation for the chosen seat: first in, facing an open, or opened and then 3-bet.
	let facing = $state('first');
	const spotListResult = $derived(attempt(getSpotList));
	const spotList = $derived(spotListResult.value);
	const hero = $derived(seat ? positionClass(seat.behind, seat.behind === players - 1) : '');
	const situations = $derived.by(() => {
		const out: {
			id: string;
			label: string;
			kind: 'vs_open' | 'vs_3bet' | null;
			villain: string;
		}[] = [];
		if (seat && seat.source !== 'none') {
			out.push({ id: 'first', label: 'First in', kind: null, villain: '' });
		}
		for (const v of spotList?.vs_open[hero] ?? []) {
			out.push({ id: `open:${v}`, label: `vs ${v} open`, kind: 'vs_open', villain: v });
		}
		for (const v of spotList?.vs_3bet[hero] ?? []) {
			out.push({ id: `3bet:${v}`, label: `open, ${v} 3-bets`, kind: 'vs_3bet', villain: v });
		}
		return out;
	});
	const situation = $derived(situations.find((x) => x.id === facing) ?? situations[0]);
	const villain = $derived(situation?.villain ?? '');
	const spotResult = $derived(
		situation?.kind ? attempt(() => spotChart(situation.kind!, hero, villain)) : undefined
	);
	const spot = $derived(spotResult?.value);
	/** Per hand: [raise for value, raise as a bluff, call] shares. */
	const spotMix = $derived.by(() => {
		if (!spot) return undefined;
		const out: Record<string, [number, number, number]> = {};
		for (const hand of new Set([
			...Object.keys(spot.call),
			...Object.keys(spot.bluff),
			...Object.keys(spot.value)
		])) {
			out[hand] = [spot.value[hand] ?? 0, spot.bluff[hand] ?? 0, spot.call[hand] ?? 0];
		}
		return out;
	});

	// Solved heads-up view: the small blind and big blind at 10 / 20 / 40 / 80bb.
	let duelStack = $state(20);
	let duelPlayer = $state<'SB' | 'BB'>('SB');
	let duelNode = $state('root');
	const duelResult = $derived(
		mode === 'headsup' ? attempt(() => headsUpChart(duelStack)) : undefined
	);
	const duel = $derived(duelResult?.value);
	const duelNodes = $derived(
		duel
			? Object.entries(duel.nodes)
					.filter(([, n]) => n.player === duelPlayer)
					.sort(
						(a, b) => a[0].split('/').length - b[0].split('/').length || b[1].reach - a[1].reach
					)
			: []
	);
	const duelCurrent = $derived(duelNodes.find(([id]) => id === duelNode) ?? duelNodes[0]);
	const duelLabel = (id: string) => {
		if (id === 'root') return 'First in';
		const parts = id.split('/');
		const last = parts[parts.length - 1];
		const what = last === 'allin' ? 'all-in' : last;
		if (parts[0] === 'limp') return parts.length === 1 ? 'vs limp' : `limp, then ${what}`;
		return parts.length === 2
			? last === 'allin'
				? 'vs all-in'
				: `vs open ${what}`
			: `facing ${what}`;
	};
	/** Per hand: [raise, all-in, call/limp/check] shares; the rest folds. */
	const duelMix = $derived.by(() => {
		const node = duelCurrent?.[1];
		if (!node) return undefined;
		const out: Record<string, [number, number, number]> = {};
		for (const [hand, shares] of Object.entries(node.freq)) {
			const row: [number, number, number] = [0, 0, 0];
			node.actions.forEach((action, i) => {
				if (action === 'fold') return;
				if (action === 'allin') row[1] += shares[i];
				else if (action.startsWith('raise')) row[0] += shares[i];
				else row[2] += shares[i];
			});
			out[hand] = row;
		}
		return out;
	});

	// Short-stack view.
	let stack = $state(10);
	let side = $state<'push' | 'call'>('push');
	const pushFoldResult = $derived(
		mode === 'pushfold' ? attempt(() => pushFoldChart(stack)) : undefined
	);
	const pushFold = $derived(pushFoldResult?.value);

	// Strength data comes with any opening chart (it does not depend on the table size).
	const baseResult = $derived(mode === 'open' ? chartResult : attempt(() => rangeChart(2)));
	const base = $derived(baseResult.value);
	const percentile = $derived(base ? percentiles(base.order) : {});

	const shown = $derived(
		new Set(
			mode === 'open'
				? (seat?.hands ?? [])
				: mode === 'pushfold'
					? side === 'push'
						? (pushFold?.push ?? [])
						: (pushFold?.call ?? [])
					: []
		)
	);
	// Per hand: how often it opens / shoves / calls (the rest folds), drawn as a partly filled cell.
	const openMix = $derived.by(() => {
		const freq =
			mode === 'open' ? seat?.mix : side === 'push' ? pushFold?.push_mix : pushFold?.call_mix;
		if (!freq) return undefined;
		return Object.fromEntries(
			Object.entries(freq).map(([hand, f]) => [
				hand,
				[f, 0, mode === 'open' ? (seat?.limp?.[hand] ?? 0) : 0] as [number, number, number]
			])
		);
	});
	const verb = $derived(mode === 'open' ? 'raise' : side === 'push' ? 'shove' : 'call');

	// Whatever the shown view could not get: the reason, in place of its chart.
	const problem = $derived(
		(mode === 'open'
			? (chartResult.error ?? spotListResult.error ?? spotResult?.error)
			: mode === 'headsup'
				? duelResult?.error
				: pushFoldResult?.error) ?? baseResult.error
	);

	const sourceText = $derived(
		!seat
			? ''
			: seat.source === 'table'
				? `Chart: ${seat.label}`
				: seat.source === 'computed'
					? `Computed: ${seat.label}`
					: seat.source === 'solver'
						? `Solver: ${seat.label}`
						: seat.label
	);

	// Where the picked hand opens at this table size.
	const raisesFrom = $derived(
		picked && chart
			? chart.seats.filter((s) => s.source !== 'none' && s.hands.includes(picked!))
			: []
	);
	const foldsFrom = $derived(
		picked && chart
			? chart.seats.filter((s) => s.source !== 'none' && !s.hands.includes(picked!))
			: []
	);

	const pct = (n: number) => `${Math.round(n * 100)}%`;
	/** The picked hand's mix in the facing-a-raise view, as readable parts. */
	const pickedMix = $derived.by(() => {
		if (!picked || !spot) return [];
		const value = spot.value[picked] ?? 0;
		const bluff = spot.bluff[picked] ?? 0;
		const call = spot.call[picked] ?? 0;
		const fold = Math.max(0, 1 - value - bluff - call);
		return [
			{ label: spot.labels.value, share: value, color: '#d97706' },
			{ label: spot.labels.bluff, share: bluff, color: '#f43f5e' },
			{ label: spot.labels.call, share: call, color: '#0284c7' },
			{ label: 'Fold', share: fold, color: '#475569' }
		].filter((part) => part.share > 0.004);
	});
</script>

<div class="flex flex-col gap-3">
	<div class="flex flex-wrap gap-1" role="tablist" aria-label="Chart type">
		<button
			class="seat-tab"
			class:selected={mode === 'open'}
			role="tab"
			aria-selected={mode === 'open'}
			onclick={() => (mode = 'open')}
		>
			Preflop
		</button>
		<button
			class="seat-tab"
			class:selected={mode === 'headsup'}
			role="tab"
			aria-selected={mode === 'headsup'}
			onclick={() => (mode = 'headsup')}
		>
			Heads-up solved
		</button>
		<button
			class="seat-tab"
			class:selected={mode === 'pushfold'}
			role="tab"
			aria-selected={mode === 'pushfold'}
			onclick={() => (mode = 'pushfold')}
		>
			Short stack (heads-up)
		</button>
	</div>

	{#if mode === 'open'}
		<label class="flex items-center gap-3 text-sm font-semibold">
			Players
			<input
				class="min-w-0 flex-1 accent-amber-500"
				type="range"
				min="2"
				max="9"
				step="1"
				bind:value={players}
				aria-label="Players at the table"
			/>
			<span class="w-4 text-right tabular-nums">{players}</span>
		</label>

		{#if chart}
			<div class="flex flex-wrap gap-1" role="tablist" aria-label="Seat">
				{#each chart.seats as s (s.name)}
					<button
						class="seat-tab seat"
						class:selected={seat === s}
						role="tab"
						aria-label={s.name}
						aria-selected={seat === s}
						onclick={() => (seatName = s.name)}
					>
						<span>{s.name}</span>
						{#if s.source !== 'none'}<span class="pct">{s.pct}%</span>{/if}
					</button>
				{/each}
			</div>

			{#if seat}
				<div class="flex flex-wrap items-center gap-1" role="tablist" aria-label="Situation">
					{#each situations as option (option.id)}
						<button
							class="seat-tab"
							class:selected={situation?.id === option.id}
							role="tab"
							aria-selected={situation?.id === option.id}
							onclick={() => (facing = option.id)}
						>
							{option.label}
						</button>
					{/each}
				</div>

				<HandGrid
					hands={shown}
					{verb}
					mix={situation?.kind ? spotMix : openMix}
					label={situation?.kind ? `${seat.name} ${situation.label}` : `${seat.name} opening range`}
					selected={picked}
					onpick={(hand) => (picked = hand)}
				/>
				<ul class="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-300" aria-label="Legend">
					{#if situation?.kind && spot}
						<li class="flex items-center gap-1">
							<span class="swatch" style:background="#d97706"></span>{spot.labels.value}
						</li>
						<li class="flex items-center gap-1">
							<span class="swatch" style:background="#f43f5e"></span>{spot.labels.bluff}
						</li>
						<li class="flex items-center gap-1">
							<span class="swatch" style:background="#0284c7"></span>{spot.labels.call}
						</li>
					{:else}
						<li class="flex items-center gap-1">
							<span class="swatch" style:background="#d97706"></span>Raise
						</li>
						{#if seat?.limp}
							<li class="flex items-center gap-1">
								<span class="swatch" style:background="#0284c7"></span>Limp
							</li>
						{/if}
					{/if}
					<li class="flex items-center gap-1">
						<span class="swatch" style:background="#1e293b"></span>Fold
					</li>
				</ul>

				{#if situation?.kind && spot}
					<div class="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 text-sm">
						<span class="tabular-nums">
							<span class="font-bold">{spot.combos.call}</span> combos call ·
							<span class="font-bold">{spot.combos.bluff + spot.combos.value}</span> raise
						</span>
						<a
							class="text-xs text-amber-400 underline"
							href={spot.url}
							target="_blank"
							rel="noreferrer"
						>
							Chart: PokerStrategy 6-max, {spot.hero}
						</a>
					</div>
					<p class="text-xs text-slate-500">{spot.note}</p>
				{:else}
					<div class="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 text-sm">
						<span>
							<span class="font-bold tabular-nums">{seat.pct}%</span> of hands ·
							{seat.behind}
							{seat.behind === 1 ? 'player' : 'players'} behind
						</span>
						<span class="text-xs text-slate-400">
							{#if seat.url}
								<a
									class="text-amber-400 underline"
									href={seat.url}
									target="_blank"
									rel="noreferrer"
								>
									{sourceText}
								</a>
							{:else}
								{sourceText}
							{/if}
						</span>
					</div>
					{#if seat.source === 'solver'}
						<p class="rounded-md bg-slate-800 px-2 py-1 text-xs text-slate-300">
							No published chart for heads-up: this is a solver approximation (see the Heads-up
							solved view), so some hands are mixed between raising, limping and folding.
						</p>
					{:else if seat.source === 'computed' || seat.source === 'fallback'}
						<p class="rounded-md bg-slate-800 px-2 py-1 text-xs text-slate-300">
							{seat.source === 'computed'
								? 'No published chart for this seat: the hands are the strongest ones by the strength order built from the other charts.'
								: 'No source found for this spot, so this is a stand-in. Real heads-up opening ranges are wider.'}
						</p>
					{/if}
				{/if}
			{/if}
		{:else if problem}
			<p class="text-sm text-rose-300">{problem}</p>
		{:else}
			<p class="text-sm text-slate-400">Loading…</p>
		{/if}
	{:else if mode === 'headsup'}
		<div class="flex flex-wrap items-center gap-x-4 gap-y-1">
			<div class="flex items-center gap-1" role="tablist" aria-label="Stack depth">
				<span class="text-xs font-semibold text-slate-400">Stack</span>
				{#each headsUpStacks as depth (depth)}
					<button
						class="seat-tab"
						class:selected={duelStack === depth}
						role="tab"
						aria-selected={duelStack === depth}
						onclick={() => ((duelStack = depth), (duelNode = 'root'))}
					>
						{depth}bb
					</button>
				{/each}
			</div>
			<div class="flex items-center gap-1" role="tablist" aria-label="Seat">
				{#each ['SB', 'BB'] as who (who)}
					<button
						class="seat-tab"
						class:selected={duelPlayer === who}
						role="tab"
						aria-selected={duelPlayer === who}
						onclick={() => (
							(duelPlayer = who as 'SB' | 'BB'), (duelNode = who === 'SB' ? 'root' : 'open/2.5')
						)}
					>
						{who}
					</button>
				{/each}
			</div>
		</div>

		{#if duel && duelCurrent && duelMix}
			<div class="flex flex-wrap items-center gap-1" role="tablist" aria-label="Situation">
				{#each duelNodes as [id, node] (id)}
					<button
						class="seat-tab"
						class:selected={duelCurrent[0] === id}
						role="tab"
						aria-selected={duelCurrent[0] === id}
						title={`${(node.reach * 100).toFixed(1)}% of hands get here`}
						onclick={() => (duelNode = id)}
					>
						{duelLabel(id)}
					</button>
				{/each}
			</div>
			<HandGrid
				hands={new Set()}
				verb="act"
				mix={duelMix}
				label={`${duelPlayer} ${duelLabel(duelCurrent[0])} at ${duelStack}bb`}
				selected={picked}
				onpick={(hand) => (picked = hand)}
			/>
			<ul class="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-300" aria-label="Legend">
				<li class="flex items-center gap-1">
					<span class="swatch" style:background="#d97706"></span>Raise
				</li>
				<li class="flex items-center gap-1">
					<span class="swatch" style:background="#f43f5e"></span>All-in
				</li>
				<li class="flex items-center gap-1">
					<span class="swatch" style:background="#0284c7"></span>Call / limp / check
				</li>
				<li class="flex items-center gap-1">
					<span class="swatch" style:background="#1e293b"></span>Fold
				</li>
			</ul>
			<p class="text-xs text-slate-400">
				Actions here: {duelCurrent[1].actions.join(', ')}. About
				<span class="font-semibold tabular-nums">{(duelCurrent[1].reach * 100).toFixed(0)}%</span> of
				hands reach this decision.
			</p>
			<p class="rounded-md bg-slate-800 px-2 py-1 text-xs text-slate-300">
				Approximate solve: preflop only, CFR+ on hand-class equities with a position-based
				realisation of equity after the flop (the small blind is in position). Not a full postflop
				solve. Exploitability of this model: {duel.exploitability_bb.toFixed(3)} bb.
			</p>
		{:else if problem}
			<p class="text-sm text-rose-300">{problem}</p>
		{:else}
			<p class="text-sm text-slate-400">Loading…</p>
		{/if}
	{:else}
		<label class="flex items-center gap-3 text-sm font-semibold">
			Stack
			<input
				class="min-w-0 flex-1 accent-amber-500"
				type="range"
				min="2"
				max="20"
				step="1"
				bind:value={stack}
				aria-label="Effective stack in big blinds"
			/>
			<span class="w-12 text-right tabular-nums">{stack}bb</span>
		</label>

		<div class="flex gap-1" role="tablist" aria-label="Who acts">
			<button
				class="seat-tab"
				class:selected={side === 'push'}
				role="tab"
				aria-selected={side === 'push'}
				onclick={() => (side = 'push')}
			>
				SB shoves
			</button>
			<button
				class="seat-tab"
				class:selected={side === 'call'}
				role="tab"
				aria-selected={side === 'call'}
				onclick={() => (side = 'call')}
			>
				BB calls
			</button>
		</div>

		{#if pushFold}
			<HandGrid
				hands={shown}
				{verb}
				mix={openMix}
				label={side === 'push' ? 'Small blind shoving range' : 'Big blind calling range'}
				selected={picked}
				onpick={(hand) => (picked = hand)}
			/>
			<div class="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 text-sm">
				<span>
					<span class="font-bold tabular-nums">
						{side === 'push' ? pushFold.push_pct : pushFold.call_pct}%
					</span>
					of hands
				</span>
				<span class="text-xs text-slate-400">
					Computed · off by ≤ {pushFold.exploitability_bb.toFixed(3)} bb per hand
				</span>
			</div>
			<p class="rounded-md bg-slate-800 px-2 py-1 text-xs text-slate-300">{pushFold.note}</p>
		{:else if problem}
			<p class="text-sm text-rose-300">{problem}</p>
		{:else}
			<p class="text-sm text-slate-400">Loading…</p>
		{/if}
	{/if}

	<!-- The hand being studied -->
	<section
		class="rounded-lg border border-slate-700 bg-slate-900/60 p-3 text-sm"
		aria-live="polite"
	>
		{#if picked}
			<div class="flex flex-wrap items-baseline gap-x-2">
				<h3 class="text-base font-bold">{handTitle(picked)}</h3>
				<span class="text-slate-400">{picked} · {handCombos(picked)} combos</span>
			</div>
			{#if percentile[picked] !== undefined}
				<p class="mt-1 text-slate-300">
					Top <span class="font-semibold tabular-nums">{percentile[picked].toFixed(1)}%</span> of
					starting hands{#if base?.equity_vs_random[picked]}, wins
						<span class="font-semibold tabular-nums">
							{(base.equity_vs_random[picked] * 100).toFixed(1)}%
						</span>
						against one random hand{/if}.
				</p>
			{/if}
			{#if mode === 'open' && seat && !situation?.kind}
				<p class="mt-1">
					From <span class="font-semibold">{seat.name}</span>:
					<span class={shown.has(picked) ? 'font-bold text-amber-400' : 'font-bold text-slate-400'}>
						{shown.has(picked) ? 'raise' : 'fold'}
					</span>
				</p>
				{#if raisesFrom.length > 0}
					<p class="mt-1 text-xs text-slate-400">
						Opens from {raisesFrom.map((s) => s.name).join(', ')}{foldsFrom.length > 0
							? `; folds from ${foldsFrom.map((s) => s.name).join(', ')}`
							: ''} at {players} players.
					</p>
				{:else}
					<p class="mt-1 text-xs text-slate-400">Never opened first-in at {players} players.</p>
				{/if}
			{:else if mode === 'open' && situation?.kind && spot}
				<p class="mt-1 flex flex-wrap gap-x-3 gap-y-1">
					<span class="text-slate-400">
						{spot.hero}
						{spot.kind === 'vs_open' ? 'vs open from' : 'vs 3-bet from'}
						{spot.villain}:
					</span>
					{#each pickedMix as part (part.label)}
						<span class="flex items-center gap-1">
							<span class="swatch" style:background={part.color}></span>
							<span class="font-bold tabular-nums">{pct(part.share)}</span>
							{part.label}
						</span>
					{/each}
				</p>
			{:else if mode === 'pushfold' && pushFold}
				<p class="mt-1">
					At {stack}bb: small blind
					<span class="font-bold">{pushFold.push.includes(picked) ? 'shoves' : 'folds'}</span>, big
					blind
					<span class="font-bold">{pushFold.call.includes(picked) ? 'calls' : 'folds'}</span>
					a shove.
				</p>
			{/if}
		{:else}
			<p class="text-slate-400">Click a hand to see how strong it is and what to do with it.</p>
		{/if}
	</section>

	{#if mode === 'open'}
		<p class="text-xs text-slate-500">
			Raise-first-in, about {chart?.stack_bb ?? 100}bb deep, {chart?.open_size_bb ?? 2.5}x open.
			Amber = raise, grey = fold. Robots at this table follow these charts.
		</p>
	{:else if mode === 'pushfold'}
		<p class="text-xs text-slate-500">
			Heads-up, chips only (no tournament pressure): the small blind shoves or folds, the big blind
			calls or folds. Robots use these when heads-up with 10bb or less.
		</p>
	{/if}
</div>

<style>
	.seat-tab {
		border: 1px solid #334155;
		border-radius: 0.4rem;
		background: #1e293b;
		padding: 0.2rem 0.55rem;
		font-size: 0.78rem;
		font-weight: 700;
		color: #cbd5e1;
	}

	.seat-tab.seat {
		display: flex;
		flex-direction: column;
		align-items: center;
		line-height: 1.15;
	}

	.pct {
		font-size: 0.62rem;
		font-weight: 600;
		color: #94a3b8;
	}

	.seat-tab.selected {
		border-color: var(--accent);
		background: #78350f;
		color: #fff7e6;
	}

	.seat-tab.selected .pct {
		color: #fcd9a0;
	}

	.seat-tab:disabled {
		opacity: 0.35;
	}

	.swatch {
		display: inline-block;
		width: 0.7rem;
		height: 0.7rem;
		border: 1px solid rgb(255 255 255 / 0.25);
		border-radius: 0.15rem;
	}
</style>

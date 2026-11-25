<!--
	Equity of the two ranges after every step of a hand: one line per player on one 0-100% axis.
	Hovering reads a step; clicking selects it. The step table next to it holds the same numbers.
-->

<script lang="ts">
	let {
		points,
		names,
		selected,
		onselect
	}: {
		/** One entry per state: the label of the step and both equities (null where there is none). */
		points: { label: string; equity: [number, number] | null }[];
		names: [string, string];
		selected: number;
		onselect: (index: number) => void;
	} = $props();

	const COLORS = ['#3987e5', '#d95926'];
	// Drawn at the width it has, so the text stays its size on a narrow screen.
	let width = $state(640);
	const W = $derived(Math.max(280, width));
	const H = 220;
	const L = 36;
	const R = 76;
	const T = 10;
	const B = 22;

	let hovered = $state<number | null>(null);

	const x = (i: number) => L + (points.length > 1 ? (i / (points.length - 1)) * (W - L - R) : 0);
	const y = (v: number) => T + (1 - v) * (H - T - B);
	const shown = $derived(points.flatMap((p, i) => (p.equity ? [{ i, equity: p.equity }] : [])));
	const path = (s: number) =>
		shown.map((p, k) => `${k ? 'L' : 'M'}${x(p.i)},${y(p.equity[s])}`).join(' ');
	const last = $derived(shown[shown.length - 1]);
	// The two end labels sit at their line ends, pushed apart when the lines meet.
	const labelY = $derived.by(() => {
		if (!last) return [0, 0];
		const at = [y(last.equity[0]), y(last.equity[1])];
		if (Math.abs(at[0] - at[1]) < 14) {
			const mid = (at[0] + at[1]) / 2;
			const up = at[0] <= at[1] ? 0 : 1;
			at[up] = mid - 7;
			at[1 - up] = mid + 7;
		}
		return at;
	});
	const read = $derived(points[hovered ?? selected]);
	const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

	function nearest(event: MouseEvent): number {
		const box = (event.currentTarget as SVGElement).getBoundingClientRect();
		const px = ((event.clientX - box.left) / box.width) * W;
		const step = points.length > 1 ? (W - L - R) / (points.length - 1) : 1;
		return Math.max(0, Math.min(points.length - 1, Math.round((px - L) / step)));
	}
</script>

<figure class="m-0" data-testid="equity-steps" bind:clientWidth={width}>
	<div class="mb-1 flex flex-wrap items-center gap-x-4 text-xs text-slate-300">
		{#each names as name, s (s)}
			<span><span class="key" style:background={COLORS[s]}></span>{name}</span>
		{/each}
		{#if read}
			<span class="text-slate-400 tabular-nums">
				{read.label}:
				{#if read.equity}
					{names[0]} {pct(read.equity[0])}, {names[1]} {pct(read.equity[1])}
				{:else}
					no equity yet
				{/if}
			</span>
		{/if}
	</div>
	<!-- The step table below gives the keyboard route to the same selection. -->
	<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
	<svg
		viewBox="0 0 {W} {H}"
		class="w-full cursor-pointer"
		role="img"
		aria-label="Equity of both ranges after each step"
		onmousemove={(e) => (hovered = nearest(e))}
		onmouseleave={() => (hovered = null)}
		onclick={(e) => onselect(nearest(e))}
	>
		{#each [0, 0.25, 0.5, 0.75, 1] as v (v)}
			<line x1={L} x2={W - R} y1={y(v)} y2={y(v)} class={v === 0.5 ? 'mid' : 'grid'} />
			<text x={L - 6} y={y(v) + 4} text-anchor="end">{v * 100}%</text>
		{/each}
		{#each points.keys() as i (i)}
			<text x={x(i)} y={H - 6} text-anchor="middle">{i}</text>
		{/each}
		<line x1={x(selected)} x2={x(selected)} y1={T} y2={H - B} class="chosen" />
		{#if hovered !== null}
			<line x1={x(hovered)} x2={x(hovered)} y1={T} y2={H - B} class="cross" />
		{/if}
		{#each [0, 1] as s (s)}
			<path d={path(s)} fill="none" stroke={COLORS[s]} stroke-width="2" />
			{#each shown as p (p.i)}
				<circle cx={x(p.i)} cy={y(p.equity[s])} r="4" fill={COLORS[s]} class="dot" />
			{/each}
			{#if last}
				<text x={x(last.i) + 9} y={labelY[s] + 4} class="end">{names[s]} {pct(last.equity[s])}</text
				>
			{/if}
		{/each}
	</svg>
</figure>

<style>
	.key {
		display: inline-block;
		width: 0.9rem;
		height: 0.2rem;
		margin-right: 0.35rem;
		vertical-align: middle;
	}

	text {
		fill: #94a3b8;
		font-size: 11px;
	}

	text.end {
		fill: #e2e8f0;
	}

	.grid {
		stroke: #1f2a44;
	}

	.mid {
		stroke: #3b4a6b;
	}

	.cross {
		stroke: #64748b;
		stroke-dasharray: 3 3;
	}

	.chosen {
		stroke: var(--accent);
		opacity: 0.6;
	}

	.dot {
		stroke: #111a2e;
		stroke-width: 2;
	}
</style>

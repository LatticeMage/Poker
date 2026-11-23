<!--
	Equity distribution: an XY chart with one line per player. The x axis is the percentile of the
	range (its weakest hands on the left, its strongest on the right); the y axis is the equity those
	hands have against the other player's range. Where one line sits above the other is where that
	player is ahead; a steep right end means a strong top of the range (many nuts), a high left end a
	range with few weak hands. Every hand class is a small dot on its line; an arrow points at your own hand when you know it.
-->

<script lang="ts">
	import type { CurveHand } from '$lib/types/ranges';

	let {
		mine,
		theirs,
		mineHands = [],
		theirHands = [],
		hand = null,
		mineLabel = 'You',
		theirLabel = 'Them'
	}: {
		mine: number[];
		theirs: number[];
		/** Every hand class of each range (one dot each); optional. */
		mineHands?: CurveHand[];
		theirHands?: CurveHand[];
		/** Where your hand sits: percentile (0 weakest .. 1 strongest), its equity and its name. */
		hand?: { percentile: number; equity: number; name?: string } | null;
		mineLabel?: string;
		theirLabel?: string;
	} = $props();

	const W = 440;
	const H = 290;
	const LEFT = 44;
	const RIGHT = 12;
	const TOP = 14;
	const BOTTOM = 42;

	const x = (p: number) => LEFT + p * (W - LEFT - RIGHT);
	const y = (e: number) => TOP + (1 - e) * (H - TOP - BOTTOM);

	/** The curve, extended to both edges so it spans the whole axis. */
	function path(values: number[]): string {
		if (values.length === 0) return '';
		const points: [number, number][] = [
			[0, values[0]],
			...values.map((e, i): [number, number] => [(i + 0.5) / values.length, e]),
			[1, values[values.length - 1]]
		];
		return points
			.map(([p, e], i) => `${i === 0 ? 'M' : 'L'}${x(p).toFixed(1)},${y(e).toFixed(1)}`)
			.join(' ');
	}

	const average = (values: number[]) =>
		values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
	type Dot = CurveHand & { side: 'you' | 'them'; sideLabel: string; key: string };
	const dots = $derived<Dot[]>([
		...theirHands.map((h) => ({
			...h,
			side: 'them' as const,
			sideLabel: theirLabel,
			key: `t-${h.hand}`
		})),
		...mineHands.map((h) => ({
			...h,
			side: 'you' as const,
			sideLabel: mineLabel,
			key: `y-${h.hand}`
		}))
	]);
	const describe = (d: Dot) =>
		`${d.hand}, ${d.sideLabel}: ${pct(d.equity)} equity, ${(d.weight * 100).toFixed(1)}% of the range`;

	/** The dot whose tooltip is open (hover, keyboard focus or tap). */
	let active = $state<string | null>(null);
	let pinned = $state<string | null>(null);
	const shown = $derived(dots.find((d) => d.key === (active ?? pinned)) ?? null);
	const tip = $derived.by(() => {
		if (!shown) return null;
		const lines = [
			`${shown.hand} · ${shown.sideLabel}`,
			`${(shown.equity * 100).toFixed(1)}% equity · ${(shown.weight * 100).toFixed(1)}% of range`
		];
		const w = Math.max(...lines.map((l) => l.length)) * 5.6 + 14;
		const h = 34;
		const cx = x(shown.percentile);
		const cy = y(shown.equity);
		const tx = Math.min(Math.max(cx - w / 2, 2), W - w - 2);
		const ty = cy - h - 9 < 2 ? cy + 10 : cy - h - 9;
		return { lines, w, h, tx, ty };
	});
	function toggle(key: string) {
		pinned = pinned === key ? null : key;
	}

	// The arrow comes from below the point unless there is no room, then from above.
	const arrow = $derived.by(() => {
		if (!hand) return null;
		const px = x(hand.percentile);
		const py = y(hand.equity);
		const below = py < H - BOTTOM - 46;
		const dir = below ? 1 : -1;
		const tipY = py + dir * 6;
		const tailY = py + dir * 34;
		return {
			px,
			tipY,
			tailY,
			labelY: tailY + dir * (below ? 11 : 4),
			anchor: px > W - 70 ? 'end' : px < LEFT + 30 ? 'start' : 'middle',
			d: `M${px},${tailY} L${px},${tipY + dir * 2} M${px - 5},${tipY + dir * 8} L${px},${tipY} L${px + 5},${tipY + dir * 8}`
		};
	});
	const ticks = [0, 0.25, 0.5, 0.75, 1];
	const pct = (v: number) => `${Math.round(v * 100)}%`;
</script>

<figure class="m-0">
	<svg
		viewBox={`0 0 ${W} ${H}`}
		class="h-auto w-full"
		role="img"
		aria-label={`Equity distribution. ${mineLabel}: average ${pct(average(mine))}. ${theirLabel}: average ${pct(average(theirs))}.`}
	>
		<!-- grid and axes -->
		{#each ticks as t (t)}
			<line
				x1={LEFT}
				x2={W - RIGHT}
				y1={y(t)}
				y2={y(t)}
				stroke="#334155"
				stroke-width="1"
				opacity="0.6"
			/>
			<line
				x1={x(t)}
				x2={x(t)}
				y1={TOP}
				y2={H - BOTTOM}
				stroke="#334155"
				stroke-width="1"
				opacity="0.35"
			/>
			<text x={LEFT - 6} y={y(t) + 3} text-anchor="end" font-size="10" fill="#94a3b8">{pct(t)}</text
			>
			<text x={x(t)} y={H - BOTTOM + 14} text-anchor="middle" font-size="10" fill="#94a3b8"
				>{pct(t)}</text
			>
		{/each}
		<line
			x1={LEFT}
			x2={W - RIGHT}
			y1={y(0.5)}
			y2={y(0.5)}
			stroke="#94a3b8"
			stroke-width="1"
			stroke-dasharray="4 4"
		/>
		<text x={(LEFT + W - RIGHT) / 2} y={H - 6} text-anchor="middle" font-size="10.5" fill="#cbd5e1">
			Percentile of the range (weakest → strongest)
		</text>
		<text
			transform={`translate(11 ${(TOP + H - BOTTOM) / 2}) rotate(-90)`}
			text-anchor="middle"
			font-size="10.5"
			fill="#cbd5e1"
		>
			Equity
		</text>

		<!-- the two distributions -->
		<path
			d={path(theirs)}
			fill="none"
			stroke="#38bdf8"
			opacity="0.55"
			stroke-width="2"
			stroke-linejoin="round"
			data-line="them"
		/>
		<path
			d={path(mine)}
			fill="none"
			stroke="#f59e0b"
			opacity="0.55"
			stroke-width="2"
			stroke-linejoin="round"
			data-line="you"
		/>

		<!-- every hand class, one dot on its own line -->
		<g role="group" aria-label="Hands in each range">
			{#each dots as d (d.key)}
				<circle
					cx={x(d.percentile)}
					cy={y(d.equity)}
					r={shown?.key === d.key ? 3.6 : 2}
					fill={d.side === 'you' ? '#f59e0b' : '#38bdf8'}
					stroke="#0f172a"
					stroke-width="0.6"
					tabindex="0"
					role="button"
					aria-label={describe(d)}
					data-dot={d.side}
					data-hand={d.hand}
					class="cursor-pointer outline-none"
					onmouseenter={() => (active = d.key)}
					onmouseleave={() => (active = null)}
					onfocus={() => (active = d.key)}
					onblur={() => (active = null)}
					onclick={() => toggle(d.key)}
					onkeydown={(e) => {
						if (e.key === 'Enter' || e.key === ' ') {
							e.preventDefault();
							toggle(d.key);
						} else if (e.key === 'Escape') pinned = null;
					}}
				>
					<title>{describe(d)}</title>
				</circle>
			{/each}
		</g>

		{#if tip && shown}
			<g pointer-events="none" aria-hidden="true" data-testid="dot-tip">
				<rect
					x={tip.tx}
					y={tip.ty}
					width={tip.w}
					height={tip.h}
					rx="4"
					fill="#0f172a"
					stroke={shown.side === 'you' ? '#f59e0b' : '#38bdf8'}
				/>
				<text x={tip.tx + 7} y={tip.ty + 14} font-size="10.5" font-weight="700" fill="#f8fafc"
					>{tip.lines[0]}</text
				>
				<text x={tip.tx + 7} y={tip.ty + 27} font-size="10" fill="#cbd5e1">{tip.lines[1]}</text>
			</g>
		{/if}

		<!-- your own hand: an arrow pointing at it -->
		{#if hand && arrow}
			<g
				data-marker="hand"
				role="img"
				aria-label={`Your hand${hand.name ? ` ${hand.name}` : ''}: ${pct(hand.equity)} equity, stronger than ${pct(hand.percentile)} of your range`}
			>
				<title
					>Your hand{hand.name ? ` ${hand.name}` : ''}: {pct(hand.equity)} equity, stronger than {pct(
						hand.percentile
					)} of your range</title
				>
				<path
					d={arrow.d}
					fill="none"
					stroke="#f8fafc"
					stroke-width="2.5"
					stroke-linecap="round"
					stroke-linejoin="round"
				/>
				<text
					x={arrow.px}
					y={arrow.labelY}
					text-anchor={arrow.anchor}
					font-size="11"
					font-weight="700"
					fill="#f8fafc">{hand.name ?? 'Your hand'}</text
				>
			</g>
		{/if}

		<!-- legend -->
		<g font-size="11" fill="#e2e8f0">
			<line
				x1={LEFT + 10}
				x2={LEFT + 30}
				y1={TOP + 14}
				y2={TOP + 14}
				stroke="#f59e0b"
				stroke-width="3"
			/>
			<text x={LEFT + 36} y={TOP + 18}>{mineLabel} · avg {pct(average(mine))}</text>
			<line
				x1={LEFT + 10}
				x2={LEFT + 30}
				y1={TOP + 30}
				y2={TOP + 30}
				stroke="#38bdf8"
				stroke-width="3"
			/>
			<text x={LEFT + 36} y={TOP + 34}>{theirLabel} · avg {pct(average(theirs))}</text>
			{#if hand}
				<path
					d={`M${LEFT + 20},${TOP + 53} L${LEFT + 20},${TOP + 41} M${LEFT + 16},${TOP + 45} L${LEFT + 20},${TOP + 41} L${LEFT + 24},${TOP + 45}`}
					fill="none"
					stroke="#f8fafc"
					stroke-width="2"
					stroke-linecap="round"
					stroke-linejoin="round"
				/>
				<text x={LEFT + 36} y={TOP + 50}
					>{hand.name ? `${hand.name} · ` : 'Your hand · '}{pct(hand.equity)}</text
				>
			{/if}
		</g>
	</svg>
</figure>

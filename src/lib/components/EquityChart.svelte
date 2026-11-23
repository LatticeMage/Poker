<!--
	Equity distribution: an XY chart with one line per player. The x axis is the percentile of the
	range (its weakest hands on the left, its strongest on the right); the y axis is the equity those
	hands have against the other player's range. Where one line sits above the other is where that
	player is ahead; a steep right end means a strong top of the range (many nuts), a high left end a
	range with few weak hands. A dot marks your own hand when you know it.
-->

<script lang="ts">
	let {
		mine,
		theirs,
		hand = null,
		mineLabel = 'You',
		theirLabel = 'Them'
	}: {
		mine: number[];
		theirs: number[];
		/** Where your hand sits: percentile (0 weakest .. 1 strongest) and its equity. */
		hand?: { percentile: number; equity: number } | null;
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
			stroke-width="2.5"
			stroke-linejoin="round"
			data-line="them"
		/>
		<path
			d={path(mine)}
			fill="none"
			stroke="#f59e0b"
			stroke-width="2.5"
			stroke-linejoin="round"
			data-line="you"
		/>

		<!-- your own hand -->
		{#if hand}
			<circle
				cx={x(hand.percentile)}
				cy={y(hand.equity)}
				r="5.5"
				fill="#f8fafc"
				stroke="#f59e0b"
				stroke-width="2.5"
				data-marker="hand"
			>
				<title
					>Your hand: {pct(hand.equity)} equity, stronger than {pct(hand.percentile)} of your range</title
				>
			</circle>
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
				<circle
					cx={LEFT + 20}
					cy={TOP + 46}
					r="4.5"
					fill="#f8fafc"
					stroke="#f59e0b"
					stroke-width="2"
				/>
				<text x={LEFT + 36} y={TOP + 50}>Your hand · {pct(hand.equity)}</text>
			{/if}
		</g>
	</svg>
</figure>

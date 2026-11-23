<!--
	The 13×13 starting-hand grid: pairs on the diagonal, suited hands above it, offsuit below.
	`hands` are the highlighted ones; `selected` is the hand being studied; `marked` is a hand
	to call out (the quiz question).
-->

<script lang="ts">
	import { INDEXES, handAt } from '$lib/utils/hands';

	let {
		hands,
		verb,
		label,
		selected = null,
		marked = null,
		mix,
		heat,
		onpick
	}: {
		hands: Set<string>;
		/** What highlighted hands do, for tooltips: "raise", "shove", "call". */
		verb: string;
		label: string;
		selected?: string | null;
		marked?: string | null;
		/** Per hand: shares of its combos that [raise for value, raise as a bluff, call]; the rest folds. */
		mix?: Record<string, [number, number, number]>;
		/** Per hand: a value from 0 to 1 (equity), drawn as a colour from red through grey to green. */
		heat?: Record<string, number>;
		onpick?: (hand: string) => void;
	} = $props();

	const COLORS = ['#d97706', '#f43f5e', '#0284c7'];

	function background(hand: string): string | undefined {
		const shares = mix?.[hand];
		if (!shares) return undefined;
		let at = 0;
		const stops: string[] = [];
		shares.forEach((share, i) => {
			if (share <= 0) return;
			stops.push(`${COLORS[i]} ${at * 100}% ${(at + share) * 100}%`);
			at += share;
		});
		return stops.length
			? `linear-gradient(90deg, ${stops.join(', ')}, #1e293b ${at * 100}%)`
			: undefined;
	}

	// Red below a coin flip, grey around it, green above.
	const STOPS: [number, [number, number, number]][] = [
		[0.2, [185, 28, 28]],
		[0.5, [51, 65, 85]],
		[0.85, [21, 128, 61]]
	];

	function heatColor(hand: string): string | undefined {
		const value = heat?.[hand];
		if (value === undefined) return undefined;
		const v = Math.min(STOPS[2][0], Math.max(STOPS[0][0], value));
		const [[a, ca], [b, cb]] = v <= STOPS[1][0] ? [STOPS[0], STOPS[1]] : [STOPS[1], STOPS[2]];
		const t = (v - a) / (b - a);
		return `rgb(${ca.map((c, i) => Math.round(c + (cb[i] - c) * t)).join(',')})`;
	}

	function mixTitle(hand: string): string {
		const [value, bluff, call] = mix?.[hand] ?? [0, 0, 0];
		const parts = [
			value > 0 && `raise (value) ${Math.round(value * 100)}%`,
			bluff > 0 && `raise (bluff) ${Math.round(bluff * 100)}%`,
			call > 0 && `call ${Math.round(call * 100)}%`
		].filter(Boolean);
		const fold = Math.max(0, 1 - value - bluff - call);
		return `${hand} — ${parts.length ? parts.join(', ') : 'fold'}${parts.length && fold > 0.005 ? `, fold ${Math.round(fold * 100)}%` : ''}`;
	}
</script>

<div class="grid grid-cols-13 gap-px" role="group" aria-label={label}>
	{#each INDEXES as row (row)}
		{#each INDEXES as col (col)}
			{@const hand = handAt(row, col)}
			<button
				type="button"
				class="cell"
				class:on={!mix && !heat && hands.has(hand)}
				class:heated={!!heat && heat[hand] !== undefined}
				class:mixed={!!mix && !!mix[hand]}
				class:pair={row === col}
				class:selected={selected === hand}
				class:marked={marked === hand}
				style:background={heat ? heatColor(hand) : background(hand)}
				title={heat
					? heat[hand] === undefined
						? `${hand} — not in the range`
						: `${hand} — ${Math.round(heat[hand] * 100)}% equity`
					: mix
						? mixTitle(hand)
						: `${hand}${hands.has(hand) ? ` — ${verb}` : ' — fold'}`}
				tabindex={(selected ?? 'AA') === hand ? 0 : -1}
				onclick={() => onpick?.(hand)}
			>
				{hand}
			</button>
		{/each}
	{/each}
</div>

<style>
	.cell {
		display: grid;
		place-items: center;
		aspect-ratio: 1;
		border-radius: 0.15rem;
		background: #1e293b;
		color: #64748b;
		font-size: clamp(0.5rem, 2vw, 0.72rem);
		font-weight: 600;
		letter-spacing: -0.04em;
	}

	.cell.pair {
		background: #243247;
	}

	.cell.on {
		background: #d97706;
		color: #fff7e6;
	}

	.cell.heated {
		color: #f8fafc;
		text-shadow: 0 0 2px rgb(0 0 0 / 0.7);
	}

	.cell.mixed {
		color: #fff7e6;
		text-shadow: 0 0 2px rgb(0 0 0 / 0.7);
	}

	.cell.on.pair {
		background: #b45309;
	}

	.cell:hover {
		outline: 1px solid #94a3b8;
	}

	.cell.selected {
		outline: 2px solid #f8fafc;
		outline-offset: -1px;
		z-index: 1;
	}

	.cell.marked {
		outline: 3px solid #38bdf8;
		outline-offset: -2px;
		z-index: 1;
	}
</style>

<!-- Settings form. Speed and auto-deal apply at once; table size and stacks to the next new game. -->

<script lang="ts">
	import { LIMITS, settings } from '$lib/stores/settings.svelte';
	import type { GameState } from '$lib/types/game';

	let { game, onnewgame }: { game: GameState | null; onnewgame: () => void } = $props();

	const speedLabel = $derived(
		settings.speed === 0
			? 'Instant'
			: settings.speed <= 300
				? 'Fast'
				: settings.speed <= 800
					? 'Normal'
					: 'Slow'
	);
</script>

<div class="flex flex-col gap-5">
	<label class="flex flex-col gap-1.5">
		<span class="flex items-baseline justify-between text-sm font-semibold">
			Robot speed <span class="font-normal text-slate-400">{speedLabel}</span>
		</span>
		<input
			class="accent-amber-500"
			type="range"
			min={LIMITS.speed.min}
			max={LIMITS.speed.max}
			step={LIMITS.speed.step}
			bind:value={settings.speed}
		/>
		<span class="flex justify-between text-xs text-slate-400"
			><span>Instant</span><span>Slow</span></span
		>
	</label>

	<label class="flex items-center justify-between gap-3 text-sm font-semibold">
		Deal the next hand automatically
		<input class="h-5 w-5 accent-amber-500" type="checkbox" bind:checked={settings.autoDeal} />
	</label>

	<label class="flex items-center justify-between gap-3 text-sm font-semibold">
		<span>
			Show chart hints while playing
			<span class="block text-xs font-normal text-slate-400">
				Before the flop, says whether your hand is in the chart for your seat.
			</span>
		</span>
		<input class="h-5 w-5 accent-amber-500" type="checkbox" bind:checked={settings.hints} />
	</label>

	<hr class="border-slate-700" />

	<label class="flex flex-col gap-1.5">
		<span class="flex items-baseline justify-between text-sm font-semibold">
			Players <span class="tabular-nums">{settings.players}</span>
		</span>
		<input
			class="accent-amber-500"
			type="range"
			min={LIMITS.players.min}
			max={LIMITS.players.max}
			step={LIMITS.players.step}
			bind:value={settings.players}
		/>
		<span class="text-xs text-slate-400">You plus {settings.players - 1} robots.</span>
	</label>

	<label class="flex flex-col gap-1.5">
		<span class="flex items-baseline justify-between text-sm font-semibold">
			Starting chips
			<span class="tabular-nums">{settings.chips.toLocaleString('en-US')}</span>
		</span>
		<input
			class="accent-amber-500"
			type="range"
			min={LIMITS.chips.min}
			max={LIMITS.chips.max}
			step={LIMITS.chips.step}
			bind:value={settings.chips}
		/>
		<span class="text-xs text-slate-400">
			Blinds start at 10/20{game ? ` and rise every ${game.hands_per_level} hands` : ''}.
		</span>
	</label>

	<div class="flex items-center justify-between gap-3">
		<p class="text-xs text-slate-400">Players and chips apply to the next new game.</p>
		<button
			class="shrink-0 rounded-lg bg-amber-600 px-3 py-2 text-sm font-bold text-amber-950 hover:bg-amber-500"
			onclick={onnewgame}
		>
			Start new game
		</button>
	</div>
</div>

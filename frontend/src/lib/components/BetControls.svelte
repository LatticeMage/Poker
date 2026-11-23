<!--
	The human's turn: fold/check, call, and a sized bet or raise.
	Mount it keyed on `legal` so the amount resets for every decision.
	Shortcuts: F fold · C check/call · R bet/raise.
-->

<script lang="ts">
	import type { Action, Legal } from '$lib/types/game';

	let {
		legal,
		smallBlind,
		onact
	}: { legal: Legal; smallBlind: number; onact: (action: Action) => void } = $props();

	const fmt = (n: number) => n.toLocaleString('en-US');

	/** Selectable "raise to" amounts: the minimum, tidy steps, then the maximum. */
	const marks = $derived.by(() => {
		if (!legal.can_raise) return [0];
		const step = Math.max(1, smallBlind);
		const out = [legal.min_to];
		for (let v = Math.floor(legal.min_to / step) * step + step; v < legal.max_to; v += step) {
			out.push(v);
		}
		if (legal.max_to > legal.min_to) out.push(legal.max_to);
		return out;
	});

	let index = $state(0);
	const amount = $derived(marks[Math.min(index, marks.length - 1)]);
	const allIn = $derived(amount >= legal.stack_to);
	const raiseLabel = $derived(allIn ? 'All-in' : legal.opening ? 'Bet' : 'Raise to');

	function pick(target: number) {
		// Nearest selectable amount.
		let best = 0;
		for (let i = 1; i < marks.length; i++) {
			if (Math.abs(marks[i] - target) < Math.abs(marks[best] - target)) best = i;
		}
		index = best;
	}

	/** Bet a fraction of the pot (after calling, when facing a bet). */
	const potFraction = (fraction: number) =>
		pick(legal.current_bet + fraction * (legal.pot + legal.to_call));

	const presets = $derived([
		{ label: 'Min', run: () => (index = 0) },
		{ label: '½ pot', run: () => potFraction(0.5) },
		{ label: 'Pot', run: () => potFraction(1) },
		{ label: 'Max', run: () => (index = marks.length - 1) }
	]);

	const fold = () => onact({ type: 'fold' });
	const checkOrCall = () => onact(legal.can_check ? { type: 'check' } : { type: 'call' });
	const raise = () => legal.can_raise && onact({ type: 'raise', amount });

	function onkeydown(event: KeyboardEvent) {
		if (event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
		if (document.querySelector('dialog[open]')) return;
		const el = event.target as HTMLElement | null;
		if (el?.tagName === 'INPUT' && (el as HTMLInputElement).type !== 'range') return;
		const key = event.key.toLowerCase();
		if (key === 'f' && !legal.can_check) fold();
		else if (key === 'c') checkOrCall();
		else if (key === 'r') raise();
		else return;
		event.preventDefault();
	}
</script>

<svelte:window {onkeydown} />

<div class="flex w-full max-w-2xl flex-col gap-2">
	<div class="grid grid-cols-3 gap-2">
		<button
			class="act bg-rose-700 text-white hover:bg-rose-600"
			disabled={legal.can_check}
			onclick={fold}
			title="Shortcut: F"
		>
			Fold
		</button>
		<button
			class="act bg-sky-700 text-white hover:bg-sky-600"
			onclick={checkOrCall}
			title="Shortcut: C"
		>
			{#if legal.can_check}
				Check
			{:else}
				Call <span class="tabular-nums">{fmt(legal.to_call)}</span>
				{#if legal.to_call >= legal.chips}<span class="text-xs font-semibold">all-in</span>{/if}
			{/if}
		</button>
		<button
			class="act bg-amber-600 text-amber-950 hover:bg-amber-500"
			disabled={!legal.can_raise}
			onclick={raise}
			title="Shortcut: R"
		>
			{#if legal.can_raise}
				{raiseLabel} <span class="tabular-nums">{fmt(amount)}</span>
			{:else}
				Raise
			{/if}
		</button>
	</div>

	<div class="flex items-center gap-2" class:invisible={!legal.can_raise}>
		<input
			class="min-w-0 flex-1 accent-amber-500"
			type="range"
			min="0"
			max={marks.length - 1}
			step="1"
			bind:value={index}
			disabled={!legal.can_raise || marks.length < 2}
			aria-label="Bet amount"
			aria-valuetext={fmt(amount)}
		/>
		<div class="flex gap-1">
			{#each presets as preset (preset.label)}
				<button
					class="rounded-md border border-slate-600 bg-slate-800 px-2 py-1 text-xs font-semibold text-slate-200 hover:bg-slate-700"
					onclick={preset.run}
					disabled={!legal.can_raise}
				>
					{preset.label}
				</button>
			{/each}
		</div>
	</div>
</div>

<style>
	.act {
		min-height: 2.9rem;
		border-radius: 0.6rem;
		padding: 0.4rem 0.5rem;
		font-size: 1rem;
		font-weight: 800;
		box-shadow: 0 2px 0 rgb(0 0 0 / 0.35);
		transition:
			background-color 120ms ease,
			transform 80ms ease;
	}

	.act:active:not(:disabled) {
		transform: translateY(1px);
	}

	.act:disabled {
		opacity: 0.35;
	}
</style>

<!--
	Practice mode: choose your hole cards and the community cards for upcoming
	deals. The highlighted slot receives the next card you click; ← → move it.
-->

<script lang="ts">
	import { settings } from '$lib/stores/settings.svelte';
	import {
		type Card,
		SUIT_SYMBOLS,
		cardName,
		isRed,
		makeCard,
		rankLabel,
		rankOf,
		suitOf
	} from '$lib/utils/cards';

	// Slots 0–1 are your hole cards, 2–6 the board (flop, turn, river).
	const SLOTS = 7;
	const RANKS = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2];
	const BOARD_LABELS = ['Flop', 'Flop', 'Flop', 'Turn', 'River'];

	let cursor = $state(0);

	const slots = $derived([...settings.pickedHole, ...settings.pickedBoard]);
	const used = $derived(new Set(slots.filter((c) => c !== null)));

	const put = (slot: number, card: Card | null) => settings.setPick(slot, card);

	function choose(card: Card | null) {
		put(cursor, card);
		cursor = (cursor + 1) % SLOTS;
	}

	function clear(from: number, to: number) {
		for (let slot = from; slot < to; slot++) put(slot, null);
		cursor = from;
	}

	function onkeydown(event: KeyboardEvent) {
		if (event.key === 'ArrowRight') cursor = (cursor + 1) % SLOTS;
		else if (event.key === 'ArrowLeft') cursor = (cursor + SLOTS - 1) % SLOTS;
		else return;
		event.preventDefault();
	}
</script>

<svelte:window {onkeydown} />

{#snippet slot(index: number, label: string)}
	{@const card = slots[index]}
	<button
		class="slot"
		class:current={cursor === index}
		class:red={card !== null && isRed(card)}
		class:empty={card === null}
		onclick={() => (cursor = index)}
		aria-label={`${label}: ${card === null ? 'random' : cardName(card)}`}
		aria-pressed={cursor === index}
	>
		{#if card === null}
			<span class="text-[0.6rem] font-semibold tracking-wide uppercase">Any</span>
		{:else}
			<span>{rankLabel(rankOf(card))}</span><span>{SUIT_SYMBOLS[suitOf(card)]}</span>
		{/if}
	</button>
{/snippet}

<div class="flex flex-col gap-4">
	<p class="text-sm text-slate-300">
		Pick cards for the next deals — handy for practising a spot. Empty slots are dealt at random.
		Picks start with the next hand and stay until you clear them.
	</p>

	<div class="flex flex-wrap items-end gap-x-6 gap-y-3">
		<div>
			<div
				class="mb-1 flex items-center justify-between gap-3 text-xs font-semibold text-slate-400"
			>
				Your cards
				<button class="link" onclick={() => clear(0, 2)}>Clear</button>
			</div>
			<div class="flex gap-1.5">
				{@render slot(0, 'Your first card')}
				{@render slot(1, 'Your second card')}
			</div>
		</div>
		<div>
			<div
				class="mb-1 flex items-center justify-between gap-3 text-xs font-semibold text-slate-400"
			>
				Board
				<button class="link" onclick={() => clear(2, SLOTS)}>Clear</button>
			</div>
			<div class="flex gap-1.5">
				{#each BOARD_LABELS as label, i (i)}
					{@render slot(i + 2, `${label} card`)}
				{/each}
			</div>
		</div>
	</div>

	<div>
		<div class="mb-1.5 flex items-center justify-between text-xs font-semibold text-slate-400">
			Deck
			<button class="link" onclick={() => choose(null)}>Set highlighted slot to random</button>
		</div>
		<div class="grid gap-1">
			{#each [0, 1, 2, 3] as suit (suit)}
				<div class="grid grid-cols-13 gap-1">
					{#each RANKS as rank (rank)}
						{@const card = makeCard(rank, suit)}
						<button
							class="deck-card"
							class:red={isRed(card)}
							disabled={used.has(card)}
							onclick={() => choose(card)}
							aria-label={cardName(card)}
						>
							{rankLabel(rank)}<span class="text-[0.85em]">{SUIT_SYMBOLS[suit]}</span>
						</button>
					{/each}
				</div>
			{/each}
		</div>
	</div>
</div>

<style>
	.slot {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		width: 2.6rem;
		height: 3.6rem;
		border: 2px solid transparent;
		border-radius: 0.4rem;
		background: #fdfdfb;
		color: #161b26;
		font-size: 1.05rem;
		font-weight: 800;
		line-height: 1.1;
	}

	.slot.empty {
		border: 2px dashed #475569;
		background: #1e293b;
		color: #94a3b8;
	}

	.slot.current {
		border: 2px solid var(--accent);
		box-shadow: 0 0 0 3px rgb(245 185 66 / 0.35);
	}

	.red {
		color: #cf2337;
	}

	.deck-card {
		display: flex;
		align-items: center;
		justify-content: center;
		aspect-ratio: 3 / 4;
		border-radius: 0.3rem;
		background: #fdfdfb;
		color: #161b26;
		font-size: clamp(0.7rem, 2.6vw, 0.95rem);
		font-weight: 800;
		letter-spacing: -0.04em;
	}

	.deck-card.red {
		color: #cf2337;
	}

	.deck-card:hover:not(:disabled) {
		outline: 2px solid var(--accent);
	}

	.deck-card:disabled {
		background: #334155;
		color: #64748b;
	}

	.link {
		color: #fbbf24;
		font-weight: 600;
	}

	.link:hover {
		text-decoration: underline;
	}
</style>

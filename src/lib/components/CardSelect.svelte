<!--
	Pick a few cards: a row of slots and a deck to click. The highlighted slot takes the next card
	and the highlight moves on. `blocked` cards (used elsewhere) cannot be picked.
-->

<script lang="ts">
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

	let {
		cards = $bindable(),
		labels,
		name,
		blocked = new Set<Card>(),
		compact = false,
		groups
	}: {
		cards: (Card | null)[];
		/** Names this picker's deck for assistive tech and tests: "Flop", "Hand". */
		name: string;
		/** One label per slot, e.g. ["Flop 1", "Flop 2", "Flop 3"]. */
		labels: string[];
		blocked?: Set<Card>;
		compact?: boolean;
		/** Optional titled groups of slots shown side by side, e.g. Board (0-4) and Hand (5-6). */
		groups?: { title: string; from: number; to: number }[];
	} = $props();
	const shown = $derived(groups ?? [{ title: '', from: 0, to: cards.length }]);

	const RANKS = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2];
	let cursor = $state(0);

	const mine = $derived(new Set(cards.filter((c): c is Card => c !== null)));

	function put(card: Card) {
		cards[cursor] = card;
		// Move to the next empty slot, else the next slot.
		const empty = cards.findIndex((c, i) => i !== cursor && c === null);
		cursor = empty >= 0 ? empty : (cursor + 1) % cards.length;
	}

	function clearSlot(index: number) {
		cards[index] = null;
		cursor = index;
	}
</script>

<div class="flex flex-col gap-2">
	<div class="flex flex-wrap items-end gap-x-4 gap-y-2">
		{#each shown as group (group.title)}
			<div class="flex flex-col gap-1" role="group" aria-label={group.title || undefined}>
				{#if group.title}
					<span class="text-xs font-semibold text-slate-400">{group.title}</span>
				{/if}
				<div class="flex flex-wrap items-center gap-2">
					{#each cards.slice(group.from, group.to) as card, j (group.from + j)}
						{@const i = group.from + j}
						<button
							type="button"
							class="slot"
							class:current={cursor === i}
							class:red={card !== null && isRed(card)}
							class:empty={card === null}
							aria-label={`${labels[i]}: ${card === null ? 'empty' : cardName(card)}`}
							aria-pressed={cursor === i}
							onclick={() => (cursor === i && card !== null ? clearSlot(i) : (cursor = i))}
							title={card === null ? `Choose ${labels[i]}` : 'Click again to clear'}
						>
							{#if card === null}
								<span class="text-[0.6rem] font-semibold tracking-wide uppercase">{labels[i]}</span>
							{:else}
								<span>{rankLabel(rankOf(card))}</span><span>{SUIT_SYMBOLS[suitOf(card)]}</span>
							{/if}
						</button>
					{/each}
				</div>
			</div>
		{/each}
	</div>

	<div class="grid gap-0.5" role="group" aria-label={`${name} deck`}>
		{#each [0, 1, 2, 3] as suit (suit)}
			<div class={compact ? 'grid grid-cols-13 gap-0.5' : 'grid grid-cols-13 gap-1'}>
				{#each RANKS as rank (rank)}
					{@const card = makeCard(rank, suit)}
					<button
						type="button"
						class="deck-card"
						class:red={isRed(card)}
						class:taken={mine.has(card)}
						disabled={blocked.has(card) || (mine.has(card) && cards[cursor] !== card)}
						aria-label={cardName(card)}
						onclick={() => put(card)}
					>
						{rankLabel(rank)}<span class="text-[0.85em]">{SUIT_SYMBOLS[suit]}</span>
					</button>
				{/each}
			</div>
		{/each}
	</div>
</div>

<style>
	.slot {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		width: 2.6rem;
		height: 3.4rem;
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
		font-size: clamp(0.6rem, 2.2vw, 0.85rem);
		font-weight: 800;
		letter-spacing: -0.04em;
	}

	.deck-card.red {
		color: #cf2337;
	}

	.deck-card:hover:not(:disabled) {
		outline: 2px solid var(--accent);
	}

	.deck-card.taken {
		outline: 2px solid #38bdf8;
	}

	.deck-card:disabled:not(.taken) {
		background: #334155;
		color: #64748b;
	}
</style>

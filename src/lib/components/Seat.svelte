<!-- One player at the table: hole cards, name plate, stack and last action. -->

<script lang="ts">
	import type { PlayerState } from '$lib/types/game';
	import PlayingCard from './PlayingCard.svelte';

	let {
		player,
		active,
		dealer,
		handNumber
	}: { player: PlayerState; active: boolean; dealer: boolean; handNumber: number } = $props();

	function ordinal(n: number): string {
		if (n % 100 >= 11 && n % 100 <= 13) return `${n}th`;
		return `${n}${['th', 'st', 'nd', 'rd'][n % 10] ?? 'th'}`;
	}

	const out = $derived(player.chips === 0 && player.place !== null);
	const faceUp = $derived(player.is_human || player.revealed);
	// Bots muck when they fold; you keep seeing your own cards.
	const hasCards = $derived(
		player.in_hand && player.hole.length === 2 && (!player.folded || player.is_human)
	);
	const winner = $derived(player.won > 0);
	const showBest = $derived(winner && player.revealed && player.best_cards.length > 0);
</script>

<div
	class="seat"
	class:hero={player.is_human}
	class:active
	class:folded={player.folded && !out}
	class:out
	class:winner
>
	<div class="cards">
		{#if hasCards}
			<!-- The game state holds every hole card; only yours and revealed ones are drawn, the rest are backs. -->
			{#each [0, 1] as i (`${handNumber}-${i}`)}
				{@const card = faceUp ? (player.hole[i] ?? null) : null}
				<PlayingCard
					{card}
					highlight={showBest && card !== null && player.best_cards.includes(card)}
					dim={player.folded || (showBest && card !== null && !player.best_cards.includes(card))}
				/>
			{/each}
		{/if}
	</div>

	<div
		class="plate"
		title={player.persona ? `${player.name} — ${player.persona.style}` : undefined}
	>
		{#if dealer}
			<span class="button" title="Dealer button">D</span>
		{/if}
		<div class="name">
			{#if !player.is_human}<span aria-hidden="true">🤖</span>{/if}
			{player.name}
		</div>
		<div class="stack">
			{#if out}
				Out · {ordinal(player.place ?? 0)}
			{:else}
				{player.chips.toLocaleString('en-US')}
			{/if}
		</div>
		{#if winner}
			<span class="tag win">+{player.won.toLocaleString('en-US')}</span>
		{:else if player.badge && !out}
			<span class="tag" data-badge={player.badge}>{player.badge}</span>
		{/if}
	</div>

	<!-- Your own hand is named in the action bar; there is no room below the hero seat. -->
	{#if player.revealed && player.hand_name && !player.folded && !player.is_human}
		<div class="hand-name">{player.hand_name}</div>
	{/if}
</div>

<style>
	.seat {
		display: flex;
		flex-direction: column;
		align-items: center;
		width: 7.4em;
		transition: opacity 200ms ease;
	}

	.hero {
		font-size: 1.2em;
	}

	.cards {
		display: flex;
		gap: 0.2em;
		height: 3.7em;
		margin-bottom: -0.7em;
	}

	.plate {
		position: relative;
		z-index: 1;
		width: 100%;
		padding: 0.3em 0.5em 0.35em;
		border: 0.1em solid var(--panel-edge);
		border-radius: 0.6em;
		background: linear-gradient(180deg, #1a2540, #0f1729);
		box-shadow: 0 0.3em 0.8em rgb(0 0 0 / 0.5);
		text-align: center;
		line-height: 1.15;
		transition:
			border-color 200ms ease,
			box-shadow 200ms ease;
	}

	.name {
		overflow: hidden;
		font-size: 0.82em;
		font-weight: 600;
		color: #b9c4dc;
		white-space: nowrap;
		text-overflow: ellipsis;
	}

	.stack {
		font-size: 1em;
		font-weight: 800;
		font-variant-numeric: tabular-nums;
		color: #f4f6fb;
	}

	.active .plate {
		border-color: var(--accent);
		box-shadow:
			0 0 0 0.14em rgb(245 185 66 / 0.55),
			0 0 1.4em rgb(245 185 66 / 0.55);
		animation: pulse 1.4s ease-in-out infinite;
	}

	.winner .plate {
		border-color: var(--accent);
		box-shadow:
			0 0 0 0.14em var(--accent),
			0 0 1.8em rgb(245 185 66 / 0.7);
	}

	.folded {
		opacity: 0.5;
	}

	.out {
		opacity: 0.32;
	}

	.button {
		position: absolute;
		top: -0.65em;
		right: -0.55em;
		display: grid;
		place-items: center;
		width: 1.65em;
		height: 1.65em;
		border-radius: 50%;
		background: radial-gradient(circle at 35% 30%, #ffffff, #d9d4c6);
		box-shadow: 0 0.12em 0.3em rgb(0 0 0 / 0.6);
		color: #1a1f2b;
		font-size: 0.72em;
		font-weight: 900;
	}

	.tag {
		position: absolute;
		bottom: -0.85em;
		left: 50%;
		padding: 0.12em 0.6em;
		border-radius: 999px;
		background: #33415f;
		color: #e6ebf5;
		font-size: 0.68em;
		font-weight: 800;
		letter-spacing: 0.03em;
		text-transform: uppercase;
		white-space: nowrap;
		transform: translateX(-50%);
	}

	.tag[data-badge='Fold'] {
		background: #2a3145;
		color: #9aa6c0;
	}

	.tag[data-badge='Check'],
	.tag[data-badge='Call'] {
		background: #1f6f9c;
	}

	.tag[data-badge='Bet'],
	.tag[data-badge='Raise'] {
		background: #b7791f;
		color: #fff7e6;
	}

	.tag[data-badge='All-in'] {
		background: #c0303f;
		color: #fff;
	}

	.tag.win {
		background: var(--accent);
		color: #2a1c00;
		font-size: 0.78em;
	}

	.hand-name {
		margin-top: 1em;
		max-width: 11em;
		padding: 0.1em 0.55em;
		border-radius: 0.4em;
		background: rgb(0 0 0 / 0.55);
		color: #f4f6fb;
		font-size: 0.7em;
		font-weight: 600;
		text-align: center;
		line-height: 1.25;
	}

	@keyframes pulse {
		50% {
			box-shadow:
				0 0 0 0.14em rgb(245 185 66 / 0.85),
				0 0 2em rgb(245 185 66 / 0.8);
		}
	}
</style>

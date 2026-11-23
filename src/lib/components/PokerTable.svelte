<!--
	The table: felt, community cards, pot, and the seats laid out on an ellipse
	with the human at the bottom and play running clockwise. Everything inside
	is sized in em off the table's font-size, which tracks the available space.
-->

<script lang="ts">
	import type { GameState } from '$lib/types/game';
	import type { Card } from '$lib/utils/cards';
	import PlayingCard from './PlayingCard.svelte';
	import Seat from './Seat.svelte';

	let { game }: { game: GameState } = $props();

	const fmt = (n: number) => n.toLocaleString('en-US');

	/** Unit vector from the table centre to each seat (screen coordinates, y down). */
	const spots = $derived(
		game.players.map((_, i) => {
			const angle = Math.PI / 2 + (i * 2 * Math.PI) / game.players.length;
			return { x: Math.cos(angle), y: Math.sin(angle) };
		})
	);

	const winningCards = $derived(
		new Set<Card>(game.players.filter((p) => p.won > 0 && p.revealed).flatMap((p) => p.best_cards))
	);

	// Side pots only mean something once a betting round has closed. When the
	// hand is over the chips are already in the winners' stacks, so the middle is empty.
	const sidePots = $derived(game.phase === 'dealing' && game.pots.length > 1 ? game.pots : []);
	const middle = $derived(game.pot);
</script>

<div class="wrap">
	<div class="table">
		<div class="felt"></div>

		<div class="centre">
			<div class="board">
				{#each [0, 1, 2, 3, 4] as slot (slot)}
					<div class="slot">
						{#if game.board[slot] !== undefined}
							{#key `${game.hand_number}-${game.board[slot]}`}
								<PlayingCard
									card={game.board[slot]}
									highlight={winningCards.has(game.board[slot])}
									dim={winningCards.size > 0 && !winningCards.has(game.board[slot])}
								/>
							{/key}
						{/if}
					</div>
				{/each}
			</div>
			<div class="pot" class:empty={middle === 0}>
				<span class="chip" aria-hidden="true"></span>
				{#if sidePots.length > 0}
					<span>
						Main {fmt(sidePots[0].amount)}
						{#each sidePots.slice(1) as pot, i (i)}
							<span class="side">· Side {fmt(pot.amount)}</span>
						{/each}
					</span>
				{:else}
					<span>Pot {fmt(middle)}</span>
				{/if}
			</div>
		</div>

		{#each game.players as player (player.seat)}
			{@const spot = spots[player.seat]}
			{#if player.bet > 0}
				<div class="at bet" style:--x={spot.x} style:--y={spot.y} style:--r="0.5">
					<span class="chip" aria-hidden="true"></span>{fmt(player.bet)}
				</div>
			{/if}
			<div class="at" style:--x={spot.x} style:--y={spot.y} style:--r="1">
				<Seat
					{player}
					active={game.to_act === player.seat}
					dealer={game.dealer === player.seat && game.hand_number > 0}
					handNumber={game.hand_number}
				/>
			</div>
		{/each}
	</div>
</div>

<style>
	.wrap {
		display: grid;
		place-items: center;
		width: 100%;
		height: 100%;
		container-type: size;
	}

	.table {
		--w: min(100cqw, 185cqh);
		--rx: 44%;
		--ry: 38%;
		position: relative;
		width: var(--w);
		height: calc(var(--w) / 1.85);
		font-size: max(8px, calc(var(--w) / 62));
	}

	.felt {
		position: absolute;
		inset: 12% 7%;
		border: 0.9em solid var(--rail);
		border-radius: 999px;
		background: radial-gradient(ellipse at 50% 40%, var(--felt-hi), var(--felt-lo) 75%);
		box-shadow:
			inset 0 0 0 0.12em var(--brass),
			inset 0 0 4em rgb(0 0 0 / 0.45),
			0 1.2em 2.5em rgb(0 0 0 / 0.55);
	}

	/* Portrait: a tall table so seats stay readable on a phone. */
	@container (max-aspect-ratio: 1/1) {
		.table {
			--w: min(100cqw, 74cqh);
			--rx: 37%;
			--ry: 40%;
			height: calc(var(--w) / 0.74);
			font-size: max(8px, calc(var(--w) / 31));
		}

		.felt {
			inset: 9% 9%;
			border-radius: 34% / 22%;
		}
	}

	.centre {
		position: absolute;
		top: 44%;
		left: 50%;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.7em;
		transform: translate(-50%, -50%);
	}

	.board {
		display: flex;
		gap: 0.3em;
		font-size: 1.15em;
	}

	.slot {
		width: 2.6em;
		height: 3.7em;
		border-radius: 0.3em;
		box-shadow: inset 0 0 0 0.08em rgb(255 255 255 / 0.14);
	}

	.pot {
		display: flex;
		align-items: center;
		gap: 0.45em;
		padding: 0.25em 0.9em 0.25em 0.5em;
		border-radius: 999px;
		background: rgb(0 0 0 / 0.38);
		color: #f6f3e7;
		font-size: 1.05em;
		font-weight: 700;
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
		transition: opacity 200ms ease;
	}

	.pot.empty {
		opacity: 0;
	}

	.side {
		color: #d5c9a2;
	}

	.at {
		position: absolute;
		top: calc(50% + var(--y) * var(--ry) * var(--r));
		left: calc(50% + var(--x) * var(--rx) * var(--r));
		transform: translate(-50%, -50%);
	}

	.bet {
		display: flex;
		align-items: center;
		gap: 0.35em;
		padding: 0.12em 0.6em 0.12em 0.3em;
		border-radius: 999px;
		background: rgb(0 0 0 / 0.42);
		color: #fff;
		font-size: 0.92em;
		font-weight: 700;
		font-variant-numeric: tabular-nums;
		animation: push 220ms ease-out both;
	}

	.chip {
		width: 1.05em;
		height: 1.05em;
		border: 0.2em dashed #fff7e0;
		border-radius: 50%;
		background: #d6453f;
		box-shadow: 0 0.08em 0.2em rgb(0 0 0 / 0.5);
	}

	@keyframes push {
		from {
			opacity: 0;
			transform: translate(-50%, -50%) scale(0.7);
		}
	}
</style>

<!-- A single card; `card = null` shows the back. Sized in em, so the parent's font-size scales it. -->

<script lang="ts">
	import {
		type Card,
		SUIT_SYMBOLS,
		cardName,
		isRed,
		rankLabel,
		rankOf,
		suitOf
	} from '$lib/utils/cards';

	let {
		card = null,
		highlight = false,
		dim = false
	}: { card?: Card | null; highlight?: boolean; dim?: boolean } = $props();
</script>

{#if card !== null}
	<div
		class="card face"
		class:red={isRed(card)}
		class:highlight
		class:dim
		role="img"
		aria-label={cardName(card)}
	>
		<span class="rank">{rankLabel(rankOf(card))}</span>
		<span class="pip">{SUIT_SYMBOLS[suitOf(card)]}</span>
		<span class="suit">{SUIT_SYMBOLS[suitOf(card)]}</span>
	</div>
{:else}
	<div class="card back" class:dim role="img" aria-label="Face-down card"></div>
{/if}

<style>
	.card {
		position: relative;
		flex: none;
		width: 2.6em;
		height: 3.7em;
		border-radius: 0.3em;
		box-shadow:
			0 0.08em 0.15em rgb(0 0 0 / 0.45),
			0 0.3em 0.6em rgb(0 0 0 / 0.25);
		animation: deal 260ms cubic-bezier(0.2, 0.8, 0.3, 1) both;
		transition:
			transform 200ms ease,
			filter 200ms ease,
			box-shadow 200ms ease;
		user-select: none;
	}

	.face {
		background: linear-gradient(160deg, #ffffff, #f1efe8);
		color: #161b26;
		font-weight: 800;
		line-height: 1;
	}

	.face.red {
		color: #cf2337;
	}

	.rank {
		position: absolute;
		top: 0.22em;
		left: 0.24em;
		font-size: 1.12em;
		letter-spacing: -0.06em;
	}

	.pip {
		position: absolute;
		top: 1.85em;
		left: 0.36em;
		font-size: 0.72em;
	}

	.suit {
		position: absolute;
		right: 0.14em;
		bottom: 0.12em;
		font-size: 1.75em;
	}

	.back {
		border: 0.16em solid #f4f1ea;
		background:
			repeating-linear-gradient(45deg, rgb(255 255 255 / 0.13) 0 0.14em, transparent 0.14em 0.42em),
			repeating-linear-gradient(
				-45deg,
				rgb(255 255 255 / 0.13) 0 0.14em,
				transparent 0.14em 0.42em
			),
			linear-gradient(160deg, #3556b8, #1d2f73);
	}

	.highlight {
		transform: translateY(-0.35em);
		box-shadow:
			0 0 0 0.13em var(--accent),
			0 0.4em 0.9em rgb(245 185 66 / 0.45);
	}

	.dim {
		filter: brightness(0.55) saturate(0.6);
	}

	@keyframes deal {
		from {
			opacity: 0;
			transform: translateY(-0.7em) scale(0.92);
		}
	}
</style>

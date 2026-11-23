<!--
	Practice: a random seat at a random table size, everyone folds to you, and you decide
	whether the hand is a raise. Answers come from the same charts as everything else.
	Keys: R raise · F fold · N next.
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import { loadRangeChart } from '$lib/actions/ranges';
	import { rangeStore } from '$lib/stores/ranges.svelte';
	import type { Card } from '$lib/utils/cards';
	import { cardsForHand, handTitle, percentiles, randomHand } from '$lib/utils/hands';
	import HandGrid from './HandGrid.svelte';
	import PlayingCard from './PlayingCard.svelte';

	const STORAGE_KEY = 'robo-holdem:quiz';

	interface Question {
		players: number;
		seatName: string;
		seatPct: number;
		source: string;
		label: string;
		hand: string;
		cards: [Card, Card];
		raise: boolean;
		hands: Set<string>;
	}

	let size = $state<'any' | number>('any');
	let question = $state<Question | null>(null);
	let answer = $state<'raise' | 'fold' | null>(null);
	let score = $state({ right: 0, total: 0, streak: 0, best: 0 });
	let loading = $state(false);

	const correct = $derived(
		question !== null && answer !== null && (answer === 'raise') === question.raise
	);
	const order = $derived(question ? rangeStore.charts[question.players]?.order : undefined);
	const percentile = $derived(order && question ? percentiles(order)[question.hand] : undefined);

	onMount(() => {
		try {
			const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null');
			if (saved && [saved.right, saved.total, saved.streak, saved.best].every(Number.isFinite)) {
				score = { right: saved.right, total: saved.total, streak: saved.streak, best: saved.best };
			}
		} catch {
			// No saved score: start from zero.
		}
		void next();
	});

	function save() {
		try {
			localStorage.setItem(STORAGE_KEY, JSON.stringify(score));
		} catch {
			// Storage unavailable: the score lasts for this visit only.
		}
	}

	async function next() {
		answer = null;
		loading = true;
		const players = size === 'any' ? 2 + Math.floor(Math.random() * 8) : size;
		await loadRangeChart(players);
		const chart = rangeStore.charts[players];
		loading = false;
		if (!chart) return;
		const seats = chart.seats.filter((s) => s.source !== 'none');
		const seat = seats[Math.floor(Math.random() * seats.length)];
		const hand = randomHand();
		question = {
			players,
			seatName: seat.name,
			seatPct: seat.pct,
			source: seat.source,
			label: seat.label,
			hand,
			cards: cardsForHand(hand),
			raise: seat.hands.includes(hand),
			hands: new Set(seat.hands)
		};
	}

	function choose(choice: 'raise' | 'fold') {
		if (!question || answer) return;
		answer = choice;
		const right = (choice === 'raise') === question.raise;
		const streak = right ? score.streak + 1 : 0;
		score = {
			right: score.right + (right ? 1 : 0),
			total: score.total + 1,
			streak,
			best: Math.max(score.best, streak)
		};
		save();
	}

	function resetScore() {
		score = { right: 0, total: 0, streak: 0, best: 0 };
		save();
	}

	function onkeydown(event: KeyboardEvent) {
		if (event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
		const el = event.target as HTMLElement | null;
		if (el && ['INPUT', 'SELECT', 'TEXTAREA'].includes(el.tagName)) return;
		const key = event.key.toLowerCase();
		if (key === 'r') choose('raise');
		else if (key === 'f') choose('fold');
		else if (key === 'n' && answer) void next();
		else return;
		event.preventDefault();
	}
</script>

<svelte:window {onkeydown} />

<div class="flex flex-col gap-3">
	<div class="flex flex-wrap items-center justify-between gap-2 text-sm">
		<label class="flex items-center gap-2 font-semibold">
			Table size
			<select
				class="rounded-md border border-slate-600 bg-slate-800 px-2 py-1"
				bind:value={size}
				onchange={() => next()}
			>
				<option value="any">Any</option>
				{#each [2, 3, 4, 5, 6, 7, 8, 9] as n (n)}
					<option value={n}>{n} players</option>
				{/each}
			</select>
		</label>
		<div class="flex items-center gap-3 tabular-nums" aria-label="Score">
			<span><span class="font-bold">{score.right}</span>/{score.total} right</span>
			<span>Streak <span class="font-bold">{score.streak}</span></span>
			<span class="text-slate-400">Best {score.best}</span>
			<button class="text-xs text-slate-400 underline hover:text-slate-200" onclick={resetScore}>
				Reset
			</button>
		</div>
	</div>

	{#if question}
		<div class="rounded-lg border border-slate-700 bg-slate-900/60 p-3">
			<p class="text-sm text-slate-300">
				You are <span class="font-bold text-slate-100">{question.seatName}</span> at a
				<span class="font-bold text-slate-100">{question.players}-player</span> table. Everyone before
				you folds.
			</p>
			<div class="my-3 flex items-center justify-center gap-3 text-[1.1rem]" aria-label="Your hand">
				{#each question.cards as card, i (i)}
					<PlayingCard {card} />
				{/each}
			</div>
			<div class="flex justify-center gap-2">
				<button
					class="choice bg-amber-600 text-amber-950 hover:bg-amber-500"
					disabled={answer !== null}
					onclick={() => choose('raise')}
				>
					Raise
				</button>
				<button
					class="choice bg-slate-600 text-white hover:bg-slate-500"
					disabled={answer !== null}
					onclick={() => choose('fold')}
				>
					Fold
				</button>
				<button
					class="choice bg-sky-700 text-white hover:bg-sky-600"
					disabled={answer === null}
					onclick={() => next()}
				>
					Next hand
				</button>
			</div>
		</div>

		{#if answer}
			<section
				class="rounded-lg border p-3 text-sm"
				class:border-emerald-600={correct}
				class:bg-emerald-950={correct}
				class:border-rose-600={!correct}
				class:bg-rose-950={!correct}
				aria-live="polite"
			>
				<p class="font-bold">{correct ? 'Correct' : 'Not quite'}</p>
				<p class="mt-1 text-slate-200">
					{handTitle(question.hand)} ({question.hand}) is
					<span class="font-bold">{question.raise ? 'in' : 'outside'}</span>
					the {question.seatName} opening range ({question.seatPct}% of hands) at
					{question.players} players:
					<span class="font-bold">{question.raise ? 'raise' : 'fold'}</span>.
					{#if percentile !== undefined}
						It is a top {percentile.toFixed(1)}% hand.
					{/if}
				</p>
				{#if question.source !== 'table'}
					<p class="mt-1 text-xs text-slate-300">
						{question.source === 'computed'
							? 'This seat has no published chart; its range is computed from the strength order.'
							: 'Heads-up has no source: this is a stand-in, and real heads-up ranges are wider.'}
					</p>
				{/if}
				<div class="mt-3">
					<HandGrid
						hands={question.hands}
						verb="raise"
						label={`${question.seatName} opening range`}
						marked={question.hand}
					/>
				</div>
			</section>
		{/if}
	{:else if loading || rangeStore.loading}
		<p class="text-sm text-slate-400">Shuffling…</p>
	{:else if rangeStore.error}
		<p class="text-sm text-rose-300">{rangeStore.error}</p>
	{/if}
</div>

<style>
	.choice {
		min-width: 7rem;
		border-radius: 0.6rem;
		padding: 0.55rem 1.2rem;
		font-size: 1rem;
		font-weight: 800;
		box-shadow: 0 2px 0 rgb(0 0 0 / 0.35);
	}

	.choice:disabled {
		opacity: 0.45;
	}
</style>

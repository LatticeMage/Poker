<!--
	Practice: a random seat at a random table size, everyone folds to you, and you decide
	whether the hand is a raise. Answers come from the same charts as everything else.
	Keys: R raise · F fold · N next.
-->

<script lang="ts">
	import { quiz } from '$lib/stores/quiz.svelte';
	import { handTitle } from '$lib/utils/hands';
	import HandGrid from './HandGrid.svelte';
	import PlayingCard from './PlayingCard.svelte';

	function onkeydown(event: KeyboardEvent) {
		if (event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
		const el = event.target as HTMLElement | null;
		if (el && ['INPUT', 'SELECT', 'TEXTAREA'].includes(el.tagName)) return;
		const key = event.key.toLowerCase();
		if (key === 'r') quiz.choose('raise');
		else if (key === 'f') quiz.choose('fold');
		else if (key === 'n' && quiz.answer) quiz.next();
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
				value={quiz.size}
				onchange={(event) => quiz.setSize(event.currentTarget.value)}
			>
				<option value="any">Any</option>
				{#each [2, 3, 4, 5, 6, 7, 8, 9] as n (n)}
					<option value={n}>{n} players</option>
				{/each}
			</select>
		</label>
		<div class="flex items-center gap-3 tabular-nums" aria-label="Score">
			<span><span class="font-bold">{quiz.score.right}</span>/{quiz.score.total} right</span>
			<span>Streak <span class="font-bold">{quiz.score.streak}</span></span>
			<span class="text-slate-400">Best {quiz.score.best}</span>
			<button
				class="text-xs text-slate-400 underline hover:text-slate-200"
				onclick={() => quiz.resetScore()}
			>
				Reset
			</button>
		</div>
	</div>

	{#if quiz.question}
		<div class="rounded-lg border border-slate-700 bg-slate-900/60 p-3">
			<p class="text-sm text-slate-300">
				You are <span class="font-bold text-slate-100">{quiz.question.seatName}</span> at a
				<span class="font-bold text-slate-100">{quiz.question.players}-player</span> table. Everyone
				before you folds.
			</p>
			<div class="my-3 flex items-center justify-center gap-3 text-[1.1rem]" aria-label="Your hand">
				{#each quiz.question.cards as card, i (i)}
					<PlayingCard {card} />
				{/each}
			</div>
			<div class="flex justify-center gap-2">
				<button
					class="choice bg-amber-600 text-amber-950 hover:bg-amber-500"
					disabled={quiz.answer !== null}
					onclick={() => quiz.choose('raise')}
				>
					Raise
				</button>
				<button
					class="choice bg-slate-600 text-white hover:bg-slate-500"
					disabled={quiz.answer !== null}
					onclick={() => quiz.choose('fold')}
				>
					Fold
				</button>
				<button
					class="choice bg-sky-700 text-white hover:bg-sky-600"
					disabled={quiz.answer === null}
					onclick={() => quiz.next()}
				>
					Next hand
				</button>
			</div>
		</div>

		{#if quiz.answer}
			<section
				class="rounded-lg border p-3 text-sm"
				class:border-emerald-600={quiz.correct}
				class:bg-emerald-950={quiz.correct}
				class:border-rose-600={!quiz.correct}
				class:bg-rose-950={!quiz.correct}
				aria-live="polite"
			>
				<p class="font-bold">{quiz.correct ? 'Correct' : 'Not quite'}</p>
				<p class="mt-1 text-slate-200">
					{handTitle(quiz.question.hand)} ({quiz.question.hand}) is
					<span class="font-bold">{quiz.question.raise ? 'in' : 'outside'}</span>
					the {quiz.question.seatName} opening range ({quiz.question.seatPct}% of hands) at
					{quiz.question.players} players:
					<span class="font-bold">{quiz.question.raise ? 'raise' : 'fold'}</span>.
					{#if quiz.percentile !== undefined}
						It is a top {quiz.percentile.toFixed(1)}% hand.
					{/if}
				</p>
				{#if quiz.question.source !== 'table'}
					<p class="mt-1 text-xs text-slate-300">
						{quiz.question.source === 'computed'
							? 'This seat has no published chart; its range is computed from the strength order.'
							: quiz.question.source === 'solver'
								? 'Heads-up has no published chart: this is a solver approximation, and a hand counts as a raise when it raises at least half the time.'
								: 'Heads-up has no source: this is a stand-in, and real heads-up ranges are wider.'}
					</p>
				{/if}
				<div class="mt-3">
					<HandGrid
						hands={quiz.question.hands}
						verb="raise"
						label={`${quiz.question.seatName} opening range`}
						marked={quiz.question.hand}
					/>
				</div>
			</section>
		{/if}
	{:else if quiz.error}
		<p class="text-sm text-rose-300">{quiz.error}</p>
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

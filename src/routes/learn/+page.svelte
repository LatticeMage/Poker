<!--routes/learn/+page.svelte-->

<script lang="ts">
	import { resolve } from '$app/paths';
	import BoardEquity from '$lib/components/BoardEquity.svelte';
	import HandQuiz from '$lib/components/HandQuiz.svelte';
	import RangeChart from '$lib/components/RangeChart.svelte';
	import SizeMatrix from '$lib/components/SizeMatrix.svelte';
	import { settings } from '$lib/stores/settings.svelte';

	let players = $state(settings.players);
	let seatName = $state('');

	function select(n: number, seat: string) {
		players = n;
		seatName = seat;
	}
</script>

<svelte:head>
	<title>Learn · Robo Hold'em</title>
</svelte:head>

<div class="min-h-dvh">
	<header
		class="sticky top-0 z-10 flex items-center gap-3 border-b border-slate-800 bg-slate-950/90 px-3 py-2 backdrop-blur sm:px-4"
	>
		<a class="back" href={resolve('/')}>← Table</a>
		<h1 class="text-base font-black tracking-tight sm:text-lg">
			<span class="text-amber-400">♠</span> Learn
		</h1>
	</header>

	<main
		class="mx-auto grid max-w-6xl gap-6 px-3 py-4 sm:px-4 lg:grid-cols-[minmax(0,34rem)_minmax(0,1fr)]"
	>
		<div class="flex min-w-0 flex-col gap-6">
			<section aria-labelledby="charts-title">
				<h2 id="charts-title" class="mb-2 text-sm font-bold tracking-wide text-slate-400 uppercase">
					Starting-hand charts
				</h2>
				<RangeChart bind:players bind:seatName />
			</section>

			<section aria-labelledby="sizes-title">
				<h2 id="sizes-title" class="mb-2 text-sm font-bold tracking-wide text-slate-400 uppercase">
					Seat by table size
				</h2>
				<SizeMatrix {players} {seatName} onselect={select} />
				<p class="mt-1 text-xs text-slate-500">
					The fewer players left to act behind you, the more hands you can open. Click a number to
					see that chart.
				</p>
			</section>
		</div>

		<div class="flex min-w-0 flex-col gap-6">
			<section aria-labelledby="quiz-title">
				<h2 id="quiz-title" class="mb-2 text-sm font-bold tracking-wide text-slate-400 uppercase">
					Practice
				</h2>
				<HandQuiz />
			</section>

			<section
				class="rounded-lg border border-slate-800 bg-slate-950/40 p-3 text-xs leading-relaxed text-slate-400"
				aria-labelledby="about-title"
			>
				<h2 id="about-title" class="mb-1 text-sm font-bold tracking-wide text-slate-400 uppercase">
					Where the charts come from
				</h2>
				<ul class="list-disc space-y-1 pl-4">
					<li>
						<span class="font-semibold text-slate-300">Opening</span>: hand lists copied from public
						6-max and 9-max charts (100bb cash, raise first in), matched by how many players act
						behind you.
					</li>
					<li>
						<span class="font-semibold text-slate-300">Facing a raise</span>: the PokerStrategy
						6-max lists for every pair of positions, kept as mixes (a hand's combos can be split
						between actions). At other table sizes seats map to the nearest 6-max seat.
					</li>
					<li>
						<span class="font-semibold text-slate-300">Computed</span>: seats that only have a
						percentage in the sources take the strongest hands by a strength order built from the
						charts and from each hand's equity against a random hand.
					</li>
					<li>
						<span class="font-semibold text-slate-300">Adjustments</span> (approximations, not source
						data): robots defend less against bigger opens (minimum defence frequency), bluff less with
						callers already in, and play tighter with shallower stacks.
					</li>
					<li>
						<span class="font-semibold text-slate-300">Stand-in</span>: heads-up opening has no
						source here; real heads-up ranges are wider.
					</li>
					<li>
						<span class="font-semibold text-slate-300">Flop, turn and river (heads-up)</span>:
						robots work out both players' ranges from these charts, measure range-against-range
						equity on the actual board, and bet, bluff and defend by it (bluffs in the s / (1 + 2s)
						ratio, defence at the minimum defence frequency). Ranges are not narrowed by earlier
						betting. An approximation, not a solver.
					</li>
					<li>
						Short-stack push/fold is an offline approximation of the equilibrium (chips only, no
						tournament pressure) and is only a good model up to about 10bb.
					</li>
				</ul>
				<p class="mt-2">
					These are simplified teaching charts, not solver output for every spot. Not covered yet:
					limped pots, squeezes, cold 4-bets, 5-bet pots, and pots with three or more players.
				</p>
			</section>
		</div>
		<section class="min-w-0 lg:col-span-2" aria-labelledby="equity-title">
			<h2 id="equity-title" class="mb-2 text-sm font-bold tracking-wide text-slate-400 uppercase">
				Board equity
			</h2>
			<BoardEquity />
		</section>
	</main>
</div>

<style>
	.back {
		border: 1px solid #334155;
		border-radius: 0.5rem;
		background: #1e293b;
		padding: 0.3rem 0.6rem;
		font-size: 0.8rem;
		font-weight: 600;
		white-space: nowrap;
		color: #e2e8f0;
	}

	.back:hover {
		background: #334155;
	}
</style>

<!--routes/+page.svelte-->

<script lang="ts">
	import { resolve } from '$app/paths';
	import { act, nextHand, retry, startGame } from '$lib/actions/games';
	import BetControls from '$lib/components/BetControls.svelte';
	import CardPicker from '$lib/components/CardPicker.svelte';
	import ChartHint from '$lib/components/ChartHint.svelte';
	import GameLog from '$lib/components/GameLog.svelte';
	import Modal from '$lib/components/Modal.svelte';
	import PokerTable from '$lib/components/PokerTable.svelte';
	import RangeChart from '$lib/components/RangeChart.svelte';
	import SettingsPanel from '$lib/components/SettingsPanel.svelte';
	import { table } from '$lib/stores/game.svelte';
	import { settings } from '$lib/stores/settings.svelte';

	let showSettings = $state(false);
	let showPicker = $state(false);
	let showLog = $state(false);
	let showCharts = $state(false);

	const actor = $derived(
		table.game && table.game.to_act >= 0 ? table.game.players[table.game.to_act] : null
	);
	const champion = $derived(
		table.game?.phase === 'game_over'
			? (table.game.players.find((p) => p.place === 1) ?? null)
			: null
	);
	const handsToNextLevel = $derived(
		table.game
			? table.game.hands_per_level - ((table.game.hand_number - 1) % table.game.hands_per_level)
			: 0
	);

	function ordinal(n: number): string {
		if (n % 100 >= 11 && n % 100 <= 13) return `${n}th`;
		return `${n}${['th', 'st', 'nd', 'rd'][n % 10] ?? 'th'}`;
	}

	function newGame() {
		const inProgress =
			table.game && table.game.phase !== 'game_over' && !table.heroOut && !table.error;
		if (inProgress && !confirm('Abandon this game and start a new one?')) return;
		showSettings = false;
		void startGame();
	}
</script>

<div class="flex h-dvh flex-col overflow-hidden">
	<header
		class="flex shrink-0 flex-wrap items-center gap-x-2 gap-y-1 border-b border-slate-800 bg-slate-950/60 px-3 py-2 sm:flex-nowrap sm:gap-3 sm:px-4"
	>
		<h1 class="text-base font-black tracking-tight whitespace-nowrap sm:text-lg">
			<span class="text-amber-400">♠</span><span class="max-[420px]:sr-only"> Robo Hold'em</span>
		</h1>
		{#if table.game && table.game.hand_number > 0}
			<div
				class="order-last min-w-0 basis-full truncate text-xs text-slate-400 tabular-nums sm:order-none sm:basis-auto sm:text-sm"
			>
				Hand {table.game.hand_number}
				<span class="mx-1 text-slate-600">·</span>
				Blinds {table.game.small_blind}/{table.game.big_blind}
				{#if table.game.next_big_blind > table.game.big_blind && table.game.phase !== 'game_over'}
					<span class="hidden md:inline">
						<span class="mx-1 text-slate-600">·</span>
						{table.game.next_big_blind / 2}/{table.game.next_big_blind} in {handsToNextLevel}
						{handsToNextLevel === 1 ? 'hand' : 'hands'}
					</span>
				{/if}
			</div>
		{/if}
		<div class="ml-auto flex flex-wrap justify-end gap-1.5">
			<button class="top-btn lg:hidden" onclick={() => (showLog = true)}>Log</button>
			<button class="top-btn" onclick={() => (showCharts = true)}>Charts</button>
			<a class="top-btn" href={resolve('/learn')}>Learn</a>
			<button class="top-btn relative" onclick={() => (showPicker = true)}>
				Pick cards
				{#if settings.hasPicks}
					<span
						class="absolute -top-1 -right-1 h-2.5 w-2.5 rounded-full bg-amber-400"
						title="Picked cards are active"
					></span>
				{/if}
			</button>
			<button class="top-btn" onclick={() => (showSettings = true)}>Settings</button>
			<button class="top-btn" onclick={newGame}>New game</button>
		</div>
	</header>

	<div class="flex min-h-0 flex-1">
		<main class="flex min-w-0 flex-1 flex-col">
			<div class="min-h-0 flex-1 p-2 sm:p-3">
				{#if table.game}
					<PokerTable game={table.game} />
				{/if}
			</div>

			<section
				class="flex min-h-[8.5rem] shrink-0 flex-col items-center justify-center gap-2 border-t border-slate-800 bg-slate-950/60 px-3 py-3"
				aria-live="polite"
			>
				{#if table.error}
					<p class="text-sm font-semibold text-rose-300">{table.error}</p>
					<div class="flex gap-2">
						{#if !false}
							<button
								class="main-btn bg-slate-700 text-white hover:bg-slate-600"
								onclick={() => retry()}
							>
								Try again
							</button>
						{/if}
						<button
							class="main-btn bg-amber-600 text-amber-950 hover:bg-amber-500"
							onclick={newGame}
						>
							New game
						</button>
					</div>
				{:else if !table.game}
					<p class="text-sm text-slate-400">Shuffling up…</p>
				{:else if table.game.phase === 'game_over' && !table.replaying}
					<p class="text-center text-lg font-bold">
						{#if champion?.is_human}
							🏆 You won the game!
						{:else}
							{champion?.name} wins the table.game.
							{#if table.hero?.place}You finished {ordinal(table.hero.place)} of {table.game.players
									.length}.{/if}
						{/if}
					</p>
					<button class="main-btn bg-amber-600 text-amber-950 hover:bg-amber-500" onclick={newGame}>
						Play again
					</button>
				{:else if table.legal}
					{#if table.hero?.hand_name}
						<p class="text-xs text-slate-400">
							You have <span class="font-semibold text-slate-200">{table.hero.hand_name}</span>
						</p>
					{/if}
					{#if settings.hints}
						<ChartHint />
					{/if}
					{#key table.legal}
						<BetControls legal={table.legal} smallBlind={table.game.small_blind} onact={act} />
					{/key}
				{:else if table.awaitingNext}
					<p class="text-center text-sm font-semibold text-amber-300 sm:text-base">
						{table.game.headline}
					</p>
					{#if table.heroOut && table.hero?.place}
						<p class="text-xs text-slate-400">
							You finished {ordinal(table.hero.place)} — the robots are playing it out.
						</p>
					{/if}
					<div class="flex gap-2">
						<button
							class="main-btn bg-sky-700 text-white hover:bg-sky-600"
							onclick={() => nextHand()}
						>
							{settings.autoDeal ? 'Deal now' : 'Next hand'}
						</button>
						{#if table.heroOut}
							<button
								class="main-btn bg-amber-600 text-amber-950 hover:bg-amber-500"
								onclick={newGame}
							>
								New game
							</button>
						{/if}
					</div>
				{:else}
					<p class="text-sm text-slate-400">
						{#if actor && !actor.is_human}
							{actor.name} is thinking…
						{:else if table.heroOut}
							Watching the robots play it out…
						{:else}
							Dealing…
						{/if}
					</p>
					{#if table.hero?.hand_name && table.hero.in_hand && !table.hero.folded}
						<p class="text-xs text-slate-400">
							You have <span class="font-semibold text-slate-200">{table.hero.hand_name}</span>
						</p>
					{/if}
				{/if}
			</section>
		</main>

		<aside class="hidden w-72 shrink-0 flex-col border-l border-slate-800 bg-slate-950/40 lg:flex">
			<h2
				class="border-b border-slate-800 px-3 py-2 text-xs font-bold tracking-wide text-slate-400 uppercase"
			>
				Hand history
			</h2>
			<div class="min-h-0 flex-1"><GameLog lines={table.log} /></div>
		</aside>
	</div>
</div>

<Modal bind:open={showSettings} title="Settings">
	<SettingsPanel game={table.game} onnewgame={newGame} />
</Modal>

<Modal bind:open={showPicker} title="Pick cards">
	<CardPicker />
</Modal>

<Modal bind:open={showCharts} title="Starting-hand charts">
	<RangeChart />
</Modal>

<Modal bind:open={showLog} title="Hand history">
	<div class="h-[60dvh]"><GameLog lines={table.log} /></div>
</Modal>

<style>
	.top-btn {
		border: 1px solid #334155;
		border-radius: 0.5rem;
		background: #1e293b;
		padding: 0.3rem 0.6rem;
		font-size: 0.8rem;
		font-weight: 600;
		white-space: nowrap;
		color: #e2e8f0;
	}

	.top-btn:hover {
		background: #334155;
	}

	.main-btn {
		border-radius: 0.6rem;
		padding: 0.55rem 1.2rem;
		font-size: 0.95rem;
		font-weight: 800;
		box-shadow: 0 2px 0 rgb(0 0 0 / 0.35);
	}
</style>

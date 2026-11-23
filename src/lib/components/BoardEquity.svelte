<!--
	Board equity. Pick the preflop story (single-raised, 3-bet or 4-bet pot, and who you are), the
	flop (and the turn and river, if you like), and optionally your own two cards. You then see:
	  · neither hand known: range against range — the average equity of each side, an XY equity
	    distribution with a line per player, and a heat map of every hand's equity;
	  · your hand known (theirs not): your hand against their whole range, where it ranks in your
	    range, and what the postflop model would do with it.
-->

<script lang="ts">
	import { boardRequest, loadBoard, loadSpotList } from '$lib/actions/ranges';
	import { rangeStore } from '$lib/stores/ranges.svelte';
	import type { BoardEquity, BoardQuery } from '$lib/types/ranges';
	import { type Card, makeCard } from '$lib/utils/cards';
	import { holeToHand } from '$lib/utils/hands';
	import CardSelect from './CardSelect.svelte';
	import EquityChart from './EquityChart.svelte';
	import HandGrid from './HandGrid.svelte';
	import PlayingCard from './PlayingCard.svelte';

	type Pot = BoardQuery['pot'];

	let pot = $state<Pot>('srp');
	let openerPick = $state('BTN');
	let otherPick = $state('BB');
	let side = $state<'opener' | 'other'>('other');
	// Flop, turn, river. The turn and river are optional; a later card only counts if the ones before it are set.
	let board = $state<(Card | null)[]>([
		makeCard(14, 0),
		makeCard(7, 2),
		makeCard(2, 3),
		null,
		null
	]);
	let hole = $state<(Card | null)[]>([null, null]);
	let bet = $state<number | null>(null);
	let view = $state<'mine' | 'theirs'>('mine');

	const POTS: { id: Pot; label: string }[] = [
		{ id: 'srp', label: 'Single-raised pot' },
		{ id: '3bet', label: '3-bet pot' },
		{ id: '4bet', label: '4-bet pot' }
	];
	const BETS: { value: number | null; label: string }[] = [
		{ value: null, label: 'No bet' },
		{ value: 0.33, label: '⅓ pot' },
		{ value: 0.5, label: '½ pot' },
		{ value: 0.66, label: '⅔ pot' },
		{ value: 1, label: 'Pot' }
	];
	const PRESETS: { label: string; cards: Card[] }[] = [
		{ label: 'A-high dry', cards: [makeCard(14, 0), makeCard(7, 2), makeCard(2, 3)] },
		{ label: 'Low connected', cards: [makeCard(7, 1), makeCard(6, 1), makeCard(5, 2)] },
		{ label: 'Paired', cards: [makeCard(13, 0), makeCard(13, 2), makeCard(4, 3)] },
		{ label: 'Monotone', cards: [makeCard(12, 1), makeCard(8, 1), makeCard(3, 1)] },
		{ label: 'Broadway', cards: [makeCard(11, 3), makeCard(10, 2), makeCard(9, 0)] }
	];

	// Who can face whom comes from the server (the charts that exist).
	const spotList = $derived(rangeStore.spotList);
	const openers = $derived(spotList ? Object.keys(spotList.vs_3bet) : []);
	const opener = $derived(openers.includes(openerPick) ? openerPick : (openers[0] ?? ''));
	const others = $derived(spotList?.vs_3bet[opener] ?? []);
	const other = $derived(others.includes(otherPick) ? otherPick : (others[0] ?? ''));
	const otherRole = $derived(pot === 'srp' ? 'caller' : '3-bettor');

	const story = $derived(
		pot === 'srp'
			? `${opener} opened, ${other} called.`
			: pot === '3bet'
				? `${opener} opened, ${other} 3-bet, ${opener} called.`
				: `${opener} opened, ${other} 3-bet, ${opener} 4-bet, ${other} called.`
	);

	/** The cards in play: the flop and, in order, the turn and river if they are set. */
	const inPlay = $derived.by(() => {
		const cards: Card[] = [];
		for (const card of board) {
			if (card === null) break;
			cards.push(card);
		}
		return cards;
	});

	const query = $derived.by((): BoardQuery | null => {
		if (!opener || !other || inPlay.length < 3) return null;
		return {
			pot,
			opener,
			other,
			side,
			board: inPlay,
			hole: hole.every((c) => c !== null) ? (hole as Card[]) : null,
			bet,
			view
		};
	});
	const key = $derived(query ? boardRequest(query) : null);
	const fresh = $derived(key ? rangeStore.boards[key] : undefined);

	// Keep showing the previous answer (dimmed) while the next one loads.
	let last = $state.raw<BoardEquity | null>(null);
	const result = $derived(fresh ?? last);
	const stale = $derived(!fresh);

	$effect(() => {
		void loadSpotList();
	});
	$effect(() => {
		if (query) void loadBoard(query);
	});
	$effect(() => {
		if (fresh) last = fresh;
	});

	const blockedForBoard = $derived(new Set(hole.filter((c): c is Card => c !== null)));
	const blockedForHole = $derived(new Set(inPlay));

	function setBoard(cards: Card[]) {
		board = [...cards, null, null];
		// A card on the new flop cannot also be in your hand.
		hole = hole.map((c) => (c !== null && cards.includes(c) ? null : c));
	}

	function randomFlop() {
		const taken = new Set(hole.filter((c): c is Card => c !== null));
		const deck = Array.from({ length: 52 }, (_, i) => i).filter((c) => !taken.has(c));
		const picked: Card[] = [];
		while (picked.length < 3) {
			picked.push(...deck.splice(Math.floor(Math.random() * deck.length), 1));
		}
		board = [...picked, null, null];
	}

	const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;
	const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(1)}`;
	const youLabel = $derived(side === 'opener' ? opener : other);
	const themLabel = $derived(side === 'opener' ? other : opener);
	const handPoint = $derived(
		result?.known
			? {
					percentile: 1 - result.known.rank,
					equity: result.known.equity,
					name: result.known.hand
				}
			: null
	);
</script>

<div class="grid gap-5 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
	<!-- The setup -->
	<div class="flex min-w-0 flex-col gap-4">
		<div class="flex flex-col gap-1.5">
			<div class="text-xs font-semibold text-slate-400">Preflop</div>
			<div class="flex flex-wrap gap-1" role="tablist" aria-label="Pot type">
				{#each POTS as option (option.id)}
					<button
						class="chip"
						class:selected={pot === option.id}
						role="tab"
						aria-selected={pot === option.id}
						onclick={() => (pot = option.id)}
					>
						{option.label}
					</button>
				{/each}
			</div>
			{#if spotList}
				<div class="flex flex-wrap items-center gap-x-3 gap-y-1">
					<div class="flex items-center gap-1" role="tablist" aria-label="Opener">
						<span class="text-xs font-semibold text-slate-400">Opener</span>
						{#each openers as option (option)}
							<button
								class="chip"
								class:selected={opener === option}
								role="tab"
								aria-selected={opener === option}
								onclick={() => (openerPick = option)}
							>
								{option}
							</button>
						{/each}
					</div>
					<div class="flex items-center gap-1" role="tablist" aria-label={otherRole}>
						<span class="text-xs font-semibold text-slate-400 capitalize">{otherRole}</span>
						{#each others as option (option)}
							<button
								class="chip"
								class:selected={other === option}
								role="tab"
								aria-selected={other === option}
								onclick={() => (otherPick = option)}
							>
								{option}
							</button>
						{/each}
					</div>
				</div>
				<p class="text-xs text-slate-400">{story} 6-max charts, ~100bb.</p>
				<div class="flex items-center gap-1" role="tablist" aria-label="You are">
					<span class="text-xs font-semibold text-slate-400">I am</span>
					{#each [{ id: 'opener', name: opener }, { id: 'other', name: other }] as option (option.id)}
						<button
							class="chip"
							class:selected={side === option.id}
							role="tab"
							aria-selected={side === option.id}
							onclick={() => (side = option.id as 'opener' | 'other')}
						>
							{option.name}
						</button>
					{/each}
				</div>
			{:else}
				<p class="text-sm text-slate-400">Loading…</p>
			{/if}
		</div>

		<div class="flex flex-col gap-1.5">
			<div class="flex flex-wrap items-center justify-between gap-2">
				<span class="text-xs font-semibold text-slate-400">
					Board <span class="font-normal"
						>(flop required; add the turn and river for later streets)</span
					>
				</span>
				<div class="flex flex-wrap gap-1">
					{#each PRESETS as preset (preset.label)}
						<button class="chip small" onclick={() => setBoard(preset.cards)}>{preset.label}</button
						>
					{/each}
					<button class="chip small" onclick={randomFlop}>Random</button>
				</div>
			</div>
			<CardSelect
				bind:cards={board}
				name="Board"
				labels={['Flop 1', 'Flop 2', 'Flop 3', 'Turn', 'River']}
				blocked={blockedForBoard}
				compact
			/>
		</div>

		<div class="flex flex-col gap-1.5">
			<div class="flex flex-wrap items-center justify-between gap-2">
				<span class="text-xs font-semibold text-slate-400">
					My hand <span class="font-normal">(optional — the other player's cards stay unknown)</span
					>
				</span>
				{#if hole.some((c) => c !== null)}
					<button class="chip small" onclick={() => (hole = [null, null])}>Clear</button>
				{/if}
			</div>
			<CardSelect
				bind:cards={hole}
				name="Hand"
				labels={['Card 1', 'Card 2']}
				blocked={blockedForHole}
				compact
			/>
		</div>

		<label class="flex items-center gap-2 text-xs font-semibold text-slate-400">
			If the other player bets
			<select
				class="rounded-md border border-slate-600 bg-slate-800 px-2 py-1 text-sm text-slate-100"
				bind:value={bet}
			>
				{#each BETS as option (option.label)}
					<option value={option.value}>{option.label}</option>
				{/each}
			</select>
		</label>
	</div>

	<!-- What it shows -->
	<div class="flex min-w-0 flex-col gap-5" class:opacity-60={stale && !!result} aria-busy={stale}>
		{#if result}
			<section aria-labelledby="unknown-title" class="flex flex-col gap-3">
				<h3 id="unknown-title" class="text-sm font-bold tracking-wide text-slate-300 uppercase">
					Neither hand known <span class="font-normal text-slate-500 normal-case"
						>· range vs range</span
					>
				</h3>

				<div data-testid="range-equity">
					<div class="flex items-baseline justify-between text-sm">
						<span>
							<span class="font-bold text-amber-400">{youLabel}</span> (you)
							<span class="text-lg font-black tabular-nums">{pct(result.unknown.my_range_eq)}</span>
						</span>
						<span>
							<span class="text-lg font-black tabular-nums"
								>{pct(result.unknown.their_range_eq)}</span
							>
							<span class="font-bold text-sky-400">{themLabel}</span>
						</span>
					</div>
					<div
						class="flex h-3 overflow-hidden rounded-full"
						role="img"
						aria-label="Range equity split"
					>
						<div class="bg-amber-500" style:width={pct(result.unknown.my_range_eq)}></div>
						<div class="bg-sky-500" style:width={pct(result.unknown.their_range_eq)}></div>
					</div>
					<p class="mt-1 text-xs text-slate-400">
						{#if result.unknown.advantage === 'you'}
							You have the range advantage on this {result.texture} {result.street}.
						{:else if result.unknown.advantage === 'them'}
							They have the range advantage on this {result.texture} {result.street}.
						{:else}
							The ranges are about even on this {result.texture} {result.street}.
						{/if}
						{result.unknown.my_combos} combos for you, {result.unknown.their_combos} for them.
					</p>
				</div>

				<EquityChart
					mine={result.unknown.my_curve}
					theirs={result.unknown.their_curve}
					mineHands={result.unknown.my_hands}
					theirHands={result.unknown.their_hands}
					hand={handPoint}
					mineLabel={`${youLabel} (you)`}
					theirLabel={themLabel}
				/>

				<div class="flex flex-col gap-1.5">
					<div class="flex flex-wrap items-center justify-between gap-2">
						<span class="text-xs font-semibold text-slate-400">Equity of each hand</span>
						<div class="flex gap-1" role="tablist" aria-label="Heat map of">
							<button
								class="chip small"
								class:selected={view === 'mine'}
								role="tab"
								aria-selected={view === 'mine'}
								onclick={() => (view = 'mine')}
							>
								Your range
							</button>
							<button
								class="chip small"
								class:selected={view === 'theirs'}
								role="tab"
								aria-selected={view === 'theirs'}
								onclick={() => (view = 'theirs')}
							>
								Their range
							</button>
						</div>
					</div>
					<HandGrid
						hands={new Set<string>()}
						verb=""
						label={`Equity of ${view === 'mine' ? 'your' : 'their'} range`}
						heat={result.unknown.grid}
						marked={view === 'mine' && result.known ? result.known.hand : null}
					/>
					<div class="flex items-center gap-2 text-[0.65rem] text-slate-400">
						<span>0%</span>
						<span class="legend-bar flex-1"></span>
						<span>100% equity</span>
					</div>
				</div>

				<div
					class="rounded-lg border border-slate-700 bg-slate-900/60 p-3 text-sm leading-relaxed"
					data-testid="flop-plan"
				>
					{#if result.plan.role === 'aggressor'}
						<p>
							You were the preflop aggressor: bet about
							<span class="font-bold tabular-nums">{pct(result.plan.bet_frequency, 0)}</span> of
							your range for
							<span class="font-bold tabular-nums">{pct(result.plan.bet_size, 0)}</span> of the pot.
							The top
							<span class="font-bold tabular-nums">{pct(result.plan.value_cut, 0)}</span> bets for
							value and about
							<span class="font-bold tabular-nums">{pct(result.plan.bluff_share_of_bets, 0)}</span> of
							your bets are bluffs.
						</p>
					{:else}
						<p>
							You called preflop: when checked to, bet about
							<span class="font-bold tabular-nums">{pct(result.plan.bet_frequency, 0)}</span> of
							your range for
							<span class="font-bold tabular-nums">{pct(result.plan.bet_size, 0)}</span> of the pot.
						</p>
					{/if}
					{#if result.facing}
						<p class="mt-1">
							Facing a <span class="font-bold">{pct(result.facing.bet, 0)}</span>-pot bet you need
							<span class="font-bold tabular-nums">{pct(result.facing.required_equity)}</span>
							equity to call; defend
							<span class="font-bold tabular-nums">{pct(result.facing.defend_frequency, 0)}</span> of
							your range.
						</p>
					{/if}
				</div>
			</section>

			<section aria-labelledby="known-title" class="flex flex-col gap-3">
				<h3 id="known-title" class="text-sm font-bold tracking-wide text-slate-300 uppercase">
					Your hand known <span class="font-normal text-slate-500 normal-case"
						>· their cards unknown</span
					>
				</h3>
				{#if result.known}
					{@const known = result.known}
					<div class="flex flex-wrap items-center gap-4" data-testid="known-hand">
						<div class="flex gap-1 text-[1rem]">
							{#each known.cards as card (card)}
								<PlayingCard {card} />
							{/each}
						</div>
						<div>
							<div class="text-xs text-slate-400">
								{holeToHand(known.cards)} · {known.made_hand}
							</div>
							<div class="text-3xl leading-tight font-black text-amber-400 tabular-nums">
								{pct(known.equity)}
							</div>
							<div class="text-xs text-slate-400">equity against their whole range</div>
						</div>
					</div>
					<dl class="grid grid-cols-2 gap-x-4 gap-y-1 text-sm sm:grid-cols-4">
						<div>
							<dt class="text-xs text-slate-400">Your range's average</dt>
							<dd class="font-semibold tabular-nums">{pct(result.unknown.my_range_eq)}</dd>
						</div>
						<div>
							<dt class="text-xs text-slate-400">This hand vs that</dt>
							<dd class="font-semibold tabular-nums">{signed(known.vs_range)} pts</dd>
						</div>
						<div>
							<dt class="text-xs text-slate-400">Rank in your range</dt>
							<dd class="font-semibold tabular-nums">
								top {Math.max(1, Math.round(known.rank * 100))}%
							</dd>
						</div>
						<div>
							<dt class="text-xs text-slate-400">Their range vs your hand</dt>
							<dd class="font-semibold tabular-nums">{pct(known.their_equity)}</dd>
						</div>
					</dl>
					<p
						class="rounded-lg border border-slate-700 bg-slate-900/60 p-3 text-sm"
						data-testid="flop-advice"
					>
						The flop model would <span class="font-bold text-amber-400">{known.action}</span>: {known.reason}.
					</p>
					<p class="text-xs text-slate-500">
						Knowing your cards changes the answer: the range averages {pct(
							result.unknown.my_range_eq
						)}, but this exact hand has {pct(known.equity)} — and your two cards are also removed from
						what they can hold.
					</p>
				{:else}
					<p class="text-sm text-slate-400">
						Pick your two cards on the left to see your hand's equity against their range, where it
						ranks in yours, and what to do with it.
					</p>
				{/if}
			</section>
			<p class="text-xs text-slate-500">{result.note}</p>
		{:else if rangeStore.error}
			<p class="text-sm text-rose-300">{rangeStore.error}</p>
		{:else}
			<p class="text-sm text-slate-400">Loading…</p>
		{/if}
	</div>
</div>

<style>
	.legend-bar {
		height: 0.5rem;
		border-radius: 999px;
		background: linear-gradient(90deg, rgb(185 28 28), rgb(51 65 85), rgb(21 128 61));
	}

	.chip {
		border: 1px solid #334155;
		border-radius: 0.4rem;
		background: #1e293b;
		padding: 0.2rem 0.55rem;
		font-size: 0.78rem;
		font-weight: 700;
		color: #cbd5e1;
	}

	.chip.small {
		padding: 0.1rem 0.45rem;
		font-size: 0.7rem;
	}

	.chip.selected {
		border-color: var(--accent);
		background: #78350f;
		color: #fff7e6;
	}
</style>

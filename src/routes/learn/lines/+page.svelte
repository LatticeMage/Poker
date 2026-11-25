<!--routes/learn/lines/+page.svelte-->
<!--
	A heads-up hand, step by step: set the preflop story, the board and the actions; see for every
	stage both ranges, both equities and both strategies. The whole line is replayed from the first
	step on every change (lib/poker/line.ts).
-->

<script lang="ts">
	import { resolve } from '$app/paths';
	import CardSelect from '$lib/components/CardSelect.svelte';
	import EquitySteps from '$lib/components/EquitySteps.svelte';
	import HandGrid from '$lib/components/HandGrid.svelte';
	import HandLine from '$lib/components/HandLine.svelte';
	import { type LineState, type PotKind, type Side, lineSteps } from '$lib/poker/line';
	import { attempt, handLine, spotList as getSpotList } from '$lib/poker/lookup';
	import { byClass } from '$lib/poker/range';
	import { boardInputs as inputs } from '$lib/stores/boardEquity.svelte';
	import { holeToHand } from '$lib/utils/hands';
	import { describeStep } from '$lib/utils/lines';

	const POTS: { id: PotKind; label: string }[] = [
		{ id: 'srp', label: 'Single-raised pot' },
		{ id: '3bet', label: '3-bet pot' },
		{ id: '4bet', label: '4-bet pot' }
	];
	const SIDES: Side[] = ['opener', 'other'];

	const spotList = $derived(attempt(getSpotList).value);
	const openers = $derived(spotList ? Object.keys(spotList.vs_3bet) : []);
	const opener = $derived(openers.includes(inputs.opener) ? inputs.opener : (openers[0] ?? ''));
	const others = $derived(spotList?.vs_3bet[opener] ?? []);
	const other = $derived(others.includes(inputs.other) ? inputs.other : (others[0] ?? ''));
	const pot = $derived(inputs.pot);
	const side = $derived(inputs.side);
	const names = $derived<[string, string]>([opener, other]);
	const seat = (s: Side) => (s === 'opener' ? opener : other);

	const built = $derived(lineSteps(inputs.inPlay, inputs.line));
	const hole = $derived(inputs.holeCards);
	const answer = $derived(
		opener && other && built.steps.length
			? attempt(() =>
					handLine({
						pot,
						opener,
						other,
						players: inputs.players,
						hole: hole ? (side === 'opener' ? { opener: hole } : { other: hole }) : undefined,
						steps: built.steps
					})
				)
			: undefined
	);
	const states = $derived(answer?.value ?? []);

	// The picked row; null follows the last step, so a new action shows at once.
	let picked = $state<number | null>(null);
	const selected = $derived(
		picked !== null && picked < states.length ? picked : Math.max(0, states.length - 1)
	);
	const now = $derived<LineState | undefined>(states[selected]);
	const pick = (index: number) => (picked = index === states.length - 1 ? null : index);

	const points = $derived(states.map((s) => ({ label: describeStep(s, seat), equity: s.equity })));
	const pct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`;

	/** The share of every hand class still in a player's range, as the grid wants it. */
	function rangeMix(state: LineState, s: number): Record<string, [number, number, number]> {
		const share = byClass(state.board, (i) => state.ranges[s][i]);
		return Object.fromEntries(
			Object.entries(share)
				.filter(([, w]) => w > 0)
				.map(([hand, w]) => [hand, [Math.min(1, w), 0, 0]])
		);
	}

	/** A player's strategy at the selected stage: the one it has now, else its latest decision. */
	function lastDecision(s: Side): { at: number; state: LineState } | null {
		for (let i = selected; i >= 0; i--) {
			if (states[i]?.strategy?.by === s) return { at: i, state: states[i] };
		}
		return null;
	}

	/** Per hand: [bet or raise, -, call]; the rest checks or folds. Combos count by their weight. */
	function actionMix(state: LineState): Record<string, [number, number, number]> {
		const st = state.strategy!;
		const range = state.ranges[st.by === 'opener' ? 0 : 1];
		const share = (option: number) =>
			byClass(
				state.board,
				(i) => st.combos[i]![option],
				(i) => (st.combos[i] ? range[i] : 0)
			);
		const raise = share(st.options.length === 2 ? 1 : 2);
		const call = st.options.length === 2 ? {} : share(1);
		return Object.fromEntries(
			Object.keys(raise).map((hand) => [
				hand,
				[raise[hand], 0, (call as Record<string, number>)[hand] ?? 0]
			])
		);
	}
	const mineClass = $derived(hole ? holeToHand(hole) : null);
</script>

<svelte:head>
	<title>Hand lines · Robo Hold'em</title>
</svelte:head>

<div class="min-h-dvh">
	<header
		class="sticky top-0 z-10 flex items-center gap-3 border-b border-slate-800 bg-slate-950/90 px-3 py-2 backdrop-blur sm:px-4"
	>
		<a class="back" href={resolve('/learn')}>← Learn</a>
		<h1 class="text-base font-black tracking-tight sm:text-lg">
			<span class="text-amber-400">♠</span> Hand lines
		</h1>
	</header>

	<main
		class="mx-auto grid max-w-6xl gap-5 px-3 py-4 sm:px-4 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]"
	>
		<!-- The setup -->
		<section class="flex min-w-0 flex-col gap-3" aria-label="Setup">
			<div class="flex flex-wrap gap-1" role="group" aria-label="Pot type">
				{#each POTS as option (option.id)}
					<button
						class="chip"
						class:selected={pot === option.id}
						aria-pressed={pot === option.id}
						onclick={() => (inputs.pot = option.id)}>{option.label}</button
					>
				{/each}
			</div>
			<div class="flex flex-wrap items-center gap-1" role="group" aria-label="Opener">
				<span class="label">Opener</span>
				{#each openers as option (option)}
					<button
						class="chip"
						class:selected={opener === option}
						aria-pressed={opener === option}
						onclick={() => (inputs.opener = option)}>{option}</button
					>
				{/each}
			</div>
			<div class="flex flex-wrap items-center gap-1" role="group" aria-label="Other player">
				<span class="label">{pot === 'srp' ? 'Caller' : '3-bettor'}</span>
				{#each others as option (option)}
					<button
						class="chip"
						class:selected={other === option}
						aria-pressed={other === option}
						onclick={() => (inputs.other = option)}>{option}</button
					>
				{/each}
			</div>
			<div class="flex flex-wrap items-center gap-1" role="group" aria-label="Players in the pot">
				<span class="label">Players in the pot</span>
				{#each [2, 3, 4, 5, 6] as n (n)}
					<button
						class="chip"
						class:selected={inputs.players === n}
						aria-pressed={inputs.players === n}
						onclick={() => (inputs.players = n)}>{n}</button
					>
				{/each}
			</div>
			<div class="flex flex-wrap items-center gap-1" role="group" aria-label="Your side">
				<span class="label">You are</span>
				{#each SIDES as s (s)}
					<button
						class="chip"
						class:selected={side === s}
						aria-pressed={side === s}
						onclick={() => (inputs.side = s)}>{seat(s)}</button
					>
				{/each}
			</div>
			<CardSelect
				cards={inputs.slots}
				onpick={(index, card) => inputs.setSlot(index, card)}
				name="Cards"
				labels={['Flop 1', 'Flop 2', 'Flop 3', 'Turn', 'River', 'Hand 1', 'Hand 2']}
				groups={[
					{ title: 'Board', from: 0, to: 5 },
					{ title: 'My hand', from: 5, to: 7 }
				]}
				compact
			/>
		</section>

		<!-- The line and the stage picked in it -->
		<div class="flex min-w-0 flex-col gap-5">
			<HandLine
				{states}
				error={answer?.error}
				used={built.used}
				{opener}
				{other}
				{side}
				known={hole !== null}
				{selected}
				onselect={pick}
			/>

			{#if states.length && now}
				<section aria-labelledby="equity-title">
					<h2 id="equity-title" class="title">Equity, step by step</h2>
					<EquitySteps {points} {names} {selected} onselect={pick} />
				</section>

				<section aria-labelledby="range-title">
					<h2 id="range-title" class="title">
						Ranges after step {selected}: {describeStep(now, seat)}
					</h2>
					<div class="grid gap-4 md:grid-cols-2">
						{#each [0, 1] as s (s)}
							<div class="min-w-0" data-testid="line-range">
								<p class="mb-1 text-xs text-slate-300">
									<span class="font-semibold">{names[s]}</span>: {now.combos[s].toFixed(0)} combos,
									{pct(now.reach[s])} of its starting range
									{#if now.equity}· equity {pct(now.equity[s], 1)}{/if}
								</p>
								<HandGrid
									hands={new Set()}
									verb="in range"
									label={`${names[s]} range`}
									mix={rangeMix(now, s)}
									names={{ value: 'in the range', bluff: '', call: '', rest: 'gone' }}
									marked={SIDES[s] === side ? mineClass : null}
								/>
							</div>
						{/each}
					</div>
					<p class="mt-1 text-xs text-slate-500">
						The filled part of a cell is the share of that hand's combos still in the range.
					</p>
				</section>

				<section aria-labelledby="strategy-title">
					<h2 id="strategy-title" class="title">Strategies at this stage</h2>
					<div class="grid gap-4 md:grid-cols-2">
						{#each SIDES as s (s)}
							{@const found = lastDecision(s)}
							<div class="min-w-0" data-testid="line-strategy">
								{#if found}
									{@const st = found.state.strategy!}
									<p class="mb-1 text-xs text-slate-300">
										<span class="font-semibold">{seat(s)}</span>
										{found.at === selected && found.state.to_act === s
											? 'to act now'
											: `last decision, after step ${found.at}`}:
										{#each st.options as option, i (option)}
											{i ? ', ' : ''}{option}
											<span class="font-semibold tabular-nums">{pct(st.range[i])}</span>
										{/each}
										<span class="text-slate-400">
											({st.source === 'solved' ? 'solved flop table' : 'rules'})
										</span>
										{#if st.hand && s === side}
											· your hand:
											{#each st.options as option, i (option)}
												{i ? ', ' : ''}{option}
												<span class="font-semibold tabular-nums">{pct(st.hand[i])}</span>
											{/each}
										{/if}
									</p>
									<HandGrid
										hands={new Set()}
										verb="bet"
										label={`${seat(s)} strategy`}
										mix={actionMix(found.state)}
										names={st.options.length === 2
											? { value: 'bet', bluff: '', call: '', rest: 'check' }
											: { value: 'raise', bluff: '', call: 'call', rest: 'fold' }}
										marked={s === side ? mineClass : null}
									/>
								{:else}
									<p class="text-xs text-slate-400">
										<span class="font-semibold">{seat(s)}</span> has not had a decision yet.
									</p>
								{/if}
							</div>
						{/each}
					</div>
					<p class="mt-1 text-xs text-slate-500">
						<span class="swatch" style:background="#d97706"></span>bet or raise
						<span class="swatch ml-3" style:background="#0284c7"></span>call
						<span class="swatch ml-3" style:background="#1e293b"></span>check or fold. "Solved flop
						table" means the range totals come from the solved game for this kind of flop; which
						hands take the action is still ranked by the model's rules. Turn, river and unsolved
						matchups use the rules on the ranges that are left.
					</p>
				</section>
			{/if}
		</div>
	</main>
</div>

<style>
	.back,
	.chip {
		border: 1px solid #334155;
		border-radius: 0.45rem;
		background: #1e293b;
		padding: 0.2rem 0.55rem;
		font-size: 0.78rem;
		font-weight: 700;
		white-space: nowrap;
		color: #cbd5e1;
	}

	.chip.selected {
		border-color: var(--accent);
		background: #78350f;
		color: #fff7e6;
	}

	.label {
		font-size: 0.75rem;
		font-weight: 600;
		color: #94a3b8;
	}

	.title {
		margin-bottom: 0.4rem;
		font-size: 0.8rem;
		font-weight: 700;
		color: #94a3b8;
	}

	.swatch {
		display: inline-block;
		width: 0.7rem;
		height: 0.7rem;
		margin-right: 0.3rem;
		border: 1px solid #475569;
		border-radius: 0.15rem;
		vertical-align: -0.1rem;
	}
</style>

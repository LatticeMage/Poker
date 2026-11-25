<!--
	The steps of the hand as a table, and the buttons that add the next action. Picking a row shows
	that stage in detail on the page. The page computes the states; this only shows them.
-->

<script lang="ts">
	import type { LineAction, LineState, Side } from '$lib/poker/line';
	import { boardInputs as inputs } from '$lib/stores/boardEquity.svelte';
	import { describeStep } from '$lib/utils/lines';

	let {
		states,
		error,
		used,
		opener,
		other,
		side,
		known,
		selected,
		onselect
	}: {
		states: LineState[];
		error?: string;
		/** How many of the stored actions are in the states. */
		used: number;
		opener: string;
		other: string;
		side: Side;
		/** Both of your cards are given. */
		known: boolean;
		selected: number;
		onselect: (index: number) => void;
	} = $props();

	const last = $derived<LineState | undefined>(states[states.length - 1]);
	const waiting = $derived(inputs.line.length - used);
	const seat = (s: Side) => (s === 'opener' ? opener : other);
	const pct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`;
	const bb = (v: number) => `${Number(v.toFixed(2))}bb`;
	const me = $derived(side === 'opener' ? 0 : 1);

	// What the player to act can do now: sizes are a third and two thirds of the pot, a raise is three
	// times the bet.
	const options = $derived.by((): { label: string; action: LineAction }[] => {
		if (!last?.to_act) return [];
		const by = last.to_act;
		const mine = last.in[by === 'opener' ? 0 : 1];
		const high = Math.max(...last.in);
		const round = (x: number) => Math.round(x * 10) / 10;
		if (high - mine <= 0) {
			return [
				{ label: 'Check', action: { kind: 'check', by } },
				{
					label: `Bet ⅓ (${bb(round(last.pot / 3))})`,
					action: { kind: 'bet', by, to: round(last.pot / 3) }
				},
				{
					label: `Bet ⅔ (${bb(round((2 * last.pot) / 3))})`,
					action: { kind: 'bet', by, to: round((2 * last.pot) / 3) }
				}
			];
		}
		return [
			{ label: 'Fold', action: { kind: 'fold', by } },
			{ label: `Call ${bb(high - mine)}`, action: { kind: 'call', by } },
			{
				label: `Raise to ${bb(round(high * 3))}`,
				action: { kind: 'raise', by, to: round(high * 3) }
			}
		];
	});
</script>

<div class="flex flex-col gap-2" data-testid="hand-line">
	<p class="text-xs text-slate-400">
		Add what the two players did, in order, then pick any row to see that stage. Each step keeps, of
		the acting player's range, the share of every hand that takes that step; everything is worked
		out again from the first step.
	</p>

	{#if error}
		<p class="text-sm text-rose-300" data-testid="line-error">{error}</p>
	{:else if states.length}
		<div class="overflow-x-auto">
			<table class="w-full min-w-[36rem] text-left text-xs tabular-nums">
				<thead class="text-slate-400">
					<tr>
						<th class="py-1 pr-2 font-semibold">Step</th>
						<th class="px-2 font-semibold">Pot</th>
						<th class="px-2 font-semibold">{opener}: combos · equity</th>
						<th class="px-2 font-semibold">{other}: combos · equity</th>
						<th class="px-2 font-semibold">Range does it</th>
						{#if known}<th class="px-2 font-semibold">Your hand</th>{/if}
					</tr>
				</thead>
				<tbody>
					{#each states as state (state.index)}
						<tr
							class="cursor-pointer border-t border-slate-800 text-slate-200"
							class:picked={selected === state.index}
							data-testid="line-row"
							onclick={() => onselect(state.index)}
						>
							<td class="py-1 pr-2 font-semibold">
								<button
									class="text-left font-semibold"
									aria-pressed={selected === state.index}
									onclick={() => onselect(state.index)}
								>
									{state.index}. {describeStep(state, seat)}
								</button>
							</td>
							<td class="px-2">{bb(state.pot)}</td>
							{#each [0, 1] as s (s)}
								<td class="px-2">
									{state.combos[s].toFixed(0)}
									<span class="text-slate-400">({pct(state.reach[s])})</span>
									· {state.equity ? pct(state.equity[s], 1) : '–'}
								</td>
							{/each}
							<td class="px-2">{state.taken === null ? '' : pct(state.taken)}</td>
							{#if known}
								<td class="px-2">
									{state.hand_equity[me] === null ? '–' : pct(state.hand_equity[me]!, 1)}
									{#if state.hand_taken !== null}
										<span class="text-slate-400">(does it {pct(state.hand_taken)})</span>
									{/if}
								</td>
							{/if}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>

		{#if last?.over}
			<p class="text-sm text-slate-300" data-testid="line-over">
				The hand is over: {last.over === 'fold' ? 'a player folded' : 'showdown'}.
			</p>
		{:else if last && !last.to_act}
			<p class="text-sm text-slate-300" data-testid="line-waiting">
				The betting on this street is over: put the next card on the board to go on.
			</p>
		{/if}
	{:else}
		<p class="text-sm text-slate-400">Put a flop on the board to start.</p>
	{/if}

	<div class="flex flex-wrap items-center gap-1">
		{#each options as option (option.label)}
			<button class="chip" onclick={() => inputs.act(option.action, used)}>
				{option.label}
			</button>
		{/each}
		{#if inputs.line.length}
			<button class="chip" onclick={() => inputs.undo()}>Undo</button>
			<button class="chip" onclick={() => inputs.clearLine()}>Clear</button>
		{/if}
		{#if waiting > 0}
			<span class="text-xs text-slate-400"
				>{waiting} later action{waiting > 1 ? 's' : ''} not shown (they come after a card that is not
				on the board, or after a fold).</span
			>
		{/if}
	</div>
</div>

<style>
	.picked {
		background: #78350f55;
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
</style>

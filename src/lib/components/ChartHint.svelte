<!--
	While you play (Settings → chart hints): before the flop, what the charts say about your hand
	from your seat — opening first in, facing an open, or facing a 3-bet after you opened. The charts
	are 6-max, so at other table sizes the seats map to the nearest 6-max seat.
-->

<script lang="ts">
	import { loadRangeChart, loadSpot } from '$lib/actions/ranges';
	import { game, hero } from '$lib/stores/game.store';
	import { rangeStore, spotKey } from '$lib/stores/ranges.svelte';
	import { holeToHand, positionClass, preflopOrder } from '$lib/utils/hands';

	const dealt = $derived($game ? $game.players.filter((p) => p.in_hand).length : 0);

	/** Which branch of the preflop tree you are in, from who has raised. */
	const situation = $derived.by(() => {
		const g = $game;
		const me = $hero;
		if (!g || !me || g.street !== 'preflop' || !me.in_hand || me.hole.length !== 2) return null;
		const order = preflopOrder(g);
		const behind = (seat: number) => order.length - 1 - order.indexOf(seat);
		const raisers = g.raisers ?? [];
		const hand = holeToHand(me.hole);
		if (raisers.length === 0)
			return { type: 'first-in' as const, hand, index: order.indexOf(me.seat) };
		if (raisers.length === 1 && raisers[0] !== me.seat) {
			return {
				type: 'vs_open' as const,
				hand,
				hero: positionClass(behind(me.seat)),
				villain: positionClass(behind(raisers[0]), true)
			};
		}
		if (raisers.length === 2 && raisers[0] === me.seat) {
			return {
				type: 'vs_3bet' as const,
				hand,
				hero: positionClass(behind(me.seat), true),
				villain: positionClass(behind(raisers[1]))
			};
		}
		return { type: 'deep' as const, hand };
	});

	$effect(() => {
		const s = situation;
		if (!s) return;
		if (s.type === 'first-in' && dealt >= 2) void loadRangeChart(dealt);
		else if (s.type === 'vs_open' || s.type === 'vs_3bet') void loadSpot(s.type, s.hero, s.villain);
	});

	const hint = $derived.by(() => {
		const s = situation;
		if (!s) return null;
		if (s.type === 'deep') {
			return { hand: s.hand, text: 'Deep in a raised pot: only the strongest hands continue.' };
		}
		if (s.type === 'first-in') {
			const seat = rangeStore.charts[dealt]?.seats[s.index];
			if (!seat) return null;
			if (seat.source === 'none')
				return { hand: s.hand, text: 'Big blind: check unless someone raises.' };
			const inRange = seat.hands.includes(s.hand);
			return {
				hand: s.hand,
				text: `${inRange ? 'In' : 'Outside'} the ${seat.name} range at ${dealt} players (${seat.pct}% of hands): ${inRange ? 'raise' : 'fold'}.`
			};
		}
		const spot = rangeStore.spots[spotKey(s.type, s.hero, s.villain)];
		if (!spot) return null;
		const raise = (spot.value[s.hand] ?? 0) + (spot.bluff[s.hand] ?? 0);
		const call = spot.call[s.hand] ?? 0;
		const fold = Math.max(0, 1 - raise - call);
		const parts = [
			raise > 0.004 && `${s.type === 'vs_open' ? '3-bet' : '4-bet'} ${Math.round(raise * 100)}%`,
			call > 0.004 && `call ${Math.round(call * 100)}%`,
			fold > 0.004 && `fold ${Math.round(fold * 100)}%`
		].filter(Boolean);
		const where =
			s.type === 'vs_open'
				? `${s.hero} vs an open from ${s.villain}`
				: `${s.hero} opened, ${s.villain} 3-bet`;
		return { hand: s.hand, text: `${where} (6-max chart): ${parts.join(' · ')}.` };
	});
</script>

{#if hint}
	<p
		class="max-w-2xl rounded-md border border-slate-700 bg-slate-900/70 px-3 py-1 text-center text-xs text-slate-300"
		data-testid="chart-hint"
	>
		<span class="font-bold text-amber-400">{hint.hand}</span> · {hint.text}
	</p>
{/if}

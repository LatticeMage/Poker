<!-- Scrolling hand history; follows the newest line. -->

<script lang="ts">
	import type { LogLine } from '$lib/stores/game.store';

	let { lines }: { lines: LogLine[] } = $props();

	let box: HTMLDivElement | undefined = $state();

	$effect(() => {
		// Re-run whenever a line is added.
		void lines.length;
		if (box) box.scrollTop = box.scrollHeight;
	});
</script>

<div
	bind:this={box}
	class="h-full overflow-y-auto px-3 py-2 text-[13px] leading-snug"
	role="log"
	aria-label="Hand history"
>
	{#each lines as line (line.id)}
		{#if line.kind === 'hand'}
			<div class="mt-3 mb-1 border-t border-slate-700 pt-2 font-bold text-slate-100 first:mt-0">
				{line.text}
			</div>
		{:else}
			<div
				class={{
					'text-slate-300': line.kind === 'action',
					'text-emerald-300': line.kind === 'street',
					'font-semibold text-amber-300': line.kind === 'win',
					'text-slate-400': line.kind === 'info'
				}}
			>
				{line.text}
			</div>
		{/if}
	{/each}
</div>

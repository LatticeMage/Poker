<!-- Native <dialog> modal: Esc and a click on the backdrop close it. -->

<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		open = $bindable(false),
		title,
		children
	}: { open: boolean; title: string; children: Snippet } = $props();

	let dialog: HTMLDialogElement | undefined = $state();

	$effect(() => {
		if (!dialog) return;
		if (open && !dialog.open) dialog.showModal();
		else if (!open && dialog.open) dialog.close();
	});
</script>

<dialog
	bind:this={dialog}
	class="m-auto w-[min(34rem,calc(100vw-1.5rem))] rounded-2xl border border-slate-700 bg-slate-900 p-0 text-slate-100 shadow-2xl backdrop:bg-black/60"
	onclose={() => (open = false)}
	onclick={(event) => {
		if (event.target === dialog) open = false;
	}}
>
	{#if open}
		<header class="flex items-center justify-between border-b border-slate-700 px-4 py-3">
			<h2 class="text-base font-bold">{title}</h2>
			<button
				class="grid h-8 w-8 place-items-center rounded-lg text-lg text-slate-300 hover:bg-slate-800"
				onclick={() => (open = false)}
				aria-label="Close"
			>
				✕
			</button>
		</header>
		<div class="max-h-[calc(100dvh-8rem)] overflow-y-auto p-4">
			{@render children()}
		</div>
	{/if}
</dialog>

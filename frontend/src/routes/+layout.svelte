<!--routes/+layout.svelte-->

<script lang="ts">
	import '../app.css';

	let { children } = $props();
</script>

<svelte:head>
	<title>Robo Hold'em</title>
	<meta name="description" content="No-limit Texas Hold'em against robot opponents." />
</svelte:head>

{@render children()}

// svelte.config.js
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),

	compilerOptions: {
		runes: true
	},

	kit: {
		// Client-only game: build a static SPA.
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			fallback: 'index.html',
			precompress: false,
			strict: true
		}),

		appDir: '_app',

		// GitHub Pages serves a project site under /<repo>; the build sets BASE_PATH for that.
		paths: { base: process.env.BASE_PATH ?? '' }
	}
};

export default config;

// vite.config.ts
import { version } from './package.json';

import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vitest/config';
import { sveltekit } from '@sveltejs/kit/vite';

// Single nginx port — front + back are both reachable through it. The vite
// dev server binds NGX_PORT inside its container so nginx can proxy to it.
const port = parseInt(process.env.NGX_PORT ?? '15173');

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	server: {
		host: '0.0.0.0',
		port,
		strictPort: true,
		allowedHosts: true
	},
	define: {
		'import.meta.env.VITE_APP_VERSION': JSON.stringify(version)
	},
	test: {
		environment: 'node',
		include: ['src/**/*.{test,spec}.ts'],
		testTimeout: 60_000
	}
});

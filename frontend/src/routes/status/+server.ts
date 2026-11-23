import { json } from '@sveltejs/kit';

// Health endpoint for the docker-compose healthcheck; prerendered so it also
// exists in the static build.
export const prerender = true;

export function GET() {
	return json({
		status: 'ok',
		version: import.meta.env.VITE_APP_VERSION
	});
}

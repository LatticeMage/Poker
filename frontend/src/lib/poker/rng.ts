// Randomness for the game engine. Every function takes the generator, so a test can pass a seeded
// one and get the same hand every time; the app passes Math.random.

/** A float in [0, 1). */
export type Rng = () => number;

export const systemRng: Rng = Math.random;

/** Small seedable generator for reproducible tests and for the range statistics. */
export function mulberry32(seed: number): Rng {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** Integer in [0, n). */
export function randrange(n: number, rng: Rng): number {
	return Math.floor(rng() * n);
}

/** Fisher–Yates shuffle, in place. */
export function shuffle<T>(items: T[], rng: Rng): T[] {
	for (let i = items.length - 1; i > 0; i--) {
		const j = randrange(i + 1, rng);
		[items[i], items[j]] = [items[j], items[i]];
	}
	return items;
}

/** `k` distinct items chosen at random (the input is not modified). */
export function sample<T>(items: readonly T[], k: number, rng: Rng): T[] {
	const pool = items.slice();
	const out: T[] = [];
	for (let i = 0; i < k && pool.length > 0; i++) {
		const j = randrange(pool.length, rng);
		out.push(pool[j]);
		pool[j] = pool[pool.length - 1];
		pool.pop();
	}
	return out;
}

/** CRC-32 of a string: a stable seed derived from a cache key. */
export function crc32(text: string): number {
	let crc = 0xffffffff;
	for (let i = 0; i < text.length; i++) {
		crc ^= text.charCodeAt(i);
		for (let k = 0; k < 8; k++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
	}
	return (crc ^ 0xffffffff) >>> 0;
}

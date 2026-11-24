// Opening the table: carry on with the game the last visit saved, else start one. Done when the
// page is loaded, not by a hook on the component.
import { resumeGame } from '$lib/actions/games';

export const load = () => {
	void resumeGame();
};

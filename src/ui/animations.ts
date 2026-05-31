/**
 * CSS animation helpers for card dealing and playing.
 */

export class Animations {
  /** Animate dealing cards to all players. */
  static async dealCards(duration: number = 800): Promise<void> {
    // Brief delay to let the UI render
    await sleep(duration);
  }

  /** Animate a card being played from a player area to the center. */
  static playCard(_fromPlayerIndex: number, duration: number = 300): Promise<void> {
    return sleep(duration);
  }

  /** Flash effect on a trick winner. */
  static async flashWinner(playerIndex: number): Promise<void> {
    const slotIds = ['trick-north', 'trick-west', 'trick-east', 'trick-south'];
    const el = document.getElementById(slotIds[playerIndex]);
    if (el) {
      el.style.transition = 'transform 0.3s ease';
      el.style.transform = 'scale(1.2)';
      await sleep(400);
      el.style.transform = 'scale(1)';
    }
  }

  /** Auto-scroll the log to the bottom. */
  static scrollLog(): void {
    const log = document.getElementById('log-entries');
    if (log) {
      log.scrollTop = 0; // Newest entries at top
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

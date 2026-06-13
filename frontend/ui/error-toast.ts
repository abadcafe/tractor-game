/** Default duration (ms) before a toast auto-removes. */
const TOAST_DURATION_MS = 3000;

/** Maximum number of visible toasts at once. Excess toasts are removed (oldest first). */
const MAX_VISIBLE_TOASTS = 3;

/**
 * Display a transient error message as a toast notification.
 * The toast is appended to the given container (defaults to #app)
 * and auto-removes after TOAST_DURATION_MS.
 * Limits visible toasts to MAX_VISIBLE_TOASTS to prevent spam.
 */
export function showErrorToast(message: string, container?: Element): void {
  const target = container ?? (typeof document !== "undefined" ? document.querySelector("#app") : null);
  if (!target) {
    console.warn(`showErrorToast: no container found for message "${message}"`);
    return;
  }

  const doc = target.ownerDocument ?? (typeof document !== "undefined" ? document : null);
  if (!doc) {
    console.warn(`showErrorToast: no document available for message "${message}"`);
    return;
  }

  // Remove excess toasts (oldest first) to prevent spam
  const existing = target.querySelectorAll(".error-toast");
  while (existing.length >= MAX_VISIBLE_TOASTS) {
    existing[0].remove();
  }

  const toast = doc.createElement("div");
  toast.className = "error-toast";
  toast.textContent = message;
  target.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, TOAST_DURATION_MS);
}

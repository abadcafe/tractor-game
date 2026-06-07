/**
 * Display a transient error message as a toast notification.
 * The toast is appended to the given container (defaults to #app)
 * and auto-removes after 3 seconds.
 */
export function showErrorToast(message: string, container?: Element): void {
  const target = container ?? (typeof document !== "undefined" ? document.querySelector("#app") : null);
  if (!target) return;

  const doc = target.ownerDocument ?? (typeof document !== "undefined" ? document : null);
  if (!doc) return;

  const toast = doc.createElement("div");
  toast.className = "error-toast";
  toast.textContent = message;
  target.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, 3000);
}

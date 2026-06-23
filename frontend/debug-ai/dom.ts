export function actionButton(
  label: string,
  onClick: () => void,
): HTMLButtonElement {
  const button = document.createElement("button");
  button.className = "copy-btn";
  button.textContent = label;
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    onClick();
  });
  return button;
}

export function copyButton(
  label: string,
  getText: () => string,
): HTMLButtonElement {
  const button = document.createElement("button");
  button.className = "copy-btn";
  button.textContent = label;
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    void copyText(getText(), button, label);
  });
  return button;
}

export function textSpan(value: string): HTMLSpanElement {
  const span = document.createElement("span");
  span.textContent = value;
  return span;
}

export function requiredElement<T extends HTMLElement>(
  id: string,
  ctor: { new (): T },
): T {
  const element = document.getElementById(id);
  if (!(element instanceof ctor)) {
    throw new Error(`missing element #${id}`);
  }
  return element;
}

async function copyText(
  text: string,
  button: HTMLButtonElement,
  label: string,
): Promise<void> {
  if (text === "") return;
  try {
    if (navigator.clipboard && globalThis.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      fallbackCopy(text);
    }
    button.textContent = "已复制";
    setTimeout(() => {
      button.textContent = label;
    }, 900);
  } catch (error: unknown) {
    console.warn("copy failed", error);
    button.textContent = "复制失败";
    setTimeout(() => {
      button.textContent = label;
    }, 1200);
  }
}

function fallbackCopy(text: string): void {
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.left = "-9999px";
  area.style.top = "0";
  document.body.appendChild(area);
  area.focus();
  area.select();
  const ok = document.execCommand("copy");
  area.remove();
  if (!ok) throw new Error("document.execCommand copy failed");
}

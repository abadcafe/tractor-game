/**
 * DOM utility functions for query and element creation.
 */

/**
 * Query a single element by CSS selector.
 * Returns null if not found.
 */
export function $<T extends Element = Element>(
  selector: string,
  parent?: ParentNode | null,
): T | null {
  return ((parent ?? document) as ParentNode).querySelector(selector) as T | null;
}

/**
 * Query all matching elements by CSS selector.
 */
export function $$<T extends Element = Element>(
  selector: string,
  parent?: ParentNode | null,
): NodeListOf<T> {
  return ((parent ?? document) as ParentNode).querySelectorAll(selector) as NodeListOf<T>;
}

/**
 * Create an HTML element with optional attributes and children.
 */
export function el(
  tag: string,
  attrs?: Record<string, string>,
  ...children: (string | Node)[]
): HTMLElement {
  const element = document.createElement(tag);
  if (attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      element.setAttribute(key, value);
    }
  }
  for (const child of children) {
    if (typeof child === "string") {
      element.appendChild(document.createTextNode(child));
    } else {
      element.appendChild(child);
    }
  }
  return element;
}

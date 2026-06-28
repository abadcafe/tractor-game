const STATIC_DIR = new URL("../static/", import.meta.url);
const VERSION = Date.now().toString(36);

async function main(): Promise<void> {
  for await (const path of walkStaticFiles(STATIC_DIR)) {
    if (path.endsWith(".js")) {
      await rewriteJavaScriptImports(path);
    } else if (path.endsWith(".html")) {
      await rewriteHtmlScripts(path);
    }
  }
}

async function* walkStaticFiles(
  directory: URL,
): AsyncGenerator<URL> {
  for await (const entry of Deno.readDir(directory)) {
    const child = new URL(entry.name, directory);
    if (entry.isDirectory) {
      yield* walkStaticFiles(new URL(`${entry.name}/`, directory));
    } else if (entry.isFile) {
      yield child.pathname;
    }
  }
}

async function rewriteJavaScriptImports(path: URL): Promise<void> {
  const source = await Deno.readTextFile(path);
  const rewritten = source.replaceAll(
    /((?:from\s*|import\s*\()\s*["'])(\.{1,2}\/[^"']+\.js)(["'])/g,
    (
      _match: string,
      prefix: string,
      specifier: string,
      suffix: string,
    ) => `${prefix}${withVersion(specifier)}${suffix}`,
  );
  if (rewritten !== source) {
    await Deno.writeTextFile(path, rewritten);
  }
}

async function rewriteHtmlScripts(path: URL): Promise<void> {
  const source = await Deno.readTextFile(path);
  const rewritten = source.replaceAll(
    /(src=")(\/[^"]+\.js)(?:\?v=[^"]*)?(")/g,
    (
      _match: string,
      prefix: string,
      specifier: string,
      suffix: string,
    ) => `${prefix}${withVersion(specifier)}${suffix}`,
  );
  if (rewritten !== source) {
    await Deno.writeTextFile(path, rewritten);
  }
}

function withVersion(specifier: string): string {
  return `${specifier}?v=${VERSION}`;
}

if (import.meta.main) {
  await main();
}

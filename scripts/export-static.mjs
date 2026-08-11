#!/usr/bin/env node
import { writeFile } from "node:fs/promises";

const workerUrl = new URL("../dist/server/index.js", import.meta.url);
workerUrl.searchParams.set("static-export", Date.now().toString());
const { default: worker } = await import(workerUrl.href);
const response = await worker.fetch(
  new Request("http://localhost/", { headers: { accept: "text/html" } }),
  { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
  { waitUntil() {}, passThroughOnException() {} },
);

if (!response.ok) throw new Error(`Static render failed with HTTP ${response.status}`);
const html = (await response.text()).replaceAll('/_next/', './_next/');
await writeFile(new URL("../dist/client/index.html", import.meta.url), html, "utf8");
console.log("Wrote dist/client/index.html for static hosting");

// Static server for the built demo under a sub-path, like the blog will host it:
//   node e2e/serve.mjs [port] [dir]   ->   http://127.0.0.1:<port>/echostream/demo/
// JSON / JS / CSS / HTML are sent gzip-encoded when the client accepts it (as GitHub Pages does);
// model.bin.gz is sent as is (the page decompresses it itself).
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize, resolve } from "node:path";
import { gzipSync } from "node:zlib";

const PREFIX = "/echostream/demo/";
const port = Number(process.argv[2] ?? 4178);
const root = resolve(process.argv[3] ?? new URL("../dist-demo", import.meta.url).pathname);
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
  ".svg": "image/svg+xml", ".gz": "application/gzip", ".md": "text/markdown; charset=utf-8", ".png": "image/png" };
const COMPRESS = new Set([".html", ".js", ".css", ".json", ".svg", ".md"]);

createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", "http://x");
  if (url.pathname === "/echostream/demo") return res.writeHead(301, { Location: PREFIX }).end();
  if (!url.pathname.startsWith(PREFIX)) return res.writeHead(404).end("not found");
  let rel = normalize(decodeURIComponent(url.pathname.slice(PREFIX.length))).replace(/^(\.\.[/\\])+/, "");
  if (rel === "" || rel === ".") rel = "index.html";
  const file = join(root, rel);
  try {
    if ((await stat(file)).isDirectory()) throw new Error("dir");
    let body = await readFile(file);
    const ext = extname(file);
    const headers = { "Content-Type": TYPES[ext] ?? "application/octet-stream", "Cache-Control": "no-cache" };
    if (COMPRESS.has(ext) && /\bgzip\b/.test(req.headers["accept-encoding"] ?? "")) {
      body = gzipSync(body);
      headers["Content-Encoding"] = "gzip";
    }
    res.writeHead(200, headers).end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
}).listen(port, "127.0.0.1", () => console.log(`serving ${root} at http://127.0.0.1:${port}${PREFIX}`));

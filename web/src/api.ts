// All network access lives here, so tests only need to mock `fetch`.

export const API: string = import.meta.env.VITE_API ?? "http://127.0.0.1:17000";

async function check(r: Response): Promise<Response> {
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return r;
}

/** fetch() only throws when the server is unreachable; say what to do about it. */
async function call(path: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(API + path, init);
  } catch {
    throw new Error(`cannot reach the API at ${API} (start the cluster: make demo)`);
  }
}

export async function getJSON<T>(path: string): Promise<T> {
  return (await check(await call(path))).json();
}

export async function postJSON<T>(path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method: "POST" };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  return (await check(await call(path, init))).json();
}

/** Static measured results from public/data/. Missing or unreadable file -> null. */
export async function loadData<T>(name: string): Promise<T | null> {
  try {
    const r = await fetch(`${import.meta.env.BASE_URL}data/${name}`);
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    return null; // e.g. the dev server answered with index.html instead of JSON
  }
}

export const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms));

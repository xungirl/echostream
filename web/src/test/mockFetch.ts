import { vi } from "vitest";

/** A minimal Response stand-in (status + json + text). */
export function reply(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

type Handler = (url: string, init?: RequestInit) => Response | undefined;

/** Install a fetch mock; unmatched URLs get a 404. Returns the vi.fn for call assertions. */
export function mockFetch(handler: Handler) {
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => handler(String(input), init) ?? reply({ detail: "not found" }, 404));
  vi.stubGlobal("fetch", fn);
  return fn;
}

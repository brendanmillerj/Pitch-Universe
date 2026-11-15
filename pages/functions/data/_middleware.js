// Guards /data/*: only the current deployment's data files are served. Data files are named by the hash of their own
// contents and cached forever, so an address from an older deployment could otherwise keep answering from
// Cloudflare's cache (2026-11-15: the pre-paywall files, with the Evolution data, stayed reachable at their old names
// on pitchomics.com and on pitchomics.pages.dev, which a zone purge can't reach). Running here, the Function answers
// before any cache: a name the current deployment doesn't have gets a plain 404, never the old bytes and never the
// site's home page (which Pages hands out for unknown paths).
export async function onRequest(context) {
  const res = await context.next();
  const type = res.headers.get("Content-Type") || "";
  if (!res.ok || type.startsWith("text/html")) {
    return new Response("Not found", { status: 404, headers: { "Content-Type": "text/plain", "Cache-Control": "no-store" } });
  }
  return res;
}

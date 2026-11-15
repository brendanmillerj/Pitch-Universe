// POST /api/logout -> clears the session cookie.
import { clearCookie, json } from "../_lib/session.js";

export async function onRequestPost() {
  return json({ ok: true }, 200, { "Set-Cookie": clearCookie() });
}

// any other method: refused, rather than falling through to the site's pages
export const onRequest = () => json({ error: "Use POST." }, 405, { Allow: "POST" });

// Pitch-omics paywall: shared pieces for the Cloudflare Pages Functions (user, 2026-11-15).
//
// A subscriber signs in with a one-time emailed link; the site then keeps a signed cookie. Nothing is stored on the
// server: tokens and cookies carry their own data and an HMAC-SHA256 signature made with SESSION_SECRET, which only
// Cloudflare knows. Who has paid is Buttondown's answer (its subscriber `type`), re-asked at least once a day.
//
// Secrets (wrangler pages secret put ...): SESSION_SECRET, BUTTONDOWN_API_KEY, RESEND_API_KEY.
// PAYWALL_MOCK=1 (local tests only): addresses starting "paid" count as paying, no email is sent, and the login
// reply carries the link so a test can follow it. Never set it on the live site.

export const COOKIE = "pitchomics_session";
export const SESSION_DAYS = 30;           // the cookie's life; then a new sign-in link
export const RECHECK_HOURS = 24;          // how stale Buttondown's "paid" may get before it is asked again
export const LINK_MINUTES = 15;           // a sign-in link's life
export const FROM = "Pitch-omics <login@pitchomics.com>";   // override with LOGIN_FROM if Resend verified another (sub)domain
// Buttondown's own definition: "premium, gifted, trialed, or churning have access to premium content".
// The newsletter uses the same list (scripts/newsletter.py PAID_STATUSES).
export const PAID_STATUSES = ["premium", "gifted", "trialed", "churning"];

const enc = new TextEncoder();
const now = () => Math.floor(Date.now() / 1000);

function b64url(bytes) {
  let s = "";
  for (const b of new Uint8Array(bytes)) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
function unb64url(s) {
  const t = atob(s.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((s.length + 3) % 4));
  return Uint8Array.from(t, c => c.charCodeAt(0));
}
async function key(secret) {
  if (!secret || secret.length < 32) throw new Error("SESSION_SECRET is missing or shorter than 32 characters");
  return crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]);
}

// a signed token: base64url(JSON) "." base64url(HMAC). `kind` keeps a sign-in link from being used as a session.
export async function sign(obj, secret) {
  const body = b64url(enc.encode(JSON.stringify(obj)));
  const sig = await crypto.subtle.sign("HMAC", await key(secret), enc.encode(body));
  return body + "." + b64url(sig);
}
export async function verify(token, secret, kind) {
  if (typeof token !== "string" || token.length > 2048) return null;
  const dot = token.indexOf(".");
  if (dot < 1) return null;
  const body = token.slice(0, dot);
  let ok = false;
  try {   // subtle.verify compares in constant time
    ok = await crypto.subtle.verify("HMAC", await key(secret), unb64url(token.slice(dot + 1)), enc.encode(body));
  } catch (e) { return null; }
  if (!ok) return null;
  let obj;
  try { obj = JSON.parse(new TextDecoder().decode(unb64url(body))); } catch (e) { return null; }
  if (!obj || obj.k !== kind || typeof obj.e !== "string" || !(obj.x > now())) return null;
  return obj;
}

export function normEmail(s) {
  const e = String(s || "").trim().toLowerCase();
  return e.length <= 254 && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e) ? e : null;
}

// Buttondown: does this address have access to paid content? true / false; throws if Buttondown can't answer.
export async function isPaid(email, env) {
  if (env.PAYWALL_MOCK === "1") return email.startsWith("paid");
  const r = await fetch("https://api.buttondown.com/v1/subscribers/" + encodeURIComponent(email), {
    headers: { Authorization: "Token " + env.BUTTONDOWN_API_KEY },
  });
  if (r.status === 404) return false;
  if (!r.ok) throw new Error("Buttondown answered " + r.status + ": " + (await r.text()).slice(0, 300));
  const s = await r.json();
  return PAID_STATUSES.includes(s.type);
}

// the sign-in email, through Resend (from login@pitchomics.com, a domain verified in Resend)
export async function sendLink(email, link, env) {
  if (env.PAYWALL_MOCK === "1") return;
  const text = "Sign in to Pitch-omics with this link. It works for " + LINK_MINUTES + " minutes.\n\n" + link +
    "\n\nIf you didn't ask to sign in, you can ignore this email.";
  const html = '<p style="font:16px/1.5 Georgia,serif;color:#1b2430">Sign in to Pitch-omics with this link. It works for ' +
    LINK_MINUTES + ' minutes.</p><p><a href="' + link + '" style="font:600 16px Helvetica,Arial,sans-serif;color:#1b2430;' +
    'background:#e2b13c;border:2px solid #1b2430;border-radius:6px;padding:10px 18px;text-decoration:none;display:inline-block">' +
    'Sign in to Pitch-omics</a></p><p style="font:13px/1.5 Helvetica,Arial,sans-serif;color:#576270">If you didn’t ask to sign in, ' +
    'you can ignore this email.</p>';
  const r = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: { Authorization: "Bearer " + env.RESEND_API_KEY, "Content-Type": "application/json" },
    body: JSON.stringify({ from: env.LOGIN_FROM || FROM, to: [email], subject: "Your Pitch-omics sign-in link", text, html }),
  });
  if (!r.ok) throw new Error("Resend answered " + r.status + ": " + (await r.text()).slice(0, 300));   // its reason, e.g. an unverified sender
}

// sign-in links and sessions
export const linkToken = (email, env) => sign({ k: "link", e: email, x: now() + LINK_MINUTES * 60 }, env.SESSION_SECRET);
export const sessionToken = (email, checked, expires, env) =>
  sign({ k: "s", e: email, c: checked, x: expires }, env.SESSION_SECRET);

export function cookieHeader(value, maxAge) {
  return `${COOKIE}=${value}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${maxAge}`;
}
export const clearCookie = () => cookieHeader("", 0);
export function readCookie(request) {
  const c = request.headers.get("Cookie") || "";
  for (const part of c.split(";")) {
    const i = part.indexOf("=");
    if (i > 0 && part.slice(0, i).trim() === COOKIE) return part.slice(i + 1).trim();
  }
  return null;
}

// The current subscriber, or null. When Buttondown's answer is older than RECHECK_HOURS it is asked again: a
// cancelled subscription ends the session (setCookie clears it), a still-paid one gets a refreshed cookie. If
// Buttondown can't answer, the session stands until the next try (no lock-out on their outage).
export async function currentSession(request, env) {
  const s = await verify(readCookie(request), env.SESSION_SECRET, "s");
  if (!s) return { session: null, setCookie: readCookie(request) ? clearCookie() : null };
  if (now() - s.c < RECHECK_HOURS * 3600) return { session: s, setCookie: null };
  let paid;
  try { paid = await isPaid(s.e, env); } catch (e) { return { session: s, setCookie: null }; }
  if (!paid) return { session: null, setCookie: clearCookie() };
  const fresh = await sessionToken(s.e, now(), s.x, env);
  return { session: { ...s, c: now() }, setCookie: cookieHeader(fresh, s.x - now()) };
}

export function json(obj, status = 200, extra = {}) {
  return new Response(JSON.stringify(obj), {
    status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store", ...extra },
  });
}
export { now };

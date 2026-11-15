// POST /api/login {email, back} -> emails a sign-in link if the address has a paid subscription.
// The reply is the same either way, so this form can't be used to learn who subscribes.
import { normEmail, isPaid, linkToken, sendLink, json } from "../_lib/session.js";

const SAME = { ok: true, message: "If that address has a subscription, a sign-in link is on its way." };

export async function onRequestPost({ request, env }) {
  let body = {};
  try { body = await request.json(); } catch (e) { /* empty or not JSON: treated as no email */ }
  const email = normEmail(body.email);
  if (!email) return json({ ok: false, message: "Enter the email address you subscribed with." }, 400);
  const back = typeof body.back === "string" && /^#[\w\-\/]{0,120}$/.test(body.back) ? body.back : "";
  let paid;
  try { paid = await isPaid(email, env); } catch (e) {
    return json({ ok: false, message: "The subscription check is down. Try again in a few minutes." }, 502);
  }
  if (!paid) return json(SAME);
  const link = new URL(request.url).origin + "/api/auth?t=" + encodeURIComponent(await linkToken(email, env)) +
    (back ? "&b=" + encodeURIComponent(back) : "");
  try { await sendLink(email, link, env); } catch (e) {
    return json({ ok: false, message: "The sign-in email could not be sent. Try again in a few minutes." }, 502);
  }
  return json(env.PAYWALL_MOCK === "1" ? { ...SAME, mock_link: link } : SAME);
}

// any other method: refused, rather than falling through to the site's pages
export const onRequest = () => json({ error: "Use POST." }, 405, { Allow: "POST" });

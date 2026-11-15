// GET /api/auth?t=<link token>&b=<#map address> -> sets the 30-day session cookie and returns to the page.
// A link works until it expires (LINK_MINUTES); a used or expired one sends the reader back with #signin-expired.
import { verify, sessionToken, cookieHeader, now, SESSION_DAYS } from "../_lib/session.js";

export async function onRequestGet({ request, env }) {
  const url = new URL(request.url);
  const b = url.searchParams.get("b") || "";
  const back = /^#[\w\-\/]{0,120}$/.test(b) ? b : "";
  const link = await verify(url.searchParams.get("t"), env.SESSION_SECRET, "link");
  if (!link) return Response.redirect(url.origin + "/#signin-expired", 302);
  const expires = now() + SESSION_DAYS * 86400;
  const token = await sessionToken(link.e, now(), expires, env);
  return new Response(null, {
    status: 302,
    headers: { Location: url.origin + "/" + back, "Set-Cookie": cookieHeader(token, SESSION_DAYS * 86400), "Cache-Control": "no-store" },
  });
}

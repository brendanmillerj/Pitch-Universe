// GET /api/me -> {paid, email}. The page asks this at boot to decide whether the Lab and Evolution are unlocked.
import { currentSession, json } from "../_lib/session.js";

export async function onRequestGet({ request, env }) {
  const { session, setCookie } = await currentSession(request, env);
  return json(session ? { paid: true, email: session.e } : { paid: false }, 200, setCookie ? { "Set-Cookie": setCookie } : {});
}

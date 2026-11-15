// Guards /p/*: the paid Evolution data (deploy_space.py --premium) goes only to a signed-in subscriber.
// Anyone else gets 401, whatever the page shows. Responses are private: no shared cache may keep them.
import { currentSession, json } from "../_lib/session.js";

export async function onRequest(context) {
  const { session, setCookie } = await currentSession(context.request, context.env);
  if (!session) {
    return json({ error: "Sign in with a Pitch-omics subscription to load this." }, 401, setCookie ? { "Set-Cookie": setCookie } : {});
  }
  const res = await context.next();
  const out = new Response(res.body, res);
  out.headers.set("Cache-Control", "private, no-store");
  if (setCookie) out.headers.append("Set-Cookie", setCookie);
  return out;
}

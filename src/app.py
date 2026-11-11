"""The website: a live demo page and a method-and-results page.
Run locally:  .venv/bin/streamlit run src/app.py
The page pauses itself after 10 minutes without a mouse or key touch: the browser drops its connection to the server and
stops reconnecting, so an abandoned tab costs nothing. Reloading the page starts again.
"""
import asyncio
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
IDLE_MINUTES = 10


def drop_connection():
    """Close this browser's connection from the server side. The page's own script stops it reconnecting."""
    try:
        from streamlit.runtime import get_instance
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        runtime, ctx = get_instance(), get_script_run_ctx()
        websocket = getattr(runtime.get_client(ctx.session_id), "_websocket", None)
        if websocket is not None:
            asyncio.run_coroutine_threadsafe(websocket.close(code=1000), runtime._get_async_objs().eventloop)
    except Exception:
        pass    # the browser closes the connection as well, so nothing is lost if this internal call ever changes


# The watcher runs inside the page itself (the small frame below only plants it there once). After IDLE_MINUTES with
# no touch it shows a "paused" cover, presses the hidden pause button so one last message goes to the server, closes the
# connection the moment that message is out, and takes away the page's means of reconnecting or pinging the server.
IDLE_WATCHER = """
(function () {
  if (window.__idlePause) return;
  window.__idlePause = true;
  var LIMIT = __MIN__ * 60 * 1000, last = Date.now();
  var touch = function () { last = Date.now(); };
  ["mousemove", "mousedown", "keydown", "touchstart", "wheel", "scroll"].forEach(function (e) {
    document.addEventListener(e, touch, {capture: true, passive: true});
  });
  function pause() {
    if (window.__paused) return;
    window.__paused = true;
    var WS = window.WebSocket, send = WS.prototype.send;
    WS.prototype.send = function () {                       // the next message reveals the live connection: send it, then hang up
      var r = send.apply(this, arguments);
      try { this.close(1000, "idle"); } catch (e) {}
      return r;
    };
    var dead = function () { throw new Error("paused after inactivity: reload the page to continue"); };
    dead.CONNECTING = 0; dead.OPEN = 1; dead.CLOSING = 2; dead.CLOSED = 3;
    window.WebSocket = dead; window.fetch = dead; window.XMLHttpRequest = dead;   // no reconnecting, no pinging
    var box = document.createElement("div");
    box.id = "idle-pause-cover";
    box.innerHTML = "<div style='position:fixed;inset:0;background:rgba(26,26,25,.93);color:#fff;z-index:2147483647;display:flex;align-items:center;justify-content:center;text-align:center;padding:2rem;font-family:sans-serif'>"
      + "<div><h2 style='margin:0 0 .6rem 0;font-weight:600'>Paused after __MIN__ minutes without activity</h2>"
      + "<p style='margin:0 0 1.4rem 0;max-width:34rem;line-height:1.5;color:#d9d8d3'>This free demo sleeps when nobody is using it, which is what keeps it free. Nothing is stored on the server.</p>"
      + "<button onclick='location.reload()' style='font-size:1rem;padding:.6rem 1.6rem;border-radius:8px;border:0;background:#2A78D6;color:#fff;cursor:pointer'>Continue</button></div></div>";
    document.body.appendChild(box);
    var b = document.querySelector(".st-key-idle_pause button");
    if (b) b.click();
  }
  window.__idlePauseNow = pause;
  setInterval(function () { if (Date.now() - last >= LIMIT) pause(); }, 15000);
})();
""".replace("__MIN__", str(IDLE_MINUTES))
PLANTER = "<script>(function(){var P=window.parent;if(!P||P.__idlePause)return;var s=P.document.createElement('script');" \
          "s.textContent=" + repr(IDLE_WATCHER) + ";P.document.head.appendChild(s);})();</script>"

st.set_page_config(page_title="Car damage AI", page_icon=":material/car_crash:", layout="wide")
# the hidden pause button comes first, so a pause request ends the run before any model work is done
st.html("<style>.st-key-idle_pause {display: none;}</style>")
with st.sidebar:
    if st.button("pause", key="idle_pause"):
        drop_connection()
        st.stop()
    components.html(PLANTER, height=0)

page = st.navigation([
    st.Page("demo.py", title="Live demo", icon=":material/photo_camera:", default=True),
    st.Page("research.py", title="Method and results", icon=":material/science:"),
])
page.run()

# shared footer at the bottom of the sidebar
with st.sidebar:
    st.divider()
    st.caption("Mohammad Adnan · November 2026")

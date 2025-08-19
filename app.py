import os, sys, json, time, base64, asyncio, threading
import pyaudio
from websockets.asyncio.client import connect
import streamlit as st
from streamlit_autorefresh import st_autorefresh

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 512

DEFAULT_HOST = "generativelanguage.googleapis.com"
DEFAULT_MODEL = "gemini-2.0-flash-live-001"

class RealtimeClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 on_log=None, on_error=None):
        self.api_key = api_key
        self.model = model
        self.host = host

        self.ws = None
        self.loop = None
        self._running = False

        self.audio_in_queue = None   
        self.out_queue = None        
        self.user_text_queue = None  

        self._pya_in = None
        self._pya_out = None
        self._mic_stream = None

        self.on_log = on_log or (lambda s: None)
        self.on_error = on_error or (lambda e: None)

    @property
    def uri(self):
        return (f"wss://{self.host}/ws/"
                f"google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
                f"?key={self.api_key}")

    async def _startup(self):
        setup_msg = {"setup": {"model": f"models/{self.model}"}}
        await self.ws.send(json.dumps(setup_msg))
        raw_response = await self.ws.recv(decode=False)
        _ = json.loads(raw_response.decode("ascii"))
        self.on_log("Setup complete")

    async def _send_user_texts(self):
        while self._running:
            text = await self.user_text_queue.get()
            if text is None:
                break
            msg = {
                "client_content": {
                    "turn_complete": True,
                    "turns": [{"role": "user", "parts": [{"text": text}]}],
                }
            }
            await self.ws.send(json.dumps(msg))
            self.on_log(f"Sent: {text}")

    def queue_text(self, text: str):
        if self.loop and self.user_text_queue:
            asyncio.run_coroutine_threadsafe(self.user_text_queue.put(text), self.loop)

    async def _listen_audio(self):
        self._pya_in = pyaudio.PyAudio()
        mic_info = self._pya_in.get_default_input_device_info()
        self._mic_stream = self._pya_in.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        self.on_log("Mic streaming…")
        while self._running:
            data = await asyncio.to_thread(self._mic_stream.read, CHUNK_SIZE)
            msg = {"realtime_input": {"media_chunks": [
                {"data": base64.b64encode(data).decode(), "mime_type": "audio/pcm"}]}}
            await self.out_queue.put(msg)

    async def _play_audio(self):
        self._pya_out = pyaudio.PyAudio()
        stream = self._pya_out.open(format=FORMAT, channels=CHANNELS,
                                    rate=RECEIVE_SAMPLE_RATE, output=True)
        try:
            while self._running:
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)
        finally:
            stream.stop_stream()
            stream.close()
            if self._pya_out:
                self._pya_out.terminate()
                self._pya_out = None

    async def _send_realtime(self):
        while self._running:
            msg = await self.out_queue.get()
            await self.ws.send(json.dumps(msg))

    async def _receive_audio(self):
        async for raw_response in self.ws:
            try:
                response = json.loads(raw_response.decode("ascii"))
            except Exception:
                self.on_log("Received non-JSON frame")
                continue

            try:
                b64data = response["serverContent"]["modelTurn"]["parts"][0]["inlineData"]["data"]
            except KeyError:
                pass
            else:
                pcm_data = base64.b64decode(b64data)
                self.audio_in_queue.put_nowait(pcm_data)

            if response.get("serverContent", {}).get("turnComplete"):
                self.on_log("End of turn — flushing buffered audio")
                while not self.audio_in_queue.empty():
                    try:
                        self.audio_in_queue.get_nowait()
                    except Exception:
                        break

    async def run(self):
        self._running = True
        self.loop = asyncio.get_running_loop()
        try:
            async with (
                await connect(self.uri, additional_headers={"Content-Type": "application/json"}) as ws,
                asyncio.TaskGroup() as tg,
            ):
                self.ws = ws
                await self._startup()

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                self.user_text_queue = asyncio.Queue()

                tg.create_task(self._send_realtime())
                tg.create_task(self._listen_audio())
                tg.create_task(self._receive_audio())
                tg.create_task(self._play_audio())
                tg.create_task(self._send_user_texts())

                while self._running:
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.on_error(e)
        finally:
            try:
                if self._mic_stream:
                    self._mic_stream.close()
                if self._pya_in:
                    self._pya_in.terminate()
                    self._pya_in = None
            except Exception:
                pass
            self._running = False
            self.on_log("Client stopped.")

    def stop(self):
        if self.loop:
            self._running = False
            if self.user_text_queue:
                asyncio.run_coroutine_threadsafe(self.user_text_queue.put(None), self.loop)
            if self.ws:
                asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)

class ClientManager:
    def __init__(self):
        self.client: RealtimeClient | None = None
        self.thread: threading.Thread | None = None
        self._running = False
        self._log_lines = []
        self._lock = threading.Lock()

    def _log(self, msg: str):
        with self._lock:
            ts = time.strftime("%H:%M:%S")
            self._log_lines.append(f"[{ts}] {msg}")
            if len(self._log_lines) > 800:
                self._log_lines = self._log_lines[-800:]

    def start(self, api_key: str, model: str):
        self.stop()
        self._log(f"Launching client (model={model})")
        self.client = RealtimeClient(
            api_key=api_key, model=model,
            on_log=self._log,
            on_error=lambda e: self._log(f"{type(e).__name__}: {e}"),
        )

        def runner():
            try:
                asyncio.run(self.client.run())
            except Exception as e:
                self._log(f"Fatal: {e}")
            finally:
                with self._lock:
                    self._running = False

        self.thread = threading.Thread(target=runner, daemon=True)
        self.thread.start()
        with self._lock:
            self._running = True

    def stop(self):
        if self.client:
            self._log("Stopping client…")
            try:
                self.client.stop()
            except Exception as e:
                self._log(f"Stop error: {e}")
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.5)
        self.client = None
        self.thread = None
        with self._lock:
            self._running = False

    def send_text(self, text: str):
        if self.client:
            self._log(f"User: {text}")
            self.client.queue_text(text)
        else:
            self._log("Client not running; cannot send.")

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_log_text(self) -> str:
        with self._lock:
            return "\n".join(self._log_lines)

@st.cache_resource(show_spinner=False)
def get_manager():
    return ClientManager()

st.set_page_config(page_title="Gemini Live", page_icon="", layout="wide")
st_autorefresh(interval=450, key="live-refresh")

if st.session_state.get("_clear_prompt", False):
    st.session_state["_clear_prompt"] = False
    st.session_state["prompt_input"] = ""



left, right = st.columns([6,4], gap="large")

with left:

    m = get_manager()

    prompt = st.text_input("Type a prompt and press Send", key="prompt_input")

    c1, c2, _ = st.columns([1,1,6])
    def on_send():
        text = st.session_state.get("prompt_input","").strip()
        if text:
            m.send_text(text)
        st.session_state["_clear_prompt"] = True
        st.rerun()

    def on_stop():
        m.stop()
        st.toast("Session stopped.")

    with c1:
        st.button("Send", use_container_width=True, type="primary", on_click=on_send)
    with c2:
        st.button("Stop", use_container_width=True, on_click=on_stop)

    st.write("---")
    st.markdown("##### Console Log")
    st.text_area(label="", value=m.get_log_text(), height=380, key="log_area")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### Session")
    api_key = st.text_input("API Key", value=os.getenv("GOOGLE_API_KEY",""), type="password", key="api_key")
    model = st.text_input("Model", value=DEFAULT_MODEL, key="model")

    def on_start():
        if not st.session_state.get("api_key"):
            st.toast("Isi API Key dulu ya bro.", icon="⚠️")
        else:
            get_manager().start(st.session_state["api_key"], st.session_state.get("model") or DEFAULT_MODEL)
            st.toast("Session started.", icon="✅")

    st.button("Start Session", use_container_width=True, on_click=on_start)

    st.write("")
    status = "Running" if get_manager().is_running() else "nothing"
    st.metric("Status", status)
    st.markdown('</div>', unsafe_allow_html=True)

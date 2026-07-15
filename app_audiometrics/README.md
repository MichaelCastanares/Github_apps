# Audio Metrics — SNR & WER

A Streamlit app that measures **signal-to-noise ratio** and **word error rate**
from a spoken passage. The user reads a displayed passage aloud (~60s); the app
records the audio, estimates SNR, transcribes with faster-whisper, scores WER
against the reference, and holds the results in memory until you export them.

## Setup

Requires **Python 3.10+** (3.13 recommended). Note: Python **3.9.7 specifically
will not work** — streamlit blacklists that exact patch release (`!=3.9.7`), so
`pip` can only resolve streamlit 1.12, which predates `st.audio_input`. Use any
other 3.10+ interpreter.

```bash
python3.13 -m venv ~/.venvs/audiometrics   # keep the venv OUT of Google Drive
source ~/.venvs/audiometrics/bin/activate
pip install -r requirements.txt            # or: uv pip install -r requirements.txt
streamlit run app.py
```

> Keep the virtualenv outside any Google Drive / synced folder — Drive locks
> files mid-sync and breaks venv creation/removal.

The first run downloads the Whisper model (`small`, ~500 MB). Mic recording
needs a browser; `http://localhost` works, but any cloud deployment must serve
over **HTTPS** — `getUserMedia` (the recorder's mic access) requires a secure
context.

## How it works

| Stage | Method |
|-------|--------|
| Passages | `passages.json` — each keyed passage holds a list of `lines`. Shown line-by-line; joined into one string as the WER reference. |
| Recording | Custom component (`components/recorder/`) — browser WebAudio capture + a paced reading guide in one iframe. One button starts mic recording *and* the line-by-line guide together; both auto-stop at 60s. Audio is encoded to a 16 kHz mono WAV in JS and returned to Python as base64. |
| SNR | Energy-based: frame the signal, split frames into speech vs. noise at the geometric-mean midpoint of the quiet floor and loud peak, `SNR = 10·log10((P_speech − P_noise) / P_noise)`. Relies on natural pauses in the reading. |
| WER | `faster-whisper` transcribes → normalized (lowercase, punctuation-stripped) → `jiwer`. |
| Storage | **None — nothing touches disk.** A session begins when you enter the password; readings you save accumulate in memory (`st.session_state`) for that login only. Export writes them to a zip you download. Refreshing, logging in again, or closing the tab discards them. |

## Adding passages

Edit `passages.json`:

```json
{
  "my_passage": {
    "title": "My Passage",
    "lines": ["First sentence.", "Second sentence."]
  }
}
```

## Deploy to Streamlit Community Cloud

1. **Push this folder to a GitHub repo.** These deploy files are already in place:
   - `requirements.txt` — Python deps
   - `packages.txt` — system libs (`libsndfile1`, `ffmpeg`) for audio decoding
   - `.python-version` → `3.11` (avoids the 3.9.7 streamlit blacklist)
   - `.gitignore` — keeps stray local data out of the repo
2. Go to **share.streamlit.io** → *New app* → pick the repo, branch, and
   `app.py` as the entrypoint.
3. In **Advanced settings**, set **Python version to 3.11+** and add your
   **secrets** (also settable later under *Manage app → Settings → Secrets*):
   ```toml
   APP_PASSWORD = "your-strong-password"   # gates the app; omit to run open
   # WHISPER_MODEL_SIZE = "base"           # optional: smaller model if RAM is tight
   ```
4. Deploy. First boot is slow (it downloads the Whisper model); later loads are
   fast. HTTPS is automatic, so the microphone works.

### Password protection

The app reads `APP_PASSWORD` from `st.secrets` (or the `APP_PASSWORD` env var).
When set, visitors must enter it before the app loads; when unset, the app runs
open (handy for local dev). For local testing, copy
`.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` (git-ignored) and
set the value. This is a single shared password — fine for limiting access, not
a substitute for per-user accounts.

> ⚠️ **Nothing is persisted.** Your readings live in memory for the current
> login only. Use **📥 Data export** in the sidebar to download them as a zip
> (`sessions.csv` + `recordings/*.wav`) **before** you refresh or close the tab —
> otherwise they're gone. This is deliberate: the container's disk is wiped on
> every restart anyway, and a shared on-disk store would let one user's export
> carry away another user's voice recordings. For durable multi-session storage,
> add a real backend (Postgres + S3/GCS) with per-user scoping.

Because the recordings are voice data (PII), consider adding access control
(Streamlit's built-in authentication) before sharing the URL widely. The single
shared `APP_PASSWORD` gates access but does not distinguish between users.

## Notes / roadmap

- **SNR** is a simple energy estimate — upgrade to VAD-based or WADA-SNR later
  without touching the UI (swap `compute_snr`). Exported WAVs allow re-analysis.
- **Whisper model size** is the main cloud-cost knob (`WHISPER_MODEL_SIZE` in
  `app.py`). `small` runs ~real-time on CPU; `medium`/`large` want a GPU.
- **No cross-session history.** Each login starts empty by design. If you need
  data to survive a logout, add a backend that scopes rows to an authenticated
  user — don't reintroduce a shared local SQLite file.

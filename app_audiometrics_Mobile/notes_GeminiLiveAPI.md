 gemini-2.5-flash-native-audio-preview-12-2025 is a Live API model

  It's not a batch transcription endpoint. Per Google's model page (https://ai.google.dev/gemi
  ni-api/docs/models/gemini-2.5-flash-native-audio-preview-12-2025), it's the Live API model —
  low-latency bidirectional voice dialog over a WebSocket, with 30 HD voices and
  emotion-aware responses. Its output modality is audio; there's a known issue
  (https://github.com/livekit/agents/issues/4423) that you can't even ask it for text-only
  responses. It's built to converse, not to transcribe a file you hand it.

  You can still get a transcript out of it, via the input_audio_transcription feature — the
  Live API will hand you an ASR transcript of whatever audio you stream in. So the path
  exists. But it means opening a WebSocket, streaming your 60-second WAV as if it were live
  mic input, collecting the input-transcription events, and discarding the spoken reply the
  model generates (which you pay for). There are also active reliability complaints on this
  preview — one forum thread reports code=1011 disconnects mid-turn at ~80% rate
  (https://discuss.ai.google.dev/t/gemini-live-api-gemini-2-5-flash-native-audio-preview-12-20
  25-returns-code-1011-mid-turn-at-80-rate-started-2026-05-27/167186).

  If your goal is "transcribe a recorded WAV," the right Gemini model is plain
  gemini-2.5-flash, which takes audio through the ordinary generateContent call. I'll show
  both.

  The seam is transcribe()
  
  Whichever you pick, this is a genuinely clean swap. Nothing else in app.py touches Whisper —
  process_recording() calls transcribe(wav_bytes) and hands the string to compute_wer(). So
  you replace one function and delete get_whisper_model().

  Option A — gemini-2.5-flash via generateContent (what I'd recommend)

  Your recorder already produces 16 kHz mono WAVs of ~1.9 MB, comfortably under the
  inline-data limit, so you don't even need the Files API:

  from google import genai
  from google.genai import types
  
  @st.cache_resource                                
  def get_genai_client():
      return genai.Client(api_key=get_setting("GEMINI_API_KEY"))

  def transcribe(wav_bytes):
      resp = get_genai_client().models.generate_content(
          model="gemini-2.5-flash",
          contents=[
              types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
              "Transcribe this recording verbatim. Output only the transcript.",
          ],
          config=types.GenerateContentConfig(
              temperature=0,
              thinking_config=types.ThinkingConfig(thinking_budget=0),
          ),
      )
      return (resp.text or "").strip()
  
  One HTTP call, synchronous, drops straight into Streamlit's execution model.

  Option B — the native-audio model via Live API (what you asked for)

  Handily, the Live API wants raw 16-bit PCM at 16 kHz mono, which is exactly what your
  recorder's encodeWAV() produces — so you just strip the 44-byte WAV header:

  import asyncio
  from google import genai
  from google.genai import types                    

  MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

  async def _transcribe_live(wav_bytes):
      client = genai.Client(api_key=get_setting("GEMINI_API_KEY"))
      config = {
          "response_modalities": ["AUDIO"],   # native-audio models cannot return text-only
          "input_audio_transcription": {},    # <- the ASR of *your* audio; this is the 
  payload
      }
      parts = []
      async with client.aio.live.connect(model=MODEL, config=config) as session:
          await session.send_realtime_input(        
              audio=types.Blob(data=wav_bytes[44:],            # strip WAV header -> raw PCM
                               mime_type="audio/pcm;rate=16000")
          )
          await session.send_realtime_input(audio_stream_end=True)
          async for response in session.receive():
              sc = response.server_content
              if sc and sc.input_transcription and sc.input_transcription.text:
                  parts.append(sc.input_transcription.text)
              if sc and sc.turn_complete:
                  break
      return "".join(parts).strip()                 

  def transcribe(wav_bytes):
      return asyncio.run(_transcribe_live(wav_bytes))

  Note you're paying for an audio reply you immediately throw away, and asyncio.run() inside a
  Streamlit rerun is workable but adds a failure mode Option A simply doesn't have.

  The thing that actually worries me

  Your app's entire purpose is measuring word error rate. An LLM-based transcriber is a poor
  instrument for that, in a way that's easy to miss because the numbers still look plausible.

  Whisper does mostly-verbatim ASR. Gemini — especially a conversational native-audio model
  tuned for dialog — tends to tidy up as it goes: it silently repairs disfluencies, normalizes
  wording, and infers what you meant to say from context. Since your reference text is a
  well-known passage (the Rainbow and Grandfather passages are in every speech-pathology
  corpus on the internet, and therefore in the training data), a capable LLM can partially
  reconstruct the correct text even where the audio was unclear. That drives WER down for
  reasons that have nothing to do with the speaker's intelligibility or the room's noise —
  which is precisely the signal you're trying to measure. Your WER_ACCEPTABLE = 0.03 threshold
  was calibrated against Whisper's behavior and would not mean the same thing.

  The temperature=0 and the "verbatim" instruction above blunt this, but they don't eliminate
  it.

  Concretely: your existing readings scored 2.6%–25% WER on Whisper. Those numbers become
  incomparable the moment you switch engines. If you have any intention of comparing across
  the two, run both transcribers on the same WAVs first and check how far apart they land
  before you commit.

  Practical notes if you go ahead
  
  You'd add google-genai to requirements.txt and drop faster-whisper (which also removes the
  ~500 MB model download and most of your cold-start time and RAM pressure — a real win on
  Streamlit Cloud). The API key goes in .streamlit/secrets.toml as GEMINI_API_KEY, read via
  your existing get_setting(). You can also delete packages.txt's ffmpeg if nothing else needs
  it.

  The one non-technical consideration: this sends voice recordings to Google. Right now
  Whisper runs entirely locally, so audio never leaves the machine. Your README already flags
  the recordings as PII, and the login gate implies these are identifiable participants — so
  this is a change worth being deliberate about, and possibly worth reflecting in the consent
  text on the password screen.

  Happy to implement whichever you pick — including a version that runs both engines side by
  side so you can see the WER delta before committing.

  Sources:                                          
  - Gemini 2.5 Flash Live Preview — model page
  (https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-native-audio-preview-12-2025)
  - Gemini 2.5 Native Audio upgrade announcement
  (https://blog.google/products-and-platforms/products/gemini/gemini-audio-model-updates/)
  - livekit/agents #4423 — native-audio model can't use text modality
  (https://github.com/livekit/agents/issues/4423)
  - Forum: code=1011 mid-turn disconnects on this preview model
  (https://discuss.ai.google.dev/t/gemini-live-api-gemini-2-5-flash-native-audio-preview-12-20
  25-returns-code-1011-mid-turn-at-80-rate-started-2026-05-27/167186)

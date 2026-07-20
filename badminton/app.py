import os
import subprocess
import tempfile
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import streamlit as st
from scipy.signal import butter, filtfilt, find_peaks

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a Butterworth bandpass filter to isolate impact frequencies."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def isolate_smash_segments(wav_path, video_fps, min_sec_between_hits=0.6):
    """
    Parses audio to isolate the frame indices of multiple hitting/smash events.
    """
    # 1. Load and conform audio to Mono
    sample_rate, data = wav.read(wav_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
        
    # Normalize audio array dynamically between -1.0 and 1.0
    data = data / np.max(np.abs(data))

    # 2. Filter out non-impact frequencies (Keep 1kHz - 4kHz)
    filtered_data = butter_bandpass_filter(data, lowcut=1000, highcut=4000, fs=sample_rate)

    # 3. Compute Amplitude Envelope (Rectify and Smooth)
    abs_data = np.abs(filtered_data)
    
    # 20ms window for moving average smoothing
    window_len = int(sample_rate * 0.02) 
    window = np.ones(window_len) / window_len
    smoothed_envelope = np.convolve(abs_data, window, mode='same')

    # 4. Detect Peaks with Biomechanical Constraints
    min_samples_spacing = int(sample_rate * min_sec_between_hits)
    
    peaks, _ = find_peaks(
        smoothed_envelope, 
        distance=min_samples_spacing,
        prominence=0.03,      # Adjust based on noise floor (0.0 to 1.0)
        height=0.02          # Absolute minimum volume threshold
    )

    # 5. Convert audio sample indices to timestamps and video frame numbers
    smash_events = []
    for peak_sample in peaks:
        timestamp_sec = peak_sample / sample_rate
        frame_idx = int(timestamp_sec * video_fps)
        
        smash_events.append({
            "timestamp": timestamp_sec,
            "frame_index": frame_idx
        })

    return smash_events, smoothed_envelope

# Set up page configurations
st.set_page_config(page_title="Multi-Smash Video & Audio Analyzer", layout="wide")

st.title("🏸 Intelligent Smash Event Detection Dashboard")
st.markdown(
    "Upload a sports video to isolate hitting signatures using DSP transient analysis and map them directly to video frame indices."
)

# 1. File Uploader Component
uploaded_file = st.file_uploader(
    "Choose a video file...", type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_file is not None:
    # Create side-by-side columns for a clean dashboard layout
    col1, col2 = st.columns([1, 1])

    # Save the uploaded video bytes to a temporary file so FFmpeg/OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    # --- Video Property Extraction ---
    # Open the video via OpenCV to accurately query its intrinsic frame rate
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Fallback default if FPS metadata is corrupted or missing
    if video_fps == 0 or np.isnan(video_fps):
        video_fps = 30.0

    # --- Column 1: Video Playback & Metadata ---
    with col1:
        st.subheader("Video Media Player")
        st.video(video_path)
        
        # Display basic clip stats
        st.info(f"🎞️ **Video Metadata:** {video_fps:.2f} FPS | {total_frames} Total Frames")

    # --- Column 2: Audio Waveform Extraction & Multi-Peak Processing ---
    with col2:
        st.subheader("Synchronized Audio Waveform Analysis")

        with st.spinner("Processing audio track for impact events..."):
            temp_wav_path = os.path.join(
                tempfile.gettempdir(), "extracted_audio.wav"
            )

            # Extract the audio using FFmpeg
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                temp_wav_path,
            ]

            try:
                subprocess.run(
                    ffmpeg_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )

                # 1. Read the audio data for plotting
                sample_rate, data = wav.read(temp_wav_path)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)

                # Normalize raw audio array between -1.0 and 1.0 for unified visual mapping
                data = data / np.max(np.abs(data))

                # Build the plotting time axis
                duration = len(data) / sample_rate
                time_axis = np.linspace(0, duration, num=len(data))

                # 2. Run the DSP transient isolation pipeline
                smash_events, smoothed_envelope = isolate_smash_segments(temp_wav_path, video_fps)

                # 3. Generate the Matplotlib Waveform Figure
                fig, ax = plt.subplots(2,1, figsize=(10, 5))

                # Downsample by 10 for lightning-fast UI rendering speeds
                ax[0].plot(
                    time_axis[::10],
                    data[::10],
                    color="#2b5c8f",
                    alpha=0.5,
                    label="Audio Track Signal",
                )

                # Plot the smoothed envelope
                ax[1].plot(
                    time_axis[::10],
                    smoothed_envelope[::10],
                    color="#5cb85c",
                    alpha=0.7,
                    label="Smoothed Envelope",
                )

                # Plot vertical indicator lines for every detected smash occurrence
                for idx, event in enumerate(smash_events):
                    ax[0].axvline(
                        x=event["timestamp"],
                        color="#d9534f",
                        linestyle="--",
                        linewidth=1.5,
                        label="Isolated Smash Event" if idx == 0 else "",  # Prevent duplicate legend keys
                    )

                # Apply styling
                ax[0].set_xlabel("Time (seconds)", fontsize=10)
                ax[0].set_ylabel("Normalized Amplitude", fontsize=10)
                ax[1].set_xlabel("Time (seconds)", fontsize=10)
                ax[1].set_ylabel("Smoothed Envelope", fontsize=10)
                ax[0].grid(True, linestyle=":", alpha=0.5)
                ax[0].legend(loc="upper right")
                plt.tight_layout()

                # Render the plot inside Streamlit
                st.pyplot(fig)

                # 4. Present the localized event coordinates
                st.subheader("🎯 Isolated Hitting Segments")
                if smash_events:
                    # Map structural dictionary lists into a clean tabular data element
                    table_data = [
                        {
                            "Hit #": i + 1,
                            "Timestamp (s)": f"{e['timestamp']:.3f}s",
                            "Center Frame Index": e["frame_index"],
                        }
                        for i, e in enumerate(smash_events)
                    ]
                    st.dataframe(table_data, use_container_width=True)
                else:
                    st.warning("No smash events crossed your threshold parameters. Try adjusting the filter prominence.")

            except subprocess.CalledProcessError:
                st.error(
                    "FFmpeg failed to extract audio. Ensure your video contains a valid audio track."
                )
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
            finally:
                # Clean up the temporary WAV file safely
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)

else:
    st.info("Please upload a video file to begin the analysis.")
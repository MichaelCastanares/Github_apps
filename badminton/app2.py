import os
import subprocess
import tempfile
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import scipy.io.wavfile as wav
import streamlit as st
from scipy.signal import butter, filtfilt, find_peaks

# --- Initialize MediaPipe Pose Globally ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_angle(a, b, c):
    """Calculates the 3D angle between three joints (a -> b -> c)."""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def evaluate_smash_mechanics(shoulder, elbow, wrist, hip):
    """Evaluates structural requirements for a forehand smash."""
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    is_high_contact = wrist.y < shoulder.y

    if 155.0 <= elbow_angle <= 185.0 and is_high_contact:
        status = "Optimal Contact Point"
        color = (0, 255, 0)
    elif elbow_angle < 155.0 and is_high_contact:
        status = "Early/Late Contact (Bent Elbow)"
        color = (0, 165, 255)
    else:
        status = "Suboptimal Contact (Low Point)"
        color = (0, 0, 255)

    return status, elbow_angle, color

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def isolate_smash_segments(wav_path, video_fps, min_sec_between_hits=0.6):
    sample_rate, data = wav.read(wav_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    data = data / np.max(np.abs(data))

    filtered_data = butter_bandpass_filter(data, lowcut=1000, highcut=4000, fs=sample_rate)
    abs_data = np.abs(filtered_data)
    
    window_len = int(sample_rate * 0.02) 
    window = np.ones(window_len) / window_len
    smoothed_envelope = np.convolve(abs_data, window, mode='same')

    min_samples_spacing = int(sample_rate * min_sec_between_hits)
    peaks, _ = find_peaks(
        smoothed_envelope, 
        distance=min_samples_spacing,
        prominence=0.03,      
        height=0.02      
    )

    smash_events = []
    for peak_sample in peaks:
        timestamp_sec = peak_sample / sample_rate
        smash_events.append({
            "timestamp": timestamp_sec,
            "frame_index": int(timestamp_sec * video_fps)
        })
    return smash_events

# --- Streamlit Config ---
st.set_page_config(page_title="Kinematic Smash Segmenter", layout="wide")
st.title("🏸 Intelligent Smash Event Detection & Frame Segmenter")
st.markdown("Isolate impact transients, review wave structures, and automatically slice biomechanically tracked 30-frame datasets.")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    # Read Video Meta Attributes
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if video_fps == 0 or np.isnan(video_fps):
        video_fps = 30.0

    with col1:
        st.subheader("Source Media Playback")
        st.video(video_path)
        st.info(f"🎞️ **Properties:** {video_fps:.2f} FPS | {total_frames} Frames | {v_width}x{v_height}")

    with col2:
        st.subheader("Audio Transient Mapping")
        temp_wav_path = os.path.join(tempfile.gettempdir(), "extracted_audio.wav")

        # Extract Audio via FFmpeg
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-vn", 
            "-acodec", "pcm_s16le", "-ar", "16000", temp_wav_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        sample_rate, data = wav.read(temp_wav_path)
        if len(data.shape) > 1: data = data.mean(axis=1)
        data = data / np.max(np.abs(data))
        time_axis = np.linspace(0, len(data)/sample_rate, num=len(data))

        smash_events = isolate_smash_segments(temp_wav_path, video_fps)

        # Plot Output
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(time_axis[::10], data[::10], color="#2b5c8f", alpha=0.5)
        for idx, event in enumerate(smash_events):
            ax.axvline(x=event["timestamp"], color="#d9534f", linestyle="--", alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

        st.subheader("🎯 Detected Impact Points")
        if smash_events:
            table_data = [{
                "Segment ID": i + 1,
                "Center Frame": e["frame_index"],
                "Timestamp": f"{e['timestamp']:.2f}s"
            } for i, e in enumerate(smash_events)]
            st.dataframe(table_data, use_container_width=True)
        else:
            st.warning("No impacts detected.")

    # --- Section 3: 30-Frame Dataset Generation ---
    if smash_events:
        st.markdown("---")
        st.subheader("🎬 Biomechanical 30-Frame Video Slicing")
        st.markdown("Extract exactly 30 frames around each peak. The segment will auto-process through the MediaPipe Pose tracking layout.")
        
        if st.button("⚡ Generate & Process 30-Frame Segments", type="primary"):
            N_FRAMES = 30
            
            # Create a multi-column row to layout isolated video segments cleanly
            clip_cols = st.columns(len(smash_events))
            
            for idx, event in enumerate(smash_events):
                center = event["frame_index"]
                
                # Math window alignment centered on peak frame
                start_frame = center - (N_FRAMES // 2)
                end_frame = start_frame + N_FRAMES
                
                # Robust boundary management constraints
                if start_frame < 0:
                    start_frame = 0
                    end_frame = min(total_frames, N_FRAMES)
                elif end_frame > total_frames:
                    end_frame = total_frames
                    start_frame = max(0, total_frames - N_FRAMES)

                with clip_cols[idx]:
                    st.markdown(f"**Segment {idx + 1} (Frames {start_frame}-{end_frame})**")
                    
                    with st.spinner(f"Slicing segment {idx+1}..."):
                        # Set up temporary file wrappers for local compilation
                        raw_segment_out = os.path.join(tempfile.gettempdir(), f"raw_seg_{idx}.mp4")
                        final_h264_out = os.path.join(tempfile.gettempdir(), f"smash_seg_{idx}.mp4")
                        
                        # Open original video pointer
                        v_cap = cv2.VideoCapture(video_path)
                        v_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                        
                        # Initialize raw frame writer buffer
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        v_writer = cv2.VideoWriter(raw_segment_out, fourcc, video_fps, (v_width, v_height))
                        
                        current_f = start_frame
                        while v_cap.isOpened() and current_f < end_frame:
                            ret, frame = v_cap.read()
                            if not ret: break
                            
                            # MediaPipe Kinematic Tracking Pass
                            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            rgb_image.flags.writeable = False
                            results = pose.process(rgb_image)
                            rgb_image.flags.writeable = True
                            bgr_frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                            
                            try:
                                landmarks = results.pose_landmarks.landmark
                                shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                                elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                                wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                                hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                                
                                status, elbow_angle, text_color = evaluate_smash_mechanics(shoulder, elbow, wrist, hip)
                                cv2.putText(bgr_frame, f"Elbow: {int(elbow_angle)} Deg", (30, 50), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                                cv2.putText(bgr_frame, f"Status: {status}", (30, 95), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)
                            except Exception:
                                pass
                            
                            if results.pose_landmarks:
                                mp_drawing.draw_landmarks(bgr_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                            
                            v_writer.write(bgr_frame)
                            current_f += 1
                            
                        v_cap.release()
                        v_writer.release()
                        
                        # Fast FFmpeg H.264 Transcode pipeline for HTML5 component playback compatibility
                        ffmpeg_transcode = [
                            "ffmpeg", "-y", "-i", raw_segment_out, 
                            "-vcodec", "libx264", "-pix_fmt", "yuv420p", final_h264_out
                        ]
                        subprocess.run(ffmpeg_transcode, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                        
                        # Serve Video to dashboard layout
                        st.video(final_h264_out)
                        
                        # Add Binary File Download Link
                        with open(final_h264_out, "rb") as file:
                            st.download_button(
                                label=f"💾 Download Segment {idx + 1}",
                                data=file,
                                file_name=f"smash_segment_{idx + 1}.mp4",
                                mime="video/mp4"
                            )
                            
                        # Cleanup temp buffers
                        if os.path.exists(raw_segment_out): os.remove(raw_segment_out)
else:
    st.info("Please upload a video file to begin the analysis.")
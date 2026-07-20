import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
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
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


def evaluate_smash_mechanics(shoulder, elbow, wrist, hip):
    """Evaluates the structural requirements for a forehand smash."""
    # 1. Elbow Extension Angle
    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    # 2. Contact Point Height (Wrist relative to Shoulder)
    # In image coordinates, lower y means higher spatially
    is_high_contact = wrist.y < shoulder.y

    # 3. Contact Point Position (Wrist relative to Hip on horizontal plane)
    # Negative z indicates coordinates closer to the camera/in front
    is_in_front = wrist.z < shoulder.z

    # Determine accuracy probability based on biomechanical alignment thresholds
    # Ideal smash contact requires near-full elbow extension (160-180 degrees)
    if 155.0 <= elbow_angle <= 185.0 and is_high_contact:
        status = "Optimal Contact Point"
        color = (0, 255, 0)  # Green
    elif elbow_angle < 155.0 and is_high_contact:
        status = "Early/Late Contact (Bent Elbow)"
        color = (0, 165, 255)  # Orange
    else:
        status = "Suboptimal Contact (Low Point)"
        color = (0, 0, 255)  # Red

    return status, elbow_angle, color


# Open video file or webcam stream (Replace with your video path)
cap = cv2.VideoCapture("test3.mov")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Recolor image to RGB for MediaPipe processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make pose detection
    results = pose.process(image)

    # Recolor back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark

        # Extract key skeletal points for a right-handed smash
        # (Swap to LEFT_ prefixed landmarks if analyzing a left-handed player)
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        # Evaluate kinematic data
        status, elbow_angle, text_color = evaluate_smash_mechanics(
            shoulder, elbow, wrist, hip)

        # Render visual metrics overlay onto video frame
        cv2.putText(image, f"Elbow Angle: {int(elbow_angle)} Deg", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Smash Status: {status}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

    except Exception as e:
        pass

    # Draw skeletal wireframe connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66),
                                   thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230),
                                   thickness=2, circle_radius=2)
        )

    cv2.imshow('Skeletal Smash Analysis', image)

    # Break loop cleanly by pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

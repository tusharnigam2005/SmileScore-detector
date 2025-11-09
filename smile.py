import cv2
import mediapipe as mp
import time
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
smile_score = 0
smiling = False

# Mouth landmark indices (outer + inner)
MOUTH_INDICES = [61, 291, 78, 308, 13, 14]

def mouth_aspect_ratio(landmarks, w, h):
    # Vertical distance between top (13) and bottom (14)
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[61]
    right = landmarks[291]

    top_y, bottom_y = int(top.y * h), int(bottom.y * h)
    left_x, right_x = int(left.x * w), int(right.x * w)

    mouth_height = abs(bottom_y - top_y)
    mouth_width = abs(right_x - left_x)
    return mouth_height / mouth_width if mouth_width != 0 else 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape
    status = "No Face"
    color = (255, 255, 255)  # default white color


    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        mar = mouth_aspect_ratio(landmarks, w, h)
        


        # Detect smile based on mouth aspect ratio (tuned threshold)
        if mar > 0.08:
            status = "Smiling"
            smile_score += 1
            smiling = True
            color = (0, 255, 0)
        else:
            status = "Not Smiling"
            smiling = False
            color = (0, 0, 255)

        # Draw mouth landmarks
        for idx in MOUTH_INDICES:
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 2, color, -1)

    # Display score
    cv2.putText(frame, f"Smile Score: {smile_score}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Status: {status}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Smile Score Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

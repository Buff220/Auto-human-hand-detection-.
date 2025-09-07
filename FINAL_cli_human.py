import cv2
import numpy as np
import mss
import mediapipe as mp
import socket

# Init
ip = "127.0.0.1"
port = 4545
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def capture_region(x, y, w, h):
    with mss.mss() as sct, mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        monitor = {"top": y, "left": x, "width": w, "height": h}

        while True:
            # Capture screen
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process pose
            result = pose.process(img_rgb)

            if result.pose_landmarks:
                points = []
                for lm in result.pose_landmarks.landmark:
                    px, py = int(lm.x * w), int(lm.y * h)
                    points.append((px, py))

                # Calculate body center (average of all points)
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                cx = sum(x_coords) // len(x_coords)
                cy = sum(y_coords) // len(y_coords)

                # Draw for debug
                mp_draw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Send offset
                alpha = cy - h // 2
                beta = cx - w // 2
                
                try:
                    sock.sendto(f"{alpha},{beta}".encode(), (ip, port))
                    print(f"Sending: alpha={alpha}, beta={beta}")
                except Exception as e:
                    print("Socket error:", e)

            # Show window
            cv2.imshow("Pose Detection", img)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

x, y = (562, 278)
w, h = (797, 596)
capture_region(x, y, w, h)

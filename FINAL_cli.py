import cv2
import numpy as np
import mss
import mediapipe as mp
import socket

# Init
ip = "127.0.0.1"
port = 4545
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def capture_region(x, y, w, h):
    with mss.mss() as sct, mp_hands.Hands(max_num_hands=1) as hands:
        monitor = {"top": y, "left": x, "width": w, "height": h}

        while True:
            # Capture
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process hands
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    points = []
                    for lm in handLms.landmark:
                        px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * 1000)
                        points.append((px, py, pz))

                    # Center
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    cx = sum(x_coords) // len(x_coords)
                    cy = sum(y_coords) // len(y_coords)

                    # Send offset
                    alpha = cy - h // 2
                    beta = cx - w // 2
                    try:
                        sock.sendto(f"{alpha},{beta}".encode(), (ip, port))
                        print(f"Sending: alpha={alpha}, beta={beta}")
                    except Exception as e:
                        print("Socket error:", e)

            # Show window (must call waitKey!)
            if cv2.waitKey(20) & 0xFF == ord('q'):  # ESC to exit
                break

x, y = (562, 278)
w, h = (797, 596)
capture_region(x, y, w, h)

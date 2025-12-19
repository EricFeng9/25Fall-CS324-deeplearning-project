import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict

def gstreamer_pipeline(sensor_id=0, sensor_mode=4, capture_width=640, capture_height=360, display_width=1280,
                       display_height=720, framerate=60, flip_method=2) -> str:
    return f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, format=(string)NV12, framerate=(fraction){framerate}/1 ! nvvidconv flip-method={flip_method} ! video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"


class HandTracker:
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7,
                                            min_tracking_confidence=0.5)
        self.drawer = mp.solutions.drawing_utils

    def detect(self, frame: np.ndarray) -> List[Dict]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        hands = []
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                # 8: Index_Tip
                p8 = lm.landmark[8]
                pos = (int(p8.x * w), int(p8.y * h))

                # --- 启发式手势识别 ---
                # Tip	Finger Tip	指尖	
                # DIP	Distal Interphalangeal  靠近指尖的第一个关节
                # PIP	Proximal Interphalangeal  手指中间的那个关节
                # IP	Interphalangeal	拇指关节	
                # MCP	Metacarpophalangeal	 掌指关节	
                # 关键点索引:
                # Thumb: 4(Tip), 3(IP), 2(MCP)
                # Index: 8(Tip), 6(PIP)
                # Middle: 12(Tip), 10(PIP)
                # Ring: 16(Tip), 14(PIP)
                # Pinky: 20(Tip), 18(PIP)   

                fingers_up = []
                
                # 拇指判断 (垂直方向): Tip.y < IP.y , Tip.y < MCP.y , Tip.y < others' PIP.y 为 Up
                # 注意: 坐标系原点在左上角，y向下增大。所以 y 小是上方。
                if lm.landmark[4].y < lm.landmark[3].y and lm.landmark[4].y < lm.landmark[2].y and \
                lm.landmark[4].y < lm.landmark[6].y and lm.landmark[4].y < lm.landmark[10].y and \
                lm.landmark[4].y < lm.landmark[14].y and lm.landmark[4].y < lm.landmark[18].y:
                    fingers_up.append(4)
                
                # 其他四指 (Tip.y < PIP.y)
                for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                    if lm.landmark[tip].y < lm.landmark[pip].y:
                        fingers_up.append(tip)
                
                count = len(fingers_up)
                gesture = "UNKNOWN"

                # 逻辑分类
                # 1. Fist (Closed_Fist): 0 fingers up OR only thumb is slightly tricky (usually 0 for perfect fist)
                if count == 0:
                    gesture = "FIST"
                
                # 2. Pointing_Up (食指): 仅 Index(8) Up
                # 有时候大拇指可能会误判，所以放宽条件: 8 in up, 12/16/20 not in up
                elif 8 in fingers_up and 12 not in fingers_up and 16 not in fingers_up and 20 not in fingers_up:
                    gesture = "POINTING_UP"
                
                # 3. Victory (耶): Index(8) + Middle(12) Up
                elif 8 in fingers_up and 12 in fingers_up and 16 not in fingers_up and 20 not in fingers_up:
                    gesture = "VICTORY"

                # 4. Pinky_Up (小指): Pinky(20) Up, 8, 12, 16 Down
                elif 20 in fingers_up and 8 not in fingers_up and 12 not in fingers_up and 16 not in fingers_up:
                    gesture = "PINKY_UP"

                # 5. Thumb_Up (赞): Thumb(4) Up, 8, 12, 16, 20 Down
                elif 4 in fingers_up and 8 not in fingers_up and 12 not in fingers_up and 16 not in fingers_up and 20 not in fingers_up:
                    gesture = "THUMB_UP"
                
                else:
                    gesture = "UNKNOWN"


                hands.append({"pos": pos, "gesture": gesture})
                # 淡色骨骼
                self.drawer.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS,
                                           self.drawer.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1),
                                           self.drawer.DrawingSpec(color=(230, 230, 230), thickness=1, circle_radius=1))
        return hands

"""
手势贪吃蛇（进阶版：即时闯关 + 策略对战）
==================================================

玩法概述
----------------------------------
- 单人模式（闯关）：
    - 机制：共10关，每关独立计分。
    - 规则：在规定时间内需要吃到目标数量的食物。
    - 胜利：一旦达到目标分数 -> 立即通关进入下一关（无需等待时间结束）。
    - 失败：时间耗尽仍未达标 -> 游戏结束。
    - 物理：撞墙不死。自撞安全。
- 双人模式（对抗）：
    - 机制：红蓝对抗，开局先绑定两只手（玩家1/玩家2），倒计时后开始。
    - 胜利：抢先达到目标长度 或 时间结束分高者胜 。
    - 失败：身体长度小于3（被吃掉太多了，初始为6）。
    - 物理：互撞头平局；撞身吃尾巴；撞墙输；自撞安全。

-------------------------------------
依赖：pip install -r requirements.txt

"""

from __future__ import annotations
import random
import time
from typing import Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

# -------------------- 可调配置 --------------------
USE_JETSON = False

# --- 单人模式关卡配置 (10关) ---
SINGLE_LEVELS = [
    {"time": 60, "score": 5},   # Level 1
    {"time": 60, "score": 8},   # Level 2
    {"time": 50, "score": 8},   # Level 3
    {"time": 50, "score": 10},  # Level 4
    {"time": 50, "score": 12},  # Level 5
    {"time": 50, "score": 15},  # Level 6
    {"time": 40, "score": 15},  # Level 7
    {"time": 40, "score": 16},  # Level 8
    {"time": 30, "score": 16},  # Level 9
    {"time": 30, "score": 20},  # Level 10
]

# --- 双人模式配置 ---
DUAL_GAME_DURATION = 120    # 秒
DUAL_TARGET_LENGTH = 20     # 目标长度

# --- 基础参数 ---
PADDING_RATIO = 0.10
HAND_LOSS_GRACE = 2.0
BIND_DIST = 80          # 双人模式下的控制圈半径

PointF = Tuple[float, float]
PointI = Tuple[int, int]

def gstreamer_pipeline(sensor_id=0, sensor_mode=4, capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=30, flip_method=2) -> str:
    return f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, format=(string)NV12, framerate=(fraction){framerate}/1 ! nvvidconv flip-method={flip_method} ! video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

class HandTracker:
    """手部检测器封装类"""
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.7, min_tracking_confidence=0.5
        )
        self.drawer = mp.solutions.drawing_utils

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """仅返回坐标"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        hands = []
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                p = lm.landmark[8]  # 食指
                pos = (int(p.x * w), int(p.y * h))
                hands.append({"pos": pos})
                self.drawer.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
                cv2.circle(frame, pos, 10, (0, 255, 0), -1)
        return hands

class SnakeGame:
    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height

        self.pad_x = int(self.width * PADDING_RATIO)
        self.pad_bottom = int(self.height * PADDING_RATIO) * 2
        self.pad_top = 0

        self.segment_length = 18
        self.min_segments = 6
        self.head_radius = 12
        self.body_radius = 9
        self.food_radius = 12
        self.alpha = 0.35
        self.max_step = 42
        self.follow_deadzone = 5
        self.move_interval = 0.02

        self.mode = "single"
        self.num_snakes = 1
        self.snakes: List[List[PointF]] = []
        self.scores: List[int] = []
        self.game_overs: List[bool] = []
        self.target_pos: List[PointI] = []
        self.food: PointI = (0, 0)
        self.last_move_times: List[float] = []
        self.result_text: str = ""
        self.sub_text: str = ""

        self.current_level_idx = 0
        self.level_start_time = 0.0
        self.end_time = 0.0

        self.total_paused_time = 0.0
        self.pause_start_time = 0.0
        self.is_paused_now = False

        self.dual_winner = -1

        self.tracker = HandTracker()
        self.hand_slots = [None, None]
        self.binding_complete = False
        self.manual_paused = False
        self.auto_paused = False

        if USE_JETSON:
            self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._init_state()

    def play_bounds(self):
        return self.pad_x, self.width - self.pad_x - 1, self.pad_top, self.height - self.pad_bottom - 1

    def clamp(self, v, lo, hi): return max(lo, min(hi, v))
    def dist2(self, p1, p2): return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

    def clamp_to_play(self, x, y):
        xm, xM, ym, yM = self.play_bounds()
        # 允许手稍微出界，以便能把蛇头带到墙边
        margin = 50
        return self.clamp(x, xm - margin, xM + margin), self.clamp(y, ym - margin, yM + margin)

    def _init_state(self) -> None:
        self.current_level_idx = 0
        self.dual_winner = -1
        self._init_level()

    def _init_level(self) -> None:
        xm, xM, ym, yM = self.play_bounds()
        cx, cy = (xm + xM) // 2, (ym + yM) // 2
        span_x = xM - xm
        offsets = [0] if self.num_snakes == 1 else [-span_x // 4, span_x // 4]

        self.snakes = []
        for i in range(self.num_snakes):
            bx = cx + offsets[i]
            body = [(bx - j * self.segment_length, cy) for j in range(self.min_segments)]
            self.snakes.append(body)

        self.scores = [0] * self.num_snakes if self.mode == "single" else [self.min_segments] * self.num_snakes
        self.game_overs = [False] * self.num_snakes
        self.target_pos = [(cx, cy) for _ in range(self.num_snakes)]
        self.last_move_times = [0.0] * self.num_snakes
        self.food = self.generate_food()
        self.result_text = ""
        self.sub_text = ""
        self.level_start_time = time.time()
        self.end_time = 0.0

        self.total_paused_time = 0.0
        self.pause_start_time = 0.0
        self.is_paused_now = False

        self.hand_slots = [None, None]
        self.manual_paused = False
        self.auto_paused = False
        if self.mode == "dual":
            self.binding_complete = False

    def select_mode(self) -> bool:
        win = "Mode Select"
        canvas = np.zeros((450, 700, 3), np.uint8)
        while True:
            canvas[:] = 0
            cv2.putText(canvas, "AI Snake Game", (220, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            cv2.putText(canvas, "1: Single Player (Stage Mode)", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(canvas, "   - Reach Target Score to pass.", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(canvas, "   - Safe Wall: You won't die hitting walls.", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.putText(canvas, "2: Dual Player (Battle)", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            cv2.putText(canvas, "   - Hit Wall = LOSE immediately.", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(canvas, "   - Head hit Body -> Steal Tail.", (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.putText(canvas, "Q / Esc: Quit", (40, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("1"):
                self.mode = "single"
                self.num_snakes = 1
                self.binding_complete = True
                cv2.destroyWindow(win)
                return True
            if key == ord("2"):
                self.mode = "dual"
                self.num_snakes = 2
                self.binding_complete = False
                cv2.destroyWindow(win)
                return True
            if key in (ord("q"), ord("Q"), 27):
                cv2.destroyWindow(win)
                return False

    def bind_players(self, win: str) -> bool:
        self.hand_slots = [None, None]
        start_t = 0
        counting = False
        while True:
            ret, frame = self.cap.read()
            if not ret: return False
            frame = cv2.flip(frame, 1)
            hands = self.tracker.detect(frame)

            self._draw_snakes_simple(frame)
            cv2.putText(frame, "Binding (Place finger on head)", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            heads = [s[0] for s in self.snakes]
            self._assign_hands(hands, heads, time.time())

            if self._slot_active(0, time.time()) and self._slot_active(1, time.time()):
                if not counting:
                    counting = True
                    start_t = time.time()
                rem = 3 - (time.time() - start_t)
                cv2.putText(frame, f"Starting in {int(rem)+1}...", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                if rem <= 0:
                    self.binding_complete = True
                    self.level_start_time = time.time()
                    self.total_paused_time = 0.0
                    return True
            else:
                counting = False

            cv2.imshow(win, frame)
            k = cv2.waitKey(1)
            if k in (ord('q'), 27): return False
            if k == ord('r'): self.hand_slots = [None, None]; counting = False

    def _assign_hands(self, hands, heads, ts):
        if self.mode == "single":
            if not hands: return
            closest_hand = None
            min_d2 = float('inf')
            snake_head = heads[0]
            for h in hands:
                d2 = self.dist2(h["pos"], snake_head)
                if d2 < min_d2:
                    min_d2 = d2
                    closest_hand = h["pos"]
            if closest_hand:
                self.hand_slots[0] = {"pos": closest_hand, "ts": ts}
        else:
            cands = []
            for h in hands:
                pos = h["pos"]
                for i, hd in enumerate(heads):
                    d2 = self.dist2(pos, hd)
                    if d2 < (BIND_DIST * 1.5)**2:
                        cands.append((d2, i, pos))
            cands.sort(key=lambda x: x[0])
            used = set()
            for d2, i, p in cands:
                if i not in used:
                    if d2 < BIND_DIST**2:
                        self.hand_slots[i] = {"pos": p, "ts": ts}
                        used.add(i)

    def _slot_active(self, i, ts):
        s = self.hand_slots[i]
        return s and (ts - s["ts"] < HAND_LOSS_GRACE)

    def get_elapsed_time(self) -> int:
        if any(self.game_overs) and self.end_time > 0:
            raw_duration = self.end_time - self.level_start_time
            return int(max(0, raw_duration - self.total_paused_time))
        if self.is_paused_now:
            raw_duration = self.pause_start_time - self.level_start_time
            return int(max(0, raw_duration - self.total_paused_time))
        raw_duration = time.time() - self.level_start_time
        return int(max(0, raw_duration - self.total_paused_time))

    def generate_food(self):
        margin = 50
        xm, xM, ym, yM = self.play_bounds()
        while True:
            x = random.randint(xm + margin, xM - margin)
            y = random.randint(ym + margin, yM - margin)
            if not any(self.dist2((x,y), s[0]) < 2500 for s in self.snakes):
                return (x, y)

    def step_head(self, idx):
        hx, hy = self.snakes[idx][0]
        tx, ty = self.target_pos[idx]
        dx, dy = tx - hx, ty - hy
        d = (dx*dx + dy*dy)**0.5
        if d < self.follow_deadzone: return (hx, hy)
        step = min(self.max_step, d) * self.alpha
        return (hx + dx/d*step, hy + dy/d*step)

    def rebuild_body(self, idx, head):
        pts = [head]
        prev = head
        for i, seg in enumerate(self.snakes[idx][1:]):
            dx, dy = prev[0]-seg[0], prev[1]-seg[1]
            d = (dx*dx+dy*dy)**0.5
            if d < 1e-4: continue
            nx = prev[0] - dx/d * self.segment_length
            ny = prev[1] - dy/d * self.segment_length
            pts.append((nx, ny))
            prev = (nx, ny)
        self.snakes[idx] = pts

    def move_snake(self, idx):
        if self.game_overs[idx]: return
        if time.time() - self.last_move_times[idx] < self.move_interval: return
        self.last_move_times[idx] = time.time()

        new_head = self.step_head(idx)

        # --- 提前进行撞墙判定 ---
        x, y = new_head
        xm, xM, ym, yM = self.play_bounds()
        radius_buffer = self.head_radius

        # 判断是否出界
        hit_wall = (x - radius_buffer < xm) or (x + radius_buffer > xM) or \
                   (y - radius_buffer < ym) or (y + radius_buffer > yM)

        if hit_wall:
            if self.mode == "single":
                # 【修改点】单人模式：撞墙不死，强制将头限制在边界内（滑墙效果）
                # 这样蛇就会贴着墙走，不会因为手移出去了而判定失败
                nx = self.clamp(x, xm + radius_buffer, xM - radius_buffer)
                ny = self.clamp(y, ym + radius_buffer, yM - radius_buffer)
                new_head = (nx, ny)
            else:
                # 双人模式：撞墙直接判负
                self.game_overs[idx] = True
                self.end_time = time.time()
                self.dual_winner = 1 - idx
                self.game_overs[1-idx] = True
                self.result_text = "Green hit wall!" if idx == 0 else "Yellow hit wall!"
                return

        # 吃食物 (用可能修正过的 new_head)
        if self.dist2(new_head, self.food) < (self.head_radius + self.food_radius)**2:
            self.snakes[idx].append(self.snakes[idx][-1])
            self.food = self.generate_food()
            if self.mode == "single":
                self.scores[idx] += 1
                target = SINGLE_LEVELS[self.current_level_idx]["score"]
                if self.scores[idx] >= target:
                    self._next_level()
                    return

        self.rebuild_body(idx, new_head)

        # --- 双人互撞 ---
        if self.mode == "dual":
            other = 1 - idx
            for k, seg in enumerate(self.snakes[other]):
                if self.dist2(new_head, seg) < (self.head_radius + self.body_radius)**2:
                    if k < 2:
                        self.game_overs[0] = True
                        self.game_overs[1] = True
                        self.dual_winner = 2
                        self.result_text = "Head to Head! Draw!"
                        self.end_time = time.time()
                    else:
                        stolen_part = self.snakes[other][k:]
                        self.snakes[other] = self.snakes[other][:k]
                        self.snakes[idx].extend(stolen_part)

                    if len(self.snakes[other]) < 3:
                        self.game_overs[other] = True
                        self.game_overs[idx] = True
                        self.dual_winner = idx
                        self.result_text = "Elimination!"
                        self.end_time = time.time()
                    return

        if self.mode == "dual":
            self.scores[idx] = len(self.snakes[idx])
            if self.scores[idx] >= DUAL_TARGET_LENGTH:
                self.dual_winner = idx
                self.game_overs[0] = self.game_overs[1] = True
                self.result_text = "Target Score Reached!"
                self.end_time = time.time()

    def _next_level(self):
        if self.current_level_idx < len(SINGLE_LEVELS) - 1:
            self.current_level_idx += 1
            self._init_level()
            self.sub_text = f"Level {self.current_level_idx+1} Start!"
        else:
            self.game_overs[0] = True
            self.result_text = "VICTORY! All Levels Cleared!"
            self.end_time = time.time()

    def check_time_limit(self):
        if any(self.game_overs): return

        elapsed = self.get_elapsed_time()

        if self.mode == "single":
            limit = SINGLE_LEVELS[self.current_level_idx]["time"]
            if elapsed > limit:
                self.game_overs[0] = True
                self.result_text = "Time's Up! Level Failed."
                self.end_time = time.time()
        else:
            limit = DUAL_GAME_DURATION
            if elapsed > limit:
                self.game_overs[0] = self.game_overs[1] = True
                self.end_time = time.time()
                if self.scores[0] > self.scores[1]:
                    self.dual_winner = 0
                    self.result_text = "Time Up! Green Wins!"
                elif self.scores[1] > self.scores[0]:
                    self.dual_winner = 1
                    self.result_text = "Time Up! Yellow Wins!"
                else:
                    self.dual_winner = 2
                    self.result_text = "Time Up! Draw!"

    def draw(self, frame):
        canvas = np.zeros_like(frame)
        colors = [(0, 255, 0), (0, 200, 200)]

        xm, xM, ym, yM = self.play_bounds()
        cv2.rectangle(frame, (xm, ym), (xM, yM), (50,50,50), 2)

        cv2.circle(canvas, self.food, self.food_radius, (0, 0, 255), -1)

        for idx, snake in enumerate(self.snakes):
            if not snake: continue
            base = colors[idx]
            if len(snake) > 1:
                pts = np.array(snake, np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], False, base, 12, cv2.LINE_AA)
            for i, p in enumerate(snake):
                r = self.head_radius if i==0 else max(6, int(self.body_radius*(0.9-0.4*(i/len(snake)))))
                col = base if i==0 else (base[0], int(base[1]*0.8), base[2])
                cv2.circle(canvas, (int(p[0]), int(p[1])), r, col, -1)

            head_pt = (int(snake[0][0]), int(snake[0][1]))
            cv2.circle(canvas, head_pt, self.head_radius, (255,255,255), 2)

            if self.mode == "dual":
                cv2.circle(frame, head_pt, BIND_DIST, (200, 200, 200), 1, cv2.LINE_AA)
                if self._slot_active(idx, time.time()):
                     cv2.circle(frame, head_pt, BIND_DIST, base, 2, cv2.LINE_AA)

        cv2.addWeighted(canvas, 0.7, frame, 0.3, 0, frame)

        elapsed = self.get_elapsed_time()

        if self.mode == "single":
            cfg = SINGLE_LEVELS[self.current_level_idx]
            rem_time = max(0, cfg["time"] - elapsed)
            target = cfg["score"]
            current = self.scores[0]

            cv2.putText(frame, f"Level {self.current_level_idx+1}/10", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, f"Goal: {target}  Current: {current}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            time_col = (0,255,255) if rem_time > 10 else (0,0,255)
            cv2.putText(frame, f"Time: {rem_time}s", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, time_col, 2)
        else:
            rem_time = max(0, DUAL_GAME_DURATION - elapsed)
            cv2.putText(frame, f"P1(Green): {self.scores[0]}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"P2(Yellow): {self.scores[1]}", (self.width-250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,200), 2)
            cv2.putText(frame, f"Win Length: {DUAL_TARGET_LENGTH}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
            cv2.putText(frame, f"Time: {rem_time}s", (self.width//2-60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        if any(self.game_overs) and self.result_text:
            sz = cv2.getTextSize(self.result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cx = (self.width - sz[0]) // 2
            cy = self.height // 2
            cv2.putText(frame, self.result_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            cv2.putText(frame, "Press R to Restart, Q to Quit", (cx+20, cy+50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        elif self.sub_text:
            cv2.putText(frame, self.sub_text, (self.width//2-150, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 2)
            if time.time() - self.level_start_time > 2: self.sub_text = ""

        if (self.auto_paused or self.manual_paused) and not any(self.game_overs):
            cv2.putText(frame, "PAUSED (Hand Lost or Space)", (self.width//2-200, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    def _draw_snakes_simple(self, frame):
        colors = [(0, 255, 0), (0, 200, 200)]
        for idx, s in enumerate(self.snakes):
            for i, p in enumerate(s):
                cv2.circle(frame, (int(p[0]), int(p[1])), 10, colors[idx], -1)
            cv2.circle(frame, (int(s[0][0]), int(s[0][1])), BIND_DIST, (200,200,200), 1)

    def run(self):
        while True:
            if not self.select_mode(): break
            win = "Snake Game"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            self._init_state()

            if self.mode == "dual":
                if not self.bind_players(win):
                    cv2.destroyAllWindows()
                    continue

            while True:
                ret, frame = self.cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                now = time.time()

                hands = self.tracker.detect(frame)
                heads = [s[0] if s else (0,0) for s in self.snakes]
                self._assign_hands(hands, heads, now)

                active = [self._slot_active(i, now) for i in range(self.num_snakes)]
                self.auto_paused = False
                if self.mode == "single":
                    if not active[0]: self.auto_paused = True
                else:
                    if not all(active): self.auto_paused = True

                is_currently_paused = self.auto_paused or self.manual_paused

                if is_currently_paused and not self.is_paused_now:
                    self.pause_start_time = time.time()

                if not is_currently_paused and self.is_paused_now:
                    duration = time.time() - self.pause_start_time
                    self.total_paused_time += duration

                self.is_paused_now = is_currently_paused

                if not is_currently_paused and not any(self.game_overs):
                    self.check_time_limit()
                    for i in range(self.num_snakes):
                        if active[i]:
                            p = self.hand_slots[i]["pos"]
                            self.target_pos[i] = self.clamp_to_play(int(p[0]/frame.shape[1]*self.width), int(p[1]/frame.shape[0]*self.height))
                            self.move_snake(i)

                self.draw(frame)

                cv2.imshow(win, frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), ord('Q')):
                    cv2.destroyWindow(win)
                    break
                if k in (ord('r'), ord('R')):
                    self._init_state()
                    if self.mode == "dual":
                        if not self.bind_players(win): break
                if k == 32: self.manual_paused = not self.manual_paused

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    SnakeGame().run()
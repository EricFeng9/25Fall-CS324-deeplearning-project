"""
==================================================
玩法概述
----------------------------------
- 单人模式（简单模式闯关，困难模式食物只存在3秒）：
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

更新说明：
1.单人模式增加困难模式（食物三秒消失）；
2.不同类型食物得分不同，吃毒药死；
3.可以同时出现多个食物；
4.UI改了一下可以再调整.
-------------------------------------
依赖：pip install -r requirements.txt

"""

from __future__ import annotations
import random
import time
import math
from typing import Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -------------------- 配置与常量 --------------------
USE_JETSON = False

# 颜色定义 (RGB) - 马卡龙色系
COL_BG_BLUE = (224, 247, 250)   # 浅蓝背景
COL_SNAKE_BODY = (179, 229, 252) # 蛇身浅蓝
COL_SNAKE_BORDER = (100, 180, 220) # 蛇身边框
COL_HUD_BG = (255, 240, 245)    # HUD 背景浅粉
COL_PROGRESS_OK = (144, 238, 144) # 进度条绿
COL_PROGRESS_LOW = (255, 99, 71)  # 进度条红

# 食物配置
FOOD_TYPES = {
    'insect': {'score': 1, 'color': (165, 214, 167), 'prob': 0.5}, # 浅绿
    'fish':   {'score': 2, 'color': (144, 202, 249), 'prob': 0.3}, # 浅蓝
    'mouse':  {'score': 3, 'color': (255, 224, 130), 'prob': 0.15}, # 浅黄
    'poison': {'score': 0, 'color': (206, 147, 216), 'prob': 0.05}  # 浅紫
}

# 游戏参数
SINGLE_LEVELS = [
    {"time": 60, "score": 5}, {"time": 60, "score": 8},
    {"time": 50, "score": 8}, {"time": 50, "score": 10},
    {"time": 50, "score": 12}, {"time": 50, "score": 15},
    {"time": 40, "score": 15}, {"time": 40, "score": 16},
    {"time": 30, "score": 16}, {"time": 30, "score": 20},
]
DUAL_GAME_DURATION = 120
DUAL_TARGET_LENGTH = 30
PADDING_RATIO = 0.05
HAND_LOSS_GRACE = 2.0
BIND_DIST = 80
HARD_MODE_FOOD_TIME = 3.0 # 食物存在3秒
HUD_HEIGHT = 100

PointF = Tuple[float, float]
PointI = Tuple[int, int]

# -------------------- 辅助类 --------------------

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 8)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = 1.0
        self.decay = random.uniform(0.05, 0.1)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay

    def is_dead(self):
        return self.life <= 0

class FoodItem:
    def __init__(self, pos, ftype, spawn_time):
        self.pos = pos
        self.type = ftype
        self.spawn_time = spawn_time
        self.active = True

def get_safe_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "calibrib.ttf", "segoeui.ttf", "msyh.ttc", "simhei.ttf", "arial.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except IOError:
            continue
    return ImageFont.load_default()

def draw_text_pill(
    draw: ImageDraw.ImageDraw,
    pos: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 160),
    padding: int = 10,
) -> None:
    x, y = pos
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    bg_box = (x - padding, y - padding, x + w + padding, y + h + padding)
    draw.rounded_rectangle(bg_box, radius=8, fill=bg_color)
    draw.text((x, y), text, font=font, fill=text_color)

def gstreamer_pipeline(sensor_id=0, sensor_mode=4, capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=30, flip_method=2) -> str:
    return f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, format=(string)NV12, framerate=(fraction){framerate}/1 ! nvvidconv flip-method={flip_method} ! video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

class HandTracker:
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.drawer = mp.solutions.drawing_utils

    def detect(self, frame: np.ndarray) -> List[Dict]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        hands = []
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                p = lm.landmark[8]
                pos = (int(p.x * w), int(p.y * h))
                hands.append({"pos": pos})
                self.drawer.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS,
                    self.drawer.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=1),
                    self.drawer.DrawingSpec(color=(230,230,230), thickness=1, circle_radius=1))
        return hands

class SnakeGame:
    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height

        self.pad_x = int(self.width * PADDING_RATIO)
        self.pad_y = int(self.height * PADDING_RATIO)
        self.play_area = (self.pad_x, HUD_HEIGHT + 20, self.width - self.pad_x, self.height - self.pad_y)

        self.segment_length = 18
        self.min_segments = 6
        self.head_radius = 16
        self.body_radius = 14
        self.food_radius = 15
        self.alpha = 0.3
        self.max_step = 45
        self.follow_deadzone = 5
        self.move_interval = 0.02

        self.font_title = get_safe_font(65)
        self.font_lg = get_safe_font(40)
        self.font_md = get_safe_font(28)
        self.font_sm = get_safe_font(18)

        self.mode = "single"
        self.difficulty = "easy"
        self.num_snakes = 1
        self.snakes = []
        self.scores = []
        self.game_overs = []
        self.target_pos = []

        self.foods: List[FoodItem] = []
        self.particles: List[Particle] = []

        self.last_move_times = []
        self.result_text = ""
        self.sub_text = ""
        self.current_level_idx = 0
        self.level_start_time = 0.0
        self.end_time = 0.0
        self.total_paused_time = 0.0
        self.pause_start_time = 0.0
        self.is_paused_now = False
        self.dual_winner = -1
        self.hit_wall_flash = 0

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

    def clamp(self, v, lo, hi): return max(lo, min(hi, v))
    def dist2(self, p1, p2): return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    def clamp_to_play(self, x, y):
        xm, ym, xM, yM = self.play_area
        return self.clamp(x, xm, xM), self.clamp(y, ym, yM)

    def get_elapsed_time(self) -> int:
        if any(self.game_overs) and self.end_time > 0:
            raw = self.end_time - self.level_start_time
        elif self.is_paused_now:
            raw = self.pause_start_time - self.level_start_time
        else:
            raw = time.time() - self.level_start_time
        return int(max(0, raw - self.total_paused_time))

    def _slot_active(self, i, ts):
        s = self.hand_slots[i]
        return s and (ts - s["ts"] < HAND_LOSS_GRACE)

    def _init_state(self) -> None:
        self.current_level_idx = 0
        self.dual_winner = -1
        self._init_level()

    def _init_level(self) -> None:
        xm, ym, xM, yM = self.play_area
        cx, cy = (xm + xM) // 2, (ym + yM) // 2
        offsets = [0] if self.num_snakes == 1 else [-(xM-xm)//4, (xM-xm)//4]

        self.snakes = []
        for i in range(self.num_snakes):
            bx = cx + offsets[i]
            body = [(bx - j * self.segment_length, cy) for j in range(self.min_segments)]
            self.snakes.append(body)

        self.scores = [0] * self.num_snakes if self.mode == "single" else [self.min_segments] * self.num_snakes
        self.game_overs = [False] * self.num_snakes
        self.target_pos = [(cx, cy) for _ in range(self.num_snakes)]
        self.last_move_times = [0.0] * self.num_snakes

        self.foods = []
        self._spawn_foods(initial=True)
        self.particles = []
        self.hit_wall_flash = 0

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

    def _spawn_foods(self, initial=False):
        target_count = 3 if self.mode == "single" else 2
        xm, ym, xM, yM = self.play_area
        margin = 30

        while len(self.foods) < target_count:
            r = random.random()
            if r < 0.5: ftype = 'insect'
            elif r < 0.8: ftype = 'fish'
            elif r < 0.95: ftype = 'mouse'
            else: ftype = 'poison'

            x = random.randint(xm + margin, xM - margin)
            y = random.randint(ym + margin, yM - margin)

            collision = False
            for snake in self.snakes:
                if self.dist2((x,y), snake[0]) < 5000: collision = True

            if not collision:
                self.foods.append(FoodItem((x,y), ftype, time.time()))

    def _spawn_particles(self, x, y, color):
        for _ in range(10):
            self.particles.append(Particle(x, y, color))

    def draw_cartoon_head(self, draw, pos, radius, color, direction_vec):
        x, y = pos
        dx, dy = direction_vec
        angle = math.atan2(dy, dx)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline=COL_SNAKE_BORDER, width=2)

        eye_offset_x = math.cos(angle - 0.6) * (radius * 0.5)
        eye_offset_y = math.sin(angle - 0.6) * (radius * 0.5)
        eye2_offset_x = math.cos(angle + 0.6) * (radius * 0.5)
        eye2_offset_y = math.sin(angle + 0.6) * (radius * 0.5)

        eye_r = 4.5
        draw.ellipse((x+eye_offset_x-eye_r, y+eye_offset_y-eye_r, x+eye_offset_x+eye_r, y+eye_offset_y+eye_r), fill="white")
        draw.ellipse((x+eye2_offset_x-eye_r, y+eye2_offset_y-eye_r, x+eye2_offset_x+eye_r, y+eye2_offset_y+eye_r), fill="white")
        p_r = 2
        draw.ellipse((x+eye_offset_x-p_r, y+eye_offset_y-p_r, x+eye_offset_x+p_r, y+eye_offset_y+p_r), fill="black")
        draw.ellipse((x+eye2_offset_x-p_r, y+eye2_offset_y-p_r, x+eye2_offset_x+p_r, y+eye2_offset_y+p_r), fill="black")

    def draw_food_icon(self, draw, pos, ftype, radius=None):
        x, y = pos
        r = radius if radius else self.food_radius
        cfg = FOOD_TYPES[ftype]
        color = cfg['color']

        draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline="white", width=2)

        if ftype == 'insect':
            draw.ellipse((x-r*0.4, y-r*0.4, x+r*0.4, y+r*0.4), fill="red")
            draw.line((x-r*0.4, y, x+r*0.4, y), fill="black", width=1)
            draw.line((x, y-r*0.4, x, y+r*0.4), fill="black", width=1)
        elif ftype == 'fish':
            draw.ellipse((x-r*0.5, y-r*0.3, x+r*0.3, y+r*0.3), fill="blue")
            draw.polygon([(x+r*0.2, y), (x+r*0.6, y-r*0.3), (x+r*0.6, y+r*0.3)], fill="blue")
        elif ftype == 'mouse':
            draw.ellipse((x-r*0.5, y-r*0.5, x, y), fill="grey")
            draw.ellipse((x, y-r*0.5, x+r*0.5, y), fill="grey")
            draw.ellipse((x-r*0.4, y-r*0.2, x+r*0.4, y+r*0.4), fill="lightgrey")
        elif ftype == 'poison':
            draw.rectangle((x-r*0.3, y-r*0.2, x+r*0.3, y+r*0.4), fill="purple")
            draw.line((x-r*0.2, y, x+r*0.2, y+r*0.3), fill="white", width=1)
            draw.line((x-r*0.2, y+r*0.3, x+r*0.2, y), fill="white", width=1)

    def draw_ui_layer(self, frame):
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil, 'RGBA')

        # 1. 顶部 HUD
        draw.rectangle((0, 0, self.width, HUD_HEIGHT), fill=COL_HUD_BG)
        draw.line((0, HUD_HEIGHT, self.width, HUD_HEIGHT), fill=(255, 255, 255), width=3)

        # 2. 时间进度条
        elapsed = self.get_elapsed_time()
        if self.mode == "single":
            cfg = SINGLE_LEVELS[self.current_level_idx]
            total_time = cfg["time"]
            current_level_text = f"LEVEL {self.current_level_idx + 1}"
            target_score = cfg["score"]
            current_score = self.scores[0]
        else:
            total_time = DUAL_GAME_DURATION
            current_level_text = "BATTLE MODE"

        rem_time = max(0, total_time - elapsed)
        progress = rem_time / total_time

        bar_color = COL_PROGRESS_OK
        if rem_time <= 3:
            if int(time.time() * 5) % 2 == 0:
                bar_color = COL_PROGRESS_LOW
            else:
                bar_color = (255, 255, 255)

        bar_x, bar_y, bar_w, bar_h = 20, 10, self.width - 40, 15
        draw.rounded_rectangle((bar_x, bar_y, bar_x+bar_w, bar_y+bar_h), radius=5, fill=(220, 220, 220))
        draw.rounded_rectangle((bar_x, bar_y, bar_x+int(bar_w*progress), bar_y+bar_h), radius=5, fill=bar_color)

        time_str = f"{int(rem_time)}s"
        tb_time = draw.textbbox((0,0), time_str, font=self.font_md)
        draw.text(((self.width - (tb_time[2]-tb_time[0]))//2, bar_y + 25), time_str, font=self.font_md, fill=(100, 100, 100))

        # 3. HUD 信息
        draw.text((30, 50), current_level_text, font=self.font_md, fill=(100, 100, 100))
        if self.mode == "single":
            draw.text((30, 80), f"SCORE: {current_score}/{target_score}", font=self.font_md, fill=(50, 150, 250))
        else:
            draw.text((30, 80), f"P1: {self.scores[0]}  P2: {self.scores[1]}", font=self.font_md, fill=(50, 150, 250))

        # 4. 图例
        legend_start_x = self.width - 450
        legends = [('insect', '1'), ('fish', '2'), ('mouse', '3'), ('poison', 'Die')]

        for i, (ftype, val) in enumerate(legends):
            lx = legend_start_x + i * 90
            ly = 65
            self.draw_food_icon(draw, (lx, ly), ftype, radius=12)
            draw.text((lx + 20, ly - 10), val, font=self.font_sm, fill=(150,150,150))

        # 5. 蛇
        for idx, snake in enumerate(self.snakes):
            if not snake: continue

            for i in range(len(snake)-1, 0, -1):
                curr = snake[i]
                if i >= len(snake) - 3:
                    scale = (len(snake) - i) / 3.0
                    r = max(4, int(self.body_radius * scale))
                else:
                    r = self.body_radius

                x, y = int(curr[0]), int(curr[1])
                draw.ellipse((x-r, y-r, x+r, y+r), fill=COL_SNAKE_BODY, outline=COL_SNAKE_BORDER)

            hx, hy = snake[0]
            if len(snake) > 1:
                nx, ny = snake[1]
                direction = (hx - nx, hy - ny)
            else:
                direction = (0, -1)

            head_color = (255, 182, 193) if idx == 0 else (255, 255, 224)
            self.draw_cartoon_head(draw, (int(hx), int(hy)), self.head_radius, head_color, direction)

            if self.mode == "dual":
                head_pt = (int(hx), int(hy))
                outline_col = (100, 255, 100) if self._slot_active(idx, time.time()) else (200, 200, 200)
                draw.ellipse((head_pt[0]-BIND_DIST, head_pt[1]-BIND_DIST, head_pt[0]+BIND_DIST, head_pt[1]+BIND_DIST), outline=outline_col, width=2)

        # 6. 食物 (带闪烁逻辑)
        curr_t = time.time()
        for f in self.foods:
            should_draw = True
            if self.mode == "single" and self.difficulty == "hard" and not self.is_paused_now:
                age = curr_t - f.spawn_time
                time_left = HARD_MODE_FOOD_TIME - age
                if time_left <= 2.0:
                    if int(curr_t * 15) % 2 == 0:
                        should_draw = False

            if should_draw:
                self.draw_food_icon(draw, (int(f.pos[0]), int(f.pos[1])), f.type)

        # 7. 粒子
        for p in self.particles:
            draw.rectangle((p.x-3, p.y-3, p.x+3, p.y+3), fill=p.color + (int(255*p.life),))

        # 8. 撞墙红闪
        if self.hit_wall_flash > 0:
            draw.rectangle((0, 0, self.width, self.height), outline=(255, 0, 0, 100), width=20)
            self.hit_wall_flash -= 1

        # 9. 结果弹窗
        if any(self.game_overs) and self.result_text:
            cx, cy = self.width//2, self.height//2
            draw.rounded_rectangle((cx-200, cy-100, cx+200, cy+100), radius=20, fill=(255, 255, 255, 230), outline=(100,100,100), width=2)
            draw.text((cx-180, cy-60), "GAME OVER", font=self.font_lg, fill=(255, 100, 100))
            draw.text((cx-150, cy+10), self.result_text, font=self.font_md, fill=(50, 50, 50))
            draw.text((cx-120, cy+50), "Press R:Restart  Q:Quit", font=self.font_sm, fill=(150, 150, 150))

        elif self.sub_text:
            cx, cy = self.width//2, self.height//2
            draw.text((cx-100, cy), self.sub_text, font=self.font_lg, fill=(255, 215, 0), stroke_width=2, stroke_fill="black")
            if time.time() - self.level_start_time > 2: self.sub_text = ""

        # 暂停提示
        if (self.auto_paused or self.manual_paused) and not any(self.game_overs):
            cx, cy = self.width//2, self.height//2
            draw.text((cx-100, cy+80), "PAUSED", font=self.font_lg, fill=(100, 100, 255))

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGR)

    # --- 游戏循环与逻辑 ---
    def select_mode(self) -> bool:
        win = "Mode Select"
        menu_state = 'main'

        while True:
            bg = Image.new('RGB', (640, 360), color=COL_BG_BLUE)
            draw = ImageDraw.Draw(bg)
            w, h = 640, 360

            if menu_state == 'main':
                title = "SNAKE FUSION"
                tb = draw.textbbox((0, 0), title, font=self.font_title)
                tw = tb[2] - tb[0]
                draw.text(((w - tw) // 2, 20), title, font=self.font_title, fill=(255, 105, 180))

                draw.rounded_rectangle((50, 100, 590, 180), radius=15, fill=(255, 240, 245), outline=(255, 182, 193), width=3)
                draw.text((70, 110), "1. Adventure (Single)", font=self.font_lg, fill=(100, 100, 100))
                draw.text((70, 155), "Safe wall, Clear levels", font=self.font_sm, fill=(150, 150, 150))

                draw.rounded_rectangle((50, 200, 590, 280), radius=15, fill=(240, 248, 255), outline=(135, 206, 250), width=3)
                draw.text((70, 210), "2. Battle (Dual)", font=self.font_lg, fill=(100, 100, 100))
                draw.text((70, 255), "Hit wall dies, Eat opponent", font=self.font_sm, fill=(150, 150, 150))

                quit_text = "Q: Quit Game"
                tb_q = draw.textbbox((0, 0), quit_text, font=self.font_sm)
                tw_q = tb_q[2] - tb_q[0]
                draw.text(((w - tw_q)//2, 320), quit_text, font=self.font_sm, fill=(200, 100, 100))

            elif menu_state == 'difficulty':
                draw.text((180, 30), "Select Difficulty", font=self.font_lg, fill=(100, 100, 100))

                draw.rounded_rectangle((100, 90, 540, 160), radius=15, fill=(224, 255, 255), outline=(0, 200, 200), width=3)
                draw.text((120, 100), "1. Easy Mode", font=self.font_lg, fill=(0, 150, 150))
                draw.text((120, 140), "Food stays forever", font=self.font_sm, fill=(100, 150, 150))

                draw.rounded_rectangle((100, 180, 540, 250), radius=15, fill=(255, 228, 225), outline=(255, 100, 100), width=3)
                draw.text((120, 190), "2. Hard Mode", font=self.font_lg, fill=(200, 50, 50))
                draw.text((120, 230), "Food vanishes in 3s!", font=self.font_sm, fill=(200, 100, 100))

                draw.text((280, 310), "B: Back", font=self.font_sm, fill=(150, 150, 150))

            frame = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
            cv2.imshow(win, frame)

            key = cv2.waitKey(20) & 0xFF

            if menu_state == 'main':
                if key == ord('1'): menu_state = 'difficulty'
                elif key == ord('2'):
                    self.mode = "dual"
                    self.num_snakes = 2
                    self.binding_complete = False
                    cv2.destroyWindow(win)
                    return True
                elif key in (ord('q'), 27): return False

            elif menu_state == 'difficulty':
                if key == ord('1'):
                    self.mode = "single"; self.difficulty = "easy"; self.num_snakes = 1; self.binding_complete = True
                    cv2.destroyWindow(win)
                    return True
                elif key == ord('2'):
                    self.mode = "single"; self.difficulty = "hard"; self.num_snakes = 1; self.binding_complete = True
                    cv2.destroyWindow(win)
                    return True
                elif key in (ord('b'), 27): menu_state = 'main'

    def bind_players(self, win: str) -> bool:
        self.hand_slots = [None, None]
        start_t = 0
        counting = False

        while True:
            ret, frame = self.cap.read()
            if not ret: return False
            frame = cv2.flip(frame, 1)
            hands = self.tracker.detect(frame)

            heads = [s[0] for s in self.snakes]
            self._assign_hands(hands, heads, time.time())

            both_active = self._slot_active(0, time.time()) and self._slot_active(1, time.time())

            if both_active:
                if not counting: counting = True; start_t = time.time()
                rem = 3 - (time.time() - start_t)
                msg = f"Starting in {int(rem)+1}..."
                if rem <= 0:
                    self.binding_complete = True
                    self.level_start_time = time.time()
                    self.total_paused_time = 0.0
                    return True
            else:
                counting = False
                msg = "Place fingers on snake heads"

            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil, 'RGBA')

            for i, snake in enumerate(self.snakes):
                hx, hy = int(snake[0][0]), int(snake[0][1])
                col = (100, 255, 100) if self._slot_active(i, time.time()) else (200, 200, 200)
                draw.ellipse((hx-BIND_DIST, hy-BIND_DIST, hx+BIND_DIST, hy+BIND_DIST), outline=col, width=3)
                draw.ellipse((hx-10, hy-10, hx+10, hy+10), fill=col)

            draw.rectangle((0, 0, self.width, 80), fill=(0, 0, 0, 150))
            draw.text((50, 20), "Dual Mode Binding", font=self.font_md, fill="white")
            draw.text((50, 50), msg, font=self.font_sm, fill="yellow")

            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGR)
            cv2.imshow(win, frame)

            k = cv2.waitKey(1)
            if k in (ord('q'), 27): return False
            if k == ord('r'): self.hand_slots = [None, None]; counting = False

    def _assign_hands(self, hands, heads, ts):
        if self.mode == "single":
            if not hands: return
            closest, min_d = None, float('inf')
            for h in hands:
                d = self.dist2(h["pos"], heads[0])
                if d < min_d: min_d = d; closest = h["pos"]
            if closest: self.hand_slots[0] = {"pos": closest, "ts": ts}
        else:
            cands = []
            for h in hands:
                for i, hd in enumerate(heads):
                    d = self.dist2(h["pos"], hd)
                    if d < (BIND_DIST*1.5)**2: cands.append((d, i, h["pos"]))
            cands.sort(key=lambda x: x[0])
            used = set()
            for d, i, p in cands:
                if i not in used and d < BIND_DIST**2:
                    self.hand_slots[i] = {"pos": p, "ts": ts}
                    used.add(i)

    def move_snake(self, idx):
        if self.game_overs[idx]: return
        if time.time() - self.last_move_times[idx] < self.move_interval: return
        self.last_move_times[idx] = time.time()

        if self.mode == "single" and self.difficulty == "hard" and not self.is_paused_now:
            now = time.time()
            self.foods = [f for f in self.foods if (now - f.spawn_time) < HARD_MODE_FOOD_TIME]
            self._spawn_foods()

        new_head = self.step_head(idx)

        x, y = new_head
        xm, ym, xM, yM = self.play_area
        rad = self.head_radius
        hit_wall = (x-rad < xm) or (x+rad > xM) or (y-rad < ym) or (y+rad > yM)

        if hit_wall:
            if self.mode == "single":
                new_head = (self.clamp(x, xm+rad, xM-rad), self.clamp(y, ym+rad, yM-rad))
            else:
                self.game_overs[idx] = True
                self.end_time = time.time()
                self.dual_winner = 1 - idx
                self.game_overs[1-idx] = True
                self.result_text = "HIT WALL!"
                self.hit_wall_flash = 5
                return

        ate_idx = -1
        for i, f in enumerate(self.foods):
            if self.dist2(new_head, f.pos) < (self.head_radius + self.food_radius)**2:
                ate_idx = i
                break

        if ate_idx != -1:
            food = self.foods.pop(ate_idx)
            self._spawn_particles(food.pos[0], food.pos[1], FOOD_TYPES[food.type]['color'])

            if food.type == 'poison':
                self.game_overs[idx] = True
                self.end_time = time.time()
                self.result_text = "POISONED!"
                self.hit_wall_flash = 5
                return
            else:
                score = FOOD_TYPES[food.type]['score']
                self.snakes[idx].append(self.snakes[idx][-1])
                if self.mode == "single":
                    self.scores[idx] += score
                    target = SINGLE_LEVELS[self.current_level_idx]["score"]
                    if self.scores[idx] >= target:
                        self._next_level()
                        return

            self._spawn_foods()

        self.rebuild_body(idx, new_head)

        if self.mode == "dual":
            other = 1 - idx
            for k, seg in enumerate(self.snakes[other]):
                if self.dist2(new_head, seg) < (self.head_radius + self.body_radius)**2:
                    if k < 2:
                        self.game_overs = [True, True]
                        self.end_time = time.time()
                        self.result_text = "HEAD CRASH!"
                    else:
                        stolen = self.snakes[other][k:]
                        self.snakes[other] = self.snakes[other][:k]
                        self.snakes[idx].extend(stolen)

                    if len(self.snakes[other]) < 3:
                        self.game_overs = [True, True]
                        self.end_time = time.time()
                        self.dual_winner = idx
                        self.result_text = "ELIMINATED!"
                    return

            self.scores[idx] = len(self.snakes[idx])
            if self.scores[idx] >= DUAL_TARGET_LENGTH:
                self.dual_winner = idx
                self.game_overs = [True, True]
                self.end_time = time.time()
                self.result_text = "WINNER!"

    def run(self):
        while True:
            if not self.select_mode(): break
            win = "Snake Game"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, self.width, self.height)
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

                is_paused = self.auto_paused or self.manual_paused
                if is_paused and not self.is_paused_now: self.pause_start_time = time.time()
                if not is_paused and self.is_paused_now: self.total_paused_time += (time.time() - self.pause_start_time)
                self.is_paused_now = is_paused

                if not is_paused and not any(self.game_overs):
                    self.check_time_limit()
                    for i in range(self.num_snakes):
                        if active[i]:
                            p = self.hand_slots[i]["pos"]
                            self.target_pos[i] = self.clamp_to_play(int(p[0]/frame.shape[1]*self.width), int(p[1]/frame.shape[0]*self.height))
                            self.move_snake(i)

                    for p in self.particles: p.update()
                    self.particles = [p for p in self.particles if not p.is_dead()]

                final_frame = self.draw_ui_layer(frame)
                cv2.imshow(win, final_frame)

                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27):
                    cv2.destroyWindow(win)
                    break
                if k == ord('r'):
                    self._init_state()
                    if self.mode == "dual" and not self.bind_players(win): break
                if k == 32: self.manual_paused = not self.manual_paused

        self.cap.release()
        cv2.destroyAllWindows()

    def check_time_limit(self):
        if any(self.game_overs): return
        elapsed = self.get_elapsed_time()
        if self.mode == "single":
            limit = SINGLE_LEVELS[self.current_level_idx]["time"]
            if elapsed > limit:
                self.game_overs[0] = True; self.end_time = time.time(); self.result_text = "TIME UP!"
        else:
            if elapsed > DUAL_GAME_DURATION:
                self.game_overs = [True, True]; self.end_time = time.time()
                if self.scores[0] > self.scores[1]: self.result_text = "P1 WINS"
                elif self.scores[1] > self.scores[0]: self.result_text = "P2 WINS"
                else: self.result_text = "DRAW"

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

    def _next_level(self):
        if self.current_level_idx < len(SINGLE_LEVELS) - 1:
            self.current_level_idx += 1
            self._init_level()
            self.sub_text = f"LEVEL {self.current_level_idx+1}"
        else:
            self.game_overs[0] = True; self.end_time = time.time(); self.result_text = "ALL CLEAR!"

if __name__ == "__main__":
    SnakeGame().run()
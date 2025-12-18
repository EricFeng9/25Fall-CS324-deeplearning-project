from __future__ import annotations
import random
import time
import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import *
from hand_detector import HandTracker, gstreamer_pipeline
from ui import *

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
        self.quit_trigger_start = 0.0 # NEW: For in-game quit timer

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

    # --- 辅助函数 ---
    def clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def dist2(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def clamp_to_play(self, x, y):
        xm, ym, xM, yM = self.play_area
        # 允许手稍微出界以便贴墙，但在 move_snake 中会进行严格判定
        margin = 50
        return self.clamp(x, xm - margin, xM + margin), self.clamp(y, ym - margin, yM + margin)

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

    # --- 初始化 ---
    def _init_state(self) -> None:
        self.current_level_idx = 0
        self.dual_winner = -1
        self._init_level()

    def _init_level(self) -> None:
        xm, ym, xM, yM = self.play_area
        cx, cy = (xm + xM) // 2, (ym + yM) // 2
        offsets = [0] if self.num_snakes == 1 else [-(xM - xm) // 4, (xM - xm) // 4]

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
        self.quit_trigger_start = 0.0

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
        self.go_trigger_start = 0.0 # Game Over trigger timer
        self.go_trigger_action = None # 'restart' or 'quit'
        if self.mode == "dual":
            self.binding_complete = False

    def _spawn_foods(self, initial=False):
        target_count = 3 if self.mode == "single" else 2
        xm, ym, xM, yM = self.play_area
        margin = 30

        while len(self.foods) < target_count:
            r = random.random()
            if r < 0.5:
                ftype = 'insect'
            elif r < 0.8:
                ftype = 'fish'
            elif r < 0.95:
                ftype = 'mouse'
            else:
                ftype = 'poison'

            x = random.randint(xm + margin, xM - margin)
            y = random.randint(ym + margin, yM - margin)

            collision = False
            for snake in self.snakes:
                if self.dist2((x, y), snake[0]) < 5000: collision = True

            if not collision:
                self.foods.append(FoodItem((x, y), ftype, time.time()))

    def _spawn_particles(self, x, y, color):
        for _ in range(10):
            self.particles.append(Particle(x, y, color))

    # --- 游戏循环与逻辑 ---
    def select_mode(self) -> bool:
        """
        主菜单逻辑 (重构版):
        - 界面：中央显示摄像头预览，四周显示选项文本。
        - 交互：检测到特定手势后，框变绿 + 倒计时 3 秒确认。
        """
        win = "Mode Select"
        menu_state = 'main' # 'main', 'difficulty'
        
        # 计时器状态
        selection_start_time = 0.0
        current_selection = None # 存储当前正在选中的目标 (e.g., 'single', 'dual', 'back'...)
        
        # UI 布局参数
        cam_w, cam_h = 160, 160 # 缩小一倍显示
        cam_x = (self.width - cam_w) // 2
        cam_y = (self.height - cam_h) // 2
        
        while True:
            ret, frame = self.cap.read()
            if not ret: return False
            frame = cv2.flip(frame, 1)
            hands = self.tracker.detect(frame)
            
            # 1. 识别手势
            gesture = "UNKNOWN"
            if hands:
                gesture = hands[0].get("gesture", "UNKNOWN")
            
            # 2. 映射手势到“意图”
            intended_selection = None
            if menu_state == 'main':
                if gesture == "POINTING_UP": intended_selection = 'single' # 预选单人(进入难度选择)
                elif gesture == "VICTORY": intended_selection = 'dual'
                elif gesture == "FIST": intended_selection = 'quit'
            elif menu_state == 'difficulty':
                if gesture == "PINKY_UP": intended_selection = 'easy'
                elif gesture == "THUMB_UP": intended_selection = 'hard'
                elif gesture == "FIST": intended_selection = 'back'
            
            # 3. 倒计时逻辑
            # 如果意图和当前选中一致，且不是None -> 继续计时
            if intended_selection is not None:
                if intended_selection == current_selection:
                    # 持续保持
                    pass
                else:
                    # 新意图
                    current_selection = intended_selection
                    selection_start_time = time.time()
            else:
                # 意图丢失 -> 重置
                current_selection = None
                selection_start_time = 0.0

            # 计算剩余时间
            confirm_progress = 0.0
            triggered = False
            if current_selection is not None:
                elapsed = time.time() - selection_start_time
                confirm_progress = min(1.0, elapsed / 3.0)
                if elapsed >= 3.0:
                    triggered = True
            
            # 4. 绘制 UI
            # 背景
            bg_pil = Image.new('RGB', (self.width, self.height), color=COL_BG_BLUE)
            draw = ImageDraw.Draw(bg_pil, 'RGBA')

            accent = (255, 120, 150)
            text_main = (70, 70, 70)
            text_muted = (120, 120, 120)
            card_fill = (255, 255, 255, 235)
            card_shadow = (0, 0, 0, 35)

            # 摄像头卡片阴影 + 背板
            shadow_offset = 8
            draw.rounded_rectangle((cam_x - shadow_offset, cam_y - shadow_offset,
                                    cam_x + cam_w + shadow_offset, cam_y + cam_h + shadow_offset),
                                   radius=18, fill=card_shadow)
            draw.rounded_rectangle((cam_x - 6, cam_y - 6, cam_x + cam_w + 6, cam_y + cam_h + 6),
                                   radius=16, fill=(235, 240, 240))

            # 标题
            if menu_state == 'main':
                title = "SNAKE FUSION"
                title_color = accent
            else:
                title = "SELECT DIFFICULTY"
                title_color = (100, 130, 255)
            tb = draw.textbbox((0, 0), title, font=self.font_title)
            draw.text(((self.width - (tb[2]-tb[0]))//2, 60), title, font=self.font_title, fill=title_color)

            if menu_state == 'main':
                # 左右卡片布局
                card_w, card_h = 320, 200
                left_x = 120
                right_x = self.width - card_w - 120
                card_y = cam_y - 40

                def draw_card(x, y, title_text, hint_text):
                    draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=18, fill=card_fill, outline=(230, 235, 235), width=2)
                    title_tb = draw.textbbox((0, 0), title_text, font=self.font_lg)
                    draw.text((x + 20, y + 40), title_text, font=self.font_lg, fill=text_main)
                    draw.text((x + 20, y + 95), hint_text, font=self.font_md, fill=text_muted)

                draw_card(left_x, card_y, "Single Player", "Pointing up")
                draw_card(right_x, card_y, "Dual Player", "Victory")

                # 底部退出提示 pill
                pill_text = "Quit: Fist"
                tb_q = draw.textbbox((0, 0), pill_text, font=self.font_md)
                pill_w = tb_q[2] - tb_q[0] + 36
                pill_h = tb_q[3] - tb_q[1] + 18
                px = (self.width - pill_w) // 2
                py = self.height - 110
                draw.rounded_rectangle((px, py, px + pill_w, py + pill_h), radius=30, fill=(255, 235, 235, 220), outline=(255, 180, 180), width=2)
                draw.text((px + 18, py + (pill_h - (tb_q[3]-tb_q[1]))//2), pill_text, font=self.font_md, fill=(200, 80, 80))

            elif menu_state == 'difficulty':
                card_w, card_h = 300, 180
                left_x = 160
                right_x = self.width - card_w - 160
                card_y = cam_y - 20

                draw.rounded_rectangle((left_x, card_y, left_x + card_w, card_y + card_h), radius=18, fill=card_fill, outline=(210, 235, 235), width=2)
                draw.rounded_rectangle((right_x, card_y, right_x + card_w, card_y + card_h), radius=18, fill=card_fill, outline=(235, 210, 210), width=2)

                draw.text((left_x + 20, card_y + 40), "Easy Mode", font=self.font_lg, fill=(0, 140, 140))
                draw.text((left_x + 20, card_y + 95), "Pinky up", font=self.font_md, fill=(80, 150, 150))

                draw.text((right_x + 20, card_y + 40), "Hard Mode", font=self.font_lg, fill=(200, 70, 70))
                draw.text((right_x + 20, card_y + 95), "Thumbs up", font=self.font_md, fill=(200, 90, 90))

                # Back pill
                back_text = "Back: Fist"
                tb_b = draw.textbbox((0, 0), back_text, font=self.font_md)
                pill_w = tb_b[2] - tb_b[0] + 36
                pill_h = tb_b[3] - tb_b[1] + 18
                px = (self.width - pill_w) // 2
                py = self.height - 110
                draw.rounded_rectangle((px, py, px + pill_w, py + pill_h), radius=30, fill=(240, 240, 240, 220), outline=(200, 200, 200), width=2)
                draw.text((px + 18, py + (pill_h - (tb_b[3]-tb_b[1]))//2), back_text, font=self.font_md, fill=text_muted)

            # 转换回 OpenCV
            final_img = cv_bg = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)
            
            # 贴入摄像头画面 (居中)
            frame_resized = cv2.resize(frame, (cam_w, cam_h))
            final_img[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = frame_resized
            
            # 绘制边框和进度
            border_theme = (200, 200, 200) # 默认灰
            if current_selection:
                border_theme = (0, 255, 0) # 选中时变绿
            
            # 边框
            cv2.rectangle(final_img, (cam_x, cam_y), (cam_x+cam_w, cam_y+cam_h), border_theme, 4)
            
            # 进度提示
            if current_selection:
                status_text = f"Selecting: {current_selection.upper()} {3.0 - (time.time() - selection_start_time):.1f}s"
                (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                text_x = cam_x + (cam_w - tw) // 2
                text_y = max(30, cam_y - 20)
                cv2.putText(final_img, status_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_theme, 2)

                # 居中的进度条
                bar_w = cam_w
                bar_valid_w = int(bar_w * confirm_progress)
                bar_x = cam_x
                bar_y = cam_y + cam_h + 10
                cv2.rectangle(final_img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (220, 220, 220), -1)
                cv2.rectangle(final_img, (bar_x, bar_y), (bar_x + bar_valid_w, bar_y + 10), border_theme, -1)
            
            cv2.imshow(win, final_img)
            
            # 5. 触发确认
            if triggered:
                if current_selection == 'single':
                    menu_state = 'difficulty'
                    current_selection = None
                    time.sleep(0.5) # 防止连续误触
                elif current_selection == 'dual':
                    self.mode = "dual"
                    self.num_snakes = 2
                    self.binding_complete = False
                    cv2.destroyWindow(win)
                    return True
                elif current_selection == 'quit':
                    return False
                elif current_selection == 'easy':
                    self.mode = "single"
                    self.difficulty = "easy"
                    self.num_snakes = 1
                    self.binding_complete = True
                    cv2.destroyWindow(win)
                    return True
                elif current_selection == 'hard':
                    self.mode = "single"
                    self.difficulty = "hard"
                    self.num_snakes = 1
                    self.binding_complete = True
                    cv2.destroyWindow(win)
                    return True
                elif current_selection == 'back':
                    menu_state = 'main'
                    current_selection = None
                    time.sleep(0.5)
            
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): return False
            # 键盘后门保留
            if key == ord('1'):
                if menu_state == 'main': menu_state = 'difficulty'
                else: 
                    self.mode = "single"; self.difficulty = "easy"; self.num_snakes = 1; self.binding_complete = True
                    cv2.destroyWindow(win); return True
            if key == ord('2'):
                if menu_state == 'main':
                    self.mode = "dual"; self.num_snakes = 2; self.binding_complete = False; cv2.destroyWindow(win); return True
                else:
                    self.mode = "single"; self.difficulty = "hard"; self.num_snakes = 1; self.binding_complete = True
                    cv2.destroyWindow(win); return True

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
                msg = f"Starting in {int(rem) + 1}..."
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
                draw.ellipse((hx - BIND_DIST, hy - BIND_DIST, hx + BIND_DIST, hy + BIND_DIST), outline=col, width=3)
                draw.ellipse((hx - 10, hy - 10, hx + 10, hy + 10), fill=col)

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
                    if d < (BIND_DIST * 1.5) ** 2: cands.append((d, i, h["pos"]))
            cands.sort(key=lambda x: x[0])
            used = set()
            for d, i, p in cands:
                if i not in used and d < BIND_DIST ** 2:
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

        # 只要蛇头边缘(rad)碰到边界就算撞墙
        hit_wall = (x - rad < xm) or (x + rad > xM) or (y - rad < ym) or (y + rad > yM)

        if hit_wall:
            if self.mode == "single":
                # 单人模式：滑墙不死
                new_head = (self.clamp(x, xm + rad, xM - rad), self.clamp(y, ym + rad, yM - rad))
            else:
                # 双人模式：碰到边界就输！
                self.game_overs[idx] = True
                self.end_time = time.time()
                self.dual_winner = 1 - idx
                self.game_overs[1 - idx] = True
                self.result_text = "HIT WALL!"
                self.hit_wall_flash = 5
                return


        ate_idx = -1
        for i, f in enumerate(self.foods):
            if self.dist2(new_head, f.pos) < (self.head_radius + self.food_radius) ** 2:
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
                if self.dist2(new_head, seg) < (self.head_radius + self.body_radius) ** 2:
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

            # In-game quit timer
            # self.quit_trigger_start = 0.0 # Done in init_state

            while True:
                ret, frame = self.cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                now = time.time()

                hands = self.tracker.detect(frame)
                heads = [s[0] if s else (0, 0) for s in self.snakes]
                self._assign_hands(hands, heads, now)

                active = [self._slot_active(i, now) for i in range(self.num_snakes)]
                
                # In-Game Quit Logic: FIST
                fist_detected = False
                for h in hands:
                    if h.get("gesture") == "FIST":
                        fist_detected = True
                        break
                
                if fist_detected:
                    if self.quit_trigger_start == 0.0:
                        self.quit_trigger_start = now
                    
                    elapsed_quit = now - self.quit_trigger_start
                    if elapsed_quit >= 3.0:
                        cv2.destroyWindow(win)
                        break
                else:
                    self.quit_trigger_start = 0.0

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
                            self.target_pos[i] = self.clamp_to_play(int(p[0] / frame.shape[1] * self.width),
                                                                    int(p[1] / frame.shape[0] * self.height))
                            self.move_snake(i)

                    for p in self.particles: p.update()
                    self.particles = [p for p in self.particles if not p.is_dead()]

                # Game Over Gesture Control
                if any(self.game_overs):
                    detected_action = None
                    for h in hands:
                        g = h.get("gesture")
                        
                        # 根据模式判断重启手势
                        # 单人模式: 食指向上 (POINTING_UP)
                        # 双人模式: 剪刀手 (VICTORY)
                        target_gesture = "POINTING_UP" if self.mode == "single" else "VICTORY"

                        if g == target_gesture:
                            detected_action = 'restart'
                            break
                        elif g == "FIST":
                            detected_action = 'quit'
                            break
                    
                    if detected_action:
                        if self.go_trigger_action != detected_action:
                            self.go_trigger_action = detected_action
                            self.go_trigger_start = now
                        
                        elapsed = now - self.go_trigger_start
                        if elapsed >= 3.0:
                            if self.go_trigger_action == 'restart':
                                self._init_state()
                                if self.mode == "dual":
                                    if not self.bind_players(win): 
                                        cv2.destroyAllWindows()
                                        break
                            elif self.go_trigger_action == 'quit':
                                cv2.destroyWindow(win)
                                break
                    else:
                         self.go_trigger_start = 0.0
                         self.go_trigger_action = None

                final_frame = display_game_overlay(self, frame)

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
                self.game_overs[0] = True;
                self.end_time = time.time();
                self.result_text = "TIME UP!"
        else:
            if elapsed > DUAL_GAME_DURATION:
                self.game_overs = [True, True];
                self.end_time = time.time()
                if self.scores[0] > self.scores[1]:
                    self.result_text = "P1 WINS"
                elif self.scores[1] > self.scores[0]:
                    self.result_text = "P2 WINS"
                else:
                    self.result_text = "DRAW"

    def step_head(self, idx):
        hx, hy = self.snakes[idx][0]
        tx, ty = self.target_pos[idx]
        dx, dy = tx - hx, ty - hy
        d = (dx * dx + dy * dy) ** 0.5
        if d < self.follow_deadzone: return (hx, hy)
        step = min(self.max_step, d) * self.alpha
        return (hx + dx / d * step, hy + dy / d * step)

    def rebuild_body(self, idx, head):
        pts = [head]
        prev = head
        for i, seg in enumerate(self.snakes[idx][1:]):
            dx, dy = prev[0] - seg[0], prev[1] - seg[1]
            d = (dx * dx + dy * dy) ** 0.5
            if d < 1e-4: continue
            nx = prev[0] - dx / d * self.segment_length
            ny = prev[1] - dy / d * self.segment_length
            pts.append((nx, ny))
            prev = (nx, ny)
        self.snakes[idx] = pts

    def _next_level(self):
        if self.current_level_idx < len(SINGLE_LEVELS) - 1:
            self.current_level_idx += 1
            self._init_level()
            self.sub_text = f"LEVEL {self.current_level_idx + 1}"
        else:
            self.game_overs[0] = True;
            self.end_time = time.time();
            self.result_text = "ALL CLEAR!"

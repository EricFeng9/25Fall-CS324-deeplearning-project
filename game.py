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
        主菜单逻辑
        """
        win = "Mode Select"
        menu_state = 'main'  # 'main' 或 'difficulty'

        # 计时器状态初始化
        selection_start_time = 0.0
        current_selection = None  # 当前正在“注视”/选中的选项标识符

        # 布局参数 关于摄像头比例长度和位置
        cam_h = 465
        cam_w = int(cam_h * ( 5 / 3))
        cam_x = 60
        cam_y = (self.height - cam_h) // 2 + 30

        while True:
            # 1. 读取摄像头
            ret, frame = self.cap.read()
            if not ret: return False
            frame = cv2.flip(frame, 1)

            # 2. 检测手势
            hands = self.tracker.detect(frame)
            gesture = "UNKNOWN"
            if hands:
                gesture = hands[0].get("gesture", "UNKNOWN")

            # 3. 意图映射：给手势赋予含义 (定义 quit_app 的地方)
            intended_selection = None

            if gesture == "FIST":
                if menu_state == 'main':
                    intended_selection = 'quit_app'  # 这里定义了 quit_app
                else:
                    intended_selection = 'back_to_main'  # 这里定义了 back_to_main
            else:
                if menu_state == 'main':
                    if gesture == "POINTING_UP":
                        intended_selection = 'single'
                    elif gesture == "VICTORY":
                        intended_selection = 'dual'
                elif menu_state == 'difficulty':
                    if gesture == "PINKY_UP":
                        intended_selection = 'easy'
                    elif gesture == "THUMB_UP":
                        intended_selection = 'hard'

            # 4. 倒计时逻辑
            if intended_selection is not None:
                if intended_selection == current_selection:
                    pass  # 手势没变，继续保持
                else:
                    # 手势变了，重置计时器
                    current_selection = intended_selection
                    selection_start_time = time.time()
            else:
                current_selection = None
                selection_start_time = 0.0

            # 5. 计算进度
            confirm_progress = 0.0
            triggered = False

            if current_selection is not None:
                elapsed = time.time() - selection_start_time
                confirm_progress = min(1.0, elapsed / 3.0)  # 3秒倒计时

                # 只有这里满足 3秒，triggered 才会变成 True
                if elapsed >= 3.0:
                    triggered = True

            # 6. 绘制 UI
            bg_color_bgr = (COL_BG_BLUE[2], COL_BG_BLUE[1], COL_BG_BLUE[0])
            final_img = np.full((self.height, self.width, 3), bg_color_bgr, dtype=np.uint8)
            frame_resized = cv2.resize(frame, (cam_w, cam_h))
            final_img[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = frame_resized

            # 调用 ui.py 绘制 (红色框和倒计时文字在这里画)
            final_img = draw_menu_interface(
                final_img,
                menu_state,
                current_selection,
                confirm_progress,
                (cam_x, cam_y, cam_w, cam_h)
            )
            cv2.imshow(win, final_img)

            # 7. 处理触发逻辑
            if triggered:

                # 退出程序
                if current_selection == 'quit_app':
                    return False

                    # 返回上级菜单
                elif current_selection == 'back_to_main':
                    menu_state = 'main'
                    current_selection = None  # 重置防止连续触发

                # 进入难度选择
                elif current_selection == 'single':
                    menu_state = 'difficulty'
                    current_selection = None

                # 进入双人模式
                elif current_selection == 'dual':
                    self.mode = "dual"
                    self.num_snakes = 2
                    self.binding_complete = False
                    cv2.destroyWindow(win)
                    return True

                # 选择简单
                elif current_selection == 'easy':
                    self.mode = "single"
                    self.difficulty = "easy"
                    self.num_snakes = 1
                    self.binding_complete = True
                    cv2.destroyWindow(win)
                    return True

                # 选择困难
                elif current_selection == 'hard':
                    self.mode = "single"
                    self.difficulty = "hard"
                    self.num_snakes = 1
                    self.binding_complete = True
                    cv2.destroyWindow(win)
                    return True

            # 8. 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
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
        """
        游戏主入口：包含最外层的应用循环和内层的游戏循环。
        """
        while True:
            # 1. 模式选择循环
            # 如果 select_mode 返回 False，说明用户想要退出程序
            if not self.select_mode():
                break

                # 初始化游戏窗口和状态
            win = "Snake Game"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, self.width, self.height)
            self._init_state()

            # 双人模式需要先进行绑定
            if self.mode == "dual":
                if not self.bind_players(win):
                    cv2.destroyAllWindows()
                    continue  # 如果绑定失败或退出，回到模式选择

            # 2. 游戏内主循环
            back_to_menu = False  # 标记：是否需要返回主菜单

            while True:
                ret, frame = self.cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                now = time.time()

                # 手势检测
                hands = self.tracker.detect(frame)

                # 只有当手势有效时才更新蛇的控制
                heads = [s[0] if s else (0, 0) for s in self.snakes]
                self._assign_hands(hands, heads, now)

                # 检查活跃状态 (是否丢手)
                active = [self._slot_active(i, now) for i in range(self.num_snakes)]
                self.auto_paused = False
                if self.mode == "single":
                    if not active[0]: self.auto_paused = True
                else:
                    if not all(active): self.auto_paused = True

                # --- 游戏内退出逻辑 (FIST 长按 3秒) ---
                fist_detected = False
                for h in hands:
                    if h.get("gesture") == "FIST":
                        fist_detected = True
                        break

                if fist_detected:
                    # 如果是第一次检测到，记录开始时间
                    if self.quit_trigger_start == 0.0:
                        self.quit_trigger_start = now

                    elapsed_quit = now - self.quit_trigger_start
                    if elapsed_quit >= 3.0:
                        # 触发退出：关闭当前窗口，设置标志位
                        cv2.destroyWindow(win)
                        back_to_menu = True
                        break
                else:
                    # 手势中断，重置计时器
                    self.quit_trigger_start = 0.0

                # 暂停逻辑处理
                is_paused = self.auto_paused or self.manual_paused
                if is_paused and not self.is_paused_now: self.pause_start_time = time.time()
                if not is_paused and self.is_paused_now: self.total_paused_time += (time.time() - self.pause_start_time)
                self.is_paused_now = is_paused

                # 游戏逻辑更新 (未暂停且未结束时)
                if not is_paused and not any(self.game_overs):
                    self.check_time_limit()
                    for i in range(self.num_snakes):
                        if active[i]:
                            p = self.hand_slots[i]["pos"]
                            self.target_pos[i] = self.clamp_to_play(int(p[0] / frame.shape[1] * self.width),
                                                                    int(p[1] / frame.shape[0] * self.height))
                            self.move_snake(i)

                    # 更新粒子
                    for p in self.particles: p.update()
                    self.particles = [p for p in self.particles if not p.is_dead()]

                # --- 游戏结束后的手势控制 (重启/退出) ---
                if any(self.game_overs):
                    detected_action = None
                    for h in hands:
                        g = h.get("gesture")

                        # 单人: 食指重启; 双人: 剪刀手重启
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
                                # 重启游戏：重新初始化状态，如果双人则重新绑定
                                self._init_state()
                                if self.mode == "dual":
                                    if not self.bind_players(win):
                                        cv2.destroyAllWindows()
                                        break
                            elif self.go_trigger_action == 'quit':
                                # 退出到菜单
                                cv2.destroyWindow(win)
                                back_to_menu = True
                                break
                    else:
                        self.go_trigger_start = 0.0
                        self.go_trigger_action = None

                # 绘制最终画面 (含 UI)
                final_frame = display_game_overlay(self, frame)
                cv2.imshow(win, final_frame)

                # 键盘逻辑
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27):
                    cv2.destroyWindow(win)
                    back_to_menu = True  # 按 Q 返回菜单
                    break
                if k == ord('r'):
                    self._init_state()
                    if self.mode == "dual" and not self.bind_players(win): break
                if k == 32: self.manual_paused = not self.manual_paused

        # 彻底退出应用
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

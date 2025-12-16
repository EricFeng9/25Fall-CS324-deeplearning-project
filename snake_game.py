"""
手势贪吃蛇（连续坐标版）精简重构版
==================================================

玩法概述
----------------------------------
- 单人模式：任意一只手食指控制一条蛇；没手自动暂停。
- 双人模式：开局先绑定两只手（玩家1/玩家2），倒计时后开始；
  游戏过程中若任一绑定的手离开画面，会自动暂停并重新绑定。
- 食物公共，先到 10 分胜利；自撞/撞墙判负；两蛇互撞判平。

依赖
----------------------------------
pip install -r requirements.txt
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# -------------------- 可调常量 --------------------
USE_JETSON = False  # Jetson Nano 上用 GStreamer，否则用普通摄像头
# 目标分数：达到即获胜
TARGET_SCORE = 10
# 游戏区域外侧留白比例（上下左右各 10%）
PADDING_RATIO = 0.10
# 手丢失后继续前进的缓冲秒数（不立刻暂停）
HAND_LOSS_GRACE = 2.0
# 绑定时，手指与蛇头的距离阈值（像素）
BIND_DIST = 80

PointF = Tuple[float, float]
PointI = Tuple[int, int]


def gstreamer_pipeline(
    sensor_id=0,
    sensor_mode=4,          # 1280x720 @60fps
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=2,
) -> str:
    """Jetson Nano 上使用的 GStreamer 管线。"""
    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, "
        "format=(string)BGRx ! videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class HandTracker:
    """手检测封装：只检测手，不做持久 ID；返回手指尖坐标列表。"""

    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.drawer = mp.solutions.drawing_utils

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        返回 [{'pos': (x,y), 'label': 'green'/'yellow'}] 列表（颜色标签仅作调试参考，游戏逻辑不依赖）。
        仅负责检测与可视化，不生成持久 ID。
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        hands = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                h, w, _ = frame.shape
                p = lm.landmark[8]  # 食指指尖
                pos = (int(p.x * w), int(p.y * h))
                # 将 mediapipe 的左右手标签映射为绿色方/黄色方（仅用于调试展示）
                # 这里 raw_label 仍然是 mediapipe 内部的左右手英文标签，这里不再在代码中写出具体文案
                raw_label = handed.classification[0].label
                # 统一映射为绿色方/黄色方标签，便于后续如需调试
                label = "green" if raw_label == handed.classification[0].label else "yellow"
                self.drawer.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
                cv2.circle(frame, pos, 10, (0, 255, 0), -1)
                hands.append({"pos": pos, "label": label})
        return hands


class SnakeGame:
    """核心游戏类：负责状态、更新、绘制、控制。"""

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        # 画面 & 参数
        self.width = width
        self.height = height
        self.pad_x = int(self.width * PADDING_RATIO)
        base_pad_y = int(self.height * PADDING_RATIO)
        self.pad_top = 0
        self.pad_bottom = base_pad_y * 2  # 上方去掉的 padding 全部加到底部
        self.segment_length = 18
        self.min_segments = 6
        self.head_radius = 12
        self.body_radius = 9
        self.food_radius = 12
        self.alpha = 0.35
        self.max_step = 42
        self.follow_deadzone = 5
        self.move_interval = 0.02
        self.target_score = TARGET_SCORE

        # 状态
        self.mode = "single"           # "single" / "dual"
        self.num_snakes = 1
        self.snakes: List[List[PointF]] = []
        self.scores: List[int] = []
        self.game_overs: List[bool] = []
        self.target_pos: List[PointI] = []
        self.food: PointI = (0, 0)
        self.last_move_times: List[float] = []
        self.result_text: str = ""

        # 手部检测器
        self.tracker = HandTracker()

        # 手槽（不使用持久 ID）：hand_slots[i] = {"pos": 指尖像素坐标, "last_seen": 时间戳}
        # 槽位 0 -> 绿蛇；槽位 1 -> 橙蛇
        self.hand_slots: List[Optional[Dict]] = [None, None]

        # 绑定状态
        self.binding_complete = False
        self.rebinding = False

        # 暂停状态开关
        self.manual_paused: bool = False  # 空格键手动暂停
        self.auto_paused: bool = False    # 无手/手丢失自动暂停
        self.need_rebind_prompt: bool = False  # 丢手后，等待玩家确认是否重连/重启/退出

        # 摄像头
        if USE_JETSON:
            pipeline = gstreamer_pipeline(
                capture_width=width,
                capture_height=height,
                display_width=width,
                display_height=height,
                framerate=30,
                flip_method=2,
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                raise RuntimeError("❌ 无法通过 GStreamer 打开摄像头")
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._init_state()

    # ---------- 基础工具 ----------
    @staticmethod
    def clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    @staticmethod
    def dist2(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]
        return dx * dx + dy * dy

    # ---------- 辅助：游戏区域（考虑 padding） ----------
    def play_bounds(self) -> Tuple[int, int, int, int]:
        """返回有效游戏区域 [xmin, xmax, ymin, ymax]（含边界）。"""
        xmin = self.pad_x
        xmax = self.width - self.pad_x - 1
        ymin = self.pad_top
        ymax = self.height - self.pad_bottom - 1
        return xmin, xmax, ymin, ymax

    def clamp_to_play(self, x: int, y: int) -> Tuple[int, int]:
        xmin, xmax, ymin, ymax = self.play_bounds()
        return self.clamp(x, xmin, xmax), self.clamp(y, ymin, ymax)

    # ---------- 状态初始化 ----------
    def _init_state(self) -> None:
        xmin, xmax, ymin, ymax = self.play_bounds()
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        span_x = xmax - xmin
        spawn_offsets = [0] if self.num_snakes == 1 else [-span_x // 4, span_x // 4]

        self.snakes = []
        for i in range(self.num_snakes):
            base_x = cx + spawn_offsets[i]
            body = [(base_x - j * self.segment_length, cy) for j in range(self.min_segments)]
            self.snakes.append(body)

        self.scores = [0 for _ in range(self.num_snakes)]
        self.game_overs = [False for _ in range(self.num_snakes)]
        self.target_pos = [(cx, cy) for _ in range(self.num_snakes)]
        self.last_move_times = [0.0 for _ in range(self.num_snakes)]
        self.food = self.generate_food()
        self.result_text = ""
        # 每次重置时清除暂停状态
        self.manual_paused = False
        self.auto_paused = False
        self.need_rebind_prompt = False
        # 清空手槽
        self.hand_slots = [None, None]
        if self.mode == "dual":
            self.binding_complete = False
            self.rebinding = False

    def _anchor_snake(self, idx: int, pos: PointI) -> None:
        """把某条蛇的头放到 pos，身体向左展开，目标点也同步。"""
        hx = self.clamp(int(pos[0]), 0, self.width - 1)
        hy = self.clamp(int(pos[1]), 0, self.height - 1)
        hx, hy = self.clamp_to_play(hx, hy)
        body = [(hx - j * self.segment_length, hy) for j in range(self.min_segments)]
        # 防御性扩展列表
        while len(self.snakes) <= idx:
            self.snakes.append([])
        while len(self.target_pos) <= idx:
            self.target_pos.append((hx, hy))
        while len(self.game_overs) <= idx:
            self.game_overs.append(False)
        while len(self.last_move_times) <= idx:
            self.last_move_times.append(0.0)
        self.snakes[idx] = body
        self.target_pos[idx] = (hx, hy)
        self.game_overs[idx] = False
        self.last_move_times[idx] = 0.0

    # ---------- 手槽匹配 ----------
    def _slot_active(self, idx: int, now_ts: float) -> bool:
        slot = self.hand_slots[idx]
        return slot is not None and (now_ts - slot["last_seen"]) < HAND_LOSS_GRACE

    def _assign_hands_to_slots(
        self, hands: List[Dict], heads: List[PointF], now_ts: float, bind_thresh: int = BIND_DIST
    ) -> None:
        """
        按“当前位置最近的蛇头”分配当帧检测到的手到槽位。
        - 逐帧重算，不依赖历史 ID，避免左右手交换导致串槽。
        - 只有进入绑定/控制阈值 (bind_thresh) 的手才会候选。
        """
        candidates: List[Tuple[float, int, PointI]] = []
        thresh2 = bind_thresh * bind_thresh
        for h in hands:
            pos = h["pos"]
            for idx, head in enumerate(heads):
                if idx >= len(self.hand_slots):
                    continue
                d2 = self.dist2(pos, head)
                if d2 <= thresh2:
                    candidates.append((d2, idx, pos))
        # 最短距离优先分配，避免同一头被多只手覆盖
        candidates.sort(key=lambda x: x[0])
        used = set()
        for _, idx, pos in candidates:
            if idx in used:
                continue
            self.hand_slots[idx] = {"pos": pos, "last_seen": now_ts}
            used.add(idx)

    # ---------- 绑定流程 ----------
    def select_mode(self) -> bool:
        """选择单/双人模式的弹窗。"""
        win = "Mode Select"
        canvas = np.zeros((360, 640, 3), np.uint8)
        cv2.putText(canvas, "Hand Snake Game", (120, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        rules = [
            "Rules (Dual):",
            "- First to reach 10 points wins",
            "- Green side hits itself => Yellow side wins",
            "- Yellow side hits itself => Green side wins",
            "- Snakes collide => Draw",
        ]
        for i, t in enumerate(rules):
            cv2.putText(canvas, t, (40, 130 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(canvas, "1: Single (one hand -> one snake)", (60, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, "2: Dual (left/right hand -> two snakes)", (60, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
        cv2.putText(canvas, "Q / Esc: Quit", (60, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        while True:
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
        """
        双人绑定（新版规则）：
        - 蛇头不动，玩家需把手指移到蛇头上方才判定绑定。
        - 绿蛇=玩家1，橙蛇=玩家2，按蛇头邻近绑定，不看左右手标签。
        - 两人都绑定后直接倒计时开始。
        - 途中按 R 重新开始绑定；Q/Esc 退出到模式选择。
        - 每帧重新分配槽位，手交换也能正确匹配最近的蛇头。
        """
        bind_thresh = BIND_DIST  # 手指与蛇头的距离阈值（像素）
        countdown_start = 0.0
        counting = False
        self.hand_slots = [None, None]

        while True:
            ret, frame = self.cap.read()
            if not ret:
                return False
            frame = cv2.flip(frame, 1)
            hands = self.tracker.detect(frame)
            now_ts = time.time()

            # 先把当前蛇的位置画出来，方便玩家将手指移到蛇头处绑定
            self._draw_snakes_binding(frame)

            cv2.putText(frame, "Binding players (Dual mode)", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "Move fingertip onto the snake head to bind.", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, "Green=Player1  Orange=Player2", (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "R: restart bind   Q/Esc: quit", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # 当前蛇头位置（保持不动）
            heads = [self.snakes[0][0], self.snakes[1][0]]

            # 当帧按最近蛇头分配槽位
            self._assign_hands_to_slots(hands, heads, now_ts, bind_thresh)

            both = self._slot_active(0, now_ts) and self._slot_active(1, now_ts)
            if both:
                if not counting:
                    counting = True
                    countdown_start = time.time()
                remain = 3 - (time.time() - countdown_start)
                cv2.putText(frame, f"Both bound! Starting in {int(remain)+1}", (40, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if remain <= 0:
                    self.binding_complete = True
                    return True
            else:
                counting = False

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return False
            if key in (ord("r"), ord("R")):
                self.hand_slots = [None, None]
                counting = False

    # ---------- 核心逻辑 ----------
    def generate_food(self) -> PointI:
        """随机生成食物，需与蛇头保持足够距离。"""
        margin = 50
        xmin, xmax, ymin, ymax = self.play_bounds()
        while True:
            x = random.randint(xmin + margin, xmax - margin)
            y = random.randint(ymin + margin, ymax - margin)
            ok = True
            for snake in self.snakes:
                if self.dist2((x, y), snake[0]) <= (self.food_radius * 5) ** 2:
                    ok = False
                    break
            if ok:
                return (x, y)

    def step_head(self, idx: int) -> PointF:
        """蛇头向目标点平滑移动。"""
        hx, hy = self.snakes[idx][0]
        tx, ty = self.target_pos[idx]
        dx, dy = tx - hx, ty - hy
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < self.follow_deadzone:
            return (hx, hy)
        ux, uy = dx / dist, dy / dist
        step = min(self.max_step, dist) * self.alpha
        return (hx + ux * step, hy + uy * step)

    def rebuild_body(self, idx: int, new_head: PointF) -> None:
        """按固定节距重建蛇身，保持平滑。"""
        pts = [new_head]
        prev = new_head
        for seg in self.snakes[idx][1:]:
            dx, dy = prev[0] - seg[0], prev[1] - seg[1]
            d = (dx * dx + dy * dy) ** 0.5
            if d < 1e-5:
                pts.append(prev)
                continue
            dirx, diry = dx / d, dy / d
            nx = prev[0] - dirx * self.segment_length
            ny = prev[1] - diry * self.segment_length
            pts.append((nx, ny))
            prev = (nx, ny)
        self.snakes[idx] = pts

    def move_snake(self, idx: int) -> None:
        """推进一次蛇的移动（包含吃食物、碰撞检测）。"""
        if self.game_overs[idx]:
            return
        now = time.time()
        if now - self.last_move_times[idx] < self.move_interval:
            return
        self.last_move_times[idx] = now

        new_head = self.step_head(idx)
        if self.dist2(new_head, self.snakes[idx][0]) < 1e-6:
            return

        # 吃食物
        if self.dist2(new_head, self.food) < (self.head_radius + self.food_radius) ** 2:
            self.scores[idx] += 1
            self.snakes[idx].append(self.snakes[idx][-1])
            self.food = self.generate_food()
            if self.scores[idx] >= self.target_score:
                for i in range(self.num_snakes):
                    self.game_overs[i] = True
                if self.num_snakes == 1:
                    self.result_text = f"You win! Score {self.scores[0]} >= {self.target_score}"
                else:
                    self.result_text = "Green side wins by score" if idx == 0 else "Yellow side wins by score"
                return

        # 重建身体
        self.rebuild_body(idx, new_head)

        # 自撞
        for seg in self.snakes[idx][3:]:
            if self.dist2(new_head, seg) < (self.head_radius + self.body_radius) ** 2:
                self._handle_self_crash(idx)
                return

        # 互撞
        for j in range(self.num_snakes):
            if j == idx:
                continue
            for seg in self.snakes[j]:
                if self.dist2(new_head, seg) < (self.head_radius + self.body_radius) ** 2:
                    self._handle_head_to_other(idx, j)
                    return

        # 撞墙
        x, y = new_head
        xmin, xmax, ymin, ymax = self.play_bounds()
        if x - self.head_radius < xmin or x + self.head_radius > xmax or y - self.head_radius < ymin or y + self.head_radius > ymax:
            self._handle_wall_hit(idx)
            return

    def _handle_self_crash(self, idx: int) -> None:
        self.game_overs[idx] = True
        if self.num_snakes == 1:
            self.result_text = "You lost (self collision)"
        else:
            other = 1 - idx
            self.game_overs[other] = True
            self.result_text = (
                "Yellow side wins! Green side crashed into itself."
                if idx == 0
                else "Green side wins! Yellow side crashed into itself."
            )

    def _handle_head_to_other(self, idx: int, other: int) -> None:
        # 两条蛇都死 => 平局
        self.game_overs[idx] = True
        self.game_overs[other] = True
        self.result_text = "Draw! Snakes crashed into each other."

    def _handle_wall_hit(self, idx: int) -> None:
        self.game_overs[idx] = True
        if self.num_snakes == 1:
            self.result_text = "You lost (hit wall)"
        else:
            other = 1 - idx
            self.game_overs[other] = True
            self.result_text = (
                "Yellow side wins! Green side hit the wall."
                if idx == 0
                else "Green side wins! Yellow side hit the wall."
            )

    # ---------- 绘制 ----------
    def draw(self, frame: np.ndarray) -> None:
        canvas = np.zeros_like(frame)
        colors = [(0, 255, 0), (0, 200, 200)]

        # 画 padding 区域（使可玩区域更明显）：上下左右各 10% 填充深色
        xmin, xmax, ymin, ymax = self.play_bounds()
        pad_color = (30, 30, 30)
        alpha_pad = 0.6
        pad_layer = np.zeros_like(frame)
        # 上（取消上方 padding）
        if self.pad_top > 0:
            cv2.rectangle(pad_layer, (0, 0), (self.width, ymin), pad_color, -1)
        # 下（加大 padding）
        cv2.rectangle(pad_layer, (0, ymax + 1), (self.width, self.height), pad_color, -1)
        # 左
        cv2.rectangle(pad_layer, (0, ymin), (xmin, ymax + 1), pad_color, -1)
        # 右
        cv2.rectangle(pad_layer, (xmax + 1, ymin), (self.width, ymax + 1), pad_color, -1)
        frame[:] = cv2.addWeighted(pad_layer, alpha_pad, frame, 1 - alpha_pad, 0)

        # 食物
        cv2.circle(canvas, (int(self.food[0]), int(self.food[1])), self.food_radius, (0, 0, 255), -1)
        cv2.circle(canvas, (int(self.food[0]), int(self.food[1])), self.food_radius, (255, 255, 255), 2)

        # 蛇
        for idx, snake in enumerate(self.snakes):
            base = colors[idx % len(colors)]
            if len(snake) > 1:
                for i in range(len(snake) - 1):
                    p1 = (int(snake[i][0]), int(snake[i][1]))
                    p2 = (int(snake[i + 1][0]), int(snake[i + 1][1]))
                    alpha = 0.6 - (i / len(snake)) * 0.3
                    color = (base[0], int(base[1] * alpha), base[2])
                    cv2.line(canvas, p1, p2, color, 12, cv2.LINE_AA)
            for i, seg in enumerate(snake):
                if i == 0:
                    color, r = base, self.head_radius
                else:
                    alpha = max(0.4, 1.0 - (i / len(snake)) * 0.6)
                    color = (base[0], int(220 * alpha), base[2])
                    r = max(6, int(self.body_radius * (0.9 - 0.4 * (i / len(snake)))))
                cv2.circle(canvas, (int(seg[0]), int(seg[1])), r, color, -1)
                cv2.circle(canvas, (int(seg[0]), int(seg[1])), r, (255, 255, 255), 1)
            # 眼睛
            hx, hy = int(snake[0][0]), int(snake[0][1])
            eye = max(2, self.head_radius // 4)
            for ex, ey in [(hx + eye, hy - eye), (hx + eye, hy + eye)]:
                cv2.circle(canvas, (ex, ey), eye, (0, 0, 0), -1)
            # 识别圈：提示手指需进入阈值内才能控制
            cv2.circle(canvas, (hx, hy), BIND_DIST, (255, 255, 255), 2, cv2.LINE_AA)

        # 叠加到原始帧
        cv2.addWeighted(canvas, 0.7, frame, 0.3, 0, frame)

        # HUD：分数 + 目标分
        cv2.putText(frame, f"Score Green: {self.scores[0]}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        if self.num_snakes > 1:
            cv2.putText(frame, f"Score Yellow: {self.scores[1]}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(frame, f"Target: {self.target_score}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        if any(self.game_overs):
            title = "Game Over"
            msg = self.result_text or "Press R to restart"
            hint = "Press R to restart, Q to quit"
            self._center_text(frame, title, dy=-30, color=(0, 0, 255), scale=1.2, thick=3)
            self._center_text(frame, msg, dy=10, color=(0, 0, 255), scale=0.9, thick=2)
            self._center_text(frame, hint, dy=45, color=(255, 255, 255), scale=0.7, thick=2)

    def _center_text(self, frame: np.ndarray, text: str, dy: int, color, scale=1.0, thick=2) -> None:
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
        pos = ((frame.shape[1] - size[0]) // 2, frame.shape[0] // 2 + dy)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    def _draw_snakes_binding(self, frame: np.ndarray) -> None:
        """在绑定界面上绘制当前蛇的位置，方便手指对准蛇头。"""
        colors = [(0, 255, 0), (0, 200, 200)]
        xmin, xmax, ymin, ymax = self.play_bounds()
        pad_color = (30, 30, 30)
        alpha_pad = 0.6
        pad_layer = np.zeros_like(frame)
        if self.pad_top > 0:
            cv2.rectangle(pad_layer, (0, 0), (self.width, ymin), pad_color, -1)
        cv2.rectangle(pad_layer, (0, ymax + 1), (self.width, self.height), pad_color, -1)
        cv2.rectangle(pad_layer, (0, ymin), (xmin, ymax + 1), pad_color, -1)
        cv2.rectangle(pad_layer, (xmax + 1, ymin), (self.width, ymax + 1), pad_color, -1)
        frame[:] = cv2.addWeighted(pad_layer, alpha_pad, frame, 1 - alpha_pad, 0)

        for idx, snake in enumerate(self.snakes):
            base = colors[idx % len(colors)]
            # 画身体线
            if len(snake) > 1:
                for i in range(len(snake) - 1):
                    p1 = (int(snake[i][0]), int(snake[i][1]))
                    p2 = (int(snake[i + 1][0]), int(snake[i + 1][1]))
                    alpha = 0.6 - (i / len(snake)) * 0.3
                    color = (base[0], int(base[1] * alpha), base[2])
                    cv2.line(frame, p1, p2, color, 12, cv2.LINE_AA)
            # 画节点
            for i, seg in enumerate(snake):
                if i == 0:
                    color, r = base, self.head_radius
                else:
                    alpha = max(0.4, 1.0 - (i / len(snake)) * 0.6)
                    color = (base[0], int(220 * alpha), base[2])
                    r = max(6, int(self.body_radius * (0.9 - 0.4 * (i / len(snake)))))
                cv2.circle(frame, (int(seg[0]), int(seg[1])), r, color, -1)
                cv2.circle(frame, (int(seg[0]), int(seg[1])), r, (255, 255, 255), 1)
            # 画眼睛
            hx, hy = int(snake[0][0]), int(snake[0][1])
            eye = max(2, self.head_radius // 4)
            for ex, ey in [(hx + eye, hy - eye), (hx + eye, hy + eye)]:
                cv2.circle(frame, (ex, ey), eye, (0, 0, 0), -1)
            # 识别圈：提示手指需进入阈值内才能控制
            cv2.circle(frame, (hx, hy), BIND_DIST, (255, 255, 255), 2, cv2.LINE_AA)

    # ---------- 主循环 ----------
    def run(self) -> None:
        print("贪吃蛇（连续坐标，无网格版）启动")
        print("在弹出的窗口中按 1 选择单人，2 选择双人，Q/Esc 退出")

        running = True
        while running:
            # 模式选择：返回 False 代表真正退出
            if not self.select_mode():
                break

            win = "Snake Game - Continuous"
            # 全屏显示（Windows 下用 WINDOW_NORMAL 创建，再强制 FULLSCREEN，避免保留标题栏）
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self._init_state()

        # 双人模式开局绑定
            if self.mode == "dual":
                if not self.bind_players(win):
                    cv2.destroyAllWindows()
                    # 返回模式选择
                    continue

            print("操作：空格 暂停/继续；R 重开；Q 返回模式选择")

            back_to_menu = False
            fatal_error = False

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    fatal_error = True
                    break
                frame = cv2.flip(frame, 1)
                now_ts = time.time()

                # 1) 检测手，得到食指指尖位置
                hands = self.tracker.detect(frame)
                # 2) 每帧按距离最近的蛇头分配槽位（含缓冲）
                heads = [snake[0] for snake in self.snakes[: self.num_snakes]]
                self._assign_hands_to_slots(hands, heads, now_ts, BIND_DIST)

                # 3) 计算槽位是否仍在“有效期”（2 秒内）
                active = [self._slot_active(i, now_ts) for i in range(self.num_snakes)]
                if self.mode == "single":
                    if active[0]:
                        self._update_target(0, self.hand_slots[0]["pos"], frame.shape)
                    self.auto_paused = not active[0]
                    self.need_rebind_prompt = False
                else:
                    # 双人模式：两个槽都有效才继续；否则自动暂停并提示重绑
                    if self.binding_complete:
                        for i in range(self.num_snakes):
                            if active[i]:
                                self._update_target(i, self.hand_slots[i]["pos"], frame.shape)
                        self.auto_paused = not all(active)
                        self.need_rebind_prompt = self.auto_paused
                    else:
                        # 还未绑定或正在重绑
                        self.auto_paused = True
                        self.need_rebind_prompt = True

                paused = self.auto_paused or self.manual_paused
                if not paused:
                    for i in range(self.num_snakes):
                        if active[i]:
                            self.move_snake(i)

                self.draw(frame)
                if paused:
                    cv2.putText(frame, "Paused", (self.width // 2 - 80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
                    if self.auto_paused:
                        if self.mode == "single":
                            msg = "No hand detected"
                        else:
                            if self.need_rebind_prompt:
                                msg = "Hand lost: Space=rebind  R=restart  Q=mode select"
                            else:
                                msg = "Need two hands (Green+Yellow) to continue"
                        cv2.putText(frame, msg, (self.width // 2 - 250, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    # 返回模式选择
                    back_to_menu = True
                    break
                if key in (ord("r"), ord("R")):
                    self._init_state()
                    if self.mode == "dual":
                        if not self.bind_players(win):
                            back_to_menu = True
                            break
                if key == 32:  # 空格
                    if self.need_rebind_prompt and self.mode == "dual":
                        # 重新绑定再继续
                        if not self.bind_players(win):
                            back_to_menu = True
                            break
                        self.need_rebind_prompt = False
                        self.manual_paused = False
                        self.auto_paused = False
                        continue
                    else:
                        self.manual_paused = not self.manual_paused

                # 在暂停且丢手提示时，提供快速模式选择返回
                if self.need_rebind_prompt and key in (ord("q"), ord("Q")):
                    back_to_menu = True
                    break

            cv2.destroyAllWindows()

            if fatal_error:
                # 摄像头读帧失败等致命错误，直接退出
                running = False
            elif back_to_menu:
                # 回到最外层 while，再次弹出模式选择
                continue
            else:
                # 其它情况（例如窗口被关闭），也直接结束
                running = False

        self.cap.release()
        cv2.destroyAllWindows()

    def _update_target(self, idx: int, finger_pos: PointI, shape) -> None:
        """像素坐标 -> 游戏坐标，写入 target_pos。"""
        fh, fw = shape[:2]
        tx = self.clamp(int(finger_pos[0] / fw * self.width), 0, self.width - 1)
        ty = self.clamp(int(finger_pos[1] / fh * self.height), 0, self.height - 1)
        tx, ty = self.clamp_to_play(tx, ty)
        self.target_pos[idx] = (tx, ty)


if __name__ == "__main__":
    SnakeGame().run()


"""
基于手势的贪吃蛇（连续坐标版）。
使用 MediaPipe Hands 检测食指指尖位置，并将蛇头以平滑方式朝目标点移动。
"""

from __future__ import annotations

import random
import time
from typing import Final, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# 类型别名：PointF 为连续坐标点；PointI 为整数像素坐标点。
PointF = Tuple[float, float]
PointI = Tuple[int, int]


class SnakeGame:
    """
    连续坐标、无网格：平滑且跟手。
    - 头部指数平滑朝指尖移动，限制单帧最大位移防抖。
    - 身体每帧按固定节距重建，无需网格。
    - 碰撞与食物用圆形距离判定。
    """

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        """初始化游戏，并设置影响手感/难度/画面表现的超参数。

        参数：
            width, height:
                游戏逻辑坐标系的宽高（同时也用于设置摄像头期望分辨率）。

        关键超参数说明（按功能分组）：

        1) 身体与视觉（影响“长度观感 / 碰撞判定尺度”）
            segment_length:
                相邻蛇身节点之间的目标间距。越大蛇看起来越“松”、转弯更钝。
            min_segments:
                初始蛇身节点数量。越大初始更长，难度更高。
            head_radius / body_radius:
                绘制半径，同时也间接影响自碰撞阈值（以圆形距离判定）。
            food_radius:
                食物绘制半径，同时影响“吃到食物”的判定范围。

        2) 跟手与平滑（影响“响应速度 / 抖动 / 手感”）
            alpha:
                跟随强度（平滑系数）。越大越跟手但更抖；越小越平滑但更迟滞。
            max_step:
                单次更新蛇头最大位移（限速）。用于避免手抖或目标跳变导致瞬移。
            follow_deadzone:
                死区半径：当蛇头离目标很近时不移动，减少细小抖动造成的晃动。

        3) 更新频率（影响“速度 / 机器负载”）
            move_interval:
                最小移动间隔（秒）。越小移动越频繁、整体更快；越大更慢更稳。

        4) 暂停状态（影响交互）
            manual_paused:
                空格键手动暂停开关。
            auto_paused:
                未检测到手时自动暂停，用于避免控制丢失时蛇继续乱跑。
        """
        self.width: int = width
        self.height: int = height

        # 身体与视觉参数
        self.segment_length: int = 18
        self.min_segments: int = 6
        self.head_radius: int = 12
        self.body_radius: int = 9
        self.food_radius: int = 12

        # 跟手与平滑
        self.alpha: float = 0.35
        self.max_step: float = 42
        self.follow_deadzone: float = 5

        # 更新频率
        self.move_interval: float = 0.02
        self.last_move_time: float = 0.0

        # 状态
        cx, cy = self.width // 2, self.height // 2
        # 初始蛇体按固定间距展开，避免重叠导致开局自撞
        self.snake: list[PointF] = [(cx - i * self.segment_length, cy) for i in range(self.min_segments)]
        self.score: int = 0
        self.game_over: bool = False
        self.target_pos: PointI = (cx, cy)
        self.food: PointI = self.generate_food()
        self.manual_paused: bool = False  # 空格手动暂停
        self.auto_paused: bool = False    # 无手检测自动暂停

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # ---------------- 工具 ----------------
    def clamp(self, v: int, lo: int, hi: int) -> int:
        """将整数 v 限制在闭区间 [lo, hi] 内。"""
        return max(lo, min(hi, v))

    def dist2(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """二维点之间的欧式距离平方。"""
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]
        return dx * dx + dy * dy

    # ---------------- 核心逻辑 ----------------
    def generate_food(self) -> PointI:
        """随机生成食物位置，并与蛇头保持一定距离。"""
        margin: Final[int] = 50
        while True:
            x = random.randint(margin, self.width - margin)
            y = random.randint(margin, self.height - margin)
            if self.dist2((x, y), self.snake[0]) > (self.food_radius * 5) ** 2:
                return (x, y)

    def detect_hand(self, frame: np.ndarray) -> Optional[PointI]:
        """检测当前帧中的食指指尖（landmark 8）。

        返回：
            (x, y) 指尖在摄像头画面中的像素坐标；若未检测到手则返回 None。
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        finger_pos: Optional[PointI] = None
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            p = lm.landmark[8]
            finger_pos = (int(p.x * w), int(p.y * h))
            cv2.circle(frame, finger_pos, 10, (0, 255, 0), -1)
        return finger_pos

    def update_target(self, finger_pos: Optional[PointI], frame_shape: Tuple[int, int, int]) -> None:
        """根据指尖像素坐标更新游戏坐标系中的目标点。"""
        if finger_pos is None:
            return
        fh, fw = frame_shape[:2]
        # 将摄像头像素坐标映射到游戏坐标系。
        tx = self.clamp(int(finger_pos[0] / fw * self.width), 0, self.width - 1)
        ty = self.clamp(int(finger_pos[1] / fh * self.height), 0, self.height - 1)
        self.target_pos = (tx, ty)

    def step_head(self) -> PointF:
        """计算下一步蛇头位置（连续坐标），朝当前目标点移动。"""
        hx, hy = self.snake[0]
        tx, ty = self.target_pos
        dx, dy = tx - hx, ty - hy
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < self.follow_deadzone:
            return (hx, hy)
        ux, uy = dx / dist, dy / dist
        step = min(self.max_step, dist) * self.alpha
        return (hx + ux * step, hy + uy * step)

    def rebuild_body(self, new_head: PointF) -> None:
        """重建蛇身，使相邻节点间距保持为 segment_length。"""
        points: list[PointF] = [new_head]
        prev: PointF = new_head

        # 从旧蛇的第1段(索引1)开始重建；旧头(索引0)不作为身体参考
        for i in range(1, len(self.snake)):
            cur = self.snake[i]
            dx, dy = prev[0] - cur[0], prev[1] - cur[1]
            d = (dx * dx + dy * dy) ** 0.5
            if d < 1e-5:
                points.append(prev)
                continue
            dirx, diry = dx / d, dy / d
            nx = prev[0] - dirx * self.segment_length
            ny = prev[1] - diry * self.segment_length
            points.append((nx, ny))
            prev = (nx, ny)

        self.snake = points

    def check_collision(self, new_head: PointF) -> bool:
        """判断给定蛇头位置是否会发生碰撞。"""
        x, y = new_head
        # 撞墙
        if x - self.head_radius < 0 or x + self.head_radius > self.width or y - self.head_radius < 0 or y + self.head_radius > self.height:
            return True
        # 撞身体（跳过头后三节，减少误判）
        for seg in self.snake[3:]:
            if self.dist2(new_head, seg) < (self.head_radius + self.body_radius) ** 2:
                return True
        return False

    def move_snake(self) -> None:
        """推进一次蛇的移动（带频率限制）。"""
        if self.game_over:
            return
        now = time.time()
        if now - self.last_move_time < self.move_interval:
            return
        self.last_move_time = now

        new_head = self.step_head()

        # 若头部未移动，直接跳过本帧，避免身体挤压到头部
        if self.dist2(new_head, self.snake[0]) < 1e-6:
            return

        # 食物检测（宽容）
        if self.dist2(new_head, self.food) < (self.head_radius + self.food_radius) ** 2:
            self.score += 1
            tail = self.snake[-1]
            self.snake.append(tail)
            self.food = self.generate_food()

        # 先按新头重建身体，再做自碰撞检测，避免用旧身体位置误判
        self.rebuild_body(new_head)

        # 自碰撞检测（跳过头后三节，减少误判）
        for seg in self.snake[3:]:
            if self.dist2(new_head, seg) < (self.head_radius + self.body_radius) ** 2:
                self.game_over = True
                return

        # 撞墙检测
        x, y = new_head
        if x - self.head_radius < 0 or x + self.head_radius > self.width or y - self.head_radius < 0 or y + self.head_radius > self.height:
            self.game_over = True
            return

    # ---------------- 绘制 ----------------
    def draw(self, frame: np.ndarray) -> None:
        """将游戏元素（蛇、食物、分数等）绘制到摄像头画面上。"""
        canvas = np.zeros_like(frame)

        # 食物
        fx, fy = int(self.food[0]), int(self.food[1])
        cv2.circle(canvas, (fx, fy), self.food_radius, (0, 0, 255), -1)
        cv2.circle(canvas, (fx, fy), self.food_radius, (255, 255, 255), 2)

        # 身体线
        if len(self.snake) > 1:
            for i in range(len(self.snake) - 1):
                p1 = (int(self.snake[i][0]), int(self.snake[i][1]))
                p2 = (int(self.snake[i + 1][0]), int(self.snake[i + 1][1]))
                alpha = 0.6 - (i / len(self.snake)) * 0.3
                color = (0, int(200 * alpha), 0)
                cv2.line(canvas, p1, p2, color, 12, cv2.LINE_AA)

        # 节点
        for i, seg in enumerate(self.snake):
            if i == 0:
                color = (0, 255, 0)
                r = self.head_radius
            else:
                alpha = max(0.4, 1.0 - (i / len(self.snake)) * 0.6)
                color = (0, int(220 * alpha), 0)
                r = max(6, int(self.body_radius * (0.9 - 0.4 * (i / len(self.snake)))))
            cv2.circle(canvas, (int(seg[0]), int(seg[1])), r, color, -1)
            cv2.circle(canvas, (int(seg[0]), int(seg[1])), r, (255, 255, 255), 1)

        # 头部眼睛
        hx, hy = int(self.snake[0][0]), int(self.snake[0][1])
        eye = max(2, self.head_radius // 4)
        eyes = [(hx + eye, hy - eye), (hx + eye, hy + eye)]
        for ex, ey in eyes:
            cv2.circle(canvas, (ex, ey), eye, (0, 0, 0), -1)

        # 叠加
        cv2.addWeighted(canvas, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"Score: {self.score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

        if self.game_over:
            text = "Game Over! Press R to restart"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.putText(frame, text, ((frame.shape[1] - size[0]) // 2, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ---------------- 控制 ----------------
    def reset(self) -> None:
        """重置游戏状态（蛇、分数、食物等）。"""
        cx, cy = self.width // 2, self.height // 2
        # 重置时同样按间距展开，防止重叠自撞
        self.snake = [(cx - i * self.segment_length, cy) for i in range(self.min_segments)]
        self.score = 0
        self.game_over = False
        self.target_pos = (cx, cy)
        self.last_move_time = 0
        self.food = self.generate_food()

    def run(self) -> None:
        """主循环：读取摄像头帧、更新游戏、渲染并处理按键。"""
        print("贪吃蛇（连续坐标，无网格版）启动")
        print("操作：食指指尖控制蛇头；R 重开；Q 退出；空格 暂停/继续；无手自动暂停")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            finger_pos = self.detect_hand(frame)

            # 自动暂停：检测不到手时暂停，检测到手时恢复（若未手动暂停）
            self.auto_paused = finger_pos is None

            if not self.auto_paused:
                self.update_target(finger_pos, frame.shape)

            paused = self.manual_paused or self.auto_paused

            if not paused:
                self.move_snake()
            self.draw(frame)
            if paused:
                cv2.putText(frame, "Paused", (self.width // 2 - 80, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
                if self.auto_paused:
                    cv2.putText(frame, "No hand detected", (self.width // 2 - 150, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow("Snake Game - Continuous", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                self.reset()
            if key == 32:  # 空格键
                self.manual_paused = not self.manual_paused

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    game = SnakeGame()
    game.run()
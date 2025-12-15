"""
基于手势的贪吃蛇（连续坐标版）。
使用 MediaPipe Hands 检测食指指尖位置，并将蛇头以平滑方式朝目标点移动。
"""

import random
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

USE_JETSON = False  # False 本地调试，True Jetson Nano

# 类型别名：PointF 为连续坐标点；PointI 为整数像素坐标点。
PointF = Tuple[float, float]
PointI = Tuple[int, int]


def gstreamer_pipeline(
    sensor_id=0,
    sensor_mode=4,          # 1280x720 @ 60fps
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=2
):
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

        # -------- 多蛇状态（支持单人/双人模式） --------
        # 默认单人，run 时会询问模式并重置
        self.mode: str = "single"  # "single" 或 "dual"
        self.num_snakes: int = 1
        self.snakes: list[list[PointF]] = []
        self.scores: list[int] = []
        self.game_overs: list[bool] = []
        self.target_pos: list[PointI] = []
        self.food: PointI = (0, 0)
        self.last_move_times: list[float] = []
        self._init_state()

        # 全局暂停状态
        self.manual_paused: bool = False  # 空格手动暂停
        self.auto_paused: bool = False    # 无手（两只手都没检测到）自动暂停

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 摄像头
        if USE_JETSON:
            pipeline = gstreamer_pipeline(
                capture_width=width,
                capture_height=height,
                display_width=width,
                display_height=height,
                framerate=30,
                flip_method=2
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                raise RuntimeError("❌ 无法通过 GStreamer 打开摄像头")
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # ---------------- 工具 ----------------

    def _init_state(self) -> None:
        """根据当前 num_snakes 初始化/重置状态。"""
        cx, cy = self.width // 2, self.height // 2
        if self.num_snakes == 1:
            spawn_offsets = [0]
        else:
            # 两条蛇左右分开
            spawn_offsets = [-self.width // 4, self.width // 4]

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
    def clamp(self, v: int, lo: int, hi: int) -> int:
        """将整数 v 限制在闭区间 [lo, hi] 内。"""
        return max(lo, min(hi, v))

    def dist2(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """二维点之间的欧式距离平方。"""
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]
        return dx * dx + dy * dy

    # ---------------- 核心逻辑 ----------------
    def generate_food(self) -> PointI:
        """随机生成公共食物位置，并与所有蛇的蛇头保持一定距离。"""
        margin: Final[int] = 50
        while True:
            x = random.randint(margin, self.width - margin)
            y = random.randint(margin, self.height - margin)
            ok = True
            for snake in self.snakes:
                if self.dist2((x, y), snake[0]) <= (self.food_radius * 5) ** 2:
                    ok = False
                    break
            if ok:
                return (x, y)

    def detect_hands(self, frame: np.ndarray) -> list[dict]:
        """检测当前帧中的双手食指指尖，返回每只手的信息列表。

        返回的每个元素为：
            {
                "pos": (x, y),   # 指尖像素坐标
                "label": "Left" / "Right"  # 手的左右属性
            }
        若未检测到手，则返回空列表。
        """
        # 将图像从 BGR 格式转换为 RGB 格式
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 检测手部
        results = self.hands.process(rgb)
        hands_info: list[dict] = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 绘制骨架
                self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
                h, w, _ = frame.shape
                # 拿到食指指尖点位
                p = lm.landmark[8]
                # 转换为像素坐标
                finger_pos: PointI = (int(p.x * w), int(p.y * h))
                # 在指尖位置绘制绿色圆圈
                cv2.circle(frame, finger_pos, 10, (0, 255, 0), -1)

                label = handedness.classification[0].label  # "Left" 或 "Right"
                hands_info.append({"pos": finger_pos, "label": label})

        return hands_info

    def update_target(self, snake_index: int, finger_pos: PointI, frame_shape: Tuple[int, int, int]) -> None:
        """根据某只手的指尖像素坐标，更新对应蛇的目标点。"""
        fh, fw = frame_shape[:2]
        # 将摄像头像素坐标映射到游戏坐标系。
        tx = self.clamp(int(finger_pos[0] / fw * self.width), 0, self.width - 1)
        ty = self.clamp(int(finger_pos[1] / fh * self.height), 0, self.height - 1)
        self.target_pos[snake_index] = (tx, ty)

    def step_head(self, snake_index: int) -> PointF:
        """计算某条蛇下一步蛇头位置（连续坐标），朝当前目标点移动。"""
        hx, hy = self.snakes[snake_index][0]
        tx, ty = self.target_pos[snake_index]
        dx, dy = tx - hx, ty - hy
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < self.follow_deadzone:
            return (hx, hy)
        ux, uy = dx / dist, dy / dist
        step = min(self.max_step, dist) * self.alpha
        return (hx + ux * step, hy + uy * step)

    def rebuild_body(self, snake_index: int, new_head: PointF) -> None:
        """重建某条蛇的蛇身，使相邻节点间距保持为 segment_length。"""
        points: list[PointF] = [new_head]
        prev: PointF = new_head

        old_snake = self.snakes[snake_index]

        # 从旧蛇的第1段(索引1)开始重建；旧头(索引0)不作为身体参考
        for i in range(1, len(old_snake)):
            cur = old_snake[i]
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

        self.snakes[snake_index] = points

    def check_collision(self, snake_index: int, new_head: PointF) -> bool:
        """判断给定蛇头位置是否会发生碰撞（与墙或自身）。"""
        x, y = new_head
        # 撞墙
        if x - self.head_radius < 0 or x + self.head_radius > self.width or y - self.head_radius < 0 or y + self.head_radius > self.height:
            return True
        # 撞身体（跳过头后三节，减少误判）
        for seg in self.snakes[snake_index][3:]:
            if self.dist2(new_head, seg) < (self.head_radius + self.body_radius) ** 2:
                return True
        return False

    def move_snake(self, snake_index: int) -> None:
        """推进某条蛇的移动（带频率限制）。"""
        if self.game_overs[snake_index]:
            return
        now = time.time()
        if now - self.last_move_times[snake_index] < self.move_interval:
            return
        self.last_move_times[snake_index] = now

        new_head = self.step_head(snake_index)

        # 若头部未移动，直接跳过本帧，避免身体挤压到头部
        if self.dist2(new_head, self.snakes[snake_index][0]) < 1e-6:
            return

        # 食物检测（宽容，公共食物，谁先吃到算谁的）
        if self.dist2(new_head, self.food) < (self.head_radius + self.food_radius) ** 2:
            self.scores[snake_index] += 1
            tail = self.snakes[snake_index][-1]
            self.snakes[snake_index].append(tail)
            self.food = self.generate_food()

        # 先按新头重建身体，再做自碰撞检测，避免用旧身体位置误判
        self.rebuild_body(snake_index, new_head)

        # 自碰撞检测（跳过头后三节，减少误判）
        for seg in self.snakes[snake_index][3:]:
            if self.dist2(new_head, seg) < (self.head_radius + self.body_radius) ** 2:
                self.game_overs[snake_index] = True
                return

        # 撞墙检测
        x, y = new_head
        if x - self.head_radius < 0 or x + self.head_radius > self.width or y - self.head_radius < 0 or y + self.head_radius > self.height:
            self.game_overs[snake_index] = True
            return

    # ---------------- 绘制 ----------------
    def draw(self, frame: np.ndarray) -> None:
        """将游戏元素（多条蛇、食物、分数等）绘制到摄像头画面上。"""
        canvas = np.zeros_like(frame)

        # 为不同蛇设置不同的主体颜色（绿色 / 青色）
        snake_colors = [(0, 255, 0), (0, 200, 200)]

        # 公共食物（只画一次）
        fx, fy = int(self.food[0]), int(self.food[1])
        cv2.circle(canvas, (fx, fy), self.food_radius, (0, 0, 255), -1)
        cv2.circle(canvas, (fx, fy), self.food_radius, (255, 255, 255), 2)

        for idx in range(self.num_snakes):
            snake = self.snakes[idx]

            # 身体线
            if len(snake) > 1:
                for i in range(len(snake) - 1):
                    p1 = (int(snake[i][0]), int(snake[i][1]))
                    p2 = (int(snake[i + 1][0]), int(snake[i + 1][1]))
                    alpha = 0.6 - (i / len(snake)) * 0.3
                    base_color = snake_colors[idx % len(snake_colors)]
                    color = (base_color[0], int(base_color[1] * alpha), base_color[2])
                    cv2.line(canvas, p1, p2, color, 12, cv2.LINE_AA)

            # 节点
            for i, seg in enumerate(snake):
                if i == 0:
                    color = snake_colors[idx % len(snake_colors)]
                    r = self.head_radius
                else:
                    alpha = max(0.4, 1.0 - (i / len(snake)) * 0.6)
                    base_color = snake_colors[idx % len(snake_colors)]
                    color = (base_color[0], int(220 * alpha), base_color[2])
                    r = max(6, int(self.body_radius * (0.9 - 0.4 * (i / len(snake)))))
                cv2.circle(canvas, (int(seg[0]), int(seg[1])), r, color, -1)
                cv2.circle(canvas, (int(seg[0]), int(seg[1])), r, (255, 255, 255), 1)

            # 头部眼睛
            hx, hy = int(snake[0][0]), int(snake[0][1])
            eye = max(2, self.head_radius // 4)
            eyes = [(hx + eye, hy - eye), (hx + eye, hy + eye)]
            for ex, ey in eyes:
                cv2.circle(canvas, (ex, ey), eye, (0, 0, 0), -1)

        # 叠加
        cv2.addWeighted(canvas, 0.7, frame, 0.3, 0, frame)

        # 分数显示：左蛇 / 右蛇
        cv2.putText(frame, f"Score L: {self.scores[0]}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        if self.num_snakes > 1:
            cv2.putText(frame, f"Score R: {self.scores[1]}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        if any(self.game_overs):
            text = "Game Over! Press R to restart"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.putText(frame, text, ((frame.shape[1] - size[0]) // 2, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ---------------- 控制 ----------------
    def reset(self) -> None:
        """重置游戏状态（多条蛇、分数、食物等）。"""
        self._init_state()

    def run(self) -> None:
        """主循环：读取摄像头帧、更新游戏、渲染并处理按键。"""
        print("贪吃蛇（连续坐标，无网格版）启动")
        print("选择模式：1 单人（单手控制一条蛇），2 双人（双手各控一条蛇）")

        # 选择模式
        while True:
            choice = input("请选择模式 (1 单人 / 2 双人): ").strip()
            if choice in ("1", "2"):
                break
            print("输入无效，请输入 1 或 2")

        if choice == "1":
            self.mode = "single"
            self.num_snakes = 1
        else:
            self.mode = "dual"
            self.num_snakes = 2

        self.reset()

        print("操作：")
        if self.mode == "single":
            print("  单人模式：任意一只手食指指尖控制蛇；无手自动暂停；空格 暂停/继续；R 重开；Q 退出")
        else:
            print("  双人模式：左手控制左蛇，右手控制右蛇；若仅一只手被检测到则自动暂停；空格 暂停/继续；R 重开；Q 退出")

        while True:
            # cv2.VideoCapture对象捕获摄像头图像帧
            # ret是Boolean值标志是否成功
            ret, frame = self.cap.read()
            if not ret:
                break
            # 水平翻转摄像头
            frame = cv2.flip(frame, 1)

            # 拿到所有检测到的手的信息
            hands_info = self.detect_hands(frame)

            # 为每条蛇记录当前帧是否有对应的手
            has_hand_for_snake = [False for _ in range(self.num_snakes)]

            if self.mode == "single":
                # 单人：只控制蛇 0。检测到任何一只手即控制；未检测到手则自动暂停
                if hands_info:
                    pos: PointI = hands_info[0]["pos"]
                    has_hand_for_snake[0] = True
                    self.update_target(0, pos, frame.shape)
                self.auto_paused = not has_hand_for_snake[0]
            else:
                # 双人：需要左右手各控制一条蛇；若未检测到两只手则自动暂停
                if len(hands_info) >= 2:
                    assigned = [False, False]
                    # 先按标签分配
                    for hinfo in hands_info:
                        pos: PointI = hinfo["pos"]
                        label: str = hinfo["label"]
                        if label == "Left":
                            has_hand_for_snake[0] = True
                            self.update_target(0, pos, frame.shape)
                            assigned[0] = True
                        elif label == "Right":
                            has_hand_for_snake[1] = True
                            self.update_target(1, pos, frame.shape)
                            assigned[1] = True
                    # 若标签不全，按检测顺序补齐前两只手
                    idx_fill = 0
                    for hinfo in hands_info:
                        if assigned[0] and assigned[1]:
                            break
                        pos: PointI = hinfo["pos"]
                        if not assigned[0]:
                            has_hand_for_snake[0] = True
                            self.update_target(0, pos, frame.shape)
                            assigned[0] = True
                            continue
                        if not assigned[1]:
                            has_hand_for_snake[1] = True
                            self.update_target(1, pos, frame.shape)
                            assigned[1] = True
                    # 需要两只手都在，才允许移动
                    self.auto_paused = not all(has_hand_for_snake)
                else:
                    # 只检测到 0/1 只手，双人模式下自动暂停
                    self.auto_paused = True

            paused = self.manual_paused or self.auto_paused

            if not paused:
                # 只有检测到对应手的蛇才会移动
                for idx in range(self.num_snakes):
                    if has_hand_for_snake[idx]:
                        self.move_snake(idx)

            # 绘制
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
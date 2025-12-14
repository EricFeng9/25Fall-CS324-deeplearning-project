   
import cv2
import mediapipe as mp
import numpy as np
import random
import time


class SnakeGame:
    """
    连续坐标、去网格版本：更平滑、更跟手、低抖动。
    - 蛇使用连续坐标，各节间距固定 segment_length
    - 头部朝指尖指数平滑逼近，限制单帧最大位移，防抖同时更跟手
    - 身体每帧按固定间距重建，无需网格、无四向离散
    - 碰撞使用圆形距离，宽容但可靠；食物也用圆形距离判定
    """

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        # 身体与视觉参数
        self.segment_length = 18          # 每节长度
        self.min_segments = 6             # 初始节数
        self.head_radius = 12
        self.body_radius = 9
        self.food_radius = 12

        # 跟手与平滑
        self.alpha = 0.35                 # 头部朝目标的指数平滑系数
        self.max_step = 42                # 头部单帧最大位移
        self.follow_deadzone = 5          # 指尖距离小于此值时不动

        # 更新频率
        self.move_interval = 0.02         # 50 FPS
        self.last_move_time = 0

        # 状态
        cx, cy = self.width // 2, self.height // 2
        self.snake = [(cx, cy) for _ in range(self.min_segments)]
        self.score = 0
        self.game_over = False
        self.target_pos = (cx, cy)
        self.food = self.generate_food()

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
    def clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def dist2(self, p1, p2):
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]
        return dx * dx + dy * dy

    # ---------------- 核心逻辑 ----------------
    def generate_food(self):
        margin = 50
        while True:
            x = random.randint(margin, self.width - margin)
            y = random.randint(margin, self.height - margin)
            if self.dist2((x, y), self.snake[0]) > (self.food_radius * 5) ** 2:
                return (x, y)

    def detect_hand(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        finger_pos = None
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            p = lm.landmark[8]
            finger_pos = (int(p.x * w), int(p.y * h))
            cv2.circle(frame, finger_pos, 10, (0, 255, 0), -1)
        return finger_pos

    def update_target(self, finger_pos, frame_shape):
        if finger_pos is None:
            return
        fh, fw = frame_shape[:2]
        tx = self.clamp(int(finger_pos[0] / fw * self.width), 0, self.width)
        ty = self.clamp(int(finger_pos[1] / fh * self.height), 0, self.height)
        self.target_pos = (tx, ty)

    def step_head(self):
        hx, hy = self.snake[0]
        tx, ty = self.target_pos
        dx, dy = tx - hx, ty - hy
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < self.follow_deadzone:
            return (hx, hy)
        ux, uy = dx / dist, dy / dist
        step = min(self.max_step, dist) * self.alpha
        return (hx + ux * step, hy + uy * step)

    def rebuild_body(self, new_head):
        points = [new_head]
        prev = new_head
        for i in range(len(self.snake) - 1):
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

    def check_collision(self, new_head):
        x, y = new_head
        # 撞墙
        if x - self.head_radius < 0 or x + self.head_radius > self.width or y - self.head_radius < 0 or y + self.head_radius > self.height:
            return True
        # 撞身体（跳过头后一两节，减少误判）
        for seg in self.snake[3:]:
            if self.dist2(new_head, seg) < (self.head_radius + self.body_radius) ** 2:
                return True
        return False

    def move_snake(self):
        if self.game_over:
            return
        now = time.time()
        if now - self.last_move_time < self.move_interval:
            return
        self.last_move_time = now

        new_head = self.step_head()

        # 食物检测在移动前后都宽容
        if self.dist2(new_head, self.food) < (self.head_radius + self.food_radius) ** 2:
            self.score += 1
            # 延长：在尾部追加一节（复制尾巴）
            tail = self.snake[-1]
            self.snake.append(tail)
            self.food = self.generate_food()

        if self.check_collision(new_head):
            self.game_over = True
            return

        self.rebuild_body(new_head)

    # ---------------- 绘制 ----------------
    def draw(self, frame):
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
    def reset(self):
        cx, cy = self.width // 2, self.height // 2
        self.snake = [(cx, cy) for _ in range(self.min_segments)]
        self.score = 0
        self.game_over = False
        self.target_pos = (cx, cy)
        self.last_move_time = 0
        self.food = self.generate_food()

    def run(self):
        print("贪吃蛇（连续坐标，无网格版）启动")
        print("操作：食指指尖控制蛇头；R 重开；Q 退出")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            finger_pos = self.detect_hand(frame)
            self.update_target(finger_pos, frame.shape)
            self.move_snake()
            self.draw(frame)

            cv2.imshow("Snake Game - Continuous", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                self.reset()

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    game = SnakeGame()
    game.run()
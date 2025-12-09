import cv2
import mediapipe as mp
import numpy as np
import random
import time


class SnakeGame:
    """
    连续坐标、无网格：平滑且跟手。
    - 头部指数平滑朝指尖移动，限制单帧最大位移防抖。
    - 身体每帧按固定节距重建，无需网格。
    - 碰撞与食物用圆形距离判定。
    """

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        # 身体与视觉参数
        self.segment_length = 18
        self.min_segments = 6
        self.head_radius = 12
        self.body_radius = 9
        self.food_radius = 12

        # 跟手与平滑
        self.alpha = 0.35
        self.max_step = 42
        self.follow_deadzone = 5

        # 更新频率
        self.move_interval = 0.02
        self.last_move_time = 0

        # 状态
        cx, cy = self.width // 2, self.height // 2
        # 初始蛇体按固定间距展开，避免重叠导致开局自撞
        self.snake = [(cx - i * self.segment_length, cy) for i in range(self.min_segments)]
        self.score = 0
        self.game_over = False
        self.target_pos = (cx, cy)
        self.food = self.generate_food()
        self.manual_paused = False  # 空格手动暂停
        self.auto_paused = False    # 无手检测自动暂停

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
        # 撞身体（跳过头后三节，减少误判）
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
        # 重置时同样按间距展开，防止重叠自撞
        self.snake = [(cx - i * self.segment_length, cy) for i in range(self.min_segments)]
        self.score = 0
        self.game_over = False
        self.target_pos = (cx, cy)
        self.last_move_time = 0
        self.food = self.generate_food()

    def run(self):
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
import cv2
import mediapipe as mp
import numpy as np
import random
import time

class SnakeGame:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.grid_size = 20  # 网格大小
        self.safe_margin = self.grid_size * 2  # 安全边界，食物不会生成在太靠近边缘的位置
        
        # 初始化贪吃蛇
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = (0, 0)  # 初始方向
        self.food = self.generate_food()
        self.score = 0
        self.game_over = False
        
        # MediaPipe手部追踪
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # OpenCV摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 用于平滑移动
        self.last_finger_pos = None
        # 响应更快：进一步减小平滑系数
        self.smooth_factor = 0.12
        self.target_pos = None  # 目标位置
        # 只有距离超过此阈值才更新方向，减小以更灵敏
        self.move_threshold = self.grid_size * 0.5
        # 步长（动态调整）
        self.step_size = self.grid_size
        
        # 控制移动频率，避免移动太快
        self.last_move_time = 0
        # 再提高刷新频率，更跟手
        self.move_interval = 0.02  # 每0.02秒移动一次（50次/秒）
        
    def generate_food(self):
        """生成食物位置，避免生成在边缘"""
        max_attempts = 100  # 最大尝试次数，避免无限循环
        attempts = 0
        
        while attempts < max_attempts:
            # 留出安全边界，确保食物不会生成在太靠近边缘的位置
            min_x = self.safe_margin
            max_x = self.width - self.grid_size - self.safe_margin
            min_y = self.safe_margin
            max_y = self.height - self.grid_size - self.safe_margin
            
            # 确保有足够的空间
            if max_x < min_x or max_y < min_y:
                # 如果空间太小，使用原来的逻辑
                food_x = random.randint(0, (self.width - self.grid_size) // self.grid_size) * self.grid_size
                food_y = random.randint(0, (self.height - self.grid_size) // self.grid_size) * self.grid_size
            else:
                # 在安全区域内生成
                food_x = random.randint(min_x // self.grid_size, max_x // self.grid_size) * self.grid_size
                food_y = random.randint(min_y // self.grid_size, max_y // self.grid_size) * self.grid_size
            
            food_pos = (food_x, food_y)
            if food_pos not in self.snake:
                return food_pos
            
            attempts += 1
        
        # 如果尝试多次都失败，使用原来的逻辑作为后备
        food_x = random.randint(0, (self.width - self.grid_size) // self.grid_size) * self.grid_size
        food_y = random.randint(0, (self.height - self.grid_size) // self.grid_size) * self.grid_size
        return (food_x, food_y)
    
    def detect_hand(self, frame):
        """检测手部并返回食指指尖位置"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        finger_pos = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # 获取食指指尖位置（landmark 8）
                h, w, _ = frame.shape
                landmark_8 = hand_landmarks.landmark[8]
                finger_x = int(landmark_8.x * w)
                finger_y = int(landmark_8.y * h)
                finger_pos = (finger_x, finger_y)
                
                # 在指尖位置绘制圆圈
                cv2.circle(frame, finger_pos, 10, (0, 255, 0), -1)
        
        return finger_pos
    
    def update_direction(self, finger_pos, frame_shape):
        """根据食指位置更新贪吃蛇移动方向"""
        if finger_pos is None:
            return
        
        # 获取摄像头画面尺寸
        frame_h, frame_w = frame_shape[:2]
        
        # 将食指位置映射到游戏画布坐标系统
        # 因为游戏画布会被resize到摄像头画面大小，所以需要映射
        target_x = int((finger_pos[0] / frame_w) * self.width)
        target_y = int((finger_pos[1] / frame_h) * self.height)
        
        # 对齐到网格
        target_x = (target_x // self.grid_size) * self.grid_size
        target_y = (target_y // self.grid_size) * self.grid_size
        
        # 确保在边界内
        target_x = max(0, min(target_x, self.width - self.grid_size))
        target_y = max(0, min(target_y, self.height - self.grid_size))
        
        target_pos = (target_x, target_y)
        
        # 平滑处理
        if self.last_finger_pos is None:
            self.last_finger_pos = target_pos
        
        # 使用平滑因子减少抖动
        smooth_x = int(self.last_finger_pos[0] * (1 - self.smooth_factor) + 
                      target_pos[0] * self.smooth_factor)
        smooth_y = int(self.last_finger_pos[1] * (1 - self.smooth_factor) + 
                      target_pos[1] * self.smooth_factor)
        smooth_pos = (smooth_x, smooth_y)
        self.last_finger_pos = smooth_pos
        
        # 保存目标位置
        self.target_pos = smooth_pos
        
        # 计算蛇头到目标位置的方向
        head_x, head_y = self.snake[0]
        dx = smooth_x - head_x
        dy = smooth_y - head_y
        
        # 计算距离
        distance = (dx**2 + dy**2)**0.5

        # 根据距离动态调整步长，距离越远步长越大
        if distance > self.grid_size * 6:
            self.step_size = self.grid_size * 3
        elif distance > self.grid_size * 3:
            self.step_size = self.grid_size * 2
        else:
            self.step_size = self.grid_size
        
        # 只有当距离超过阈值时才改变方向，减少抖动
        if distance < self.move_threshold:
            # 距离太近，保持当前方向或停止
            if self.direction == (0, 0):
                return  # 如果还没开始移动，不改变方向
            # 否则保持当前方向不变
            return
        
        # 确定移动方向（只允许上下左右，不允许斜向）
        # 降低方向切换阈值，提升跟手性
        threshold = self.grid_size * 0.6
        
        if abs(dx) > abs(dy):
            if dx > threshold:
                self.direction = (self.step_size, 0)  # 右
            elif dx < -threshold:
                self.direction = (-self.step_size, 0)  # 左
            # 如果dx在阈值内，保持当前方向（不改变）
        else:
            if dy > threshold:
                self.direction = (0, self.step_size)  # 下
            elif dy < -threshold:
                self.direction = (0, -self.step_size)  # 上
            # 如果dy在阈值内，保持当前方向（不改变）
    
    def move_snake(self):
        """移动贪吃蛇"""
        if self.game_over or self.direction == (0, 0):
            return
        
        # 控制移动频率
        current_time = time.time()
        if current_time - self.last_move_time < self.move_interval:
            return  # 还没到移动时间
        self.last_move_time = current_time
        
        # 计算新头部位置
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        # 检查碰撞（允许走到当前尾巴位置，因为本回合尾巴会被移除）
        will_hit_body = new_head in self.snake and not (len(self.snake) > 1 and new_head == self.snake[-1])
        if self.check_wall(new_head) or will_hit_body:
            self.game_over = True
            return
        
        # 记录移动前的长度
        old_length = len(self.snake)
        
        # 添加新头部
        self.snake.insert(0, new_head)
        
        # 移除尾部（延长已经在检测到食物时立即处理了）
        self.snake.pop()
    
    def check_wall(self, pos):
        """检查撞墙"""
        x, y = pos
        return x < 0 or x + self.grid_size > self.width or y < 0 or y + self.grid_size > self.height
    
    def draw_game(self, frame, finger_pos=None):
        """在帧上绘制游戏元素"""
        h, w = frame.shape[:2]
        
        # 创建游戏画布（使用摄像头画面尺寸，便于直接映射）
        game_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 将游戏坐标映射到摄像头画面坐标
        scale_x = w / self.width
        scale_y = h / self.height
        
        # 绘制食物（圆形）
        food_x = int((self.food[0] + self.grid_size // 2) * scale_x)
        food_y = int((self.food[1] + self.grid_size // 2) * scale_y)
        food_radius = int(self.grid_size * scale_x // 2)
        cv2.circle(game_canvas, (food_x, food_y), food_radius, (0, 0, 255), -1)
        cv2.circle(game_canvas, (food_x, food_y), food_radius, (255, 255, 255), 2)
        
        # 绘制目标位置（如果存在，圆形）
        if self.target_pos is not None:
            target_x = int((self.target_pos[0] + self.grid_size // 2) * scale_x)
            target_y = int((self.target_pos[1] + self.grid_size // 2) * scale_y)
            target_radius = int(self.grid_size * scale_x // 2)
            # 绘制半透明的目标位置指示（圆形）
            overlay_target = game_canvas.copy()
            cv2.circle(overlay_target, (target_x, target_y), target_radius, (255, 255, 0), -1)
            cv2.addWeighted(overlay_target, 0.2, game_canvas, 0.8, 0, game_canvas)
        
        # 绘制贪吃蛇（圆形，更流畅）
        radius = int(self.grid_size * scale_x // 2)
        
        # 先绘制连接线，让蛇看起来更流畅
        if len(self.snake) > 1:
            for i in range(len(self.snake) - 1):
                x1 = int((self.snake[i][0] + self.grid_size // 2) * scale_x)
                y1 = int((self.snake[i][1] + self.grid_size // 2) * scale_y)
                x2 = int((self.snake[i+1][0] + self.grid_size // 2) * scale_x)
                y2 = int((self.snake[i+1][1] + self.grid_size // 2) * scale_y)
                # 身体连接线，颜色逐渐变暗
                body_alpha = 0.6 - (i / len(self.snake)) * 0.3
                line_color = (0, int(200 * body_alpha), 0)
                line_thickness = max(1, int(radius * 1.5))
                cv2.line(game_canvas, (x1, y1), (x2, y2), line_color, line_thickness)
        
        # 绘制蛇的每个节点（圆形）
        for i, segment in enumerate(self.snake):
            # 头部更亮更大，身体逐渐变小变暗
            if i == 0:
                color = (0, 255, 0)  # 头部：亮绿色
                seg_radius = int(radius * 1.2)  # 头部稍大
            else:
                # 身体：逐渐变暗
                alpha = max(0.5, 1.0 - (i / len(self.snake)) * 0.5)
                color = (0, int(200 * alpha), 0)
                seg_radius = int(radius * (1.0 - i / len(self.snake) * 0.2))  # 身体逐渐变小
            
            seg_x = int((segment[0] + self.grid_size // 2) * scale_x)
            seg_y = int((segment[1] + self.grid_size // 2) * scale_y)
            
            # 绘制圆形
            cv2.circle(game_canvas, (seg_x, seg_y), seg_radius, color, -1)
            cv2.circle(game_canvas, (seg_x, seg_y), seg_radius, (255, 255, 255), 1)
            
            # 头部添加眼睛
            if i == 0:
                eye_size = max(2, seg_radius // 4)
                # 根据移动方向确定眼睛位置
                if self.direction == (self.grid_size, 0):  # 向右
                    cv2.circle(game_canvas, (seg_x + eye_size, seg_y - eye_size), eye_size, (0, 0, 0), -1)
                    cv2.circle(game_canvas, (seg_x + eye_size, seg_y + eye_size), eye_size, (0, 0, 0), -1)
                elif self.direction == (-self.grid_size, 0):  # 向左
                    cv2.circle(game_canvas, (seg_x - eye_size, seg_y - eye_size), eye_size, (0, 0, 0), -1)
                    cv2.circle(game_canvas, (seg_x - eye_size, seg_y + eye_size), eye_size, (0, 0, 0), -1)
                elif self.direction == (0, self.grid_size):  # 向下
                    cv2.circle(game_canvas, (seg_x - eye_size, seg_y + eye_size), eye_size, (0, 0, 0), -1)
                    cv2.circle(game_canvas, (seg_x + eye_size, seg_y + eye_size), eye_size, (0, 0, 0), -1)
                elif self.direction == (0, -self.grid_size):  # 向上
                    cv2.circle(game_canvas, (seg_x - eye_size, seg_y - eye_size), eye_size, (0, 0, 0), -1)
                    cv2.circle(game_canvas, (seg_x + eye_size, seg_y - eye_size), eye_size, (0, 0, 0), -1)
        
        # 创建半透明叠加
        overlay = frame.copy()
        alpha = 0.6
        cv2.addWeighted(game_canvas, alpha, overlay, 1 - alpha, 0, frame)
        
        # 绘制分数
        cv2.putText(frame, f'Score: {self.score}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 游戏结束提示
        if self.game_over:
            text = 'Game Over! Press R to restart'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = frame.shape[0] // 2
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    def reset_game(self):
        """重置游戏"""
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = (0, 0)
        self.food = self.generate_food()
        self.score = 0
        self.game_over = False
        self.last_finger_pos = None
        self.target_pos = None
        self.last_move_time = 0
    
    def run(self):
        """运行游戏主循环"""
        print("贪吃蛇游戏启动！")
        print("操作说明：")
        print("- 将食指移动到摄像头前，食指指尖控制贪吃蛇移动")
        print("- 按 'R' 键重新开始游戏")
        print("- 按 'Q' 键退出游戏")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 水平翻转画面（镜像效果）
            frame = cv2.flip(frame, 1)
            
            # 检测手部
            finger_pos = self.detect_hand(frame)
            
            # 更新游戏状态
            if not self.game_over:
                # 每帧都检查当前蛇头是否与食物重叠（确保不会错过）
                if len(self.snake) > 0:
                    head_x, head_y = self.snake[0]
                    head_grid_x = head_x // self.grid_size
                    head_grid_y = head_y // self.grid_size
                    food_grid_x = self.food[0] // self.grid_size
                    food_grid_y = self.food[1] // self.grid_size
                    
                    if head_grid_x == food_grid_x and head_grid_y == food_grid_y:
                        # 吃到食物
                        self.score += 1
                        old_length = len(self.snake)
                        self.food = self.generate_food()
                        
                        # 立即延长蛇：添加一个新的身体段（在当前位置添加，下次移动时会自然延伸）
                        # 如果蛇还没有开始移动，直接添加一个身体段
                        if len(self.snake) > 0:
                            # 在蛇的尾部添加一个新的身体段（复制最后一个位置）
                            last_segment = self.snake[-1]
                            self.snake.append(last_segment)
                            print(f"检测到食物！当前分数: {self.score}, 蛇长度: {old_length} -> {len(self.snake)}")  # 调试信息
                        else:
                            print(f"检测到食物！但蛇为空，无法延长")
                
                if finger_pos is not None:
                    # 如果蛇还没有开始移动，直接将蛇头对齐到食指位置
                    if self.direction == (0, 0) and len(self.snake) == 1:
                        frame_h, frame_w = frame.shape[:2]
                        target_x = int((finger_pos[0] / frame_w) * self.width)
                        target_y = int((finger_pos[1] / frame_h) * self.height)
                        target_x = (target_x // self.grid_size) * self.grid_size
                        target_y = (target_y // self.grid_size) * self.grid_size
                        target_x = max(0, min(target_x, self.width - self.grid_size))
                        target_y = max(0, min(target_y, self.height - self.grid_size))
                        self.snake[0] = (target_x, target_y)
                        self.last_finger_pos = (target_x, target_y)
                    
                    self.update_direction(finger_pos, frame.shape)
                    self.move_snake()
                # 如果检测不到手部，保持当前方向继续移动
                elif self.direction != (0, 0):
                    self.move_snake()
            
            # 绘制游戏
            self.draw_game(frame, finger_pos)
            
            # 显示画面
            cv2.imshow('Snake Game - Hand Tracking', frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                self.reset_game()
        
        # 清理资源
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    game = SnakeGame(width=1280, height=720)
    try:
        game.run()
    except KeyboardInterrupt:
        print("\n游戏退出")
    finally:
        if game.cap.isOpened():
            game.cap.release()
        cv2.destroyAllWindows()


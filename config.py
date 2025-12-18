# -------------------- 配置与常量 --------------------
USE_JETSON = False

# 颜色定义 (RGB) - 马卡龙色系
COL_BG_BLUE = (224, 247, 250)  # 浅蓝背景
COL_SNAKE_BODY = (179, 229, 252)  # 蛇身浅蓝
COL_SNAKE_BORDER = (100, 180, 220)  # 蛇身边框
COL_HUD_BG = (255, 240, 245)  # HUD 背景浅粉
COL_PROGRESS_OK = (144, 238, 144)  # 进度条绿
COL_PROGRESS_LOW = (255, 99, 71)  # 进度条红

# 食物配置 (icon 字段不再用于显示，仅作标记)
FOOD_TYPES = {
    'insect': {'score': 1, 'color': (165, 214, 167), 'prob': 0.5},  # 浅绿
    'fish': {'score': 2, 'color': (144, 202, 249), 'prob': 0.3},  # 浅蓝
    'mouse': {'score': 3, 'color': (255, 224, 130), 'prob': 0.15},  # 浅黄
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
HARD_MODE_FOOD_TIME = 3.0
HUD_HEIGHT = 100

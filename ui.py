import cv2
import numpy as np
import math
import time
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple
from config import *

def get_safe_font(size: int) -> ImageFont.ImageFont:
    """优先加载圆润/现代的系统字体"""
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

def draw_cartoon_head(draw, pos, radius, color, direction_vec):
    """绘制卡通蛇头"""
    x, y = pos
    dx, dy = direction_vec
    angle = math.atan2(dy, dx)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=COL_SNAKE_BORDER, width=2)

    eye_offset_x = math.cos(angle - 0.6) * (radius * 0.5)
    eye_offset_y = math.sin(angle - 0.6) * (radius * 0.5)
    eye2_offset_x = math.cos(angle + 0.6) * (radius * 0.5)
    eye2_offset_y = math.sin(angle + 0.6) * (radius * 0.5)

    eye_r = 4.5
    draw.ellipse(
        (x + eye_offset_x - eye_r, y + eye_offset_y - eye_r, x + eye_offset_x + eye_r, y + eye_offset_y + eye_r),
        fill="white")
    draw.ellipse((x + eye2_offset_x - eye_r, y + eye2_offset_y - eye_r, x + eye2_offset_x + eye_r,
                  y + eye2_offset_y + eye_r), fill="white")
    p_r = 2
    draw.ellipse((x + eye_offset_x - p_r, y + eye_offset_y - p_r, x + eye_offset_x + p_r, y + eye_offset_y + p_r),
                 fill="black")
    draw.ellipse(
        (x + eye2_offset_x - p_r, y + eye2_offset_y - p_r, x + eye2_offset_x + p_r, y + eye2_offset_y + p_r),
        fill="black")

def draw_food_icon(draw, pos, ftype, radius=None, food_radius=15):
    """手绘食物图标"""
    x, y = pos
    r = radius if radius else food_radius
    cfg = FOOD_TYPES[ftype]
    color = cfg['color']

    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline="white", width=2)

    if ftype == 'insect':
        draw.ellipse((x - r * 0.4, y - r * 0.4, x + r * 0.4, y + r * 0.4), fill="red")
        draw.line((x - r * 0.4, y, x + r * 0.4, y), fill="black", width=1)
        draw.line((x, y - r * 0.4, x, y + r * 0.4), fill="black", width=1)
    elif ftype == 'fish':
        draw.ellipse((x - r * 0.5, y - r * 0.3, x + r * 0.3, y + r * 0.3), fill="blue")
        draw.polygon([(x + r * 0.2, y), (x + r * 0.6, y - r * 0.3), (x + r * 0.6, y + r * 0.3)], fill="blue")
    elif ftype == 'mouse':
        draw.ellipse((x - r * 0.5, y - r * 0.5, x, y), fill="grey")
        draw.ellipse((x, y - r * 0.5, x + r * 0.5, y), fill="grey")
        draw.ellipse((x - r * 0.4, y - r * 0.2, x + r * 0.4, y + r * 0.4), fill="lightgrey")
    elif ftype == 'poison':
        draw.rectangle((x - r * 0.3, y - r * 0.2, x + r * 0.3, y + r * 0.4), fill="purple")
        draw.line((x - r * 0.2, y, x + r * 0.2, y + r * 0.3), fill="white", width=1)
        draw.line((x - r * 0.2, y + r * 0.3, x + r * 0.2, y), fill="white", width=1)

def draw_game_overlay(game, frame):
    """Refactored draw_ui_layer taking game instance"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil, 'RGBA')

    # 1. 顶部 HUD
    draw.rectangle((0, 0, game.width, HUD_HEIGHT), fill=COL_HUD_BG)
    draw.line((0, HUD_HEIGHT, game.width, HUD_HEIGHT), fill=(255, 255, 255), width=3)

    # 绘制游戏边界
    xm, ym, xM, yM = game.play_area
    cx, cy = (xm + xM) // 2, (ym + yM) // 2  # use play-area center for overlays
    draw.rectangle((xm, ym, xM, yM), outline=(100, 180, 220), width=3)

    # 2. 时间进度条
    elapsed = game.get_elapsed_time()
    if game.mode == "single":
        cfg = SINGLE_LEVELS[game.current_level_idx]
        total_time = cfg["time"]
        current_level_text = f"LEVEL {game.current_level_idx + 1}"
        target_score = cfg["score"]
        current_score = game.scores[0]
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

    bar_x, bar_y, bar_w, bar_h = 20, 10, game.width - 40, 15
    draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=5, fill=(220, 220, 220))
    draw.rounded_rectangle((bar_x, bar_y, bar_x + int(bar_w * progress), bar_y + bar_h), radius=5, fill=bar_color)

    time_str = f"{int(rem_time)}s"
    tb_time = draw.textbbox((0, 0), time_str, font=game.font_md)
    draw.text(((game.width - (tb_time[2] - tb_time[0])) // 2, bar_y + 25), time_str, font=game.font_md,
                fill=(100, 100, 100))

    # 3. HUD 信息
    if game.mode == "single":
        draw.text((30, 50), current_level_text, font=game.font_md, fill=(100, 100, 100))
        # 单人模式：显示分数
        draw.text((30, 80), f"SCORE: {current_score}/{target_score}", font=game.font_md, fill=(50, 150, 250))
        # 单人模式：显示图例（右侧）
        legend_start_x = game.width - 450
        legends = [('insect', '1'), ('fish', '2'), ('mouse', '3'), ('poison', 'Die')]
        for i, (ftype, val) in enumerate(legends):
            lx = legend_start_x + i * 90
            ly = 65
            draw_food_icon(draw, (lx, ly), ftype, radius=12, food_radius=game.food_radius)
            draw.text((lx + 20, ly - 10), val, font=game.font_sm, fill=(150, 150, 150))
    else:
        # === 双人模式 HUD 布局修复 ===
        draw.text((100, 55), current_level_text, font=game.font_md, fill=(100, 100, 100))
        # 左侧 P1
        draw_text_pill(draw, (20, 55), f"P1: {game.scores[0]}", game.font_md, text_color=(50, 200, 50))
        # 右侧 P2
        p2_text = f"P2: {game.scores[1]}"
        tb = draw.textbbox((0, 0), p2_text, font=game.font_md)
        p2_w = tb[2] - tb[0]
        draw_text_pill(draw, (game.width - p2_w - 40, 55), p2_text, game.font_md, text_color=(0, 200, 200))
        # 中间目标分数 (金色)
        target_str = f"GOAL: {DUAL_TARGET_LENGTH}"
        tb_tgt = draw.textbbox((0, 0), target_str, font=game.font_lg)
        t_w = tb_tgt[2] - tb_tgt[0]
        draw.text(((game.width - t_w) // 2, 60), target_str, font=game.font_lg, fill=(255, 165, 0))

    # 4. 蛇 (更圆润的身体绘制)
    for idx, snake in enumerate(game.snakes):
        if not snake: continue
        # 从尾部向前绘制，实现覆盖效果
        for i in range(len(snake) - 1, 0, -1):
            curr = snake[i]
            # 尾部收束处理
            if i >= len(snake) - 3:
                scale = (len(snake) - i) / 3.0
                r = max(4, int(game.body_radius * scale))
            else:
                r = game.body_radius

            # 绘制椭圆/圆形身体
            x, y = int(curr[0]), int(curr[1])
            draw.ellipse((x - r, y - r, x + r, y + r), fill=COL_SNAKE_BODY, outline=COL_SNAKE_BORDER)

        # 头部
        hx, hy = snake[0]
        if len(snake) > 1:
            nx, ny = snake[1]
            direction = (hx - nx, hy - ny)
        else:
            direction = (0, -1)

        head_color = (255, 182, 193) if idx == 0 else (255, 255, 224)
        draw_cartoon_head(draw, (int(hx), int(hy)), game.head_radius, head_color, direction)

        if game.mode == "dual":
            head_pt = (int(hx), int(hy))
            outline_col = (100, 255, 100) if game._slot_active(idx, time.time()) else (200, 200, 200)
            draw.ellipse(
                (head_pt[0] - BIND_DIST, head_pt[1] - BIND_DIST, head_pt[0] + BIND_DIST, head_pt[1] + BIND_DIST),
                outline=outline_col, width=2)

    # 5. 食物 (带闪烁逻辑)
    curr_t = time.time()
    for f in game.foods:
        should_draw = True
        if game.mode == "single" and game.difficulty == "hard" and not game.is_paused_now:
            age = curr_t - f.spawn_time
            time_left = HARD_MODE_FOOD_TIME - age
            if time_left <= 2.0:
                if int(curr_t * 15) % 2 == 0:
                    should_draw = False

        if should_draw:
            draw_food_icon(draw, (int(f.pos[0]), int(f.pos[1])), f.type, food_radius=game.food_radius)

    # 6. 粒子
    for p in game.particles:
        draw.rectangle((p.x - 3, p.y - 3, p.x + 3, p.y + 3), fill=p.color + (int(255 * p.life),))

    # 6.5 In-game quit countdown (FIST hold, both modes)
    if hasattr(game, 'quit_trigger_start') and game.quit_trigger_start > 0.0 and not any(game.game_overs):
        rem = max(0.0, 3.0 - (time.time() - game.quit_trigger_start))
        col = (255, 0, 0)  # red border for quitting
        draw.rectangle((0, 0, game.width, game.height), outline=col + (255,), width=20)
        text = f"QUITTING... {rem:.1f}s"
        tb = draw.textbbox((0, 0), text, font=game.font_title)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.text(((game.width - tw)//2, (game.height - th)//2), text, font=game.font_title, fill=col)

    # 7. Game Over overlay (persist until gesture/keyboard action)
    if any(game.game_overs):
        # Optional red flash for a few frames
        if game.hit_wall_flash > 0:
            draw.rectangle((0, 0, game.width, game.height), outline=(255, 0, 0, 100), width=20)
            game.hit_wall_flash -= 1

        box_w, box_h = 400, 240
        x1, y1 = cx - box_w//2, cy - box_h//2
        x2, y2 = cx + box_w//2, cy + box_h//2
        
        draw.rounded_rectangle((x1, y1, x2, y2), radius=20, fill=(255, 255, 255, 230),
                                outline=(100, 100, 100), width=2)
        
        def draw_centered(text, y, font, color):
            tb = draw.textbbox((0, 0), text, font=font)
            w = tb[2] - tb[0]
            draw.text((cx - w // 2, y), text, font=font, fill=color)

        draw_centered("GAME OVER", cy - 80, game.font_lg, (255, 100, 100))
        draw_centered(game.result_text, cy - 10, game.font_md, (50, 50, 50))
        
        restart_msg = "Index Up: Restart" if game.mode == "single" else "Victory: Restart"
        draw_centered(f"{restart_msg}   Fist: Quit", cy + 40, game.font_sm, (100, 100, 100))
        draw_centered("(or press R / Q)", cy + 70, game.font_sm, (180, 180, 180))

    elif game.sub_text:
        cx, cy = game.width // 2, game.height // 2
        draw.text((cx - 100, cy), game.sub_text, font=game.font_lg, fill=(255, 215, 0), stroke_width=2,
                    stroke_fill="black")
        if time.time() - game.level_start_time > 2: game.sub_text = ""

    # 暂停提示
    if (game.auto_paused or game.manual_paused) and not any(game.game_overs):
        cx, cy = game.width // 2, game.height // 2
        draw.text((cx - 100, cy + 80), "PAUSED", font=game.font_lg, fill=(100, 100, 255))

    # Game Over Trigger Overlay
    if hasattr(game, 'go_trigger_start') and game.go_trigger_start > 0.0 and game.go_trigger_action:
        rem = max(0.0, 3.0 - (time.time() - game.go_trigger_start))
        
        if game.go_trigger_action == 'quit':
            col = (255, 0, 0) # Red
            text = f"QUITTING... {rem:.1f}s"
        else:
            col = (0, 255, 0) # Green
            text = f"RESTARTING... {rem:.1f}s"
            
        draw.rectangle((0, 0, game.width, game.height), outline=col, width=20)
        tb = draw.textbbox((0, 0), text, font=game.font_title)
        tw, th = tb[2]-tb[0], tb[3]-tb[1]
        # Draw below the "GAME OVER" box if possible, or just overlay on top
        # Overlay on top is clearer
        draw.text(((game.width - tw)//2, (game.height - th)//2 + 150), text, font=game.font_title, fill=col)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGR)

# Alias for compatibility if needed, though game.py calls display_game_overlay now
display_game_overlay = draw_game_overlay

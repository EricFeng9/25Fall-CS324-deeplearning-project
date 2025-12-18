# 手势控制贪吃蛇游戏 (Gesture Controlled Snake Game)

一个基于 OpenCV 和 MediaPipe 的创新手势交互贪吃蛇游戏。支持单人闯关、双人对战，完全通过手势控制菜单和游戏操作。

## 功能特点 (Features)

- **实时手势追踪**：使用 MediaPipe 进行高精度手部关键点检测。
- **多种游戏模式**：
  - **单人模式 (Single Player)**：包含简单和困难两种难度。
  - **双人模式 (Dual Player)**：本地双人红蓝对抗。
- **丰富的手势交互**：菜单导航、难度选择、游戏退出全手势操作。
- **现代化 UI**：马卡龙色系界面，平滑的视觉效果，3秒倒计时手势确认机制。
- **粒子特效**：吃到食物时的视觉反馈。

## 安装依赖 (Installation)

```bash
pip install -r requirements.txt
```

需要的核心库：`opencv-python`, `mediapipe`, `numpy`, `Pillow`

## 运行游戏 (Run)

```bash
python main.py
```

## 游戏规则 (Game Rules)

### 1. 单人模式 (Single Player)
- **目标**：在规定时间内达到目标分数以进入下一关（共10关）。
- **简单模式 (Easy Mode)**：
  - 食物永久存在。
  - 撞墙不死（会滑墙），撞自己也安全。
- **困难模式 (Hard Mode)**：
  - 食物只会存在 **3秒**，如果不吃掉会消失并刷新！
  - 挑战手速和反应。
- **食物类型**：
  - 🟢 **昆虫 (Insect)**: +1 分
  - 🔵 **鱼 (Fish)**: +2 分
  - 🟡 **老鼠 (Mouse)**: +3 分
  - 🟣 **毒药 (Poison)**: 吃到直接 **死亡 (Game Over)**

### 2. 双人模式 (Dual Player)
- **目标**：红蓝贪吃蛇对抗。
- **胜利条件**：
  - 率先达到目标长度（默认30）。
  - 或者时间结束时分数高者胜。
  - 或者对手死亡（如撞墙、长度过短）。
- **规则**：
  - **互撞头**：平局。
  - **撞身**：如果你撞到了对手的身体，你的身体会被截断，对手会吃掉你断掉的部分增加长度！
  - **撞墙**：直接判负。
  - **长度惩罚**：如果长度小于3，直接被淘汰。

## 操作说明 (Controls)

### 菜单手势 (Menu Gestures)
游戏全程无需键盘，看着摄像头做动作即可：

| 动作 | 手势 | 作用 |
| :--- | :--- | :--- |
| **👆 食指向上 (Index Up)** | <img src="https://emojigraph.org/media/apple/backhand-index-pointing-up_1f446.png" width="30"/> | 预选 **单人模式** (进入难度选择) |
| **✌️ 剪刀手 (Victory)** | <img src="https://emojigraph.org/media/apple/victory-hand_270c-fe0f.png" width="30"/> | 选择 **双人模式** |
| **（找不到emoji） 小指向上 (Pinky Up)** | | 选择 **简单模式** (Easy) |
| **👍 竖大拇指 (Thumb Up)** | <img src="https://emojigraph.org/media/apple/thumbs-up_1f44d.png" width="30"/> | 选择 **困难模式** (Hard) |
| **👊 握拳 (Fist)** | <img src="https://emojigraph.org/media/apple/raised-fist_270a.png" width="30"/> | **退出 / 返回** (Back/Quit) |

*注意：所有菜单选择在这个手势保持 3 秒后才会确认（会有绿色进度条提示）。*

### 游戏内操作 (In-Game)
- **移动**：移动你的食指指尖，贪吃蛇会跟随你的指尖移动。
- **暂停**：按 `Space` 空格键（手动暂停），或当手移出屏幕时自动暂停。
- **退出**：游戏中对着摄像头 **握拳 (Fist)** 并保持 3 秒，即可直接退出游戏（显示红色警告框）。

## 技术实现
- **架构**：拆分为 `game.py` (逻辑), `ui.py` (界面), `hand_detector.py` (视觉), `config.py` (配置)。
- **渲染**：结合 OpenCV 的视频流处理与 Pillow 的高质量中文文字/图形绘制。

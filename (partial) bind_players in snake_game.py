    def bind_players(self) -> bool:
        """双人模式：在游戏开始前绑定玩家1/2 的手，并倒计时 3 秒。

        流程：
            1. 提示玩家1伸手进入画面，检测到第一只手后绑定为玩家1（track_id 记录）。
            2. 提示玩家2伸手进入画面，在玩家1的手仍在画面内的前提下，绑定第二只手为玩家2。
               若此时玩家1的手丢失，则提示按 R 重新开始绑定流程。
            3. 两只手都绑定后，进行 3 秒倒计时，然后开始游戏。

        返回：
            True  绑定成功并完成倒计时；
            False 绑定过程中按 Q/Esc 退出。
        """
        stage = 1  # 1: 绑定玩家1；2: 绑定玩家2；3: 倒计时
        countdown_started = False
        countdown_start = 0.0

        self.player_track_ids = [None, None]

        while True:
            ret, frame = self.cap.read()
            if not ret:
                return False
            frame = cv2.flip(frame, 1)

            hands_info = self.detect_hands(frame)

            # 绘制基础信息
            h, w, _ = frame.shape
            cv2.putText(frame, "Binding players (Dual mode)", (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if stage == 1:
                # 绑定玩家1：检测到至少一只手，取第一只的 id
                cv2.putText(frame, "Player 1: put your hand in front of camera",
                            (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press Q/Esc to quit", (40, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                if hands_info:
                    first = hands_info[0]
                    self.player_track_ids[0] = first["id"]
                    stage = 2
            elif stage == 2:
                # 绑定玩家2：确保玩家1仍在，且有第二只不同 id 的手
                p1_id = self.player_track_ids[0]
                ids_now = [h["id"] for h in hands_info]
                p1_present = p1_id is not None and (p1_id in ids_now)

                if not p1_present:
                    cv2.putText(frame, "Player 1 lost! Press R to restart binding",
                                (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Player 2: put your hand in front of camera",
                                (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, "Keep Player 1 hand visible", (40, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    # 找到与玩家1不同 id 的手
                    for hinfo in hands_info:
                        if hinfo["id"] != p1_id:
                            self.player_track_ids[1] = hinfo["id"]
                            stage = 3
                            countdown_started = False
                            break

            elif stage == 3:
                # 两个玩家都绑定好了：3 秒倒计时
                if not countdown_started:
                    countdown_started = True
                    countdown_start = time.time()

                remaining = 3 - (time.time() - countdown_start)
                if remaining <= 0:
                    self.binding_complete = True
                    return True

                cv2.putText(frame, "Both players bound! Get ready...",
                            (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Starting in {int(remaining) + 1}...",
                            (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            cv2.imshow("Snake Game - Binding", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return False
            if key in (ord("r"), ord("R")):
                # 重新开始绑定流程
                stage = 1
                self.player_track_ids = [None, None]
                countdown_started = False





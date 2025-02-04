
import cv2
import mediapipe as mp

class ShelfInteractionDetector:
    def __init__(self, shelf_region):
        self.shelf_region = shelf_region
        self.previous_right_hand_in_shelf = False
        self.previous_left_hand_in_shelf = False
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.right_cnt = 0
        self.left_cnt = 0

    def is_hand_in_shelf(self, hand_landmark, frame_shape):
        # 棚の領域
        shelf_x1, shelf_y1, shelf_x2, shelf_y2 = self.shelf_region

        # 手首の位置 (手首のランドマークは0番)
        wrist = hand_landmark[self.mp_hands.HandLandmark.WRIST]
        wrist_x, wrist_y = int(wrist.x * frame_shape[1]), int(wrist.y * frame_shape[0])

        # 手が棚の領域に入っているかどうかを判定
        return shelf_x1 <= wrist_x <= shelf_x2 and shelf_y1 <= wrist_y <= shelf_y2

    def detect_action(self, frame, hand_landmarks, handedness):
        current_right_hand_in_shelf = False
        current_left_hand_in_shelf = False

        if hand_landmarks:
            for i, hand_landmark in enumerate(hand_landmarks):
                hand_label = handedness[i].classification[0].label
                if hand_label == 'Left':
                    current_left_hand_in_shelf = self.is_hand_in_shelf(hand_landmark, frame.shape)
                    if not self.previous_left_hand_in_shelf and current_left_hand_in_shelf:
                        print("left hand inserted")
                        cv2.imwrite("./image/left/inserted_"+str(self.left_cnt)+".png", frame)
                        self.left_cnt += 1
                    elif self.previous_left_hand_in_shelf and not current_left_hand_in_shelf:
                        print("left hand removed")
                        cv2.imwrite("./image/left/removed_"+str(self.left_cnt)+".png", frame)
                        self.left_cnt += 1
                    self.previous_left_hand_in_shelf = current_left_hand_in_shelf  # 状態を更新
                elif hand_label == 'Right':
                    current_right_hand_in_shelf = self.is_hand_in_shelf(hand_landmark, frame.shape)
                    if not self.previous_right_hand_in_shelf and current_right_hand_in_shelf:
                        print("right hand inserted")
                        cv2.imwrite("./image/right/inserted"+str(self.right_cnt)+".png", frame)
                        self.right_cnt += 1
                    elif not self.previous_left_hand_in_shelf and current_left_hand_in_shelf:
                        print("right hand removed")
                        cv2.imwrite("./image/right/removed_"+str(self.right_cnt)+".png", frame)
                        self.right_cnt += 1
                    self.previous_right_hand_in_shelf = current_right_hand_in_shelf  # 状態を更新


    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            hand_landmarks = [hand_landmark.landmark for hand_landmark in results.multi_hand_landmarks] if results.multi_hand_landmarks else None
            handedness = results.multi_handedness if results.multi_handedness else None

            self.detect_action(frame, hand_landmarks, handedness)

            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmark, self.mp_hands.HAND_CONNECTIONS)

            frame = cv2.rectangle(frame, self.shelf_region, color=(255,0,0), lineType=cv2.LINE_AA)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# 棚の領域を定義（手動で座標を指定）
shelf_region = (0, 0, 640, 400)  # (x1, y1, x2, y2)

# クラスのインスタンスを作成し、動画を処理
detector = ShelfInteractionDetector(shelf_region)
detector.process_video('movie/test.mp4')

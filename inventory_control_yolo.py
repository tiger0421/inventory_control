import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# MediaPipe Handsのセットアップ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# YOLOv5のセットアップ
model = YOLO('yolov9c.pt')

def detect_hands(image):
    # 画像を処理して手を検出
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    hand_bboxes = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # バウンディングボックスを計算
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            xmin = min(x_coords)
            xmax = max(x_coords)
            ymin = min(y_coords)
            ymax = max(y_coords)
            h, w, _ = image.shape
            hand_bboxes.append((xmin * w, ymin * h, xmax * w, ymax * h))
    return hand_bboxes

def is_inside(hand_bbox, object_bbox):
    # 手のバウンディングボックスと物体のバウンディングボックスが重なるかチェック
    hx1, hy1, hx2, hy2 = hand_bbox
    ox1, oy1, ox2, oy2 = [int(i) for i in object_bbox.xyxy[0]]

    # 重なりの判定
    return not (hx2 < ox1 or hx1 > ox2 or hy2 < oy1 or hy1 > oy2)

# 動画の読み込み
cap = cv2.VideoCapture('movie/test.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 手の検出
    hand_bboxes = detect_hands(frame)
    if(len(hand_bboxes) == 0):
        continue

    # 物体の検出
    results = model(frame, device='mps')
    if(len(results) == 0):
        continue

    object_names   = results[0].names
    object_classes = results[0].boxes.cls
    object_bboxes  = results[0].boxes
    annotatedFrame = results[0].plot()

    # 手に持っている物体の特定
    held_objects_index = []
    for hand_bbox in hand_bboxes:
        for i, object_bbox in enumerate(object_bboxes):
            if(i == 0):
                # Skip if "Person"
                continue
            elif is_inside(hand_bbox, object_bbox):
                held_objects_index.append(i)

    # 結果の表示
    for index in held_objects_index:
        x1, y1, x2, y2 = [int(i) for i in object_bboxes[index].xyxy[0]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{object_names[int(index)]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Video with detected objects', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


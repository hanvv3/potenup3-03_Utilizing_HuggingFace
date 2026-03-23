## uv remove opencv-python
## uv add opencv-contrib-python

import sys
import cv2
import mediapipe as mp
import math

# ---------------------------
# MediaPipe 초기화
# ---------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------
# 랜드마크 인덱스
# ---------------------------
V_INDICES = [5, 6, 7, 8, 9, 10, 11, 12]
TIP_INDICES = [4, 8, 12, 16, 20]

# 모드: None / "V" / "TIPS" / "DIST"
mode = None


def landmark_to_pixel(landmark, frame_shape):
    """정규화 좌표 -> 픽셀 좌표"""
    h, w, _ = frame_shape
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    return x, y


def draw_line_between(frame, hand_landmarks, tup:tuple):
    st, end = tup
    thumb_tip = hand_landmarks.landmark[st]
    index_tip = hand_landmarks.landmark[end]

    x1, y1 = landmark_to_pixel(thumb_tip, frame.shape)
    x2, y2 = landmark_to_pixel(index_tip, frame.shape)

    # 선
    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


def draw_V(frame, hand_landmarks):
    """V 모드: 5~12번 랜드마크에 원 그리기"""
    for idx in V_INDICES:
        landmark = hand_landmarks.landmark[idx]
        point_x, point_y = landmark_to_pixel(landmark, frame.shape)
        cv2.circle(frame, (point_x, point_y), 6, (0, 255, 0), 2)
        
    for idx in range(len(V_INDICES)-1):
        if idx == 3:
            draw_line_between(frame, hand_landmarks, (5, 9))
        else:
            draw_line_between(frame, hand_landmarks, (V_INDICES[idx], V_INDICES[idx+1]))


def draw_tips(frame, hand_landmarks):
    """Tips 모드: 손가락 끝 5개에 원 그리기"""
    for idx in TIP_INDICES:
        landmark = hand_landmarks.landmark[idx]
        point_x, point_y = landmark_to_pixel(landmark, frame.shape)
        cv2.circle(frame, (point_x, point_y), 6, (255, 0, 0), 2)


def get_dist(frame, hand_landmarks):
    """
    Dist 모드:
    엄지 끝(4)과 검지 끝(8)에 원 + 선 + 거리 표시
    """
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    x1, y1 = landmark_to_pixel(thumb_tip, frame.shape)
    x2, y2 = landmark_to_pixel(index_tip, frame.shape)

    # 점
    cv2.circle(frame, (x1, y1), 8, (0, 255, 0), 2)
    cv2.circle(frame, (x2, y2), 8, (0, 255, 0), 2)

    # 선
    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 거리 계산
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # 선 중간쯤에 텍스트 표시
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2

    cv2.putText(
        frame,
        f"{dist:.2f}",
        (mid_x + 10, mid_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


# ---------------------------
# 카메라 시작
# ---------------------------
vcap = cv2.VideoCapture(0)

if not vcap.isOpened():
    print("카메라를 열 수 없습니다.")
    sys.exit()

while True:
    ret, frame = vcap.read()

    if not ret:
        print("카메라가 작동하지 않습니다.")
        break

    # 좌우 반전
    flipped_frame = cv2.flip(frame, 1)

    # MediaPipe는 RGB 입력 권장
    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    # 현재 모드 표시
    mode_text = f"MODE: {mode if mode is not None else 'NONE'}"
    cv2.putText(
        flipped_frame,
        mode_text,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # 손 감지 결과가 있으면 현재 mode에 따라 계속 그림
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # 필요하면 기본 랜드마크도 같이 그릴 수 있음
            # mp_drawing.draw_landmarks(
            #     flipped_frame,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style()
            # )

            if mode == "V":
                draw_V(flipped_frame, hand_landmarks)

            elif mode == "TIPS":
                draw_tips(flipped_frame, hand_landmarks)

            elif mode == "DIST":
                get_dist(flipped_frame, hand_landmarks)

    cv2.imshow("webcam", flipped_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:          # ESC
        break
    elif key == ord('q'):
        mode = "V"
        print("V 모드")
    elif key == ord('w'):
        mode = "TIPS"
        print("TIPS 모드")
    elif key == ord('e'):
        mode = "DIST"
        print("DIST 모드")
    elif key == ord('r'):
        mode = None
        print("모드 해제")

vcap.release()
cv2.destroyAllWindows()
hands.close()
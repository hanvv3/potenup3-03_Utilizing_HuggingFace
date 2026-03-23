import sys
import cv2
import mediapipe as mp
import math


LEFT_EYE_UP = 386
LEFT_EYE_DOWN = 374
RIGHT_EYE_UP = 159
RIGHT_EYE_DOWN = 145

THRESHOLD = 3.0

# ---------------------------
# MediaPipe 초기화
# ---------------------------
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

# 모드: None / "FaceDet" / "FaceMesh"
mode = None


def landmark_to_pixel(landmark, frame_shape):
    """정규화 좌표 -> 픽셀 좌표"""
    h, w, _ = frame_shape
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    return x, y


def draw_face_det(draw_frame, rgb_frame, detector):
    results = detector.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(draw_frame, detection)


def draw_face_mesh(draw_frame, rgb_frame, mesh_model):
    results = mesh_model.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=draw_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=1,
                    circle_radius=1
                )
            )

            # 윤곽선까지 보고 싶으면 아래도 추가 가능
            mp_drawing.draw_landmarks(
                image=draw_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 0, 255),
                    thickness=1
                )
            )


def sleepy_eye_detector(draw_frame, rgb_frame, mesh_model):
    results = mesh_model.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            r_eye_up = face_landmarks.landmark[RIGHT_EYE_UP]
            r_eye_down = face_landmarks.landmark[RIGHT_EYE_DOWN]
            l_eye_up = face_landmarks.landmark[LEFT_EYE_UP]
            l_eye_down = face_landmarks.landmark[LEFT_EYE_DOWN]
            
            x1, y1 = landmark_to_pixel(r_eye_up, draw_frame.shape)
            x2, y2 = landmark_to_pixel(r_eye_down, draw_frame.shape)
            dist1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            # 선 중간쯤에 텍스트 표시
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            cv2.putText(
                draw_frame,
                f"{dist1:.2f}",
                (mid_x - 50, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            x1, y1 = landmark_to_pixel(l_eye_up, draw_frame.shape)
            x2, y2 = landmark_to_pixel(l_eye_down, draw_frame.shape)
            dist2 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            cv2.putText(
                draw_frame,
                f"{dist2:.2f}",
                (mid_x - 20, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            if (dist1 < THRESHOLD) and (dist2 < THRESHOLD):
                cv2.putText(
                    draw_frame, f"Don't sleep!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
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

    # MediaPipe 입력용 RGB
    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False

    # 현재 모드에 따라 처리
    if mode == "FaceDet":
        draw_face_det(flipped_frame, rgb_frame, face_detection)

    elif mode == "FaceMesh":
        draw_face_mesh(flipped_frame, rgb_frame, face_mesh)

    elif mode == "SleepyEye":
        sleepy_eye_detector(flipped_frame, rgb_frame, face_mesh)

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

    cv2.imshow("webcam", flipped_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:   # ESC
        break
    elif key == ord('q'):
        mode = "FaceDet"
        print("Face Detection 모드")
    elif key == ord('w'):
        mode = "FaceMesh"
        print("Face Mesh 모드")
    elif key == ord('e'):
        mode = "SleepyEye"
        print("Sleepy Eye 모드")
    elif key == ord('r'):
        mode = None
        print("모드 해제")

vcap.release()
cv2.destroyAllWindows()
face_detection.close()
face_mesh.close()
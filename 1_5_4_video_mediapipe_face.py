import sys
import cv2
import mediapipe as mp

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
    elif key == ord('r'):
        mode = None
        print("모드 해제")

vcap.release()
cv2.destroyAllWindows()
face_detection.close()
face_mesh.close()
## 만약에 웹캠이 켜지지 않는다. 
## uv add opencv-contrib-python 

## mediapipe 라이브러리 설치 uv add mediapipe==0.10.14
import sys 
import cv2 
import mediapipe as mp 
import os
import csv
import joblib 
import numpy as np

CLASS_NAMES = {
    0: "Scissors",
    1: "Rock",
    2: "Paper"
}

# 추론모델 불러오기
model = joblib.load("./scissors_rock_paper.pkl")
model

# 모델 불러오기
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

## STEP 0. 데이터를 저장할 csv 만들기
file_path = "./hand_data.csv"
if os.path.exists(file_path):
    with open(file_path, "w") as file:
        writer = csv.writer(file)
##########################################################
# 카메라 시작
##########################################################
vcap = cv2.VideoCapture(0)

while True:
    # 카메라 이미지 추출하기 
    ret, frame = vcap.read()

    # 카메라 작동 확인
    if not ret:
        print("카메라가 작동하지 않습니다.")
        sys.exit()

    # 좌우 반전 
    flipped_frame = cv2.flip(frame, 1)  # 그릴 도화지 입니다.

    #------------------------------------------------------
    # 손 그리기 준비 
    flipped_frame.flags.writeable = True 

    # 손 감지하기 
    results = hands.process(flipped_frame)

    h, w, _ = flipped_frame.shape
    
    # 추출 및 그리기
    if results.multi_hand_landmarks:
        #print(f"{len(results.multi_hand_landmarks)}개의 손이 감지되었습니다.")
        ## 손 하나하나 가져오기 
        for hand_landmarks in results.multi_hand_landmarks:
            #print(f"이 손에는 {len(hand_landmarks.landmark)}개의 좌표가 있습니다.")

            ## STEP1. 21개의 데이터 리스트 수집하기 ([x1, y1, z1, x2, y2, z2, .....])
            collect_row_data = []
            landmarks = hand_landmarks.landmark
            for landmark in landmarks:
                collect_row_data.extend([landmark.x, landmark.y, landmark.z])

                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(flipped_frame, (x, y), 5, (0,0,255), 2)

            pred = model.predict(np.array([collect_row_data]))

            cv2.putText(
                flipped_frame,
                f"{CLASS_NAMES[pred.item()]}",
                (x - 50, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    # 화면 띄우기
    cv2.imshow("webcam", flipped_frame)

    # 화면 끄기 
    key = cv2.waitKey(1)
    if key == 27: # ESC(ASCII Code)
        break

vcap.release()
cv2.destroyAllWindows()
hands.close()
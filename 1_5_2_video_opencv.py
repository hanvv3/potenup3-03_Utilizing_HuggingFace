## 만약에 웹캠이 켜지지 않는다. 
## uv add opencv-contrib-python 
import sys 
import cv2 

vcap = cv2.VideoCapture(0)

while True:
    # 카메라 이미지 추출하기 
    ret, frame = vcap.read()

    # 카메라 작동 확인
    if not ret:
        print("카메라가 작동하지 않습니다.")
        sys.exit()

    # 좌우 반전 
    flipped_frame = cv2.flip(frame, 1)

    # 화면 띄우기
    cv2.imshow("webcam", flipped_frame)

    # 화면 끄기 
    key = cv2.waitKey(1)
    if key == 27: # ESC(ASCII Code)
        break

vcap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

def resize_video_with_white_background(input_video_path, output_video_path):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(input_video_path)

    # 출력 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (160, 160))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        # 비율 계산
        if h > w:
            new_h, new_w = 160, int(w * (160 / h))
        else:
            new_h, new_w = int(h * (160 / w)), 160

        # 프레임 리사이즈
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 새로운 흰색 배경 생성
        new_background = np.ones((160, 160, 3), dtype=np.uint8) * 255

        # 리사이즈된 프레임을 가운데에 배치
        y_offset = (160 - new_h) // 2
        x_offset = (160 - new_w) // 2
        new_background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

        # 리사이즈된 프레임을 출력 비디오에 씀
        out.write(new_background)

    # 모든 작업 후 객체 해제
    cap.release()
    out.release()

# 사용 예시
resize_video_with_white_background('images/mcdropout.mp4', 'images/mcdropout_reshaped.mp4')

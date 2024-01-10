import cv2
import numpy as np

def resize_video_with_background(input_video_path, output_video_path, output_size=(160, 160), duration=None):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(input_video_path)

    # 입력 비디오의 FPS 및 총 프레임 수 얻기
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 출력 비디오의 FPS 설정 (사용자 지정 길이가 있는 경우, 총 프레임 수를 조정)
    if duration:
        output_fps = total_frames / duration
    else:
        output_fps = fps

    # 출력 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, output_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        # 비율 계산
        if h > w:
            new_h, new_w = output_size[1], int(w * (output_size[1] / h))
        else:
            new_h, new_w = int(h * (output_size[0] / w)), output_size[0]

        # 프레임 리사이즈
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 새로운 배경 생성
        new_background = np.ones(output_size + (3,), dtype=np.uint8) * 255

        # 리사이즈된 프레임을 가운데에 배치
        y_offset = (output_size[1] - new_h) // 2
        x_offset = (output_size[0] - new_w) // 2
        new_background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

        # 리사이즈된 프레임을 출력 비디오에 씀
        out.write(new_background)

    # 모든 작업 후 객체 해제
    cap.release()
    out.release()
    
# 사용 예시 (지정된 길이 있는 경우, 예: 10초)
resize_video_with_background('images/mcdropout.mp4', 'images/mcdropout_reshaped.mp4', duration=5)
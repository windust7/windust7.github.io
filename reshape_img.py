import cv2
import numpy as np

def resize_with_aspect_ratio(image_path, output_path):
    # 이미지를 불러옴
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    h, w = image.shape[:2]

    # 비율 계산
    if h > w:
        new_h, new_w = 160, int(w * (160 / h))
    else:
        new_h, new_w = int(h * (160 / w)), 160

    # 이미지 리사이즈
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 리사이즈된 이미지에 알파 채널 추가 (투명도)
    if len(resized_image.shape) == 2: # 흑백 이미지인 경우
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGRA)
    elif len(resized_image.shape) == 3: # 컬러 이미지인 경우
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2BGRA)

    # 새로운 투명한 배경 생성
    new_background = np.zeros((160, 160, 4), dtype=np.uint8)

    # 리사이즈된 이미지를 가운데에 배치
    y_offset = (160 - new_h) // 2
    x_offset = (160 - new_w) // 2
    new_background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image

    # 이미지 저장
    cv2.imwrite(output_path, new_background)

# 사용 예시
resize_with_aspect_ratio('images/mcdropout.jpeg', 'images/mcdropout_reshaped.png')

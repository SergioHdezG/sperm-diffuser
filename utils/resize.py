import os
import cv2

img_path = 'data/real_test/images'
save_path = 'data/real_test_500x500/images'
os.makedirs(save_path, exist_ok=True)
list_path = os.listdir(img_path)

for path in list_path:
    img = cv2.imread(os.path.join(img_path, path))
    resize = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(save_path, path), resize)



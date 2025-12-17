import cv2
import numpy as np

# Загружаем шаблоны
img_ghost_m = [
    cv2.imread('ghosts/candy_ghost.png'),
    cv2.imread('ghosts/scary_ghost.png'),
    cv2.imread('ghosts/pampkin_ghost.png')
]

# создаём перевёрнутые варианты
img_ghost_m2 = []
for ghost in img_ghost_m:
    img_ghost_m2.append(ghost)
    img_ghost_m2.append(cv2.flip(ghost, 0))
    img_ghost_m2.append(cv2.flip(ghost, 1))
    img_ghost_m2.append(cv2.flip(ghost, -1))

# Загружаем сцену
img_result = cv2.imread('ghosts/lab7.png')
img_result_c = img_result.copy()
gray_scene_full = cv2.cvtColor(img_result_c, cv2.COLOR_BGR2GRAY)

# Зоны поиска: (y1, y2, x1, x2, x_offset, y_offset)
zones = [
    (0, 520, 1280, 1920, 1280, 0),  # правая верхняя
    (0, 520, 0, 640, 0, 0),         # левая верхняя
    (0, gray_scene_full.shape[0], 0, gray_scene_full.shape[1], 0, 0)  # вся сцена
]

# Основной цикл: по зонам и по призракам
for y1, y2, x1, x2, x_offset, y_offset in zones:
    gray_scene = gray_scene_full[y1:y2, x1:x2]

    for idx, img_ghost in enumerate(img_ghost_m2):
        gray_temp = cv2.cvtColor(img_ghost, cv2.COLOR_BGR2GRAY)
        detector = cv2.SIFT_create()
        kp_temp, des_temp = detector.detectAndCompute(gray_temp, None)
        kp_scene, des_scene = detector.detectAndCompute(gray_scene, None)

        # FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_temp, des_scene, k=2)

        # фильтруем хорошие совпадения
        good_matches = []
        for m, n in matches:
            if m.distance < 0.55 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:
            src_pts = np.float32([kp_temp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h, w = img_ghost.shape[:2]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)

            # добавляем смещение зоны
            dst_shifted = dst + np.array([[[x_offset, y_offset]]], dtype=np.float32)
            img_result = cv2.polylines(img_result, [np.int32(dst_shifted)], True, (0, 255, 0), 3)

cv2.imshow("Found Ghost", img_result)
cv2.imwrite('result_with_box.jpg', img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
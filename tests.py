import cv2
import numpy as np

cap = cv2.VideoCapture(0)

filters = ['Laplacian', 'Color Edges', 'Pencil Sketch', 'Canny']
current_filter = 0

print("Камера запущена. Управление:")
print("'q' - выход, 'f' - сменить фильтр, 's' - сохранить кадр")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if filters[current_filter] == 'Laplacian':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        result = np.uint8(np.absolute(laplacian))

    elif filters[current_filter] == 'Color Edges':
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        laplacian_b = cv2.Laplacian(blurred[:, :, 0], cv2.CV_64F)
        laplacian_g = cv2.Laplacian(blurred[:, :, 1], cv2.CV_64F)
        laplacian_r = cv2.Laplacian(blurred[:, :, 2], cv2.CV_64F)
        result = cv2.merge([
            np.uint8(np.absolute(laplacian_b)),
            np.uint8(np.absolute(laplacian_g)),
            np.uint8(np.absolute(laplacian_r))
        ])
        result = cv2.convertScaleAbs(result, alpha=3.0, beta=0)

    elif filters[current_filter] == 'Pencil Sketch':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        result = cv2.divide(gray, 255 - blurred, scale=256)

    elif filters[current_filter] == 'Canny':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(gray, 50, 150)

    # Добавляем текст с названием фильтра
    cv2.putText(result, f'Filter: {filters[current_filter]}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, "Press 'f' to change filter, 'q' to quit", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Interactive Filters', result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        current_filter = (current_filter + 1) % len(filters)
        print(f"Фильтр изменен на: {filters[current_filter]}")
    elif key == ord('s'):
        filename = f'{filters[current_filter]}_capture.jpg'
        cv2.imwrite(filename, result)
        print(f"Сохранено: {filename}")

cap.release()
cv2.destroyAllWindows()
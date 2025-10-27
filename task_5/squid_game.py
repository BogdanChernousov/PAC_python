import cv2
import numpy as np
import time

def motion_detection_with_timer():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Motion Mask', cv2.WINDOW_NORMAL)
    prev_gray = None
    motion_enabled = True
    timer_start = time.time()
    timer_duration = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Проверяем таймер и переключаем режим
        if time.time() - timer_start > timer_duration:
            motion_enabled = not motion_enabled
            timer_start = time.time()

        # Отображаем текущий режим (Red light / Green light)
        status_text = "Red light" if motion_enabled else "Green light"
        color = (0, 0, 255) if motion_enabled else (0, 255, 0)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Обрабатываем кадр
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
        if prev_gray is not None:
            # Рассчитываем оптический поток
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mask = magnitude > 5.0

            motion_mask = motion_mask.astype(np.uint8) * 255
            motion_area = np.sum(motion_mask == 255)
            motion_detected = motion_area > 6000

            # Отображаем маску движения
            mask_display = np.zeros_like(frame)
            mask_display[motion_mask == 255] = [0, 0, 255]
            mask_display[motion_mask == 0] = [0, 255, 0]
            cv2.imshow('Motion Mask', mask_display)

            # Отображаем статус движения только при Red light
            if motion_enabled:
                status = "MOTION!" if motion_detected else "NO MOTION"
                color = (0, 0, 255) if motion_detected else (0, 255, 0)
                cv2.putText(frame, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            # Первый кадр - инициализация
            mask_display = np.zeros_like(frame)
            mask_display[:] = [0, 255, 0]
            cv2.imshow('Motion Mask', mask_display)
            if motion_enabled:
                cv2.putText(frame, "Initializing...", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        prev_gray = gray.copy()

        cv2.imshow('Camera', frame)

        # Выход по ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    motion_detection_with_timer()

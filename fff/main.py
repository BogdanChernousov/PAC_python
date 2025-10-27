import cv2
import time
import HandTrackingModule as htm

# Настройки камеры
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Инициализация детектора
pTime = 0
detector = htm.handDetector(detectionCon=0.75, maxHands=1)
totalFingers = 0

# Цвета для визуализации
COLOR_FINGERS = (0, 255, 0)  # Зеленый для поднятых пальцев
COLOR_FPS = (255, 0, 255)  # Розовый для FPS
COLOR_COUNT = (0, 255, 255)  # Желтый для счетчика

print("Запуск программы... Покажите руку в кадре. Нажмите 'q' для выхода.")

# Основной цикл
while True:
    success, img = cap.read()
    if not success:
        print("Ошибка: не удалось получить кадр с камеры")
        break

    img = cv2.flip(img, 1)  # зеркальное отражение

    # Обнаружение руки
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)

    # Подсчет поднятых пальцев
    if lmList:
        fingersUp = detector.fingersUp()
        totalFingers = fingersUp.count(1)

        # Визуализация поднятых пальцев
        for i, is_up in enumerate(fingersUp):
            if is_up:
                # Рисуем кружки на кончиках поднятых пальцев
                x, y = lmList[detector.tipIds[i]][1], lmList[detector.tipIds[i]][2]
                cv2.circle(img, (x, y), 12, COLOR_FINGERS, cv2.FILLED)
                cv2.circle(img, (x, y), 15, COLOR_FINGERS, 3)

        # Отображение bounding box вокруг руки
        if bbox:
            xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

    # Вывод FPS и количества пальцев
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Панель информации
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, COLOR_FPS, 2)
    cv2.putText(img, f'Fingers: {totalFingers}', (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, COLOR_COUNT, 2)

    # Большой счетчик пальцев в углу
    cv2.rectangle(img, (wCam - 100, 20), (wCam - 20, 100), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, str(totalFingers), (wCam - 80, 80), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

    # Отладочная информация (можно убрать)
    if lmList:
        cv2.putText(img, 'Hand Detected', (10, hCam - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    else:
        cv2.putText(img, 'No Hand', (10, hCam - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    cv2.imshow("Hand Finger Counter", img)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Программа завершена")
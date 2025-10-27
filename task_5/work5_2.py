import cv2

# Укажите путь к вашему видеофайлу
video_path = "test.mp4"

# Открываем видеофайл
cap = cv2.VideoCapture(video_path)

while True:
    # Читаем кадр
    ret, frame = cap.read()

    # Если кадр не получен (конец видео)
    if not ret:
        break

    # Конвертируем в градации серого
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Показываем кадр в градациях серого
    cv2.imshow('testVideoGrayscale', gray_frame)

    # Выход по нажатию ESC
    if cv2.waitKey(25) & 0xFF == 27:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
print("Воспроизведение завершено")
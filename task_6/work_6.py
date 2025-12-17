import os
import numpy as np
import cv2

def random_augmentation(image, mask, target_size=(256, 256)):

    app_aug = []

    # A
    # Случайный поворот
    if np.random.random() > 0.5:
        angle = np.random.randint(-180, 180)
        app_aug.append(f"Rotate: {angle}deg")
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    # B
    # Случайное отражение по горизонтали
    if np.random.random() > 0.5:
        app_aug.append("Flip H")
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    # Случайное отражение по вертикали
    if np.random.random() > 0.5:
        app_aug.append("Flip V")
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    # C
    # Случайное вырезание
    if np.random.random() > 0.5:
        crop_size = np.random.randint(int(target_size[0] * 0.8), target_size[0])
        x = np.random.randint(0, target_size[1] - crop_size)
        y = np.random.randint(0, target_size[0] - crop_size)
        app_aug.append(f"Crop: {crop_size}px")

        image_crop = image[y:y + crop_size, x:x + crop_size]
        mask_crop = mask[y:y + crop_size, x:x + crop_size]

        # Возвращаем к исходному размеру
        image = cv2.resize(image_crop, target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask_crop, target_size, interpolation=cv2.INTER_NEAREST)

    # D
    # Случайное размытие
    if np.random.random() > 0.7:
        kernel_size = np.random.choice([3, 5, 7])
        app_aug.append(f"Blur: {kernel_size}x{kernel_size}")
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Если не было применено аугментаций
    if not app_aug:
        app_aug.append("No aug")

    return image, mask, app_aug


def nails_generator(data_dir='archive', shuffle=True, target_size=(256, 256), augment=True):

    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'labels')

    # Получаем списки файлов
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    # Создаем пары по именам
    pairs = []
    for img_file in image_files:
        name = os.path.splitext(img_file)[0]
        for mask_file in mask_files:
            mask_name = os.path.splitext(mask_file)[0]
            if name == mask_name:
                pairs.append((img_file, mask_file))
                break

    indices = np.arange(len(pairs))

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            img_file, mask_file = pairs[idx]

            # Загружаем изображение
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img_array = img.copy()

            # Загружаем маску
            mask_path = os.path.join(masks_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, target_size)
            mask_array = mask

            # Применяем аугментацию
            app_aug = ["No aug"]
            if augment:
                img_array, mask_array, app_aug = random_augmentation(img_array, mask_array, target_size)

            yield img_array, mask_array, app_aug


def display_pair_with_overlay(image, mask, augmentations, pair_number):

    image_display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB в BGR

    mask_display = mask

    # Создаем наложение
    overlay = image_display.copy()
    mask_binary = (mask > 127).astype(np.uint8)

    # Добавляем зеленый цвет для маски
    overlay[mask_binary == 1] = [0, 255, 0]

    # Собираем все в один вид
    mask_bgr = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
    display = np.hstack([image_display, mask_bgr, overlay])

    # Добавляем текст с информацией об аугментациях
    aug_text = " | ".join(augmentations)

    # Добавляем черную подложку для текста
    cv2.rectangle(display, (0, 0), (display.shape[1], 70), (0, 0, 0), -1)

    # Текст сверху
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display, f"Pair {pair_number}", (10, 20), font, 0.6, (255, 255, 255), 1)
    cv2.putText(display, aug_text, (10, 45), font, 0.5, (255, 255, 255), 1)

    return display


if __name__ == "__main__":
    # Создаем генератор
    gen = nails_generator(shuffle=True, augment=True)

    cv2.namedWindow('Nails Segmentation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Nails Segmentation', 900, 300)

    pair_counter = 1

    while True:
        # Получаем следующую пару
        image, mask, aug_list = next(gen)

        # Создаем отображение
        display = display_pair_with_overlay(image, mask, aug_list, pair_counter)

        # Показываем
        cv2.imshow('Nails Segmentation', display)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC - выход из программы
                cv2.destroyAllWindows()
                exit()
            elif key == 32:  # Пробел - следующая пара
                pair_counter += 1
                break

    cv2.destroyAllWindows()
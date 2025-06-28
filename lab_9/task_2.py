from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def magic_wand(
    image_array: np.ndarray, seed: Tuple[int, int], thresh: float
) -> np.ndarray:
    """Реализация алгоритма волшебной палочки"""
    if len(image_array.shape) == 3:
        image_array = np.mean(image_array, axis=2).astype(np.uint8)

    # Вычисляем координаты изначального пикселя
    height, width = image_array.shape
    x, y = seed

    # Нормализация и создание матрицы расстояний
    c0 = image_array[y, x] / 255.0
    normalized = image_array / 255.0
    diff = np.abs(normalized - c0)

    # Бинарная матрица
    B = np.zeros((height, width), dtype=np.uint8)
    B[diff <= thresh] = 1

    # Простая заливка с помощью BFS
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    stack = [(x, y)]

    while stack:
        x, y = stack.pop()
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        if mask[y + 1, x + 1] == 1 or B[y, x] == 0:
            continue

        mask[y + 1, x + 1] = 1
        stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

    return mask[1:-1, 1:-1]


def apply_mask(image_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Накладываем маску на изображение"""
    masked_image = np.zeros_like(image_array)
    for c in range(3):
        masked_image[:, :, c] = np.where(mask == 1, image_array[:, :, c], 0)

    return masked_image


def select_point(img_array: np.ndarray) -> Optional[Tuple[int, int]]:
    """Окно для выбора точки пользователем"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img_array)
    ax.set_title(
        "Кликните по изображению чтобы выбрать точку (закройте окно для продолжения)"
    )

    point = [None]

    def on_click(event) -> None:
        if event.inaxes == ax:
            point[0] = (int(event.xdata), int(event.ydata))
            ax.plot(event.xdata, event.ydata, "ro")
            fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    return point[0]


def main() -> None:
    # Параметры
    image_path = "origins/rafal.jpg"
    threshold = 0.05  # Порог от 0 до 1!

    # Загрузка изображения
    img = Image.open(image_path)
    img_array = np.array(img)

    # Выбор точки
    seed_point = select_point(img_array)

    if seed_point:
        print(f"Выбрана точка: {seed_point}")

        # Применяем алгоритм
        mask = magic_wand(img_array, seed_point, threshold)

        # Применяем маску к изображению
        result = apply_mask(img_array, mask)

        # Визуализация
        plt.figure(figsize=(18, 6))

        # Исходное изображение
        plt.subplot(1, 2, 1)
        plt.imshow(img_array)
        plt.scatter([seed_point[0]], [seed_point[1]], c="red", s=40)
        plt.title("Исходное изображение")

        # Результат с наложенной маской
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.title("Результат работы алгоритма")

        plt.tight_layout()
        plt.show()
    else:
        print("Точка не выбрана")


if __name__ == "__main__":
    main()

import heapq
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from PIL import Image


def compute_edge_weights(image_array: np.ndarray, K: float) -> np.ndarray:
    """Вычисляем веса рёбер между пикселями"""
    if len(image_array.shape) == 3:
        image_array = np.mean(image_array, axis=2).astype(np.uint8)

    height, width = image_array.shape
    weights = np.zeros((height, width, 4))

    # Разности цветов между соседними пикселями
    diff_right = np.abs(image_array[:, 1:] - image_array[:, :-1])
    diff_down = np.abs(image_array[1:, :] - image_array[:-1, :])

    # Вычисление весов
    weights[:, :-1, 0] = 1.0 / (K + diff_right)  # Право
    weights[:, 1:, 1] = 1.0 / (K + diff_right)  # Лево
    weights[:-1, :, 2] = 1.0 / (K + diff_down)  # Низ
    weights[1:, :, 3] = 1.0 / (K + diff_down)  # Верх

    return weights


def find_shortest_path(
    image_array: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    weights: np.ndarray,
) -> List[Tuple[int, int]]:
    """Алгоритм Дейкстры для поиска кратчайшего пути между двумя точками"""
    height, width = image_array.shape[:2]

    dist = np.full((height, width), np.inf)
    prev = np.full((height, width, 2), -1)
    visited = np.zeros((height, width), dtype=bool)

    x, y = start
    dist[y, x] = 0
    heap = [(0, x, y)]

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while heap:
        current_dist, x, y = heapq.heappop(heap)

        if (x, y) == end:
            break

        if visited[y, x]:
            continue

        visited[y, x] = True

        for i, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy

            if 0 <= nx < width and 0 <= ny < height:
                edge_weight = weights[y, x, i]
                new_dist = current_dist + edge_weight

                if new_dist < dist[ny, nx]:
                    dist[ny, nx] = new_dist
                    prev[ny, nx] = (x, y)
                    heapq.heappush(heap, (new_dist, nx, ny))

    path = []
    x, y = end

    while (x, y) != (-1, -1):
        path.append((x, y))
        x, y = prev[y, x] if (x, y) != start else (-1, -1)

    return path[::-1]


def apply_mask(image_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Накладываем маску на изображение"""
    masked_image = np.zeros_like(image_array)
    for c in range(3):
        masked_image[:, :, c] = np.where(mask == 1, image_array[:, :, c], 0)

    return masked_image


def select_multiple_points(img_array: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """Интерактивный выбор нескольких точек"""

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_array)
    ax.set_title("Кликните чтобы добавить точки контура (Правая кнопка - завершить)")

    points = []

    def on_click(event):
        if event.inaxes != ax:
            return

        # Левая кнопка - добавить точку
        if event.button == 1:
            x, y = int(event.xdata), int(event.ydata)
            points.append((x, y))
            ax.plot(x, y, "ro" if len(points) == 1 else "bo")
            if len(points) > 1:
                ax.plot([points[-2][0], x], [points[-2][1], y], "g-")
            fig.canvas.draw()
            print(f"Добавлена точка {len(points)}: {(x, y)}")

        # Правая кнопка - завершить
        elif event.button == 3 and len(points) >= 3:
            # Замыкаем контур
            ax.plot([points[-1][0], points[0][0]], [points[-1][1], points[0][1]], "g-")
            fig.canvas.draw()
            plt.close()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    if len(points) >= 3:
        return points
    else:
        return None


def main() -> None:
    # Параметры
    image_path = "origins/rafal.jpg"
    K = 1.0

    # Загрузка изображения
    img = Image.open(image_path)
    img_array = np.array(img)

    # Вычисляем веса рёбер
    edge_weights = compute_edge_weights(img_array, K)

    # Выбираем точки контура
    points = select_multiple_points(img_array)

    if not points or len(points) < 3:
        print("Нужно выбрать минимум 3 точки для создания контура")
        return

    # Находим кратчайшие пути между всеми точками
    full_path = []
    for i in range(len(points)):
        start = points[i]
        end = points[(i + 1) % len(points)]
        path_segment = find_shortest_path(img_array, start, end, edge_weights)
        full_path.extend(path_segment)

    # Создаем маску на основе контура
    height, width = img_array.shape[:2]
    y, x = np.mgrid[:height, :width]
    points_array = np.array(full_path)

    path = Path(points_array)
    mask = path.contains_points(np.vstack((x.flatten(), y.flatten())).T)
    mask = mask.reshape(height, width).astype(np.uint8)

    # Накладываем маску на результат
    result = apply_mask(img_array, mask)

    # Визуализация
    plt.figure(figsize=(15, 7))

    # Исходное изображение с контуром
    plt.subplot(1, 2, 1)
    plt.imshow(img_array)
    plt.plot(
        [p[0] for p in points + [points[0]]],
        [p[1] for p in points + [points[0]]],
        "ro-",
    )
    plt.title("Выбранные точки и контур")

    # Результат сегментации
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title("Маска сегментации")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

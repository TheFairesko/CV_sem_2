from collections import deque
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def build_graph(
    image_array: np.ndarray, lambda_: float = 1.0, sigma: float = 1.0
) -> Dict[Tuple[int, int], Dict[Tuple[int, int], float]]:
    """Построение графа смежности для изображения."""
    height, width = image_array.shape[:2]
    graph = {}

    # Создаем вершины для всех пикселей
    for y in range(height):
        for x in range(width):
            graph[(x, y)] = {}

    # Добавляем рёбра между соседними пикселями
    for y in range(height):
        for x in range(width):
            current_color = image_array[y, x]

            # Право
            if x < width - 1:
                neighbor_color = image_array[y, x + 1]
                weight = lambda_ * np.exp(
                    -sigma * np.linalg.norm(current_color - neighbor_color)
                )
                graph[(x, y)][(x + 1, y)] = weight
                graph[(x + 1, y)][(x, y)] = weight

            # Низ
            if y < height - 1:
                neighbor_color = image_array[y + 1, x]
                weight = lambda_ * np.exp(
                    -sigma * np.linalg.norm(current_color - neighbor_color)
                )
                graph[(x, y)][(x, y + 1)] = weight
                graph[(x, y + 1)][(x, y)] = weight

    return graph


def calculate_tlink_weights(
    pixel: Tuple[int, int],
    image_array: np.ndarray,
    object_seeds: Set[Tuple[int, int]],
    background_seeds: Set[Tuple[int, int]],
) -> Tuple[float, float]:
    """
    Вычисление весов рёбер между пикселем и истоком/стоком
    Возвращает (вес к истоку, вес к стоку)
    """
    # Если пиксель в object seeds - бесконечный вес к истоку
    if pixel in object_seeds:
        return float("inf"), 0.0

    # Если пиксель в background seeds - бесконечный вес к стоку
    if pixel in background_seeds:
        return 0.0, float("inf")

    # Вычисляем схожесть с object seeds
    obj_weight = 0.0
    if object_seeds:
        similarities = [
            np.exp(-np.linalg.norm(image_array[y, x] - image_array[pixel[1], pixel[0]]))
            for x, y in object_seeds
        ]
        if similarities:
            obj_weight = np.mean(similarities)
        else:
            obj_weight = 0.0

    # Вычисляем схожесть с background seeds
    bg_weight = 0.0
    if background_seeds:
        similarities = [
            np.exp(-np.linalg.norm(image_array[y, x] - image_array[pixel[1], pixel[0]]))
            for x, y in background_seeds
        ]
        if similarities:
            bg_weight = np.mean(similarities)
        else:
            bg_weight = 0.0

    return obj_weight, bg_weight


def add_source_sink(
    graph: Dict[Tuple[int, int], Dict[Tuple[int, int], float]],
    image_array: np.ndarray,
    object_seeds: Set[Tuple[int, int]],
    background_seeds: Set[Tuple[int, int]],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Добавляет исток и сток в граф для алгоритма минимального разреза."""
    source = (-1, -1)
    sink = (-2, -2)

    graph[source] = {}
    graph[sink] = {}

    height, width = image_array.shape[:2]

    # Добавляем рёбра от истока к object seeds
    for pixel in object_seeds:
        graph[source][pixel] = float("inf")
        graph[pixel][source] = float("inf")

    # Добавляем рёбра от background seeds к стоку
    for pixel in background_seeds:
        graph[pixel][sink] = float("inf")
        graph[sink][pixel] = float("inf")

    # Добавляем рёбра от истока и стока к остальным пикселям
    for y in range(height):
        for x in range(width):
            pixel = (x, y)
            if pixel not in object_seeds and pixel not in background_seeds:
                obj_weight, bg_weight = calculate_tlink_weights(
                    pixel, image_array, object_seeds, background_seeds
                )
                graph[source][pixel] = obj_weight
                graph[pixel][source] = obj_weight
                graph[pixel][sink] = bg_weight
                graph[sink][pixel] = bg_weight

    return source, sink


def bfs(
    residual_graph: Dict[Tuple[int, int], Dict[Tuple[int, int], float]],
    source: Tuple[int, int],
    sink: Tuple[int, int],
    parent: Dict[Tuple[int, int], Tuple[int, int]],
) -> bool:
    """Поиск в ширину для нахождения увеличивающего пути в алгоритме Форда-Фалкерсона."""
    visited = set()
    queue = deque()
    queue.append(source)
    visited.add(source)

    while queue:
        u = queue.popleft()

        for v, capacity in residual_graph[u].items():
            if v not in visited and capacity > 0:
                queue.append(v)
                visited.add(v)
                parent[v] = u
                if v == sink:
                    return True
    return False


def ford_fulkerson(
    graph: Dict[Tuple[int, int], Dict[Tuple[int, int], float]],
    source: Tuple[int, int],
    sink: Tuple[int, int],
) -> Tuple[Dict[Tuple[int, int], Dict[Tuple[int, int], float]], Set[Tuple[int, int]]]:
    """Реализация алгоритма Форда-Фалкерсона для нахождения минимального разреза."""
    # Создаем остаточный граф
    residual_graph = {
        u: {v: capacity for v, capacity in neighbors.items()}
        for u, neighbors in graph.items()
    }

    parent = {}
    max_flow = 0

    # Находим увеличивающие пути пока это возможно
    while bfs(residual_graph, source, sink, parent):
        path_flow = float("inf")
        s = sink

        # Находим минимальный поток в пути
        while s != source:
            path_flow = min(path_flow, residual_graph[parent[s]][s])
            s = parent[s]

        # Обновляем остаточные пропускные способности
        v = sink
        while v != source:
            u = parent[v]
            residual_graph[u][v] -= path_flow
            residual_graph[v][u] += path_flow
            v = u

        max_flow += path_flow
        parent = {}

    # Находим все достижимые вершины из истока в остаточном графе
    visited = set()
    queue = deque()
    queue.append(source)
    visited.add(source)

    while queue:
        u = queue.popleft()

        for v, capacity in residual_graph[u].items():
            if v not in visited and capacity > 0:
                visited.add(v)
                queue.append(v)

    return residual_graph, visited


def graph_cut_segmentation(
    image_array: np.ndarray,
    object_points: List[Tuple[int, int]],
    background_points: List[Tuple[int, int]],
    lambda_: float = 1.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """Основная функция для выполнения сегментации с помощью разрезов."""
    # Преобразуем списки точек в множества
    object_seeds = set((x, y) for x, y in object_points)
    background_seeds = set((x, y) for x, y in background_points)

    # Строим граф
    graph = build_graph(image_array, lambda_, sigma)

    # Добавляем исток и сток
    source, sink = add_source_sink(graph, image_array, object_seeds, background_seeds)

    # Находим минимальный разрез
    residual_graph, reachable = ford_fulkerson(graph, source, sink)

    # Создаем маску (1 - объект, 0 - фон)
    height, width = image_array.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            if (x, y) in reachable:
                mask[y, x] = 1

    for x, y in object_seeds:
        mask[y, x] = 1

    for x, y in background_seeds:
        mask[y, x] = 0

    return mask


def select_points(image_array: np.ndarray, title: str) -> List[Tuple[int, int]]:
    """Интерактивный выбор точек пользователем"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image_array)
    ax.set_title(
        f"{title} (Кликните левой кнопкой чтобы добавить точку, правой - завершить)"
    )

    points = []

    def on_click(event):
        if event.inaxes != ax:
            return

        if event.button == 1:
            x, y = int(event.xdata), int(event.ydata)
            points.append((x, y))
            ax.plot(x, y, "ro")
            fig.canvas.draw()
            print(f"Добавлена точка: {(x, y)}")
        elif event.button == 3:
            plt.close()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    return points


def apply_mask(image_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Накладываем маску на изображение"""
    masked_image = np.zeros_like(image_array)
    for c in range(3):
        masked_image[:, :, c] = np.where(mask == 1, image_array[:, :, c], 0)
    return masked_image


def main() -> None:
    # Параметры алгоритма
    lambda_ = 1.0
    sigma = 1.0
    image_path = "origins/bb.jpg"

    # Загрузка изображения
    img = Image.open(image_path)
    img_array = np.array(img)

    # Выбор точек для объекта
    print("Выберите точки принадлежащие объекту (object):")
    object_points = select_points(img_array, "Выбор точек объекта")

    # Выбор точек для фона
    print("Выберите точки принадлежащие фону (background):")
    background_points = select_points(img_array, "Выбор точек фона")

    # Выполняем сегментацию
    mask = graph_cut_segmentation(
        img_array, object_points, background_points, lambda_, sigma
    )

    # Применяем маску к изображению
    result = apply_mask(img_array, mask)

    # Визуализация результатов
    plt.figure(figsize=(15, 7))

    # Исходное изображение с отмеченными точками
    plt.subplot(1, 2, 1)
    plt.imshow(img_array)
    if object_points:
        fx, fy = zip(*object_points)
        plt.scatter(fx, fy, c="r", label="Object")
    if background_points:
        bx, by = zip(*background_points)
        plt.scatter(bx, by, c="b", label="Background")
    plt.title("Исходное изображение с seeds")
    plt.legend()

    # Результат сегментации
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title("Результат сегментации")

    plt.tight_layout()
    plt.savefig("results/segmentation_result.png")
    plt.show()


if __name__ == "__main__":
    main()

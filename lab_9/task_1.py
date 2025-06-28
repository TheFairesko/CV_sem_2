import heapq
from collections import defaultdict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Вычисляет евклидово расстояние между двумя векторами."""
    return np.linalg.norm(a - b)


class Neuron:
    def __init__(self, position: np.ndarray):
        """Инициализирует нейрон с заданной позицией (вектором признаков)."""
        self.position = np.array(position, dtype=np.float64)
        self.error = 0.0
        self.neighbors = set()
        self.age = 0

    def add_neighbor(self, neuron) -> None:
        """Добавляет соседний нейрон."""
        self.neighbors.add(neuron)
        neuron.neighbors.add(self)

    def remove_neighbor(self, neuron) -> None:
        """Удаляет соседний нейрон."""
        if neuron in self.neighbors:
            self.neighbors.remove(neuron)
        if self in neuron.neighbors:
            neuron.neighbors.remove(self)


def image_to_samples(
    image_path: str, max_samples: int = 10000
) -> Tuple[np.ndarray, StandardScaler]:
    """Преобразование изображения в набор векторов признаков (x,y,r,g,b)"""
    img = Image.open(image_path)
    img_array = np.array(img)

    h, w = img_array.shape[:2]

    # Создаем сетку координат
    yy, xx = np.mgrid[:h, :w]

    # Объединяем координаты и цвета
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    colors = img_array.reshape(-1, 3)
    samples = np.column_stack([coords, colors])

    # Выбираем случайное подмножество пикселей для ускорения обработки
    if len(samples) > max_samples:
        indices = np.random.choice(len(samples), max_samples, replace=False)
        samples = samples[indices]

    # Нормализация данных
    scaler = StandardScaler()
    return scaler.fit_transform(samples), scaler


def growing_neural_gas(
    samples: np.ndarray,
    max_neurons: int = 50,
    max_iter: int = 5000,
    ew: float = 0.1,
    en: float = 0.01,
    alpha: float = 0.5,
    beta: float = 0.999,
    max_age: int = 50,
    lambda_iter: int = 100,
) -> List[Neuron]:
    """Реализация алгоритма растущего нейронного газа"""

    # Инициализация с двух случайных нейронов
    neurons = [Neuron(samples[np.random.choice(len(samples))]) for _ in range(2)]
    neurons[0].add_neighbor(neurons[1])

    for iteration in range(max_iter):
        # 1. Выбор случайного пикселя
        sample = samples[np.random.choice(len(samples))]

        # 2. Поиск двух ближайших нейронов
        distances = [(euclidean_distance(sample, n.position), n) for n in neurons]
        heapq.heapify(distances)
        d1, winner = heapq.heappop(distances)  # Ближайший
        d2, second = heapq.heappop(distances)  # Второй ближайший

        # 3. Обновление ошибки победителя
        winner.error += d1**2

        # 4. Адаптация позиций
        winner.position += ew * (sample - winner.position)  # Двигаем победителя
        for neighbor in winner.neighbors:  # Двигаем соседей
            neighbor.position += en * (sample - neighbor.position)

        # 5. Обновление топологии
        for neighbor in list(winner.neighbors):
            neighbor.age += 1
            winner.age += 1

            # Удаляем старые соединения
            if neighbor.age > max_age:
                winner.remove_neighbor(neighbor)

        # 6. Создание нового соединения
        if second not in winner.neighbors:
            winner.add_neighbor(second)
            winner.age = 0
            second.age = 0

        # 7. Удаление нейронов без соединений
        if len(neurons) > 2:
            neurons = [n for n in neurons if len(n.neighbors) > 0]

        # 8. Добавление новых нейронов
        if iteration % lambda_iter == 0 and len(neurons) < max_neurons:
            u = max(neurons, key=lambda n: n.error)  # Нейрон с max ошибкой
            if u.neighbors:
                v = max(u.neighbors, key=lambda n: n.error)  # Его сосед с max ошибкой

                # Создаем новый нейрон между u и v
                r_pos = 0.5 * (u.position + v.position)
                r = Neuron(r_pos)

                # Обновляем соединения
                u.remove_neighbor(v)
                u.add_neighbor(r)
                r.add_neighbor(v)

                # Корректируем ошибки
                r.error = u.error
                u.error *= alpha
                v.error *= alpha

                neurons.append(r)

        # 9. Уменьшение ошибок
        for neuron in neurons:
            neuron.error *= beta

    return neurons


def assign_pixels_to_neurons(
    image_path: str, neurons: List[Neuron], scaler: StandardScaler
) -> np.ndarray:
    """Присваивает каждый пиксель изображения ближайшему нейрону."""
    img = Image.open(image_path)
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    # Создаем пустую карту кластеров
    cluster_map = np.zeros((h, w), dtype=np.int32)

    # Подготовка данных пикселей
    yy, xx = np.mgrid[:h, :w]
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    colors = img_array.reshape(-1, 3)
    pixel_samples = np.column_stack([coords, colors])
    pixel_samples = scaler.transform(pixel_samples)

    # Создаем список позиций нейронов для быстрого доступа
    neuron_positions = np.array([n.position for n in neurons])

    # Для каждого пикселя находим ближайший нейрон
    for i in range(len(pixel_samples)):
        x, y = int(coords[i][0]), int(coords[i][1])
        distances = np.linalg.norm(neuron_positions - pixel_samples[i], axis=1)
        closest_idx = np.argmin(distances)
        cluster_map[y, x] = closest_idx

    return cluster_map


def visualize_clusters(
    image_path: str, cluster_map: np.ndarray, neurons: List[Neuron]
) -> None:
    """Визуализирует результаты кластеризации."""
    img = Image.open(image_path)
    img_array = np.array(img)

    # Получаем цвета для кластеров из позиций нейронов
    cluster_colors = np.array([n.position[-3:] for n in neurons])

    # Масштабируем цвета обратно к диапазону [0, 255]
    cluster_colors = (
        255
        * (cluster_colors - cluster_colors.min())
        / (cluster_colors.max() - cluster_colors.min())
    )
    cluster_colors = cluster_colors.astype(np.uint8)

    # Создаем изображение кластеров
    clustered_img = cluster_colors[cluster_map]

    # Визуализация
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_array)
    plt.title("Исходное изображение")

    plt.subplot(1, 3, 2)
    plt.imshow(clustered_img)
    plt.title("Результат кластеризации")

    plt.subplot(1, 3, 3)
    plt.imshow(cluster_map, cmap="tab20")
    plt.title("Карта кластеров")

    plt.tight_layout()
    plt.savefig("results/gng_clustering_1.png")
    plt.show()


def main() -> None:
    """Основная функция для выполнения кластеризации изображения."""
    # Параметры алгоритма
    image_path = "origins/friren.jpg"
    max_neurons = 50  # Максимальное количество нейронов (кластеров)
    max_samples = 10000  # Максимальное количество пикселей для обучения

    # 1. Преобразуем изображение в набор векторов признаков
    samples, scaler = image_to_samples(image_path, max_samples)

    # 2. Применяем алгоритм растущего нейронного газа
    neurons = growing_neural_gas(samples, max_neurons=max_neurons)
    print(f"Обучение завершено. Создано {len(neurons)} нейронов.")

    # 3. Присваиваем каждый пиксель ближайшему нейрону
    cluster_map = assign_pixels_to_neurons(image_path, neurons, scaler)

    # 4. Визуализируем результаты
    visualize_clusters(image_path, cluster_map, neurons)


if __name__ == "__main__":
    main()

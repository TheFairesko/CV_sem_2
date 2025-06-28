from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import os
import random
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

# Ссылка на датасет -- https://www.kaggle.com/datasets/akhiljethwa/forest-vs-desert

# Устанавливаем seed для повторяемости
random_state = 2025
np.random.seed(random_state)

def extract_features(image_path: str) -> NDArray[np.float64]:
    """Извлекает признаки изображения."""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img) / 255.0 
    
    brightness = np.mean(img_array)
    green_ratio = np.mean(img_array[:, :, 1]) 
    contrast = np.std(img_array)
    
    return np.array([brightness, green_ratio, contrast])

class ActiveLearningKNN:
    def __init__(self, k: int = 3) -> None:
        self.k = k  
        self.X_labeled = None  
        self.y_labeled = None 
        self.X_unlabeled = None
        self.image_paths_unlabeled = None

    def initialize_labeled_data(self, X_init: NDArray[np.float64], y_init: NDArray[str]) -> None:
        """Инициализирует начальный размеченный набор данных."""
        self.X_labeled = X_init
        self.y_labeled = y_init

    def set_unlabeled_pool(self, X_pool: NDArray[np.float64], image_paths_pool: List[str]) -> None:
        """Устанавливает пул неразмеченных данных."""
        self.X_unlabeled = X_pool
        self.image_paths_unlabeled = image_paths_pool

    def predict_proba(self, X_test: NDArray[np.float64]) -> List[Dict[str, float]]:
        """Возвращает вероятности классов для каждого тестового образца."""
        probas = []
        for x in X_test:
            distances = [np.linalg.norm(x - train_x) for train_x in self.X_labeled]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_labeled[i] for i in k_indices]
            
            # Вычисляем вероятности классов
            class_counts = Counter(k_nearest_labels)
            proba = {cls: count/self.k for cls, count in class_counts.items()}
            probas.append(proba)
        
        return probas

    def query_samples(self, n_samples: int = 5) -> NDArray[np.int64]:
        """Выбирает наиболее информативные образцы для разметки."""
        probas = self.predict_proba(self.X_unlabeled)
        
        # Минимальная уверенность (Least Confidence)
        # Вычисляем неопределенность (1 - максимальная вероятность)
        uncertainties = []
        for p in probas:
            if len(p) == 0:
                uncertainties.append(1.0)
            else:
                max_proba = max(p.values())
                uncertainties.append(1 - max_proba)
        
        # Выбираем индексы наиболее неопределенных образцов
        query_indices = np.argsort(uncertainties)[-n_samples:]
        
        return query_indices

    def teach(self, query_indices: NDArray[np.int64], new_labels: List[str]) -> None:
        """Добавляет новые размеченные образцы в обучающий набор."""
        new_X = [self.X_unlabeled[i] for i in query_indices]
        new_y = new_labels
        
        # Добавляем к размеченным данным
        if self.X_labeled is None:
            self.X_labeled = np.array(new_X)
            self.y_labeled = np.array(new_y)
        else:
            self.X_labeled = np.vstack((self.X_labeled, new_X))
            self.y_labeled = np.concatenate((self.y_labeled, new_y))
        
        # Удаляем из неразмеченного пула
        mask = np.ones(len(self.X_unlabeled), dtype=bool)
        mask[query_indices] = False
        self.X_unlabeled = self.X_unlabeled[mask]
        self.image_paths_unlabeled = [self.image_paths_unlabeled[i] for i in range(len(mask)) if mask[i]]

def display_image_with_prediction(
    image_path: str, 
    predicted_class: str, 
    predicted_proba: float
) -> None:
    """Отображает изображение с информацией о предсказании."""
    img = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    
    prediction_text = f"Предсказание: {predicted_class} ({predicted_proba*100:.1f}%)"
    plt.title(prediction_text, fontsize=14, pad=20)
    plt.axis('off')
    
    plt.text(10, img.size[1] + 30, 
             "Введите класс изображения (1 - лес, 2 - пустыня, 3 - пропустить):", 
             fontsize=12, color='black')
    
    plt.tight_layout()
    plt.show()

def get_user_input() -> str:
    """Получает ввод от пользователя."""
    while True:
        try:
            user_input = input("Ваш выбор (1/2/3): ").strip()
            if user_input in ['1', '2', '3']:
                return user_input
            print("Пожалуйста, введите 1, 2 или 3")
        except:
            print("Некорректный ввод. Попробуйте еще раз.")

def active_learning_demo() -> None:
    # Путь к датасету
    DATASET_PATH = "origins/data"
    
    # 1. Загружаем датасет
    X_all, y_all, paths_all = [], [], []
    for class_name in ['forest', 'desert']:
        class_dir = os.path.join(DATASET_PATH, class_name)
        images = os.listdir(class_dir)
        for img_file in images:
            img_path = os.path.join(class_dir, img_file)
            features = extract_features(img_path)
            X_all.append(features)
            y_all.append(class_name)
            paths_all.append(img_path)
    
    # 2. Однократное разделение (30% init, 50% pool, 20% test)
    indices = np.arange(len(X_all))
    np.random.shuffle(indices)
    
    split_init = int(0.3 * len(X_all))
    split_pool = int(0.8 * len(X_all))
    
    # Init (30%)
    init_indices = indices[:split_init]
    X_init = np.array([X_all[i] for i in init_indices])
    y_init = np.array([y_all[i] for i in init_indices])
    
    # Pool (50%)
    pool_indices = indices[split_init:split_pool]
    X_pool = np.array([X_all[i] for i in pool_indices])
    pool_paths = [paths_all[i] for i in pool_indices]
    
    # Test (20%)
    test_indices = indices[split_pool:]
    X_test = np.array([X_all[i] for i in test_indices])
    y_test = np.array([y_all[i] for i in test_indices])
    
    print("=== Оптимизированное разделение ===")
    print(f"Начальный размеченный набор: {len(X_init)} изображений")
    print(f"Пул неразмеченных данных: {len(X_pool)} изображений") 
    print(f"Тестовый набор: {len(X_test)} изображений")
    
    # 3. Инициализация модели
    al_knn = ActiveLearningKNN(k=5)
    al_knn.initialize_labeled_data(X_init, y_init)
    al_knn.set_unlabeled_pool(X_pool, pool_paths)
    
    # Количество итераций активного обучения
    n_iterations = 3
    n_queries = 5
    
    print("\n=== Начало Active Learning ===")
    print(f"Начальный размеченный набор: {len(al_knn.X_labeled)} изображений")
    print(f"Пул неразмеченных данных: {len(al_knn.X_unlabeled)} изображений")
    print(f"Тестовый набор: {len(X_test)} изображений (для оценки модели)")
    
    # Оценка начальной точности
    y_pred = []
    for x in X_test:
        probas = al_knn.predict_proba([x])[0]
        pred_class = max(probas.items(), key=lambda x: x[1])[0] if probas else "forest"
        y_pred.append(pred_class)
    initial_accuracy = np.mean(np.array(y_pred) == y_test)
    print(f"\nНачальная точность на тестовом наборе: {initial_accuracy * 100:.1f}%")
    
    for i in range(n_iterations):
        print(f"\n=== Итерация {i+1}/{n_iterations} ===")
        
        # 1. Выбираем наиболее информативные образцы для разметки
        query_indices = al_knn.query_samples(n_queries)
        query_image_paths = [al_knn.image_paths_unlabeled[idx] for idx in query_indices]
        
        new_labels = []
        for j, img_path in enumerate(query_image_paths):
            # 2. Получаем предсказание модели
            features = extract_features(img_path)
            probas = al_knn.predict_proba([features])[0]
            predicted_class = max(probas.items(), key=lambda x: x[1])[0] if probas else "не определено"
            predicted_proba = max(probas.values()) if probas else 0
            
            # 3. Показываем изображение и запрашиваем разметку
            clear_output(wait=True)
            display_image_with_prediction(img_path, predicted_class, predicted_proba)
            
            # 4. Получаем ввод пользователя
            user_input = get_user_input()
            
            if user_input == '1':
                new_labels.append('forest')
                print("Вы выбрали: лес")
            elif user_input == '2':
                new_labels.append('desert')
                print("Вы выбрали: пустыня")
            else:
                print("Изображение пропущено")
                continue
            
            time.sleep(0.5)
        
        # 5. Добавляем размеченные образцы в обучающий набор
        if new_labels:
            al_knn.teach(query_indices[:len(new_labels)], new_labels)
        
        # 6. Оцениваем точность на тестовом наборе
        y_pred = []
        for x in X_test:
            probas = al_knn.predict_proba([x])[0]
            pred_class = max(probas.items(), key=lambda x: x[1])[0] if probas else "forest"
            y_pred.append(pred_class)
        accuracy = np.mean(np.array(y_pred) == y_test)
        
        print(f"\nТочность на тестовом наборе: {accuracy * 100:.1f}%")
        print(f"Размер обучающего набора: {len(al_knn.X_labeled)}")
        print(f"Осталось неразмеченных изображений: {len(al_knn.X_unlabeled)}")
        
        if len(al_knn.X_unlabeled) == 0:
            print("\nВсе изображения размечены!")
            break
    
    print("\n=== Active Learning завершен ===")
    print(f"Итоговый размер обучающего набора: {len(al_knn.X_labeled)}")
    print(f"Итоговая точность на тестовом наборе: {accuracy * 100:.1f}%")

# Запуск демонстрации
if __name__ == "__main__":
    active_learning_demo()
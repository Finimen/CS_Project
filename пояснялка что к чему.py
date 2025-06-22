import cv2  # Основная библиотека для работы с компьютерным зрением
import numpy as np  # Для числовых операций и работы с массивами
import pandas as pd  # Для хранения и обработки данных статистики
from datetime import datetime  # Для работы с временными метками
from matplotlib import pyplot as plt  # Для визуализации графиков

class PeopleCounter:
    def __init__(self, entry_line_position=150):
        """
        Инициализация счетчика людей

        Args:
            entry_line_position (int): Y-координата линии входа/выхода
        """
        # Загрузка каскада Хаара для детекции лиц
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.trackers = []  # Список активных трекеров
        self.people_count = 0  # Текущее количество людей
        self.entry_line_y = entry_line_position  # Позиция контрольной линии
        self.positions = {}  # Словарь для хранения позиций объектов (id: y_position)
        self.log = pd.DataFrame(columns=['timestamp', 'event', 'count'])  # Лог событий
        self.initial_detection_done = False  # Флаг первичной детекции

    def process_frame(self, frame):
        """
        Основной метод обработки кадра

        Args:
            frame (numpy.ndarray): Входной кадр видео

        Returns:
            numpy.ndarray: Кадр с визуализацией
        """
        # Конвертация в grayscale для детекции
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Детекция лиц с параметрами:
        # scaleFactor - компенсация разных масштабов
        # minNeighbors - фильтрация ложных срабатываний
        # minSize - минимальный размер объекта
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))

        # Первичная инициализация счетчика
        if not self.initial_detection_done and len(faces) > 0:
            self.people_count = len(faces)
            for _ in faces:
                self._log_event('initial_detection')
            self.initial_detection_done = True

        # Обновление трекеров и проверка пересечений
        self._update_trackers(frame, faces)
        self._check_line_crossings()

        return self._draw_visualization(frame)

    def _update_trackers(self, frame, detections):
        """
        Обновление состояния трекеров

        Args:
            frame: Текущий кадр
            detections: Обнаруженные лица
        """
        active_trackers = []

        # Обновление существующих трекеров
        for tracker, rect in self.trackers:
            success, box = tracker.update(frame)
            if success:
                active_trackers.append((tracker, box))

        self.trackers = active_trackers

        # Добавление новых трекеров
        for (x, y, w, h) in detections:
            if not self._is_duplicate((x, y, w, h)):
                tracker = cv2.TrackerKCF_create()  # Используем трекер KCF
                tracker.init(frame, (x, y, w, h))
                self.trackers.append((tracker, (x, y, w, h)))

    def _is_duplicate(self, new_rect):
        """
        Проверка дублирования обнаружений

        Args:
            new_rect: Координаты нового обнаружения (x,y,w,h)

        Returns:
            bool: True если это дубликат существующего трекера
        """
        x, y, w, h = new_rect
        new_center = (x + w / 2, y + h / 2)  # Центр нового прямоугольника

        # Проверка расстояния до существующих трекеров
        for tracker, box in self.trackers:
            tx, ty, tw, th = map(int, box)
            t_center = (tx + tw / 2, ty + th / 2)
            distance = np.sqrt((new_center[0] - t_center[0]) ** 2 +
                             (new_center[1] - t_center[1]) ** 2)
            if distance < 50:  # Пороговое значение для объединения
                return True
        return False

    def _check_line_crossings(self):
        """
        Проверка пересечения контрольной линии
        """
        for tracker, box in self.trackers:
            _, y, _, h = map(int, box)
            center_y = y + h / 2  # Y-координата центра объекта
            tracker_id = id(tracker)  # Уникальный идентификатор трекера

            if tracker_id in self.positions:
                prev_y = self.positions[tracker_id]

                # Логика определения пересечения:
                # Сверху вниз - выход
                if prev_y < self.entry_line_y and center_y >= self.entry_line_y:
                    self.people_count = max(0, self.people_count - 1)
                    self._log_event('exit')
                # Снизу вверх - вход
                elif prev_y > self.entry_line_y and center_y <= self.entry_line_y:
                    self.people_count += 1
                    self._log_event('entry')

            self.positions[tracker_id] = center_y

    def _log_event(self, event_type):
        """
        Логирование события

        Args:
            event_type: Тип события ('entry'/'exit')
        """
        self.log = pd.concat([
            self.log,
            pd.DataFrame([{
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'event': event_type,
                'count': self.people_count
            }])
        ], ignore_index=True)

    def _draw_visualization(self, frame):
        """
        Визуализация результатов

        Args:
            frame: Исходный кадр

        Returns:
            Кадр с элементами визуализации
        """
        # Рисование контрольной линии
        cv2.line(frame,
                 (0, self.entry_line_y),
                 (frame.shape[1], self.entry_line_y),
                 (0, 255, 255), 2)  # Желтый цвет

        # Отображение счетчика
        cv2.putText(frame, f"People: {self.people_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Рисование bounding box'ов
        for tracker, box in self.trackers:
            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return frame

    def generate_report(self):
        """
        Генерация отчетов
        """
        # Сохранение лога в CSV
        self.log.to_csv('people_count.csv', index=False)

        # Построение графика
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(self.log['timestamp']), self.log['count'])
        plt.title('People Count Over Time')
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.grid()
        plt.tight_layout()
        plt.savefig('people_count.png')
        plt.close()


if __name__ == "__main__":
    # Инициализация видеозахвата
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: камера не доступна")
        exit()

    # Создание счетчика с линией на y=200
    counter = PeopleCounter(entry_line_position=200)

    # Основной цикл обработки
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обработка и отображение кадра
        frame = counter.process_frame(frame)
        cv2.imshow('People Counter', frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

    # Генерация отчетов
    counter.generate_report()

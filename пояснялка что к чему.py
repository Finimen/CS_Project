import cv2
import numpy as np
import pandas as pd
from datetime import datetime


class PeopleCounter:
    def __init__(self, entry_line_position=150):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.trackers = {}  # {id: {'tracker': tracker, 'positions': [], 'counted': bool, 'skin_confirmed': bool}}
        self.next_id = 0
        self.people_count = 0
        self.entry_line_y = entry_line_position
        self.crossing_threshold = 15
        self.log = pd.DataFrame(columns=['timestamp', 'event', 'count'])

        # Параметры для фильтрации ложных срабатываний
        self.min_face_ratio = 0.7  # Минимальное соотношение width/height для лица
        self.max_face_ratio = 1.5  # Максимальное соотношение width/height
        self.skin_lower = np.array([0, 48, 80], dtype=np.uint8)  # Нижняя граница цвета кожи (HSV)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)  # Верхняя граница
        self.min_skin_pixels = 0.15  # Минимальный процент пикселей кожи в области лица
        self.match_threshold = 0.7  # Порог для сопоставления детекций с трекерами

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Детекция лиц с дополнительными проверками
        faces = []
        raw_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))

        for (x, y, w, h) in raw_faces:
            # Проверка пропорций лица
            ratio = w / h
            if ratio < self.min_face_ratio or ratio > self.max_face_ratio:
                continue

            # Проверка цвета кожи
            face_roi = frame[y:y + h, x:x + w]
            face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(face_hsv, self.skin_lower, self.skin_upper)
            skin_percent = cv2.countNonZero(skin_mask) / (w * h)

            if skin_percent >= self.min_skin_pixels:
                faces.append((x, y, w, h))
            else:
                # Визуализация отбракованных объектов
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(frame, f"Not face: {skin_percent:.1%}",
                            (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        self._update_trackers(frame, faces)
        self._check_line_crossings()

        return self._draw_visualization(frame)

    def _update_trackers(self, frame, detections):
        # Удаляем трекеры, которые вышли за границы кадра
        dead_trackers = []
        for t_id in self.trackers:
            if len(self.trackers[t_id]['positions']) > 0:
                last_pos = self.trackers[t_id]['positions'][-1]
                if (len(self.trackers[t_id]['positions']) > 10 and 
                    (last_pos < 0 or last_pos > frame.shape[0])):
                    dead_trackers.append(t_id)

        for t_id in dead_trackers:
            del self.trackers[t_id]

        active_ids = set()

        # Обновляем существующие трекеры
        for tracker_id in list(self.trackers.keys()):
            tracker_data = self.trackers[tracker_id]
            success, box = tracker_data['tracker'].update(frame)

            if success:
                x, y, w, h = map(int, box)
                center_y = y + h / 2
                tracker_data['positions'].append(center_y)

                if len(tracker_data['positions']) > 10:
                    tracker_data['positions'].pop(0)

                active_ids.add(tracker_id)
            else:
                del self.trackers[tracker_id]

        # Обрабатываем новые обнаружения
        for (x, y, w, h) in detections:
            center = (x + w / 2, y + h / 2)
            matched = False
            min_dist = h * self.match_threshold

            # Сначала пытаемся сопоставить с существующими трекерами
            for t_id in list(self.trackers.keys()):
                if t_id in active_ids:  # Уже обновленный трекер
                    continue
                    
                if not self.trackers[t_id]['positions']:
                    continue

                last_pos = self.trackers[t_id]['positions'][-1]
                dist = abs(center[1] - last_pos)

                if dist < min_dist:
                    # Обновляем существующий трекер
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, (x, y, w, h))
                    self.trackers[t_id]['tracker'] = tracker
                    self.trackers[t_id]['positions'].append(center[1])
                    matched = True
                    active_ids.add(t_id)
                    break

            if not matched:
                # Проверяем пересечение с существующими прямоугольниками
                overlapping = False
                current_rect = (x, y, w, h)
                
                for t_id in self.trackers:
                    if t_id in active_ids:
                        _, t_box = self.trackers[t_id]['tracker'].update(frame)
                        if self._is_overlapping(current_rect, tuple(map(int, t_box))):
                            overlapping = True
                            break
                
                if not overlapping:
                    # Создаем новый трекер только если не нашли соответствия
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, (x, y, w, h))
                    self.next_id += 1
                    self.trackers[self.next_id] = {
                        'tracker': tracker,
                        'positions': [y + h / 2],
                        'counted': False,
                        'skin_confirmed': True
                    }
                    active_ids.add(self.next_id)

    def _is_overlapping(self, rect1, rect2):
        """Проверяет пересекаются ли два прямоугольника"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Проверка пересечения по оси X
        x_overlap = not (x1 + w1 < x2 or x2 + w2 < x1)
        
        # Проверка пересечения по оси Y
        y_overlap = not (y1 + h1 < y2 or y2 + h2 < y1)
        
        return x_overlap and y_overlap

    def _check_line_crossings(self):
        for tracker_id in self.trackers:
            tracker_data = self.trackers[tracker_id]

            if len(tracker_data['positions']) < 2:
                continue

            prev_pos = tracker_data['positions'][-2]
            curr_pos = tracker_data['positions'][-1]

            direction = 'down' if curr_pos > prev_pos else 'up'

            if ((prev_pos < self.entry_line_y and curr_pos >= self.entry_line_y) or
                    (prev_pos > self.entry_line_y and curr_pos <= self.entry_line_y)):

                if not tracker_data['counted'] and tracker_data.get('skin_confirmed', False):
                    if direction == 'down':
                        self.people_count = max(0, self.people_count - 1)
                        self._log_event('exit')
                    else:
                        self.people_count += 1
                        self._log_event('entry')

                    tracker_data['counted'] = True
            else:
                if abs(curr_pos - self.entry_line_y) > self.crossing_threshold:
                    tracker_data['counted'] = False

    def _log_event(self, event_type):
        self.log = pd.concat([
            self.log,
            pd.DataFrame([{
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'event': event_type,
                'count': self.people_count
            }])
        ], ignore_index=True)

    def _draw_visualization(self, frame):
        cv2.line(frame,
                 (0, self.entry_line_y),
                 (frame.shape[1], self.entry_line_y),
                 (0, 255, 255), 2)

        cv2.putText(frame, f"People: {self.people_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for tracker_id, tracker_data in self.trackers.items():
            if len(tracker_data['positions']) < 3:  # Показываем только трекеры с историей
                continue
                
            success, box = tracker_data['tracker'].update(frame)
            if success:
                x, y, w, h = map(int, box)
                color = (0, 0, 255) if tracker_data['counted'] else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                status = "counted" if tracker_data['counted'] else "tracking"
                cv2.putText(frame, f"ID:{tracker_id} {status}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return frame

    def generate_report(self):
        self.log.to_csv('people_count_log.csv', index=False)
        print("Report generated: people_count_log.csv")


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()

    counter = PeopleCounter(entry_line_position=200)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = counter.process_frame(frame)
        cv2.imshow('People Counter', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    counter.generate_report()

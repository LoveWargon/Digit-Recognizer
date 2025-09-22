import sys
import os
import io
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort
import cv2
from scipy import ndimage

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSlider, QFrame, QMessageBox, QProgressBar,
    QDialog, QGraphicsOpacityEffect, QGroupBox, QSizePolicy, QGridLayout,
    QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QThread, Signal, QByteArray, QBuffer, QIODevice, QPropertyAnimation, QEasingCurve, QMargins
from PySide6.QtGui import (
    QPainter, QPen, QColor, QImage, QPixmap, QIcon, QKeySequence, QFont, QShortcut, QLinearGradient # Добавлено QLinearGradient
)
# Qt Charts
from PySide6.QtCharts import QChart, QChartView, QBarSeries, QBarSet, QBarCategoryAxis, QValueAxis

# ---------- Helper ----------
def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # type: ignore
        rel = os.path.join('src', relative_path)
    except Exception:
        base_path = os.path.abspath(".")
        rel = relative_path
    return os.path.join(base_path, rel)

# ---------- Worker ----------
# Используем улучшенную версию из второго файла
class InferenceWorker(QThread):
    result_ready = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, session: ort.InferenceSession, img_array: np.ndarray):
        super().__init__()
        self.session = session
        self.img = img_array.astype(np.float32)

    def run(self):
        try:
            name = self.session.get_inputs()[0].name
            outs = self.session.run(None, {name: self.img})
            # Берём первый тензор выхода и аккуратно приводим к форме (n_classes,)
            out = outs[0]
            if hasattr(out, "shape") and out.ndim > 1:
                out = out[0]
            out = np.asarray(out).astype(np.float32)
            # Нормализуем (на всякий случай), если суммы не 1
            if out.sum() > 0:
                out = out / out.sum()
            self.result_ready.emit(out)
        except Exception as e:
            self.error.emit(str(e))

# ---------- Drawing Widget ----------
# Используем UI версию из первого файла
class DrawingWidget(QWidget):
    def __init__(self, size: int = 280, brush: int = 12):
        super().__init__()
        self.setFixedSize(size, size)
        self.size_px = size
        self.brush_size = brush
        self.pen_color = QColor("black")
        self._image = QImage(self.size_px, self.size_px, QImage.Format_Grayscale8)
        self.clear()
        self.last_pos = None
        self.setCursor(Qt.CrossCursor)
        # Добавляем рамку для визуального выделения холста
        self.setStyleSheet("border: 2px solid #4a90e2; border-radius: 8px;")

    def clear(self):
        self._image.fill(255)
        self.update()

    def set_brush(self, size: int):
        self.brush_size = max(1, int(size))

    def paintEvent(self, event):
        p = QPainter(self)
        pix = QPixmap.fromImage(self._image)
        p.drawPixmap(0, 0, pix)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_pos = event.position() if hasattr(event, 'position') else event.pos()
            self._draw_point(self.last_pos)
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.last_pos is not None:
            pos = event.position() if hasattr(event, 'position') else event.pos()
            self._draw_line(self.last_pos, pos)
            self.last_pos = pos
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_pos = None

    def _draw_point(self, pos):
        painter = QPainter(self._image)
        pen = QPen(self.pen_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        x, y = int(pos.x()), int(pos.y())
        painter.drawPoint(x, y)
        painter.end()

    def _draw_line(self, p1, p2):
        painter = QPainter(self._image)
        pen = QPen(self.pen_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(int(p1.x()), int(p1.y()), int(p2.x()), int(p2.y()))
        painter.end()

    def get_pil_image(self) -> Image.Image:
        buffer = QByteArray()
        buf = QBuffer(buffer)
        buf.open(QIODevice.WriteOnly)
        self._image.save(buf, "PNG")
        buf.close()
        pil_img = Image.open(io.BytesIO(buffer.data()))
        if pil_img.mode != "L":
            pil_img = pil_img.convert("L")
        return pil_img

# ---------- Theme transition overlay ----------
# Используем улучшенную версию из второго файла
class ThemeTransitionOverlay(QWidget):
    """Полупрозрачный оверлей с градиентом, который мы показываем/скрываем при смене темы."""
    def __init__(self, parent: QWidget, start_color: str, end_color: str):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setGeometry(parent.rect())
        # Используем стиль градиента от start_color до end_color
        self.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {start_color}, stop:1 {end_color});
        """)
        self._opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._opacity.setOpacity(0.0)
        self.show()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.setGeometry(self.parent().rect())

# ---------- Probability Dialog ----------
# Используем улучшенную UI версию из первого файла
class ProbabilityDialog(QDialog):
    def __init__(self, probabilities: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Вероятности предсказаний")
        self.resize(750, 500)
        self.setModal(True)
        # Добавляем декоративные элементы
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2c3e50, stop:1 #1a2530);
                border-radius: 12px;
            }
            QLabel {
                color: #ecf0f1;
                font-family: "Segoe UI", Arial;
            }
        """)

        # Основной layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Заголовок с декорацией
        header = QLabel("Распределение вероятностей")
        header.setStyleSheet("""
            QLabel {
                color: #3498db;
                font-size: 20px;
                font-weight: bold;
                padding: 12px;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 8px;
                border: 1px solid #3498db;
            }
        """)
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # Создаем график
        chart = QChart()
        chart.setBackgroundRoundness(0)
        chart.setMargins(QMargins(0, 0, 0, 0))
        chart.setAnimationOptions(QChart.SeriesAnimations)

        # Создаем серии данных
        series = QBarSeries()
        max_idx = int(np.argmax(probabilities))

        # Создаем отдельные наборы для каждой цифры
        bar_sets = []
        for i, p in enumerate(probabilities):
            bar_set = QBarSet(str(i))
            bar_set.append(float(p))
            # === Улучшенная цветовая схема с градиентами ===
            if i == max_idx:
                # Зеленый градиент для максимального значения
                gradient = QLinearGradient(0, 0, 0, 400)
                gradient.setColorAt(0, QColor("#27ae60"))
                gradient.setColorAt(1, QColor("#1e8449"))
                bar_set.setBrush(gradient)
                bar_set.setBorderColor(QColor("#2ecc71"))
            elif p > 0.3:
                # Синий градиент для высоких значений
                gradient = QLinearGradient(0, 0, 0, 400)
                gradient.setColorAt(0, QColor("#3498db"))
                gradient.setColorAt(1, QColor("#2874a6"))
                bar_set.setBrush(gradient)
                bar_set.setBorderColor(QColor("#5dade2"))
            elif p > 0.1:
                # Оранжевый градиент для средних значений
                gradient = QLinearGradient(0, 0, 0, 400)
                gradient.setColorAt(0, QColor("#f39c12"))
                gradient.setColorAt(1, QColor("#d35400"))
                bar_set.setBrush(gradient)
                bar_set.setBorderColor(QColor("#f5b041"))
            else:
                # Серый градиент для низких значений
                gradient = QLinearGradient(0, 0, 0, 400)
                gradient.setColorAt(0, QColor("#95a5a6"))
                gradient.setColorAt(1, QColor("#7f8c8d"))
                bar_set.setBrush(gradient)
                bar_set.setBorderColor(QColor("#bdc3c7"))
            # ================================================
            series.append(bar_set)
        chart.addSeries(series)

        # Настройка осей
        categories = [str(i) for i in range(len(probabilities))]
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        axis_x.setLabelsFont(QFont("Arial", 11, QFont.Bold))
        axis_x.setLabelsBrush(QColor("#ecf0f1"))
        axis_x.setTitleText("Цифры")
        axis_x.setTitleFont(QFont("Arial", 12, QFont.Bold))
        axis_x.setTitleBrush(QColor("#3498db"))
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setRange(0, 1)
        axis_y.setTickCount(6)
        axis_y.setLabelsFont(QFont("Arial", 10))
        axis_y.setLabelsBrush(QColor("#ecf0f1"))
        axis_y.setTitleText("Вероятность")
        axis_y.setTitleFont(QFont("Arial", 12, QFont.Bold))
        axis_y.setTitleBrush(QColor("#3498db"))
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        chart.legend().setVisible(True)
        chart.legend().setLabelBrush(QColor("#ecf0f1"))
        chart.legend().setFont(QFont("Arial", 11, QFont.Bold))
        chart.legend().setAlignment(Qt.AlignBottom)

        # Создаем виджет графика
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        # === Улучшенный декор для области графика ===
        chart_view.setStyleSheet("""
            QChartView {
                background: rgba(0, 0, 0, 0.2);
                border-radius: 10px;
                border: 2px solid #34495e;
                padding: 10px;
                /* Добавляем эффект тени */
                border-image: url(:/border_shadow.png) 0 0 0 0 stretch stretch;
            }
        """)
        # Добавляем эффект тени для графика
        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(15)
        shadow_effect.setColor(QColor(0, 0, 0, 100))
        shadow_effect.setOffset(0, 5)
        chart_view.setGraphicsEffect(shadow_effect)
        # =========================================
        main_layout.addWidget(chart_view)

        # Добавляем информационную панель
        info_panel = QFrame()
        info_panel.setStyleSheet("""
            QFrame {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                border: 2px solid #34495e;
                padding: 15px;
            }
        """)
        info_layout = QVBoxLayout()
        # Наиболее вероятная цифра
        max_digit = np.argmax(probabilities)
        max_prob = probabilities[max_digit]
        max_label = QLabel(
            f"Наиболее вероятная цифра: <span style='color: #27ae60; font-weight: bold; font-size: 16px;'>{max_digit}</span> "
            f"(<span style='color: #3498db; font-weight: bold;'>{max_prob:.1%}</span>)")
        max_label.setStyleSheet("font-size: 14px;")
        # Вторая по вероятности
        sorted_indices = np.argsort(probabilities)[::-1]
        second_digit = sorted_indices[1] if len(sorted_indices) > 1 else None
        second_prob = probabilities[second_digit] if second_digit is not None else 0
        second_label = QLabel(
            f"Вторая по вероятности: <span style='color: #3498db; font-weight: bold; font-size: 14px;'>{second_digit if second_digit is not None else 'нет'}</span> "
            f"(<span style='color: #f39c12; font-weight: bold;'>{second_prob:.1%}</span>)" if second_digit is not None else "")
        second_label.setStyleSheet("font-size: 13px;")
        info_layout.addWidget(max_label)
        if second_digit is not None and second_prob > 0.01:
            info_layout.addWidget(second_label)
        info_panel.setLayout(info_layout)
        main_layout.addWidget(info_panel)

        # Кнопка закрытия
        close_btn = QPushButton("Закрыть")
        close_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2980b9, stop:1 #2573a7);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2573a7, stop:1 #1c5a85);
            }
        """)
        close_btn.clicked.connect(self.accept)
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        close_layout.addWidget(close_btn)
        close_layout.addStretch()
        main_layout.addLayout(close_layout)

        self.setLayout(main_layout)

# ---------- Preview Dialog ----------
# Используем улучшенную UI версию из первого файла
class PreviewDialog(QDialog):
    def __init__(self, processed_array: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Анализ обработанного изображения")
        self.resize(380, 480)
        self.setModal(True)
        # Стилизация окна
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2c3e50, stop:1 #1a2530);
                border-radius: 12px;
            }
            QLabel {
                color: #ecf0f1;
                font-family: "Segoe UI", Arial;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Заголовок
        header = QLabel("Обработанное изображение")
        header.setStyleSheet("""
            QLabel {
                color: #3498db;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background: rgba(0, 0, 0, 0.2);
                border-radius: 8px;
            }
        """)
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # Отображение изображения
        arr = processed_array.reshape(28, 28)
        arr_u8 = (arr * 255).astype(np.uint8)
        h, w = arr_u8.shape
        img = QImage(arr_u8.data, w, h, w, QImage.Format_Grayscale8).copy()
        pix = QPixmap.fromImage(img).scaled(280, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img_label = QLabel()
        img_label.setPixmap(pix)
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setStyleSheet("""
            QLabel {
                background: rgba(0, 0, 0, 0.2);
                border-radius: 8px;
                border: 1px solid #34495e;
                padding: 10px;
            }
        """)
        main_layout.addWidget(img_label)

        # Статистика
        stats = arr_u8.flatten()
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 8px;
                border: 1px solid #34495e;
                padding: 15px;
            }
        """)
        stats_layout = QVBoxLayout()
        stats_title = QLabel("Статистика изображения")
        stats_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #3498db; margin-bottom: 10px;")
        stats_title.setAlignment(Qt.AlignCenter)
        stats_label = QLabel(
            f"<table style='width: 100%;'>"
            f"<tr><td>Минимальное значение:</td><td align='right'><span style='color: #27ae60;'>{stats.min() / 255:.3f}</span></td></tr>"
            f"<tr><td>Максимальное значение:</td><td align='right'><span style='color: #e74c3c;'>{stats.max() / 255:.3f}</span></td></tr>"
            f"<tr><td>Среднее значение:</td><td align='right'><span style='color: #3498db;'>{stats.mean() / 255:.3f}</span></td></tr>"
            f"<tr><td>Стандартное отклонение:</td><td align='right'><span style='color: #f39c12;'>{stats.std() / 255:.3f}</span></td></tr>"
            f"</table>"
        )
        stats_label.setStyleSheet("font-size: 12px;")
        stats_layout.addWidget(stats_title)
        stats_layout.addWidget(stats_label)
        stats_frame.setLayout(stats_layout)
        main_layout.addWidget(stats_frame)

        # Кнопка закрытия
        close_btn = QPushButton("Закрыть")
        close_btn.setStyleSheet("""
            QPushButton {
                background: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #2980b9;
            }
            QPushButton:pressed {
                background: #2573a7;
            }
        """)
        close_btn.clicked.connect(self.accept)
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        close_layout.addWidget(close_btn)
        close_layout.addStretch()
        main_layout.addLayout(close_layout)

        self.setLayout(main_layout)

# ---------- Main Window ----------
class ModernDigitRecognizerMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Распознавание цифр — QtCharts & Animations")
        self.setMinimumSize(620, 820)
        self._load_model()
        self._init_themes()
        self.current_theme = "dark"
        self._build_ui()
        self.apply_theme(self.current_theme)  # Применяем тему при запуске

        # храним анимации, overlay и др. (из второго файла)
        self._theme_overlay = None
        self._theme_animation_in = None
        self._theme_animation_out = None
        self._result_anim = None
        self._confidence_anim = None

    def _load_model(self):
        # Используем логику из второго файла
        model_paths = [
            resource_path("resources/models/improved_digit_recognition_model.onnx"),
            "resources/models/improved_digit_recognition_model.onnx",
            "improved_digit_recognition_model.onnx",
        ]
        self.model_session = None
        for p in model_paths:
            try:
                if os.path.exists(p):
                    self.model_session = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
                    break
            except Exception:
                continue
        if self.model_session is None:
            raise FileNotFoundError("Не удалось загрузить ONNX модель. Проверьте пути.")

    def _init_themes(self):
        # Используем стилизованные темы из первого файла
        base_font = "Segoe UI, Arial"
        self.themes = {
            "light": {
                "qss": f"""
                    QMainWindow {{
                        background: #f5f7fa;
                        border: 2px solid #4a90e2; /* Рамка главного окна */
                    }}
                    QWidget {{
                        background: #f5f7fa;
                        color: #2c3e50;
                        font-family: {base_font};
                    }}
                    QFrame#panel {{
                        background: #ffffff;
                        border-radius: 12px;
                        border: 1px solid #e4e7eb;
                    }}
                    QPushButton {{
                        background: #4a90e2;
                        color: white;
                        border: none;
                        padding: 10px 16px;
                        border-radius: 8px;
                        font-weight: 500;
                        font-size: 11px;
                    }}
                    QPushButton:hover {{
                        background: #3a7bc8;
                    }}
                    QPushButton:pressed {{
                        background: #2a6bb8;
                    }}
                    QPushButton#accent {{
                        background: #50c878;
                    }}
                    QPushButton#accent:hover {{
                        background: #42b366;
                    }}
                    QPushButton#danger {{
                        background: #ff6b6b;
                    }}
                    QPushButton#danger:hover {{
                        background: #e55a5a;
                    }}
                    QPushButton#theme {{
                        background: #9b59b6;
                    }}
                    QPushButton#theme:hover {{
                        background: #8e44ad;
                    }}
                    QSlider::groove:horizontal {{
                        background: #e9eef6;
                        height: 8px;
                        border-radius: 4px;
                    }}
                    QSlider::handle:horizontal {{
                        background: #4a90e2;
                        width: 16px;
                        border-radius: 8px;
                        margin: -4px 0;
                    }}
                    QSlider::handle:horizontal:hover {{
                        background: #3a7bc8;
                    }}
                    QLabel#title {{
                        font-size: 20px;
                        font-weight: 700;
                    }}
                    QLabel#subtitle {{
                        color: #7f8c8d;
                        font-size: 13px;
                    }}
                    QProgressBar {{
                        background: #eceff1;
                        border-radius: 8px;
                        height: 16px;
                        text-align: center;
                        border: none;
                        font-size: 10px;
                    }}
                    QProgressBar::chunk {{
                        background: #4a90e2;
                        border-radius: 8px;
                    }}
                    QFrame {{
                        border: none;
                    }}
                    QGroupBox {{
                        font-weight: bold;
                        border: 1px solid #dcdcdc;
                        border-radius: 8px;
                        margin-top: 1ex;
                        padding-top: 10px;
                    }}
                    QGroupBox::title {{
                        subline: 0;
                        left: 10px;
                        padding: 0 5px 0 5px;
                    }}
                    QDialog {{
                        background: #ffffff;
                    }}
                """,
                "border_color": "#4a90e2"  # Цвет рамки для темы
            },
            "dark": {
                "qss": f"""
                    QMainWindow {{
                        background: #1f2d3a;
                        border: 2px solid #3498db; /* Рамка главного окна */
                    }}
                    QWidget {{
                        background: #1f2d3a;
                        color: #ecf0f1;
                        font-family: {base_font};
                    }}
                    QMainWindow {{
                        background: #1f2d3a;
                    }}
                    QFrame#panel {{
                        background: #2b3b47;
                        border-radius: 12px;
                        border: 1px solid #30424f;
                    }}
                    QPushButton {{
                        background: #3498db;
                        color: white;
                        border: none;
                        padding: 10px 16px;
                        border-radius: 8px;
                        font-weight: 500;
                        font-size: 11px;
                    }}
                    QPushButton:hover {{
                        background: #2980b9;
                    }}
                    QPushButton:pressed {{
                        background: #2573a7;
                    }}
                    QPushButton#accent {{
                        background: #27ae60;
                    }}
                    QPushButton#accent:hover {{
                        background: #229954;
                    }}
                    QPushButton#danger {{
                        background: #e74c3c;
                    }}
                    QPushButton#danger:hover {{
                        background: #c0392b;
                    }}
                    QPushButton#theme {{
                        background: #9b59b6;
                    }}
                    QPushButton#theme:hover {{
                        background: #8e44ad;
                    }}
                    QSlider::groove:horizontal {{
                        background: #2e3a46;
                        height: 8px;
                        border-radius: 4px;
                    }}
                    QSlider::handle:horizontal {{
                        background: #3498db;
                        width: 16px;
                        border-radius: 8px;
                        margin: -4px 0;
                    }}
                    QSlider::handle:horizontal:hover {{
                        background: #2980b9;
                    }}
                    QLabel#title {{
                        font-size: 20px;
                        font-weight: 700;
                    }}
                    QLabel#subtitle {{
                        color: #bdc3c7;
                        font-size: 13px;
                    }}
                    QProgressBar {{
                        background: #2a3a45;
                        border-radius: 8px;
                        height: 16px;
                        text-align: center;
                        border: none;
                        font-size: 10px;
                    }}
                    QProgressBar::chunk {{
                        background: #27ae60;
                        border-radius: 8px;
                    }}
                    QFrame {{
                        border: none;
                    }}
                    QGroupBox {{
                        font-weight: bold;
                        border: 1px solid #3a4b5c;
                        border-radius: 8px;
                        margin-top: 1ex;
                        padding-top: 10px;
                    }}
                    QGroupBox::title {{
                        subline: 0;
                        left: 10px;
                        padding: 0 5px 0 5px;
                    }}
                    QDialog {{
                        background: #2b3b47;
                    }}
                """,
                "border_color": "#3498db"  # Цвет рамки для темы
            },
            "blue": {
                "qss": f"""
                    QMainWindow {{
                        background: #0f2b3d;
                        border: 2px solid #4abdac; /* Рамка главного окна */
                    }}
                    QWidget {{
                        background: #0f2b3d;
                        color: #e6f2f7;
                        font-family: {base_font};
                    }}
                    QMainWindow {{
                        background: #0f2b3d;
                    }}
                    QFrame#panel {{
                        background: #153847;
                        border-radius: 12px;
                        border: 1px solid #1e4054;
                    }}
                    QPushButton {{
                        background: #4abdac;
                        color: #01303a;
                        border: none;
                        padding: 10px 16px;
                        border-radius: 8px;
                        font-weight: 500;
                        font-size: 11px;
                    }}
                    QPushButton:hover {{
                        background: #3d9d9c;
                    }}
                    QPushButton:pressed {{
                        background: #358a89;
                    }}
                    QPushButton#accent {{
                        background: #4ecdc4;
                        color: #01303a;
                    }}
                    QPushButton#accent:hover {{
                        background: #3db9b0;
                    }}
                    QPushButton#danger {{
                        background: #ff6b6b;
                        color: #01303a;
                    }}
                    QPushButton#danger:hover {{
                        background: #e55a5a;
                    }}
                    QPushButton#theme {{
                        background: #9b59b6;
                        color: white;
                    }}
                    QPushButton#theme:hover {{
                        background: #8e44ad;
                    }}
                    QSlider::groove:horizontal {{
                        background: #123744;
                        height: 8px;
                        border-radius: 4px;
                    }}
                    QSlider::handle:horizontal {{
                        background: #4abdac;
                        width: 16px;
                        border-radius: 8px;
                        margin: -4px 0;
                    }}
                    QSlider::handle:horizontal:hover {{
                        background: #3d9d9c;
                    }}
                    QLabel#title {{
                        font-size: 20px;
                        font-weight: 700;
                    }}
                    QLabel#subtitle {{
                        color: #d4e6f1;
                        font-size: 13px;
                    }}
                    QProgressBar {{
                        background: #163345;
                        border-radius: 8px;
                        height: 16px;
                        text-align: center;
                        border: none;
                        font-size: 10px;
                    }}
                    QProgressBar::chunk {{
                        background: #4abdac;
                        border-radius: 8px;
                    }}
                    QFrame {{
                        border: none;
                    }}
                    QGroupBox {{
                        font-weight: bold;
                        border: 1px solid #1e4054;
                        border-radius: 8px;
                        margin-top: 1ex;
                        padding-top: 10px;
                    }}
                    QGroupBox::title {{
                        subline: 0;
                        left: 10px;
                        padding: 0 5px 0 5px;
                    }}
                    QDialog {{
                        background: #153847;
                    }}
                """,
                "border_color": "#4abdac"  # Цвет рамки для темы
            }
        }

    def _build_ui(self):
        # Используем UI структуру из первого файла
        central = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Header
        header = QHBoxLayout()
        header.setSpacing(12)
        title_container = QVBoxLayout()
        title_container.setSpacing(4)
        title_label = QLabel("Распознавание цифр")
        title_label.setObjectName("title")
        subtitle = QLabel("Нарисуйте цифру в поле ниже")
        subtitle.setObjectName("subtitle")
        title_container.addWidget(title_label)
        title_container.addWidget(subtitle)
        header.addLayout(title_container)
        header.addStretch()
        btn_theme = QPushButton("Тема")
        btn_theme.setObjectName("theme")
        btn_theme.setFixedWidth(90)
        btn_theme.clicked.connect(self._cycle_theme)
        btn_full = QPushButton("Полный экран")
        btn_full.setFixedWidth(130)
        btn_full.clicked.connect(self._toggle_fullscreen)
        header.addWidget(btn_theme)
        header.addWidget(btn_full)
        main_layout.addLayout(header)

        # Panel
        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(16, 16, 16, 16)
        panel_layout.setSpacing(15)

        # Canvas group
        canvas_group = QGroupBox("Холст для рисования")
        canvas_layout = QHBoxLayout()
        canvas_layout.setSpacing(20)
        self.drawing = DrawingWidget(size=340, brush=16)
        canvas_layout.addWidget(self.drawing)

        # Controls group
        controls_group = QGroupBox("Управление")
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(12)

        # Brush size control
        brush_container = QVBoxLayout()
        brush_container.setSpacing(6)
        brush_label = QLabel("Размер кисти")
        brush_label.setAlignment(Qt.AlignCenter)
        self.slider_brush = QSlider(Qt.Horizontal)
        self.slider_brush.setMinimum(2)
        self.slider_brush.setMaximum(40)
        self.slider_brush.setValue(16)
        self.slider_brush.setFixedWidth(180)
        self.slider_brush.valueChanged.connect(lambda v: self.drawing.set_brush(v))

        # Brush size display
        self.brush_size_label = QLabel("16")
        self.brush_size_label.setAlignment(Qt.AlignCenter)
        self.brush_size_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        brush_container.addWidget(brush_label)
        brush_container.addWidget(self.slider_brush)
        brush_container.addWidget(self.brush_size_label)
        controls_layout.addLayout(brush_container)

        # Action buttons
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(8)
        btn_predict = QPushButton("Распознать (Enter/Пробел)")
        btn_predict.clicked.connect(self._predict)
        btn_predict.setObjectName("accent")
        btn_preview = QPushButton("Предпросмотр")
        btn_preview.clicked.connect(self._show_preview)
        btn_probs = QPushButton("Вероятности")
        btn_probs.clicked.connect(self._show_probabilities)
        btn_clear = QPushButton("Очистить (Ctrl+C)")
        btn_clear.setObjectName("danger")
        btn_clear.clicked.connect(self._clear_canvas)
        buttons_layout.addWidget(btn_predict)
        buttons_layout.addWidget(btn_preview)
        buttons_layout.addWidget(btn_probs)
        buttons_layout.addWidget(btn_clear)
        controls_layout.addLayout(buttons_layout)
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        canvas_layout.addWidget(controls_group)
        canvas_group.setLayout(canvas_layout)
        panel_layout.addWidget(canvas_group)

        # Result group
        result_group = QGroupBox("Результаты")
        result_layout = QVBoxLayout()
        result_layout.setSpacing(10)
        self.result_label = QLabel("Готов к распознаванию...")
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.result_label.setFont(font)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFixedHeight(50)
        self.result_label.setStyleSheet("color: #3498db;")
        self.result_opacity = QGraphicsOpacityEffect()
        self.result_label.setGraphicsEffect(self.result_opacity)
        self.result_opacity.setOpacity(1.0)

        self.details_label = QLabel("")
        self.details_label.setAlignment(Qt.AlignCenter)
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet("font-size: 12px;")

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFixedHeight(20)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #34495e;
                border-radius: 10px;
                text-align: center;
                background: #2c3e50;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #27ae60, stop:1 #2ecc71);
                border-radius: 9px;
            }
        """)

        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.details_label)
        result_layout.addWidget(self.confidence_bar)
        result_group.setLayout(result_layout)
        panel_layout.addWidget(result_group)
        panel.setLayout(panel_layout)
        main_layout.addWidget(panel)

        # Status bar with tips
        tips = QLabel(
            "Советы: Рисуйте четкие цифры по центру. Ctrl+R: Распознать | Ctrl+T: Сменить тему | F11: Полный экран")
        tips.setObjectName("subtitle")
        tips.setWordWrap(True)
        tips.setAlignment(Qt.AlignCenter)
        tips.setStyleSheet(
            "padding: 10px; background: rgba(52, 152, 219, 0.1); border-radius: 6px; border: 1px solid #3498db;")
        main_layout.addWidget(tips)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # Keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self._clear_canvas)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._predict)
        QShortcut(QKeySequence("Ctrl+P"), self, activated=self._show_preview)
        QShortcut(QKeySequence("Ctrl+T"), self, activated=self._cycle_theme)
        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key_Return), self, activated=self._predict)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._predict)

        # Update brush size label when slider changes
        self.slider_brush.valueChanged.connect(self._update_brush_size_label)
        self.last_prediction: Optional[np.ndarray] = None
        self.busy_progress = QProgressBar()
        self.busy_progress.setRange(0, 0)
        self.busy_progress.setVisible(False)
        self.busy_progress.setFixedHeight(16)
        main_layout.addWidget(self.busy_progress)

    def _update_brush_size_label(self, value):
        self.brush_size_label.setText(str(value))

    def apply_theme(self, theme_name: str):
        # Используем логику из второго файла, но стили из первого
        theme_data = self.themes.get(theme_name, {})
        qss = theme_data.get("qss", "")
        self.setStyleSheet(qss)
        self.current_theme = theme_name

    def apply_theme_animated(self, theme_name: str, duration: int = 500):
        """Плавная смена темы: показываем градиентный overlay (от текущ border -> новый border),
           затем применяем QSS и уходим."""
        # Используем улучшенную анимацию из второго файла
        if theme_name == self.current_theme:
            return

        # Если уже анимация идёт, не стартуем новую
        if self._theme_animation_in is not None and self._theme_animation_in.state() == QPropertyAnimation.Running:
            return

        start_color = self.themes.get(self.current_theme, {}).get("border_color", "#3498db")
        end_color = self.themes.get(theme_name, {}).get("border_color", "#3498db")

        # Создаём overlay
        overlay = ThemeTransitionOverlay(self, start_color, end_color)
        overlay.raise_()
        self._theme_overlay = overlay

        # Animate in
        anim_in = QPropertyAnimation(overlay.graphicsEffect(), b"opacity", self)
        anim_in.setStartValue(0.0)
        anim_in.setEndValue(1.0)
        anim_in.setDuration(duration // 2)
        anim_in.setEasingCurve(QEasingCurve.InOutCubic)
        self._theme_animation_in = anim_in

        def on_in_finished():
            # Применяем тему ровно в момент покрытия экрана
            self.apply_theme(theme_name)
            # Анимируем исчезновение overlay
            anim_out = QPropertyAnimation(overlay.graphicsEffect(), b"opacity", self)
            anim_out.setStartValue(1.0)
            anim_out.setEndValue(0.0)
            anim_out.setDuration(duration // 2)
            anim_out.setEasingCurve(QEasingCurve.InOutCubic)
            self._theme_animation_out = anim_out

            def on_out_finished():
                try:
                    overlay.hide()
                    overlay.setParent(None)
                except Exception:
                    pass
                self._theme_overlay = None
                self._theme_animation_in = None
                self._theme_animation_out = None

            anim_out.finished.connect(on_out_finished)
            anim_out.start()

        anim_in.finished.connect(on_in_finished)
        anim_in.start()

    def _cycle_theme(self):
        # Используем улучшенную анимацию из второго файла
        keys = list(self.themes.keys())
        idx = keys.index(self.current_theme)
        idx = (idx + 1) % len(keys)
        self.apply_theme_animated(keys[idx])

    def _toggle_fullscreen(self):
        # Используем логику из второго файла
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _clear_canvas(self):
        # Используем логику из первого файла
        self.drawing.clear()
        self.result_label.setText("Холст очищен. Нарисуйте цифру.")
        self.result_label.setStyleSheet("color: #3498db;")
        self.details_label.setText("")
        self.confidence_bar.setValue(0)
        self.last_prediction = None

    @staticmethod
    def get_best_shift(img: np.ndarray) -> Tuple[int, int]:
        # Используем улучшенную логику из второго файла
        cy, cx = ndimage.center_of_mass(img)
        rows, cols = img.shape
        shiftx = int(np.round(cols / 2.0 - cx))
        shifty = int(np.round(rows / 2.0 - cy))
        return shiftx, shifty

    @staticmethod
    def shift(img: np.ndarray, sx: int, sy: int) -> np.ndarray:
        # Используем улучшенную логику из второго файла
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows))
        return shifted

    def preprocess_image(self) -> np.ndarray:
        # Используем улучшенную логику из второго файла
        pil = self.drawing.get_pil_image()
        img_resized = pil.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img_resized).astype(np.uint8)
        img_array = 255 - img_array
        img_array = img_array / 255.0
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
        bbox = ImageOps.invert(img_pil).getbbox()
        if bbox:
            img_cropped = img_pil.crop(bbox)
            img_pil = ImageOps.pad(img_cropped, (28, 28), color=0)
            img_array = np.array(img_pil) / 255.0
        shiftx, shifty = self.get_best_shift(img_array)
        img_array = self.shift(img_array, shiftx, shifty)
        img_array = img_array.reshape(1, 28, 28, 1).astype(np.float32)
        return img_array

    def _predict(self):
        # Используем логику из второго файла
        try:
            img_array = self.preprocess_image()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при подготовке изображения:\n{e}")
            return
        if self.model_session is None:
            QMessageBox.critical(self, "Ошибка", "Модель не загружена.")
            return
        self.busy_progress.setVisible(True)
        self.result_label.setText("Анализ...")
        self.result_label.setStyleSheet("color: #f39c12;")
        self.details_label.setText("")
        self.repaint()
        self.worker = InferenceWorker(self.model_session, img_array)
        self.worker.result_ready.connect(self._on_prediction)
        self.worker.error.connect(self._on_inference_error)
        self.worker.start()

    def _on_inference_error(self, err: str):
        # Используем логику из второго файла
        self.busy_progress.setVisible(False)
        QMessageBox.critical(self, "Ошибка inference", err)
        self.result_label.setText("Ошибка")
        self.result_label.setStyleSheet("color: #e74c3c;")
        self.details_label.setText(err)

    def _animate_result_appearance(self):
        # Используем логику из второго файла
        anim = QPropertyAnimation(self.result_opacity, b"opacity", self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(280)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._result_anim = anim  # сохранить ссылку

    def _animate_confidence_bar(self, target_value: int):
        # Используем логику из второго файла
        anim = QPropertyAnimation(self.confidence_bar, b"value", self)
        anim.setStartValue(self.confidence_bar.value())
        anim.setEndValue(target_value)
        anim.setDuration(420)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._confidence_anim = anim  # сохранить ссылку

    def _on_prediction(self, prediction: np.ndarray):
        # Используем улучшенную логику из второго файла
        self.busy_progress.setVisible(False)
        probs = np.asarray(prediction).astype(np.float32).flatten()
        if probs.sum() > 0:
            probs = probs / probs.sum()
        pred_digit = int(np.argmax(probs))
        confidence = float(np.max(probs))
        self.last_prediction = probs

        # Отображаем в одном лейбле цифру и уверенность
        self.result_label.setText(f"Цифра: {pred_digit}   (Уверенность: {confidence:.1%})")
        self.result_label.setStyleSheet("color: #27ae60; font-size: 20px; font-weight: bold;")

        # Топ-альтернативы
        sorted_idx = np.argsort(probs)[::-1]
        top3 = sorted_idx[:3]
        alt_list = []
        for idx in top3:
            if idx != pred_digit and probs[idx] > 0.005:
                alt_list.append(f"{idx}: {probs[idx]:.1%}")
        details_text = ""
        if alt_list:
            details_text = "Альтернативы: " + "  |  ".join(alt_list)

        # Дополнительно показываем краткую уверенность
        details_text = f"Уверенность: {confidence:.1%}" + ((" | " + details_text) if details_text else "")
        self.details_label.setText(details_text)
        self._animate_result_appearance()
        self._animate_confidence_bar(int(confidence * 100))

    def _show_probabilities(self):
        # Используем логику из первого файла
        if self.last_prediction is None:
            QMessageBox.information(self, "Информация", "Сначала выполните распознавание!")
            return
        dlg = ProbabilityDialog(self.last_prediction, parent=self)
        dlg.exec()

    def _show_preview(self):
        # Используем логику из первого файла
        try:
            arr = self.preprocess_image()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при предобработке: {e}")
            return
        dlg = PreviewDialog(arr, parent=self)
        dlg.exec()

# ---------- Main ----------
def main():
    app = QApplication(sys.argv)
    try:
        w = ModernDigitRecognizerMain()
    except Exception as e:
        QMessageBox.critical(None, "Ошибка запуска", str(e))
        raise
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
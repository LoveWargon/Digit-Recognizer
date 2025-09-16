import sys
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import onnxruntime as ort
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os

def resource_path(relative_path):
    """Получение пути к ресурсу, работает для dev и для PyInstaller"""
    try:
        # PyInstaller создает временную папку _MEIPASS
        base_path = sys._MEIPASS
        # Для PyInstaller: файлы добавлены с префиксом 'src'
        # Поэтому добавляем 'src' к относительному пути для поиска внутри _MEIPASS
        relative_path = os.path.join('src', relative_path)
    except Exception:
        # Для разработки: файлы ищем относительно текущей директории проекта
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)



class ModernDigitRecognizer:
    def __init__(self):

        model_loaded = False

        model_paths = [
            resource_path("resources/models/improved_digit_recognition_model.onnx"),
            "resources/models/improved_digit_recognition_model.onnx",
            "improved_digit_recognition_model.onnx",
        ]
        for model_path in model_paths:
            try:
                if os.path.exists(model_path):
                    print(f"Попытка загрузки ONNX модели: {model_path}")

                    self.model = ort.InferenceSession(model_path, providers=[
                        'CPUExecutionProvider'])
                    print(f" ONNX модель загружена: {model_path}")
                    model_loaded = True
                    break
                else:
                    print(f" ONNX модель не найдена: {model_path}")
            except Exception as e:
                print(f" Ошибка загрузки ONNX модели {model_path}: {e}")
                continue
        if not model_loaded:
            raise FileNotFoundError("Не удалось загрузить ONNX модель нейросети. Проверьте наличие файлов моделей.")

        # Темы
        self.themes = {
            'light': {
                'bg': '#f0f0f0',
                'fg': '#2c3e50',
                'canvas_bg': 'white',
                'button_bg': '#4a90e2',
                'clear_bg': '#ff6b6b',
                'show_bg': '#50c878',
                'text_secondary': '#7f8c8d',
                'frame_bg': '#f8f9fa',
                'border_color': '#2c3e50',
                'result_text': '#2c3e50',
                'window_controls': '#e0e0e0',
                'window_border': '#cccccc'
            },
            'dark': {
                'bg': '#2c3e50',
                'fg': '#ecf0f1',
                'canvas_bg': '#34495e',
                'button_bg': '#3498db',
                'clear_bg': '#e74c3c',
                'show_bg': '#27ae60',
                'text_secondary': '#bdc3c7',
                'frame_bg': '#34495e',
                'border_color': '#ecf0f1',
                'result_text': '#ecf0f1',
                'window_controls': '#3c4e60',
                'window_border': '#1a2530'
            },
            'blue': {
                'bg': '#1e3d59',
                'fg': '#ffffff',
                'canvas_bg': '#2a4d69',
                'button_bg': '#4abdac',
                'clear_bg': '#ff6b6b',
                'show_bg': '#4ecdc4',
                'text_secondary': '#d4e6f1',
                'frame_bg': '#2a4d69',
                'border_color': '#4abdac',
                'result_text': '#ffffff',
                'window_controls': '#2d5d79',
                'window_border': '#152a3d'
            }
        }

        self.current_theme = 'light'
        self.theme_transition_active = False
        self.is_fullscreen = False

        # Создаем главное окно
        self.root = tk.Tk()
        self.root.title("🧠 AI Распознавание рукописных цифр")
        self.root.geometry("450x700")
        self.root.configure(bg=self.themes[self.current_theme]['bg'])
        self.root.minsize(400, 600)

        # Теперь можно создавать переменные Tkinter
        self.brush_size = tk.DoubleVar(value=12)

        # Создаем изображение для рисования
        self.image = Image.new("L", (280, 280), 255)  # белый фон
        self.draw = ImageDraw.Draw(self.image)

        # Хранение предыдущей точки для рисования линии
        self.last_x, self.last_y = None, None

        # Оптимизация
        self.debounce_timer = None
        self.tooltip = None

        # История предсказаний
        self.prediction_history = []

        # Создаем интерфейс
        self.create_ui()

        # Подключаем события
        self.bind_events()

        # Применяем начальную тему
        self.apply_theme()

    def create_ui(self):
        """Создание пользовательского интерфейса"""
        # Заголовок с переключателем темы
        header_frame = tk.Frame(self.root, bg=self.themes[self.current_theme]['bg'])
        header_frame.pack(pady=10, padx=20, fill='x')

        title_frame = tk.Frame(header_frame, bg=self.themes[self.current_theme]['bg'])
        title_frame.pack(side=tk.LEFT)

        title_label = tk.Label(
            title_frame,
            text="🧠 Распознавание цифр",
            font=("Segoe UI", 16, "bold"),
            bg=self.themes[self.current_theme]['bg'],
            fg=self.themes[self.current_theme]['fg']
        )
        title_label.pack(anchor='w')

        subtitle_label = tk.Label(
            title_frame,
            text="Нарисуйте цифру в поле ниже",
            font=("Segoe UI", 10),
            bg=self.themes[self.current_theme]['bg'],
            fg=self.themes[self.current_theme]['text_secondary']
        )
        subtitle_label.pack(anchor='w')

        # Переключатель темы и полноэкранный режим
        control_frame = tk.Frame(header_frame, bg=self.themes[self.current_theme]['bg'])
        control_frame.pack(side=tk.RIGHT)

        # Кнопка полноэкранного режима
        fullscreen_label = tk.Label(
            control_frame,
            text="⬜",
            font=("Segoe UI", 12),
            bg=self.themes[self.current_theme]['bg'],
            fg=self.themes[self.current_theme]['fg'],
            cursor="hand2"
        )
        fullscreen_label.pack(side=tk.LEFT, padx=5)
        fullscreen_label.bind("<Button-1>", self.toggle_fullscreen)
        self.fullscreen_label = fullscreen_label
        self.create_tooltip(fullscreen_label, "Полноэкранный режим (F11)")

        # Переключатель темы
        theme_label = tk.Label(
            control_frame,
            text="🌙",
            font=("Segoe UI", 14),
            bg=self.themes[self.current_theme]['bg'],
            fg=self.themes[self.current_theme]['fg'],
            cursor="hand2"
        )
        theme_label.pack(side=tk.LEFT)
        theme_label.bind("<Button-1>", self.toggle_theme)
        self.theme_label = theme_label
        self.create_tooltip(theme_label, "Переключить тему (Ctrl+T)")

        # Слайдер размера кисти
        brush_frame = tk.Frame(self.root, bg=self.themes[self.current_theme]['bg'])
        brush_frame.pack(pady=5, padx=20, fill='x')

        brush_label = tk.Label(brush_frame, text="🎨 Размер кисти:",
                               font=("Segoe UI", 9),
                               bg=self.themes[self.current_theme]['bg'],
                               fg=self.themes[self.current_theme]['fg'])
        brush_label.pack(side=tk.LEFT)
        self.create_tooltip(brush_label, "Используйте колесо мыши или стрелки ↑↓")

        brush_slider = tk.Scale(
            brush_frame,
            from_=2,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self.brush_size,
            bg=self.themes[self.current_theme]['bg'],
            fg=self.themes[self.current_theme]['fg'],
            highlightthickness=0,
            length=100,
            font=("Segoe UI", 8),
            cursor="hand2"
        )
        brush_slider.pack(side=tk.LEFT, padx=10)

        # Холст для рисования с рамкой
        canvas_frame = tk.Frame(
            self.root,
            bg=self.themes[self.current_theme]['border_color'],
            bd=2,
            relief='solid'
        )
        canvas_frame.pack(pady=10, padx=20)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=280,
            height=280,
            bg=self.themes[self.current_theme]['canvas_bg'],
            cursor="cross",
            highlightthickness=0
        )
        self.canvas.pack(padx=2, pady=2)

        # Панель кнопок
        button_frame = tk.Frame(self.root, bg=self.themes[self.current_theme]['bg'])
        button_frame.pack(pady=15)

        # Кнопки с улучшенным дизайном
        self.clear_button = self.create_feedback_button(
            button_frame,
            "🧹 Очистить",
            self.clear_canvas,
            self.themes[self.current_theme]['clear_bg']
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.clear_button, "Очистить холст (Ctrl+C или Delete)")

        self.predict_button = self.create_feedback_button(
            button_frame,
            "🔍 Распознать",
            self.predict_digit,
            self.themes[self.current_theme]['button_bg']
        )
        self.predict_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.predict_button, "Распознать цифру (Ctrl+R или Enter)")

        self.show_image_button = self.create_feedback_button(
            button_frame,
            "🖼️ Показать",
            self.show_processed_image,
            self.themes[self.current_theme]['show_bg']
        )
        self.show_image_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.show_image_button, "Показать обработанное изображение (Ctrl+P)")

        # Кнопка показа вероятностей
        self.probability_button = self.create_feedback_button(
            button_frame,
            "📈 Вероятности",
            self.show_probability_chart,
            self.themes[self.current_theme]['button_bg']
        )
        self.probability_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.probability_button, "Показать вероятности предсказаний")

        # Панель результатов с деталями
        self.result_frame = tk.Frame(self.root, bg=self.themes[self.current_theme]['bg'])
        self.result_frame.pack(pady=10, padx=20, fill='x')

        # Основной результат
        self.result_label = tk.Label(
            self.result_frame,
            text="Готов к распознаванию...",
            font=("Segoe UI", 14, "bold"),
            bg=self.themes[self.current_theme]['bg'],
            fg=self.themes[self.current_theme]['result_text']
        )
        self.result_label.pack(pady=5)

        # Детали результата
        self.details_label = tk.Label(
            self.result_frame,
            text="",
            font=("Segoe UI", 10),
            bg=self.themes[self.current_theme]['bg'],
            fg=self.themes[self.current_theme]['text_secondary']
        )
        self.details_label.pack()

        # Индикатор уверенности
        self.confidence_frame = tk.Frame(self.result_frame, bg=self.themes[self.current_theme]['bg'])
        self.confidence_frame.pack(pady=5)

        self.confidence_label = tk.Label(
            self.confidence_frame,
            text="",
            font=("Segoe UI", 9),
            bg=self.themes[self.current_theme]['bg'],
            fg=self.themes[self.current_theme]['fg']
        )
        self.confidence_label.pack()

        self.confidence_canvas = tk.Canvas(
            self.confidence_frame,
            width=200,
            height=15,
            bg=self.themes[self.current_theme]['frame_bg'],
            highlightthickness=0
        )
        self.confidence_canvas.pack()

        # Прогресс бар с текстом
        progress_container = tk.Frame(self.result_frame, bg=self.themes[self.current_theme]['bg'])
        progress_container.pack(pady=5)

        self.progress_text = tk.Label(
            progress_container,
            text="",
            font=("Segoe UI", 9),
            bg=self.themes[self.current_theme]['bg'],
            fg=self.themes[self.current_theme]['text_secondary']
        )
        self.progress_text.pack()

        self.progress = ttk.Progressbar(
            progress_container,
            mode='indeterminate',
            length=200
        )
        self.progress.pack()

        # Информационная панель
        info_frame = tk.Frame(self.root, bg=self.themes[self.current_theme]['bg'])
        info_frame.pack(pady=10, padx=20, fill='x')

        info_label = tk.Label(
            info_frame,
            text="💡 Совет: Рисуйте четкие цифры по центру",
            font=("Segoe UI", 9),
            bg=self.themes[self.current_theme]['bg'],
            fg=self.themes[self.current_theme]['text_secondary']
        )
        info_label.pack()

        # Горячие клавиши информация
        hotkeys_label = tk.Label(
            info_frame,
            text="⌨️ Ctrl+C: Очистить | Ctrl+R: Распознать | Ctrl+P: Показать | Ctrl+T: Тема | F11: Полный экран",
            font=("Segoe UI", 8),
            bg=self.themes[self.current_theme]['bg'],
            fg=self.themes[self.current_theme]['text_secondary']
        )
        hotkeys_label.pack(pady=(5, 0))

    def create_feedback_button(self, parent, text, command, bg_color):
        """Кнопка с визуальной обратной связью"""

        def on_enter(e):
            button.config(bg=self.lighten_color(bg_color))

        def on_leave(e):
            button.config(bg=bg_color)

        def on_click(e):
            button.config(relief='sunken')
            self.root.after(100, lambda: button.config(relief='raised'))
            command()

        button = tk.Button(
            parent,
            text=text,
            font=("Segoe UI", 10, "bold"),
            width=12,
            relief='raised',
            bd=2,
            fg='white',
            bg=bg_color,
            activebackground=self.darken_color(bg_color),
            activeforeground='white',
            padx=10,
            pady=5,
            cursor='hand2'
        )

        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
        button.bind("<Button-1>", on_click)

        return button

    def lighten_color(self, color):
        """Осветление цвета"""
        color_map = {
            '#4a90e2': '#6bb3ff',
            '#ff6b6b': '#ff8e8e',
            '#50c878': '#73ff9b',
            '#3498db': '#5dade2',
            '#e74c3c': '#ec7063',
            '#27ae60': '#2ecc71',
            '#4abdac': '#5fd3c2'
        }
        return color_map.get(color, color)

    def toggle_theme(self, event=None):
        """Переключение темы с анимацией"""
        if self.theme_transition_active:
            return

        themes_list = list(self.themes.keys())
        current_index = themes_list.index(self.current_theme)
        next_index = (current_index + 1) % len(themes_list)
        next_theme = themes_list[next_index]

        self.animate_theme_transition(self.current_theme, next_theme)

    def toggle_fullscreen(self, event=None):
        """Переключение полноэкранного режима"""
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes('-fullscreen', self.is_fullscreen)

        # Обновляем иконку
        if self.is_fullscreen:
            self.fullscreen_label.config(text="⧉")
            self.fullscreen_label.config(text="⧉")
        else:
            self.fullscreen_label.config(text="⬜")

        # Обновляем подсказку
        if hasattr(self, 'tooltip') and self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def animate_theme_transition(self, from_theme, to_theme, step=0):
        """Анимация перехода между темами"""
        self.theme_transition_active = True

        if step > 20:  # Завершаем анимацию
            self.current_theme = to_theme
            self.apply_theme()
            self.theme_transition_active = False
            return

        # Интерполируем цвета
        progress = step / 20.0

        # Обновляем основное окно
        bg_color = self.interpolate_color(
            self.themes[from_theme]['bg'],
            self.themes[to_theme]['bg'],
            progress
        )
        self.root.configure(bg=bg_color)

        # Обновляем все фреймы и виджеты
        self.animate_all_widgets(from_theme, to_theme, progress)

        # Обновляем иконки
        icons = {'light': '🌙', 'dark': '☀️', 'blue': '🌊'}
        icon_color = self.interpolate_color(
            self.themes[from_theme]['fg'],
            self.themes[to_theme]['fg'],
            progress
        )
        self.theme_label.configure(
            bg=bg_color,
            fg=icon_color
        )
        self.fullscreen_label.configure(
            bg=bg_color,
            fg=icon_color
        )

        # Продолжаем анимацию
        self.root.after(20, lambda: self.animate_theme_transition(from_theme, to_theme, step + 1))

    def animate_all_widgets(self, from_theme, to_theme, progress):
        """Полная анимация всех виджетов"""
        for widget in self.root.winfo_children():
            self.animate_widget(widget, from_theme, to_theme, progress)

    def animate_widget(self, widget, from_theme, to_theme, progress):
        """Анимация отдельного виджета"""
        if isinstance(widget, tk.Frame):
            # Анимируем фрейм
            bg_color = self.interpolate_color(
                self.themes[from_theme]['bg'],
                self.themes[to_theme]['bg'],
                progress
            )
            widget.configure(bg=bg_color)

            # Анимируем дочерние элементы
            for child in widget.winfo_children():
                self.animate_widget(child, from_theme, to_theme, progress)

        elif isinstance(widget, tk.Label):
            # Анимируем надписи
            self.animate_label(widget, from_theme, to_theme, progress)

        elif isinstance(widget, tk.Button):
            # Анимируем кнопки
            self.animate_button(widget, from_theme, to_theme, progress)

        elif isinstance(widget, tk.Scale):
            # Анимируем слайдеры
            self.animate_scale(widget, from_theme, to_theme, progress)

        elif isinstance(widget, tk.Canvas):
            # Анимируем холсты
            self.animate_canvas(widget, from_theme, to_theme, progress)

    def animate_label(self, label, from_theme, to_theme, progress):
        """Анимация надписи"""
        text = label.cget('text')
        bg_color = self.interpolate_color(
            self.themes[from_theme]['bg'],
            self.themes[to_theme]['bg'],
            progress
        )

        if 'Распознавание цифр' in text:
            fg_color = self.interpolate_color(
                self.themes[from_theme]['fg'],
                self.themes[to_theme]['fg'],
                progress
            )
        elif any(keyword in text for keyword in ['Нарисуйте цифру', 'Совет:', '⌨️', 'Уровень уверенности:', '📊']):
            fg_color = self.interpolate_color(
                self.themes[from_theme]['text_secondary'],
                self.themes[to_theme]['text_secondary'],
                progress
            )
        else:
            fg_color = self.interpolate_color(
                self.themes[from_theme]['fg'],
                self.themes[to_theme]['fg'],
                progress
            )

        label.configure(bg=bg_color, fg=fg_color)

    def animate_button(self, button, from_theme, to_theme, progress):
        """Анимация кнопки"""
        # Анимируем фон родительского фрейма
        parent_bg = self.interpolate_color(
            self.themes[from_theme]['bg'],
            self.themes[to_theme]['bg'],
            progress
        )
        button.configure(bg=parent_bg)

        # Для текста кнопки используем цвет фона родительского фрейма
        button.configure(fg='white')  # Текст остается белым для лучшей читаемости

    def animate_scale(self, scale, from_theme, to_theme, progress):
        """Анимация слайдера"""
        bg_color = self.interpolate_color(
            self.themes[from_theme]['bg'],
            self.themes[to_theme]['bg'],
            progress
        )
        fg_color = self.interpolate_color(
            self.themes[from_theme]['fg'],
            self.themes[to_theme]['fg'],
            progress
        )
        trough_color = self.interpolate_color(
            self.themes[from_theme]['frame_bg'],
            self.themes[to_theme]['frame_bg'],
            progress
        )
        scale.configure(bg=bg_color, fg=fg_color, troughcolor=trough_color)

    def animate_canvas(self, canvas, from_theme, to_theme, progress):
        """Анимация холста"""
        if canvas == self.canvas:
            # Основной холст для рисования
            bg_color = self.interpolate_color(
                self.themes[from_theme]['canvas_bg'],
                self.themes[to_theme]['canvas_bg'],
                progress
            )
            canvas.configure(bg=bg_color)
        elif canvas == self.confidence_canvas:
            # Холст индикатора уверенности
            bg_color = self.interpolate_color(
                self.themes[from_theme]['frame_bg'],
                self.themes[to_theme]['frame_bg'],
                progress
            )
            canvas.configure(bg=bg_color)

    def interpolate_color(self, color1, color2, factor):
        """Интерполяция между двумя цветами"""
        if color1 == color2:
            return color1

        # Преобразуем цвета в RGB
        try:
            r1 = int(color1[1:3], 16)
            g1 = int(color1[3:5], 16)
            b1 = int(color1[5:7], 16)

            r2 = int(color2[1:3], 16)
            g2 = int(color2[3:5], 16)
            b2 = int(color2[5:7], 16)

            # Интерполируем
            r = int(r1 + (r2 - r1) * factor)
            g = int(g1 + (g2 - g1) * factor)
            b = int(b1 + (b2 - b1) * factor)

            return f"#{r:02x}{g:02x}{b:02x}"
        except:
            return color1  # В случае ошибки возвращаем исходный цвет

    def apply_theme(self):
        """Применение текущей темы ко всем элементам"""
        theme = self.themes[self.current_theme]

        # Обновляем основное окно
        self.root.configure(bg=theme['bg'])

        # Обновляем все фреймы и виджеты
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame):
                widget.configure(bg=theme['bg'])
                for child in widget.winfo_children():
                    self.update_widget_theme(child, theme)

        # Обновляем холст
        canvas_frame = self.canvas.master
        canvas_frame.configure(bg=theme['border_color'])
        self.canvas.configure(bg=theme['canvas_bg'])

        # Обновляем кнопки
        buttons_config = [
            (self.clear_button, theme['clear_bg']),
            (self.predict_button, theme['button_bg']),
            (self.show_image_button, theme['show_bg']),
            (self.probability_button, theme['button_bg'])
        ]

        for button, color in buttons_config:
            button.configure(
                bg=color,
                activebackground=self.darken_color(color)
            )

        # Обновляем панель результатов
        self.result_frame.configure(bg=theme['bg'])
        self.result_label.configure(
            bg=theme['bg'],
            fg=theme['result_text']
        )
        self.details_label.configure(
            bg=theme['bg'],
            fg=theme['text_secondary']
        )
        self.confidence_label.configure(
            bg=theme['bg'],
            fg=theme['fg']
        )
        self.confidence_canvas.configure(
            bg=theme['frame_bg']
        )
        self.progress_text.configure(
            bg=theme['bg'],
            fg=theme['text_secondary']
        )

        # Обновляем слайдер
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Scale):
                        child.configure(
                            bg=theme['bg'],
                            fg=theme['fg'],
                            troughcolor=theme['frame_bg']
                        )

        # Обновляем иконки
        icons = {'light': '🌙', 'dark': '☀️', 'blue': '🌊'}
        self.theme_label.configure(
            text=icons.get(self.current_theme, '🌙'),
            bg=theme['bg'],
            fg=theme['fg']
        )

        fullscreen_icons = {'normal': '⬜', 'fullscreen': '⧉'}
        self.fullscreen_label.configure(
            bg=theme['bg'],
            fg=theme['fg']
        )

    def update_widget_theme(self, widget, theme):
        """Рекурсивное обновление темы для виджета"""
        if isinstance(widget, tk.Label):
            if 'Распознавание цифр' in widget.cget('text'):
                widget.configure(bg=theme['bg'], fg=theme['fg'])
            elif any(text in widget.cget('text') for text in
                     ['Нарисуйте цифру', 'Совет:', '⌨️', 'Уровень уверенности:', '📊']):
                widget.configure(bg=theme['bg'], fg=theme['text_secondary'])
            else:
                widget.configure(bg=theme['bg'], fg=theme['fg'])
        elif isinstance(widget, tk.Frame):
            widget.configure(bg=theme['bg'])
            for subchild in widget.winfo_children():
                self.update_widget_theme(subchild, theme)
        elif isinstance(widget, tk.Scale):
            widget.configure(
                bg=theme['bg'],
                fg=theme['fg'],
                troughcolor=theme['frame_bg']
            )
        elif isinstance(widget, tk.Canvas):
            if widget == self.canvas:
                widget.configure(bg=theme['canvas_bg'])
            elif widget == self.confidence_canvas:
                widget.configure(bg=theme['frame_bg'])

    def darken_color(self, color):
        """Затемнение цвета для activebackground"""
        color_map = {
            '#ff6b6b': '#c0392b',  # clear_bg light
            '#4a90e2': '#2980b9',  # button_bg light
            '#50c878': '#27ae60',  # show_bg light
            '#e74c3c': '#c0392b',  # clear_bg dark
            '#3498db': '#2980b9',  # button_bg dark
            '#27ae60': '#229954',  # show_bg dark
            '#4abdac': '#3d9a8d'  # blue theme
        }
        return color_map.get(color, color)

    def bind_events(self):
        """Подключение событий мыши и клавиатуры"""
        self.canvas.bind("<B1-Motion>", self.draw_digit)
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_coords)
        self.canvas.bind("<Double-Button-1>", lambda e: self.clear_canvas())
        self.canvas.bind("<MouseWheel>", self.change_brush_size)

        # Горячие клавиши
        self.root.bind("<Control-c>", lambda e: self.clear_canvas())
        self.root.bind("<Control-r>", lambda e: self.predict_digit())
        self.root.bind("<Control-p>", lambda e: self.show_processed_image())
        self.root.bind("<Control-t>", lambda e: self.toggle_theme())
        self.root.bind("<Delete>", lambda e: self.clear_canvas())
        self.root.bind("<Return>", lambda e: self.predict_digit())
        self.root.bind("<space>", lambda e: self.predict_digit())
        self.root.bind("<Up>", lambda e: self.adjust_brush_size(1))
        self.root.bind("<Down>", lambda e: self.adjust_brush_size(-1))
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", lambda e: self.exit_fullscreen())

    def exit_fullscreen(self, event=None):
        """Выход из полноэкранного режима"""
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.root.attributes('-fullscreen', False)
            self.fullscreen_label.config(text="⬜")

    def change_brush_size(self, event):
        """Изменение размера кисти колесом мыши"""
        delta = 1 if event.delta > 0 else -1
        current = self.brush_size.get()
        new_size = max(2, min(20, current + delta))
        self.brush_size.set(new_size)

    def adjust_brush_size(self, delta):
        """Изменение размера кисти стрелками"""
        current = self.brush_size.get()
        new_size = max(2, min(20, current + delta))
        self.brush_size.set(new_size)

    def start_draw(self, event):
        """Начало рисования - устанавливаем первую точку"""
        self.last_x, self.last_y = event.x, event.y

    def draw_digit(self, event):
        """Рисование на холсте"""
        x, y = event.x, event.y

        if self.last_x and self.last_y:
            # Определяем цвет в зависимости от темы
            line_color = '#2c3e50' if self.current_theme == 'light' else '#ecf0f1'

            # Рисуем линию на холсте
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                fill=line_color,
                width=self.brush_size.get(),
                capstyle=tk.ROUND,
                smooth=True
            )
            # Рисуем линию в PIL изображении
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill="black",
                width=int(self.brush_size.get())
            )

        self.last_x, self.last_y = x, y

    def reset_last_coords(self, event=None):
        """Сброс координат"""
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        """Очистка холста"""
        self.canvas.delete("all")
        self.image.paste(255, (0, 0, 280, 280))
        self.draw.rectangle([0, 0, 280, 280], fill=255)
        self.result_label.config(text="Холст очищен. Нарисуйте цифру.")
        self.details_label.config(text="")
        self.confidence_label.config(text="")
        self.confidence_canvas.delete("all")
        self.result_label.config(fg=self.themes[self.current_theme]['text_secondary'])

    def animate_result(self, text, color, details="", confidence=None):
        """Плавное изменение текста результата"""

        def update_text(step=0):
            if step <= len(text):
                self.result_label.config(text=text[:step], fg=color)
                self.root.after(20, update_text, step + 1)
            else:
                if details:
                    self.details_label.config(text=details)
                if confidence is not None:
                    self.update_confidence_meter(confidence)

        self.result_label.config(text="")
        self.details_label.config(text="")
        self.confidence_label.config(text="")
        self.confidence_canvas.delete("all")
        update_text()

    def update_confidence_meter(self, confidence):
        """Обновление индикатора уверенности"""
        # Текст уверенности
        self.confidence_label.config(text=f"Уровень уверенности: {confidence:.1%}")

        # Визуальный индикатор
        self.confidence_canvas.delete("all")

        # Фон
        self.confidence_canvas.create_rectangle(0, 0, 200, 15,
                                                fill=self.themes[self.current_theme]['frame_bg'],
                                                outline="")

        # Заполнение
        fill_width = int(200 * confidence)
        color = self.get_confidence_color(confidence)

        if fill_width > 0:
            self.confidence_canvas.create_rectangle(0, 0, fill_width, 15,
                                                    fill=color, outline="")

    def get_confidence_color(self, confidence):
        """Цвет в зависимости от уровня уверенности"""
        if confidence > 0.8:
            return '#27ae60'  # Зеленый
        elif confidence > 0.5:
            return '#f39c12'  # Оранжевый
        else:
            return '#e74c3c'  # Красный

    def show_notification(self, message, duration=2000):
        """Показ всплывающего уведомления"""
        notification_frame = tk.Frame(
            self.root,
            bg='#3498db',
            relief='raised',
            bd=2
        )
        notification_frame.place(relx=0.5, rely=0.1, anchor='center')

        tk.Label(
            notification_frame,
            text=message,
            font=("Segoe UI", 10, "bold"),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=10
        ).pack()

        self.root.after(duration, notification_frame.destroy)

    def get_best_shift(self, img):
        """Нахождение центра масс для центрирования"""
        cy, cx = ndimage.center_of_mass(img)
        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)
        return shiftx, shifty

    def shift(self, img, sx, sy):
        """Сдвиг изображения"""
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows))
        return shifted

    def preprocess_image(self):
        """Предобработка изображения для модели"""

        img_resized = self.image.resize((28, 28))
        img_array = np.array(img_resized)

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


        img_array = img_array.reshape(1, 28, 28, 1)
        return img_array.astype(np.float32)

    def predict_digit(self):
        """Распознавание цифры с анимацией загрузки"""
        # Отменяем предыдущий таймер
        if self.debounce_timer:
            self.root.after_cancel(self.debounce_timer)

        # Задержка для предотвращения множественных вызовов
        self.debounce_timer = self.root.after(300, self._perform_prediction)

    def _perform_prediction(self):
        """Выполнение предсказания с ONNX моделью"""

        def predict_thread():
            try:
                self.progress.start()
                self.progress_text.config(text="🧠 Анализируем изображение...")
                theme = self.themes[self.current_theme]
                self.result_label.config(text="Анализ...", fg=theme['button_bg'])
                self.root.update()

                img_array = self.preprocess_image()

                input_name = self.model.get_inputs()[0].name

                prediction = self.model.run(None, {input_name: img_array})[0][
                    0]

                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)

                self.last_prediction = prediction

                confidence_color = self.get_confidence_color(confidence)
                result_text = f"🎯 Результат: {predicted_digit}"
                details_text = f"Уверенность: {confidence:.1%} | Альтернативы: "

                top_3_indices = np.argsort(prediction)[-3:][::-1]
                alternatives = [f"{i}: {prediction[i]:.1%}" for i in top_3_indices[1:] if prediction[i] > 0.01]
                details_text += " | ".join(alternatives) if alternatives else "нет"
                self.animate_result(result_text, confidence_color, details_text, confidence)
            except Exception as e:
                import traceback
                print(traceback.format_exc())  # Для отладки
                theme = self.themes[self.current_theme]
                self.animate_result(f"❌ Ошибка", theme['clear_bg'], str(e))
            finally:
                self.progress.stop()
                self.progress_text.config(text="")

        threading.Thread(target=predict_thread, daemon=True).start()

    def show_probability_chart(self):
        """Показ диаграммы вероятностей"""
        if not hasattr(self, 'last_prediction'):
            messagebox.showinfo("Информация", "Сначала выполните распознавание!")
            return

        try:
            prediction = self.last_prediction

            # Создаем новое окно
            chart_window = tk.Toplevel(self.root)
            chart_window.title("📈 Вероятности предсказаний")
            chart_window.geometry("800x600")
            chart_window.configure(bg=self.themes[self.current_theme]['bg'])

            # Добавляем обводку окна в цвет темы
            chart_window.configure(highlightbackground=self.themes[self.current_theme]['window_border'])
            chart_window.configure(highlightthickness=2)

            # Сохраняем ссылку на окно для возможности изменения темы
            self.probability_window = chart_window

            # Центрируем окно
            chart_window.transient(self.root)
            chart_window.grab_set()

            # Создаем панель управления окном
            self.create_window_controls(chart_window)

            # Заголовок окна
            title_label = tk.Label(
                chart_window,
                text="📈 Вероятности предсказаний нейросети",
                font=("Segoe UI", 14, "bold"),
                bg=self.themes[self.current_theme]['bg'],
                fg=self.themes[self.current_theme]['fg']
            )
            title_label.pack(pady=10)

            # Создаем matplotlib фигуру с цветом фона темы
            fig, ax = plt.subplots(figsize=(10, 6),
                                   facecolor=self.themes[self.current_theme]['frame_bg'])

            digits = list(range(10))
            probabilities = [prediction[i] for i in digits]

            # Создаем цветовую палитру
            colors = []
            max_prob = max(probabilities)
            for prob in probabilities:
                if prob == max_prob:
                    colors.append('#27ae60')  # Зеленый для максимальной вероятности
                elif prob > 0.1:
                    colors.append('#3498db')  # Синий для средних
                else:
                    colors.append('#95a5a6')  # Серый для низких

            # Создаем столбчатую диаграмму
            bars = ax.bar(digits, probabilities, color=colors, alpha=0.8,
                          edgecolor=self.themes[self.current_theme]['border_color'],
                          linewidth=1)

            # Настройки диаграммы
            ax.set_xlabel('Цифры', fontsize=12, color=self.themes[self.current_theme]['fg'])
            ax.set_ylabel('Вероятность', fontsize=12, color=self.themes[self.current_theme]['fg'])
            ax.set_title('Вероятности предсказаний', fontsize=14,
                         color=self.themes[self.current_theme]['fg'], pad=20)
            ax.set_ylim(0, 1)

            # Добавляем значения на столбцы
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                height = bar.get_height()
                if height > 0.01:  # Показываем только значимые значения
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{prob:.1%}',
                            ha='center', va='bottom',
                            fontsize=10,
                            color=self.themes[self.current_theme]['fg'],
                            weight='bold')

            # Настройка цветов осей и сетки
            ax.tick_params(colors=self.themes[self.current_theme]['fg'], labelsize=10)
            ax.spines['bottom'].set_color(self.themes[self.current_theme]['fg'])
            ax.spines['top'].set_color(self.themes[self.current_theme]['fg'])
            ax.spines['left'].set_color(self.themes[self.current_theme]['fg'])
            ax.spines['right'].set_color(self.themes[self.current_theme]['fg'])

            # Добавляем сетку
            ax.grid(True, alpha=0.3, color=self.themes[self.current_theme]['fg'])

            # Настройка меток по оси X
            ax.set_xticks(digits)
            ax.set_xticklabels(digits, color=self.themes[self.current_theme]['fg'])

            # Встраиваем в tkinter
            canvas = FigureCanvasTkAgg(fig, chart_window)
            canvas.draw()

            # Добавляем панель информации
            info_frame = tk.Frame(chart_window, bg=self.themes[self.current_theme]['bg'])
            info_frame.pack(fill='x', padx=20, pady=10)

            # Информация о максимальной вероятности
            max_digit = np.argmax(prediction)
            max_prob = prediction[max_digit]
            info_text = f"Наиболее вероятная цифра: {max_digit} ({max_prob:.1%})"

            tk.Label(
                info_frame,
                text=info_text,
                font=("Segoe UI", 11, "bold"),
                bg=self.themes[self.current_theme]['bg'],
                fg=self.themes[self.current_theme]['fg']
            ).pack()

            canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=10)

            # Обновляем тему окна
            self.update_probability_window_theme()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при отображении диаграммы: {str(e)}")

    def create_window_controls(self, window):
        """Создание панели управления окном"""
        control_frame = tk.Frame(window, bg=self.themes[self.current_theme]['window_controls'],
                                 height=30)
        control_frame.pack(fill='x')
        control_frame.pack_propagate(False)

        # Кнопки управления окном
        button_frame = tk.Frame(control_frame, bg=self.themes[self.current_theme]['window_controls'])
        button_frame.pack(side=tk.RIGHT, padx=5)

        # Кнопка свернуть
        minimize_btn = tk.Button(
            button_frame,
            text="−",
            font=("Arial", 12, "bold"),
            width=2,
            height=1,
            bg=self.themes[self.current_theme]['window_controls'],
            fg=self.themes[self.current_theme]['fg'],
            activebackground=self.themes[self.current_theme]['button_bg'],
            activeforeground='white',
            relief='flat',
            bd=0,
            command=window.iconify
        )
        minimize_btn.pack(side=tk.LEFT, padx=2)

        # Кнопка закрыть
        close_btn = tk.Button(
            button_frame,
            text="×",
            font=("Arial", 12, "bold"),
            width=2,
            height=1,
            bg=self.themes[self.current_theme]['window_controls'],
            fg=self.themes[self.current_theme]['fg'],
            activebackground=self.themes[self.current_theme]['clear_bg'],
            activeforeground='white',
            relief='flat',
            bd=0,
            command=window.destroy
        )
        close_btn.pack(side=tk.LEFT, padx=2)

        # Делаем панель перетаскиваемой
        def start_move(event):
            window.x = event.x
            window.y = event.y

        def do_move(event):
            deltax = event.x - window.x
            deltay = event.y - window.y
            x = window.winfo_x() + deltax
            y = window.winfo_y() + deltay
            window.geometry(f"+{x}+{y}")

        control_frame.bind("<ButtonPress-1>", start_move)
        control_frame.bind("<B1-Motion>", do_move)

    def update_probability_window_theme(self):
        """Обновление темы окна вероятностей"""
        if hasattr(self, 'probability_window') and self.probability_window.winfo_exists():
            theme = self.themes[self.current_theme]
            self.probability_window.configure(bg=theme['bg'])
            self.probability_window.configure(highlightbackground=theme['window_border'])

            # Обновляем все дочерние элементы
            for widget in self.probability_window.winfo_children():
                if isinstance(widget, tk.Label):
                    if 'Вероятности предсказаний' in widget.cget('text') or 'Наиболее вероятная' in widget.cget('text'):
                        widget.configure(bg=theme['bg'], fg=theme['fg'])
                    else:
                        widget.configure(bg=theme['bg'], fg=theme['fg'])
                elif isinstance(widget, tk.Frame):
                    widget.configure(bg=theme['bg'])
                    # Рекурсивно обновляем дочерние элементы
                    for child in widget.winfo_children():
                        if isinstance(child, tk.Label):
                            child.configure(bg=theme['bg'], fg=theme['fg'])
                        elif isinstance(child, tk.Button):
                            child.configure(bg=theme['window_controls'], fg=theme['fg'])

    def show_processed_image(self):
        """Улучшенный предпросмотр с дополнительной информацией"""
        try:
            img_array = self.preprocess_image()

            preview_window = tk.Toplevel(self.root)
            preview_window.title("📊 Анализ изображения")
            preview_window.geometry("400x550")
            preview_window.configure(bg=self.themes[self.current_theme]['bg'])
            preview_window.resizable(False, False)

            # Добавляем обводку окна в цвет темы
            preview_window.configure(highlightbackground=self.themes[self.current_theme]['window_border'])
            preview_window.configure(highlightthickness=2)

            # Центрируем окно
            preview_window.transient(self.root)
            preview_window.grab_set()

            # Заголовок окна
            tk.Label(
                preview_window,
                text="📊 Анализ обработанного изображения",
                font=("Segoe UI", 12, "bold"),
                bg=self.themes[self.current_theme]['bg'],
                fg=self.themes[self.current_theme]['fg']
            ).pack(pady=10)

            # Создаем matplotlib фигуру
            fig, ax = plt.subplots(figsize=(3, 3), facecolor=self.themes[self.current_theme]['frame_bg'])
            ax.imshow(img_array.reshape(28, 28), cmap='gray')
            ax.set_title("28×28 пикселей", fontsize=10, color=self.themes[self.current_theme]['fg'])
            ax.axis('off')

            # Настраиваем цвета для matplotlib
            if self.current_theme == 'dark' or self.current_theme == 'blue':
                ax.tick_params(colors='white')

            # Встраиваем в tkinter
            canvas = FigureCanvasTkAgg(fig, preview_window)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=10)

            # Информация о предобработке
            info_frame = tk.Frame(preview_window, bg=self.themes[self.current_theme]['bg'])
            info_frame.pack(pady=10)

            tk.Label(
                info_frame,
                text="✓ Изменение размера до 28×28\n✓ Центрирование по массе\n✓ Нормализация значений",
                font=("Segoe UI", 9),
                bg=self.themes[self.current_theme]['bg'],
                fg=self.themes[self.current_theme]['text_secondary'],
                justify='left'
            ).pack()

            # Статистика изображения
            stats_frame = tk.Frame(preview_window, bg=self.themes[self.current_theme]['bg'])
            stats_frame.pack(pady=10)

            img_stats = img_array.flatten()
            tk.Label(
                stats_frame,
                text=f"📊 Статистика:\n"
                     f"Мин. значение: {img_stats.min():.3f}\n"
                     f"Макс. значение: {img_stats.max():.3f}\n"
                     f"Среднее: {img_stats.mean():.3f}\n"
                     f"Стандартное отклонение: {img_stats.std():.3f}",
                font=("Segoe UI", 9),
                bg=self.themes[self.current_theme]['bg'],
                fg=self.themes[self.current_theme]['fg'],
                justify='left'
            ).pack()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при отображении: {str(e)}")

    def create_tooltip(self, widget, text):
        """Создание всплывающей подсказки"""

        def on_enter(event):
            self.tooltip = tk.Toplevel()
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

            label = tk.Label(
                self.tooltip,
                text=text,
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                font=("Segoe UI", 8),
                padx=5,
                pady=2
            )
            label.pack()

        def on_leave(event):
            if hasattr(self, 'tooltip') and self.tooltip:
                self.tooltip.destroy()
                self.tooltip = None

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def run(self):
        """Запуск приложения"""
        # Центрируем окно
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (450 // 2)
        y = (self.root.winfo_screenheight() // 2) - (700 // 2)
        self.root.geometry(f"450x700+{x}+{y}")

        self.root.mainloop()


if __name__ == "__main__":
    try:
        app = ModernDigitRecognizer()
        app.run()
    except Exception as e:
        print(f"Ошибка запуска приложения: {e}")
        import traceback

        traceback.print_exc()
        import sys

        sys.exit(1)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Загрузка обученной модели
model = load_model("../resources/models/improved_digit_recognition_model.h5")

# Загружаем данные MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализуем данные
x_test = x_test / 255.0


# Тестирование модели на тестовых данных
def test_model(model, x_test, y_test):
    correct_predictions = 0
    total_predictions = len(y_test)

    # Перебираем тестовые данные
    for i in range(total_predictions):
        img = x_test[i]
        true_label = y_test[i]

        # Изменяем размер изображения до (28, 28, 1) для подачи в модель
        img = img.reshape(1, 28, 28, 1)

        # Прогнозируем
        prediction = model.predict(img)
        predicted_label = np.argmax(prediction)

        # Сравниваем с истинной меткой
        if predicted_label == true_label:
            correct_predictions += 1

        # Показываем картинку и предсказания
        if i % 100 == 0:  # показываем изображение каждые 100 примеров
            plt.imshow(img.reshape(28, 28), cmap="gray")
            plt.title(f"True: {true_label}, Predicted: {predicted_label}")
            plt.show()

    # Печатаем общую точность
    accuracy = correct_predictions / total_predictions
    print(f"Точность на тестовых данных: {accuracy * 100:.2f}%")


# Тестируем модель
test_model(model, x_test, y_test)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import cv2
import os


def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return project_root


def find_last_conv_layer(model):
    """
    Находит имя последнего сверточного слоя в модели
    """
    for i in reversed(range(len(model.layers))):
        layer = model.layers[i]
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"Найден последний сверточный слой: {layer.name} (индекс: {i})")
            return layer.name
    print("Сверточные слои не найдены!")
    return None


# Загружаем модель
project_root = get_project_root()
model_path = os.path.join(project_root, "src", "resources", "models", "improved_digit_recognition_model.keras")
model = tf.keras.models.load_model(model_path)

print("Model type:", type(model))
print("Trainable params:", model.count_params())

# Показываем все слои для отладки
print("\nСлои модели:")
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} ({type(layer).__name__})")

# Находим последний сверточный слой автоматически
last_conv_layer_name = find_last_conv_layer(model)
if last_conv_layer_name is None:
    raise ValueError("Не найден ни один сверточный слой в модели!")

# Загружаем тестовые данные MNIST
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = np.expand_dims(x_test, axis=-1)

# ВАЖНО: Прогоняем модель через данные для полной инициализации
print("Инициализация модели...")
dummy_input = np.expand_dims(x_test[0], axis=0)
dummy_prediction = model(dummy_input)
print("Модель инициализирована")

# Создаем копию модели для Grad-CAM (без обучения)
print("Создание копии модели для Grad-CAM...")
model_copy = tf.keras.models.clone_model(model)
model_copy.set_weights(model.get_weights())
model_copy.trainable = False


# Функция Grad-CAM (исправленная)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Преобразуем numpy массив в TensorFlow тензор
    img_array = tf.cast(img_array, tf.float32)
    img_array = tf.expand_dims(img_array, axis=0)

    # Получаем последний сверточный слой
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Создаем модель для получения градиентов
    grad_model = tf.keras.models.Model(
        inputs=model.layers[0].input,
        outputs=[last_conv_layer.output, model.layers[-1].output]
    )

    with tf.GradientTape() as tape:
        tape.watch(img_array)
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Градиенты равны None. Проверьте имя слоя и модель.")

    # Вычисляем heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap


# Визуализация Grad-CAM
def show_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    superimposed_img = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img


# Пример на случайном изображении
print("Выбираем случайное изображение...")
i = np.random.randint(0, len(x_test))
test_img, true_label = x_test[i], y_test[i]

print(f"Изображение {i}, истинная метка: {true_label}")

# Вызываем Grad-CAM с автоматически найденным слоем
try:
    print("Выполняем Grad-CAM...")
    heatmap = make_gradcam_heatmap(test_img, model_copy, last_conv_layer_name)

    if heatmap is not None:
        gradcam_img = show_gradcam(test_img.squeeze(), heatmap)

        # Визуализация
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(test_img.squeeze(), cmap="gray")
        plt.title(f"Оригинал (label={true_label})")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap="jet")
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(gradcam_img)
        plt.title("Grad-CAM Overlay")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        print("Grad-CAM успешно выполнен!")
    else:
        print("Не удалось создать heatmap")

except Exception as e:
    print(f"Ошибка: {e}")
    import traceback

    traceback.print_exc()
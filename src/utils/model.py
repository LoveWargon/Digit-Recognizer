import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def get_project_root():
    """Возвращает путь к корневой директории проекта."""
    # Получаем путь к текущему файлу
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Поднимаемся на два уровня вверх: src/utils -> проект
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return project_root

# Отключаем предупреждения
tf.get_logger().setLevel('ERROR')

# Загружаем данные MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализуем данные
x_train, x_test = x_train / 255.0, x_test / 255.0

# Добавляем размерность для сверточной сети
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# Разделяем обучающую выборку на train и validation
x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train_cnn, y_train, test_size=0.1, random_state=42, stratify=y_train
)


# Улучшенная CNN модель
def create_improved_cnn_model():
    model = models.Sequential([
        # Первый блок
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Второй блок
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Третий блок
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Классификационный блок
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model


# Создаем модель
model = create_improved_cnn_model()

# Компилируем с улучшенным оптимизатором
model.compile(
    optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks для лучшего обучения
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
]

# Аугментация данных
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)

datagen.fit(x_train_split)

# Обучаем модель с аугментацией и правильной валидацией
print("Начинаем обучение...")
history = model.fit(
    datagen.flow(x_train_split, y_train_split, batch_size=128),
    epochs=200, # 200
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Оцениваем модель на тестовых данных
test_loss, test_acc = model.evaluate(x_test_cnn, y_test, verbose=0)
print(f"\nТочность на тестовых данных: {test_acc:.4f}")


# Визуализация обучения
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # График потерь
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # График точности
    ax2.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# Показываем графики обучения
plot_training_history(history)

# Сохраняем модель
project_root = get_project_root()
model_save_path = os.path.join(project_root, "src", "resources", "models", "improved_digit_recognition_model.h5")
model.save(model_save_path)
print(f"Модель сохранена как '{model_save_path}'")

# Сохраняем модель в формате Keras
keras_save_path = os.path.join(project_root, "src", "resources", "models", "improved_digit_recognition_model.keras")
model.save(keras_save_path)
print(f"Модель сохранена в формате Keras как '{keras_save_path}'")

# Также сохраняем в формате SavedModel
savedmodel_path = os.path.join(project_root, "src", "resources", "models", "improved_digit_recognition_model_savedmodel")
model.export(savedmodel_path)

print(f"Модель также сохранена в формате SavedModel как '{savedmodel_path}'")

# Выводим информацию о модели
print("\nАрхитектура модели:")
model.summary()


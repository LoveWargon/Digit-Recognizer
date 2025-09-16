import subprocess
import shlex
import os
import sys

# --- Настройки ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
saved_model_path = os.path.join(project_root, "src", "resources", "models",
                                "improved_digit_recognition_model_savedmodel")
onnx_output_path = os.path.join(project_root, "src", "resources", "models", "improved_digit_recognition_model.onnx")
opset = 15
# ------------------

# 1. Используем тот же Python, что и для запуска скрипта
python_executable = sys.executable # <-- Ключевое изменение

command_list = [
    python_executable, "-m", "tf2onnx.convert", # <-- Используем sys.executable
    "--saved-model", saved_model_path,
    "--output", onnx_output_path,
    "--opset", str(opset)
]

print("Планируемая команда:", shlex.join(command_list))

try:
    result = subprocess.run(command_list, check=True, text=True)
    print(" Конвертация завершена успешно!")
except subprocess.CalledProcessError as e:
    print(f" Ошибка при выполнении команды (код возврата {e.returncode}).")
    print("Вы можете попробовать выполнить эту команду вручную в терминале:")
    print(shlex.join(command_list))
except FileNotFoundError:
    print(" Ошибка: Python не найден.")
except Exception as e:
    print(f" Неожиданная ошибка: {e}")
    print("Вы можете попробовать выполнить эту команду вручную в терминале:")
    print(shlex.join(command_list))

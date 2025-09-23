import os
import sys

print("="*50)
print("=== ЗАПУСК ДІАГНОСТИЧНОГО СКРИПТА ===")
print("="*50)

try:
    # Показуємо, звідки запускається скрипт
    cwd = os.getcwd()
    print(f"ПОТОЧНА РОБОЧА ДИРЕКТОРІЯ (os.getcwd()): {cwd}")

    # Показуємо, де фізично лежить сам файл скрипта
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    print(f"АБСОЛЮТНИЙ ШЛЯХ ДО СКРИПТА (__file__): {script_path}")
    print(f"ПАПКА, В ЯКІЙ ЛЕЖИТЬ СКРИПТ (dirname): {script_dir}")

    # Тепер ми дослідимо папки навколо нашого скрипта
    # Це найважливіша частина. Вона покаже нам, де лежить папка 'documents'
    print("\n" + "="*20 + " ВМІСТ БАТЬКІВСЬКОЇ ПАПКИ " + "="*20)
    parent_of_script_dir = os.path.dirname(script_dir)
    os.system(f"ls -laR {parent_of_script_dir}") # Рекурсивно показуємо вміст

    print("\n" + "="*50)
    print("=== ДІАГНОСТИКА ЗАВЕРШЕНА. ПРИМУСОВИЙ ВИХІД. ===")
    print("="*50)

    # Ми спеціально викликаємо помилку, щоб зупинити процес збірки
    # і дати нам можливість спокійно подивитись логи.
    sys.exit(1)

except Exception as e:
    print(f"!!! Помилка під час діагностики: {e} !!!")
    sys.exit(1)
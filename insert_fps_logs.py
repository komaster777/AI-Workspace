import time

def insert_fps_logs(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Строки, добавляемые в начале файла
    header_lines = [
        "import time\n",
        "\n",
        "def log_with_fps(tag, start_time):\n",
        "    elapsed = time.time() - start_time\n",
        "    fps = 1 / elapsed if elapsed > 0 else float('inf')\n",
        "    print(f\"[INFO] {tag}: {fps:.2f} FPS\")\n",
        "    if fps < 10:\n",
        "        print(f\"[WARNING] Низкий FPS: {fps:.2f} сек\")\n",
        "\n",
    ]

    new_lines = header_lines.copy()
    current_line_number = 1 + len(header_lines)  # Нумерация строк с учетом добавленных строк в начале
    start_logging = False
    inside_multiline_block = False  # Флаг для отслеживания многострочных конструкций
    block_indent = 0  # Уровень отступа для текущей многострочной конструкции
    inside_any_function = False  # Флаг для отслеживания нахождения внутри any()
    inside_list_comprehension = False  # Флаг для отслеживания нахождения внутри list comprehension

    for original_line_number, line in enumerate(lines, 1):
        stripped_line = line.strip()

        # Проверяем, достигли ли строки для начала логирования
        if "start_time = time.time()" in stripped_line:
            start_logging = True

        # Добавляем текущую строку
        new_lines.append(line)
        current_line_number += 1

        # Пропускаем комментарии и пустые строки
        if not stripped_line or stripped_line.startswith("#"):
            continue

        # Определяем отступ текущей строки
        indent = len(line) - len(line.lstrip())
        indent_space = " " * indent

        if start_logging:
            # Проверяем начало многострочной конструкции (например, any(), all(), словарь, список)
            if any(stripped_line.startswith(start) for start in ["any(", "all(", "{", "["]) and not inside_multiline_block:
                inside_multiline_block = True
                block_indent = indent  # Запоминаем отступ для текущей конструкции
                if stripped_line.startswith("any("):
                    inside_any_function = True  # Входим в конструкцию any()
                elif stripped_line.startswith("[") or stripped_line.startswith("{"):
                    inside_list_comprehension = True  # Входим в list comprehension
                continue

            # Проверяем закрытие многострочной конструкции
            if inside_any_function and stripped_line.endswith(")"):
                inside_any_function = False  # Выход из any()

            if inside_list_comprehension and stripped_line.endswith("]") or stripped_line.endswith("}"):
                inside_list_comprehension = False  # Выход из list comprehension

            # Если мы внутри any() или list comprehension, пропускаем вставку log_with_fps
            if inside_any_function or inside_list_comprehension:
                continue

            # Проверяем закрытие многострочной конструкции
            if inside_multiline_block:
                if stripped_line.endswith((")", "]", "}")) and indent == block_indent:
                    inside_multiline_block = False
                    new_lines.append(f"{indent_space}log_with_fps('FPS №{current_line_number}', start_time)\n")
                    current_line_number += 1
                continue  # Пропускаем вставку log_with_fps внутри многострочной конструкции

            # Если строка заканчивается двоеточием (начало блока), добавляем log_with_fps внутри блока
            if stripped_line.endswith(":"):
                new_lines.append(f"{indent_space}    log_with_fps('FPS №{current_line_number}', start_time)\n")
                current_line_number += 1
            else:
                # Вставляем log_with_fps на том же уровне отступа
                new_lines.append(f"{indent_space}log_with_fps('FPS №{current_line_number}', start_time)\n")
                current_line_number += 1

    # Записываем обработанный файл
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


if __name__ == "__main__":
    # Пример использования
    insert_fps_logs("Main Recognition Processed.py", "Main Recognition with FPS.py")

import re

def remove_indentation(code: str) -> str:
    def collapse_multiline_block(start_keyword: str, code: str) -> str:
        result = ""
        i = 0
        while i < len(code):
            if code.startswith(start_keyword, i):
                start = i + len(start_keyword)
                if code[start] != '(':
                    result += code[i]
                    i += 1
                    continue

                bracket_level = 1
                j = start + 1
                while j < len(code) and bracket_level > 0:
                    if code[j] == '(':
                        bracket_level += 1
                    elif code[j] == ')':
                        bracket_level -= 1
                    j += 1

                block = code[i:j]
                # Удаляем лишние пробелы без нарушения структуры
                block_single_line = re.sub(r'\s+', ' ', block).strip()
                result += block_single_line
                i = j
            else:
                result += code[i]
                i += 1
        return result

    def preserve_strings(code: str, func) -> str:
        """Сохраняет содержимое строк (в кавычках) и восстанавливает его после обработки."""
        string_pattern = re.compile(r'(["\'])(.*?)(\1)')
        strings = []  # Сохраняем строки
        def replacer(match):
            strings.append(match.group(0))
            return f"__STRING_{len(strings) - 1}__"

        # Заменяем строки на маркеры
        code = string_pattern.sub(replacer, code)

        # Обрабатываем код без строк
        code = func(code)

        # Восстанавливаем строки
        for i, string in enumerate(strings):
            code = code.replace(f"__STRING_{i}__", string)

        return code

    # Обрабатываем блоки с any()
    code = collapse_multiline_block("any", code)

    # Обрабатываем словари и списки в коде, с сохранением строк
    def process_code_without_strings(code: str) -> str:
        code = re.sub(r"\{\s*(.*?)\s*\}", lambda m: '{ ' + re.sub(r'\s+', ' ', m.group(1)).strip() + ' }', code, flags=re.DOTALL)
        code = re.sub(r"\[\s*(.*?)\s*\]", lambda m: '[ ' + re.sub(r'\s+', ' ', m.group(1)).strip() + ' ]', code, flags=re.DOTALL)
        return code

    # Сохраняем строки, обрабатываем код, затем восстанавливаем строки
    code = preserve_strings(code, process_code_without_strings)

    return code


def process_file(input_filename: str, output_filename: str):
    """Функция для обработки файла и записи преобразованного кода."""
    with open(input_filename, 'r', encoding='utf-8') as f:
        code = f.read()

    # Преобразуем код с помощью функции
    processed_code = remove_indentation(code)

    # Записываем обработанный код в новый файл
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(processed_code)


if __name__ == "__main__":
    input_filename = "Main Recognition.py"  # Исходный файл
    output_filename = "Main Recognition Processed.py"  # Новый файл
    process_file(input_filename, output_filename)

# Тестовый пример
code1 = """
still_present = any(
    p['name'] == current_name and bbox_intersects_polygon(p['bbox'], zone_bbox)
    for track_id2, p in tracked_people.items() if track_id2 != track_id
)

cv2.putText(frame, f"{confidence:.2f}", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
"""
result1 = remove_indentation(code1)
print(f"result1: {result1}")
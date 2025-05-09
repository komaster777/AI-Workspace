import re

def remove_indentation(code: str) -> str:
    # Убираем новые строки и лишние пробелы
    return re.sub(r"\s*\n\s*", " ", code.strip())

# Пример 1
code1 = """
tracked_people[track_id] = {
                            'name': matched_name,
                            'bbox': assigned_body['bbox'],
                            'face_bbox': (x, y, w, h),
                            'last_seen': now,
                            'first_seen': now,
                            'last_retry': now,
                            'last_check1': now,
                            'last_check2': now,
                            'IDs': 0,
                            'confidence_window': deque(maxlen=5)
                        }
"""
result1 = remove_indentation(code1)
print(f"result1: {result1}")

# Пример 2
code2 = """
still_present = any(
                                    p['name'] == current_name and bbox_intersects_polygon(p['bbox'], zone_bbox)
                                    for track_id2, p in tracked_people.items() if track_id2 != track_id
                                )
"""
result2 = remove_indentation(code2)
print(f"result2: {result2}")
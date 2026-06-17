import os
import json
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

def call_ollama(system_prompt: str, user_prompt: str, json_mode: bool = False):
    url = f"{OLLAMA_URL}/api/generate"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}",
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    if json_mode:
        payload["format"] = "json"
        
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        res_json = response.json()
        return res_json.get("response", "")
    except Exception as e:
        print(f"Error calling Ollama ({url}): {e}")
        return ""

def process_receipt_ocr(ocr_lines):
    raw_text = "\n".join([f"{l['text']}" for l in ocr_lines])
    
    system_prompt = (
        "Ты — AI-ассистент для разбора чеков и нормализации товаров. "
        "Тебе на вход даётся распознанный текст чека. Твоя задача — извлечь данные и вернуть строго JSON-объект.\n"
        "Требования к структуре JSON:\n"
        "{\n"
        '  "date": "YYYY-MM-DD" (или текущая дата, если не найдена в чеке),\n'
        '  "total_amount": float (итоговая сумма чека),\n'
        '  "category": "продукты" | "транспорт" | "аптека" | "развлечения" | "другое" (категория чека),\n'
        '  "items": [\n'
        "    {\n"
        '      "name": "оригинальное наименование товара из чека",\n'
        '      "normalized_name": "короткое нормализованное название товара на русском языке (например, Сыр Российский, Молоко 3.2%, Батон нарезной, Шоколад Alpen Gold)",\n'
        '      "price": float (цена за 1 ед. товара),\n'
        '      "qty": float (количество),\n'
        '      "amount": float (общая стоимость этой позиции),\n'
        '      "category": "категория товара (например, продукты, медикаменты, бытовая химия)"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Отвечай строго в формате JSON, без лишнего текста и без markdown-разметки."
    )
    
    user_prompt = f"Распознанный текст чека:\n{raw_text}"
    response_text = call_ollama(system_prompt, user_prompt, json_mode=True)
    try:
        return json.loads(response_text)
    except Exception as e:
        print(f"Failed to parse JSON: {e}. Raw: {response_text}")
        return {
            "date": "",
            "total_amount": 0.0,
            "category": "другое",
            "items": []
        }

def process_lab_results_ocr(ocr_lines):
    raw_text = "\n".join([f"{l['text']}" for l in ocr_lines])
    
    system_prompt = (
        "Ты — медицинский AI-помощник по структурированию результатов анализов. "
        "Тебе даётся распознанный текст бланка лаборатории. Извлеки показатели и верни строго JSON.\n"
        "Требования к структуре JSON:\n"
        "{\n"
        '  "date": "YYYY-MM-DD" (или текущая дата, если не найдена),\n'
        '  "metrics": [\n'
        "    {\n"
        '      "title": "название показателя на русском или латинском (например, Гемоглобин, Ферритин, Холестерин, АЛТ)",\n'
        '      "value": "числовое значение показателя в виде строки (например, 142 или 4.5)",\n'
        '      "unit": "единицы измерения (например, г/л, ммоль/л, %)",\n'
        '      "reference_range": "референсный диапазон лаборатории в виде строки (например, 130-160 или < 5.0)"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Важно: не ставь медицинских диагнозов. Извлекай только точные данные из текста.\n"
        "Отвечай строго в формате JSON, без markdown-разметки."
    )
    
    user_prompt = f"Текст бланка анализов:\n{raw_text}"
    response_text = call_ollama(system_prompt, user_prompt, json_mode=True)
    try:
        return json.loads(response_text)
    except Exception as e:
        print(f"Failed to parse JSON: {e}. Raw: {response_text}")
        return {
            "date": "",
            "metrics": []
        }

def generate_recommendations(user_data_json):
    system_prompt = (
        "Ты — заботливый AI-помощник LifeHub. Твоя задача — проанализировать последние данные "
        "пользователя (расходы, сон, давление, настроение/заметки, проекты/дела) и дать короткие советы.\n"
        "Напиши отчет в красивом формате Markdown на русском языке. Отчет должен включать:\n"
        "1. **Анализ расходов** (краткий обзор трат, превышение лимитов, аномалии).\n"
        "2. **Оценка здоровья** (качество сна, давление, пульс, тренды по показателям анализов).\n"
        "3. **Практичные рекомендации** (советы по оптимизации бюджета, балансу работы и отдыха, самочувствию).\n"
        "Предупреждение: не ставь диагнозы и не назначай лечение. Пиши лаконично, дружелюбно, структурированно."
    )
    
    user_prompt = f"Данные пользователя за последние 30 дней:\n{json.dumps(user_data_json, ensure_ascii=False, indent=2)}"
    return call_ollama(system_prompt, user_prompt, json_mode=False)

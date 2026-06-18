import os
import json
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
PROVERKACHEKA_TOKEN = os.environ.get("PROVERKACHEKA_TOKEN", "")

def get_receipt_from_proverkacheka(qr_raw: str) -> dict:
    if not PROVERKACHEKA_TOKEN:
        print("[ProverkaCheka] No token configured (PROVERKACHEKA_TOKEN). Skipping.")
        return None
        
    url = "https://proverkacheka.com/api/v1/check/get"
    data = {
        "token": PROVERKACHEKA_TOKEN,
        "qrraw": qr_raw
    }
    
    try:
        print(f"[ProverkaCheka] Querying FNS for QR code raw: {qr_raw}")
        res = requests.post(url, data=data, timeout=30)
        res.raise_for_status()
        res_json = res.json()
        
        code = res_json.get("code")
        if code == 1:
            print("[ProverkaCheka] Check data successfully fetched from FNS.")
            return res_json.get("data", {}).get("json")
        else:
            error_msg = res_json.get("data", "Unknown error")
            print(f"[ProverkaCheka] API error code {code}: {error_msg}")
            return None
    except Exception as e:
        print(f"[ProverkaCheka] Connection error: {e}")
        return None

def normalize_receipt_items(receipt_json: dict) -> dict:
    """
    Takes the raw receipt JSON from FNS/Proverkacheka and normalizes it using Ollama.
    """
    items_raw = receipt_json.get("items", [])
    date_str = receipt_json.get("dateTime", "")
    if "T" in date_str:
        date_str = date_str.split("T")[0]
    elif len(date_str) >= 10:
        date_str = date_str[:10]  # Take first 10 chars (YYYY-MM-DD)
    else:
        date_str = ""
        
    # FNS totalSum is in kopecks (e.g. 19590 = 195.9)
    total_amount_raw = receipt_json.get("totalSum", 0)
    total_amount = float(total_amount_raw) / 100.0 if isinstance(total_amount_raw, int) else float(total_amount_raw)
    
    items_formatted = []
    for item in items_raw:
        name = item.get("name", "")
        # FNS prices and sums are in kopecks (integers)
        raw_price = item.get("price", 0)
        raw_sum = item.get("sum", 0)
        
        price = float(raw_price) / 100.0 if isinstance(raw_price, int) else float(raw_price)
        amount = float(raw_sum) / 100.0 if isinstance(raw_sum, int) else float(raw_sum)
        qty = float(item.get("quantity", 1.0))
        
        items_formatted.append({
            "name": name,
            "price": price,
            "qty": qty,
            "amount": amount
        })
        
    system_prompt = (
        "Ты — AI-ассистент для нормализации товаров в чеке. "
        "Тебе на вход дается список товаров с их оригинальными названиями, ценами и количеством. "
        "Твоя задача — вернуть строго JSON-объект с нормализованными названиями и категориями для каждого товара, "
        "а также определить общую категорию чека.\n"
        "Требования к структуре JSON:\n"
        "{\n"
        f'  "date": "{date_str}",\n'
        f'  "total_amount": {total_amount},\n'
        '  "category": "продукты" | "транспорт" | "аптека" | "развлечения" | "другое",\n'
        '  "items": [\n'
        "    {\n"
        '      "name": "оригинальное наименование товара",\n'
        '      "normalized_name": "короткое нормализованное название товара на русском (например, Сыр Российский, Молоко 3.2%, Батон)",\n'
        '      "price": float,\n'
        '      "qty": float,\n'
        '      "amount": float,\n'
        '      "category": "категория товара (продукты, медикаменты, бытовая химия, алкоголь, табак, одежда, прочее)"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Отвечай строго в формате JSON, без лишнего текста и без markdown-разметки."
    )
    
    user_prompt = f"Список товаров из чека:\n{json.dumps(items_formatted, ensure_ascii=False, indent=2)}"
    response_text = call_ollama(system_prompt, user_prompt, json_mode=True, timeout=15)
    
    try:
        res_dict = json.loads(response_text)
        res_dict["date"] = date_str
        res_dict["total_amount"] = total_amount
        return res_dict
    except Exception as e:
        print(f"Failed to parse Ollama normalization JSON: {e}. Raw: {response_text}")
        # Return fallback with raw items
        items_fallback = []
        for it in items_formatted:
            items_fallback.append({
                "name": it["name"],
                "normalized_name": it["name"],
                "price": it["price"],
                "qty": it["qty"],
                "amount": it["amount"],
                "category": "другое"
            })
        return {
            "date": date_str,
            "total_amount": total_amount,
            "category": "другое",
            "items": items_fallback
        }

def call_ollama(system_prompt: str, user_prompt: str, json_mode: bool = False, timeout: int = 60):
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
        response = requests.post(url, json=payload, timeout=timeout)
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
    response_text = call_ollama(system_prompt, user_prompt, json_mode=True, timeout=40)
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
    response_text = call_ollama(system_prompt, user_prompt, json_mode=True, timeout=40)
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

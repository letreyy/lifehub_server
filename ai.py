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

KEYWORD_MAPPINGS = {
    "food": [
        "молоко", "кефир", "сыр", "масло", "творог", "сметана", "йогурт", "сливки", "ряженка", "простокваша",
        "хлеб", "батон", "булка", "булочка", "лаваш", "сушки", "багет", "чиабатта", "лепешка",
        "картофель", "картошка", "лук", "зелень", "укроп", "петрушка", "кинза", "салат", "шпинат", "чеснок", "морковь", "свекла", "капуста", "яблоки", "бананы", "апельсин", "лимон", "мандарин", "грейпфрут", "груша", "виноград", "слива", "фрукты", "ягоды", "помидор", "томат", "огурец", "огурцы", "перец", "кабачок", "баклажан", "грибы", "шампиньоны",
        "вода", "сок", "нектар", "пиво", "вино", "водка", "напиток", "кола", "лимонад", "квас", "чай", "кофе", "какао", "минералка", "минеральн",
        "яйцо", "яйца", "мясо", "курица", "цыпленок", "колбаса", "сосиски", "ветчина", "рыба", "фарш", "говядина", "свинина", "индейка", "стейк", "шницель", "котлет", "окорок", "грудинка", "карбонад",
        "сахар", "соль", "мука", "крупа", "рис", "гречка", "макароны", "паста", "вермишель", "спагетти",
        "шоколад", "конфеты", "печенье", "вафли", "торт", "пирожное", "мороженое", "чипсы", "орехи", "сухарики", "пряники",
        "майонез", "кетчуп", "соус", "приправа", "уксус", "масло подсолн", "растительное масло", "оливковое масло",
        "бакалея", "консервы", "пюре", "каша", "хлопья", "мюсли", "джем", "мед", "варенье", "зефир"
    ],
    "health": [
        "таблетки", "лекарство", "сироп", "мазь", "пластырь", "бинт", "витамины", "аспирин", "парацетамол", "анальгин",
        "аптека", "препарат", "капли", "спрей", "гель для десен", "антисептик", "маска мед", "терафлю", "нурофен", "но-шпа"
    ],
    "home": [
        "мыло", "шампунь", "гель", "порошок", "кондиционер", "салфетки", "паста", "щетка", "бумага", "освежитель",
        "губка", "тряпка", "средство для", "вешалка", "посуда", "тарелка", "вилка", "ложка", "нож",
        "лампочка", "батарейка", "клей", "скотч", "порошок стир", "ополаскиватель", "белизна", "доместос", "туалетн", "влажные",
        "пена для", "бритья", "дезодорант", "ватные", "зубная", "освежитель", "чистки", "мытья"
    ],
    "transport": [
        "билет", "проезд", "метро", "автобус", "трамвай", "электричка", "поезд"
    ],
    "fuel": [
        "бензин", "топливо", "дизель", "аи-95", "аи-92", "дт"
    ],
    "other": [
        "пакет", "мешок", "коробка", "упаковка"
    ]
}

CATEGORY_MAP = {
    "продукты": "food",
    "еда": "food",
    "food": "food",
    "аптека": "health",
    "здоровье": "health",
    "медикаменты": "health",
    "health": "health",
    "бытовая химия": "home",
    "дом": "home",
    "для дома": "home",
    "home": "home",
    "транспорт": "transport",
    "transport": "transport",
    "топливо": "fuel",
    "бензин": "fuel",
    "fuel": "fuel",
    "другое": "other",
    "разное": "other",
    "пакет": "other",
    "other": "other"
}

def classify_item_by_keywords(name: str) -> str:
    name_lower = name.lower()
    for category, keywords in KEYWORD_MAPPINGS.items():
        for kw in keywords:
            if kw in name_lower:
                return category
    return "other"

def map_category_key(cat: str) -> str:
    cat_clean = cat.lower().strip()
    return CATEGORY_MAP.get(cat_clean, "other")

def normalize_receipt_items(receipt_json: dict) -> dict:
    """
    Takes the raw receipt JSON from FNS/Proverkacheka, classifies items using fast rules,
    and falls back to Ollama only for unrecognized items.
    """
    items_raw = receipt_json.get("items", [])
    date_str = receipt_json.get("dateTime", "")
    if "T" in date_str:
        date_str = date_str.split("T")[0]
    elif len(date_str) >= 10:
        date_str = date_str[:10]
    else:
        date_str = ""
        
    total_amount_raw = receipt_json.get("totalSum", 0)
    total_amount = float(total_amount_raw) / 100.0 if isinstance(total_amount_raw, int) else float(total_amount_raw)
    
    # Pre-classify items using fast keyword rules
    classified_items = []
    unrecognized_items = []
    
    for item in items_raw:
        name = item.get("name", "")
        raw_price = item.get("price", 0)
        raw_sum = item.get("sum", 0)
        
        price = float(raw_price) / 100.0 if isinstance(raw_price, int) else float(raw_price)
        amount = float(raw_sum) / 100.0 if isinstance(raw_sum, int) else float(raw_sum)
        qty = float(item.get("quantity", 1.0))
        
        category = classify_item_by_keywords(name)
        
        item_entry = {
            "name": name,
            "normalized_name": name,
            "price": price,
            "qty": qty,
            "amount": amount,
            "category": category
        }
        
        if category == "other" and not any(pkg in name.lower() for pkg in ["пакет", "мешок", "коробка"]):
            unrecognized_items.append(item_entry)
        
        classified_items.append(item_entry)
        
    # If we have unrecognized items and Ollama is available, run Ollama to classify ONLY those
    if unrecognized_items and OLLAMA_URL:
        print(f"[ProverkaCheka] Running Ollama fallback on {len(unrecognized_items)} unrecognized items...")
        
        system_prompt = (
            "Ты — AI-ассистент для категоризации товаров. Тебе на вход дается список товаров с их названиями.\n"
            "Твоя задача — сопоставить каждый товар с одной из категорий: "
            "\"food\" (еда, продукты, напитки, алкоголь), "
            "\"health\" (лекарства, витамины, аптека), "
            "\"home\" (бытовая химия, салфетки, товары для дома), "
            "\"transport\" (проездные, билеты), "
            "\"fuel\" (бензин, дизель), "
            "или \"other\" (все остальное).\n"
            "Верни строго JSON-список следующего формата (без markdown-разметки):\n"
            "[\n"
            "  {\n"
            '    "name": "оригинальное наименование товара",\n'
            '    "category": "food" | "health" | "home" | "transport" | "fuel" | "other"\n'
            "  }\n"
            "]"
        )
        
        user_prompt = f"Список товаров:\n{json.dumps([{'name': it['name']} for it in unrecognized_items], ensure_ascii=False)}"
        
        # We set a tight 10 seconds timeout for this small batch Ollama request
        response_text = call_ollama(system_prompt, user_prompt, json_mode=True, timeout=10)
        
        try:
            ai_results = json.loads(response_text)
            if isinstance(ai_results, list):
                # Map names to categories from AI
                ai_map = {item.get("name"): map_category_key(item.get("category", "other")) for item in ai_results}
                
                # Update unrecognized items with AI categories
                for item in classified_items:
                    if item["name"] in ai_map:
                        item["category"] = ai_map[item["name"]]
                        print(f"[ProverkaCheka] AI categorized '{item['name']}' as '{item['category']}'")
        except Exception as e:
            print(f"[ProverkaCheka] AI categorization failed or timed out: {e}. Keeping fast-rule categories.")

    # Determine general check category by largest sum
    category_sums = {}
    for it in classified_items:
        c = it["category"]
        category_sums[c] = category_sums.get(c, 0.0) + it["amount"]
        
    main_category = max(category_sums, key=category_sums.get) if category_sums else "other"
    
    return {
        "date": date_str,
        "total_amount": total_amount,
        "category": main_category,
        "items": classified_items
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

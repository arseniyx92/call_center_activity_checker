"""
Примеры использования модуля llm_corrector.py

Этот файл содержит примеры различных способов использования
RAG-модуля для коррекции и обогащения транскрипций звонков.
"""

from llm_corrector import CallCorrector
import json


def example_1_basic_usage():
    """Пример 1: Базовое использование"""
    print("=" * 60)
    print("Пример 1: Базовое использование")
    print("=" * 60)
    
    # Инициализация
    corrector = CallCorrector()
    
    # Данные звонка
    transcription = """
    эээ здравствуйте да я хотел бы узнать про возврат товара 
    номер заказа орд 12345 я покупал его две недели назад 
    сумма была тысяча пятьсот рублей
    """
    
    metadata = {
        "id": "call_001",
        "client": "+79001234567",
        "duration": 180,
        "type": "in",
        "start": "2024-01-15T10:30:00"
    }
    
    # Обработка
    result = corrector.process_call(
        transcription=transcription,
        call_metadata=metadata
    )
    
    # Вывод результатов
    print("\n📝 Исходная транскрипция:")
    print(result['original_transcription'])
    
    print("\n✅ Исправленная транскрипция:")
    print(result['corrected_transcription'])
    
    print("\n🏷️ Извлеченные сущности:")
    print(json.dumps(result.get('entities', {}), ensure_ascii=False, indent=2))
    
    print("\n📊 Классификация:")
    print(json.dumps(result.get('classification', {}), ensure_ascii=False, indent=2))
    
    print("\n📋 Резюме:")
    print(json.dumps(result.get('summary', {}), ensure_ascii=False, indent=2))


def example_2_only_correction():
    """Пример 2: Только коррекция текста"""
    print("\n" + "=" * 60)
    print("Пример 2: Только коррекция текста")
    print("=" * 60)
    
    corrector = CallCorrector()
    
    raw_text = "эээ да я хотел бы эээ узнать про доставку заказа номер орд 56789"
    
    # Коррекция с контекстом
    context = [
        "Клиент спрашивает про доставку заказа ORD-56789",
        "Вопрос о статусе доставки заказа номер 56789"
    ]
    
    corrected = corrector.correct_text(text=raw_text, context=context)
    
    print(f"\nДо: {raw_text}")
    print(f"После: {corrected}")


def example_3_only_entities():
    """Пример 3: Только извлечение сущностей"""
    print("\n" + "=" * 60)
    print("Пример 3: Только извлечение сущностей")
    print("=" * 60)
    
    corrector = CallCorrector()
    
    text = """
    Здравствуйте, меня зовут Иван Петров, мой телефон 89001234567,
    email ivan@example.com. Я хочу вернуть заказ номер ORD-12345,
    который я купил на сумму 2500 рублей 10 января.
    Адрес доставки: г. Москва, ул. Ленина, д. 10, кв. 5.
    """
    
    entities = corrector.extract_entities(text)
    
    print("\nИзвлеченные сущности:")
    print(json.dumps(entities, ensure_ascii=False, indent=2))


def example_4_batch_processing():
    """Пример 4: Пакетная обработка нескольких звонков"""
    print("\n" + "=" * 60)
    print("Пример 4: Пакетная обработка")
    print("=" * 60)
    
    corrector = CallCorrector()
    
    calls = [
        {
            "transcription": "эээ да я хотел бы узнать про доставку",
            "metadata": {"id": "call_1", "client": "+79001111111"}
        },
        {
            "transcription": "здравствуйте у меня проблема с оплатой заказа орд 111",
            "metadata": {"id": "call_2", "client": "+79002222222"}
        },
        {
            "transcription": "хочу вернуть товар номер заказа орд 222",
            "metadata": {"id": "call_3", "client": "+79003333333"}
        }
    ]
    
    results = []
    for call in calls:
        result = corrector.process_call(
            transcription=call["transcription"],
            call_metadata=call["metadata"],
            include_summary=False  # Ускоряем обработку
        )
        results.append(result)
    
    print(f"\nОбработано звонков: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"\nЗвонок {i}:")
        print(f"  Тип: {result.get('classification', {}).get('type', 'неизвестно')}")
        print(f"  Тема: {result.get('classification', {}).get('topic', 'неизвестно')}")
        print(f"  Заказы: {result.get('entities', {}).get('orders', [])}")


def example_5_custom_configuration():
    """Пример 5: Кастомная конфигурация"""
    print("\n" + "=" * 60)
    print("Пример 5: Кастомная конфигурация")
    print("=" * 60)
    
    # Инициализация с кастомными параметрами
    corrector = CallCorrector(
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        llm_model="gpt-4",
        vector_db_path="custom_vector_db",
        use_cache=True
    )
    
    transcription = "эээ да я хотел бы узнать про возврат"
    metadata = {"id": "call_custom", "client": "+79004444444"}
    
    # Обработка только с коррекцией и классификацией
    result = corrector.process_call(
        transcription=transcription,
        call_metadata=metadata,
        include_similar=True,      # Искать похожие
        include_entities=False,     # Не извлекать сущности
        include_classification=True, # Классифицировать
        include_summary=False       # Не создавать резюме
    )
    
    print("\nРезультат обработки:")
    print(f"Исправлено: {result['corrected_transcription']}")
    print(f"Классификация: {result.get('classification', {})}")
    print(f"Время обработки: {result['metadata']['processing_time']} сек")


def example_6_integration_with_main():
    """Пример 6: Интеграция с main.py"""
    print("\n" + "=" * 60)
    print("Пример 6: Интеграция с основным workflow")
    print("=" * 60)
    
    # Имитация workflow из main.py
    import hosted_pbx
    
    # Получение истории звонков
    calls = hosted_pbx.get_call_history()
    
    if calls['error']:
        print(f"Ошибка: {calls['error']}")
        return
    
    # Берем последний звонок
    last_call = calls['info'][-1]
    
    # Скачивание записи (в реальности здесь будет STT)
    # hosted_pbx.download_recording(last_call['record'], 'recording.mp3')
    # transcription = stt.transcribe('recording.mp3')
    
    # Имитация транскрипции
    transcription = "эээ здравствуйте да я хотел бы узнать про статус заказа"
    
    # RAG обработка
    corrector = CallCorrector()
    result = corrector.process_call(
        transcription=transcription,
        call_metadata=last_call
    )
    
    print("\nРезультат обработки последнего звонка:")
    print(f"ID звонка: {last_call.get('id')}")
    print(f"Клиент: {last_call.get('client')}")
    print(f"Исправленная транскрипция: {result['corrected_transcription']}")
    print(f"Классификация: {result.get('classification', {}).get('type')}")
    
    # Здесь можно сохранить результат в БД или отправить в Telegram
    # save_to_database(result)
    # await Log_in_tg(f"Обработан звонок: {result['summary']['brief']}")


if __name__ == "__main__":
    # Запуск всех примеров
    print("\n" + "🚀 Примеры использования llm_corrector.py" + "\n")
    
    try:
        example_1_basic_usage()
        example_2_only_correction()
        example_3_only_entities()
        example_4_batch_processing()
        example_5_custom_configuration()
        # example_6_integration_with_main()  # Раскомментировать для реальной интеграции
        
        print("\n" + "=" * 60)
        print("✅ Все примеры выполнены!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении примеров: {e}")
        import traceback
        traceback.print_exc()


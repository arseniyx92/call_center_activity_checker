"""
RAG-модуль для коррекции и обогащения транскрипций телефонных звонков.

Модуль использует Retrieval-Augmented Generation для:
- Исправления ошибок распознавания речи
- Извлечения сущностей (имена, телефоны, заказы)
- Классификации звонков
- Резюмирования разговоров
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Импорты для RAG (раскомментировать при установке зависимостей)
# from sentence_transformers import SentenceTransformer
# import chromadb
# from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI


class CallCorrector:
    """
    Класс для коррекции и обогащения транскрипций звонков с использованием RAG.
    """
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        vector_db_path: str = "vector_db",
        use_cache: bool = True
    ):
        """
        Инициализация модуля коррекции.
        
        Args:
            embedding_model: Модель для создания эмбеддингов
            llm_model: Модель LLM для коррекции
            vector_db_path: Путь к векторной БД
            use_cache: Использовать ли кэширование
        """
        self.embedding_model_name = embedding_model or "paraphrase-multilingual-MiniLM-L12-v2"
        self.llm_model = llm_model or "gpt-3.5-turbo"
        self.vector_db_path = vector_db_path
        self.use_cache = use_cache
        
        # Инициализация компонентов (заглушки для начала)
        self.embedding_model = None
        self.vector_db = None
        self.llm = None
        self.cache = {} if use_cache else None
        
        # Инициализация при первом использовании
        # self._initialize_components()
    
    def _initialize_components(self):
        """Инициализация всех компонентов RAG."""
        # TODO: Раскомментировать после установки зависимостей
        # try:
        #     # Инициализация модели эмбеддингов
        #     self.embedding_model = SentenceTransformer(self.embedding_model_name)
        #     
        #     # Инициализация векторной БД
        #     self.vector_db = chromadb.PersistentClient(path=self.vector_db_path)
        #     self.collection = self.vector_db.get_or_create_collection("call_transcriptions")
        #     
        #     # Инициализация LLM
        #     api_key = os.getenv("OPENAI_API_KEY")
        #     if api_key:
        #         self.llm = ChatOpenAI(model_name=self.llm_model, temperature=0.3)
        #     else:
        #         print("⚠️ OPENAI_API_KEY не найден, LLM функции будут недоступны")
        #         
        # except Exception as e:
        #     print(f"⚠️ Ошибка инициализации компонентов: {e}")
        #     print("Работа в режиме заглушек")
        pass
    
    def process_call(
        self,
        transcription: str,
        call_metadata: Dict[str, Any],
        include_similar: bool = True,
        include_entities: bool = True,
        include_classification: bool = True,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Полная обработка звонка: коррекция и обогащение.
        
        Args:
            transcription: Сырая транскрипция из STT
            call_metadata: Метаданные звонка (id, client, duration, etc.)
            include_similar: Включать ли поиск похожих звонков
            include_entities: Извлекать ли сущности
            include_classification: Классифицировать ли звонок
            include_summary: Создавать ли резюме
            
        Returns:
            Словарь с обогащенными данными
        """
        start_time = time.time()
        
        result = {
            "original_transcription": transcription,
            "call_metadata": call_metadata,
            "processing_steps": []
        }
        
        # Этап 1: Поиск похожих звонков (RAG Retrieval)
        similar_calls = []
        if include_similar:
            similar_calls = self._find_similar_calls(transcription, top_k=5)
            result["similar_calls"] = similar_calls
            result["processing_steps"].append("similar_calls_search")
        
        # Этап 2: Коррекция транскрипции
        corrected_transcription = self._correct_transcription(
            transcription=transcription,
            similar_calls=similar_calls,
            call_metadata=call_metadata
        )
        result["corrected_transcription"] = corrected_transcription
        result["processing_steps"].append("correction")
        
        # Этап 3: Извлечение сущностей
        entities = {}
        if include_entities:
            entities = self._extract_entities(corrected_transcription)
            result["entities"] = entities
            result["processing_steps"].append("entity_extraction")
        
        # Этап 4: Классификация
        classification = {}
        if include_classification:
            classification = self._classify_call(
                corrected_transcription,
                call_metadata
            )
            result["classification"] = classification
            result["processing_steps"].append("classification")
        
        # Этап 5: Резюмирование
        summary = {}
        if include_summary:
            summary = self._summarize_call(
                corrected_transcription,
                call_metadata,
                classification
            )
            result["summary"] = summary
            result["processing_steps"].append("summarization")
        
        # Сохранение в векторную БД для будущих поисков
        if self.vector_db:
            self._save_to_vector_db(
                transcription=corrected_transcription,
                call_metadata=call_metadata,
                entities=entities,
                classification=classification
            )
        
        # Метаданные обработки
        result["metadata"] = {
            "processing_time": round(time.time() - start_time, 2),
            "model_version": self.llm_model,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        return result
    
    def _find_similar_calls(
        self,
        transcription: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Поиск похожих звонков в векторной БД (RAG Retrieval).
        
        Args:
            transcription: Транскрипция для поиска
            top_k: Количество похожих звонков
            similarity_threshold: Минимальный порог схожести
            
        Returns:
            Список похожих звонков с метаданными
        """
        if not self.embedding_model or not self.vector_db:
            # Заглушка: возвращаем пустой список
            print("⚠️ Векторная БД не инициализирована, похожие звонки не найдены")
            return []
        
        try:
            # Создание эмбеддинга для текущей транскрипции
            query_embedding = self.embedding_model.encode(transcription)
            
            # Поиск в векторной БД
            # TODO: Реализовать поиск через Chroma/Qdrant
            # results = self.collection.query(
            #     query_embeddings=[query_embedding.tolist()],
            #     n_results=top_k
            # )
            
            # Фильтрация по порогу схожести
            similar_calls = []
            # for i, distance in enumerate(results['distances'][0]):
            #     if 1 - distance >= similarity_threshold:  # Преобразуем расстояние в схожесть
            #         similar_calls.append({
            #             "call_id": results['ids'][0][i],
            #             "similarity": 1 - distance,
            #             "transcription": results['documents'][0][i],
            #             "metadata": results['metadatas'][0][i]
            #         })
            
            return similar_calls
            
        except Exception as e:
            print(f"❌ Ошибка поиска похожих звонков: {e}")
            return []
    
    def _correct_transcription(
        self,
        transcription: str,
        similar_calls: List[Dict[str, Any]],
        call_metadata: Dict[str, Any]
    ) -> str:
        """
        Коррекция транскрипции с использованием LLM и контекста из похожих звонков.
        
        Args:
            transcription: Сырая транскрипция
            similar_calls: Список похожих звонков для контекста
            call_metadata: Метаданные звонка
            
        Returns:
            Исправленная транскрипция
        """
        if not self.llm:
            # Заглушка: возвращаем исходный текст
            print("⚠️ LLM не инициализирован, коррекция не выполнена")
            return transcription
        
        # Проверка кэша
        cache_key = f"correction_{hash(transcription)}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Подготовка контекста из похожих звонков
            similar_context = ""
            if similar_calls:
                similar_context = "\n\nПохожие звонки для контекста:\n"
                for i, call in enumerate(similar_calls[:3], 1):
                    similar_context += f"{i}. {call.get('transcription', '')[:200]}...\n"
            
            # Подготовка промпта
            prompt = f"""Ты - эксперт по коррекции транскрипций телефонных звонков.

Исходная транскрипция:
{transcription}
{similar_context}

Задача:
1. Исправь все ошибки распознавания речи
2. Приведи термины к правильному написанию
3. Исправь имена собственные (заглавные буквы)
4. Нормализуй числа и даты
5. Удали паразитные слова ("эээ", "ммм", повторы)
6. Сохрани стиль разговорной речи
7. Не меняй смысл и структуру диалога

Исправленная транскрипция:"""
            
            # Вызов LLM
            # TODO: Реализовать вызов через LangChain
            # response = self.llm.predict(prompt)
            # corrected = response.strip()
            
            # Заглушка
            corrected = transcription
            
            # Сохранение в кэш
            if self.cache:
                self.cache[cache_key] = corrected
            
            return corrected
            
        except Exception as e:
            print(f"❌ Ошибка коррекции транскрипции: {e}")
            return transcription
    
    def _extract_entities(self, transcription: str) -> Dict[str, Any]:
        """
        Извлечение сущностей из транскрипции (NER).
        
        Args:
            transcription: Исправленная транскрипция
            
        Returns:
            Словарь с извлеченными сущностями
        """
        if not self.llm:
            print("⚠️ LLM не инициализирован, извлечение сущностей не выполнено")
            return {}
        
        try:
            prompt = f"""Извлеки из следующего текста все сущности:

Текст:
{transcription}

Извлеки:
1. Имена людей (клиенты, сотрудники)
2. Телефоны и email
3. Номера заказов, договоров, счетов
4. Суммы денег
5. Даты и время
6. Адреса

Формат ответа: JSON
{{
  "persons": [],
  "contacts": {{"phones": [], "emails": []}},
  "orders": [],
  "amounts": [],
  "dates": [],
  "addresses": []
}}"""
            
            # TODO: Реализовать вызов LLM
            # response = self.llm.predict(prompt)
            # entities = json.loads(response)
            
            # Заглушка
            entities = {
                "persons": [],
                "contacts": {"phones": [], "emails": []},
                "orders": [],
                "amounts": [],
                "dates": [],
                "addresses": []
            }
            
            return entities
            
        except Exception as e:
            print(f"❌ Ошибка извлечения сущностей: {e}")
            return {}
    
    def _classify_call(
        self,
        transcription: str,
        call_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Классификация звонка по типу, тематике, эмоциям и результату.
        
        Args:
            transcription: Исправленная транскрипция
            call_metadata: Метаданные звонка
            
        Returns:
            Словарь с классификацией
        """
        if not self.llm:
            print("⚠️ LLM не инициализирован, классификация не выполнена")
            return {}
        
        try:
            prompt = f"""Классифицируй следующий звонок:

Транскрипция:
{transcription}

Метаданные:
- Клиент: {call_metadata.get('client', 'неизвестно')}
- Длительность: {call_metadata.get('duration', 0)} сек
- Тип: {call_metadata.get('type', 'неизвестно')}

Определи:
1. Тип обращения (жалоба/вопрос/заказ/возврат/поддержка/другое)
2. Тематика (доставка/оплата/качество/гарантия/возврат/статус/другое)
3. Эмоциональная окраска (позитивная/нейтральная/негативная/очень негативная)
4. Результат (решено/не решено/отложено/передано)

Ответ в формате JSON:
{{
  "type": "...",
  "topic": "...",
  "sentiment": "...",
  "result": "...",
  "confidence": 0.95
}}"""
            
            # TODO: Реализовать вызов LLM
            # response = self.llm.predict(prompt)
            # classification = json.loads(response)
            
            # Заглушка
            classification = {
                "type": "вопрос",
                "topic": "другое",
                "sentiment": "нейтральная",
                "result": "решено",
                "confidence": 0.8
            }
            
            return classification
            
        except Exception as e:
            print(f"❌ Ошибка классификации: {e}")
            return {}
    
    def _summarize_call(
        self,
        transcription: str,
        call_metadata: Dict[str, Any],
        classification: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Создание резюме звонка.
        
        Args:
            transcription: Исправленная транскрипция
            call_metadata: Метаданные звонка
            classification: Результаты классификации
            
        Returns:
            Словарь с резюме
        """
        if not self.llm:
            print("⚠️ LLM не инициализирован, резюме не создано")
            return {}
        
        try:
            prompt = f"""Создай краткое резюме следующего звонка:

Транскрипция:
{transcription}

Метаданные:
- Клиент: {call_metadata.get('client', 'неизвестно')}
- Длительность: {call_metadata.get('duration', 0)} сек
- Классификация: {json.dumps(classification, ensure_ascii=False)}

Создай резюме в следующем формате:

КРАТКОЕ СОДЕРЖАНИЕ:
[1-2 предложения о сути звонка]

ПРОБЛЕМА КЛИЕНТА:
[Описание проблемы, если есть]

ДЕЙСТВИЯ ОПЕРАТОРА:
[Что было сделано]

РЕЗУЛЬТАТ:
[Итог разговора]

СЛЕДУЮЩИЕ ШАГИ:
[Что нужно сделать дальше, если есть]"""
            
            # TODO: Реализовать вызов LLM
            # response = self.llm.predict(prompt)
            # summary = self._parse_summary(response)
            
            # Заглушка
            summary = {
                "brief": "Клиент обратился с вопросом",
                "problem": "Не указана",
                "actions": "Оператор предоставил информацию",
                "result": "Вопрос решен",
                "next_steps": "Нет"
            }
            
            return summary
            
        except Exception as e:
            print(f"❌ Ошибка создания резюме: {e}")
            return {}
    
    def _parse_summary(self, summary_text: str) -> Dict[str, str]:
        """
        Парсинг резюме из текстового формата в структурированный.
        
        Args:
            summary_text: Текст резюме от LLM
            
        Returns:
            Словарь с полями резюме
        """
        summary = {
            "brief": "",
            "problem": "",
            "actions": "",
            "result": "",
            "next_steps": ""
        }
        
        # Простой парсинг по ключевым словам
        lines = summary_text.split('\n')
        current_key = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if 'КРАТКОЕ СОДЕРЖАНИЕ' in line.upper():
                current_key = 'brief'
            elif 'ПРОБЛЕМА' in line.upper():
                current_key = 'problem'
            elif 'ДЕЙСТВИЯ' in line.upper():
                current_key = 'actions'
            elif 'РЕЗУЛЬТАТ' in line.upper():
                current_key = 'result'
            elif 'СЛЕДУЮЩИЕ ШАГИ' in line.upper():
                current_key = 'next_steps'
            elif current_key and not line.startswith('[') and not line.startswith('['):
                summary[current_key] += line + " "
        
        # Очистка
        for key in summary:
            summary[key] = summary[key].strip()
        
        return summary
    
    def _save_to_vector_db(
        self,
        transcription: str,
        call_metadata: Dict[str, Any],
        entities: Dict[str, Any],
        classification: Dict[str, Any]
    ):
        """
        Сохранение обработанного звонка в векторную БД для будущих поисков.
        
        Args:
            transcription: Исправленная транскрипция
            call_metadata: Метаданные звонка
            entities: Извлеченные сущности
            classification: Классификация
        """
        if not self.vector_db or not self.embedding_model:
            return
        
        try:
            # Создание эмбеддинга
            embedding = self.embedding_model.encode(transcription)
            
            # Подготовка метаданных
            metadata = {
                **call_metadata,
                "classification": json.dumps(classification, ensure_ascii=False),
                "entities_count": len(entities.get('persons', [])) + len(entities.get('orders', []))
            }
            
            # Сохранение в БД
            # TODO: Реализовать сохранение через Chroma/Qdrant
            # self.collection.add(
            #     ids=[call_metadata.get('id', 'unknown')],
            #     embeddings=[embedding.tolist()],
            #     documents=[transcription],
            #     metadatas=[metadata]
            # )
            
        except Exception as e:
            print(f"❌ Ошибка сохранения в векторную БД: {e}")
    
    def correct_text(
        self,
        text: str,
        context: Optional[List[str]] = None
    ) -> str:
        """
        Упрощенная функция только для коррекции текста.
        
        Args:
            text: Текст для коррекции
            context: Дополнительный контекст (опционально)
            
        Returns:
            Исправленный текст
        """
        similar_calls = []
        if context:
            similar_calls = [{"transcription": ctx} for ctx in context]
        
        return self._correct_transcription(
            transcription=text,
            similar_calls=similar_calls,
            call_metadata={}
        )
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Упрощенная функция только для извлечения сущностей.
        
        Args:
            text: Текст для обработки
            
        Returns:
            Извлеченные сущности
        """
        return self._extract_entities(text)


# Пример использования
if __name__ == "__main__":
    # Инициализация
    corrector = CallCorrector()
    
    # Пример данных
    test_transcription = "эээ здравствуйте да я хотел бы узнать про возврат товара номер заказа орд 12345"
    test_metadata = {
        "id": "test_call_1",
        "client": "+79001234567",
        "duration": 120,
        "type": "in"
    }
    
    # Обработка
    result = corrector.process_call(
        transcription=test_transcription,
        call_metadata=test_metadata
    )
    
    # Вывод результатов
    print("Исходная транскрипция:", result['original_transcription'])
    print("Исправленная транскрипция:", result['corrected_transcription'])
    print("Извлеченные сущности:", json.dumps(result.get('entities', {}), ensure_ascii=False, indent=2))
    print("Классификация:", json.dumps(result.get('classification', {}), ensure_ascii=False, indent=2))
    print("Резюме:", json.dumps(result.get('summary', {}), ensure_ascii=False, indent=2))


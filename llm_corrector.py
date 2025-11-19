"""
Модуль для коррекции и обогащения транскрипций телефонных звонков в медицинском call-центре.

Модуль использует LLM и RAG для:
- Исправления ошибок распознавания речи
- Извлечения информации о записи к врачу (врач, специальность, дата/время) с использованием RAG из Google Sheets
- Проверки доступности врача через Google Sheets (RAG с таблицей)
- Классификации звонков
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from openai import OpenAI

# Google Sheets интеграция
try:
    from google_sheets import DoctorsSchedule
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False
    print("⚠️ Модуль google_sheets недоступен")

# Загрузка переменных окружения
load_dotenv()


class CallCorrector:
    """
    Класс для коррекции и обогащения транскрипций звонков с проверкой врачей через Google Sheets.
    """
    
    def __init__(
        self,
        llm_model: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Инициализация модуля коррекции.
        
        Args:
            llm_model: Модель LLM для коррекции
            use_cache: Использовать ли кэширование
        """
        self.llm_model = llm_model or "gpt-3.5-turbo"
        self.use_cache = use_cache
        
        # Инициализация компонентов
        self.llm = None
        self.cache = {} if use_cache else None
        
        # Google Sheets для проверки врачей
        self.doctors_schedule = None
        
        # Инициализация при создании экземпляра
        self._initialize_components()
    
    def _initialize_components(self):
        """Инициализация всех компонентов."""
        try:
            # Инициализация OpenAI клиента
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm = OpenAI(api_key=api_key)
                print(f"✅ LLM инициализирован: {self.llm_model}")
            else:
                print("⚠️ OPENAI_API_KEY не найден, LLM функции будут недоступны")
                
        except Exception as e:
            print(f"⚠️ Ошибка инициализации LLM: {e}")
            self.llm = None
        
        # Инициализация Google Sheets
        if GOOGLE_SHEETS_AVAILABLE:
            try:
                self.doctors_schedule = DoctorsSchedule()
                if self.doctors_schedule.doctors_worksheet:
                    print("✅ Подключение к Google Sheets установлено")
            except Exception as e:
                print(f"⚠️ Ошибка подключения к Google Sheets: {e}")
    
    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Вызов LLM для генерации ответа.
        
        Args:
            prompt: Промпт для LLM
            temperature: Температура генерации (0.0-2.0)
            
        Returns:
            Ответ от LLM
        """
        if not self.llm:
            return ""
        
        try:
            response = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "Ты - эксперт по обработке транскрипций телефонных звонков в медицинском call-центре для записи на прием к врачу."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Ошибка вызова LLM: {e}")
            return ""
    
    def process_call(
        self,
        transcription: str,
        call_metadata: Dict[str, Any],
        include_entities: bool = True,
        include_classification: bool = True,
        verify_doctor: bool = True
    ) -> Dict[str, Any]:
        """
        Полная обработка звонка: коррекция и обогащение с проверкой врача.
        
        Args:
            transcription: Сырая транскрипция из STT
            call_metadata: Метаданные звонка (id, client, duration, etc.)
            include_entities: Извлекать ли сущности
            include_classification: Классифицировать ли звонок
            verify_doctor: Проверять ли врача в Google Sheets
            
        Returns:
            Словарь с обогащенными данными
        """
        start_time = time.time()
        
        result = {
            "original_transcription": transcription,
            "call_metadata": call_metadata,
            "processing_steps": []
        }
        
        # Этап 1: Коррекция транскрипции
        corrected_transcription = self._correct_transcription(
            transcription=transcription,
            call_metadata=call_metadata
        )
        result["corrected_transcription"] = corrected_transcription
        result["processing_steps"].append("correction")
        
        # Этап 2: Извлечение информации о записи к врачу (с RAG)
        appointment_info = {}
        if include_entities:
            appointment_info = self._extract_appointment_info(corrected_transcription)
            result["appointment_info"] = appointment_info
            result["processing_steps"].append("appointment_extraction_with_rag")
            
            # Этап 2.1: Проверка врача в Google Sheets (RAG с таблицей)
            if verify_doctor and self.doctors_schedule:
                doctor_verification = self._verify_doctor_availability(
                    appointment_info,
                    corrected_transcription
                )
                result["doctor_verification"] = doctor_verification
                result["processing_steps"].append("doctor_verification")
        
        # Этап 3: Классификация
        classification = {}
        if include_classification:
            classification = self._classify_call(
                corrected_transcription,
                call_metadata,
                appointment_info
            )
            result["classification"] = classification
            result["processing_steps"].append("classification")
        
        # Метаданные обработки
        result["metadata"] = {
            "processing_time": round(time.time() - start_time, 2),
            "model_version": self.llm_model,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        return result
    
    def _correct_transcription(
        self,
        transcription: str,
        call_metadata: Dict[str, Any]
    ) -> str:
        """
        Коррекция транскрипции с использованием LLM.
        
        Args:
            transcription: Сырая транскрипция
            call_metadata: Метаданные звонка
            
        Returns:
            Исправленная транскрипция
        """
        if not self.llm:
            print("⚠️ LLM не инициализирован, коррекция не выполнена")
            return transcription
        
        # Проверка кэша
        cache_key = f"correction_{hash(transcription)}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Подготовка промпта
            prompt = f"""Ты - эксперт по коррекции транскрипций телефонных звонков в медицинском call-центре.

Исходная транскрипция:
{transcription}

Задача:
1. Исправь все ошибки распознавания речи
2. Приведи медицинские термины к правильному написанию
3. Исправь имена врачей и пациентов (заглавные буквы)
4. Нормализуй даты и время записи
5. Удали паразитные слова ("эээ", "ммм", повторы)
6. Сохрани стиль разговорной речи
7. Не меняй смысл и структуру диалога

Верни ТОЛЬКО исправленную транскрипцию без дополнительных комментариев:"""
            
            # Вызов LLM
            corrected = self._call_llm(prompt, temperature=0.2)
            
            # Если LLM не вернул результат, возвращаем исходный текст
            if not corrected:
                corrected = transcription
            
            # Сохранение в кэш
            if self.cache:
                self.cache[cache_key] = corrected
            
            return corrected
            
        except Exception as e:
            print(f"❌ Ошибка коррекции транскрипции: {e}")
            return transcription
    
    def _extract_appointment_info(self, transcription: str) -> Dict[str, Any]:
        """
        Извлечение информации о записи к врачу из транскрипции с использованием RAG.
        
        Args:
            transcription: Исправленная транскрипция
            
        Returns:
            Словарь с информацией о записи
        """
        if not self.llm:
            print("⚠️ LLM не инициализирован, извлечение информации о записи не выполнено")
            return {}
        
        try:
            # RAG: Получаем контекст о врачах из Google Sheets
            doctors_context = ""
            if self.doctors_schedule:
                doctors_context = self.doctors_schedule.get_context_for_rag()
            
            # Подготовка промпта с RAG контекстом
            if doctors_context and doctors_context != "База данных врачей недоступна":
                prompt = f"""Извлеки из следующего текста информацию о записи на прием к врачу.

Доступные врачи в базе данных:
{doctors_context}

Текст разговора:
{transcription}

Задача:
1. ФИО врача - извлеки точное имя из текста, если упоминается, используй контекст из базы данных для уточнения
2. Специальность врача - определи специальность (терапевт, кардиолог, стоматолог и т.д.), используй контекст из базы для проверки
3. Дата записи - день недели или конкретная дата (если упоминается)
4. Время записи - нормализуй в формат "HH" или "HH:MM" (например, "два часа" → "14", "14:00" → "14")
5. Имя пациента - если упоминается
6. Телефон пациента - если упоминается
7. Жалоба или причина обращения - кратко

Важно: Используй контекст из базы данных для правильного определения имени и специальности врача.

Верни ТОЛЬКО валидный JSON без дополнительного текста:
{{
  "doctor_name": "",
  "doctor_specialty": "",
  "appointment_date": "",
  "appointment_time": "",
  "patient_name": "",
  "patient_phone": "",
  "reason": ""
}}"""
            else:
                # Если Google Sheets недоступен, используем промпт без контекста
                prompt = f"""Извлеки из следующего текста информацию о записи на прием к врачу:

Текст:
{transcription}

Извлеки:
1. ФИО врача (если упоминается)
2. Специальность врача (терапевт, кардиолог, стоматолог и т.д.)
3. Дата записи (день недели или конкретная дата)
4. Время записи (например, "14:00", "два часа дня") - нормализуй в формат "HH"
5. Имя пациента (если упоминается)
6. Телефон пациента
7. Жалоба или причина обращения (кратко)

Верни ТОЛЬКО валидный JSON без дополнительного текста:
{{
  "doctor_name": "",
  "doctor_specialty": "",
  "appointment_date": "",
  "appointment_time": "",
  "patient_name": "",
  "patient_phone": "",
  "reason": ""
}}"""
            
            # Вызов LLM
            response = self._call_llm(prompt, temperature=0.1)
            
            # Парсинг JSON ответа
            if response:
                try:
                    # Попытка извлечь JSON из ответа
                    if "```json" in response:
                        response = response.split("```json")[1].split("```")[0].strip()
                    elif "```" in response:
                        response = response.split("```")[1].split("```")[0].strip()
                    appointment_info = json.loads(response)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Ошибка парсинга JSON информации о записи: {e}")
                    appointment_info = {
                        "doctor_name": "",
                        "doctor_specialty": "",
                        "appointment_date": "",
                        "appointment_time": "",
                        "patient_name": "",
                        "patient_phone": "",
                        "reason": ""
                    }
            else:
                appointment_info = {
                    "doctor_name": "",
                    "doctor_specialty": "",
                    "appointment_date": "",
                    "appointment_time": "",
                    "patient_name": "",
                    "patient_phone": "",
                    "reason": ""
                }
            
            return appointment_info
            
        except Exception as e:
            print(f"❌ Ошибка извлечения информации о записи: {e}")
            return {}
    
    def _verify_doctor_availability(
        self,
        appointment_info: Dict[str, Any],
        transcription: str
    ) -> Dict[str, Any]:
        """
        Проверить доступность врача через Google Sheets (RAG с таблицей).
        
        Args:
            appointment_info: Извлеченная информация о записи
            transcription: Транскрипция для дополнительного контекста
            
        Returns:
            Результат проверки доступности врача
        """
        if not self.doctors_schedule:
            return {
                "verified": False,
                "message": "База данных врачей недоступна",
                "doctor_info": None
            }
        
        doctor_name = appointment_info.get("doctor_name", "").strip()
        specialty = appointment_info.get("doctor_specialty", "").strip()
        appointment_date = appointment_info.get("appointment_date", "").strip()
        appointment_time = appointment_info.get("appointment_time", "").strip()
        
        if not doctor_name:
            return {
                "verified": False,
                "message": "Имя врача не указано в звонке",
                "doctor_info": None
            }
        
        # Получаем контекст из Google Sheets для RAG
        doctors_context = self.doctors_schedule.get_context_for_rag(
            doctor_name=doctor_name if doctor_name else None,
            specialty=specialty if specialty else None
        )
        
        # Проверка доступности через Google Sheets API
        verification_result = self.doctors_schedule.check_doctor_availability(
            doctor_name=doctor_name,
            specialty=specialty if specialty else None,
            day=appointment_date if appointment_date else None,
            time_slot=appointment_time if appointment_time else None
        )
        
        # Обогащаем результат контекстом из таблицы
        result = {
            "verified": (
                verification_result.get("doctor_exists", False) and
                verification_result.get("specialty_matches", True) and  # True если не указана
                verification_result.get("available_at_time", False)
            ),
            "doctor_exists": verification_result.get("doctor_exists", False),
            "specialty_matches": verification_result.get("specialty_matches", True),
            "available_at_time": verification_result.get("available_at_time", False),
            "message": verification_result.get("message", ""),
            "doctor_info": verification_result.get("doctor_info"),
            "doctors_context": doctors_context  # RAG контекст для улучшения понимания
        }
        
        # Если есть несоответствие, используем LLM с контекстом для уточнения
        if not result["verified"] and self.llm:
            clarification_prompt = f"""Проверка записи к врачу:

Информация из звонка:
- Врач: {doctor_name}
- Специальность: {specialty or 'не указана'}
- Дата: {appointment_date or 'не указана'}
- Время: {appointment_time or 'не указана'}

Информация из базы данных врачей:
{doctors_context}

Проблема: {result['message']}

Проанализируй ситуацию и предложи:
1. Существует ли такой врач?
2. Правильна ли специальность?
3. Доступен ли врач в указанное время?

Ответ в формате JSON:
{{
  "doctor_exists_alternative": "",
  "suggested_specialty": "",
  "alternative_doctors": [],
  "recommendation": ""
}}"""
            
            clarification = self._call_llm(clarification_prompt, temperature=0.2)
            if clarification:
                try:
                    if "```json" in clarification:
                        clarification = clarification.split("```json")[1].split("```")[0].strip()
                    elif "```" in clarification:
                        clarification = clarification.split("```")[1].split("```")[0].strip()
                    result["llm_clarification"] = json.loads(clarification)
                except (json.JSONDecodeError, ValueError) as e:
                    result["llm_clarification"] = {"recommendation": clarification}
        
        return result
    
    def _classify_call(
        self,
        transcription: str,
        call_metadata: Dict[str, Any],
        appointment_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Классификация звонка в медицинском call-центре.
        
        Args:
            transcription: Исправленная транскрипция
            call_metadata: Метаданные звонка
            appointment_info: Информация о записи
            
        Returns:
            Словарь с классификацией
        """
        if not self.llm:
            print("⚠️ LLM не инициализирован, классификация не выполнена")
            return {}
        
        try:
            prompt = f"""Классифицируй следующий звонок в медицинском call-центре:

Транскрипция:
{transcription}

Метаданные:
- Клиент: {call_metadata.get('client', 'неизвестно')}
- Длительность: {call_metadata.get('duration', 0)} сек
- Тип: {call_metadata.get('type', 'неизвестно')}

Информация о записи:
- Врач: {appointment_info.get('doctor_name', 'не указан')}
- Специальность: {appointment_info.get('doctor_specialty', 'не указана')}
- Дата: {appointment_info.get('appointment_date', 'не указана')}

Определи:
1. Тип обращения (запись_к_врачу/консультация/отмена_записи/перенос/другое)
2. Специальность врача (терапевт/кардиолог/стоматолог/невролог/другое)
3. Эмоциональная окраска (позитивная/нейтральная/негативная/очень негативная)
4. Результат (запись_создана/запись_не_создана/требуется_уточнение/отказ)

Верни ТОЛЬКО валидный JSON без дополнительного текста:
{{
  "type": "...",
  "specialty": "...",
  "sentiment": "...",
  "result": "...",
  "confidence": 0.95
}}"""
            
            # Вызов LLM
            response = self._call_llm(prompt, temperature=0.2)
            
            # Парсинг JSON ответа
            if response:
                try:
                    if "```json" in response:
                        response = response.split("```json")[1].split("```")[0].strip()
                    elif "```" in response:
                        response = response.split("```")[1].split("```")[0].strip()
                    classification = json.loads(response)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Ошибка парсинга JSON классификации: {e}")
                    classification = {
                        "type": "запись_к_врачу",
                        "specialty": "другое",
                        "sentiment": "нейтральная",
                        "result": "требуется_уточнение",
                        "confidence": 0.8
                    }
            else:
                classification = {
                    "type": "запись_к_врачу",
                    "specialty": "другое",
                    "sentiment": "нейтральная",
                    "result": "требуется_уточнение",
                    "confidence": 0.8
                }
            
            return classification
            
        except Exception as e:
            print(f"❌ Ошибка классификации: {e}")
            return {}
    
    def correct_text(self, text: str) -> str:
        """
        Упрощенная функция только для коррекции текста.
        
        Args:
            text: Текст для коррекции
            
        Returns:
            Исправленный текст
        """
        return self._correct_transcription(
            transcription=text,
            call_metadata={}
        )


# Пример использования
if __name__ == "__main__":
    # Инициализация
    corrector = CallCorrector()
    
    # Пример данных
    test_transcription = "эээ здравствуйте да я хотел бы записаться к врачу иванову терапевту на понедельник в два часа дня"
    test_metadata = {
        "id": "test_call_1",
        "client": "+79001234567",
        "duration": 120,
        "type": "in"
    }
    
    # Обработка
    result = corrector.process_call(
        transcription=test_transcription,
        call_metadata=test_metadata,
        verify_doctor=True
    )
    
    # Вывод результатов
    print("Исходная транскрипция:", result['original_transcription'])
    print("Исправленная транскрипция:", result['corrected_transcription'])
    print("\nИнформация о записи:", json.dumps(result.get('appointment_info', {}), ensure_ascii=False, indent=2))
    print("\nПроверка врача:", json.dumps(result.get('doctor_verification', {}), ensure_ascii=False, indent=2))
    print("\nКлассификация:", json.dumps(result.get('classification', {}), ensure_ascii=False, indent=2))

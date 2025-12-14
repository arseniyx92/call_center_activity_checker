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

# Web Scraper интеграция
try:
    from web_scraper import WebScraper
    WEB_SCRAPER_AVAILABLE = True
except ImportError:
    WEB_SCRAPER_AVAILABLE = False
    print("⚠️ Модуль web_scraper недоступен")

# Загрузка переменных окружения
load_dotenv()


class CallCorrector:
    """
    Класс для коррекции и обогащения транскрипций звонков с проверкой врачей через Google Sheets.
    """
    
    def __init__(
        self,
        llm_model: Optional[str] = None,
    ):
        """
        Инициализация модуля коррекции.
        
        Args:
            llm_model: Модель LLM для коррекции
            use_cache: Использовать ли кэширование
        """
        self.llm_model = llm_model or "gpt-4.5"
        
        # Инициализация компонентов
        self.llm = None
        
        # Google Sheets для проверки врачей
        self.doctors_schedule = None
        
        # Web Scraper для получения информации с сайта
        self.web_scraper = None
        
        # Инициализация при создании экземпляра
        self._initialize_components()
    
    def _initialize_components(self):
        """Инициализация всех компонентов."""
        # Инициализация OpenAI клиента
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.llm = OpenAI(api_key=api_key)
            print(f"✅ LLM инициализирован: {self.llm_model}")
        
        # Инициализация Google Sheets
        if GOOGLE_SHEETS_AVAILABLE:
            self.doctors_schedule = DoctorsSchedule()
        
        # Инициализация Web Scraper
        if WEB_SCRAPER_AVAILABLE:
            base_url = os.getenv('WEBSITE_BASE_URL')
            if base_url:
                self.web_scraper = WebScraper(base_url=base_url)
    
    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Вызов LLM для генерации ответа.
        
        Args:
            prompt: Промпт для LLM
            temperature: Температура генерации (0.0-2.0)
            
        Returns:
            Ответ от LLM
        """
        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "Ты - эксперт по обработке транскрипций телефонных звонков в медицинском call-центре для записи на прием к врачу."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    
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
        
        # # Этап 1.1: Форматирование диалога (разбивка по репликам)
        # formatted_transcription = self._format_as_dialogue(corrected_transcription)
        # result["formatted_transcription"] = formatted_transcription
        # result["processing_steps"].append("dialogue_formatting")
        
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
        prompt = f"""Ты - эксперт по коррекции транскрипций телефонных звонков в медицинском call-центре.

Тебе пришла транскрипция звонка после обработки речи с телефонной аудио-записи разговора оператора сплошным текстом.
Транскрипция звонка полна опечаток и семантических ошибок, ты должен привести ее к содержательному диалогу без ошибок.
Тебе необходимо исправить все медицинские термины, все русские слова.
Разумно преобразовать текст в диалог из двух говорящих: оператора и клиента, чтобы получился осмысленный разговор.

Исходная плохая транскрипция:
{transcription}

Верни ТОЛЬКО исправленную транскрипцию без дополнительных комментариев:"""
        
        corrected = self._call_llm(prompt, temperature=0.2)
        return corrected or transcription
    
    def _format_as_dialogue(self, transcription: str) -> str:
        """
        Разбить транскрипцию на диалог: каждая реплика с новой строки и дефисом.
        
        Args:
            transcription: Исправленная транскрипция
            
        Returns:
            Транскрипция, отформатированная как диалог
        """
        if not self.llm:
            return transcription
        
        prompt = f"""Разбей следующий текст телефонного разговора на диалог.

Текст:
{transcription}

Задача:
1. Определи границы каждой реплики (говорящего)
2. Каждая реплика должна начинаться с новой строки
3. Каждая реплика должна начинаться с дефиса и пробела: "- "
4. Сохрани все содержимое без изменений
5. Если текст уже содержит реплики с дефисами, улучши разбивку если нужно

Пример формата:
- Реплика 1
- Реплика 2
- Реплика 3

Верни ТОЛЬКО отформатированный диалог без дополнительных комментариев:"""
        
        formatted = self._call_llm(prompt, temperature=0.1)
        return formatted or transcription
    
    def _extract_appointment_info(self, transcription: str) -> Dict[str, Any]:
        """
        Извлечение информации о записи к врачу из транскрипции с использованием RAG.
        
        Args:
            transcription: Исправленная транскрипция
            
        Returns:
            Словарь с информацией о записи
        """
        # RAG: Получаем контекст о врачах и расписании из Google Sheets
        doctors_context = self.doctors_schedule.get_context_for_rag() if self.doctors_schedule else ""
        
        # RAG: Получаем контекст с веб-сайта (только телефоны, адреса, услуги)
        website_context = ""
        if self.web_scraper:
            scrape_url = os.getenv('WEBSITE_DOCTORS_PAGE', '/')
            website_context = self.web_scraper.get_context_for_rag(
                url=scrape_url,
                keywords=["услуг", "контакт", "адрес", "телефон"],
                max_length=1500,
                include_doctors=False,
                include_services=True,
                include_contacts=True
            )
        
        # Объединяем контексты
        rag_context_parts = []
        if doctors_context and doctors_context != "База данных врачей недоступна":
            rag_context_parts.append(f"База данных врачей и расписание (Google Sheets):\n{doctors_context}\n")
        if website_context and website_context not in ["Информация с сайта недоступна", "URL не указан"]:
            rag_context_parts.append(f"Контакты и услуги клиники (с сайта):\n{website_context}\n")
        
        combined_context = "\n".join(rag_context_parts)
        
        # Подготовка промпта с RAG контекстом
        if combined_context:
            prompt = f"""Извлеки из следующего текста информацию о записи на прием к врачу.

{combined_context}
Текст разговора:
{transcription}

Задача:
1. ФИО врача - извлеки точное имя из текста, если упоминается, используй контекст из базы данных (Google Sheets) для уточнения
2. Специальность врача - определи специальность (терапевт, кардиолог, стоматолог и т.д.), используй контекст из базы данных для проверки
3. Дата записи - день недели или конкретная дата (если упоминается)
4. Время записи - нормализуй в формат "HH" или "HH:MM" (например, "два часа" → "14", "14:00" → "14")
5. Имя пациента - если упоминается
6. Телефон пациента - если упоминается
7. Услуга - определи какую услугу запрашивает клиент, используй список услуг с сайта для уточнения
8. Жалоба или причина обращения - кратко

Важно: 
- Используй контекст из Google Sheets для правильного определения имени и специальности врача, а также для проверки расписания
- Используй информацию с сайта (телефоны, адреса, услуги) для валидации и уточнения данных из разговора

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
        
        response = self._call_llm(prompt, temperature=0.1)
        
        # Парсинг JSON ответа
        if not response:
            return {
                "doctor_name": "", "doctor_specialty": "", "appointment_date": "",
                "appointment_time": "", "patient_name": "", "patient_phone": "", "reason": ""
            }
        
        # Извлечение JSON из ответа
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        appointment_info = json.loads(response)
        return appointment_info
    
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
        
        doctors_context = self.doctors_schedule.get_context_for_rag(
            doctor_name=doctor_name or None,
            specialty=specialty or None
        )
        
        verification_result = self.doctors_schedule.check_doctor_availability(
            doctor_name=doctor_name,
            specialty=specialty or None,
            day=appointment_date or None,
            time_slot=appointment_time or None
        )
        
        result = {
            "verified": (
                verification_result.get("doctor_exists", False) and
                verification_result.get("specialty_matches", True) and
                verification_result.get("available_at_time", False)
            ),
            "doctor_exists": verification_result.get("doctor_exists", False),
            "specialty_matches": verification_result.get("specialty_matches", True),
            "available_at_time": verification_result.get("available_at_time", False),
            "message": verification_result.get("message", ""),
            "doctor_info": verification_result.get("doctor_info"),
            "doctors_context": doctors_context
        }
        
        # LLM уточнение при несоответствии
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
                if "```json" in clarification:
                    clarification = clarification.split("```json")[1].split("```")[0].strip()
                elif "```" in clarification:
                    clarification = clarification.split("```")[1].split("```")[0].strip()
                result["llm_clarification"] = json.loads(clarification)
        
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
        
        response = self._call_llm(prompt, temperature=0.2)
        
        if not response:
            return {
                "type": "запись_к_врачу",
                "specialty": "другое",
                "sentiment": "нейтральная",
                "result": "требуется_уточнение",
                "confidence": 0.8
            }
        
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        classification = json.loads(response)
        return classification
    
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

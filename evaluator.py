"""
Модуль для оценки качества работы LLM компонентов системы.

Использует подходы из LangChain:
- Response vs retrieved docs (для RAG участков) - проверка на галлюцинации
- Response vs reference answer (для всех участков) - проверка точности
"""

import os
import json
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class Evaluator:
    """
    Класс для оценки качества работы компонентов системы.
    """
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        """
        Инициализация оценщика.
        
        Args:
            llm_model: Модель LLM для оценки
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY не найден в переменных окружения")
        
        self.llm = OpenAI(api_key=api_key)
        self.llm_model = llm_model
    
    def _call_llm(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Вызов LLM для оценки.
        
        Args:
            prompt: Промпт для оценки
            temperature: Температура (0.0 для детерминированных оценок)
            
        Returns:
            Ответ от LLM
        """
        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "Ты - эксперт по оценке качества работы LLM систем в медицинском call-центре."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    
    def evaluate_response_vs_retrieved_docs(
        self,
        question: str,
        retrieved_docs: str,
        response: str
    ) -> Dict[str, Any]:
        """
        Оценка соответствия ответа извлеченным документам (RAG hallucination check).
        
        Проверяет, не содержит ли ответ информацию, которой нет в извлеченных документах.
        
        Args:
            question: Исходный вопрос/запрос
            retrieved_docs: Извлеченные документы/контекст для RAG
            response: Ответ модели
            
        Returns:
            Словарь с оценкой и метриками
        """
        prompt = f"""Ты - учитель, оценивающий ответ студента на экзамене.

Вам дан:
1. ВОПРОС: {question}

2. ФАКТЫ (извлеченные документы/контекст):
{retrieved_docs}

3. ОТВЕТ СТУДЕНТА (ответ LLM):
{response}

Ваша задача:
Определить, содержит ли ответ студента информацию, которой НЕТ в фактах (извлеченных документах).

Критерии оценки:
- Оценка 1: Ответ полностью основан на фактах, НЕТ галлюцинаций
- Оценка 0: Ответ содержит информацию, которой нет в фактах (есть галлюцинации)

Важно:
- Если ответ содержит информацию из фактов, но в другой формулировке - это нормально (оценка 1)
- Если ответ содержит новую информацию, которой НЕТ в фактах - это галлюцинация (оценка 0)
- Если факты не содержат ответа на вопрос, но ответ логичен и не противоречит фактам - оценка 1

Верни ТОЛЬКО JSON в формате:
{{
  "score": 1 или 0,
  "reasoning": "краткое объяснение оценки"
}}"""
        
        evaluation_result = self._call_llm(prompt, temperature=0.0)
        
        # Парсинг JSON
        if "```json" in evaluation_result:
            evaluation_result = evaluation_result.split("```json")[1].split("```")[0].strip()
        elif "```" in evaluation_result:
            evaluation_result = evaluation_result.split("```")[1].split("```")[0].strip()
        
        try:
            result = json.loads(evaluation_result)
        except json.JSONDecodeError:
            result = {"score": 0, "reasoning": "Ошибка парсинга оценки", "raw_response": evaluation_result}
        
        return {
            "metric": "response_vs_retrieved_docs",
            "score": result.get("score", 0),
            "reasoning": result.get("reasoning", ""),
            "question": question,
            "retrieved_docs": retrieved_docs[:500],  # Обрезаем для логов
            "response": response[:500]
        }
    
    def evaluate_response_vs_reference(
        self,
        question: str,
        reference_answer: str,
        response: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Оценка соответствия ответа эталонному ответу.
        
        Args:
            question: Исходный вопрос/запрос
            reference_answer: Эталонный ответ
            response: Ответ модели для оценки
            context: Дополнительный контекст (опционально)
            
        Returns:
            Словарь с оценкой и метриками
        """
        context_part = f"\nКонтекст:\n{context}\n" if context else ""
        
        prompt = f"""Ты - учитель, оценивающий ответ студента на экзамене.

Вам дан:
1. ВОПРОС: {question}
{context_part}
2. ЭТАЛОННЫЙ ОТВЕТ (правильный ответ):
{reference_answer}

3. ОТВЕТ СТУДЕНТА (ответ LLM для оценки):
{response}

Ваша задача:
Определить, насколько ответ студента соответствует эталонному ответу.

Критерии оценки:
- Оценка 1: Ответ полностью соответствует эталону или эквивалентен ему
- Оценка 0.75: Ответ в основном правильный, но есть небольшие неточности
- Оценка 0.5: Ответ частично правильный, но есть существенные отличия
- Оценка 0.25: Ответ почти не соответствует эталону
- Оценка 0: Ответ полностью неверный или не соответствует эталону

Важно:
- Учитывайте смысловое соответствие, а не только текстовое
- Разные формулировки, но одинаковый смысл - это хорошо (оценка 1)
- Если эталонный ответ - JSON, проверяйте структуру и значения

Верни ТОЛЬКО JSON в формате:
{{
  "score": число от 0 до 1,
  "reasoning": "краткое объяснение оценки",
  "accuracy": "полностью_правильно" | "в_основном_правильно" | "частично_правильно" | "почти_неправильно" | "неправильно"
}}"""
        
        evaluation_result = self._call_llm(prompt, temperature=0.0)
        
        # Парсинг JSON
        if "```json" in evaluation_result:
            evaluation_result = evaluation_result.split("```json")[1].split("```")[0].strip()
        elif "```" in evaluation_result:
            evaluation_result = evaluation_result.split("```")[1].split("```")[0].strip()
        
        try:
            result = json.loads(evaluation_result)
        except json.JSONDecodeError:
            result = {
                "score": 0,
                "reasoning": "Ошибка парсинга оценки",
                "accuracy": "неправильно",
                "raw_response": evaluation_result
            }
        
        return {
            "metric": "response_vs_reference",
            "score": result.get("score", 0),
            "reasoning": result.get("reasoning", ""),
            "accuracy": result.get("accuracy", "неправильно"),
            "question": question,
            "reference_answer": reference_answer[:500],
            "response": response[:500]
        }
    
    def evaluate_rag_extraction(
        self,
        transcription: str,
        retrieved_docs: str,
        extracted_info: Dict[str, Any],
        reference_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Комплексная оценка RAG извлечения информации о записи.
        
        Объединяет:
        1. Response vs retrieved docs - проверка на галлюцинации
        2. Response vs reference answer - проверка точности
        
        Args:
            transcription: Транскрипция звонка
            retrieved_docs: Извлеченный RAG контекст
            extracted_info: Извлеченная информация (ответ системы)
            reference_info: Эталонная информация (правильный ответ)
            
        Returns:
            Словарь с комплексной оценкой
        """
        # Преобразуем в JSON строки для сравнения
        extracted_json = json.dumps(extracted_info, ensure_ascii=False)
        reference_json = json.dumps(reference_info, ensure_ascii=False)
        
        question = f"Извлеки информацию о записи к врачу из следующего диалога:\n{transcription[:500]}"
        
        # 1. Оценка соответствия извлеченным документам
        hallucination_check = self.evaluate_response_vs_retrieved_docs(
            question=question,
            retrieved_docs=retrieved_docs,
            response=extracted_json
        )
        
        # 2. Оценка соответствия эталонному ответу
        accuracy_check = self.evaluate_response_vs_reference(
            question=question,
            reference_answer=reference_json,
            response=extracted_json,
            context=retrieved_docs
        )
        
        # Комплексная оценка
        overall_score = (hallucination_check["score"] + accuracy_check["score"]) / 2
        
        return {
            "overall_score": overall_score,
            "hallucination_check": hallucination_check,
            "accuracy_check": accuracy_check,
            "transcription": transcription[:200],
            "retrieved_docs": retrieved_docs[:300]
        }
    
    def evaluate_correction(
        self,
        original: str,
        corrected: str,
        reference_corrected: str
    ) -> Dict[str, Any]:
        """
        Оценка коррекции транскрипции.
        
        Args:
            original: Исходная транскрипция
            corrected: Исправленная транскрипция (ответ системы)
            reference_corrected: Эталонная исправленная транскрипция
            
        Returns:
            Словарь с оценкой
        """
        question = f"Исправь транскрипцию телефонного звонка:\n{original[:500]}"
        
        return self.evaluate_response_vs_reference(
            question=question,
            reference_answer=reference_corrected,
            response=corrected
        )
    
    def evaluate_classification(
        self,
        transcription: str,
        classification: Dict[str, Any],
        reference_classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Оценка классификации звонка.
        
        Args:
            transcription: Транскрипция звонка
            classification: Классификация от системы
            reference_classification: Эталонная классификация
            
        Returns:
            Словарь с оценкой
        """
        question = f"Классифицируй звонок:\n{transcription[:500]}"
        
        classification_json = json.dumps(classification, ensure_ascii=False)
        reference_json = json.dumps(reference_classification, ensure_ascii=False)
        
        return self.evaluate_response_vs_reference(
            question=question,
            reference_answer=reference_json,
            response=classification_json
        )
    
    def evaluate_dialogue_formatting(
        self,
        original: str,
        formatted: str,
        reference_formatted: str
    ) -> Dict[str, Any]:
        """
        Оценка форматирования диалога.
        
        Args:
            original: Исходный текст
            formatted: Отформатированный текст (ответ системы)
            reference_formatted: Эталонный отформатированный текст
            
        Returns:
            Словарь с оценкой
        """
        question = f"Разбей текст на диалог (каждая реплика с новой строки и дефисом):\n{original[:500]}"
        
        return self.evaluate_response_vs_reference(
            question=question,
            reference_answer=reference_formatted,
            response=formatted
        )


# Пример использования
if __name__ == "__main__":
    evaluator = Evaluator()
    
    # Пример оценки RAG извлечения
    test_transcription = "Здравствуйте, я хочу записаться к врачу Иванову на завтра в 14:00"
    test_retrieved_docs = "Врачи: Иванов Иван Иванович, терапевт. Расписание: работает с 9 до 18"
    test_extracted = {
        "doctor_name": "Иванов Иван Иванович",
        "doctor_specialty": "терапевт",
        "appointment_date": "завтра",
        "appointment_time": "14"
    }
    test_reference = {
        "doctor_name": "Иванов Иван Иванович",
        "doctor_specialty": "терапевт",
        "appointment_date": "завтра",
        "appointment_time": "14:00"
    }
    
    result = evaluator.evaluate_rag_extraction(
        transcription=test_transcription,
        retrieved_docs=test_retrieved_docs,
        extracted_info=test_extracted,
        reference_info=test_reference
    )
    
    print(json.dumps(result, ensure_ascii=False, indent=2))

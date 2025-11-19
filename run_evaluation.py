"""
Скрипт для запуска оценки компонентов системы.

Оценивает:
1. RAG участки (извлечение информации, проверка врача) - Response vs retrieved docs + Response vs reference
2. Коррекция транскрипции - Response vs reference
3. Форматирование диалога - Response vs reference
4. Классификация - Response vs reference
"""

import os
import json
import asyncio
from typing import Dict, List, Any
from dotenv import load_dotenv
from llm_corrector import CallCorrector
from llm_stt import transcribe_mp3
from hosted_pbx import get_call_history, download_recording
from evaluator import Evaluator

load_dotenv()


class EvaluationRunner:
    """Класс для запуска оценки компонентов системы."""
    
    def __init__(self):
        self.corrector = CallCorrector()
        self.evaluator = Evaluator()
        self.results = []
    
    def get_retrieved_docs_for_evaluation(self, transcription: str) -> str:
        """
        Получить извлеченные документы для оценки RAG.
        
        Args:
            transcription: Транскрипция звонка
            
        Returns:
            Строка с извлеченным контекстом
        """
        # Получаем контекст из Google Sheets
        doctors_context = ""
        if self.corrector.doctors_schedule:
            doctors_context = self.corrector.doctors_schedule.get_context_for_rag()
        
        # Получаем контекст с веб-сайта
        website_context = ""
        if self.corrector.web_scraper:
            scrape_url = os.getenv('WEBSITE_DOCTORS_PAGE', '/')
            website_context = self.corrector.web_scraper.get_context_for_rag(
                url=scrape_url,
                keywords=["услуг", "контакт", "адрес", "телефон"],
                max_length=1500,
                include_doctors=False,
                include_services=True,
                include_contacts=True
            )
        
        # Объединяем контексты
        context_parts = []
        if doctors_context and doctors_context != "База данных врачей недоступна":
            context_parts.append(f"База данных врачей:\n{doctors_context}")
        if website_context and website_context not in ["Информация с сайта недоступна", "URL не указан"]:
            context_parts.append(f"Информация с сайта:\n{website_context}")
        
        return "\n\n".join(context_parts) if context_parts else "Контекст недоступен"
    
    def evaluate_call_processing(
        self,
        transcription: str,
        reference_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Полная оценка обработки звонка.
        
        Args:
            transcription: Сырая транскрипция
            reference_data: Эталонные данные с ключами:
                - corrected_transcription (опционально)
                - formatted_transcription (опционально)
                - appointment_info (опционально)
                - classification (опционально)
            
        Returns:
            Словарь с результатами оценки всех компонентов
        """
        # Обработка звонка
        result = self.corrector.process_call(
            transcription=transcription,
            call_metadata={},
            include_entities=True,
            include_classification=True,
            verify_doctor=True
        )
        
        evaluation_results = {
            "transcription": transcription[:200],
            "evaluations": {}
        }
        
        # 1. Оценка коррекции транскрипции
        if "corrected_transcription" in reference_data and result.get("corrected_transcription"):
            evaluation_results["evaluations"]["correction"] = self.evaluator.evaluate_correction(
                original=transcription,
                corrected=result["corrected_transcription"],
                reference_corrected=reference_data["corrected_transcription"]
            )
        
        # 2. Оценка форматирования диалога
        if "formatted_transcription" in reference_data and result.get("formatted_transcription"):
            evaluation_results["evaluations"]["dialogue_formatting"] = self.evaluator.evaluate_dialogue_formatting(
                original=result.get("corrected_transcription", transcription),
                formatted=result["formatted_transcription"],
                reference_formatted=reference_data["formatted_transcription"]
            )
        
        # 3. Оценка RAG извлечения информации
        if "appointment_info" in reference_data and result.get("appointment_info"):
            retrieved_docs = self.get_retrieved_docs_for_evaluation(transcription)
            
            evaluation_results["evaluations"]["rag_extraction"] = self.evaluator.evaluate_rag_extraction(
                transcription=transcription,
                retrieved_docs=retrieved_docs,
                extracted_info=result["appointment_info"],
                reference_info=reference_data["appointment_info"]
            )
        
        # 4. Оценка классификации
        if "classification" in reference_data and result.get("classification"):
            evaluation_results["evaluations"]["classification"] = self.evaluator.evaluate_classification(
                transcription=result.get("corrected_transcription", transcription),
                classification=result["classification"],
                reference_classification=reference_data["classification"]
            )
        
        return evaluation_results
    
    def evaluate_real_call(self, call_index: int = -2) -> Dict[str, Any]:
        """
        Оценка на реальном звонке (без эталонных данных - только RAG проверка).
        
        Args:
            call_index: Индекс звонка в истории (-2 для предпоследнего)
            
        Returns:
            Результаты оценки RAG компонентов
        """
        # Получаем историю звонков
        calls = get_call_history()
        if calls['error'] is not None:
            return {"error": f"Ошибка получения звонков: {calls['error']}"}
        
        if not calls['info'] or len(calls['info']) < abs(call_index):
            return {"error": "Недостаточно звонков для обработки"}
        
        call = calls['info'][call_index]
        download_link = call.get('record')
        
        if not download_link:
            return {"error": "У звонка нет записи"}
        
        # Загружаем запись
        def get_record_name(call):
            record_name = f"{call['start']}_{call['type']}.mp3"
            return record_name
        
        filename = get_record_name(call)
        recordings_dir = os.getenv('RECORDINGS_DIR', 'recordings')
        
        if not download_recording(download_link, filename):
            return {"error": f"Не удалось загрузить запись: {download_link}"}
        
        # Транскрипция
        transcription = transcribe_mp3(f'{recordings_dir}/{filename}')
        if not transcription:
            return {"error": "Не удалось распознать речь"}
        
        # Обработка
        result = self.corrector.process_call(
            transcription=transcription,
            call_metadata=call,
            include_entities=True,
            include_classification=True,
            verify_doctor=True
        )
        
        # Оценка только RAG компонентов (без эталонных данных)
        retrieved_docs = self.get_retrieved_docs_for_evaluation(transcription)
        
        rag_evaluation = {
            "hallucination_check": None,
            "transcription": transcription[:300]
        }
        
        # Проверка RAG извлечения на галлюцинации
        if result.get("appointment_info"):
            question = f"Извлеки информацию о записи к врачу из диалога:\n{transcription[:500]}"
            
            rag_evaluation["hallucination_check"] = self.evaluator.evaluate_response_vs_retrieved_docs(
                question=question,
                retrieved_docs=retrieved_docs,
                response=json.dumps(result["appointment_info"], ensure_ascii=False)
            )
        
        return {
            "call_id": call.get("id"),
            "result": result,
            "rag_evaluation": rag_evaluation,
            "retrieved_docs": retrieved_docs[:500]
        }


def create_test_dataset() -> List[Dict[str, Any]]:
    """
    Создает тестовый датасет для оценки.
    
    Returns:
        Список тестовых примеров с транскрипциями и эталонными данными
    """
    return [
        {
            "transcription": "Алло, здравствуйте. Я хочу записаться к врачу Иванову на завтра в два часа дня. Меня зовут Петров Петр.",
            "reference_data": {
                "corrected_transcription": "Алло, здравствуйте. Я хочу записаться к врачу Иванову на завтра в 14:00. Меня зовут Петров Петр.",
                "formatted_transcription": "- Алло, здравствуйте. Я хочу записаться к врачу Иванову на завтра в 14:00.\n- Меня зовут Петров Петр.",
                "appointment_info": {
                    "doctor_name": "Иванов",
                    "doctor_specialty": "",
                    "appointment_date": "завтра",
                    "appointment_time": "14",
                    "patient_name": "Петров Петр",
                    "patient_phone": "",
                    "reason": ""
                },
                "classification": {
                    "type": "запись_к_врачу",
                    "specialty": "",
                    "sentiment": "нейтральная",
                    "result": "запись_создана",
                    "confidence": 0.9
                }
            }
        },
        {
            "transcription": "Здравствуйте, мне нужно к терапевту. У меня болит живот. Можно сегодня вечером?",
            "reference_data": {
                "corrected_transcription": "Здравствуйте, мне нужно к терапевту. У меня болит живот. Можно сегодня вечером?",
                "formatted_transcription": "- Здравствуйте, мне нужно к терапевту. У меня болит живот.\n- Можно сегодня вечером?",
                "appointment_info": {
                    "doctor_name": "",
                    "doctor_specialty": "терапевт",
                    "appointment_date": "сегодня",
                    "appointment_time": "",
                    "patient_name": "",
                    "patient_phone": "",
                    "reason": "болит живот"
                },
                "classification": {
                    "type": "запись_к_врачу",
                    "specialty": "терапевт",
                    "sentiment": "нейтральная",
                    "result": "требуется_уточнение",
                    "confidence": 0.85
                }
            }
        }
    ]


def main():
    """Основная функция для запуска оценки."""
    print("🚀 Запуск оценки компонентов системы...\n")
    
    runner = EvaluationRunner()
    
    # Вариант 1: Оценка на тестовом датасете
    print("=" * 80)
    print("Вариант 1: Оценка на тестовом датасете")
    print("=" * 80)
    
    test_dataset = create_test_dataset()
    all_results = []
    
    for i, test_case in enumerate(test_dataset, 1):
        print(f"\n📋 Тест {i}/{len(test_dataset)}")
        print(f"Транскрипция: {test_case['transcription'][:100]}...")
        
        result = runner.evaluate_call_processing(
            transcription=test_case["transcription"],
            reference_data=test_case["reference_data"]
        )
        
        all_results.append(result)
        
        # Выводим результаты
        if "evaluations" in result:
            for component, evaluation in result["evaluations"].items():
                if isinstance(evaluation, dict):
                    score = evaluation.get("score", 0) if "score" in evaluation else evaluation.get("overall_score", 0)
                    print(f"  {component}: score = {score:.2f}")
                elif isinstance(evaluation, dict) and "hallucination_check" in evaluation:
                    score = evaluation["hallucination_check"].get("score", 0)
                    print(f"  {component}: hallucination score = {score:.2f}")
    
    # Вариант 2: Оценка на реальном звонке (только RAG проверка)
    print("\n" + "=" * 80)
    print("Вариант 2: Оценка на реальном звонке (RAG проверка)")
    print("=" * 80)
    
    real_call_result = None
    try:
        real_call_result = runner.evaluate_real_call()
        if "error" in real_call_result:
            print(f"⚠️ {real_call_result['error']}")
        else:
            print(f"✅ Обработан звонок: {real_call_result.get('call_id', 'unknown')}")
            if "rag_evaluation" in real_call_result and real_call_result["rag_evaluation"].get("hallucination_check"):
                score = real_call_result["rag_evaluation"]["hallucination_check"].get("score", 0)
                print(f"  RAG hallucination score: {score:.2f}")
                print(f"  Reasoning: {real_call_result['rag_evaluation']['hallucination_check'].get('reasoning', '')}")
    except Exception as e:
        print(f"❌ Ошибка при оценке реального звонка: {e}")
    
    # Сохраняем результаты
    output_file = "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_results": all_results,
            "real_call_result": real_call_result
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Результаты сохранены в {output_file}")
    
    # Выводим итоговую статистику
    print("\n" + "=" * 80)
    print("Итоговая статистика")
    print("=" * 80)
    
    if all_results:
        total_scores = {"correction": [], "dialogue_formatting": [], "rag_extraction": [], "classification": []}
        
        for result in all_results:
            if "evaluations" in result:
                for component, evaluation in result["evaluations"].items():
                    if isinstance(evaluation, dict):
                        if "overall_score" in evaluation:
                            total_scores[component].append(evaluation["overall_score"])
                        elif "score" in evaluation:
                            total_scores[component].append(evaluation["score"])
        
        for component, scores in total_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"{component}: средний score = {avg_score:.2f} ({len(scores)} тестов)")


if __name__ == "__main__":
    main()

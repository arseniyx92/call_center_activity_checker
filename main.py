import hosted_pbx
import asyncio
import os
import json
from tg_logger import set_application, Log_in_tg
from telegram.ext import Application
from dotenv import load_dotenv
from llm_stt import transcribe_mp3
from llm_corrector import CallCorrector
from datetime import datetime
import time

# Load environment variables
load_dotenv()

recordings_dir = os.getenv('RECORDINGS_DIR', 'recordings')

def get_record_name(call):
    record_name = f"{call['start']}_{call['type']}.mp3"
    return record_name

def setup_telegram():
    """Set up Telegram application"""
    BOT_TOKEN = os.getenv("LOG_BOT_TOKEN")
    if not BOT_TOKEN:
        raise ValueError("LOG_BOT_TOKEN environment variable not set!")
    
    application = Application.builder().token(BOT_TOKEN).build()
    set_application(application)
    print("✅ Telegram application set up successfully")
    return application

async def main(missed):
    end = datetime.now()
    start = end - datetime.timedelta(minutes=15)
    
    end = end.strftime("%Y-%m-%dT%H:%M:%S")
    start = start.strftime("%Y-%m-%dT%H:%M:%S")

    calls = hosted_pbx.get_call_history(start, end)
    if calls['error'] is not None:
        await Log_in_tg(f"❌ PBX API ERROR: {calls['error']}")
        return
    
    for call in calls['info']:
        last_call = call
        await Log_in_tg(f"📞 Обрабатываем звонок:\n{last_call}")

        # if call['type'] == 'in' and call['result'] == 'missed':
        #     missed.insert(call['client'])
        
        download_link = last_call['record']
        filename = get_record_name(last_call)
        stt_result = None
        
        if not download_link:
            await Log_in_tg("⚠️ У звонка нет записи")
            continue
        
        if hosted_pbx.download_recording(download_link, filename) == False:
            await Log_in_tg(f"❌ DOWNLOAD ERROR: Couldn't download file by this link {download_link}")
            continue
        
        stt_result = transcribe_mp3(f'{recordings_dir}/{filename}')
        if not stt_result:
            await Log_in_tg("❌ STT ERROR: Не удалось распознать речь")
            continue
        
        # print("Сырая транскрипция:")
        # print(stt_result)
        
        # Отправляем сырую транскрипцию
        # await Log_in_tg(f"Сырая транскрипция:\n{stt_result[:1000]}...")  # Ограничение длины
        
        # Инициализация LLM корректора с проверкой врачей
        corrector = CallCorrector()
        
        # Обработка через LLM с проверкой врачей
        try:
            enriched_data = corrector.process_call(
                transcription=stt_result,
                call_metadata=last_call,
                include_entities=True,
                include_classification=True,
                verify_doctor=True  # Проверка врача через Google Sheets
            )
            
            # Формируем сообщение с результатами
            result_message = "Обработка через LLM завершена!\n\n"
            
            if enriched_data.get('corrected_transcription'):
                corrected = enriched_data['corrected_transcription']
                result_message += f"Исправленная транскрипция:\n{corrected[:1500]}...\n\n"
            
            # if enriched_data.get('formatted_transcription'):
            #     formatted = enriched_data['formatted_transcription']
            #     result_message += f"Диалог (разбивка по репликам):\n{formatted[:1500]}...\n\n"
            
            # # Информация о записи к врачу
            # if enriched_data.get('appointment_info'):
            #     appt = enriched_data['appointment_info']
            #     result_message += "Информация о записи:\n"
            #     if appt.get('doctor_name'):
            #         result_message += f"Врач: {appt['doctor_name']}\n"
            #     if appt.get('doctor_specialty'):
            #         result_message += f"Специальность: {appt['doctor_specialty']}\n"
            #     if appt.get('appointment_date'):
            #         result_message += f"Дата: {appt['appointment_date']}\n"
            #     if appt.get('appointment_time'):
            #         result_message += f"Время: {appt['appointment_time']}\n"
            #     if appt.get('patient_name'):
            #         result_message += f"Пациент: {appt['patient_name']}\n"
            #     if appt.get('patient_phone'):
            #         result_message += f"Телефон: {appt['patient_phone']}\n"
            #     result_message += "\n"
            
            # # Проверка врача через Google Sheets (RAG с таблицей)
            # if enriched_data.get('doctor_verification'):
            #     verification = enriched_data['doctor_verification']
            #     result_message += "Проверка врача в Google Sheets:\n"
                
            #     if verification.get('verified'):
            #         result_message += "Врач найден и доступен\n"
            #     else:
            #         result_message += "ПРОБЛЕМА: Врач не найден или недоступен\n"
                
            #     result_message += f"{verification.get('message', '')}\n"
                
            #     if verification.get('doctor_info'):
            #         doc_info = verification['doctor_info']
            #         result_message += f"\nИнформация о враче из БД:\n"
            #         result_message += f"- ФИО: {doc_info.get('name', 'неизвестно')}\n"
            #         result_message += f"- Специальность: {doc_info.get('specialty', 'неизвестно')}\n"
            #         result_message += f"- День: {doc_info.get('day', 'неизвестно')}\n"
            #         result_message += f"- Время работы: {doc_info.get('start_time', '')}-{doc_info.get('end_time', '')}\n"
                
            #     if verification.get('llm_clarification'):
            #         clarification = verification['llm_clarification']
            #         if isinstance(clarification, dict):
            #             result_message += f"\nРекомендация LLM: {clarification.get('recommendation', '')}\n"
            #         else:
            #             result_message += f"\nУточнение: {clarification}\n"
                
            #     result_message += "\n"
            
            # Классификация
            # if enriched_data.get('classification'):
            #     cls = enriched_data['classification']
            #     # result_message += f"Классификация:\n"
            #     # result_message += f"Тип: {cls.get('type', 'неизвестно')}\n"
            #     # result_message += f"Специальность: {cls.get('specialty', 'неизвестно')}\n"
            #     # result_message += f"Эмоции: {cls.get('sentiment', 'неизвестно')}\n"
            #     result_message += f"Результат: {cls.get('result', 'неизвестно')}\n"
            
            # result_message += f"\nВремя обработки: {enriched_data.get('metadata', {}).get('processing_time', 0)} сек"
            
            await Log_in_tg(result_message)
            
            # Также выводим в консоль
            print("\n🤖 Результаты обработки:")
            print(json.dumps(enriched_data, ensure_ascii=False, indent=2))
            
        except Exception as e:
            error_msg = f"❌ Ошибка обработки через LLM: {e}"
            print(error_msg)
            await Log_in_tg(error_msg)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    missed = set()
    setup_telegram()
    while True:
        asyncio.run(main(missed))
        time.sleep(900)
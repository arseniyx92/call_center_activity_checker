import requests
import json
from dotenv import load_dotenv
import os

# Получение токена из .env файла
load_dotenv()

domain = os.getenv('GRAVITEL_DOMAIN')
api_key = os.getenv('GRAVITEL_API_KEY')
base_url = f"https://crm.aicall.ru/v1/{domain}/history"
recordings_dir = 'recordings'

headers = {
    "X-API-KEY": api_key,
    "Content-Type": "application/json"
}

download_headers = {
    "X-API-KEY": api_key,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Получение истории звонков за определенный период
def get_call_history():
    payload = {
        "period": "yesterday"  # today, yesterday, this_week, last_week, this_month, last_month
        # "start": "2024-01-01T00:00:00",  # Начало периода (опционально)
        # "end": "2024-01-31T23:59:59",    # Конец периода (опционально)
        # "type": "all",  # all, in, out, missed (опционально)
        # "limit": 100     # Ограничение количества записей (опционально)
    }

    output = {'error': None, 'info': []}

    try:
        # Отправка POST запроса
        response = requests.post(base_url, headers=headers, json=payload)
        
        # Проверка статуса ответа
        if response.status_code == 200:
            calls = response.json()
            print(f"Успешно получено звонков: {len(calls)}")
            
            # Вывод информации о звонках
            for call in calls:
                output['info'].append({'id': call.get('id'),
                'type': call.get('type'),
                'client': call.get('client'),
                'start': call.get('start'),
                'wait': call.get('wait'),
                'duration': call.get('duration'),
                'record': call.get('record')})
                
        else:
            output['error'] = f"Ошибка: {response.status_code}\nСообщение: {response.text}"

    except requests.exceptions.RequestException as e:
        output['error'] = f"Ошибка при выполнении запроса: {e}"
    
    return output

def download_recording(record_url, filename):
        """Скачать одну запись звонка"""
        file_path = os.path.join(recordings_dir, filename)
        
        # Проверяем, не скачан ли уже файл
        if os.path.exists(file_path):
            print(f"⏩ Файл уже существует: {filename}")
            return True
        
        try:
            print(f"📥 Скачиваю: {filename}")
            print(f"🔗 URL: {record_url}")
            
            response = requests.get(record_url, headers=download_headers, stream=True, timeout=60)
            response.raise_for_status()
            
            # Проверяем content-type
            content_type = response.headers.get('content-type', '')
            print(f"📄 Content-Type: {content_type}")
            
            # Скачиваем файл
            total_size = 0
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
            
            # Проверяем что файл не пустой
            if total_size > 0:
                print(f"✅ Успешно скачан: {filename} ({total_size} bytes)")
                return True
            else:
                print(f"❌ Файл пустой: {filename}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка скачивания {filename}: {e}")
            # Удаляем частично скачанный файл
            if os.path.exists(file_path):
                os.remove(file_path)
            return False
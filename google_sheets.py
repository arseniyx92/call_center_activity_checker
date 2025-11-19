"""
Модуль для работы с Google Sheets - расписание врачей.

Структура таблиц:
1. Лист "Врачи" - справочник врачей:
   - Колонка A: ФИО врача
   - Колонка B: Специальность

2. Лист "Расписание" - матрица расписания:
   - Строки: время с 9:00 до 21:00 (каждый час)
   - Столбец A: время (9:00, 10:00, ..., 21:00)
   - Столбцы B, C, D, ... : врачи (ФИО)
   - Клетки на пересечении времени и врача:
     * Зеленый цвет фона = свободно
     * Красный цвет фона = занято
     * Синий цвет фона = выходной
"""

import os
import gspread
from google.oauth2.service_account import Credentials
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv()


class DoctorsSchedule:
    """
    Класс для работы с расписанием врачей в Google Sheets.
    """
    
    # Цветовые коды RGB для определения статуса
    COLOR_GREEN = {'red': 0.8509804, 'green': 0.91764706, 'blue': 0.827451}  # Свободно
    COLOR_RED = {'red': 0.95686275, 'green': 0.8, 'blue': 0.8}  # Занято
    COLOR_BLUE = {'red': 0.8, 'green': 0.87843137, 'blue': 0.95686275}  # Выходной
    
    # Время начала и конца расписания
    START_HOUR = 9
    END_HOUR = 21
    
    def __init__(
        self,
        spreadsheet_id: Optional[str] = None,
        doctors_sheet: str = "Врачи",
        schedule_sheet: str = "Расписание"
    ):
        """
        Инициализация подключения к Google Sheets.
        
        Args:
            spreadsheet_id: ID Google таблицы (из URL)
            doctors_sheet: Название листа со справочником врачей
            schedule_sheet: Название листа с расписанием
        """
        self.spreadsheet_id = spreadsheet_id or os.getenv('GOOGLE_SHEETS_SPREADSHEET_ID')
        self.doctors_sheet_name = doctors_sheet
        self.schedule_sheet_name = schedule_sheet
        
        # Настройка доступа к Google Sheets API
        scope = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.readonly'
        ]
        
        # Путь к JSON файлу с ключами сервисного аккаунта
        creds_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH', 'credentials.json')
        
        self.client = None
        self.spreadsheet = None
        self.doctors_worksheet = None
        self.schedule_worksheet = None
        
        try:
            if os.path.exists(creds_path):
                creds = Credentials.from_service_account_file(creds_path, scopes=scope)
                self.client = gspread.authorize(creds)
                
                if self.spreadsheet_id:
                    self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
                    
                    # Открываем лист со справочником врачей
                    try:
                        self.doctors_worksheet = self.spreadsheet.worksheet(doctors_sheet)
                        print(f"✅ Лист '{doctors_sheet}' загружен")
                    except Exception as e:
                        print(f"⚠️ Лист '{doctors_sheet}' не найден: {e}")
                    
                    # Открываем лист с расписанием
                    try:
                        self.schedule_worksheet = self.spreadsheet.worksheet(schedule_sheet)
                        print(f"✅ Лист '{schedule_sheet}' загружен")
                    except Exception as e:
                        print(f"⚠️ Лист '{schedule_sheet}' не найден: {e}")
                        
                    print(f"✅ Подключение к Google Sheets установлено")
                else:
                    print("⚠️ GOOGLE_SHEETS_SPREADSHEET_ID не указан")
            else:
                print(f"⚠️ Файл credentials.json не найден по пути: {creds_path}")
                print("Создайте сервисный аккаунт в Google Cloud Console")
                
        except Exception as e:
            print(f"⚠️ Ошибка инициализации Google Sheets: {e}")
            self.client = None
            self.spreadsheet = None
    
    def get_all_doctors(self) -> List[Dict[str, Any]]:
        """
        Получить всех врачей из справочника.
        
        Returns:
            Список словарей с информацией о врачах (name, specialty)
        """
        if not self.doctors_worksheet:
            return []
        
        try:
            # Получаем все данные (предполагаем, что первая строка - заголовки)
            all_records = self.doctors_worksheet.get_all_records()
            
            doctors = []
            for record in all_records:
                doctor = {
                    'name': record.get('ФИО врача', '') or record.get('Имя', '') or '',
                    'specialty': record.get('Специальность', '') or record.get('Специализация', '') or ''
                }
                if doctor['name']:  # Игнорируем пустые строки
                    doctors.append(doctor)
            
            return doctors
            
        except Exception as e:
            print(f"❌ Ошибка получения списка врачей: {e}")
            return []
    
    def find_doctor_by_name(self, doctor_name: str) -> List[Dict[str, Any]]:
        """
        Найти врача по имени (частичное совпадение).
        
        Args:
            doctor_name: Имя или фамилия врача
            
        Returns:
            Список найденных врачей
        """
        doctors = self.get_all_doctors()
        doctor_name_lower = doctor_name.lower().strip()
        
        found = []
        for doctor in doctors:
            name_lower = doctor['name'].lower()
            # Проверяем полное совпадение или совпадение фамилии/имени
            if doctor_name_lower in name_lower or name_lower in doctor_name_lower:
                found.append(doctor)
        
        return found
    
    def find_doctors_by_specialty(self, specialty: str) -> List[Dict[str, Any]]:
        """
        Найти врачей по специальности.
        
        Args:
            specialty: Название специальности
            
        Returns:
            Список врачей с указанной специальностью
        """
        doctors = self.get_all_doctors()
        specialty_lower = specialty.lower().strip()
        
        found = []
        for doctor in doctors:
            doc_specialty_lower = doctor['specialty'].lower()
            if specialty_lower in doc_specialty_lower or doc_specialty_lower in specialty_lower:
                found.append(doctor)
        
        return found
    
    def _get_cell_color_status(self, cell_color: Dict[str, float]) -> str:
        """
        Определить статус ячейки по цвету фона.
        
        Args:
            cell_color: Словарь с RGB цветом ({red, green, blue})
            
        Returns:
            'free' - свободно (зеленый)
            'busy' - занято (красный)
            'holiday' - выходной (синий)
            'unknown' - неизвестно
        """
        if not cell_color:
            return 'unknown'
        
        # Сравнение цветов с небольшой погрешностью
        def color_similar(color1: Dict[str, float], color2: Dict[str, float], threshold: float = 0.1) -> bool:
            return (
                abs(color1.get('red', 0) - color2.get('red', 0)) < threshold and
                abs(color1.get('green', 0) - color2.get('green', 0)) < threshold and
                abs(color1.get('blue', 0) - color2.get('blue', 0)) < threshold
            )
        
        if color_similar(cell_color, self.COLOR_GREEN):
            return 'free'
        elif color_similar(cell_color, self.COLOR_RED):
            return 'busy'
        elif color_similar(cell_color, self.COLOR_BLUE):
            return 'holiday'
        else:
            return 'unknown'
    
    def _get_time_row_index(self, time_slot: str) -> Optional[int]:
        """
        Получить номер строки для указанного времени.
        
        Args:
            time_slot: Время в формате "HH:MM" или "HH"
            
        Returns:
            Номер строки (1-based) или None если время вне диапазона
        """
        try:
            # Парсинг времени
            if ':' in time_slot:
                hour = int(time_slot.split(':')[0])
            else:
                hour = int(time_slot)
            
            # Проверка диапазона
            if hour < self.START_HOUR or hour > self.END_HOUR:
                return None
            
            # Строка 1 - заголовки
            # Строка 2 - 9:00
            # Строка 3 - 10:00
            # ...
            row_index = (hour - self.START_HOUR) + 2
            
            return row_index
            
        except Exception as e:
            print(f"⚠️ Ошибка парсинга времени '{time_slot}': {e}")
            return None
    
    def _get_doctor_column_index(self, doctor_name: str) -> Optional[int]:
        """
        Получить номер столбца для указанного врача.
        
        Args:
            doctor_name: Имя врача
            
        Returns:
            Номер столбца (1-based, где A=1, B=2, ...) или None если врач не найден
        """
        if not self.schedule_worksheet:
            return None
        
        try:
            # Получаем заголовки (первая строка)
            header_row = self.schedule_worksheet.row_values(1)
            
            # Ищем врача в заголовках (начиная со второго столбца, первый - время)
            doctor_name_lower = doctor_name.lower().strip()
            
            for col_index, header in enumerate(header_row[1:], start=2):  # Начинаем с колонки B (индекс 2)
                if header and doctor_name_lower in header.lower():
                    # Проверяем точное совпадение с врачами из справочника
                    found_doctors = self.find_doctor_by_name(doctor_name)
                    if found_doctors:
                        # Проверяем, совпадает ли заголовок с одним из найденных врачей
                        for doctor in found_doctors:
                            if doctor['name'].lower() in header.lower() or header.lower() in doctor['name'].lower():
                                return col_index
                    
                    # Если не нашли точное совпадение, возвращаем первое частичное
                    return col_index
            
            return None
            
        except Exception as e:
            print(f"⚠️ Ошибка поиска столбца врача '{doctor_name}': {e}")
            return None
    
    def check_doctor_availability(
        self,
        doctor_name: str,
        specialty: Optional[str] = None,
        day: Optional[str] = None,
        time_slot: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Проверить доступность врача в указанное время.
        
        Args:
            doctor_name: Имя врача
            specialty: Специальность (для дополнительной проверки)
            day: День недели (опционально, пока не используется)
            time_slot: Время записи (например, "14:00" или "14")
            
        Returns:
            Словарь с результатом проверки
        """
        result = {
            'doctor_exists': False,
            'specialty_matches': True,  # По умолчанию True, если специальность не указана
            'available_at_time': False,
            'doctor_info': None,
            'message': ''
        }
        
        # Проверка существования врача в справочнике
        found_doctors = self.find_doctor_by_name(doctor_name)
        
        if not found_doctors:
            result['message'] = f"Врач '{doctor_name}' не найден в справочнике"
            return result
        
        result['doctor_exists'] = True
        
        # Если несколько врачей с похожими именами, берем первого
        doctor = found_doctors[0]
        
        if len(found_doctors) > 1:
            result['message'] = f"Найдено несколько врачей с именем '{doctor_name}'. Используется: {doctor['name']}"
        
        # Проверка специальности
        if specialty:
            specialty_matches = specialty.lower() in doctor['specialty'].lower()
            result['specialty_matches'] = specialty_matches
            
            if not specialty_matches:
                result['message'] = f"Врач {doctor['name']} найден, но его специальность '{doctor['specialty']}' не соответствует запрошенной '{specialty}'"
                result['doctor_info'] = doctor
                return result
        
        # Проверка времени в расписании
        if time_slot and self.schedule_worksheet:
            # Получаем индексы строки и столбца
            row_index = self._get_time_row_index(time_slot)
            col_index = self._get_doctor_column_index(doctor['name'])
            
            if row_index is None:
                result['message'] = f"Время '{time_slot}' вне диапазона расписания (9:00-21:00)"
                result['doctor_info'] = doctor
                return result
            
            if col_index is None:
                result['message'] = f"Врач {doctor['name']} не найден в расписании"
                result['doctor_info'] = doctor
                return result
            
            # Получаем цвет фона ячейки через Google Sheets API
            try:
                cell = self.schedule_worksheet.cell(row_index, col_index)
                cell_value = cell.value
                
                # Получаем цвет фона через Google Sheets API
                cell_color = None
                try:
                    spreadsheet_id = self.spreadsheet.id
                    cell_a1 = gspread.utils.rowcol_to_a1(row_index, col_index)
                    range_notation = f"{self.schedule_worksheet.title}!{cell_a1}"
                    
                    # Используем Google Sheets API напрямую
                    from googleapiclient.discovery import build
                    
                    # Получаем credentials из gspread client
                    credentials = None
                    if hasattr(self.client, 'auth') and hasattr(self.client.auth, 'credentials'):
                        credentials = self.client.auth.credentials
                    elif hasattr(self.client, '_session') and hasattr(self.client._session, 'credentials'):
                        credentials = self.client._session.credentials
                    
                    if credentials:
                        service = build('sheets', 'v4', credentials=credentials)
                        
                        request = service.spreadsheets().get(
                            spreadsheetId=spreadsheet_id,
                            ranges=[range_notation],
                            fields='sheets.data.rowData.values.userEnteredFormat.backgroundColor'
                        )
                        response = request.execute()
                        
                        # Извлекаем цвет фона
                        try:
                            sheets = response.get('sheets', [])
                            if sheets and sheets[0].get('data'):
                                row_data = sheets[0]['data'][0].get('rowData', [])
                                if row_data and row_data[0].get('values'):
                                    user_format = row_data[0]['values'][0].get('userEnteredFormat', {})
                                    cell_color = user_format.get('backgroundColor', {})
                        except (KeyError, IndexError, AttributeError):
                            pass
                    
                except Exception as e:
                    print(f"⚠️ Ошибка получения цвета ячейки через API: {e}")
                    # Если не получили цвет, используем fallback
                
                # Определяем статус по цвету или значению
                if cell_color:
                    status = self._get_cell_color_status(cell_color)
                else:
                    # Fallback: определяем по значению ячейки
                    if not cell_value or cell_value.lower() in ['свободно', 'free', '']:
                        status = 'free'
                    elif cell_value.lower() in ['занято', 'busy', 'занят']:
                        status = 'busy'
                    elif cell_value.lower() in ['выходной', 'holiday', 'вых']:
                        status = 'holiday'
                    else:
                        status = 'unknown'
                
                # Определяем доступность
                result['available_at_time'] = (status == 'free')
                
                # Нормализуем формат времени для сообщения
                time_display = time_slot
                if ':' not in time_slot:
                    time_display = f"{time_slot}:00"
                
                if status == 'free':
                    result['message'] = f"✅ Врач {doctor['name']} свободен в {time_display}"
                elif status == 'busy':
                    result['message'] = f"❌ Врач {doctor['name']} занят в {time_display}"
                elif status == 'holiday':
                    result['message'] = f"🚫 Врач {doctor['name']} в выходной в {time_display}"
                else:
                    result['message'] = f"⚠️ Статус врача {doctor['name']} в {time_display} не определен"
                
            except Exception as e:
                print(f"⚠️ Ошибка чтения ячейки расписания: {e}")
                result['message'] = f"Ошибка проверки расписания врача {doctor['name']}"
        else:
            result['available_at_time'] = True  # Если время не указано, считаем доступным
            result['message'] = f"Врач {doctor['name']} найден: {doctor['specialty']}"
        
        result['doctor_info'] = doctor
        
        return result
    
    def get_context_for_rag(self, doctor_name: Optional[str] = None, specialty: Optional[str] = None) -> str:
        """
        Получить контекст для RAG (информация о врачах).
        
        Args:
            doctor_name: Фильтр по имени врача (опционально)
            specialty: Фильтр по специальности (опционально)
            
        Returns:
            Текст с контекстом о врачах
        """
        if not self.doctors_worksheet:
            return "База данных врачей недоступна"
        
        doctors = self.get_all_doctors()
        
        if doctor_name:
            doctors = self.find_doctor_by_name(doctor_name)
        elif specialty:
            doctors = self.find_doctors_by_specialty(specialty)
        
        if not doctors:
            return f"Врачи не найдены (фильтр: имя='{doctor_name}', специальность='{specialty}')"
        
        context = "Информация о врачах из базы данных:\n\n"
        for doctor in doctors[:10]:  # Ограничиваем 10 записями
            context += f"- {doctor['name']}, {doctor['specialty']}\n"
        context += "\n"
        
        return context
    
    def get_doctor_schedule(self, doctor_name: str, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Получить расписание врача на день.
        
        Args:
            doctor_name: Имя врача
            date: Дата (опционально, пока не используется)
            
        Returns:
            Словарь с расписанием: {time: status, ...}
        """
        if not self.schedule_worksheet:
            return {}
        
        col_index = self._get_doctor_column_index(doctor_name)
        if not col_index:
            return {}
        
        schedule = {}
        try:
            # Получаем все ячейки в столбце врача (строки со временем)
            for hour in range(self.START_HOUR, self.END_HOUR + 1):
                row_index = self._get_time_row_index(f"{hour}:00")
                if row_index:
                    try:
                        cell = self.schedule_worksheet.cell(row_index, col_index)
                        # Упрощенная проверка: если ячейка пустая - свободно
                        status = 'free' if not cell.value else 'unknown'
                        schedule[f"{hour}:00"] = status
                    except Exception:
                        schedule[f"{hour}:00"] = 'unknown'
        except Exception as e:
            print(f"⚠️ Ошибка получения расписания: {e}")
        
        return schedule


# Пример использования
if __name__ == "__main__":
    schedule = DoctorsSchedule()
    
    # Тест поиска врача
    result = schedule.check_doctor_availability(
        doctor_name="Иванов",
        specialty="Терапевт",
        time_slot="14"
    )
    
    print(result)

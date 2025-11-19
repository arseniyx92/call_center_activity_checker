"""
Модуль для получения информации с веб-сайта для использования в RAG.

Поддерживаемые форматы:
- HTML страницы (через BeautifulSoup)
- JSON API endpoints
- RSS/Atom фиды
- Структурированные данные (JSON-LD, Microdata)
"""

import os
import requests
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
import json
import re

load_dotenv()

try:
    from bs4 import BeautifulSoup
    BEAUTIFUL_SOUP_AVAILABLE = True
except ImportError:
    BEAUTIFUL_SOUP_AVAILABLE = False
    print("⚠️ BeautifulSoup4 не установлен. Установите: pip install beautifulsoup4")
    # Заглушка для типизации
    BeautifulSoup = None


class WebScraper:
    """
    Класс для извлечения информации с веб-сайтов для использования в RAG.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 10,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Инициализация веб-скрейпера.
        
        Args:
            base_url: Базовый URL сайта (для относительных ссылок)
            timeout: Таймаут запросов в секундах
            headers: Дополнительные HTTP заголовки
        """
        self.base_url = base_url or os.getenv('WEBSITE_BASE_URL', '')
        self.timeout = timeout
        
        # Стандартные заголовки (имитация браузера)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        if headers:
            self.headers.update(headers)
    
    def fetch_page(self, url: str) -> Optional[str]:
        """
        Получить содержимое страницы.
        
        Args:
            url: URL страницы
            
        Returns:
            HTML содержимое страницы или None при ошибке
        """
        if not url.startswith(('http://', 'https://')):
            url = urljoin(self.base_url, url)
        
        response = requests.get(url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or 'utf-8'
        
        return response.text
    
    def extract_from_html(
        self,
        html: str,
        selectors: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Извлечь структурированную информацию из HTML.
        
        Args:
            html: HTML содержимое
            selectors: Словарь CSS селекторов для извлечения данных
                      Например: {'doctors': '.doctor-card', 'services': '.service-item'}
            
        Returns:
            Словарь с извлеченными данными
        """
        if not BEAUTIFUL_SOUP_AVAILABLE:
            return {"error": "BeautifulSoup4 не установлен"}
        
        soup = BeautifulSoup(html, 'html.parser')
        result = {}
        
        # Стандартные селекторы, если не указаны
        default_selectors = {
            'title': 'title',
            'description': 'meta[name="description"]',
            'doctors': '[class*="doctor"], [class*="врач"], [data-role="doctor"]',
            'services': '[class*="service"], [class*="услуга"], [data-role="service"]',
            'price': '[class*="price"], [class*="цена"]',
            'schedule': '[class*="schedule"], [class*="расписание"]',
            'address': '[class*="address"], [class*="адрес"]',
            'phone': '[class*="phone"], [class*="телефон"], a[href^="tel:"]'
        }
        
        selectors_to_use = selectors or default_selectors
        
        # Извлечение данных по селекторам
        for key, selector in selectors_to_use.items():
            elements = soup.select(selector)
            if not elements:
                continue
            
            if key == 'description':
                result[key] = elements[0].get('content', '')
            elif key == 'title':
                result[key] = elements[0].get_text(strip=True)
            elif key == 'phone':
                phones = []
                for el in elements:
                    href = el.get('href', '')
                    if href.startswith('tel:'):
                        phones.append(href.replace('tel:', '').strip())
                    else:
                        text = el.get_text(strip=True)
                        phone_match = re.search(r'[\d\s\-\+\(\)]{10,}', text)
                        if phone_match:
                            phones.append(phone_match.group().strip())
                result[key] = list(set(phones))
            else:
                texts = [el.get_text(separator=' ', strip=True) for el in elements]
                result[key] = texts if len(texts) > 1 else (texts[0] if texts else '')
        
        # Извлечение структурированных данных (JSON-LD)
        json_ld_data = self._extract_json_ld(soup)
        if json_ld_data:
            result['structured_data'] = json_ld_data
        
        # Извлечение microdata
        microdata = self._extract_microdata(soup)
        if microdata:
            result['microdata'] = microdata
        
        return result
    
    def _extract_json_ld(self, soup) -> List[Dict[str, Any]]:
        """
        Извлечь структурированные данные JSON-LD.
        
        Args:
            soup: BeautifulSoup объект
            
        Returns:
            Список словарей со структурированными данными
        """
        json_ld_data = []
        for script in soup.find_all('script', type='application/ld+json'):
            data = json.loads(script.string)
            json_ld_data.append(data)
        
        return json_ld_data
    
    def _extract_microdata(self, soup) -> List[Dict[str, Any]]:
        """
        Извлечь данные из Microdata.
        
        Args:
            soup: BeautifulSoup объект
            
        Returns:
            Список словарей с microdata
        """
        microdata = []
        
        # Поиск элементов с itemscope
        items = soup.find_all(attrs={'itemscope': True})
        
        for item in items:
            item_data = {}
            item_type = item.get('itemtype', '')
            if item_type:
                item_data['@type'] = item_type
            
            # Извлекаем свойства
            props = item.find_all(attrs={'itemprop': True})
            for prop in props:
                prop_name = prop.get('itemprop')
                prop_value = prop.get('content') or prop.get_text(strip=True)
                if prop_name:
                    if prop_name in item_data:
                        # Если свойство уже есть, делаем список
                        if not isinstance(item_data[prop_name], list):
                            item_data[prop_name] = [item_data[prop_name]]
                        item_data[prop_name].append(prop_value)
                    else:
                        item_data[prop_name] = prop_value
            
            if item_data:
                microdata.append(item_data)
        
        return microdata
    
    def get_context_for_rag(
        self,
        url: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        max_length: int = 2000,
        include_doctors: bool = False,
        include_services: bool = True,
        include_contacts: bool = True
    ) -> str:
        """
        Получить контекст с сайта для использования в RAG.
        
        Args:
            url: URL страницы (если None, используется base_url)
            keywords: Ключевые слова для фильтрации релевантного контента
            max_length: Максимальная длина контекста в символах
            
        Returns:
            Текст с контекстом для RAG
        """
        target_url = url or self.base_url
        if not target_url:
            return "URL не указан"
        
        html = self.fetch_page(target_url)
        if not html:
            return "Не удалось получить данные с сайта"
        
        extracted = self.extract_from_html(html)
        
        # Формируем контекст для RAG
        context_parts = []
        
        # Заголовок страницы (опционально)
        if extracted.get('title'):
            context_parts.append(f"Клиника: {extracted['title']}\n")
        
        # Контакты (телефоны и адреса) - ВКЛЮЧАЕМ
        if include_contacts:
            if extracted.get('address'):
                addresses = extracted['address']
                if isinstance(addresses, list):
                    context_parts.append("Адреса клиники:\n")
                    for addr in addresses[:5]:  # Ограничиваем 5 адресами
                        context_parts.append(f"- {addr}\n")
                    context_parts.append("\n")
                else:
                    context_parts.append(f"Адрес: {addresses}\n\n")
            
            if extracted.get('phone'):
                phones = extracted['phone']
                if isinstance(phones, list):
                    context_parts.append("Телефоны клиники:\n")
                    for phone in phones:
                        context_parts.append(f"- {phone}\n")
                    context_parts.append("\n")
                else:
                    context_parts.append(f"Телефон: {phones}\n\n")
        
        # Информация о врачах - ОПЦИОНАЛЬНО (по умолчанию НЕ включаем)
        if include_doctors and extracted.get('doctors'):
            context_parts.append("Информация о врачах с сайта:\n")
            doctors = extracted['doctors']
            if isinstance(doctors, list):
                for doctor in doctors[:10]:  # Ограничиваем 10 записями
                    context_parts.append(f"- {doctor}\n")
            else:
                context_parts.append(f"- {doctors}\n")
            context_parts.append("\n")
        
        # Услуги - ВКЛЮЧАЕМ
        if include_services and extracted.get('services'):
            context_parts.append("Услуги клиники:\n")
            services = extracted['services']
            if isinstance(services, list):
                # Фильтруем дубликаты и пустые значения
                unique_services = []
                seen = set()
                for service in services:
                    service_text = str(service).strip()
                    if service_text and len(service_text) > 3:  # Игнорируем очень короткие строки
                        service_lower = service_text.lower()
                        if service_lower not in seen:
                            seen.add(service_lower)
                            unique_services.append(service_text)
                
                for service in unique_services[:20]:  # Ограничиваем 20 услугами
                    context_parts.append(f"- {service}\n")
            else:
                context_parts.append(f"- {services}\n")
            context_parts.append("\n")
        
        # Цены (если есть) - опционально
        if include_services and extracted.get('price'):
            prices = extracted['price']
            if isinstance(prices, list):
                unique_prices = []
                for price in prices[:10]:
                    price_text = str(price).strip()
                    if price_text:
                        unique_prices.append(price_text)
                if unique_prices:
                    context_parts.append("Цены на услуги:\n")
                    for price in unique_prices:
                        context_parts.append(f"- {price}\n")
                    context_parts.append("\n")
        
        # Структурированные данные (JSON-LD)
        if extracted.get('structured_data'):
            context_parts.append("Структурированная информация:\n")
            for data in extracted['structured_data'][:3]:  # Берем первые 3
                # Извлекаем релевантные поля
                if isinstance(data, dict):
                    name = data.get('name') or data.get('@name') or ''
                    description = data.get('description') or data.get('@description') or ''
                    if name:
                        context_parts.append(f"- {name}")
                        if description:
                            context_parts.append(f": {description[:100]}")
                        context_parts.append("\n")
            context_parts.append("\n")
        
        context = "".join(context_parts)
        
        # Обрезаем до max_length
        if len(context) > max_length:
            context = context[:max_length] + "...\n"
        
        return context if context else "Информация с сайта недоступна"


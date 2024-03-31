# Автоматизированная база знаний закупок RLT.HACK

## Обзор

Проект направлен на разработку системы вопросов и ответов для эффективного обслуживания запросов поставщиков, используя передовые технологии обработки естественного языка.

## Команда: Мисисково

## Результат: 
    1 место

## Начало работы

Для развертывания нашего решения выполните следующие шаги:

```bash
git clone https://github.com/danzzzlll/RAG_RltHack.git
cd RAG_RltHack
pip install -r requirements.txt
python ./utils/saving_retriever # Для получения и сохранения индексов
streamlit run main.py

## Подход к решению

Наш метод разделен на несколько ключевых этапов:

- **Парсинг данных**: Мы разобрали законы, постановления и форумы, относящиеся к Федеральным законам 44 и 223, собрав обширный набор данных.
- **Обработка данных**: Собранные документы были разделены на осмысленные части — статьи, разделы и подразделы, для удобства управления.
- **Создание ретривера**: Используя `multilingual-e5-large` для генерации векторных представлений и `Voyage` для быстрого поиска и индексации, нам удалось извлечь 40 или 60 релевантных частей на запрос.
- **Оценка**: Извлеченные части переоценивались с использованием `antoinelouis/colbert-xm` для выявления трех наиболее релевантных чанков.
- **Создание промпта**: Мы разработали промпты, нацеленные на уточнение обработки запроса.
- **Настройка сервера**: Был настроен сервер на базе Streamlit, предлагающий выбор закона, историю запросов, уровни уверенности и используемые источники.
- **Обработка промпта**: Входные данные обрабатывались с использованием GigaChat Pro и Mistral-instruct-v2 для генерации точных ответов.
- **Предоставление ответа**: Сервер отображает сгенерированный ответ, включая уровни уверенности и использованные источники.

## Особенности

- **Выбор закона**: Пользователи могут выбирать конкретные законы, относящиеся к их запросам.
- **История запросов**: Отслеживает и отображает прошлые запросы для удобного справочника.
- **Уровни уверенности и источники**: Каждый ответ включает в себя уровни уверенности и используемые источники, обеспечивая прозрачность и надежность.
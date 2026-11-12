# Публикация AI-детектива

## 1. Создать GitHub repository

1. Открой https://github.com/new
2. Repository name: `ai-detective-lab6`
3. Выбери Public или Private согласно требованиям преподавателя.
4. Не добавляй README/.gitignore/license автоматически — они уже находятся в проекте.
5. Создай пустой репозиторий.

## 2. Загрузить проект

Распакуй `ai-detective-lab6-github-ready.zip`, открой терминал в папке `ai-detective` и выполни:

```bash
git init -b main
git add .
git status
git commit -m "Submit AI Detective Lab 6"
git remote add origin https://github.com/USERNAME/ai-detective-lab6.git
git push -u origin main
```

Замени `USERNAME` на свой GitHub username. Перед `git add` убедись, что в проекте нет `.env`, токенов и личных файлов. В проекте уже настроен `.gitignore`.

## 3. Опубликовать Streamlit

1. Открой https://share.streamlit.io
2. Нажми Create app.
3. Repository: `USERNAME/ai-detective-lab6`
4. Branch: `main`
5. Main file path: `app.py`
6. Deploy.

В облачном приложении доступен просмотр сохранённых 36 основных и 84 стрессовых результатов. Новые запросы к Qwen через Ollama доступны только при локальном запуске Ollama; в Streamlit Cloud локального Ollama-сервера нет.

## 4. Проверить опубликованное приложение

- Открывается ли главная страница без ошибки.
- Выбираются ли вопросы Q01–Q12 и режимы A/B/C.
- Отображаются ли 36 основных ответов и их метрики.
- Отображаются ли 84 стресс-кейса, фильтры по типу/вопросу/режиму и подробности вмешательства.
- Не пытается ли облачное приложение вызвать локальную Ollama (кнопка live-запроса должна быть отключена при отсутствии сервера).

## 5. Что отправить преподавателю

- `report.pdf`
- ссылку на GitHub repository
- ссылку на опубликованное Streamlit app
- `lab6_submission_final.zip`, если преподаватель просит полный архив результатов.

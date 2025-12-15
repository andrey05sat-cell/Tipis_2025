# Cardiovascular Disease Risk Prediction

Учебный проект по предмету "Теория информационных процессов и систем".

Цель проекта — построить простую систему, которая по данным о человеке
(возраст, давление, холестерин, вредные привычки и т.п.) оценивает
риск развития сердечно-сосудистых заболеваний.

## Структура проекта

```text
cvd_risk_project/
├── data/
│   ├── raw/                     # исходные данные (Kaggle)
│   └── processed/               # очищенные данные
├── models/
│   └── cvd_model.pkl            # обученная модель (Pipeline)
├── notebooks/
│   ├── 01_eda.ipynb             # разведочный анализ данных
│   └── 02_preprocessing_model.ipynb  # подготовка и обучение модели
├── app/
│   └── app.py                   # веб-приложение Streamlit
├── .gitignore
├── README.md
└── requirements.txt
```

## Данные

1. На Kaggle найдите датасет по риску сердечно-сосудистых заболеваний
   (например, на основе Framingham Heart Study) и скачайте CSV.
2. Переименуйте файл, например, в `cvd_raw.csv`.
3. Поместите его в папку `data/raw/`.

В ноутбуках при необходимости поменяйте путь и имя целевой колонки.

## Установка и запуск

```bash
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>.git
cd cvd_risk_project

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Затем выполните ноутбуки по порядку:

1. `notebooks/01_eda.ipynb` — загрузка и базовая очистка данных,
   сохранение `data/processed/cvd_clean.csv`.
2. `notebooks/02_preprocessing_model.ipynb` — подготовка признаков,
   обучение модели и сохранение `models/cvd_model.pkl`.

Запуск веб-приложения:

```bash
streamlit run app/app.py
```

После этого в браузере откроется интерфейс для ввода данных пациента
или загрузки CSV-файла и получения предсказаний.

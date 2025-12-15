import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "cvd_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Система оценки риска сердечно-сосудистых заболеваний")

st.markdown(
    """
    Это учебное приложение, которое использует обученную модель
    машинного обучения для оценки риска сердечно-сосудистых заболеваний (CVD)
    по данным о пациенте.

    1. Подготовьте данные и обучите модель в ноутбуках проекта.
    2. Убедитесь, что файл `models/cvd_model.pkl` существует.
    3. Запустите приложение командой:

    ```bash
    streamlit run app/app.py
    ```
    """
)

tab_single, tab_batch = st.tabs(["Один пациент", "CSV-файл"])

with tab_single:
    st.subheader("Ввод данных одного пациента")

    st.info(
        "Список признаков нужно при необходимости адаптировать под ваш датасет. "
        "Ниже приведен пример для датасета Framingham."
    )

    age = st.number_input("Возраст", min_value=18, max_value=100, value=50)
    male = st.selectbox("Пол", options=["Женщина", "Мужчина"])
    current_smoker = st.selectbox("Текущий курильщик", options=["Нет", "Да"])
    cigs_per_day = st.number_input("Сигарет в день", min_value=0, max_value=100, value=0)
    bpm_meds = st.selectbox("Принимает гипотензивные препараты (BPMeds)", options=["Нет", "Да"])
    prevalent_stroke = st.selectbox("Был инсульт", options=["Нет", "Да"])
    prevalent_hyp = st.selectbox("Артериальная гипертензия", options=["Нет", "Да"])
    diabetes = st.selectbox("Сахарный диабет", options=["Нет", "Да"])
    tot_chol = st.number_input("Общий холестерин", min_value=50.0, max_value=400.0, value=200.0)
    sys_bp = st.number_input("Систолическое давление", min_value=80.0, max_value=250.0, value=120.0)
    dia_bp = st.number_input("Диастолическое давление", min_value=40.0, max_value=150.0, value=80.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    heart_rate = st.number_input("Пульс", min_value=30.0, max_value=200.0, value=70.0)
    glucose = st.number_input("Глюкоза", min_value=40.0, max_value=400.0, value=80.0)

    input_dict = {
        "age": age,
        "male": 1 if male == "Мужчина" else 0,
        "currentSmoker": 1 if current_smoker == "Да" else 0,
        "cigsPerDay": cigs_per_day,
        "BPMeds": 1 if bpm_meds == "Да" else 0,
        "prevalentStroke": 1 if prevalent_stroke == "Да" else 0,
        "prevalentHyp": 1 if prevalent_hyp == "Да" else 0,
        "diabetes": 1 if diabetes == "Да" else 0,
        "totChol": tot_chol,
        "sysBP": sys_bp,
        "diaBP": dia_bp,
        "BMI": bmi,
        "heartRate": heart_rate,
        "glucose": glucose,
    }

    if st.button("Рассчитать риск"):
        input_df = pd.DataFrame([input_dict])
        proba = model.predict_proba(input_df)[0, 1]
        pred = model.predict(input_df)[0]

        st.write(f"Вероятность события (CVD): **{proba:.2%}**")
        if pred == 1:
            st.error("Модель относит пациента к группе повышенного риска.")
        else:
            st.success("Модель относит пациента к группе низкого риска.")

with tab_batch:
    st.subheader("Предсказания по CSV-файлу")

    st.markdown(
        "Загрузите CSV-файл с теми же колонками, которые использовались при обучении "
        "модели (кроме целевой колонки)."
    )

    uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Первые строки загруженных данных:")
        st.dataframe(data.head())

        if st.button("Получить предсказания для файла"):
            proba = model.predict_proba(data)[:, 1]
            preds = model.predict(data)

            result = data.copy()
            result["cvd_proba"] = proba
            result["cvd_pred"] = preds

            st.subheader("Результаты предсказания")
            st.dataframe(result.head())

            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать результаты в CSV",
                data=csv,
                file_name="cvd_predictions.csv",
                mime="text/csv",
            )

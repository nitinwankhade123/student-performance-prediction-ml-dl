"""Local Streamlit demo for the saved synthetic-data Logistic Regression model.

The app loads the already-fitted estimator and the scaler fitted on the
training split. It performs inference only and never trains or updates a model.
Run from the project root with: python -m streamlit run deployment/app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "experiments" / "logistic_regression" / "model.joblib"
SCALER_PATH = PROJECT_ROOT / "artifacts" / "standard_scaler.joblib"
FEATURE_COLUMNS = ["study_hours", "attendance", "previous_marks"]
FROZEN_THRESHOLD = 0.45458775758743286

# These input bounds match the synthetic feature ranges used to train the model.
INPUT_RANGES = {
    "study_hours": (0.0, 10.0),
    "attendance": (40.0, 100.0),
    "previous_marks": (20.0, 100.0),
}


@st.cache_resource
def load_artifacts():
    """Load the saved model and training-fitted scaler once per app session."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved Logistic Regression model not found: {MODEL_PATH}")
    if not SCALER_PATH.is_file():
        raise FileNotFoundError(f"Saved preprocessing scaler not found: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if list(getattr(scaler, "feature_names_in_", [])) != FEATURE_COLUMNS:
        raise ValueError("Saved scaler feature order does not match this app's inputs")
    if int(getattr(model, "n_features_in_", -1)) != len(FEATURE_COLUMNS):
        raise ValueError("Saved model does not expect the three configured input features")
    if list(getattr(model, "classes_", [])) != [0, 1]:
        raise ValueError("Saved model classes must use fail=0 and pass=1 encoding")

    return model, scaler


def predict_student(
    study_hours: float,
    attendance: float,
    previous_marks: float,
) -> dict[str, float | int | str]:
    """Validate inputs, apply saved scaling, and predict using frozen settings."""
    values = {
        "study_hours": float(study_hours),
        "attendance": float(attendance),
        "previous_marks": float(previous_marks),
    }
    for feature, value in values.items():
        lower, upper = INPUT_RANGES[feature]
        if not lower <= value <= upper:
            raise ValueError(f"{feature} must be between {lower:g} and {upper:g}")

    model, scaler = load_artifacts()
    input_frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    transformed_features = scaler.transform(input_frame)
    probabilities = model.predict_proba(transformed_features)[0]
    class_indices = {int(label): index for index, label in enumerate(model.classes_)}
    fail_probability = float(probabilities[class_indices[0]])
    pass_probability = float(probabilities[class_indices[1]])
    predicted_label = int(pass_probability >= FROZEN_THRESHOLD)

    return {
        "predicted_label": predicted_label,
        "predicted_outcome": "Pass" if predicted_label == 1 else "Fail",
        "fail_probability": fail_probability,
        "pass_probability": pass_probability,
        "threshold": FROZEN_THRESHOLD,
    }


def main() -> None:
    st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")
    st.title("Student Performance Prediction")
    st.caption("Educational demonstration using a saved Logistic Regression model")

    st.warning(
        "This demonstration uses a synthetic educational dataset. It is not a "
        "validated real-world student assessment system and should not be used "
        "to make academic decisions. Its displayed class probabilities are model "
        "estimates and are not established as calibrated real-world probabilities."
    )

    st.write(
        "Enter values within the synthetic training ranges. The app applies the "
        "saved training-fitted StandardScaler before prediction."
    )
    st.markdown(
        "- Study hours per day: **0–10**\n"
        "- Attendance percentage: **40–100%**\n"
        "- Previous exam marks: **20–100**"
    )

    try:
        load_artifacts()
    except Exception as error:
        st.error(f"Unable to load the saved model artifacts: {error}")
        st.info(
            "From the project root, verify that `experiments/logistic_regression/"
            "model.joblib` and `artifacts/standard_scaler.joblib` exist."
        )
        return

    with st.form("student_inputs"):
        study_hours = st.number_input(
            "Study hours per day",
            min_value=INPUT_RANGES["study_hours"][0],
            max_value=INPUT_RANGES["study_hours"][1],
            value=5.0,
            step=0.5,
            key="study_hours",
        )
        attendance = st.number_input(
            "Attendance percentage",
            min_value=INPUT_RANGES["attendance"][0],
            max_value=INPUT_RANGES["attendance"][1],
            value=70.0,
            step=1.0,
            key="attendance",
        )
        previous_marks = st.number_input(
            "Previous exam marks",
            min_value=INPUT_RANGES["previous_marks"][0],
            max_value=INPUT_RANGES["previous_marks"][1],
            value=60.0,
            step=1.0,
            key="previous_marks",
        )
        submitted = st.form_submit_button("Predict result")

    st.caption(f"Frozen classification threshold: {FROZEN_THRESHOLD:.17g} (selected using validation data)")

    if submitted:
        try:
            prediction = predict_student(study_hours, attendance, previous_marks)
        except (ValueError, FileNotFoundError) as error:
            st.error(str(error))
            return

        if prediction["predicted_label"] == 1:
            st.success(f"Model prediction: **{prediction['predicted_outcome']}**")
        else:
            st.error(f"Model prediction: **{prediction['predicted_outcome']}**")
        st.subheader("Estimated class probabilities")
        col_fail, col_pass = st.columns(2)
        col_fail.metric("Fail", f"{prediction['fail_probability']:.1%}")
        col_pass.metric("Pass", f"{prediction['pass_probability']:.1%}")
        st.caption(
            "The model predicts Pass when its Pass probability is at or above "
            "the frozen validation-selected threshold; otherwise it predicts Fail."
        )


if __name__ == "__main__":
    main()

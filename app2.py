import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Analysis & Prediction")

st.write(
    "Analyze student performance, identify important factors, "
    "compare machine learning models, and generate academic insights."
)


# ---------------------------------------------------------
# DEMO DATASET
# ---------------------------------------------------------

def create_demo_data():

    np.random.seed(42)

    n = 800

    study_time = np.random.randint(1, 11, n)
    attendance = np.random.randint(50, 101, n)
    previous_score = np.random.randint(40, 96, n)
    sleep_hours = np.random.randint(4, 10, n)

    parental_education = np.random.choice(
        ["High School", "Bachelor", "Master", "PhD"],
        n
    )

    internet_access = np.random.choice(
        ["Yes", "No"],
        n,
        p=[0.85, 0.15]
    )

    extracurricular = np.random.choice(
        ["Yes", "No"],
        n
    )

    tutoring = np.random.choice(
        ["Yes", "No"],
        n,
        p=[0.3, 0.7]
    )

    final_score = (
        20
        + study_time * 2.5
        + attendance * 0.25
        + previous_score * 0.40
        + sleep_hours * 1.2
        + (parental_education == "Bachelor") * 2
        + (parental_education == "Master") * 4
        + (parental_education == "PhD") * 5
        + (internet_access == "Yes") * 2
        + (extracurricular == "Yes") * 1.5
        + (tutoring == "Yes") * 3
        + np.random.normal(0, 5, n)
    )

    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame({
        "Study_Time": study_time,
        "Attendance": attendance,
        "Previous_Score": previous_score,
        "Parental_Education": parental_education,
        "Internet_Access": internet_access,
        "Extracurricular_Activities": extracurricular,
        "Sleep_Hours": sleep_hours,
        "Tutoring": tutoring,
        "Final_Score": np.round(final_score, 2)
    })

    # Add missing values
    df.loc[10, "Study_Time"] = np.nan
    df.loc[20, "Attendance"] = np.nan
    df.loc[30, "Previous_Score"] = np.nan

    # Add duplicate records
    df = pd.concat(
        [df, df.iloc[[5, 10, 15]]],
        ignore_index=True
    )

    return df


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Student Dataset",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    source = "Uploaded Dataset"

else:

    df = create_demo_data()
    source = "Built-in Demo Dataset"


# ---------------------------------------------------------
# CLEAN COLUMN NAMES
# ---------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

raw_rows = len(df)

duplicate_count = df.duplicated().sum()

df = df.drop_duplicates()


# ---------------------------------------------------------
# HANDLE NUMERIC DATA
# ---------------------------------------------------------

numeric_columns = [
    "study_time",
    "attendance",
    "previous_score",
    "sleep_hours",
    "final_score"
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        df[col] = df[col].fillna(
            df[col].median()
        )


# ---------------------------------------------------------
# HANDLE CATEGORICAL DATA
# ---------------------------------------------------------

categorical_columns = [
    "parental_education",
    "internet_access",
    "extracurricular_activities",
    "tutoring"
]

for col in categorical_columns:

    if col in df.columns:

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )


# ---------------------------------------------------------
# PERFORMANCE CATEGORY
# ---------------------------------------------------------

def performance_group(score):

    if score >= 80:
        return "Excellent"

    elif score >= 60:
        return "Good"

    elif score >= 40:
        return "Average"

    else:
        return "Needs Improvement"


df["Performance_Group"] = (
    df["final_score"].apply(performance_group)
)

cleaned_rows = len(df)


# ---------------------------------------------------------
# INFORMATION
# ---------------------------------------------------------

st.info(
    f"📊 Data Source: **{source}** | "
    f"Raw Rows: **{raw_rows}** | "
    f"Cleaned Rows: **{cleaned_rows}**"
)


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

overview, cleaning, eda, ml, insights, downloads = st.tabs(
    [
        "📋 Overview",
        "🧹 Data Cleaning",
        "📊 EDA & Visualizations",
        "🤖 ML Prediction",
        "💡 Insights",
        "⬇️ Downloads"
    ]
)


# =========================================================
# OVERVIEW
# =========================================================

with overview:

    st.header("📋 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Students",
        len(df)
    )

    c2.metric(
        "Average Final Score",
        f"{df['final_score'].mean():.2f}"
    )

    c3.metric(
        "Average Attendance",
        f"{df['attendance'].mean():.2f}%"
    )

    c4.metric(
        "Average Study Time",
        f"{df['study_time'].mean():.2f}"
    )

    st.subheader("👀 Data Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("📊 Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# =========================================================
# DATA CLEANING
# =========================================================

with cleaning:

    st.header("🧹 Data Cleaning")

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Missing Values")

        st.dataframe(
            df.isnull().sum().to_frame(
                "Missing Values"
            ),
            use_container_width=True
        )

    with c2:

        st.subheader("Duplicate Records")

        st.metric(
            "Duplicates Removed",
            duplicate_count
        )

    st.subheader("Cleaning Operations")

    st.success("✓ Duplicate records removed")
    st.success("✓ Missing numerical values handled")
    st.success("✓ Numeric data types corrected")
    st.success("✓ Text values cleaned")
    st.success("✓ Column names standardized")

    st.subheader("Cleaned Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )


# =========================================================
# EDA
# =========================================================

with eda:

    st.header("📊 Exploratory Data Analysis")

    numeric_features = [
        "study_time",
        "attendance",
        "previous_score",
        "sleep_hours",
        "final_score"
    ]

    # Correlation
    st.subheader("🔗 Correlation with Final Score")

    correlation = (
        df[numeric_features]
        .corr()["final_score"]
        .sort_values(ascending=False)
    )

    st.dataframe(
        correlation.to_frame("Correlation"),
        use_container_width=True
    )

    # Study Time
    st.subheader("📚 Study Time vs Final Score")

    fig, ax = plt.subplots()

    sns.scatterplot(
        data=df,
        x="study_time",
        y="final_score",
        ax=ax
    )

    ax.set_xlabel("Study Time")
    ax.set_ylabel("Final Score")

    st.pyplot(fig)
    plt.close(fig)

    # Attendance
    st.subheader("🏫 Attendance vs Final Score")

    fig, ax = plt.subplots()

    sns.scatterplot(
        data=df,
        x="attendance",
        y="final_score",
        ax=ax
    )

    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Final Score")

    st.pyplot(fig)
    plt.close(fig)

    # Previous Score
    st.subheader("📚 Previous Score vs Final Score")

    fig, ax = plt.subplots()

    sns.scatterplot(
        data=df,
        x="previous_score",
        y="final_score",
        ax=ax
    )

    ax.set_xlabel("Previous Score")
    ax.set_ylabel("Final Score")

    st.pyplot(fig)
    plt.close(fig)

    # Parental Education
    st.subheader("👨‍👩‍👧 Parental Education vs Performance")

    parental = (
        df.groupby("parental_education")["final_score"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots()

    sns.barplot(
        x=parental.index,
        y=parental.values,
        ax=ax
    )

    ax.set_xlabel("Parental Education")
    ax.set_ylabel("Average Final Score")

    plt.xticks(rotation=20)

    st.pyplot(fig)
    plt.close(fig)

    # Internet
    st.subheader("🌐 Internet Access vs Performance")

    internet = (
        df.groupby("internet_access")["final_score"]
        .mean()
    )

    fig, ax = plt.subplots()

    sns.barplot(
        x=internet.index,
        y=internet.values,
        ax=ax
    )

    ax.set_xlabel("Internet Access")
    ax.set_ylabel("Average Final Score")

    st.pyplot(fig)
    plt.close(fig)

    # Extracurricular
    st.subheader("🏆 Extracurricular Activities vs Performance")

    extra = (
        df.groupby(
            "extracurricular_activities"
        )["final_score"]
        .mean()
    )

    fig, ax = plt.subplots()

    sns.barplot(
        x=extra.index,
        y=extra.values,
        ax=ax
    )

    ax.set_xlabel("Extracurricular Activities")
    ax.set_ylabel("Average Final Score")

    st.pyplot(fig)
    plt.close(fig)

    # Distribution
    st.subheader("📈 Final Score Distribution")

    fig, ax = plt.subplots()

    sns.histplot(
        df["final_score"],
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Final Score")
    ax.set_ylabel("Number of Students")

    st.pyplot(fig)
    plt.close(fig)

    # Heatmap
    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.heatmap(
        df[numeric_features].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# MACHINE LEARNING
# =========================================================

with ml:

    st.header("🤖 Student Performance Prediction")

    X = df.drop(
        columns=[
            "final_score",
            "Performance_Group"
        ]
    )

    y = df["final_score"]

    categorical_features = (
        X.select_dtypes(
            include=["object"]
        ).columns.tolist()
    )

    numeric_features_ml = (
        X.select_dtypes(
            exclude=["object"]
        ).columns.tolist()
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
            (
                "numeric",
                "passthrough",
                numeric_features_ml
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            random_state=42
        )
    }

    results = {}
    trained_models = {}

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        results[name] = {
            "MAE": mean_absolute_error(
                y_test,
                predictions
            ),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2 Score": r2_score(
                y_test,
                predictions
            )
        }

        trained_models[name] = pipeline

    results_df = pd.DataFrame(results).T

    st.subheader("📊 Model Comparison")

    st.dataframe(
        results_df.round(4),
        use_container_width=True
    )

    best_model_name = (
        results_df["R2 Score"].idxmax()
    )

    st.success(
        f"🏆 Best Model: **{best_model_name}**"
    )

    # R2 graph
    st.subheader("📊 Model R² Comparison")

    fig, ax = plt.subplots()

    sns.barplot(
        x=results_df.index,
        y=results_df["R2 Score"],
        ax=ax
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("R² Score")

    plt.xticks(rotation=20)

    st.pyplot(fig)
    plt.close(fig)

    # Actual vs predicted
    best_model = trained_models[
        best_model_name
    ]

    predictions = best_model.predict(
        X_test
    )

    st.subheader("🎯 Actual vs Predicted Scores")

    fig, ax = plt.subplots()

    ax.scatter(
        y_test,
        predictions,
        alpha=0.7
    )

    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")

    st.pyplot(fig)
    plt.close(fig)

    # Feature importance
    if best_model_name in [
        "Random Forest",
        "Gradient Boosting"
    ]:

        model = best_model.named_steps["model"]

        processor = best_model.named_steps[
            "preprocessor"
        ]

        feature_names = (
            processor
            .get_feature_names_out()
        )

        importance = model.feature_importances_

        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

        feature_importance = (
            feature_importance
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(15)
        )

        st.subheader(
            "⭐ Most Influential Factors"
        )

        st.dataframe(
            feature_importance,
            use_container_width=True
        )

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        sns.barplot(
            data=feature_importance,
            x="Importance",
            y="Feature",
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)


# =========================================================
# INSIGHTS
# =========================================================

with insights:

    st.header("💡 Academic Performance Insights")

    correlation = (
        df[numeric_features]
        .corr()["final_score"]
        .drop("final_score")
    )

    strongest_factor = (
        correlation.abs().idxmax()
    )

    st.info(
        f"📌 Strongest numerical factor related to "
        f"final score: **{strongest_factor.replace('_', ' ').title()}**"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Avg Study Time",
        f"{df['study_time'].mean():.2f}"
    )

    c2.metric(
        "Avg Attendance",
        f"{df['attendance'].mean():.2f}%"
    )

    c3.metric(
        "Avg Previous Score",
        f"{df['previous_score'].mean():.2f}"
    )

    c4.metric(
        "Avg Final Score",
        f"{df['final_score'].mean():.2f}"
    )

    st.subheader("📊 Performance Distribution")

    st.dataframe(
        df["Performance_Group"]
        .value_counts()
        .to_frame("Number of Students"),
        use_container_width=True
    )

    st.subheader("🔎 Key Observations")

    st.write(
        "• Study time can be analyzed as an indicator of academic performance."
    )

    st.write(
        "• Attendance helps measure student participation and engagement."
    )

    st.write(
        "• Previous scores provide useful information for predicting future performance."
    )

    st.write(
        "• Parental education, internet access and extracurricular activities "
        "can be compared with student outcomes."
    )

    st.write(
        "• Machine learning models can predict final performance using multiple factors."
    )


# =========================================================
# DOWNLOADS
# =========================================================

with downloads:

    st.header("⬇️ Download Results")

    cleaned_csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Cleaned Dataset",
        cleaned_csv,
        "cleaned_student_performance.csv",
        "text/csv"
    )

    st.download_button(
        "⬇️ Download Model Comparison",
        results_df.to_csv().encode("utf-8"),
        "model_comparison.csv",
        "text/csv"
    )

    st.success(
        "Your analysis files are ready to download."
    )
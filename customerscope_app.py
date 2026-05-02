import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
from datetime import datetime
import os

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from PIL import Image, UnidentifiedImageError 
def apply_global_styles():
    st.markdown(
        """
        <style>
        /* Background + main container */
        .stApp {
            background: radial-gradient(circle at top left, #0f172a 0, #020617 45%, #000000 100%);
        }
        .block-container {
            max-width: 900px;
            padding-top: 2.5rem;
            padding-bottom: 2.5rem;
        }

        /* Hide default Streamlit header bar */
        header {visibility: hidden;}

        /* Auth card styling */
        .auth-card {
            margin-top: 1.2rem;
            padding: 1.4rem 1.8rem 1.8rem;
            border-radius: 1.4rem;
            background: rgba(15,23,42,0.98);
            border: 1px solid rgba(148,163,184,0.35);
            box-shadow: 0 18px 55px rgba(15,23,42,0.9);
        }

        /* Tabs spacing & look */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.75rem;
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            background: transparent;
            color: #e5e7eb;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #0ea5e9, #22c55e);
            color: #020617 !important;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 0.6rem;
        }

        /* Inputs rounded */
        .stTextInput>div>div>input {
            border-radius: 999px;
        }
        .stTextInput>div>div {
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.7);
        }
        .stTextInput>div>div:focus-within {
            border-color: #0ea5e9;
            box-shadow: 0 0 0 1px #0ea5e9;
        }

        /* Buttons */
        button[kind="primary"], button[kind="secondary"] {
            border-radius: 999px !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# CONFIG & SECURITY SETTINGS


# Safe logo loading for page icon
def get_page_icon():
    logo_path = "customerscope_logo.png"
    if os.path.exists(logo_path):
        try:
            return Image.open(logo_path)
        except UnidentifiedImageError:
            # File exists but is not a valid image
            return "📊"
        except Exception:
            # Any other unexpected error
            return "📊"
    else:
        # File not found
        return "📊"

st.set_page_config(
    page_title="CustomerScope – Customer Segmentation & Intelligence",
    page_icon=get_page_icon(),   # use safe loader
    layout="wide"
)


# Email domain rules (edit these for your real institute/company)
INSTITUTIONAL_EMAIL_DOMAINS = [
    "example.com",      # e.g. company.com or college.edu
    "lloyd.edu.in",
]

STUDENT_EMAIL_DOMAINS = [
    ".in",  # example student domain
    "stu.lloyd.edu.in",
    "gmail.com",    
    "googlemail.com",          
]



# AUTHENTICATION LAYER


AUTH_DB = "auth.db"


def init_auth_db():
    conn = sqlite3.connect(AUTH_DB)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def register_user(full_name: str, email: str, password: str, role: str, account_type: str):
    """
    account_type: 'institutional' or 'student'
    """
    email = email.strip().lower()
    domain = email.split("@")[-1] if "@" in email else ""

    if account_type == "institutional":
        allowed_domains = INSTITUTIONAL_EMAIL_DOMAINS
        label = "institutional / staff"
    else:
        allowed_domains = STUDENT_EMAIL_DOMAINS
        label = "student"

    if allowed_domains and domain not in allowed_domains:
        return (
            False,
            f"Registration restricted: use {label} email "
            f"(allowed domains: {', '.join(allowed_domains)})",
        )

    try:
        conn = sqlite3.connect(AUTH_DB)
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO users (full_name, email, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (full_name, email, hash_password(password), role, datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
        return True, "Registration successful. You can now log in."
    except sqlite3.IntegrityError:
        return False, "This email is already registered."


def authenticate_user(email: str, password: str):
    email = email.strip().lower()
    conn = sqlite3.connect(AUTH_DB)
    c = conn.cursor()
    c.execute(
        "SELECT id, full_name, email, role, password_hash FROM users WHERE email = ?",
        (email,),
    )
    row = c.fetchone()
    conn.close()
    if row is None:
        return None

    user_id, full_name, email, role, stored_hash = row
    if stored_hash == hash_password(password):
        return {"id": user_id, "full_name": full_name, "email": email, "role": role}
    return None


def init_session_state():
    if "user" not in st.session_state:
        st.session_state.user = None
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None
    if "df_processed" not in st.session_state:
        st.session_state.df_processed = None
    if "cluster_model" not in st.session_state:
        st.session_state.cluster_model = None
    if "scaled_features" not in st.session_state:
        st.session_state.scaled_features = None
    if "scaler" not in st.session_state:
        st.session_state.scaler = None
    if "rfm_summary" not in st.session_state:
        st.session_state.rfm_summary = None
    if "churn_model" not in st.session_state:
        st.session_state.churn_model = None
    if "churn_features" not in st.session_state:
        st.session_state.churn_features = None
    if "cluster_summary" not in st.session_state:
        st.session_state.cluster_summary = None
    if "segment_mapping" not in st.session_state:
        st.session_state.segment_mapping = None

    # Numeric columns and selected cluster features
    if "numeric_cols" not in st.session_state:
        st.session_state.numeric_cols = None
    if "cluster_features" not in st.session_state:
        st.session_state.cluster_features = None


def logout():
    st.session_state.user = None


def show_auth_page():
    logo_path = "customerscope_logo.png"

    # Create two-column layout (logo left, form right)
    col_logo, col_form = st.columns([1, 2])

    
    # LEFT COLUMN → Logo (Sidebar style)
    
    with col_logo:
        st.markdown("<br><br><br>", unsafe_allow_html=True)  # vertical spacing

        if os.path.exists(logo_path):
            try:
                st.image(logo_path, width=180)   # Bigger, centered
            except Exception:
                st.markdown("<h1>📊</h1>", unsafe_allow_html=True)
        else:
            st.markdown("<h1>📊</h1>", unsafe_allow_html=True)

        st.markdown(
            """
            <p style='text-align:center; color:#cbd5f5; font-size:0.9rem; margin-top:0.5rem'>
            © CustomerScope
            </p>
            """,
            unsafe_allow_html=True
        )

    
    # RIGHT COLUMN → Title + Form
    
    with col_form:
        st.markdown("<br>", unsafe_allow_html=True)

        # Title
        st.markdown(
            """
            <h1 style="margin-bottom:0.2rem; font-size:2.3rem;">
                CustomerScope
            </h1>
            <p style="color:#9ca3af; margin-top:0;">
                AI-Powered Customer Intelligence for E-Commerce
            </p>
            """,
            unsafe_allow_html=True,
        )

        # Card with tabs inside right column
        st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["🔐 Login", "🆕 Register"])

        # ------------- LOGIN TAB ----------------
        with tab_login:
            st.subheader("Sign in to your organization workspace")

            email = st.text_input("Institution / company email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            remember_me = st.checkbox("Remember me on this device", value=True)

            if st.button("Login", type="primary", use_container_width=True):
                user = authenticate_user(email, password)
                if user:
                    st.session_state.user = user
                    st.success(f"Welcome back, {user['full_name']} 👋")
                else:
                    st.error("Invalid email or password")

            st.caption("Forgot password? Contact your system administrator.")

        # ------------- REGISTER TAB ----------------
        with tab_register:
            st.subheader("Create an account")

            account_type_label = st.radio(
                "Account type",
                ["Institutional account", "Student account"],
                horizontal=True,
            )

            if account_type_label == "Institutional account":
                account_type = "institutional"
                email_label = "Institution / staff email"
                role = st.selectbox("Role", ["Admin", "Manager", "Analyst"])
                domain_hint = ", ".join(INSTITUTIONAL_EMAIL_DOMAINS)
            else:
                account_type = "student"
                email_label = "Student email"
                role = "Student"
                domain_hint = ", ".join(STUDENT_EMAIL_DOMAINS)

            full_name = st.text_input("Full Name", key="reg_name")
            email_reg = st.text_input(email_label, key="reg_email", help=f"Allowed: {domain_hint}")
            password_reg = st.text_input("Password", type="password", key="reg_password")
            password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm")

            if st.button("Register", type="secondary", use_container_width=True):
                if password_reg != password_confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(full_name, email_reg, password_reg, role, account_type)
                    st.success(msg) if ok else st.error(msg)

        st.markdown("</div>", unsafe_allow_html=True)

# DATA LOADING & PREPROCESS


REQUIRED_COLUMNS = [
    "CustomerID",
    "Age",
    "Gender",
    "Annual Income",
    "Spending Score",
    "Purchase Frequency",
    "Recency",
    "Average Order Value",
    "Loyalty Points",
]


def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df


def preprocess_data(df: pd.DataFrame):
    # Ensure required columns exist
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    numeric_cols = [
        "Age",
        "Annual Income",
        "Spending Score",
        "Purchase Frequency",
        "Recency",
        "Average Order Value",
        "Loyalty Points",
    ]
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=numeric_cols)

    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    df_clean = df_clean.dropna(subset=numeric_cols)

    return df_clean



# CLUSTERING & RFM ANALYSIS


CLUSTER_FEATURES = [
    "Annual Income",
    "Spending Score",
    "Purchase Frequency",
    "Recency",
    "Loyalty Points",
]


def perform_kmeans(df: pd.DataFrame, feature_cols, n_clusters: int = 4):
    df_model = df.copy()
    X = df_model[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,  # compatible across sklearn versions
    )
    labels = kmeans.fit_predict(X_scaled)

    df_model["Cluster"] = labels

    if n_clusters > 1 and len(df_model) > n_clusters:
        sil = silhouette_score(X_scaled, labels)
    else:
        sil = np.nan

    centroids_scaled = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids, columns=feature_cols)
    centroids_df["Cluster"] = range(n_clusters)

    return df_model, kmeans, scaler, X_scaled, sil, centroids_df


def compute_rfm(df: pd.DataFrame):
    rfm_df = df.copy()
    rfm_df["R"] = rfm_df["Recency"]
    rfm_df["F"] = rfm_df["Purchase Frequency"]

    if "Average Order Value" in rfm_df.columns:
        rfm_df["M"] = rfm_df["Average Order Value"] * rfm_df["Purchase Frequency"]
    else:
        rfm_df["M"] = rfm_df["Spending Score"]

    rfm_summary = (
        rfm_df.groupby("Cluster")[["R", "F", "M"]]
        .agg(["mean"])
        .round(2)
    )
    return rfm_summary


def derive_segment_labels(df_model: pd.DataFrame, n_clusters: int):
    summary = (
        df_model.groupby("Cluster")[
            ["Annual Income", "Spending Score", "Purchase Frequency", "Recency"]
        ]
        .mean()
        .round(2)
    )

    income_rank = summary["Annual Income"].rank(ascending=False)
    spend_rank = summary["Spending Score"].rank(ascending=False)
    freq_rank = summary["Purchase Frequency"].rank(ascending=False)
    recency_rank = summary["Recency"].rank(ascending=True)

    total_rank = income_rank + spend_rank + freq_rank + recency_rank
    ranked_clusters = total_rank.sort_values().index.tolist()

    segment_names = ["Champions", "Loyal Customers", "At Risk", "Lost"]
    segment_mapping = {}

    for idx, cluster_id in enumerate(ranked_clusters):
        if idx < len(segment_names):
            segment_mapping[cluster_id] = segment_names[idx]
        else:
            segment_mapping[cluster_id] = f"Segment {cluster_id}"

    return summary, segment_mapping


def generate_segment_insight(row, segment_name: str):
    if segment_name == "Champions":
        return f"{segment_name}: Very high income & spending, frequent and recent shoppers — treat them as VIP premium customers."
    if segment_name == "Loyal Customers":
        return f"{segment_name}: Good income, consistent spending, moderate recency — nurture with loyalty programs and personalized offers."
    if segment_name == "At Risk":
        return f"{segment_name}: Previously good spenders but less recent and less frequent — re-engage with win-back campaigns and surveys."
    if segment_name == "Lost":
        return f"{segment_name}: Low frequency and very high recency — consider low-cost reactivation or let them churn gracefully."
    return f"{segment_name}: Mixed behaviour segment — analyze deeper for tailored strategy."



# CHURN PREDICTION


def label_churn(df: pd.DataFrame):
    df = df.copy()
    rec_q75 = df["Recency"].quantile(0.75)
    freq_q25 = df["Purchase Frequency"].quantile(0.25)

    df["ChurnLabel"] = np.where(
        (df["Recency"] >= rec_q75) & (df["Purchase Frequency"] <= freq_q25),
        1,
        0,
    )
    return df


def train_churn_model(df_labeled: pd.DataFrame):
    features = [
        "Annual Income",
        "Spending Score",
        "Purchase Frequency",
        "Recency",
        "Loyalty Points",
    ]
    X = df_labeled[features].values
    y = df_labeled["ChurnLabel"].values

    if len(np.unique(y)) < 2:
        return None, None, "Only one churn class present. Not enough variation to train a model."

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, scaler, report


def predict_churn_for_single(model, scaler, input_dict):
    features = [
        "Annual Income",
        "Spending Score",
        "Purchase Frequency",
        "Recency",
        "Loyalty Points",
    ]
    x = np.array([[input_dict[f] for f in features]])
    x_scaled = scaler.transform(x)
    prob = model.predict_proba(x_scaled)[0, 1]
    if prob < 0.33:
        category = "Low"
    elif prob < 0.66:
        category = "Medium"
    else:
        category = "High"
    return prob, category



# UI COMPONENTS


def app_header():
    st.markdown(
        """
        <style>
        .app-header {
            padding: 0.75rem 1.5rem;
            border-radius: 1rem;
            background: linear-gradient(90deg, #0f172a, #0b7285);
            color: #f9fafb;
            margin-bottom: 1.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    user = st.session_state.user
    with st.container():
        st.markdown("<div class='app-header'>", unsafe_allow_html=True)
        cols = st.columns([0.8, 3.2, 2, 1])
        with cols[0]:
            logo_path = "customerscope_logo.png"
            if os.path.exists(logo_path):
              try:
                  st.image(logo_path, width=60)
              except Exception:
               st.warning("Logo file could not be loaded in header. Check image format.")
        with cols[1]:
            st.markdown(
                "<h2 style='margin:0'>📊 CustomerScope</h2>"
                "<p style='margin:0;color:#e5e7eb'>AI-Powered Customer Intelligence for E-Commerce</p>",
                unsafe_allow_html=True,
            )
        with cols[2]:
            if user:
                st.markdown(
                    f"<p style='text-align:right;margin:0'>Signed in as <b>{user['full_name']}</b> "
                    f"({user['role']})</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='text-align:right;margin:0;color:#cbd5f5'>{user['email']}</p>",
                    unsafe_allow_html=True,
                )
        with cols[3]:
            st.write("")
            if st.button("Logout", key="logout_btn"):
                logout()
                st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def sidebar_menu():
    # 🔹 Sidebar Logo
    logo_path = "customerscope_logo.png"  # Make sure the file is in the same folder as customerscope_app.py
    if os.path.exists(logo_path):
        st.sidebar.image(
            logo_path,
            caption="CustomerScope",
            use_container_width=True
        )
    else:
        st.sidebar.markdown("## 📊 CustomerScope")

    st.sidebar.caption("AI-Powered Customer Intelligence")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")

    page = st.sidebar.radio(
        "Go to",
        [
            "Home / Overview",
            "Phase 1: Visualization Analysis",
            "Phase 2: Numerical Analysis",
            "Churn Prediction",
            "Marketing Insights",
            "Download Center",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("🔒 Internal use only – company/institute data")

    return page


def check_data_uploaded():
    if st.session_state.df_raw is None or st.session_state.df_processed is None:
        st.info("Please upload a valid `customer_data.csv` file on the **Home / Overview** page first.")
        return False
    return True


def page_home():
    st.subheader("Home / Overview")
    st.write(
        "Upload your **customer_data.csv** and get an end-to-end segmentation, RFM insights, churn risk, and marketing recommendations."
    )

    with st.expander("📁 Upload Customer Data", expanded=True):
        uploaded = st.file_uploader(
            "Upload customer_data.csv",
            type=["csv"],
            accept_multiple_files=False,
            help="File should contain CustomerID, Age, Gender, Annual Income, Spending Score, Purchase Frequency, Recency, Average Order Value, Loyalty Points",
        )

        if uploaded is not None:
            try:
                df_raw = load_data(uploaded)
                df_processed = preprocess_data(df_raw)

                st.session_state.df_raw = df_raw
                st.session_state.df_processed = df_processed

                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
                st.session_state.numeric_cols = numeric_cols

                st.success(f"File loaded successfully. {df_processed.shape[0]} clean rows available for analysis.")

                st.markdown("#### Data Preview")
                st.dataframe(df_processed.head())

                st.markdown("#### Column Summary")
                st.write(pd.DataFrame({
                    "Column": df_processed.columns,
                    "Non-Null Count": df_processed.notna().sum(),
                    "Dtype": df_processed.dtypes.astype(str),
                }))
            except Exception as e:
                st.error(f"Error while processing file: {e}")

    if st.session_state.df_processed is not None:
        df = st.session_state.df_processed
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Customers", len(df))
        with c2:
            st.metric("Avg Annual Income", f"{df['Annual Income'].mean():.2f}")
        with c3:
            st.metric("Avg Spending Score", f"{df['Spending Score'].mean():.2f}")
        with c4:
            st.metric("Avg Frequency", f"{df['Purchase Frequency'].mean():.2f}")


def page_phase1():
    st.subheader("Phase 1: Visualization Analysis")

    if not check_data_uploaded():
        return

    df = st.session_state.df_processed

    st.markdown("#### Basic Distributions")
    num_cols = ["Age", "Annual Income", "Spending Score"]

    cols = st.columns(3)
    for idx, col in enumerate(num_cols):
        with cols[idx]:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"{col} Distribution")
            st.pyplot(fig)

    st.markdown("#### Correlation Heatmap")
    corr_cols = [
        "Age",
        "Annual Income",
        "Spending Score",
        "Purchase Frequency",
        "Recency",
        "Average Order Value",
        "Loyalty Points",
    ]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.markdown("#### 2D Scatter Plot – Annual Income vs Spending Score")
    fig2d = px.scatter(
        df,
        x="Annual Income",
        y="Spending Score",
        hover_data=["CustomerID"],
    )
    st.plotly_chart(fig2d, use_container_width=True)

    st.markdown("#### 3D Scatter Plot – Income vs Spending Score vs Loyalty Points")
    fig3d = px.scatter_3d(
        df,
        x="Annual Income",
        y="Spending Score",
        z="Loyalty Points",
        hover_name="CustomerID",
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.info(
        "Once clustering is run in **Phase 2**, cluster labels and segment colors will show up on these plots for deeper insight."
    )


def page_phase2():
    st.subheader("Phase 2: Numerical Analysis – K-Means Clustering")

    if not check_data_uploaded():
        return

    df = st.session_state.df_processed.copy()

    k = st.slider("Number of clusters (K)", 2, 10, 4, help="Default K=4 → Champions, Loyal, At Risk, Lost")

    if st.session_state.numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.session_state.numeric_cols = numeric_cols
    else:
        numeric_cols = st.session_state.numeric_cols

    exclude_cols = ["CustomerID", "ChurnLabel"]
    feature_options = [c for c in numeric_cols if c not in exclude_cols]

    st.markdown("#### Select features for clustering")
    default_features = (
        st.session_state.cluster_features
        if st.session_state.cluster_features
        else [c for c in CLUSTER_FEATURES if c in feature_options] or feature_options[:5]
    )

    selected_features = st.multiselect(
        "Choose the columns you want to use for K-Means",
        options=feature_options,
        default=default_features,
        help="These features will be scaled and used to create clusters.",
    )

    if not selected_features:
        st.warning("Please select at least one feature for clustering.")
        return

    if st.button("Run K-Means Clustering", type="primary"):
        df_model, kmeans, scaler, X_scaled, sil, centroids_df = perform_kmeans(
            df, feature_cols=selected_features, n_clusters=k
        )
        st.session_state.cluster_model = kmeans
        st.session_state.scaler = scaler
        st.session_state.scaled_features = X_scaled
        st.session_state.df_processed = df_model
        st.session_state.cluster_features = selected_features

        summary, seg_mapping = derive_segment_labels(df_model, k)
        st.session_state.cluster_summary = summary
        st.session_state.segment_mapping = seg_mapping

        rfm_summary = compute_rfm(df_model)
        st.session_state.rfm_summary = rfm_summary

        st.success("Clustering completed successfully!")
        if not np.isnan(sil):
            st.metric("Silhouette Score", f"{sil:.3f}")
        else:
            st.info("Silhouette score not available (insufficient data).")

    if st.session_state.cluster_model is None or "Cluster" not in st.session_state.df_processed.columns:
        st.info("Run K-Means clustering first using the button above.")
        return

    df_model = st.session_state.df_processed
    kmeans = st.session_state.cluster_model
    summary = st.session_state.cluster_summary
    segment_mapping = st.session_state.segment_mapping
    rfm_summary = st.session_state.rfm_summary
    selected_features = st.session_state.cluster_features

    st.markdown("#### Cluster Centroids (Original Scale)")
    centroids_scaled = kmeans.cluster_centers_
    centroids = st.session_state.scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids, columns=selected_features)
    centroids_df["Cluster"] = range(len(centroids_df))
    st.dataframe(centroids_df.round(2))

    st.markdown("#### Cluster-wise Summary")
    summary_disp = summary.copy()
    summary_disp["Segment"] = summary_disp.index.map(segment_mapping)
    st.dataframe(summary_disp)

    st.markdown("#### Customer Distribution by Cluster")
    cluster_counts = df_model["Cluster"].value_counts().sort_index()
    total = len(df_model)
    dist_df = pd.DataFrame({
        "Cluster": cluster_counts.index,
        "Count": cluster_counts.values,
        "Percentage": (cluster_counts.values / total * 100).round(2),
        "Segment": [segment_mapping[c] for c in cluster_counts.index],
    })
    st.dataframe(dist_df)

    fig_bar = px.bar(
        dist_df,
        x="Segment",
        y="Count",
        text="Percentage",
        title="Segment Distribution",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### 2D Scatter with Cluster Coloring")
    if "Annual Income" in df_model.columns and "Spending Score" in df_model.columns:
        fig2d = px.scatter(
            df_model,
            x="Annual Income",
            y="Spending Score",
            color=df_model["Cluster"].astype(str),
            hover_data=["CustomerID"],
            labels={"color": "Cluster"},
        )
        st.plotly_chart(fig2d, use_container_width=True)
    else:
        st.info("2D scatter (Income vs Spending Score) is skipped because those columns are not available.")

    st.markdown("#### 3D Scatter with Cluster Coloring")
    if (
        "Annual Income" in df_model.columns
        and "Spending Score" in df_model.columns
        and "Loyalty Points" in df_model.columns
    ):
        fig3d = px.scatter_3d(
            df_model,
            x="Annual Income",
            y="Spending Score",
            z="Loyalty Points",
            color=df_model["Cluster"].astype(str),
            hover_name="CustomerID",
        )
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.info("3D scatter (Income, Spending, Loyalty) is skipped because some of those columns are not available.")

    st.markdown("#### RFM-style Summary per Cluster")
    st.dataframe(rfm_summary)

    st.markdown("#### Actionable Insights per Segment")
    for cluster_id, row in summary.iterrows():
        segment_name = segment_mapping[cluster_id]
        insight = generate_segment_insight(row, segment_name)
        st.write(f"- **Cluster {cluster_id} – {segment_name}:** {insight}")


def page_churn():
    st.subheader("Churn Prediction")

    if not check_data_uploaded():
        return

    if st.session_state.cluster_model is None:
        st.info("Run clustering in **Phase 2** first. It will enrich data with better behaviour features.")

    df = st.session_state.df_processed

    st.markdown("#### Auto-label churn based on Recency & Frequency")
    df_labeled = label_churn(df)
    churn_rate = df_labeled["ChurnLabel"].mean() * 100
    st.metric("Estimated Churn Rate", f"{churn_rate:.1f}%")

    if st.button("Train Churn Risk Model", type="primary"):
        model, scaler, report = train_churn_model(df_labeled)
        if model is None:
            st.error(report)
        else:
            st.session_state.churn_model = model
            st.session_state.churn_features = scaler
            st.success("Churn model trained successfully.")

            st.markdown("##### Evaluation (Classification Report)")
            rep_df = pd.DataFrame(report).T
            st.dataframe(rep_df)

    if st.session_state.churn_model is None:
        st.info("Train the churn model above, then you can predict risk for new customers.")
        return

    st.markdown("#### Predict Churn Risk for a New Customer")
    col1, col2 = st.columns(2)
    with col1:
        annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
        spending_score = st.number_input("Spending Score", min_value=0.0, max_value=100.0, value=50.0)
        loyalty = st.number_input("Loyalty Points", min_value=0.0, value=100.0, step=10.0)
    with col2:
        freq = st.number_input("Purchase Frequency (orders per period)", min_value=0.0, value=5.0)
        rec = st.number_input("Recency (days since last purchase)", min_value=0.0, value=30.0)

    if st.button("Predict Churn Risk", type="primary"):
        model = st.session_state.churn_model
        scaler = st.session_state.churn_features
        inputs = {
            "Annual Income": annual_income,
            "Spending Score": spending_score,
            "Purchase Frequency": freq,
            "Recency": rec,
            "Loyalty Points": loyalty,
        }
        prob, category = predict_churn_for_single(model, scaler, inputs)
        st.success(
            f"Estimated churn probability: **{prob:.2%}** → **{category} risk**"
        )

        if category == "High":
            st.warning("Consider immediate win-back campaigns, personal outreach, and strong incentives.")
        elif category == "Medium":
            st.info("Monitor closely, send targeted offers, and keep engagement high.")
        else:
            st.success("Customer seems stable. Focus on retention & value-added experiences.")


def page_marketing_insights():
    st.subheader("Marketing Insights & Recommendations")

    if not check_data_uploaded():
        return
    if st.session_state.cluster_model is None or st.session_state.cluster_summary is None:
        st.info("Run clustering in **Phase 2** first to generate marketing insights.")
        return

    summary = st.session_state.cluster_summary
    seg_mapping = st.session_state.segment_mapping
    df_model = st.session_state.df_processed

    st.markdown("#### Segment-level Marketing Playbook")

    strategies = {
        "Champions": [
            "Launch VIP reward programs and exclusive early access to new products.",
            "Offer premium bundles and upsell complementary high-value items.",
            "Invite them to referral programs to bring similar high-value customers.",
        ],
        "Loyal Customers": [
            "Send personalized recommendations based on past purchases.",
            "Use birthday/anniversary offers to strengthen relationship.",
            "Introduce tiered loyalty levels to encourage more frequent purchases.",
        ],
        "At Risk": [
            "Trigger win-back email flows with limited-time discounts.",
            "Ask for feedback to understand why engagement dropped.",
            "Offer tailored bundles that match previously purchased categories.",
        ],
        "Lost": [
            "Use low-cost, high-impact channels like email/SMS for reactivation.",
            "Send one or two strong offers, then stop communications to avoid cost.",
            "Analyze reasons for churn (pricing, product fit, competition).",
        ],
    }

    for cluster_id, row in summary.iterrows():
        segment = seg_mapping[cluster_id]
        st.markdown(f"### Cluster {cluster_id}: {segment}")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Avg Income", f"{row['Annual Income']:.1f}")
        with c2:
            st.metric("Avg Spending Score", f"{row['Spending Score']:.1f}")
        with c3:
            st.metric("Avg Frequency", f"{row['Purchase Frequency']:.1f}")
        with c4:
            st.metric("Avg Recency (days)", f"{row['Recency']:.1f}")

        count = (df_model["Cluster"] == cluster_id).sum()
        pct = count / len(df_model) * 100
        st.caption(f"{count} customers in this segment ({pct:.1f}% of base).")

        st.markdown("**Recommended Strategies:**")
        for s in strategies.get(segment, ["Analyze this segment further for tailored actions."]):
            st.write(f"- {s}")

        st.markdown("---")


def page_download_center():
    st.subheader("Download Center")

    if not check_data_uploaded():
        return
    df = st.session_state.df_processed

    if "Cluster" not in df.columns:
        st.info("Run clustering in **Phase 2** first to add cluster labels to the dataset.")
        return

    seg_mapping = st.session_state.segment_mapping

    df_export = df.copy()
    df_export["Segment"] = df_export["Cluster"].map(seg_mapping)

    st.markdown("#### Download Cluster-labeled Full Dataset")
    csv_full = df_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download full dataset with clusters & segments (CSV)",
        data=csv_full,
        file_name="customerscope_segments_full.csv",
        mime="text/csv",
    )

    st.markdown("#### Download At-Risk / Lost Customers")
    mask_at_risk = df_export["Segment"].isin(["At Risk", "Lost"])
    df_risky = df_export[mask_at_risk]
    if len(df_risky) > 0:
        csv_risky = df_risky.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download At-Risk & Lost customers only (CSV)",
            data=csv_risky,
            file_name="customerscope_at_risk_lost.csv",
            mime="text/csv",
        )
    else:
        st.info("No customers currently classified as At Risk or Lost.")

    if st.session_state.rfm_summary is not None:
        st.markdown("#### Download RFM Summary")
        rfm_csv = st.session_state.rfm_summary.to_csv().encode("utf-8")
        st.download_button(
            "⬇️ Download RFM summary per cluster (CSV)",
            data=rfm_csv,
            file_name="customerscope_rfm_summary.csv",
            mime="text/csv",
        )
    else:
        st.info("RFM summary not available yet – run clustering in Phase 2.")



# MAIN ENTRY POINT


def main():
    init_auth_db()
    init_session_state()
    apply_global_styles()

    if st.session_state.user is None:
        show_auth_page()
        return

    app_header()
    page = sidebar_menu()

    if page == "Home / Overview":
        page_home()
    elif page == "Phase 1: Visualization Analysis":
        page_phase1()
    elif page == "Phase 2: Numerical Analysis":
        page_phase2()
    elif page == "Churn Prediction":
        page_churn()
    elif page == "Marketing Insights":
        page_marketing_insights()
    elif page == "Download Center":
        page_download_center()


if __name__ == "__main__":
    main()

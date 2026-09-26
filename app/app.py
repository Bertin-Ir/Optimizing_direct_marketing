import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Follow-up prioritization", page_icon="🏦", layout="wide")


# Folder containing this file, so the app finds its files wherever it is launched from
APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "model.joblib")


@st.cache_data
def load_info():
    with open(APP_DIR / "model_info.json") as f:
        return json.load(f)


model = load_model()
info = load_info()
OPTS = info["options"]
BASE_RATE = info["base_rate"]
CUTOFF = info["decision_threshold"]
USES_DURATION = info["uses_duration"]
RAW_COLUMNS = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
               "contact", "day", "month", "campaign", "pdays", "previous", "poutcome"]
if USES_DURATION:
    RAW_COLUMNS.append("duration")

MONTH_LABELS = {"jan": "January", "feb": "February", "mar": "March", "apr": "April",
                "may": "May", "jun": "June", "jul": "July", "aug": "August",
                "sep": "September", "oct": "October", "nov": "November", "dec": "December"}
POUTCOME_LABELS = {"success": "Subscribed", "failure": "Declined",
                   "other": "Other outcome", "unknown": "Never contacted"}


def prepare_features(clients: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the model's input columns exactly as in the training notebook."""
    df = clients.copy()
    for col in ["default", "housing", "loan"]:
        df[col] = df[col].map({"yes": 1, "no": 0})
    df["high_balance"] = (df["balance"] > info["avg_balance"]).astype(int)
    df["no_loans"] = ((df["loan"] == 0) & (df["housing"] == 0)).astype(int)
    if USES_DURATION:
        df["long_duration"] = (df["duration"] > info["avg_duration"]).astype(int)
    return df[info["input_columns"]]


def predict(df: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(prepare_features(df))[:, 1], index=df.index)


st.title("Optimizing Direct Bank Marketing")
st.write(
    "Banks sell term deposits (savings locked in for a fixed period) through phone campaigns, "
    f"but most calls fail: only {BASE_RATE:.1%} of contacted clients subscribe. "
    "Calling everyone back wastes staff time and annoys clients who are not interested. "
    "This project uses machine learning on 45,211 real campaign calls to help the bank "
    "focus its effort on the clients most likely to say yes."
)

st.header("Which clients should the bank follow up with?")
st.write(
    "After a first call, the model predicts whether the client will subscribe, "
    "so the team can spend its follow-up calls on the most promising clients."
)

tab_single, tab_batch, tab_threshold, tab_about = st.tabs(
    ["Score a completed call", "Pick a cut-off", "Rank called clients", "About the model"]
)

# ---------------------------------------------------------------- single client
with tab_single:
    with st.form("client"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Client")
            age = st.number_input("Age", 18, 95, 40)
            job = st.selectbox("Job", OPTS["job"], index=OPTS["job"].index("management"))
            marital = st.selectbox("Marital status", OPTS["marital"], index=OPTS["marital"].index("married"))
            education = st.selectbox("Education", OPTS["education"], index=OPTS["education"].index("tertiary"))
        with c2:
            st.subheader("Finances")
            balance = st.number_input("Average yearly balance (€)", -10000, 100000, 1500, step=100)
            housing = st.radio("Housing loan", ["no", "yes"], horizontal=True)
            loan = st.radio("Personal loan", ["no", "yes"], horizontal=True)
            default = st.radio("Credit in default", ["no", "yes"], horizontal=True)
        with c3:
            st.subheader("The call")
            if USES_DURATION:
                duration = st.number_input("Call duration (seconds)", 0, 5000, 180, step=10)
            contact = st.selectbox("Channel", OPTS["contact"])
            month = st.selectbox("Month of the call", OPTS["month"],
                                 index=OPTS["month"].index("may"),
                                 format_func=lambda m: MONTH_LABELS[m])
            day = st.slider("Day of the call", 1, 31, 15)
            campaign = st.number_input("Calls to this client in this campaign (including this one)", 1, 60, 1)
            poutcome = st.selectbox("Previous campaign", OPTS["poutcome"],
                                    index=OPTS["poutcome"].index("unknown"),
                                    format_func=lambda p: POUTCOME_LABELS[p])
            if poutcome == "unknown":
                previous, pdays = 0, -1
            else:
                previous = st.number_input("Contacts before this campaign", 1, 300, 1)
                pdays = st.number_input("Days since last contact", 0, 900, 90)

        submitted = st.form_submit_button("Predict outcome", type="primary")

    if submitted:
        client = pd.DataFrame([{
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "balance": balance, "housing": housing, "loan": loan,
            "contact": contact, "day": day, "month": month, "campaign": campaign,
            "pdays": pdays, "previous": previous, "poutcome": poutcome,
        }])
        if USES_DURATION:
            client["duration"] = duration
        p = predict(client).iloc[0]
        m0, m1 = st.columns(2)
        m0.metric("Prediction", "Will subscribe" if p >= CUTOFF else "Will not subscribe")
        m1.metric("Subscription score", f"{p:.2f}")
        st.caption(
            f"Clients scoring {CUTOFF:.2f} or higher are predicted to subscribe."
            
        )
        if p >= 0.8:
            st.success("Strong follow-up candidate. Prioritize a callback.")
        elif p >= CUTOFF:
            st.info("Likely to subscribe. Follow up if there is capacity.")
        else:
            st.warning("Unlikely to subscribe. A follow-up is probably low-return.")
        st.caption("Tip: change only the call duration to see how strongly engagement on the call drives the score.")


# ------------------------------------------------------------------ batch rank
with tab_batch:
    st.write(
        "Upload a CSV of clients already called in this campaign, one row per client, "
        "using the same columns as the UCI Bank Marketing data (including `duration`, "
        "and text values such as `yes`/`no`). Extra columns are kept."
    )
    with open(APP_DIR / "sample_clients.csv", "rb") as f:
        st.download_button("Download a sample file (300 clients)", f,
                           file_name="sample_clients.csv", mime="text/csv")

    uploaded = st.file_uploader("Client list", type="csv")
    use_sample = st.checkbox("Use the sample file instead")

    clients = None
    if uploaded is not None:
        raw = uploaded.getvalue().decode("utf-8", errors="replace")
        sep = ";" if raw.splitlines()[0].count(";") > raw.splitlines()[0].count(",") else ","
        clients = pd.read_csv(io.StringIO(raw), sep=sep)
    elif use_sample:
        clients = pd.read_csv(APP_DIR / "sample_clients.csv")

    if clients is not None:
        missing = [c for c in RAW_COLUMNS if c not in clients.columns]
        if missing:
            st.error(f"The file is missing these columns: {', '.join(missing)}. "
                     "Download the sample file to see the expected format.")
        else:
            ranked = clients.copy()
            ranked.insert(0, "score", predict(ranked).round(4))
            ranked.insert(1, "prediction", (ranked["score"] >= CUTOFF).map(
                {True: "Will subscribe", False: "Will not subscribe"}))
            ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
            ranked.index = ranked.index + 1

            top_n = st.slider("How many follow-up calls can the team make?", 1, len(ranked),
                              min(50, len(ranked)))
            n_yes_all = int((ranked["score"] >= CUTOFF).sum())
            n_yes_top = int((ranked["score"].head(top_n) >= CUTOFF).sum())
            a, b = st.columns(2)
            a.metric("Predicted to subscribe in the whole list", f"{n_yes_all} of {len(ranked)}")
            b.metric(f"Predicted to subscribe in your top {top_n}", f"{n_yes_top} of {top_n}")
            st.caption(f"For comparison, about {BASE_RATE * top_n:.0f} of {top_n} clients picked at random would subscribe.")

            st.dataframe(
                ranked.head(top_n),
                column_config={"score": st.column_config.ProgressColumn(
                    "Subscription score", format="%.2f", min_value=0.0, max_value=1.0)},
                width="stretch",
            )
            st.download_button("Download the ranked list",
                               ranked.to_csv(index_label="rank").encode(),
                               file_name="ranked_clients.csv", mime="text/csv")

# ------------------------------------------------------------------ threshold
with tab_threshold:
    st.write(
        "A cut-off turns scores into a follow-up list: everyone above it gets a callback. "
        "A low cut-off finds more subscribers but wastes more calls; a high one is "
        "efficient but misses people. These figures come from 9,000+ held-out clients."
    )
    table = pd.DataFrame(info["thresholds"]).dropna()
    t = st.select_slider("Follow up with clients scoring at least",
                         options=table["threshold"].tolist(),
                         value=min(table["threshold"], key=lambda v: abs(v - CUTOFF)),
                         format_func=lambda v: f"{v:.2f}")
    row = table.loc[table["threshold"] == t].iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Share of clients followed up", f"{row.share_called:.1%}")
    k2.metric("Follow-ups that convert (precision)", f"{row.precision:.1%}",
              f"{row.precision / BASE_RATE:.1f}× the average")
    k3.metric("Subscribers reached (recall)", f"{row.recall:.1%}")

    chart = table.set_index("threshold")[["precision", "recall", "share_called"]].rename(
        columns={"precision": "Precision", "recall": "Recall", "share_called": "Share followed up"})
    st.line_chart(chart, x_label="Cut-off", y_label="Rate")

# ---------------------------------------------------------------------- about
with tab_about:
    duration_note = (
        "**Why it scores completed calls.** Call duration is the strongest predictor in the data, since long "
        "calls usually signal interest. It is only known once a call ends, so this app is built for "
        "prioritizing follow-ups after a first call, not for choosing whom to call first. Without duration, "
        "the same model reaches a ROC-AUC of about 0.79."
        if USES_DURATION else
        "**What the model does not use.** Call duration. It is the strongest predictor in the data, but it is "
        "only known after the call ends, so it cannot help decide whom to call. With it, ROC-AUC reaches about "
        "0.93, which would overstate how well the model works before a call."
    )
    st.markdown(f"""
**Data.** UCI Bank Marketing dataset: 45,211 phone-campaign contacts from a Portuguese bank [Moro et al., 2011].

**Model.** XGBoost with SMOTE oversampling, trained on {len(info['model_features'])} features selected by variance
threshold. Features, preprocessing, and hyperparameters come from the model comparison in the project notebooks.
On {9043:,} held-out clients (test data): ROC-AUC {info['roc_auc']:.2f}, precision {info['precision']:.0%}, recall {info['recall']:.0%}
at a cut-off of {CUTOFF:.2f}.

{duration_note}

**Limits.** Scores show association, not cause. The data comes from 2008–2010 campaigns at one bank, so
month effects partly reflect that bank's campaign schedule rather than universal seasonality.

Built by Bertin Iradukunda. Analysis and model comparison:
[GitHub repository](https://github.com/Bertin-Ir/Optimizing_direct_marketing).
""")

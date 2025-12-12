"""Streamlit dashboard for exploring the VAR analysis workflow.

The app loads ``df_var_1209.csv`` and provides interactive views for:
- Data overview and basic profiling
- Stationarity checks (ADF)
- VAR(1) model summary with impulse responses
- Scenario-scaled IRF tables
- A lightweight risk dashboard derived from latest observations
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

DATA_PATH = Path(__file__).resolve().parent / "df_var_1209.csv"
DEFAULT_VAR_COLUMNS = [
    "ret_log_1d",
    "oi_close_diff",
    "funding_close",
    "liq_total_usd_diff",
    "taker_buy_ratio",
    "sth_sopr",
    "lth_sopr",
    "sth_realized_price_usd_diff",
    "lth_realized_price_usd_diff",
    "global_m2_yoy_diff",
    "sp500_ret",
    "nasdaq_ret",
    "etf_aum_diff",
]


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    df.index = pd.to_datetime(df.index)
    return df.asfreq("D")


def _read_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


@st.cache_data(show_spinner=False)
def load_data(source: Optional[str] = None) -> pd.DataFrame:
    target = Path(source) if source else DATA_PATH
    return _prepare_dataframe(_read_csv(target))


def compute_adf_tests(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        series = df[col].dropna()
        stat, pvalue, _, _, _, _ = adfuller(series)
        rows.append(
            {
                "metric": col,
                "test_stat": stat,
                "p_value": pvalue,
                "stationary@5%": pvalue < 0.05,
            }
        )
    return pd.DataFrame(rows).set_index("metric")


@st.cache_resource(show_spinner=False)
def fit_var_model(df: pd.DataFrame, columns: List[str], lags: int):
    model = VAR(df[columns].dropna())
    return model.fit(lags)


def build_coef_table(var_result) -> pd.DataFrame:
    params = var_result.params
    params.index.name = "equation"
    params.columns.name = "lag/const"
    return params.round(4)


def impulse_response_series(var_result, impulse: str, response: str, horizon: int) -> pd.DataFrame:
    irf = var_result.irf(horizon)
    data = irf.irfs[:, var_result.names.index(response), var_result.names.index(impulse)]
    return pd.DataFrame({"day": np.arange(1, horizon + 1), "impact_pct": data[:horizon] * 100})


def scaled_irf_table(
    irf_values: np.ndarray,
    var_names: List[str],
    response_name: str,
    impulse_name: str,
    shock_list: Iterable[float],
    horizons: Iterable[int],
    label_prefix: str,
) -> pd.DataFrame:
    i_resp = var_names.index(response_name)
    i_imp = var_names.index(impulse_name)

    rows = []
    for shock in shock_list:
        row = {}
        for h in horizons:
            irf_h = irf_values[h - 1, i_resp, i_imp]
            delta_pct = irf_h * shock * 100.0
            row[f"h={h}d"] = delta_pct
        rows.append(pd.Series(row, name=f"{label_prefix}_shock={shock: .2e}"))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def make_scenario_tables(var_result, horizons: Iterable[int]):
    irf = var_result.irf(max(horizons))
    irf_values = irf.irfs
    var_names = var_result.names

    liq_shocks = [1e8, 5e8, 1e9]
    taker_shocks = [0.01, 0.05, 0.10]
    etf_shocks = [1e8, 5e8, 1e9]

    fe_irf_liq = scaled_irf_table(
        irf_values,
        var_names,
        response_name="ret_log_1d",
        impulse_name="liq_total_usd_diff",
        shock_list=liq_shocks,
        horizons=horizons,
        label_prefix="liq",
    )

    fe_irf_taker = scaled_irf_table(
        irf_values,
        var_names,
        response_name="ret_log_1d",
        impulse_name="taker_buy_ratio",
        shock_list=taker_shocks,
        horizons=horizons,
        label_prefix="taker",
    )

    fe_irf_etf = scaled_irf_table(
        irf_values,
        var_names,
        response_name="ret_log_1d",
        impulse_name="etf_aum_diff",
        shock_list=etf_shocks,
        horizons=horizons,
        label_prefix="etf",
    )

    combined = pd.concat([fe_irf_liq, fe_irf_taker, fe_irf_etf]).round(4)
    return combined


# Dashboard helpers

def signal_oi(oi_diff: float) -> str:
    if oi_diff > 1_000_000_000:
        return "🔴 HIGH (레버리지 과열)"
    if oi_diff > 300_000_000:
        return "🟡 CAUTION (레버리지 증가)"
    return "🟢 NORMAL"


def signal_funding(funding: float) -> str:
    if abs(funding) > 0.0003:
        return "🔴 HIGH (극단적 펀딩)"
    if abs(funding) > 0.0001:
        return "🟡 CAUTION"
    return "🟢 NORMAL"


def signal_liq(liq_usd: float) -> str:
    liq_abs = abs(liq_usd)
    if liq_abs > 200_000_000:
        return "🔴 HIGH (대규모 청산)"
    if liq_abs > 50_000_000:
        return "🟡 CAUTION (청산 확대)"
    return "🟢 NORMAL"


def signal_taker(taker_ratio: float) -> str:
    if taker_ratio > 0.60 or taker_ratio < 0.40:
        return "🔴 HIGH (매수/매도 한쪽 쏠림)"
    if taker_ratio > 0.55 or taker_ratio < 0.45:
        return "🟡 CAUTION (편향 존재)"
    return "🟢 NORMAL"


def signal_m2(m2_diff: float) -> str:
    if m2_diff < 0:
        return "🔴 TIGHT (유동성 축소)"
    if m2_diff < 0.01:
        return "🟡 NEUTRAL"
    return "🟢 LOOSE (유동성 확대)"


def score_from_signal(signal: str) -> int:
    if signal.startswith("🔴"):
        return 2
    if signal.startswith("🟡"):
        return 1
    return 0


def summarize_signals(latest: pd.Series) -> Dict[str, str]:
    sig_oi = signal_oi(latest["oi_close_diff"])
    sig_fund = signal_funding(latest["funding_close"])
    sig_liq = signal_liq(latest["liq_total_usd_diff"])
    sig_taker = signal_taker(latest["taker_buy_ratio"])
    sig_m2 = signal_m2(latest["global_m2_yoy_diff"])
    total_score = sum(
        [score_from_signal(sig) for sig in [sig_oi, sig_fund, sig_liq, sig_taker, sig_m2]]
    )
    if total_score >= 6:
        overall = "🔴 HIGH RISK (단기 변동성·청산 리스크 매우 큼)"
    elif total_score >= 3:
        overall = "🟡 CAUTION (포지션 관리 필요)"
    else:
        overall = "🟢 NORMAL (구조적 과열 신호 약함)"
    return {
        "oi": sig_oi,
        "funding": sig_fund,
        "liq": sig_liq,
        "taker": sig_taker,
        "m2": sig_m2,
        "overall": overall,
    }


def render_dashboard(latest: pd.Series):
    signals = summarize_signals(latest)
    st.subheader("Risk Dashboard")
    st.markdown(f"**기준일:** {latest.name.date()}")
    cols = st.columns(2)
    with cols[0]:
        st.write("[ 레버리지 구조 ]")
        st.write(f"OI 변화량: {signals['oi']} (값: {latest['oi_close_diff']:,.0f})")
        st.write(f"Funding Rate: {signals['funding']} (값: {latest['funding_close']:.5f})")
        st.write(f"Liquidations: {signals['liq']} (값: {latest['liq_total_usd_diff']:,.0f})")
    with cols[1]:
        st.write("[ 시장 흐름 / 유동성 ]")
        st.write(f"Taker Buy Ratio: {signals['taker']} (값: {latest['taker_buy_ratio']:.3f})")
        st.write(f"Global M2 YoY Diff: {signals['m2']} (값: {latest['global_m2_yoy_diff']:.3f})")
    st.success(f"종합 위험도: {signals['overall']}")


def main():
    st.set_page_config(page_title="VAR Explorer", layout="wide")
    st.title("📈 VAR 기반 비트코인 리스크 대시보드")
    st.caption("df_var_1209.csv를 활용한 시각화 데모")

    st.sidebar.header("데이터 입력")
    user_file = st.sidebar.file_uploader("CSV 업로드 (선택)", type=["csv"])
    source_label = "업로드 파일" if user_file else "기본 df_var_1209.csv"
    try:
        df = load_data(user_file) if user_file else load_data()
    except Exception as exc:  # pragma: no cover - Streamlit UI path
        st.error(f"데이터를 불러오지 못했습니다: {exc}")
        return

    st.sidebar.header("모델 옵션")
    lag_order = st.sidebar.number_input("VAR 차수(lag)", min_value=1, max_value=5, value=1)
    horizon = st.sidebar.slider("IRF Horizon (days)", min_value=3, max_value=20, value=10)

    st.sidebar.caption(f"데이터 소스: {source_label}")
    available_columns = [c for c in DEFAULT_VAR_COLUMNS if c in df.columns]
    missing_columns = sorted(set(DEFAULT_VAR_COLUMNS) - set(available_columns))
    selected_columns: Sequence[str] = st.sidebar.multiselect(
        "VAR에 사용할 컬럼",
        options=available_columns,
        default=available_columns,
        help="기본 추천 컬럼 중 데이터에 존재하는 항목만 표시됩니다.",
    )

    if missing_columns:
        st.sidebar.warning(
            "데이터에 없는 컬럼: " + ", ".join(missing_columns) + " (자동으로 제외됨)",
            icon="⚠️",
        )

    tabs = st.tabs(["Data", "Stationarity", "VAR & IRF", "Scenario", "Dashboard"])

    with tabs[0]:
        st.subheader("데이터 미리보기")
        st.write(
            "데이터는 일 단위로 맞춰져 있으며, 결측치는 VAR 학습 전에 자동으로 제거됩니다."
        )
        if not selected_columns:
            st.error("사용할 수 있는 컬럼이 없습니다. CSV에 필요한 컬럼을 포함시켜 주세요.")
            return

        st.dataframe(df[selected_columns].tail(), use_container_width=True)
        base_col = "ret_log_1d" if "ret_log_1d" in df.columns else selected_columns[0]
        stats = {
            "행 개수": len(df),
            "시작일": df.index.min().date(),
            "종료일": df.index.max().date(),
            f"결측 비율 ({base_col})": f"{df[base_col].isna().mean():.1%}",
        }
        st.json(stats)
        st.plotly_chart(
            px.line(df.reset_index(), x="time", y="ret_log_1d", title="일간 로그 수익률"),
            use_container_width=True,
        )

    with tabs[1]:
        st.subheader("ADF 정상성 테스트")
        adf_table = compute_adf_tests(df, selected_columns)
        st.dataframe(adf_table, use_container_width=True)
        st.info("p-value < 0.05 이면 5% 유의수준에서 정상성(stationary)으로 간주합니다.")

    with tabs[2]:
        st.subheader("VAR 모델 요약")
        var_result = fit_var_model(df, selected_columns, lag_order)
        highlights = {
            "AIC": round(var_result.aic, 3),
            "BIC": round(var_result.bic, 3),
            "LogLik": round(var_result.llf, 3),
        }
        st.json(highlights)
        st.markdown("**계수 테이블**")
        st.dataframe(build_coef_table(var_result), use_container_width=True)

        st.markdown("**Impulse Response (단위 shock 기준)**")
        impulse = st.selectbox("Impulse 변수", var_result.names, index=var_result.names.index("liq_total_usd_diff"))
        response = st.selectbox("Response 변수", var_result.names, index=var_result.names.index("ret_log_1d"))
        irf_df = impulse_response_series(var_result, impulse, response, horizon)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=irf_df["day"],
                y=irf_df["impact_pct"],
                mode="lines+markers",
                name=f"{impulse} → {response}",
            )
        )
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(
            xaxis_title="Days after shock",
            yaxis_title="Price impact (%)",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("시나리오 기반 IRF")
        var_result = fit_var_model(df, selected_columns, lag_order)
        horizons = list(range(1, horizon + 1))
        scenario_table = make_scenario_tables(var_result, horizons)
        st.dataframe(scenario_table, use_container_width=True)

        st.markdown("**상위 충격 시리즈 라인 차트**")
        fig = go.Figure()
        for idx, row in scenario_table.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=horizons,
                    y=row.values,
                    mode="lines+markers",
                    name=idx,
                )
            )
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(
            xaxis_title="Days after shock",
            yaxis_title="Impact (%)",
            template="plotly_white",
            legend_orientation="h",
            legend_y=-0.2,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.subheader("최신 시점 리스크 대시보드")
        latest = df[selected_columns].dropna().iloc[-1]
        render_dashboard(latest)


if __name__ == "__main__":
    main()

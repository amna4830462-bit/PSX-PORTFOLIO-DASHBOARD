import json
import os
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


MENU_OPTIONS = [
    "Show Portfolio",
    "Add Stock",
    "Risk Analysis",
    "Future Predictions & Suggestions",
    "Exit",
]


def ensure_streamlit_mode():
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is None:
        subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)], check=False)
        raise SystemExit(0)


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_symbol(symbol):
    return symbol.upper().replace(".KA", "").strip()


def yahoo_symbol(symbol):
    clean = symbol.upper().strip()
    return clean if clean.endswith(".KA") else f"{clean}.KA"


def extract_close(payload):
    if isinstance(payload, dict):
        for key in ("data", "rows", "result"):
            nested = payload.get(key)
            if nested is not None:
                value = extract_close(nested)
                if value:
                    return value
        for key in ("close", "price", "ldcp", "current"):
            value = to_float(payload.get(key))
            if value and value > 0:
                return value
        return None

    if isinstance(payload, list):
        for item in reversed(payload):
            value = extract_close(item)
            if value:
                return value
            if isinstance(item, (list, tuple)) and len(item) > 1:
                value = to_float(item[1])
                if value and value > 0:
                    return value
    return None


def get_psx_price(symbol):
    clean = normalize_symbol(symbol)
    if not clean:
        raise ValueError("Ticker is empty.")

    url = f"https://dps.psx.com.pk/timeseries/eod/{clean}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))

    price = extract_close(payload)
    if not price:
        raise ValueError("PSX response did not contain a valid close price.")
    return clean, float(price)


def get_market_price(symbol):
    try:
        clean, price = get_psx_price(symbol)
        return clean, price, "PSX"
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
        y_symbol = yahoo_symbol(symbol)
        hist = yf.Ticker(y_symbol).history(period="5d")
        if hist.empty:
            raise ValueError("No recent market data found from PSX or Yahoo.")
        return y_symbol, float(hist["Close"].iloc[-1]), "Yahoo"


def get_price_history(symbol, period):
    hist = yf.Ticker(yahoo_symbol(symbol)).history(period=period)
    if hist.empty:
        return pd.Series(dtype=float)
    return hist["Close"].dropna()


def get_prediction(symbol):
    close = get_price_history(symbol, "6mo")
    if len(close) < 60:
        return None

    x = np.arange(len(close), dtype=float)
    y = close.values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(len(close), len(close) + 7, dtype=float)
    future_values = slope * future_x + intercept

    fitted = slope * x + intercept
    residual_std = float(np.std(y - fitted))

    current = float(close.iloc[-1])
    day7 = float(future_values[-1])
    lower = day7 - 1.96 * residual_std
    upper = day7 + 1.96 * residual_std

    move_pct = ((day7 - current) / current) * 100 if current else 0
    if move_pct > 2:
        trend, suggestion = "Bullish", "Hold / Buy on dips"
    elif move_pct < -2:
        trend, suggestion = "Bearish", "Reduce risk"
    else:
        trend, suggestion = "Sideways", "Hold"

    return {
        "trend": trend,
        "suggestion": suggestion,
        "current": current,
        "predicted_day7": day7,
        "lower": float(lower),
        "upper": float(upper),
        "history": close.tail(60),
        "future_values": future_values.astype(float),
    }


def refresh_portfolio_prices():
    for item in st.session_state.portfolio:
        try:
            _, price, source = get_market_price(item["symbol"])
            item["current"] = price
            item["source"] = source
        except Exception:
            pass


def portfolio_df():
    df = pd.DataFrame(st.session_state.portfolio)
    if df.empty:
        return df
    df["total_value"] = df["qty"] * df["current"]
    df["cost"] = df["qty"] * df["buy"]
    df["pnl"] = df["total_value"] - df["cost"]
    return df


def render_add_stock():
    st.header("Add Stock to Portfolio")
    symbol = st.text_input("Enter Stock Symbol (e.g., HBL or HBL.KA):").strip().upper()
    qty = st.number_input("Quantity:", min_value=1, value=1, step=1)
    buy_price = st.number_input("Buy Price per Share:", min_value=0.01, value=100.0, step=0.01)

    if not st.button("Add Stock"):
        return
    if not symbol:
        st.error("Please enter a stock symbol.")
        return

    try:
        clean, current, source = get_market_price(symbol)
        existing = next((x for x in st.session_state.portfolio if x["symbol"] == clean), None)
        if existing:
            existing["qty"] += int(qty)
            existing["buy"] = float(buy_price)
            existing["current"] = current
            existing["source"] = source
        else:
            st.session_state.portfolio.append(
                {"symbol": clean, "qty": int(qty), "buy": float(buy_price), "current": current, "source": source}
            )
        st.success(f"{clean} added/updated. Current: {current:.2f} ({source})")
    except Exception as e:
        st.error(f"Could not fetch price: {e}")


def render_show_portfolio():
    st.header("Your Portfolio")
    if not st.session_state.portfolio:
        st.info("Portfolio is empty. Add stocks first.")
        return

    refresh_portfolio_prices()
    df = portfolio_df()

    st.dataframe(
        df.rename(
            columns={
                "symbol": "Stock",
                "qty": "Quantity",
                "buy": "Buy Price",
                "current": "Current Price",
                "source": "Price Source",
                "total_value": "Total Value",
                "pnl": "Profit/Loss",
            }
        ),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio Value", f"{df['total_value'].sum():,.2f}")
    c2.metric("Total Cost", f"{df['cost'].sum():,.2f}")
    c3.metric("Net P/L", f"{df['pnl'].sum():,.2f}")

    fig = px.pie(df, names="symbol", values="total_value", title="Portfolio Distribution")
    st.plotly_chart(fig, use_container_width=True)


def render_risk_analysis():
    st.header("Portfolio Risk Analysis")
    if not st.session_state.portfolio:
        st.info("Portfolio is empty. Add stocks first.")
        return

    refresh_portfolio_prices()
    rows = []
    for item in st.session_state.portfolio:
        close = get_price_history(item["symbol"], "3mo")
        if len(close) < 20:
            rows.append({"Stock": item["symbol"], "Volatility %": None, "Risk Level": "Insufficient Data"})
            continue

        vol_pct = float(close.pct_change().dropna().std() * 100)
        if vol_pct < 1.5:
            level = "Low"
        elif vol_pct < 3:
            level = "Medium"
        else:
            level = "High"
        rows.append({"Stock": item["symbol"], "Volatility %": round(vol_pct, 2), "Risk Level": level})

    risk_df = pd.DataFrame(rows)
    st.dataframe(risk_df, use_container_width=True)
    fig = px.bar(risk_df.dropna(), x="Stock", y="Volatility %", color="Risk Level", title="Risk by Stock")
    st.plotly_chart(fig, use_container_width=True)


def render_predictions():
    st.header("Future Predictions & Suggestions")
    if not st.session_state.portfolio:
        st.info("Portfolio is empty. Add stocks first.")
        return

    for item in st.session_state.portfolio:
        symbol = item["symbol"]
        st.subheader(symbol)
        pred = get_prediction(symbol)

        if pred is None:
            st.warning("Not enough historical data for prediction.")
            continue

        st.write(
            f"Current: {pred['current']:.2f} | Predicted (7d): {pred['predicted_day7']:.2f} | "
            f"95% Range: {pred['lower']:.2f} - {pred['upper']:.2f}"
        )
        st.info(f"Trend: {pred['trend']} | Suggestion: {pred['suggestion']}")

        hist_len = len(pred["history"])
        chart_df = pd.DataFrame(
            {
                "Day": list(range(1, hist_len + 1)) + list(range(hist_len + 1, hist_len + 8)),
                "Price": list(pred["history"].values) + list(pred["future_values"]),
                "Type": (["Historical"] * hist_len) + (["Predicted"] * 7),
            }
        )
        fig = px.line(chart_df, x="Day", y="Price", color="Type", title=f"{symbol} Historical + 7-Day Forecast")
        st.plotly_chart(fig, use_container_width=True)


def render_exit():
    st.header("Exit")
    st.write("Dashboard is running. Close the browser tab to exit.")


def main():
    ensure_streamlit_mode()
    st.set_page_config(page_title="PSX Portfolio Dashboard", layout="wide")
    st.title("PSX Portfolio Dashboard")

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

    option = st.sidebar.radio("Choose Option:", MENU_OPTIONS)
    if option == "Add Stock":
        render_add_stock()
    elif option == "Show Portfolio":
        render_show_portfolio()
    elif option == "Risk Analysis":
        render_risk_analysis()
    elif option == "Future Predictions & Suggestions":
        render_predictions()
    else:
        render_exit()


if __name__ == "__main__":
    main()

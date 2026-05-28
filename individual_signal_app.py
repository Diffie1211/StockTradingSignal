import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


# ============================================================
# SETTINGS
# ============================================================

MARKET_FILTER_SYMBOL = "QQQ"

LOOKBACK_DAYS = 260
CROSS_LOOKBACK_DAYS = 3
STOP_LOSS_PCT = 0.10
MAX_DISTANCE_ABOVE_MA20 = 1.20
WATCHLIST_MIN_BUY_SCORE = 4

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


# ============================================================
# LOAD KEYS
# ============================================================

load_dotenv()

def get_secret(name: str, default=None):
    env_value = os.getenv(name)
    if env_value:
        return env_value

    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


api_key = get_secret("ALPACA_API_KEY")
secret_key = get_secret("ALPACA_SECRET_KEY")

SEC_USER_AGENT = get_secret(
    "SEC_USER_AGENT",
    "DiffiesStockSignalScanner/1.0 diffie@example.com",
)

if not api_key or not secret_key:
    st.error("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env file.")
    st.stop()

data_client = StockHistoricalDataClient(
    api_key=api_key,
    secret_key=secret_key,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Diffie's Stock Signal Scanner",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# DATA FUNCTIONS
# ============================================================

@st.cache_data(ttl=300)
def get_daily_bars(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    symbol = symbol.strip().upper()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    last_error = None

    for feed in [DataFeed.SIP, DataFeed.IEX]:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=feed,
            )

            bars = data_client.get_stock_bars(request).df

            if bars.empty:
                continue

            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.reset_index()

            bars = bars[bars["symbol"] == symbol].copy()
            bars = bars.sort_values("timestamp").reset_index(drop=True)

            if not bars.empty:
                return bars

        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not load daily bars for {symbol}. Last error: {last_error}")


def remove_unfinished_daily_candle(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    ny_time = datetime.now(ZoneInfo("America/New_York"))
    latest_time = pd.to_datetime(df.iloc[-1]["timestamp"])
    latest_date = latest_time.date()

    if latest_date == ny_time.date() and ny_time.hour < 16:
        return df.iloc[:-1].copy()

    if latest_date == ny_time.date() and ny_time.hour == 16 and ny_time.minute < 10:
        return df.iloc[:-1].copy()

    return df


# ============================================================
# INDICATORS
# ============================================================

def add_kdj(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()

    denominator = high_max - low_min
    denominator = denominator.where(denominator != 0)

    rsv = (df["close"] - low_min) / denominator * 100
    rsv = rsv.fillna(50)

    df["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    df["D"] = df["K"].ewm(alpha=1 / 3, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]

    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df["MA10"] = df["close"].rolling(10).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    return df


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_unfinished_daily_candle(df)
    df = add_kdj(df)
    df = add_macd(df)
    df = add_moving_averages(df)
    return df


# ============================================================
# SIGNAL FUNCTIONS
# ============================================================

def cross_up(prev_a, prev_b, now_a, now_b) -> bool:
    return prev_a <= prev_b and now_a > now_b


def cross_down(prev_a, prev_b, now_a, now_b) -> bool:
    return prev_a >= prev_b and now_a < now_b


def crossed_up_within_last_n_days(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    n: int = CROSS_LOOKBACK_DAYS,
) -> bool:
    recent = df.iloc[-(n + 1):].copy()

    for i in range(1, len(recent)):
        prev = recent.iloc[i - 1]
        now = recent.iloc[i]

        if cross_up(prev[col_a], prev[col_b], now[col_a], now[col_b]):
            return True

    return False


def crossed_down_today(df: pd.DataFrame, col_a: str, col_b: str) -> bool:
    previous = df.iloc[-2]
    latest = df.iloc[-1]

    return cross_down(
        previous[col_a],
        previous[col_b],
        latest[col_a],
        latest[col_b],
    )


def three_consecutive_down_closes(df: pd.DataFrame) -> bool:
    c4 = df.iloc[-4]["close"]
    c3 = df.iloc[-3]["close"]
    c2 = df.iloc[-2]["close"]
    c1 = df.iloc[-1]["close"]

    return c3 < c4 and c2 < c3 and c1 < c2


def suggested_action(signal: str) -> str:
    if signal == "BUY SIGNAL":
        return "Consider manual buy next trading day around 9:45 AM ET using a limit order."
    if signal == "SELL SIGNAL":
        return "Review your holding and consider selling manually."
    if signal == "WATCHLIST":
        return "Almost ready. Watch the next completed daily candle."
    if signal == "HOLD":
        return "You marked this as holding, and no sell rule triggered."
    return "No action. Wait for a cleaner setup."


# ============================================================
# SEC FUNDAMENTAL FUNCTIONS
# ============================================================

def sec_headers() -> dict:
    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }


def sec_headers_for_ticker_file() -> dict:
    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }


@st.cache_data(ttl=86400)
def get_sec_ticker_map() -> dict:
    response = requests.get(
        SEC_COMPANY_TICKERS_URL,
        headers=sec_headers_for_ticker_file(),
        timeout=20,
    )
    response.raise_for_status()

    raw = response.json()
    ticker_map = {}

    for _, item in raw.items():
        ticker = item["ticker"].upper()
        cik = str(item["cik_str"]).zfill(10)
        title = item.get("title", "")
        ticker_map[ticker] = {
            "cik": cik,
            "title": title,
        }

    return ticker_map


@st.cache_data(ttl=86400)
def get_companyfacts(symbol: str) -> dict:
    symbol = symbol.strip().upper()
    ticker_map = get_sec_ticker_map()

    if symbol not in ticker_map:
        raise RuntimeError(f"No SEC CIK found for {symbol}. Fundamentals may not be available.")

    cik = ticker_map[symbol]["cik"]
    url = SEC_COMPANYFACTS_URL.format(cik=cik)

    response = requests.get(
        url,
        headers=sec_headers(),
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()
    data["_scanner_cik"] = cik
    data["_scanner_company_name"] = ticker_map[symbol]["title"]

    return data


def get_us_gaap_facts(companyfacts: dict) -> dict:
    return companyfacts.get("facts", {}).get("us-gaap", {})


def find_fact_series(
    companyfacts: dict,
    tag_candidates: list[str],
    unit_candidates: list[str],
    form_prefixes: tuple[str, ...] = ("10-K", "10-Q"),
) -> tuple[str | None, str | None, list[dict]]:
    us_gaap = get_us_gaap_facts(companyfacts)

    for tag in tag_candidates:
        if tag not in us_gaap:
            continue

        units = us_gaap[tag].get("units", {})

        for unit in unit_candidates:
            if unit not in units:
                continue

            items = []

            for item in units[unit]:
                form = item.get("form", "")
                value = item.get("val")

                if value is None:
                    continue

                if not any(form.startswith(prefix) for prefix in form_prefixes):
                    continue

                items.append(item)

            if items:
                return tag, unit, items

    return None, None, []


def latest_fact(
    companyfacts: dict,
    tag_candidates: list[str],
    unit_candidates: list[str],
    form_prefixes: tuple[str, ...] = ("10-K", "10-Q"),
) -> dict | None:
    tag, unit, items = find_fact_series(
        companyfacts,
        tag_candidates,
        unit_candidates,
        form_prefixes,
    )

    if not items:
        return None

    items = sorted(
        items,
        key=lambda x: (x.get("filed", ""), x.get("end", "")),
        reverse=True,
    )

    item = items[0].copy()
    item["_tag"] = tag
    item["_unit"] = unit
    return item


def annual_series(
    companyfacts: dict,
    tag_candidates: list[str],
    unit_candidates: list[str],
) -> list[dict]:
    tag, unit, items = find_fact_series(
        companyfacts,
        tag_candidates,
        unit_candidates,
        form_prefixes=("10-K",),
    )

    if not items:
        return []

    annual_items = []

    for item in items:
        fp = item.get("fp", "")
        form = item.get("form", "")

        if not form.startswith("10-K"):
            continue

        if fp and fp != "FY":
            continue

        copied = item.copy()
        copied["_tag"] = tag
        copied["_unit"] = unit
        annual_items.append(copied)

    latest_by_year = {}

    for item in annual_items:
        fiscal_year = item.get("fy")

        if fiscal_year is None:
            fiscal_year = pd.to_datetime(item.get("end")).year

        key = int(fiscal_year)
        existing = latest_by_year.get(key)

        if existing is None:
            latest_by_year[key] = item
        else:
            if item.get("filed", "") > existing.get("filed", ""):
                latest_by_year[key] = item

    result = list(latest_by_year.values())
    result = sorted(result, key=lambda x: int(x.get("fy", 0)))

    return result


def latest_two_annual_values(
    companyfacts: dict,
    tag_candidates: list[str],
    unit_candidates: list[str],
) -> tuple[dict | None, dict | None]:
    series = annual_series(companyfacts, tag_candidates, unit_candidates)

    if len(series) == 0:
        return None, None

    if len(series) == 1:
        return series[-1], None

    return series[-1], series[-2]


def pct_change(latest, previous):
    if latest is None or previous is None:
        return None

    if previous == 0:
        return None

    return (latest - previous) / abs(previous)


def safe_val(item):
    if not item:
        return None

    return item.get("val")


def format_money(value) -> str:
    if value is None:
        return "N/A"

    value = float(value)
    abs_value = abs(value)

    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"

    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"

    if abs_value >= 1_000:
        return f"${value / 1_000:.2f}K"

    return f"${value:.2f}"


def format_number(value) -> str:
    if value is None:
        return "N/A"

    return f"{float(value):,.2f}"


def format_pct(value) -> str:
    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


def build_fundamental_summary(symbol: str) -> dict:
    companyfacts = get_companyfacts(symbol)

    company_name = companyfacts.get("_scanner_company_name", "")
    cik = companyfacts.get("_scanner_cik", "")

    revenue_tags = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ]

    net_income_tags = ["NetIncomeLoss"]
    operating_income_tags = ["OperatingIncomeLoss"]
    operating_cash_flow_tags = ["NetCashProvidedByUsedInOperatingActivities"]
    assets_tags = ["Assets"]
    liabilities_tags = ["Liabilities"]
    equity_tags = [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ]
    eps_tags = ["EarningsPerShareDiluted"]

    revenue_latest, revenue_previous = latest_two_annual_values(companyfacts, revenue_tags, ["USD"])
    net_income_latest, net_income_previous = latest_two_annual_values(companyfacts, net_income_tags, ["USD"])
    operating_income_latest, _ = latest_two_annual_values(companyfacts, operating_income_tags, ["USD"])
    ocf_latest, _ = latest_two_annual_values(companyfacts, operating_cash_flow_tags, ["USD"])

    assets_latest = latest_fact(companyfacts, assets_tags, ["USD"])
    liabilities_latest = latest_fact(companyfacts, liabilities_tags, ["USD"])
    equity_latest = latest_fact(companyfacts, equity_tags, ["USD"])
    eps_latest, eps_previous = latest_two_annual_values(companyfacts, eps_tags, ["USD/shares"])

    revenue_value = safe_val(revenue_latest)
    revenue_prev_value = safe_val(revenue_previous)
    net_income_value = safe_val(net_income_latest)
    net_income_prev_value = safe_val(net_income_previous)
    operating_income_value = safe_val(operating_income_latest)
    ocf_value = safe_val(ocf_latest)

    assets_value = safe_val(assets_latest)
    liabilities_value = safe_val(liabilities_latest)
    equity_value = safe_val(equity_latest)

    eps_value = safe_val(eps_latest)
    eps_prev_value = safe_val(eps_previous)

    revenue_growth = pct_change(revenue_value, revenue_prev_value)
    net_income_growth = pct_change(net_income_value, net_income_prev_value)
    eps_growth = pct_change(eps_value, eps_prev_value)

    net_margin = None
    operating_margin = None

    if revenue_value and revenue_value != 0:
        if net_income_value is not None:
            net_margin = net_income_value / revenue_value

        if operating_income_value is not None:
            operating_margin = operating_income_value / revenue_value

    liability_to_assets = None

    if assets_value and liabilities_value is not None and assets_value != 0:
        liability_to_assets = liabilities_value / assets_value

    comments = []

    if revenue_growth is not None:
        if revenue_growth > 0.10:
            comments.append("Revenue growth looks strong compared with the previous annual period.")
        elif revenue_growth > 0:
            comments.append("Revenue is growing, but not aggressively.")
        else:
            comments.append("Revenue declined compared with the previous annual period.")

    if net_margin is not None:
        if net_margin > 0.15:
            comments.append("Net margin looks strong.")
        elif net_margin > 0:
            comments.append("Company is profitable, but margin is moderate.")
        else:
            comments.append("Company is currently unprofitable on a net income basis.")

    if liability_to_assets is not None:
        if liability_to_assets < 0.60:
            comments.append("Balance sheet leverage looks moderate based on liabilities/assets.")
        elif liability_to_assets < 0.80:
            comments.append("Balance sheet leverage is noticeable but not extreme.")
        else:
            comments.append("Liabilities/assets looks high, so check the balance sheet carefully.")

    if ocf_value is not None:
        if ocf_value > 0:
            comments.append("Operating cash flow is positive.")
        else:
            comments.append("Operating cash flow is negative.")

    return {
        "company_name": company_name,
        "cik": cik,

        "revenue": revenue_value,
        "revenue_prev": revenue_prev_value,
        "revenue_growth": revenue_growth,
        "revenue_fy": revenue_latest.get("fy") if revenue_latest else "N/A",
        "revenue_filed": revenue_latest.get("filed") if revenue_latest else "N/A",

        "net_income": net_income_value,
        "net_income_prev": net_income_prev_value,
        "net_income_growth": net_income_growth,

        "operating_income": operating_income_value,
        "operating_cash_flow": ocf_value,

        "assets": assets_value,
        "liabilities": liabilities_value,
        "equity": equity_value,
        "liability_to_assets": liability_to_assets,

        "eps": eps_value,
        "eps_prev": eps_prev_value,
        "eps_growth": eps_growth,

        "net_margin": net_margin,
        "operating_margin": operating_margin,

        "comments": comments,
    }


# ============================================================
# SCANNER CORE
# ============================================================

def scan_one_stock(symbol: str, entry_price):
    symbol = symbol.strip().upper()

    stock_df = get_daily_bars(symbol)
    qqq_df = get_daily_bars(MARKET_FILTER_SYMBOL)

    stock_df = prepare_indicators(stock_df)
    qqq_df = prepare_indicators(qqq_df)

    if len(stock_df) < 50:
        raise RuntimeError(f"Not enough data for {symbol}.")

    if len(qqq_df) < 50:
        raise RuntimeError("Not enough QQQ data.")

    latest = stock_df.iloc[-1]
    qqq_latest = qqq_df.iloc[-1]

    latest_close = float(latest["close"])
    ma10 = float(latest["MA10"])
    ma20 = float(latest["MA20"])

    qqq_close = float(qqq_latest["close"])
    qqq_ma20 = float(qqq_latest["MA20"])
    qqq_above_ma20 = qqq_close > qqq_ma20

    holding = entry_price is not None

    kdj_cross_recent = crossed_up_within_last_n_days(stock_df, "K", "D")
    macd_cross_recent = crossed_up_within_last_n_days(stock_df, "MACD", "MACD_SIGNAL")
    stock_above_ma20 = latest_close > ma20
    not_too_extended = latest_close <= ma20 * MAX_DISTANCE_ABOVE_MA20

    buy_checks = {
        "KDJ golden cross within last 3 completed daily candles": kdj_cross_recent,
        "MACD golden cross within last 3 completed daily candles": macd_cross_recent,
        "Stock close > stock MA20": stock_above_ma20,
        "QQQ close > QQQ MA20": qqq_above_ma20,
        f"Stock close <= MA20 * {MAX_DISTANCE_ABOVE_MA20}": not_too_extended,
    }

    buy_score = sum(1 for value in buy_checks.values() if value)
    buy_total = len(buy_checks)
    buy_test_pass = buy_score == buy_total

    three_down_days = three_consecutive_down_closes(stock_df)
    macd_cross_down = crossed_down_today(stock_df, "MACD", "MACD_SIGNAL")
    close_below_ma10 = latest_close < ma10

    stop_loss_signal = False
    stop_loss_price = None

    if holding:
        stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
        stop_loss_signal = latest_close <= stop_loss_price

    sell_checks = {
        "Three consecutive down closes": three_down_days,
        "MACD crossed down today": macd_cross_down,
        "Stock close < stock MA10": close_below_ma10,
        "Stop loss hit": stop_loss_signal,
    }

    sell_test_pass = any(sell_checks.values())

    if holding and sell_test_pass:
        final_signal = "SELL SIGNAL"
    elif holding and not sell_test_pass:
        final_signal = "HOLD"
    elif not holding and buy_test_pass:
        final_signal = "BUY SIGNAL"
    elif not holding and buy_score >= WATCHLIST_MIN_BUY_SCORE:
        final_signal = "WATCHLIST"
    else:
        final_signal = "NO ACTION"

    distance_from_ma20_pct = (latest_close / ma20 - 1) * 100
    max_allowed_price = ma20 * MAX_DISTANCE_ABOVE_MA20

    return {
        "symbol": symbol,
        "final_signal": final_signal,
        "holding": holding,
        "entry_price": entry_price,
        "latest_date": latest["timestamp"],
        "latest_close": latest_close,
        "ma10": ma10,
        "ma20": ma20,
        "distance_from_ma20_pct": distance_from_ma20_pct,
        "max_allowed_price": max_allowed_price,
        "qqq_close": qqq_close,
        "qqq_ma20": qqq_ma20,
        "buy_checks": buy_checks,
        "buy_score": buy_score,
        "buy_total": buy_total,
        "sell_checks": sell_checks,
        "sell_test_pass": sell_test_pass,
        "stop_loss_price": stop_loss_price,
        "stock_df": stock_df,
    }


# ============================================================
# UI HELPERS
# ============================================================

def show_strategy_explanation():
    st.markdown("## 🧠 How This Scanner Works")

    st.info(
        "This scanner is a manual trading helper. It does not buy or sell stocks. "
        "It checks whether a stock matches Diffie's daily momentum strategy."
    )

    with st.expander("📌 Strategy Summary", expanded=True):
        st.markdown(
            """
            This strategy looks for stocks that are starting to move upward with momentum, 
            while also avoiding weak market conditions.

            **Buy idea:**  
            The scanner looks for a stock where both **KDJ** and **MACD** recently turned bullish, 
            and the stock is trading above its 20-day moving average.

            **Sell idea:**  
            If you already hold the stock, the scanner checks whether weakness is appearing, 
            such as a MACD bearish cross, price falling below the 10-day moving average, 
            three down days in a row, or a 10% stop-loss.
            """
        )

    with st.expander("✅ Buy Conditions Explained"):
        st.markdown(
            f"""
            A stock gets a **BUY SIGNAL** only when all buy checks pass:

            1. **KDJ golden cross within last 3 daily candles**  
               K moves above D. This suggests short-term momentum is improving.

            2. **MACD golden cross within last 3 daily candles**  
               MACD moves above its signal line. This suggests trend momentum is turning bullish.

            3. **Stock close > 20-day moving average**  
               The stock is trading above its recent trend level.

            4. **QQQ close > QQQ 20-day moving average**  
               The broader tech/growth market is healthy. This helps avoid buying when the market is weak.

            5. **Stock is not too far above MA20**  
               The stock close must be less than or equal to:

               `MA20 × {MAX_DISTANCE_ABOVE_MA20}`

               This helps avoid chasing a stock after it has already moved too far.
            """
        )

    with st.expander("🔻 Sell Conditions Explained"):
        st.markdown(
            f"""
            If you mark the stock as **Holding**, the scanner checks sell conditions.

            A stock gets a **SELL SIGNAL** if any of these happen:

            1. **Three consecutive down closes**  
               The stock closed lower for three completed daily candles in a row.

            2. **MACD crossed down today**  
               MACD moved below its signal line. This can suggest momentum is weakening.

            3. **Stock close < 10-day moving average**  
               The stock has fallen below its short-term trend.

            4. **Stop loss hit**  
               If you enter your entry price, the scanner calculates:

               `Stop loss price = Entry price × {1 - STOP_LOSS_PCT}`

               With the current setting, this is a **10% stop loss**.
            """
        )

    with st.expander("📊 What Each Signal Means"):
        st.markdown(
            """
            **BUY SIGNAL**  
            All buy conditions passed. This means the stock matches the strategy setup.

            **WATCHLIST / Almost Buy**  
            Most buy conditions passed, but not all. The stock may be close to a setup.

            **HOLD**  
            You marked the stock as holding, and no sell rule was triggered.

            **SELL SIGNAL**  
            You marked the stock as holding, and at least one sell rule was triggered.

            **NO ACTION**  
            The stock does not currently match the buy or sell setup.
            """
        )

    with st.expander("⚠️ Important Notes"):
        st.markdown(
            """
            - This scanner is for **manual decision support only**.
            - It does **not** place trades.
            - It does **not** predict the future.
            - A signal does not guarantee profit.
            - The strategy uses **completed daily candles**, not unfinished intraday candles.
            - The fundamental section is only extra reference information and is **not used** in the signal.
            """
        )


def show_signal_card(result: dict):
    signal = result["final_signal"]
    symbol = result["symbol"]

    if signal == "BUY SIGNAL":
        st.success(f"🚀 {symbol}: BUY SIGNAL")
    elif signal == "SELL SIGNAL":
        st.error(f"🔻 {symbol}: SELL SIGNAL")
    elif signal == "WATCHLIST":
        st.warning(f"👀 {symbol}: WATCHLIST / Almost Buy")
    elif signal == "HOLD":
        st.info(f"🟦 {symbol}: HOLD")
    else:
        st.info(f"⚪ {symbol}: NO ACTION")

    st.write("**Suggested action:**", suggested_action(signal))


def show_reason_summary(result: dict):
    st.subheader("Reason Summary")

    signal = result["final_signal"]

    passed_buy = [k for k, v in result["buy_checks"].items() if v]
    failed_buy = [k for k, v in result["buy_checks"].items() if not v]
    passed_sell = [k for k, v in result["sell_checks"].items() if v]

    if signal in ["BUY SIGNAL", "WATCHLIST", "NO ACTION"]:
        st.write("**Passed buy checks:**")

        if passed_buy:
            for item in passed_buy:
                st.write(f"✅ {item}")
        else:
            st.write("None")

        st.write("**Failed buy checks:**")

        if failed_buy:
            for item in failed_buy:
                st.write(f"❌ {item}")
        else:
            st.write("None")

    if signal in ["SELL SIGNAL", "HOLD"]:
        st.write("**Sell checks triggered:**")

        if passed_sell:
            for item in passed_sell:
                st.write(f"✅ {item}")
        else:
            st.write("None")


def show_price_chart(result: dict):
    st.subheader("Price Chart")

    df = result["stock_df"].copy().tail(120)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    chart_df = df.set_index("date")[["close", "MA10", "MA20"]]

    st.line_chart(chart_df)


def show_fundamentals(symbol: str):
    st.subheader("Fundamental Reference")

    st.caption(
        "This section is extra reference only. It is not used in the buy/sell signal."
    )

    try:
        summary = build_fundamental_summary(symbol)

        st.write(f"**Company:** {summary['company_name']}")
        st.write(f"**CIK:** {summary['cik']}")
        st.write(
            f"Latest annual fiscal year used for revenue: {summary['revenue_fy']} "
            f"| Filed: {summary['revenue_filed']}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Annual Revenue",
            format_money(summary["revenue"]),
            format_pct(summary["revenue_growth"]),
        )

        col2.metric(
            "Annual Net Income",
            format_money(summary["net_income"]),
            format_pct(summary["net_income_growth"]),
        )

        col3.metric(
            "Diluted EPS",
            format_number(summary["eps"]),
            format_pct(summary["eps_growth"]),
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Operating Margin",
            format_pct(summary["operating_margin"]),
        )

        col5.metric(
            "Net Margin",
            format_pct(summary["net_margin"]),
        )

        col6.metric(
            "Liabilities / Assets",
            format_pct(summary["liability_to_assets"]),
        )

        st.write("**Balance Sheet / Cash Flow**")

        balance_df = pd.DataFrame(
            [
                {"Metric": "Assets", "Value": format_money(summary["assets"])},
                {"Metric": "Liabilities", "Value": format_money(summary["liabilities"])},
                {"Metric": "Stockholders' Equity", "Value": format_money(summary["equity"])},
                {"Metric": "Operating Cash Flow", "Value": format_money(summary["operating_cash_flow"])},
            ]
        )

        st.dataframe(balance_df, hide_index=True, use_container_width=True)

        st.write("**Financial performance notes:**")

        if summary["comments"]:
            for comment in summary["comments"]:
                st.write(f"- {comment}")
        else:
            st.write("- Not enough financial data available to generate notes.")

    except Exception as e:
        st.warning(f"Could not load SEC fundamentals for {symbol}: {e}")


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.title("📈 Diffie's Stock Signal Scanner")
st.caption("KDJ + MACD + MA strategy. Manual trading only. No order submission.")

show_strategy_explanation()

with st.sidebar:
    st.header("Scanner Settings")
    st.write("Market filter:", MARKET_FILTER_SYMBOL)
    st.write("Cross lookback days:", CROSS_LOOKBACK_DAYS)
    st.write("Stop loss:", f"{STOP_LOSS_PCT * 100:.0f}%")
    st.write("Max distance above MA20:", f"{(MAX_DISTANCE_ABOVE_MA20 - 1) * 100:.0f}%")
    st.divider()
    st.caption("Fundamental data uses free SEC EDGAR data when available.")

symbol = st.text_input(
    "Stock symbol",
    value="TSLA",
    help="Example: TSLA, AAPL, NVDA, MSFT",
)

holding_choice = st.radio(
    "Are you currently holding this stock?",
    ["Not Holding", "Holding"],
    horizontal=True,
)

entry_price = None

if holding_choice == "Holding":
    entry_price = st.number_input(
        "Your entry price",
        min_value=0.01,
        value=100.00,
        step=0.01,
    )

show_fundamental_info = st.checkbox(
    "Show fundamental reference from SEC filings",
    value=True,
)

run_scan = st.button("Scan Signal", type="primary")

if run_scan:
    try:
        result = scan_one_stock(symbol, entry_price)

        st.divider()
        show_signal_card(result)

        st.subheader("Price Levels")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Latest Close", f"{result['latest_close']:.2f}")
        col2.metric("MA10", f"{result['ma10']:.2f}")
        col3.metric("MA20", f"{result['ma20']:.2f}")
        col4.metric("Distance from MA20", f"{result['distance_from_ma20_pct']:.2f}%")

        st.write("Latest completed candle:", result["latest_date"])
        st.write("Max allowed buy price from MA20 rule:", f"{result['max_allowed_price']:.2f}")

        if result["holding"]:
            st.write("Entry price:", f"{result['entry_price']:.2f}")
            st.write("Stop loss price:", f"{result['stop_loss_price']:.2f}")

        st.subheader("Market Filter")

        col5, col6 = st.columns(2)

        col5.metric("QQQ Close", f"{result['qqq_close']:.2f}")
        col6.metric("QQQ MA20", f"{result['qqq_ma20']:.2f}")

        st.subheader("Buy Checks")

        st.write(f"Buy score: **{result['buy_score']}/{result['buy_total']}**")

        for check_name, passed in result["buy_checks"].items():
            if passed:
                st.write(f"✅ {check_name}")
            else:
                st.write(f"❌ {check_name}")

        st.subheader("Sell Checks")

        for check_name, passed in result["sell_checks"].items():
            if passed:
                st.write(f"✅ {check_name}")
            else:
                st.write(f"❌ {check_name}")

        show_reason_summary(result)
        show_price_chart(result)

        if show_fundamental_info:
            show_fundamentals(result["symbol"])

        st.divider()

        st.subheader("Copyable Summary")

        summary_text = (
            f"{result['symbol']} | {result['final_signal']} | "
            f"Close {result['latest_close']:.2f} | "
            f"Buy Score {result['buy_score']}/{result['buy_total']} | "
            f"Distance from MA20 {result['distance_from_ma20_pct']:.2f}%"
        )

        st.code(summary_text)

        st.caption("This page only gives a signal and reference information. It does not place trades.")

    except Exception as e:
        st.error(str(e))
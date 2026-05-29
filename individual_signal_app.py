import os
from io import StringIO
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

from subscription_utils import (
    SUBSCRIPTION_UNIVERSES,
    subscription_config_ready,
    flags_to_selected_universes,
    get_subscription_by_token,
    upsert_subscription,
    update_subscription_by_token,
    unsubscribe_by_token,
    unsubscribe_by_email,
    format_manage_link,
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
# SETTINGS
# ============================================================

MARKET_FILTER_SYMBOL = "QQQ"

LOOKBACK_DAYS = 260
CROSS_LOOKBACK_DAYS = 3
STOP_LOSS_PCT = 0.10
MAX_DISTANCE_ABOVE_MA20 = 1.20

# Box Theory uses previous 7 completed daily candles.
BOX_LOOKBACK_DAYS = 7

# There are 6 buy checks total after Box Theory is included.
WATCHLIST_MIN_BUY_SCORE = 4

INDEX_UNIVERSES = {
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "Nasdaq-100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "Dow 30": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    "S&P 400 MidCap": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
}

CHUNK_SIZE = 100

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

CACHE_TTL_SECONDS = 3600


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
    st.error(
        "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY. "
        "Add them to your local .env file or Streamlit Cloud Secrets."
    )
    st.stop()

data_client = StockHistoricalDataClient(
    api_key=api_key,
    secret_key=secret_key,
)


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=CACHE_TTL_SECONDS)
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


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_daily_bars_for_symbols(symbols: list[str], days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    all_bars = []
    last_error = None

    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i:i + CHUNK_SIZE]
        chunk_loaded = False

        for feed in [DataFeed.SIP, DataFeed.IEX]:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=chunk,
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

                bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
                all_bars.append(bars)

                chunk_loaded = True
                break

            except Exception as e:
                last_error = e

        if not chunk_loaded:
            print(f"Could not load chunk {i}-{i + CHUNK_SIZE}. Last error: {last_error}")

    if not all_bars:
        raise RuntimeError(f"No bars loaded. Last error: {last_error}")

    return pd.concat(all_bars, ignore_index=True)


def remove_unfinished_daily_candle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps completed daily candles only.
    If the app is run before 4:10 PM New York time, today's candle is removed.
    """
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


def add_box_theory_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Box Theory:
    - Box top = highest high from previous BOX_LOOKBACK_DAYS completed daily candles.
    - Box bottom = lowest low from previous BOX_LOOKBACK_DAYS completed daily candles.
    - shift(1) prevents today's candle from building today's box.
    """
    df["BOX_TOP"] = df["high"].shift(1).rolling(BOX_LOOKBACK_DAYS).max()
    df["BOX_BOTTOM"] = df["low"].shift(1).rolling(BOX_LOOKBACK_DAYS).min()
    df["BOX_MID"] = (df["BOX_TOP"] + df["BOX_BOTTOM"]) / 2
    return df


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_unfinished_daily_candle(df)
    df = add_kdj(df)
    df = add_macd(df)
    df = add_moving_averages(df)
    df = add_box_theory_levels(df)
    return df


# ============================================================
# SIGNAL HELPERS
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
        return "Best setup. Consider manual buy next trading day around 9:45 AM ET using a limit order."
    if signal == "ALMOST BUY":
        return "Very close setup. Consider waiting for the missing condition to confirm."
    if signal == "WATCHLIST":
        return "Good candidate, but not ready yet. Watch the next completed daily candle."
    if signal == "SELL SIGNAL":
        return "Review your holding carefully. A major sell rule or multiple sell warnings triggered."
    if signal == "CAUTION HOLD":
        return "One warning triggered, but not enough for a full sell signal. Monitor closely."
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
# INDIVIDUAL STOCK SCANNER CORE
# ============================================================

def scan_one_stock(symbol: str, entry_price):
    symbol = symbol.strip().upper()

    stock_df = get_daily_bars(symbol)
    qqq_df = get_daily_bars(MARKET_FILTER_SYMBOL)

    stock_df = prepare_indicators(stock_df)
    qqq_df = prepare_indicators(qqq_df)

    if len(stock_df) < max(50, BOX_LOOKBACK_DAYS + 10):
        raise RuntimeError(f"Not enough data for {symbol}.")

    if len(qqq_df) < 50:
        raise RuntimeError("Not enough QQQ data.")

    latest = stock_df.iloc[-1]
    qqq_latest = qqq_df.iloc[-1]

    latest_close = float(latest["close"])
    ma10 = float(latest["MA10"])
    ma20 = float(latest["MA20"])

    box_top = float(latest["BOX_TOP"])
    box_bottom = float(latest["BOX_BOTTOM"])
    box_mid = float(latest["BOX_MID"])

    if pd.isna(box_top) or pd.isna(box_bottom):
        raise RuntimeError(f"Not enough Box Theory data for {symbol}.")

    box_range = box_top - box_bottom

    if box_range > 0:
        box_position_pct = (latest_close - box_bottom) / box_range * 100
    else:
        box_position_pct = None

    qqq_close = float(qqq_latest["close"])
    qqq_ma20 = float(qqq_latest["MA20"])
    qqq_above_ma20 = qqq_close > qqq_ma20

    holding = entry_price is not None

    kdj_cross_recent = crossed_up_within_last_n_days(stock_df, "K", "D")
    macd_cross_recent = crossed_up_within_last_n_days(stock_df, "MACD", "MACD_SIGNAL")
    stock_above_ma20 = latest_close > ma20
    not_too_extended = latest_close <= ma20 * MAX_DISTANCE_ABOVE_MA20

    box_breakout = latest_close > box_top
    box_breakdown = latest_close < box_bottom

    buy_checks = {
        "KDJ golden cross within last 3 completed daily candles": kdj_cross_recent,
        "MACD golden cross within last 3 completed daily candles": macd_cross_recent,
        "Stock close > stock MA20": stock_above_ma20,
        "QQQ close > QQQ MA20": qqq_above_ma20,
        f"Stock close <= MA20 * {MAX_DISTANCE_ABOVE_MA20}": not_too_extended,
        f"Box Theory breakout: close > previous {BOX_LOOKBACK_DAYS}-day box top": box_breakout,
    }

    buy_score = sum(1 for value in buy_checks.values() if value)
    buy_total = len(buy_checks)

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
        f"Box Theory breakdown: close < previous {BOX_LOOKBACK_DAYS}-day box bottom": box_breakdown,
        "Stop loss hit": stop_loss_signal,
    }

    sell_warning_count = sum(1 for value in sell_checks.values() if value)

    hard_sell_signal = stop_loss_signal or box_breakdown
    medium_sell_signal = sell_warning_count >= 2

    stock_safety_filters_pass = stock_above_ma20 and qqq_above_ma20 and not_too_extended

    if holding:
        if hard_sell_signal or medium_sell_signal:
            final_signal = "SELL SIGNAL"
        elif sell_warning_count == 1:
            final_signal = "CAUTION HOLD"
        else:
            final_signal = "HOLD"
    else:
        if buy_score == buy_total:
            final_signal = "BUY SIGNAL"
        elif buy_score == buy_total - 1 and stock_safety_filters_pass:
            final_signal = "ALMOST BUY"
        elif buy_score >= WATCHLIST_MIN_BUY_SCORE and stock_safety_filters_pass:
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
        "box_top": box_top,
        "box_bottom": box_bottom,
        "box_mid": box_mid,
        "box_position_pct": box_position_pct,
        "box_breakout": box_breakout,
        "box_breakdown": box_breakdown,
        "buy_checks": buy_checks,
        "buy_score": buy_score,
        "buy_total": buy_total,
        "sell_checks": sell_checks,
        "sell_warning_count": sell_warning_count,
        "hard_sell_signal": hard_sell_signal,
        "medium_sell_signal": medium_sell_signal,
        "stop_loss_price": stop_loss_price,
        "stock_df": stock_df,
    }


# ============================================================
# INDEX / UNIVERSE SCANNER CORE
# ============================================================

def wiki_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36 "
            "DiffiesStockSignalScanner/1.0"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }


@st.cache_data(ttl=86400)
def read_wikipedia_tables(url: str) -> list[pd.DataFrame]:
    response = requests.get(
        url,
        headers=wiki_headers(),
        timeout=30,
    )
    response.raise_for_status()
    return pd.read_html(StringIO(response.text))


def clean_symbol(symbol) -> str:
    symbol = str(symbol).strip().upper()
    symbol = symbol.replace("\xa0", "")
    symbol = symbol.replace(" ", "")
    symbol = symbol.replace("\n", "")
    return symbol


def first_table_with_columns(tables: list[pd.DataFrame], required_columns: list[str]) -> pd.DataFrame:
    for table in tables:
        columns = [str(c) for c in table.columns]
        if all(col in columns for col in required_columns):
            return table.copy()

    raise RuntimeError(f"Could not find a table with columns: {required_columns}")


@st.cache_data(ttl=86400)
def get_universe_symbols(universe_name: str) -> pd.DataFrame:
    """
    Returns a standard table with these columns:
    Symbol, Security, GICS Sector, GICS Sub-Industry
    """
    if universe_name not in INDEX_UNIVERSES:
        raise RuntimeError(f"Unknown universe: {universe_name}")

    url = INDEX_UNIVERSES[universe_name]
    tables = read_wikipedia_tables(url)

    if universe_name == "S&P 500":
        df = first_table_with_columns(tables, ["Symbol", "Security"])

    elif universe_name == "Nasdaq-100":
        df = first_table_with_columns(tables, ["Ticker", "Company"])
        df = df.rename(
            columns={
                "Ticker": "Symbol",
                "Company": "Security",
            }
        )

    elif universe_name == "Dow 30":
        df = first_table_with_columns(tables, ["Company", "Symbol"])
        df = df.rename(
            columns={
                "Company": "Security",
                "Sector": "GICS Sector",
                "Industry": "GICS Sub-Industry",
            }
        )

    elif universe_name == "S&P 400 MidCap":
        df = first_table_with_columns(tables, ["Symbol", "Security"])

    else:
        raise RuntimeError(f"Unsupported universe: {universe_name}")

    # Make columns consistent.
    if "GICS Sector" not in df.columns:
        if "Sector" in df.columns:
            df["GICS Sector"] = df["Sector"]
        else:
            df["GICS Sector"] = "Unknown"

    if "GICS Sub-Industry" not in df.columns:
        if "Industry" in df.columns:
            df["GICS Sub-Industry"] = df["Industry"]
        else:
            df["GICS Sub-Industry"] = "Unknown"

    df = df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()

    df["Symbol"] = df["Symbol"].apply(clean_symbol)
    df["Security"] = df["Security"].astype(str)
    df["GICS Sector"] = df["GICS Sector"].astype(str)
    df["GICS Sub-Industry"] = df["GICS Sub-Industry"].astype(str)

    df = df[df["Symbol"].notna()].copy()
    df = df[df["Symbol"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["Symbol"]).reset_index(drop=True)

    return df


def scan_one_index_symbol(
    symbol: str,
    company_name: str,
    sector: str,
    sub_industry: str,
    df: pd.DataFrame,
    qqq_above_ma20: bool,
) -> dict:
    df = df.copy()
    df = prepare_indicators(df)

    if len(df) < max(50, BOX_LOOKBACK_DAYS + 10):
        return {
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector,
            "sub_industry": sub_industry,
            "final_signal": "ERROR",
            "error": "Not enough data",
        }

    latest = df.iloc[-1]

    latest_close = float(latest["close"])
    ma20 = float(latest["MA20"])

    box_top = latest["BOX_TOP"]
    box_bottom = latest["BOX_BOTTOM"]

    if pd.isna(box_top) or pd.isna(box_bottom):
        return {
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector,
            "sub_industry": sub_industry,
            "final_signal": "ERROR",
            "error": "Not enough Box Theory data",
        }

    box_top = float(box_top)
    box_bottom = float(box_bottom)

    kdj_cross_recent = crossed_up_within_last_n_days(df, "K", "D")
    macd_cross_recent = crossed_up_within_last_n_days(df, "MACD", "MACD_SIGNAL")
    stock_above_ma20 = latest_close > ma20
    not_too_extended = latest_close <= ma20 * MAX_DISTANCE_ABOVE_MA20
    box_breakout = latest_close > box_top

    buy_checks = {
        "KDJ recent golden cross": kdj_cross_recent,
        "MACD recent golden cross": macd_cross_recent,
        "Close > MA20": stock_above_ma20,
        "QQQ > QQQ MA20": qqq_above_ma20,
        f"Close <= MA20 * {MAX_DISTANCE_ABOVE_MA20}": not_too_extended,
        f"Close > previous {BOX_LOOKBACK_DAYS}-day box top": box_breakout,
    }

    buy_score = sum(1 for value in buy_checks.values() if value)
    buy_total = len(buy_checks)

    safety_filters_pass = stock_above_ma20 and qqq_above_ma20 and not_too_extended

    failed_checks = [
        check_name
        for check_name, passed in buy_checks.items()
        if not passed
    ]

    if buy_score == buy_total:
        final_signal = "BUY SIGNAL"
    elif buy_score == buy_total - 1 and safety_filters_pass:
        final_signal = "ALMOST BUY"
    elif buy_score >= WATCHLIST_MIN_BUY_SCORE and safety_filters_pass:
        final_signal = "WATCHLIST"
    else:
        final_signal = "NO ACTION"

    distance_from_ma20_pct = (latest_close / ma20 - 1) * 100

    return {
        "symbol": symbol,
        "company_name": company_name,
        "sector": sector,
        "sub_industry": sub_industry,
        "final_signal": final_signal,
        "latest_date": str(latest["timestamp"]),
        "latest_close": round(latest_close, 2),
        "ma20": round(ma20, 2),
        "distance_from_ma20_pct": round(distance_from_ma20_pct, 2),
        "box_top": round(box_top, 2),
        "box_bottom": round(box_bottom, 2),
        "buy_score": buy_score,
        "buy_total": buy_total,
        "failed_checks": "; ".join(failed_checks),
        "kdj_cross_recent": kdj_cross_recent,
        "macd_cross_recent": macd_cross_recent,
        "stock_above_ma20": stock_above_ma20,
        "qqq_above_ma20": qqq_above_ma20,
        "not_too_extended": not_too_extended,
        "box_breakout": box_breakout,
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def scan_universe_cached(universe_name: str) -> tuple[pd.DataFrame, dict]:
    universe_df = get_universe_symbols(universe_name)
    symbols = universe_df["Symbol"].tolist()

    symbols_to_download = sorted(set(symbols + [MARKET_FILTER_SYMBOL]))
    all_bars = get_daily_bars_for_symbols(symbols_to_download)

    qqq_df = all_bars[all_bars["symbol"] == MARKET_FILTER_SYMBOL].copy()
    qqq_df = prepare_indicators(qqq_df)

    if len(qqq_df) < 50:
        raise RuntimeError("Not enough QQQ data.")

    qqq_latest = qqq_df.iloc[-1]
    qqq_close = float(qqq_latest["close"])
    qqq_ma20 = float(qqq_latest["MA20"])
    qqq_above_ma20 = qqq_close > qqq_ma20

    market_info = {
        "universe_name": universe_name,
        "universe_count": len(universe_df),
        "qqq_close": round(qqq_close, 2),
        "qqq_ma20": round(qqq_ma20, 2),
        "qqq_above_ma20": qqq_above_ma20,
        "qqq_latest_date": str(qqq_latest["timestamp"]),
    }

    results = []

    for _, row in universe_df.iterrows():
        symbol = row["Symbol"]
        company_name = row["Security"]
        sector = row["GICS Sector"]
        sub_industry = row["GICS Sub-Industry"]

        symbol_df = all_bars[all_bars["symbol"] == symbol].copy()

        if symbol_df.empty:
            results.append({
                "symbol": symbol,
                "company_name": company_name,
                "sector": sector,
                "sub_industry": sub_industry,
                "final_signal": "ERROR",
                "error": "No bars returned from Alpaca",
            })
            continue

        result = scan_one_index_symbol(
            symbol=symbol,
            company_name=company_name,
            sector=sector,
            sub_industry=sub_industry,
            df=symbol_df,
            qqq_above_ma20=qqq_above_ma20,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    return results_df, market_info


# UI HELPERS
# ============================================================

def show_strategy_explanation():
    st.markdown("## 🧠 How This Scanner Works")

    st.info(
        "This scanner is a manual trading helper. It does not buy or sell stocks. "
        "It checks whether a stock matches Diffie's daily momentum + Box Theory strategy."
    )

    with st.expander("📌 Strategy Summary", expanded=True):
        st.markdown(
            """
            This strategy looks for stocks that are starting to move upward with momentum,
            while also avoiding weak market conditions and avoiding fake breakouts.

            **Buy idea:**  
            The scanner looks for a stock where **KDJ** and **MACD** recently turned bullish,
            the stock is above its 20-day moving average, QQQ confirms the broader market is healthy,
            and the stock breaks above its recent Box Theory resistance level.

            **Sell idea:**  
            If you already hold the stock, the scanner checks whether weakness is appearing,
            such as a MACD bearish cross, price falling below the 10-day moving average,
            three down days in a row, Box Theory breakdown, or a 10% stop-loss.
            """
        )

    with st.expander("📦 Box Theory Explained"):
        st.markdown(
            f"""
            Box Theory looks for a stock trading inside a recent price range.

            In this scanner:

            - **Box top** = highest high from the previous `{BOX_LOOKBACK_DAYS}` completed daily candles.
            - **Box bottom** = lowest low from the previous `{BOX_LOOKBACK_DAYS}` completed daily candles.
            - **Box breakout** = latest close is above the box top.
            - **Box breakdown** = latest close is below the box bottom.
            """
        )

    with st.expander("📊 Signal Meaning"):
        st.markdown(
            """
            **BUY SIGNAL**  
            Strongest setup. All buy conditions passed.

            **ALMOST BUY**  
            Very close setup. Only one buy condition is missing, and safety filters are okay.

            **WATCHLIST**  
            Interesting candidate, but not ready yet.

            **HOLD**  
            You marked the stock as holding, and no sell rule was triggered.

            **CAUTION HOLD**  
            One sell warning appeared, but not enough for a full sell signal.

            **SELL SIGNAL**  
            A major sell rule or multiple sell warnings were triggered.

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
    elif signal == "ALMOST BUY":
        st.warning(f"🟡 {symbol}: ALMOST BUY")
    elif signal == "WATCHLIST":
        st.warning(f"👀 {symbol}: WATCHLIST")
    elif signal == "SELL SIGNAL":
        st.error(f"🔻 {symbol}: SELL SIGNAL")
    elif signal == "CAUTION HOLD":
        st.warning(f"⚠️ {symbol}: CAUTION HOLD")
    elif signal == "HOLD":
        st.info(f"🟦 {symbol}: HOLD")
    else:
        st.info(f"⚪ {symbol}: NO ACTION")

    st.write("**Suggested action:**", suggested_action(signal))


def show_reason_summary(result: dict):
    st.subheader("Reason Summary")

    passed_buy = [k for k, v in result["buy_checks"].items() if v]
    failed_buy = [k for k, v in result["buy_checks"].items() if not v]
    passed_sell = [k for k, v in result["sell_checks"].items() if v]

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
    chart_df = df.set_index("date")[["close", "MA10", "MA20", "BOX_TOP", "BOX_BOTTOM"]]

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

        col4.metric("Operating Margin", format_pct(summary["operating_margin"]))
        col5.metric("Net Margin", format_pct(summary["net_margin"]))
        col6.metric("Liabilities / Assets", format_pct(summary["liability_to_assets"]))

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
# APP UI
# ============================================================

st.title("📈 Diffie's Stock Signal Scanner")
st.caption("KDJ + MACD + MA + Box Theory strategy. Manual trading only. No order submission.")

with st.sidebar:
    st.header("About Diffie")

    st.markdown(
        """
        **Diffie Liu**  
        Future CPA | Wake Forest & UCR Alum

        I built this scanner to turn my personal daily stock-analysis process
        into a simple, transparent tool.

        My approach is **not day trading**. I focus on completed daily candles,
        momentum confirmation, trend direction, Box Theory structure, and
        financial reference data.

        This scanner is for **manual decision support only**. It does not place trades.
        """
    )

    st.markdown("📧 [diffieliu@gmail.com](mailto:diffieliu@gmail.com)")
    st.write("📞 909-689-6496")

    st.divider()

    st.header("Scanner Settings")
    st.write("Market filter:", MARKET_FILTER_SYMBOL)
    st.write("Cross lookback days:", CROSS_LOOKBACK_DAYS)
    st.write("Box lookback days:", BOX_LOOKBACK_DAYS)
    st.write("Stop loss:", f"{STOP_LOSS_PCT * 100:.0f}%")
    st.write("Max distance above MA20:", f"{(MAX_DISTANCE_ABOVE_MA20 - 1) * 100:.0f}%")

    st.divider()

    if st.button("Refresh data / clear cache"):
        st.cache_data.clear()
        st.rerun()

tab_single, tab_sp500, tab_subscribe = st.tabs(
    [
        "🔎 Individual Stock Scanner",
        "📋 Index Scanner",
        "📬 Daily Email Alerts",
    ]
)


# ============================================================
# TAB 1: INDIVIDUAL STOCK SCANNER
# ============================================================

with tab_single:
    show_strategy_explanation()

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

    run_scan = st.button("Scan Individual Stock", type="primary")

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

            st.subheader("Box Theory Levels")

            box_col1, box_col2, box_col3, box_col4 = st.columns(4)

            box_col1.metric("Box Top", f"{result['box_top']:.2f}")
            box_col2.metric("Box Bottom", f"{result['box_bottom']:.2f}")
            box_col3.metric("Box Mid", f"{result['box_mid']:.2f}")

            if result["box_position_pct"] is not None:
                box_col4.metric("Position in Box", f"{result['box_position_pct']:.2f}%")
            else:
                box_col4.metric("Position in Box", "N/A")

            if result["box_breakout"]:
                st.success("Box Theory: price closed above the box top.")
            elif result["box_breakdown"]:
                st.error("Box Theory: price closed below the box bottom.")
            else:
                st.info("Box Theory: price is still inside the box.")

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

            st.write(f"Sell warnings triggered: **{result['sell_warning_count']}**")

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
                f"Distance from MA20 {result['distance_from_ma20_pct']:.2f}% | "
                f"Box Top {result['box_top']:.2f} | "
                f"Box Bottom {result['box_bottom']:.2f}"
            )

            st.code(summary_text)

            st.caption("This page only gives a signal and reference information. It does not place trades.")

        except Exception as e:
            st.error(str(e))


# ============================================================
# TAB 2: INDEX / UNIVERSE SCANNER
# ============================================================

with tab_sp500:
    st.markdown("## 📋 Index / Universe Signal Scanner")

    st.info(
        "This scans a full stock universe using the same buy setup. "
        "It does not check holding-specific sell signals because it does not know your entry prices."
    )

    selected_universe = st.selectbox(
        "Choose stock universe to scan",
        list(INDEX_UNIVERSES.keys()),
        index=0,
        help="Nasdaq-100 is usually fastest after Dow 30. S&P 400 and S&P 500 may take longer.",
    )

    with st.expander("How the index scan works", expanded=False):
        st.markdown(
            f"""
            The scanner checks every stock in the selected universe using the same buy setup:

            1. KDJ golden cross within the last `{CROSS_LOOKBACK_DAYS}` completed daily candles  
            2. MACD golden cross within the last `{CROSS_LOOKBACK_DAYS}` completed daily candles  
            3. Stock close > stock MA20  
            4. QQQ close > QQQ MA20  
            5. Stock is not more than `{(MAX_DISTANCE_ABOVE_MA20 - 1) * 100:.0f}%` above MA20  
            6. Stock closes above its previous `{BOX_LOOKBACK_DAYS}`-day Box Theory top  

            Signal judgment:

            - **BUY SIGNAL** = 6/6 checks passed  
            - **ALMOST BUY** = 5/6 checks passed and safety filters passed  
            - **WATCHLIST** = 4+ checks passed and safety filters passed  
            - **NO ACTION** = setup is not ready  
            """
        )

    run_universe_scan = st.button(f"Run {selected_universe} Scan", type="primary")

    if run_universe_scan:
        try:
            with st.spinner(f"Scanning {selected_universe}. This may take a few minutes..."):
                results_df, market_info = scan_universe_cached(selected_universe)

            st.divider()

            st.subheader("Market Filter")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Universe", market_info["universe_name"])
            col2.metric("Stocks loaded", market_info["universe_count"])
            col3.metric("QQQ Close", market_info["qqq_close"])
            col4.metric("QQQ > MA20", str(market_info["qqq_above_ma20"]))

            st.write("QQQ MA20:", market_info["qqq_ma20"])
            st.write("QQQ latest completed candle:", market_info["qqq_latest_date"])

            st.divider()

            signal_counts = results_df["final_signal"].value_counts().to_dict()

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric("BUY", signal_counts.get("BUY SIGNAL", 0))
            c2.metric("ALMOST BUY", signal_counts.get("ALMOST BUY", 0))
            c3.metric("WATCHLIST", signal_counts.get("WATCHLIST", 0))
            c4.metric("NO ACTION", signal_counts.get("NO ACTION", 0))
            c5.metric("ERROR", signal_counts.get("ERROR", 0))

            st.subheader("Filter Results")

            available_signals = [
                "BUY SIGNAL",
                "ALMOST BUY",
                "WATCHLIST",
                "NO ACTION",
                "ERROR",
            ]

            selected_signals = st.multiselect(
                "Signal filter",
                available_signals,
                default=["BUY SIGNAL", "ALMOST BUY"],
            )

            sectors = sorted(results_df["sector"].dropna().unique().tolist())

            selected_sectors = st.multiselect(
                "Sector filter",
                sectors,
                default=sectors,
            )

            search_text = st.text_input(
                "Search symbol or company",
                value="",
            ).strip().upper()

            filtered_df = results_df[
                results_df["final_signal"].isin(selected_signals)
                & results_df["sector"].isin(selected_sectors)
            ].copy()

            if search_text:
                filtered_df = filtered_df[
                    filtered_df["symbol"].str.upper().str.contains(search_text, na=False)
                    | filtered_df["company_name"].str.upper().str.contains(search_text, na=False)
                ].copy()

            if not filtered_df.empty and "buy_score" in filtered_df.columns:
                filtered_df = filtered_df.sort_values(
                    ["final_signal", "buy_score", "distance_from_ma20_pct"],
                    ascending=[True, False, True],
                )

            display_columns = [
                "symbol",
                "company_name",
                "sector",
                "sub_industry",
                "final_signal",
                "latest_close",
                "buy_score",
                "buy_total",
                "distance_from_ma20_pct",
                "box_top",
                "box_bottom",
                "failed_checks",
            ]

            display_columns = [
                col for col in display_columns if col in filtered_df.columns
            ]

            st.subheader("Scanner Results")

            st.dataframe(
                filtered_df[display_columns],
                use_container_width=True,
                hide_index=True,
            )

            csv_data = filtered_df.to_csv(index=False).encode("utf-8")

            safe_universe_name = selected_universe.replace(" ", "_").replace("&", "and")

            st.download_button(
                label="Download filtered results as CSV",
                data=csv_data,
                file_name=f"{safe_universe_name}_signal_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
                mime="text/csv",
            )

            st.subheader("Top BUY / ALMOST BUY List")

            top_df = results_df[
                results_df["final_signal"].isin(["BUY SIGNAL", "ALMOST BUY"])
            ].copy()

            if top_df.empty:
                st.write(f"No BUY SIGNAL or ALMOST BUY stocks in {selected_universe} right now.")
            else:
                top_df = top_df.sort_values(
                    ["buy_score", "distance_from_ma20_pct"],
                    ascending=[False, True],
                )

                for _, row in top_df.iterrows():
                    st.write(
                        f"**{row['symbol']}** | {row['company_name']} | "
                        f"{row['final_signal']} | "
                        f"Score {row['buy_score']}/{row['buy_total']} | "
                        f"Close {row['latest_close']} | "
                        f"Missing: {row['failed_checks']}"
                    )

        except Exception as e:
            st.error(str(e))


# ============================================================
# TAB 3: EMAIL SUBSCRIPTION
# ============================================================

with tab_subscribe:
    st.markdown("## 📬 Daily Email Alerts")

    st.info(
        "Subscribe to receive a daily email with BUY SIGNAL and ALMOST BUY stocks "
        "from the lists you choose. You can update or cancel your subscription later."
    )

    if not subscription_config_ready():
        st.warning(
            "Subscription is not configured yet. Add SUPABASE_URL, "
            "SUPABASE_SERVICE_ROLE_KEY, RESEND_API_KEY, and APP_PUBLIC_URL "
            "to Streamlit Secrets before using this page online."
        )
    else:
        query_token = None

        try:
            query_token = st.query_params.get("token")
        except Exception:
            query_token = None

        existing_subscription = None

        if query_token:
            try:
                existing_subscription = get_subscription_by_token(query_token)

                if existing_subscription:
                    st.success(
                        f"Managing subscription for {existing_subscription['email']}"
                    )
                else:
                    st.error("This subscription link is invalid or expired.")

            except Exception as e:
                st.error(f"Could not load subscription: {e}")

        default_email = ""
        default_universes = ["S&P 500", "Nasdaq-100"]

        if existing_subscription:
            default_email = existing_subscription["email"]
            loaded_universes = flags_to_selected_universes(existing_subscription)
            if loaded_universes:
                default_universes = loaded_universes

        st.subheader("Subscribe or Update")

        with st.form("daily_email_subscription_form"):
            email_input = st.text_input(
                "Email address",
                value=default_email,
                disabled=bool(existing_subscription),
                help="Daily signal emails will be sent to this address.",
            )

            selected_universes = st.multiselect(
                "Choose lists for daily updates",
                SUBSCRIPTION_UNIVERSES,
                default=default_universes,
            )

            submitted = st.form_submit_button("Subscribe / Update Subscription")

            if submitted:
                try:
                    if not selected_universes:
                        st.error("Please select at least one list.")
                    elif existing_subscription and query_token:
                        updated = update_subscription_by_token(
                            token=query_token,
                            selected_universes=selected_universes,
                        )
                        st.success("Subscription updated.")
                        st.write("Selected lists:", ", ".join(flags_to_selected_universes(updated)))
                        st.write("Manage link:")
                        st.code(format_manage_link(updated["token"]))
                    else:
                        saved = upsert_subscription(
                            email=email_input,
                            selected_universes=selected_universes,
                        )
                        st.success("Subscription saved.")
                        st.write("Selected lists:", ", ".join(flags_to_selected_universes(saved)))
                        st.write("Save this manage/cancel link:")
                        st.code(format_manage_link(saved["token"]))
                except Exception as e:
                    st.error(str(e))

        st.subheader("Cancel Subscription")

        if existing_subscription and query_token:
            if st.button("Cancel this subscription"):
                try:
                    unsubscribe_by_token(query_token)
                    st.success("Subscription cancelled.")
                except Exception as e:
                    st.error(str(e))
        else:
            with st.form("unsubscribe_by_email_form"):
                cancel_email = st.text_input(
                    "Email address to cancel",
                    value="",
                    help="This cancels the subscription for this email address.",
                )
                cancel_submitted = st.form_submit_button("Cancel by Email")

                if cancel_submitted:
                    try:
                        unsubscribe_by_email(cancel_email)
                        st.success("Subscription cancelled if the email existed.")
                    except Exception as e:
                        st.error(str(e))

        st.caption(
            "Every daily email includes a private manage link so subscribers can "
            "change their selected lists or cancel."
        )

import os
import re
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

# Extra confirmations used to label HIGH CONVICTION BUY.
VOLUME_CONFIRMATION_MULTIPLE = 1.20

STRATEGY_VERSION = "v1.4"
STRATEGY_UPDATED = "2026-05-29"

UNIVERSE_BENCHMARKS = {
    "S&P 500": "SPY",
    "Nasdaq-100": "QQQ",
    "Dow 30": "DIA",
    "S&P 400 MidCap": "MDY",
}

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
    df["VOL20"] = df["volume"].rolling(20).mean()
    df["RET20"] = df["close"] / df["close"].shift(20) - 1
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
    if signal == "HIGH CONVICTION BUY":
        return "Strongest setup. Base buy signal plus volume, relative strength, and benchmark confirmation passed."
    if signal == "BUY SIGNAL":
        return "Best base setup. Consider manual review next trading day around 9:45 AM ET using a limit order."
    if signal == "ALMOST BUY":
        return "Very close setup. Consider waiting for the missing condition to confirm."
    if signal == "WATCHLIST":
        return "Good candidate, but not ready yet. Watch the next completed daily candle."
    if signal == "SELL SIGNAL":
        return "Review your holding carefully. A hard sell rule or multiple sell warnings triggered."
    if signal == "ALMOST SELL":
        return "Early weakness appeared. Review closely, but this is not a full sell signal yet."
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
    box_position_pct = (latest_close - box_bottom) / box_range * 100 if box_range > 0 else None

    qqq_close = float(qqq_latest["close"])
    qqq_ma20 = float(qqq_latest["MA20"])
    qqq_above_ma20 = qqq_close > qqq_ma20
    qqq_ret20 = float(qqq_latest["RET20"]) if pd.notna(qqq_latest.get("RET20")) else None

    holding = entry_price is not None

    # -------------------------
    # Core BUY pillars
    # Keep these distinct. KDJ + MACD are one Momentum pillar.
    # QQQ is the individual-stock market environment pillar.
    # -------------------------
    kdj_cross_recent = crossed_up_within_last_n_days(stock_df, "K", "D")
    macd_cross_recent = crossed_up_within_last_n_days(stock_df, "MACD", "MACD_SIGNAL")
    momentum_confirmed = kdj_cross_recent and macd_cross_recent

    market_environment_pass = qqq_above_ma20
    trend_pass = latest_close > ma20
    structure_pass = latest_close > box_top
    risk_extension_pass = latest_close <= ma20 * MAX_DISTANCE_ABOVE_MA20

    core_pillars = {
        "Market Environment: QQQ > QQQ MA20": market_environment_pass,
        "Trend: stock close > stock MA20": trend_pass,
        "Momentum: KDJ and MACD golden crosses within lookback": momentum_confirmed,
        f"Structure: close > previous {BOX_LOOKBACK_DAYS}-day box top": structure_pass,
        f"Risk / Extension: close <= MA20 × {MAX_DISTANCE_ABOVE_MA20}": risk_extension_pass,
    }

    buy_score = sum(1 for value in core_pillars.values() if value)
    buy_total = len(core_pillars)
    failed_checks = [name for name, passed in core_pillars.items() if not passed]

    # -------------------------
    # Confirmation tags
    # These add perspective, but they do not duplicate the core score.
    # -------------------------
    volume_value = float(latest["volume"]) if pd.notna(latest.get("volume")) else 0.0
    vol20 = float(latest["VOL20"]) if pd.notna(latest.get("VOL20")) else None
    volume_confirmed = bool(vol20 and vol20 > 0 and volume_value >= vol20 * VOLUME_CONFIRMATION_MULTIPLE)

    stock_ret20 = float(latest["RET20"]) if pd.notna(latest.get("RET20")) else None
    relative_strength_confirmed = bool(stock_ret20 is not None and qqq_ret20 is not None and stock_ret20 > qqq_ret20)

    confirmations = {
        f"Volume: latest volume >= {VOLUME_CONFIRMATION_MULTIPLE}× 20-day average volume": volume_confirmed,
        "Relative Strength: stock 20-day return > QQQ 20-day return": relative_strength_confirmed,
        "Growth Market Context: QQQ > QQQ MA20": qqq_above_ma20,
    }

    all_core_pillars_pass = buy_score == buy_total
    high_conviction_buy = all_core_pillars_pass and volume_confirmed and relative_strength_confirmed

    # ALMOST BUY is intentionally narrow:
    # market/trend/risk must be okay, and only momentum or structure may be missing.
    required_safety_pillars_pass = market_environment_pass and trend_pass and risk_extension_pass
    momentum_or_structure_pass_count = int(momentum_confirmed) + int(structure_pass)

    if holding:
        final_signal = None
    else:
        if high_conviction_buy:
            final_signal = "HIGH CONVICTION BUY"
        elif all_core_pillars_pass:
            final_signal = "BUY SIGNAL"
        elif required_safety_pillars_pass and momentum_or_structure_pass_count == 1:
            final_signal = "ALMOST BUY"
        elif required_safety_pillars_pass:
            final_signal = "WATCHLIST"
        else:
            final_signal = "NO ACTION"

    # -------------------------
    # Sell pillars
    # Keep each weakness type separate.
    # -------------------------
    three_down_days = three_consecutive_down_closes(stock_df)
    macd_cross_down = crossed_down_today(stock_df, "MACD", "MACD_SIGNAL")
    close_below_ma10 = latest_close < ma10
    box_breakdown = latest_close < box_bottom

    stop_loss_signal = False
    stop_loss_price = None

    if holding:
        stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
        stop_loss_signal = latest_close <= stop_loss_price

    sell_pillars = {
        "Hard Risk: stop loss hit": stop_loss_signal,
        f"Structure Failure: close < previous {BOX_LOOKBACK_DAYS}-day box bottom": box_breakdown,
        "Trend Weakness: close < MA10": close_below_ma10,
        "Momentum Weakness: MACD crossed down today": macd_cross_down,
        "Price Action Weakness: three consecutive down closes": three_down_days,
    }

    medium_sell_warnings = {
        "Trend Weakness: close < MA10": close_below_ma10,
        "Momentum Weakness: MACD crossed down today": macd_cross_down,
        "Price Action Weakness: three consecutive down closes": three_down_days,
    }

    sell_warning_count = sum(1 for value in medium_sell_warnings.values() if value)
    hard_sell_signal = stop_loss_signal or box_breakdown
    medium_sell_signal = sell_warning_count >= 2

    if holding:
        if hard_sell_signal or medium_sell_signal:
            final_signal = "SELL SIGNAL"
        elif sell_warning_count == 1:
            # One warning but no hard break = caution.
            final_signal = "CAUTION HOLD"
        else:
            final_signal = "HOLD"

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
        "box_breakout": structure_pass,
        "box_breakdown": box_breakdown,

        # Core pillars
        "buy_checks": core_pillars,
        "core_pillars": core_pillars,
        "buy_score": buy_score,
        "buy_total": buy_total,
        "core_pillar_score": buy_score,
        "buy_score": buy_score,
        "buy_total": buy_total,
        "core_pillar_score": buy_score,
        "core_pillar_total": buy_total,
        "failed_checks": "; ".join(failed_checks),
        "market_environment_pass": market_environment_pass,
        "trend_pass": trend_pass,
        "momentum_pass": momentum_confirmed,
        "structure_pass": structure_pass,
        "risk_extension_pass": risk_extension_pass,
        "kdj_cross_recent": kdj_cross_recent,
        "macd_cross_recent": macd_cross_recent,
        "all_core_pillars_pass": all_core_pillars_pass,

        # Confirmations
        "confirmations": confirmations,
        "volume": volume_value,
        "vol20": vol20,
        "volume_confirmed": volume_confirmed,
        "stock_ret20": stock_ret20,
        "qqq_ret20": qqq_ret20,
        "relative_strength_confirmed": relative_strength_confirmed,
        "universe_benchmark_confirmed": market_environment_pass,
        "qqq_market_confirmed": qqq_above_ma20,
        "high_conviction_buy": high_conviction_buy,

        # Sell pillars
        "sell_checks": sell_pillars,
        "sell_pillars": sell_pillars,
        "sell_warning_count": sell_warning_count,
        "hard_sell_signal": hard_sell_signal,
        "medium_sell_signal": medium_sell_signal,
        "stop_loss_price": stop_loss_price,
        "stop_loss_hit": stop_loss_signal,
        "trend_weakness": close_below_ma10,
        "momentum_weakness": macd_cross_down,
        "price_action_weakness": three_down_days,
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
    qqq_ret20: float | None,
    universe_benchmark_above_ma20: bool,
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

    # -------------------------
    # Core BUY pillars
    # Market uses the selected universe benchmark:
    # S&P 500→SPY, Nasdaq-100→QQQ, Dow→DIA, S&P 400→MDY.
    # -------------------------
    kdj_cross_recent = crossed_up_within_last_n_days(df, "K", "D")
    macd_cross_recent = crossed_up_within_last_n_days(df, "MACD", "MACD_SIGNAL")
    momentum_confirmed = kdj_cross_recent and macd_cross_recent

    market_environment_pass = universe_benchmark_above_ma20
    trend_pass = latest_close > ma20
    structure_pass = latest_close > box_top
    risk_extension_pass = latest_close <= ma20 * MAX_DISTANCE_ABOVE_MA20

    core_pillars = {
        "Market Environment: universe benchmark > MA20": market_environment_pass,
        "Trend: stock close > stock MA20": trend_pass,
        "Momentum: KDJ and MACD golden crosses within lookback": momentum_confirmed,
        f"Structure: close > previous {BOX_LOOKBACK_DAYS}-day box top": structure_pass,
        f"Risk / Extension: close <= MA20 × {MAX_DISTANCE_ABOVE_MA20}": risk_extension_pass,
    }

    buy_score = sum(1 for value in core_pillars.values() if value)
    buy_total = len(core_pillars)

    failed_checks = [
        check_name
        for check_name, passed in core_pillars.items()
        if not passed
    ]

    # -------------------------
    # Confirmation tags
    # These are separate perspectives and do not inflate the core score.
    # -------------------------
    volume_value = float(latest["volume"]) if pd.notna(latest.get("volume")) else 0.0
    vol20 = float(latest["VOL20"]) if pd.notna(latest.get("VOL20")) else None
    volume_confirmed = bool(vol20 and vol20 > 0 and volume_value >= vol20 * VOLUME_CONFIRMATION_MULTIPLE)

    stock_ret20 = float(latest["RET20"]) if pd.notna(latest.get("RET20")) else None
    relative_strength_confirmed = bool(stock_ret20 is not None and qqq_ret20 is not None and stock_ret20 > qqq_ret20)

    confirmations = {
        f"Volume: latest volume >= {VOLUME_CONFIRMATION_MULTIPLE}× 20-day average volume": volume_confirmed,
        "Relative Strength: stock 20-day return > QQQ 20-day return": relative_strength_confirmed,
        "QQQ Growth Context: QQQ > QQQ MA20": qqq_above_ma20,
    }

    all_core_pillars_pass = buy_score == buy_total
    required_safety_pillars_pass = market_environment_pass and trend_pass and risk_extension_pass
    momentum_or_structure_pass_count = int(momentum_confirmed) + int(structure_pass)

    high_conviction_buy = (
        all_core_pillars_pass
        and volume_confirmed
        and relative_strength_confirmed
    )

    if high_conviction_buy:
        final_signal = "HIGH CONVICTION BUY"
    elif all_core_pillars_pass:
        final_signal = "BUY SIGNAL"
    elif required_safety_pillars_pass and momentum_or_structure_pass_count == 1:
        final_signal = "ALMOST BUY"
    elif required_safety_pillars_pass:
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

        # Core pillars
        "buy_score": buy_score,
        "buy_total": buy_total,
        "core_pillar_score": buy_score,
        "buy_score": buy_score,
        "buy_total": buy_total,
        "core_pillar_score": buy_score,
        "core_pillar_total": buy_total,
        "failed_checks": "; ".join(failed_checks),
        "market_environment_pass": market_environment_pass,
        "trend_pass": trend_pass,
        "momentum_pass": momentum_confirmed,
        "structure_pass": structure_pass,
        "risk_extension_pass": risk_extension_pass,
        "all_core_pillars_pass": all_core_pillars_pass,

        # Components of momentum, shown as explanation only.
        "kdj_cross_recent": kdj_cross_recent,
        "macd_cross_recent": macd_cross_recent,

        # Confirmation tags
        "volume": round(volume_value, 2),
        "vol20": round(vol20, 2) if vol20 is not None else None,
        "volume_confirmed": volume_confirmed,
        "stock_ret20": round(stock_ret20 * 100, 2) if stock_ret20 is not None else None,
        "qqq_ret20": round(qqq_ret20 * 100, 2) if qqq_ret20 is not None else None,
        "relative_strength_confirmed": relative_strength_confirmed,
        "universe_benchmark_confirmed": universe_benchmark_above_ma20,
        "qqq_market_confirmed": qqq_above_ma20,
        "high_conviction_buy": high_conviction_buy,
    }

def scan_universe_cached(universe_name: str) -> tuple[pd.DataFrame, dict]:
    universe_df = get_universe_symbols(universe_name)
    symbols = universe_df["Symbol"].tolist()

    benchmark_symbol = UNIVERSE_BENCHMARKS.get(universe_name, MARKET_FILTER_SYMBOL)
    symbols_to_download = sorted(set(symbols + [MARKET_FILTER_SYMBOL, benchmark_symbol, "SPY"]))
    all_bars = get_daily_bars_for_symbols(symbols_to_download)

    qqq_df = all_bars[all_bars["symbol"] == MARKET_FILTER_SYMBOL].copy()
    qqq_df = prepare_indicators(qqq_df)

    if len(qqq_df) < 50:
        raise RuntimeError("Not enough QQQ data.")

    qqq_latest = qqq_df.iloc[-1]
    qqq_close = float(qqq_latest["close"])
    qqq_ma20 = float(qqq_latest["MA20"])
    qqq_above_ma20 = qqq_close > qqq_ma20
    qqq_ret20 = float(qqq_latest["RET20"]) if pd.notna(qqq_latest.get("RET20")) else None

    benchmark_df = all_bars[all_bars["symbol"] == benchmark_symbol].copy()
    benchmark_df = prepare_indicators(benchmark_df) if not benchmark_df.empty else qqq_df.copy()
    benchmark_latest = benchmark_df.iloc[-1]
    benchmark_close = float(benchmark_latest["close"])
    benchmark_ma20 = float(benchmark_latest["MA20"])
    universe_benchmark_above_ma20 = benchmark_close > benchmark_ma20

    market_info = {
        "universe_name": universe_name,
        "universe_count": len(universe_df),
        "qqq_close": round(qqq_close, 2),
        "qqq_ma20": round(qqq_ma20, 2),
        "qqq_above_ma20": qqq_above_ma20,
        "qqq_latest_date": str(qqq_latest["timestamp"]),
        "qqq_ret20": round(qqq_ret20 * 100, 2) if qqq_ret20 is not None else None,
        "benchmark_symbol": benchmark_symbol,
        "benchmark_close": round(benchmark_close, 2),
        "benchmark_ma20": round(benchmark_ma20, 2),
        "universe_benchmark_above_ma20": universe_benchmark_above_ma20,
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
            qqq_ret20=qqq_ret20,
            universe_benchmark_above_ma20=universe_benchmark_above_ma20,
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
        "It organizes daily technical signals, market context, confirmations, and risk warnings."
    )

    with st.expander("🚦 Recommended Workflow", expanded=True):
        st.markdown(
            f"""
            **Best daily workflow**

            1. Run the **Market List Scanner** after the market closes.
            2. Review **HIGH CONVICTION BUY**, **BUY SIGNAL**, and **ALMOST BUY**.
            3. Copy the ticker list and save it with the signal date.
            4. In the following days, paste that list into **Signal List Tracker**.
            5. Use the tracker to check return, benchmark comparison, and SELL / ALMOST SELL warnings.

            **Strategy version:** `{STRATEGY_VERSION}`  
            **Last updated:** `{STRATEGY_UPDATED}`
            """
        )

    with st.expander("🧩 Core Strategy Pillars", expanded=True):
        st.markdown(
            f"""
            The strategy is organized into independent pillars, so the same idea is not counted multiple times.

            **1. Market Environment**  
            For index scanning, the app uses the matching universe benchmark:
            - S&P 500 → SPY
            - Nasdaq-100 → QQQ
            - Dow 30 → DIA
            - S&P 400 MidCap → MDY

            The market pillar passes when the benchmark closes above its MA20.

            **2. Trend**  
            The stock must close above its own MA20.

            **3. Momentum**  
            KDJ golden cross and MACD golden cross must both happen within the last `{CROSS_LOOKBACK_DAYS}` completed daily candles.
            KDJ and MACD are shown separately, but they count as **one Momentum pillar**.

            **4. Structure**  
            The stock must close above the previous `{BOX_LOOKBACK_DAYS}`-day Box Theory top.

            **5. Risk / Extension**  
            The stock must not be more than `{(MAX_DISTANCE_ABOVE_MA20 - 1) * 100:.0f}%` above its MA20.
            """
        )

    with st.expander("🔥 Signal Levels"):
        st.markdown(
            f"""
            **HIGH CONVICTION BUY**  
            All core pillars pass, plus:
            - Volume confirmation passes.
            - Relative strength vs QQQ passes.

            **BUY SIGNAL**  
            All core pillars pass.

            **ALMOST BUY**  
            Market, trend, and risk/extension pass, but either momentum or structure still needs confirmation.

            **WATCHLIST**  
            Market, trend, and risk/extension pass, but both momentum and structure are not ready.

            **NO ACTION**  
            Market, trend, or risk/extension is weak.
            """
        )

    with st.expander("✅ Separate Confirmation Tags"):
        st.markdown(
            f"""
            Confirmations are not duplicate buy points. They provide extra perspective.

            **Volume confirmation**  
            Latest volume >= `{VOLUME_CONFIRMATION_MULTIPLE}×` 20-day average volume.

            **Relative strength confirmation**  
            Stock 20-day return > QQQ 20-day return.

            **QQQ growth market context**  
            QQQ > QQQ MA20.

            These help decide whether a BUY SIGNAL becomes **HIGH CONVICTION BUY**.
            """
        )

    with st.expander("📦 Box Theory Explained"):
        st.markdown(
            f"""
            Box Theory treats recent price action as a short-term range.

            - **Box top** = highest high from the previous `{BOX_LOOKBACK_DAYS}` completed daily candles.
            - **Box bottom** = lowest low from the previous `{BOX_LOOKBACK_DAYS}` completed daily candles.
            - **Box breakout** = latest close is above the box top.
            - **Box breakdown** = latest close is below the box bottom.

            The buy side uses box breakout.  
            The sell side treats box breakdown as a serious structure failure.
            """
        )

    with st.expander("🔻 Sell Logic Pillars"):
        st.markdown(
            f"""
            Sell logic is separated into different weakness types:

            **Hard Risk**  
            Current close <= signal close × `{1 - STOP_LOSS_PCT:.2f}`.

            **Structure Failure**  
            Current close < previous `{BOX_LOOKBACK_DAYS}`-day box bottom.

            **Trend Weakness**  
            Current close < MA10.

            **Momentum Weakness**  
            MACD crossed down.

            **Price Action Weakness**  
            Three consecutive down closes.

            **SELL SIGNAL** = hard risk, structure failure, or at least 2 medium warnings.  
            **ALMOST SELL** = exactly 1 medium warning plus weak price structure.  
            **CAUTION HOLD** = exactly 1 medium warning but structure is still acceptable.  
            **HOLD** = no warning.
            """
        )

    with st.expander("📈 How to Judge Whether the Strategy Works"):
        st.markdown(
            """
            A stock going up after a signal is not enough. The key questions are:

            - Did the ticker return positive after the signal date?
            - Did it beat QQQ and SPY?
            - Did HIGH CONVICTION BUY perform better than normal BUY SIGNAL?
            - Did ALMOST BUY behave like a good early watchlist or a weak signal?
            - Did the sell warnings protect gains or avoid large losses?

            The Signal List Tracker starts returns from the signal-date close and compares them with QQQ/SPY.
            """
        )

    with st.expander("⚠️ Important Notes"):
        st.markdown(
            """
            - This scanner is for **education, research, and manual decision support only**.
            - It is **not financial advice**.
            - It is **not a trading recommendation**.
            - It does **not** place trades.
            - It does **not** predict the future.
            - A signal does not guarantee profit.
            - The strategy uses **completed daily candles**, not unfinished intraday candles.
            - Fundamental information is extra reference only and is **not used** in the signal.
            """
        )


def show_signal_card(result: dict):
    signal = result["final_signal"]
    symbol = result["symbol"]

    if signal == "HIGH CONVICTION BUY":
        st.success(f"🔥 {symbol}: HIGH CONVICTION BUY")
    elif signal == "BUY SIGNAL":
        st.success(f"🚀 {symbol}: BUY SIGNAL")
    elif signal == "ALMOST BUY":
        st.warning(f"🟡 {symbol}: ALMOST BUY")
    elif signal == "WATCHLIST":
        st.warning(f"👀 {symbol}: WATCHLIST")
    elif signal == "SELL SIGNAL":
        st.error(f"🔻 {symbol}: SELL SIGNAL")
    elif signal == "ALMOST SELL":
        st.warning(f"🟠 {symbol}: ALMOST SELL")
    elif signal == "CAUTION HOLD":
        st.warning(f"⚠️ {symbol}: CAUTION HOLD")
    elif signal == "HOLD":
        st.info(f"🟦 {symbol}: HOLD")
    else:
        st.info(f"⚪ {symbol}: NO ACTION")

    st.write("**Suggested action:**", suggested_action(signal))


def show_reason_summary(result: dict):
    st.subheader("Reason Summary")

    core_pillars = result.get("core_pillars", result.get("buy_checks", {}))
    confirmations = result.get("confirmations", {})
    sell_pillars = result.get("sell_pillars", result.get("sell_checks", {}))

    st.write("**Core buy pillars:**")
    for item, passed in core_pillars.items():
        st.write(f"{'✅' if passed else '❌'} {item}")

    if confirmations:
        st.write("**Separate confirmation tags:**")
        for item, passed in confirmations.items():
            st.write(f"{'✅' if passed else '❌'} {item}")

    st.write("**Sell / weakness pillars:**")
    if sell_pillars:
        for item, triggered in sell_pillars.items():
            st.write(f"{'⚠️' if triggered else '✅'} {item}")
    else:
        st.write("None")

    st.caption(
        "Core pillars decide the base signal. Confirmation tags add perspective but do not duplicate the core score."
    )

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
# MANUAL PERFORMANCE / SELL TRACKER HELPERS
# ============================================================

def parse_ticker_list(raw_text: str) -> list[str]:
    """
    Accepts tickers copied from the scanner, for example:
    TSLA, AAPL, NVDA
    or one ticker per line.
    """
    raw_text = raw_text or ""
    pieces = re.split(r"[,;\s]+", raw_text.upper().strip())

    tickers = []
    for item in pieces:
        item = item.strip()
        if not item:
            continue
        item = item.replace("$", "")
        if item not in tickers:
            tickers.append(item)

    return tickers


def date_only_from_timestamp(value):
    return pd.to_datetime(value).date()


def find_signal_candle_on_or_before(df: pd.DataFrame, selected_date) -> pd.Series | None:
    """
    Finds the completed daily candle on or before the user-input signal date.
    This handles weekends/holidays by using the most recent trading day before that date.
    """
    if df.empty:
        return None

    working = df.copy()
    working["date_only"] = pd.to_datetime(working["timestamp"]).dt.date
    eligible = working[working["date_only"] <= selected_date].copy()

    if eligible.empty:
        return None

    return eligible.iloc[-1]


def trading_rows_after_date(df: pd.DataFrame, signal_date) -> pd.DataFrame:
    if df.empty:
        return df

    working = df.copy()
    working["date_only"] = pd.to_datetime(working["timestamp"]).dt.date
    return working[working["date_only"] > signal_date].copy().reset_index(drop=True)


def n_trading_day_return(df: pd.DataFrame, signal_date, signal_close: float, n: int):
    after = trading_rows_after_date(df, signal_date)

    if len(after) < n:
        return None

    close_n = float(after.iloc[n - 1]["close"])
    return (close_n / signal_close - 1) * 100


def return_since_signal(current_close: float, signal_close: float):
    if signal_close is None or signal_close == 0:
        return None

    return (current_close / signal_close - 1) * 100


def safe_round_pct(value):
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


def get_benchmark_tracker_metrics(benchmark_symbol: str, signal_date_input, days_needed: int) -> dict:
    """
    Calculates benchmark returns from the same signal-date close rule used for stocks.
    """
    try:
        benchmark_df = get_daily_bars(benchmark_symbol, days=days_needed)
        benchmark_df = prepare_indicators(benchmark_df)

        signal_row = find_signal_candle_on_or_before(benchmark_df, signal_date_input)

        if signal_row is None or benchmark_df.empty:
            return {}

        signal_date_used = date_only_from_timestamp(signal_row["timestamp"])
        signal_close = float(signal_row["close"])
        latest_close = float(benchmark_df.iloc[-1]["close"])

        current_return = return_since_signal(latest_close, signal_close)

        return {
            f"{benchmark_symbol.lower()}_signal_close": round(signal_close, 2),
            f"{benchmark_symbol.lower()}_latest_close": round(latest_close, 2),
            f"{benchmark_symbol.lower()}_current_return_pct": safe_round_pct(current_return),
            f"{benchmark_symbol.lower()}_return_1d_pct": safe_round_pct(n_trading_day_return(benchmark_df, signal_date_used, signal_close, 1)),
            f"{benchmark_symbol.lower()}_return_3d_pct": safe_round_pct(n_trading_day_return(benchmark_df, signal_date_used, signal_close, 3)),
            f"{benchmark_symbol.lower()}_return_5d_pct": safe_round_pct(n_trading_day_return(benchmark_df, signal_date_used, signal_close, 5)),
            f"{benchmark_symbol.lower()}_return_10d_pct": safe_round_pct(n_trading_day_return(benchmark_df, signal_date_used, signal_close, 10)),
            f"{benchmark_symbol.lower()}_return_20d_pct": safe_round_pct(n_trading_day_return(benchmark_df, signal_date_used, signal_close, 20)),
        }

    except Exception:
        return {}


def add_benchmark_comparison(row: dict) -> dict:
    """
    Adds excess return and beat-benchmark fields for QQQ and SPY.
    """
    for benchmark in ["qqq", "spy"]:
        benchmark_current = row.get(f"{benchmark}_current_return_pct")
        stock_current = row.get("current_return_pct")

        if stock_current is not None and benchmark_current is not None:
            excess = stock_current - benchmark_current
            row[f"excess_vs_{benchmark}_current_pct"] = round(excess, 2)
            row[f"beat_{benchmark}_current"] = excess > 0
        else:
            row[f"excess_vs_{benchmark}_current_pct"] = None
            row[f"beat_{benchmark}_current"] = None

        for horizon in [1, 3, 5, 10, 20]:
            stock_return = row.get(f"return_{horizon}d_pct")
            benchmark_return = row.get(f"{benchmark}_return_{horizon}d_pct")

            if stock_return is not None and benchmark_return is not None:
                excess = stock_return - benchmark_return
                row[f"excess_vs_{benchmark}_{horizon}d_pct"] = round(excess, 2)
                row[f"beat_{benchmark}_{horizon}d"] = excess > 0
            else:
                row[f"excess_vs_{benchmark}_{horizon}d_pct"] = None
                row[f"beat_{benchmark}_{horizon}d"] = None

    return row


def assess_manual_sell_signal(df: pd.DataFrame, signal_close: float) -> dict:
    """
    Manual sell-status logic using independent sell pillars.

    SELL SIGNAL:
    - Hard Risk: stop loss hit, or
    - Structure Failure: box breakdown, or
    - 2+ medium warnings from Trend / Momentum / Price Action.

    ALMOST SELL:
    - Exactly 1 medium warning plus weak price structure.

    CAUTION HOLD:
    - Exactly 1 medium warning, but structure is still acceptable.

    HOLD:
    - No warning.
    """
    df = prepare_indicators(df)

    if len(df) < max(50, BOX_LOOKBACK_DAYS + 10):
        return {
            "sell_signal_type": "UNKNOWN",
            "sell_notice_date": "",
            "exit_reason": "Not enough data",
        }

    latest = df.iloc[-1]
    latest_close = float(latest["close"])
    latest_date = date_only_from_timestamp(latest["timestamp"])

    ma10 = float(latest["MA10"])
    box_bottom = float(latest["BOX_BOTTOM"])
    box_mid = float(latest["BOX_MID"])

    hard_risk_stop_loss = latest_close <= float(signal_close) * (1 - STOP_LOSS_PCT)
    structure_failure = latest_close < box_bottom
    trend_weakness = latest_close < ma10
    momentum_weakness = crossed_down_today(df, "MACD", "MACD_SIGNAL")
    price_action_weakness = three_consecutive_down_closes(df)

    sell_pillars = {
        "Hard Risk: stop loss hit": hard_risk_stop_loss,
        f"Structure Failure: close < previous {BOX_LOOKBACK_DAYS}-day box bottom": structure_failure,
        "Trend Weakness: close < MA10": trend_weakness,
        "Momentum Weakness: MACD crossed down": momentum_weakness,
        "Price Action Weakness: three consecutive down closes": price_action_weakness,
    }

    medium_warnings = {
        "Trend Weakness: close < MA10": trend_weakness,
        "Momentum Weakness: MACD crossed down": momentum_weakness,
        "Price Action Weakness: three consecutive down closes": price_action_weakness,
    }

    medium_warning_count = sum(1 for value in medium_warnings.values() if value)
    medium_reasons = [name for name, value in medium_warnings.items() if value]

    if hard_risk_stop_loss:
        sell_signal_type = "SELL SIGNAL"
        exit_reason = "Hard Risk: stop loss hit"
    elif structure_failure:
        sell_signal_type = "SELL SIGNAL"
        exit_reason = "Structure Failure: box breakdown"
    elif medium_warning_count >= 2:
        sell_signal_type = "SELL SIGNAL"
        exit_reason = "; ".join(medium_reasons)
    elif medium_warning_count == 1:
        near_ma10 = latest_close <= ma10 * 1.02
        weak_structure = latest_close < box_mid

        if near_ma10 or weak_structure:
            sell_signal_type = "ALMOST SELL"
            exit_reason = "; ".join(medium_reasons)
        else:
            sell_signal_type = "CAUTION HOLD"
            exit_reason = "; ".join(medium_reasons)
    else:
        sell_signal_type = "HOLD"
        exit_reason = "No sell warning"

    sell_notice_date = str(latest_date) if sell_signal_type in ["SELL SIGNAL", "ALMOST SELL", "CAUTION HOLD"] else ""

    return {
        "sell_signal_type": sell_signal_type,
        "sell_notice_date": sell_notice_date,
        "latest_date": str(latest_date),
        "latest_close": round(latest_close, 2),
        "sell_signal_price": round(latest_close, 2) if sell_signal_type == "SELL SIGNAL" else None,
        "ma10": round(ma10, 2),
        "box_bottom": round(box_bottom, 2),
        "box_mid": round(box_mid, 2),
        "sell_pillars": sell_pillars,
        "hard_risk_stop_loss": hard_risk_stop_loss,
        "structure_failure": structure_failure,
        "trend_weakness": trend_weakness,
        "momentum_weakness": momentum_weakness,
        "price_action_weakness": price_action_weakness,
        "stop_loss_hit": hard_risk_stop_loss,
        "box_breakdown": structure_failure,
        "macd_cross_down": momentum_weakness,
        "close_below_ma10": trend_weakness,
        "three_down_days": price_action_weakness,
        "medium_warning_count": medium_warning_count,
        "exit_reason": exit_reason,
    }

def build_manual_tracker_row(
    symbol: str,
    signal_date_input,
    original_signal_type: str,
    list_name: str,
) -> dict:
    symbol = symbol.strip().upper()

    days_needed = max(
        LOOKBACK_DAYS,
        (datetime.now().date() - signal_date_input).days + 90,
    )

    df = get_daily_bars(symbol, days=days_needed)
    df = prepare_indicators(df)

    signal_row = find_signal_candle_on_or_before(df, signal_date_input)

    if signal_row is None:
        return {
            "list_name": list_name,
            "ticker": symbol,
            "original_signal_type": original_signal_type,
            "error": "No candle found on or before selected signal date.",
        }

    latest = df.iloc[-1]

    signal_date_used = date_only_from_timestamp(signal_row["timestamp"])
    signal_close = float(signal_row["close"])

    latest_date = date_only_from_timestamp(latest["timestamp"])
    latest_close = float(latest["close"])

    days_since_signal = len(trading_rows_after_date(df, signal_date_used))

    after_or_signal = df.copy()
    after_or_signal["date_only"] = pd.to_datetime(after_or_signal["timestamp"]).dt.date
    after_or_signal = after_or_signal[after_or_signal["date_only"] >= signal_date_used].copy()

    max_gain = None
    max_drawdown = None

    if not after_or_signal.empty and signal_close:
        highest_close = float(after_or_signal["close"].max())
        lowest_close = float(after_or_signal["close"].min())
        max_gain = (highest_close / signal_close - 1) * 100
        max_drawdown = (lowest_close / signal_close - 1) * 100

    sell_info = assess_manual_sell_signal(df, signal_close)
    current_return = return_since_signal(latest_close, signal_close)

    row = {
        "list_name": list_name,
        "ticker": symbol,
        "original_signal_type": original_signal_type,
        "input_signal_date": str(signal_date_input),
        "signal_date_used": str(signal_date_used),
        "signal_close": round(signal_close, 2),
        "latest_date": str(latest_date),
        "latest_close": round(latest_close, 2),
        "days_since_signal": days_since_signal,
        "current_return_pct": safe_round_pct(current_return),
        "return_1d_pct": safe_round_pct(n_trading_day_return(df, signal_date_used, signal_close, 1)),
        "return_3d_pct": safe_round_pct(n_trading_day_return(df, signal_date_used, signal_close, 3)),
        "return_5d_pct": safe_round_pct(n_trading_day_return(df, signal_date_used, signal_close, 5)),
        "return_10d_pct": safe_round_pct(n_trading_day_return(df, signal_date_used, signal_close, 10)),
        "return_20d_pct": safe_round_pct(n_trading_day_return(df, signal_date_used, signal_close, 20)),
        "max_gain_since_signal_pct": safe_round_pct(max_gain),
        "max_drawdown_since_signal_pct": safe_round_pct(max_drawdown),
        "sell_signal_type": sell_info.get("sell_signal_type"),
        "sell_notice_date": sell_info.get("sell_notice_date"),
        "sell_signal_price": sell_info.get("sell_signal_price"),
        "exit_reason": sell_info.get("exit_reason"),
        "hard_risk_stop_loss": sell_info.get("hard_risk_stop_loss"),
        "structure_failure": sell_info.get("structure_failure"),
        "trend_weakness": sell_info.get("trend_weakness"),
        "momentum_weakness": sell_info.get("momentum_weakness"),
        "price_action_weakness": sell_info.get("price_action_weakness"),
        "stop_loss_hit": sell_info.get("stop_loss_hit"),
        "box_breakdown": sell_info.get("box_breakdown"),
        "macd_cross_down": sell_info.get("macd_cross_down"),
        "close_below_ma10": sell_info.get("close_below_ma10"),
        "three_down_days": sell_info.get("three_down_days"),
        "medium_warning_count": sell_info.get("medium_warning_count"),
        "error": "",
    }

    qqq_metrics = get_benchmark_tracker_metrics("QQQ", signal_date_input, days_needed)
    spy_metrics = get_benchmark_tracker_metrics("SPY", signal_date_input, days_needed)

    row.update(qqq_metrics)
    row.update(spy_metrics)
    row = add_benchmark_comparison(row)

    return row


def summarize_manual_tracker_results(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    valid_df = results_df[results_df["error"].fillna("") == ""].copy()

    if valid_df.empty:
        return pd.DataFrame()

    rows = []

    for sell_type, group in valid_df.groupby("sell_signal_type", dropna=False):
        current_returns = group["current_return_pct"].dropna()
        excess_qqq = group["excess_vs_qqq_current_pct"].dropna() if "excess_vs_qqq_current_pct" in group else pd.Series(dtype=float)
        excess_spy = group["excess_vs_spy_current_pct"].dropna() if "excess_vs_spy_current_pct" in group else pd.Series(dtype=float)

        rows.append(
            {
                "Sell Status": sell_type,
                "Count": len(group),
                "Average Current Return %": round(current_returns.mean(), 2) if current_returns.size else None,
                "Win Rate %": round((current_returns > 0).mean() * 100, 2) if current_returns.size else None,
                "Average Excess vs QQQ %": round(excess_qqq.mean(), 2) if excess_qqq.size else None,
                "Beat QQQ Rate %": round((excess_qqq > 0).mean() * 100, 2) if excess_qqq.size else None,
                "Average Excess vs SPY %": round(excess_spy.mean(), 2) if excess_spy.size else None,
                "Beat SPY Rate %": round((excess_spy > 0).mean() * 100, 2) if excess_spy.size else None,
            }
        )

    return pd.DataFrame(rows)


def clean_filename_text(value: str) -> str:
    value = value or "signal_list"
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "signal_list"


def build_return_horizon_scorecard(valid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes return performance at the main horizons the user wants:
    1D, 5D, 10D, and 20D.
    """
    if valid_df.empty:
        return pd.DataFrame()

    rows = []

    for label, column in [
        ("Current", "current_return_pct"),
        ("1D", "return_1d_pct"),
        ("5D", "return_5d_pct"),
        ("10D", "return_10d_pct"),
        ("20D", "return_20d_pct"),
    ]:
        values = valid_df[column].dropna() if column in valid_df.columns else pd.Series(dtype=float)

        if values.empty:
            rows.append(
                {
                    "Horizon": label,
                    "Available Signals": 0,
                    "Average Return %": None,
                    "Median Return %": None,
                    "Win Rate %": None,
                    "Best Return %": None,
                    "Worst Return %": None,
                }
            )
            continue

        rows.append(
            {
                "Horizon": label,
                "Available Signals": int(values.count()),
                "Average Return %": round(values.mean(), 2),
                "Median Return %": round(values.median(), 2),
                "Win Rate %": round((values > 0).mean() * 100, 2),
                "Best Return %": round(values.max(), 2),
                "Worst Return %": round(values.min(), 2),
            }
        )

    return pd.DataFrame(rows)


def build_manual_overall_scorecard(valid_df: pd.DataFrame) -> pd.DataFrame:
    if valid_df.empty:
        return pd.DataFrame()

    rows = []

    groups = [("All", valid_df)]

    if "original_signal_type" in valid_df.columns:
        for signal_type, group in valid_df.groupby("original_signal_type"):
            groups.append((signal_type, group))

    for label, group in groups:
        current_returns = group["current_return_pct"].dropna()
        excess_qqq = group["excess_vs_qqq_current_pct"].dropna() if "excess_vs_qqq_current_pct" in group else pd.Series(dtype=float)
        excess_spy = group["excess_vs_spy_current_pct"].dropna() if "excess_vs_spy_current_pct" in group else pd.Series(dtype=float)

        rows.append(
            {
                "Group": label,
                "Count": len(group),
                "Avg Current Return %": round(current_returns.mean(), 2) if current_returns.size else None,
                "Win Rate %": round((current_returns > 0).mean() * 100, 2) if current_returns.size else None,
                "Avg Excess vs QQQ %": round(excess_qqq.mean(), 2) if excess_qqq.size else None,
                "Beat QQQ Rate %": round((excess_qqq > 0).mean() * 100, 2) if excess_qqq.size else None,
                "Avg Excess vs SPY %": round(excess_spy.mean(), 2) if excess_spy.size else None,
                "Beat SPY Rate %": round((excess_spy > 0).mean() * 100, 2) if excess_spy.size else None,
            }
        )

    return pd.DataFrame(rows)



# ============================================================
# APP UI
# ============================================================

st.title("📈 Diffie's Stock Signal Scanner")
st.caption("A personal finance hobby tool for organizing daily stock signals. Manual decision support only. No order submission.")

with st.expander("🚀 Quick Start / How to Use This Tool", expanded=False):
    st.markdown(
        f"""
        **Recommended workflow**

        1. Open **Market List Scanner** after the market closes.
        2. Choose a list such as S&P 500, Nasdaq-100, Dow 30, or S&P 400 MidCap.
        3. Run the scan and copy the HIGH CONVICTION BUY / BUY SIGNAL / ALMOST BUY tickers.
        4. Save the ticker list together with the signal date.
        5. In the following days, paste the list into **Signal List Tracker** to check returns, QQQ/SPY comparison, and sell warnings.

        **Strategy version:** `{STRATEGY_VERSION}`  
        **Last updated:** `{STRATEGY_UPDATED}`
        """
    )

with st.expander("⚠️ Important Disclaimer and Limitations", expanded=False):
    st.markdown(
        """
        This tool is for **education, research, and manual decision support only**.  
        It is **not financial advice**, not a trading recommendation, and does not guarantee profit.

        **Known limitations**

        - Signals are based on completed daily candles, not intraday execution prices.
        - A stock can have a signal and still lose money.
        - Market gaps, earnings, news, liquidity, and macro events can change risk quickly.
        - Benchmark comparison helps evaluate context, but it does not prove future performance.
        - The tool does not manage trades or take responsibility for anyone's investment risk.
        """
    )

with st.expander("💬 Feedback / Strategy Ideas", expanded=False):
    st.markdown(
        """
        I built this as a personal finance hobby project.  
        Feel free to send feedback, bug reports, or strategy ideas.

        📧 [diffieliu@gmail.com](mailto:diffieliu@gmail.com)  
        🔗 [LinkedIn: Diffie Liu](https://www.linkedin.com/in/diffie-liu/)
        """
    )

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
    st.markdown("🔗 [LinkedIn: Diffie Liu](https://www.linkedin.com/in/diffie-liu/)")
    st.write("📞 909-689-6496")

    st.divider()

    st.header("Scanner Settings")
    st.write("Strategy version:", STRATEGY_VERSION)
    st.write("Updated:", STRATEGY_UPDATED)
    st.write("Market filter:", MARKET_FILTER_SYMBOL)
    st.write("Cross lookback days:", CROSS_LOOKBACK_DAYS)
    st.write("Box lookback days:", BOX_LOOKBACK_DAYS)
    st.write("Stop loss:", f"{STOP_LOSS_PCT * 100:.0f}%")
    st.write("Max distance above MA20:", f"{(MAX_DISTANCE_ABOVE_MA20 - 1) * 100:.0f}%")

    st.divider()

    if st.button("Refresh data / clear cache"):
        st.cache_data.clear()
        st.rerun()

tab_single, tab_sp500, tab_subscribe, tab_performance = st.tabs(
    [
        "🔎 Individual Stock Scanner",
        "📋 Market List Scanner",
        "📬 Daily Email Alerts",
        "📊 Signal List Tracker",
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

            st.subheader("Core Buy Pillars")

            st.write(f"Core pillar score: **{result['core_pillar_score']}/{result['core_pillar_total']}**")

            for pillar_name, passed in result["core_pillars"].items():
                st.write(f"{'✅' if passed else '❌'} {pillar_name}")

            st.subheader("Separate Confirmation Tags")

            for tag_name, passed in result["confirmations"].items():
                st.write(f"{'✅' if passed else '❌'} {tag_name}")

            st.caption("Confirmation tags give extra perspective but do not duplicate the core pillar score.")

            st.subheader("Sell / Weakness Pillars")

            st.write(f"Medium sell warnings triggered: **{result['sell_warning_count']}**")

            for pillar_name, triggered in result["sell_pillars"].items():
                st.write(f"{'⚠️' if triggered else '✅'} {pillar_name}")

            show_reason_summary(result)
            show_price_chart(result)

            if show_fundamental_info:
                show_fundamentals(result["symbol"])

            st.divider()

            st.subheader("Copyable Summary")

            summary_text = (
                f"{result['symbol']} | {result['final_signal']} | "
                f"Close {result['latest_close']:.2f} | "
                f"Core Score {result['core_pillar_score']}/{result['core_pillar_total']} | "
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

            c1, c2, c3, c4, c5, c6 = st.columns(6)

            c1.metric("HIGH CONVICTION", signal_counts.get("HIGH CONVICTION BUY", 0))
            c2.metric("BUY", signal_counts.get("BUY SIGNAL", 0))
            c3.metric("ALMOST BUY", signal_counts.get("ALMOST BUY", 0))
            c4.metric("WATCHLIST", signal_counts.get("WATCHLIST", 0))
            c5.metric("NO ACTION", signal_counts.get("NO ACTION", 0))
            c6.metric("ERROR", signal_counts.get("ERROR", 0))

            st.subheader("Filter Results")

            available_signals = [
                "HIGH CONVICTION BUY",
                "BUY SIGNAL",
                "ALMOST BUY",
                "WATCHLIST",
                "NO ACTION",
                "ERROR",
            ]

            selected_signals = st.multiselect(
                "Signal filter",
                available_signals,
                default=["HIGH CONVICTION BUY", "BUY SIGNAL", "ALMOST BUY"],
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

            if not filtered_df.empty and "core_pillar_score" in filtered_df.columns:
                filtered_df = filtered_df.sort_values(
                    ["final_signal", "core_pillar_score", "distance_from_ma20_pct"],
                    ascending=[True, False, True],
                )

            display_columns = [
                "symbol",
                "company_name",
                "sector",
                "sub_industry",
                "final_signal",
                "latest_close",
                "core_pillar_score",
                "core_pillar_total",
                "market_environment_pass",
                "trend_pass",
                "momentum_pass",
                "structure_pass",
                "risk_extension_pass",
                "volume_confirmed",
                "relative_strength_confirmed",
                "qqq_market_confirmed",
                "distance_from_ma20_pct",
                "stock_ret20",
                "qqq_ret20",
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

            safe_universe_name = selected_universe.replace(" ", "_").replace("&", "and")

            filtered_tickers = ", ".join(
                list(dict.fromkeys(filtered_df["symbol"].dropna().astype(str).tolist()))
            )

            st.text_area(
                "Copy tickers from current filtered table",
                value=filtered_tickers,
                height=80,
                key=f"copy_filtered_table_tickers_{safe_universe_name}",
                help="This copies tickers from the table currently shown above.",
            )

            csv_data = filtered_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download filtered results as CSV",
                data=csv_data,
                file_name=f"{safe_universe_name}_signal_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
                mime="text/csv",
            )

            st.subheader("Top HIGH CONVICTION / BUY / ALMOST BUY List")

            top_df = results_df[
                results_df["final_signal"].isin(["HIGH CONVICTION BUY", "BUY SIGNAL", "ALMOST BUY"])
            ].copy()

            if top_df.empty:
                st.write(f"No HIGH CONVICTION, BUY SIGNAL, or ALMOST BUY stocks in {selected_universe} right now.")

                st.text_area(
                    "All HIGH CONVICTION + BUY SIGNAL + ALMOST BUY tickers",
                    value="",
                    height=80,
                    key=f"copy_high_buy_almost_empty_{safe_universe_name}",
                    help="No BUY SIGNAL or ALMOST BUY tickers are available right now.",
                )

            else:
                top_df = top_df.sort_values(
                    ["core_pillar_score", "distance_from_ma20_pct"],
                    ascending=[False, True],
                )

                buy_almost_tickers = ", ".join(
                    list(dict.fromkeys(top_df["symbol"].dropna().astype(str).tolist()))
                )

                buy_only_df = top_df[top_df["final_signal"] == "BUY SIGNAL"].copy()
                almost_buy_only_df = top_df[top_df["final_signal"] == "ALMOST BUY"].copy()

                buy_only_tickers = ", ".join(
                    list(dict.fromkeys(buy_only_df["symbol"].dropna().astype(str).tolist()))
                )

                almost_buy_only_tickers = ", ".join(
                    list(dict.fromkeys(almost_buy_only_df["symbol"].dropna().astype(str).tolist()))
                )

                st.subheader("Copyable Ticker Lists")

                st.text_area(
                    "All HIGH CONVICTION + BUY SIGNAL + ALMOST BUY tickers",
                    value=buy_almost_tickers,
                    height=90,
                    key=f"copy_all_high_buy_almost_tickers_{safe_universe_name}",
                    help="Copy this list for your tracker, watchlist, or notes.",
                )

                col_buy_copy, col_almost_copy = st.columns(2)

                with col_buy_copy:
                    st.text_area(
                        "BUY SIGNAL only",
                        value=buy_only_tickers,
                        height=90,
                        key=f"copy_buy_only_tickers_{safe_universe_name}",
                    )

                with col_almost_copy:
                    st.text_area(
                        "ALMOST BUY only",
                        value=almost_buy_only_tickers,
                        height=90,
                        key=f"copy_almost_buy_only_tickers_{safe_universe_name}",
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


# ============================================================
# TAB 4: MANUAL PERFORMANCE / SELL TRACKER
# ============================================================

with tab_performance:
    st.markdown("## 📊 Signal List Tracker v2")

    st.info(
        "Paste the ticker list you copied from the Market List Scanner, choose the original signal date, "
        "and this page will calculate returns from that signal-date close, compare performance against QQQ and SPY, "
        "and check whether any ticker currently has SELL SIGNAL, ALMOST SELL, CAUTION HOLD, or HOLD."
    )

    st.caption(
        "This page does not automatically save every scanner result. It only checks the tickers you paste here."
    )

    with st.expander("How this signal list tracker works", expanded=True):
        st.markdown(
            f"""
            **Input workflow**

            1. Run the Market List Scanner.
            2. Copy the ticker list from BUY / ALMOST BUY.
            3. Paste the tickers here.
            4. Enter the date when you received the signal.
            5. Click **Run Signal List Tracker**.

            **Return rule**

            Return starts from the closing price of the completed candle on the signal date.
            If the input date is a weekend or holiday, the app uses the most recent trading day before that date.

            **Benchmark rule**

            The tracker also calculates QQQ and SPY returns from the same signal-date close.
            If your ticker return is higher than QQQ/SPY, the excess return is positive.

            **Main return windows**

            The table keeps the key horizons visible: **1D, 5D, 10D, and 20D**.
            These help you see whether the signal works immediately, over a short swing period,
            and over a medium-term follow-through period.

            **Sell-status logic**

            - **SELL SIGNAL** = stop loss hit, Box breakdown, or 2+ medium sell warnings.
            - **ALMOST SELL** = 1 medium warning plus weak price structure.
            - **CAUTION HOLD** = 1 medium warning, but structure is not weak enough for Almost Sell.
            - **HOLD** = no sell warning.

            Stop loss uses the original signal close:
            `{STOP_LOSS_PCT * 100:.0f}%` below the signal-date close.
            """
        )

    default_signal_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)

    col_input_left, col_input_right = st.columns([2, 1])

    with col_input_left:
        pasted_tickers = st.text_area(
            "Paste tickers from scanner",
            value="",
            height=130,
            placeholder="TSLA, AAPL, NVDA, MSFT",
            help="You can paste comma-separated tickers or one ticker per line.",
        )

    with col_input_right:
        list_name = st.text_input(
            "Tracking list name",
            value=f"Signal List {default_signal_date}",
            help="Example: 2026-05-29 Nasdaq-100 BUY/ALMOST BUY",
        )

        signal_date_input = st.date_input(
            "Original signal date",
            value=default_signal_date,
            help="The date you received the BUY / ALMOST BUY signal.",
        )

        original_signal_type = st.selectbox(
            "Original signal type",
            [
                "MIXED LIST",
                "HIGH CONVICTION BUY",
                "BUY SIGNAL",
                "ALMOST BUY",
            ],
            index=0,
        )

        tracking_note = st.text_input(
            "Optional note",
            value="",
            help="Example: copied from Nasdaq-100 scan after market close",
        )

        run_manual_tracker = st.button("Run Signal List Tracker", type="primary")

    if run_manual_tracker:
        tickers = parse_ticker_list(pasted_tickers)

        if not tickers:
            st.error("Please paste at least one ticker.")
        else:
            st.write(f"Tracking {len(tickers)} tickers from signal date: **{signal_date_input}**")

            rows = []
            progress = st.progress(0)
            status_text = st.empty()

            for i, ticker in enumerate(tickers, start=1):
                status_text.write(f"Checking {ticker} ({i}/{len(tickers)})...")
                try:
                    row = build_manual_tracker_row(
                        symbol=ticker,
                        signal_date_input=signal_date_input,
                        original_signal_type=original_signal_type,
                        list_name=list_name,
                    )
                except Exception as e:
                    row = {
                        "list_name": list_name,
                        "ticker": ticker,
                        "original_signal_type": original_signal_type,
                        "input_signal_date": str(signal_date_input),
                        "error": str(e),
                    }

                rows.append(row)
                progress.progress(i / len(tickers))

            status_text.empty()
            progress.empty()

            results_df = pd.DataFrame(rows)
            results_df["tracking_note"] = tracking_note

            st.divider()

            error_df = results_df[results_df["error"].fillna("") != ""].copy()
            clean_df = results_df[results_df["error"].fillna("") == ""].copy()

            if not error_df.empty:
                st.warning("Some tickers could not be checked.")
                st.dataframe(
                    error_df[["ticker", "error"]],
                    use_container_width=True,
                    hide_index=True,
                )

            if clean_df.empty:
                st.error("No valid ticker results were generated.")
            else:
                st.subheader("Manual Tracking Summary")

                sell_counts = clean_df["sell_signal_type"].value_counts().to_dict()

                c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                c1.metric("SELL SIGNAL", sell_counts.get("SELL SIGNAL", 0))
                c2.metric("ALMOST SELL", sell_counts.get("ALMOST SELL", 0))
                c3.metric("CAUTION HOLD", sell_counts.get("CAUTION HOLD", 0))
                c4.metric("HOLD", sell_counts.get("HOLD", 0))
                c5.metric("Avg Return", f"{clean_df['current_return_pct'].dropna().mean():.2f}%")
                c6.metric("Avg Excess vs QQQ", f"{clean_df['excess_vs_qqq_current_pct'].dropna().mean():.2f}%")
                c7.metric("Avg Excess vs SPY", f"{clean_df['excess_vs_spy_current_pct'].dropna().mean():.2f}%")

                scorecard_df = build_manual_overall_scorecard(clean_df)

                if not scorecard_df.empty:
                    st.write("**Overall strategy scorecard**")
                    st.dataframe(scorecard_df, use_container_width=True, hide_index=True)

                horizon_scorecard_df = build_return_horizon_scorecard(clean_df)

                if not horizon_scorecard_df.empty:
                    st.write("**Return horizon scorecard: 1D / 5D / 10D / 20D**")
                    st.dataframe(horizon_scorecard_df, use_container_width=True, hide_index=True)

                # Best / worst names are based on current return from the signal-date close.
                if not clean_df["current_return_pct"].dropna().empty:
                    best_row = clean_df.sort_values("current_return_pct", ascending=False).iloc[0]
                    worst_row = clean_df.sort_values("current_return_pct", ascending=True).iloc[0]

                    best_col, worst_col = st.columns(2)

                    best_col.success(
                        f"Best current performer: {best_row['ticker']} "
                        f"({best_row['current_return_pct']:.2f}%)"
                    )

                    worst_col.error(
                        f"Worst current performer: {worst_row['ticker']} "
                        f"({worst_row['current_return_pct']:.2f}%)"
                    )

                summary_df = summarize_manual_tracker_results(clean_df)

                if not summary_df.empty:
                    st.write("**Sell-status scorecard**")
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

                st.subheader("Filter Manual Results")

                status_options = ["SELL SIGNAL", "ALMOST SELL", "CAUTION HOLD", "HOLD", "UNKNOWN"]
                selected_status_options = st.multiselect(
                    "Sell status filter",
                    status_options,
                    default=["SELL SIGNAL", "ALMOST SELL", "CAUTION HOLD", "HOLD"],
                )

                filtered_manual_df = clean_df[
                    clean_df["sell_signal_type"].isin(selected_status_options)
                ].copy()

                display_cols = [
                    "list_name",
                    "tracking_note",
                    "ticker",
                    "original_signal_type",
                    "signal_date_used",
                    "signal_close",
                    "latest_date",
                    "latest_close",
                    "current_return_pct",
                    "return_1d_pct",
                    "return_5d_pct",
                    "return_10d_pct",
                    "return_20d_pct",
                    "return_3d_pct",
                    "qqq_current_return_pct",
                    "spy_current_return_pct",
                    "excess_vs_qqq_current_pct",
                    "excess_vs_spy_current_pct",
                    "beat_qqq_current",
                    "beat_spy_current",
                    "max_gain_since_signal_pct",
                    "max_drawdown_since_signal_pct",
                    "sell_signal_type",
                    "sell_notice_date",
                    "sell_signal_price",
                    "exit_reason",
                    "hard_risk_stop_loss",
                    "structure_failure",
                    "trend_weakness",
                    "momentum_weakness",
                    "price_action_weakness",
                    "stop_loss_hit",
                    "box_breakdown",
                    "macd_cross_down",
                    "close_below_ma10",
                    "three_down_days",
                ]

                display_cols = [col for col in display_cols if col in filtered_manual_df.columns]

                st.subheader("Signal List Tracker Results")
                st.dataframe(
                    filtered_manual_df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )

                sell_tickers = ", ".join(
                    filtered_manual_df[
                        filtered_manual_df["sell_signal_type"] == "SELL SIGNAL"
                    ]["ticker"].dropna().astype(str).tolist()
                )

                almost_sell_tickers = ", ".join(
                    filtered_manual_df[
                        filtered_manual_df["sell_signal_type"] == "ALMOST SELL"
                    ]["ticker"].dropna().astype(str).tolist()
                )

                caution_tickers = ", ".join(
                    filtered_manual_df[
                        filtered_manual_df["sell_signal_type"] == "CAUTION HOLD"
                    ]["ticker"].dropna().astype(str).tolist()
                )

                hold_tickers = ", ".join(
                    filtered_manual_df[
                        filtered_manual_df["sell_signal_type"] == "HOLD"
                    ]["ticker"].dropna().astype(str).tolist()
                )

                st.subheader("Copyable Result Lists")

                col_copy_1, col_copy_2 = st.columns(2)

                with col_copy_1:
                    st.text_area(
                        "SELL SIGNAL tickers",
                        value=sell_tickers,
                        height=80,
                        key="manual_tracker_sell_tickers",
                    )

                    st.text_area(
                        "ALMOST SELL tickers",
                        value=almost_sell_tickers,
                        height=80,
                        key="manual_tracker_almost_sell_tickers",
                    )

                with col_copy_2:
                    st.text_area(
                        "CAUTION HOLD tickers",
                        value=caution_tickers,
                        height=80,
                        key="manual_tracker_caution_tickers",
                    )

                    st.text_area(
                        "HOLD tickers",
                        value=hold_tickers,
                        height=80,
                        key="manual_tracker_hold_tickers",
                    )

                csv_data = results_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download signal list tracker results as CSV",
                    data=csv_data,
                    file_name=f"manual_tracker_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
                    mime="text/csv",
                )

                st.caption(
                    "This signal list tracker does not store your pasted list. Download the CSV if you want to keep the record."
                )

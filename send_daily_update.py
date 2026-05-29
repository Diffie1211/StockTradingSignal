import os
from io import StringIO
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import defaultdict

import pandas as pd
import requests
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from subscription_utils import (
    get_active_subscriptions,
    selected_universes_from_subscription,
    format_manage_link,
    get_secret,
)


load_dotenv()


# ============================================================
# SETTINGS
# ============================================================

MARKET_FILTER_SYMBOL = "QQQ"

LOOKBACK_DAYS = 260
CROSS_LOOKBACK_DAYS = 3
MAX_DISTANCE_ABOVE_MA20 = 1.20
BOX_LOOKBACK_DAYS = 7
WATCHLIST_MIN_BUY_SCORE = 4

INDEX_UNIVERSES = {
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "Nasdaq-100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "Dow 30": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    "S&P 400 MidCap": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
}

CHUNK_SIZE = 100


# ============================================================
# SECRETS
# ============================================================

ALPACA_API_KEY = get_secret("ALPACA_API_KEY")
ALPACA_SECRET_KEY = get_secret("ALPACA_SECRET_KEY")

RESEND_API_KEY = get_secret("RESEND_API_KEY")
RESEND_FROM_EMAIL = get_secret(
    "RESEND_FROM_EMAIL",
    "Diffie's Stock Scanner <onboarding@resend.dev>",
)

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise RuntimeError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY.")

if not RESEND_API_KEY:
    raise RuntimeError("Missing RESEND_API_KEY.")


data_client = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
)


# ============================================================
# UNIVERSE LISTS
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


def read_wikipedia_tables(url: str) -> list[pd.DataFrame]:
    response = requests.get(url, headers=wiki_headers(), timeout=30)
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


def get_universe_symbols(universe_name: str) -> pd.DataFrame:
    if universe_name not in INDEX_UNIVERSES:
        raise RuntimeError(f"Unknown universe: {universe_name}")

    tables = read_wikipedia_tables(INDEX_UNIVERSES[universe_name])

    if universe_name == "S&P 500":
        df = first_table_with_columns(tables, ["Symbol", "Security"])

    elif universe_name == "Nasdaq-100":
        df = first_table_with_columns(tables, ["Ticker", "Company"])
        df = df.rename(columns={"Ticker": "Symbol", "Company": "Security"})

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

    df = df.dropna(subset=["Symbol"])
    df = df[df["Symbol"] != ""]
    df = df.drop_duplicates(subset=["Symbol"]).reset_index(drop=True)

    return df


# ============================================================
# MARKET DATA
# ============================================================

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
    df["MA20"] = df["close"].rolling(20).mean()
    return df


def add_box_theory_levels(df: pd.DataFrame) -> pd.DataFrame:
    df["BOX_TOP"] = df["high"].shift(1).rolling(BOX_LOOKBACK_DAYS).max()
    df["BOX_BOTTOM"] = df["low"].shift(1).rolling(BOX_LOOKBACK_DAYS).min()
    return df


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_unfinished_daily_candle(df)
    df = add_kdj(df)
    df = add_macd(df)
    df = add_moving_averages(df)
    df = add_box_theory_levels(df)
    return df


def cross_up(prev_a, prev_b, now_a, now_b) -> bool:
    return prev_a <= prev_b and now_a > now_b


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


def scan_one_symbol(
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
    }


def scan_universe(universe_name: str) -> tuple[pd.DataFrame, dict]:
    universe = get_universe_symbols(universe_name)
    symbols = universe["Symbol"].tolist()
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
        "universe_count": len(universe),
        "qqq_close": round(qqq_close, 2),
        "qqq_ma20": round(qqq_ma20, 2),
        "qqq_above_ma20": qqq_above_ma20,
        "qqq_latest_date": str(qqq_latest["timestamp"]),
    }

    results = []

    for _, row in universe.iterrows():
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

        result = scan_one_symbol(
            symbol=symbol,
            company_name=company_name,
            sector=sector,
            sub_industry=sub_industry,
            df=symbol_df,
            qqq_above_ma20=qqq_above_ma20,
        )

        results.append(result)

    return pd.DataFrame(results), market_info


# ============================================================
# EMAIL
# ============================================================

def build_universe_section_html(universe_name: str, results_df: pd.DataFrame) -> str:
    top_df = results_df[
        results_df["final_signal"].isin(["BUY SIGNAL", "ALMOST BUY"])
    ].copy()

    if top_df.empty:
        return f"""
        <h3>{universe_name}</h3>
        <p>No BUY SIGNAL or ALMOST BUY stocks right now.</p>
        """

    top_df = top_df.sort_values(
        ["final_signal", "buy_score", "distance_from_ma20_pct"],
        ascending=[True, False, True],
    )

    rows_html = ""

    for _, row in top_df.iterrows():
        rows_html += f"""
        <tr>
            <td><strong>{row.get('symbol', '')}</strong></td>
            <td>{row.get('company_name', '')}</td>
            <td>{row.get('final_signal', '')}</td>
            <td>{row.get('buy_score', '')}/{row.get('buy_total', '')}</td>
            <td>{row.get('latest_close', '')}</td>
            <td>{row.get('failed_checks', '')}</td>
        </tr>
        """

    return f"""
    <h3>{universe_name}</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
        <thead>
            <tr>
                <th>Symbol</th>
                <th>Company</th>
                <th>Signal</th>
                <th>Score</th>
                <th>Close</th>
                <th>Missing checks</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """


def build_email_html(subscription: dict, universe_results: dict[str, pd.DataFrame]) -> str:
    selected_universes = selected_universes_from_subscription(subscription)

    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    manage_link = format_manage_link(subscription["token"])

    sections = ""

    for universe_name in selected_universes:
        results_df = universe_results.get(universe_name)

        if results_df is None:
            sections += f"<h3>{universe_name}</h3><p>Could not scan this list today.</p>"
        else:
            sections += build_universe_section_html(universe_name, results_df)

    return f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.5;">
        <h2>Diffie's Daily Stock Signal Update — {today}</h2>

        <p>
            This email lists <strong>BUY SIGNAL</strong> and <strong>ALMOST BUY</strong>
            stocks based on Diffie's KDJ + MACD + MA20 + QQQ + 7-day Box Theory strategy.
        </p>

        <p>
            This is for manual decision support only. It does not place trades and does not guarantee profit.
        </p>

        {sections}

        <hr>

        <p>
            Manage or cancel your subscription here:<br>
            <a href="{manage_link}">{manage_link}</a>
        </p>
    </div>
    """


def send_email(to_email: str, subject: str, html: str) -> None:
    response = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "from": RESEND_FROM_EMAIL,
            "to": [to_email],
            "subject": subject,
            "html": html,
        },
        timeout=30,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"Resend error {response.status_code}: {response.text}")


# ============================================================
# MAIN
# ============================================================

def main():
    subscribers = get_active_subscriptions()

    if not subscribers:
        print("No active subscribers.")
        return

    print(f"Active subscribers: {len(subscribers)}")

    needed_universes = set()

    for subscriber in subscribers:
        for universe_name in selected_universes_from_subscription(subscriber):
            needed_universes.add(universe_name)

    universe_results = {}

    for universe_name in sorted(needed_universes):
        print(f"Scanning {universe_name}...")
        results_df, market_info = scan_universe(universe_name)
        universe_results[universe_name] = results_df
        print(f"Finished {universe_name}: {len(results_df)} rows")

    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    for subscriber in subscribers:
        email = subscriber["email"]
        html = build_email_html(subscriber, universe_results)
        subject = f"Diffie's Daily Stock Signals — {today}"

        print(f"Sending email to {email}...")
        send_email(email, subject, html)

    print("Daily update complete.")


if __name__ == "__main__":
    main()

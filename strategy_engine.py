import os
from io import StringIO
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


MARKET_FILTER_SYMBOL = "QQQ"
BROAD_BENCHMARK_SYMBOL = "SPY"

LOOKBACK_DAYS = 260
CROSS_LOOKBACK_DAYS = 3
STOP_LOSS_PCT = 0.10
MAX_DISTANCE_ABOVE_MA20 = 1.20
BOX_LOOKBACK_DAYS = 7
WATCHLIST_MIN_BUY_SCORE = 4
VOLUME_CONFIRMATION_MULTIPLE = 1.20

INDEX_UNIVERSES = {
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "Nasdaq-100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "Dow 30": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    "S&P 400 MidCap": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
}

UNIVERSE_BENCHMARKS = {
    "S&P 500": "SPY",
    "Nasdaq-100": "QQQ",
    "Dow 30": "DIA",
    "S&P 400 MidCap": "MDY",
}

CHUNK_SIZE = 100


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

def get_daily_bars_for_symbols(data_client, symbols: list[str], days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    all_bars = []
    last_error = None

    clean_symbols = sorted(set(s.strip().upper() for s in symbols if s and str(s).strip()))

    for i in range(0, len(clean_symbols), CHUNK_SIZE):
        chunk = clean_symbols[i:i + CHUNK_SIZE]
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
    df["MA10"] = df["close"].rolling(10).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    if "volume" in df.columns:
        df["VOL20"] = df["volume"].rolling(20).mean()
    else:
        df["VOL20"] = pd.NA
    df["RET20"] = df["close"] / df["close"].shift(20) - 1
    return df


def add_box_theory_levels(df: pd.DataFrame) -> pd.DataFrame:
    df["BOX_TOP"] = df["high"].shift(1).rolling(BOX_LOOKBACK_DAYS).max()
    df["BOX_BOTTOM"] = df["low"].shift(1).rolling(BOX_LOOKBACK_DAYS).min()
    df["BOX_MID"] = (df["BOX_TOP"] + df["BOX_BOTTOM"]) / 2
    return df


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = remove_unfinished_daily_candle(df)
    df = add_kdj(df)
    df = add_macd(df)
    df = add_moving_averages(df)
    df = add_box_theory_levels(df)
    return df


# ============================================================
# SIGNAL LOGIC
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
    return cross_down(previous[col_a], previous[col_b], latest[col_a], latest[col_b])


def three_consecutive_down_closes(df: pd.DataFrame) -> bool:
    if len(df) < 4:
        return False
    c4 = df.iloc[-4]["close"]
    c3 = df.iloc[-3]["close"]
    c2 = df.iloc[-2]["close"]
    c1 = df.iloc[-1]["close"]
    return c3 < c4 and c2 < c3 and c1 < c2


def _safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def assess_buy_signal(
    df: pd.DataFrame,
    qqq_above_ma20: bool,
    qqq_ret20: float | None,
    universe_benchmark_above_ma20: bool = True,
) -> dict:
    latest = df.iloc[-1]
    latest_close = float(latest["close"])
    ma20 = float(latest["MA20"])
    box_top = float(latest["BOX_TOP"])
    box_bottom = float(latest["BOX_BOTTOM"])

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

    failed_checks = [name for name, passed in buy_checks.items() if not passed]

    volume = _safe_float(latest.get("volume"), 0)
    vol20 = _safe_float(latest.get("VOL20"), None)
    volume_confirmed = False
    if vol20 and vol20 > 0:
        volume_confirmed = volume >= vol20 * VOLUME_CONFIRMATION_MULTIPLE

    stock_ret20 = _safe_float(latest.get("RET20"), None)
    relative_strength_confirmed = False
    if stock_ret20 is not None and qqq_ret20 is not None:
        relative_strength_confirmed = stock_ret20 > qqq_ret20

    high_conviction_buy = (
        buy_score == buy_total
        and volume_confirmed
        and relative_strength_confirmed
        and universe_benchmark_above_ma20
    )

    if high_conviction_buy:
        final_signal = "HIGH CONVICTION BUY"
    elif buy_score == buy_total:
        final_signal = "BUY SIGNAL"
    elif buy_score == buy_total - 1 and safety_filters_pass:
        final_signal = "ALMOST BUY"
    elif buy_score >= WATCHLIST_MIN_BUY_SCORE and safety_filters_pass:
        final_signal = "WATCHLIST"
    else:
        final_signal = "NO ACTION"

    distance_from_ma20_pct = (latest_close / ma20 - 1) * 100 if ma20 else None

    return {
        "final_signal": final_signal,
        "latest_date": str(latest["timestamp"]),
        "latest_close": round(latest_close, 2),
        "ma20": round(ma20, 2),
        "box_top": round(box_top, 2),
        "box_bottom": round(box_bottom, 2),
        "distance_from_ma20_pct": round(distance_from_ma20_pct, 2) if distance_from_ma20_pct is not None else None,
        "buy_score": buy_score,
        "buy_total": buy_total,
        "failed_checks": "; ".join(failed_checks),
        "kdj_cross_recent": kdj_cross_recent,
        "macd_cross_recent": macd_cross_recent,
        "stock_above_ma20": stock_above_ma20,
        "qqq_above_ma20": qqq_above_ma20,
        "not_too_extended": not_too_extended,
        "box_breakout": box_breakout,
        "volume": volume,
        "vol20": round(vol20, 2) if vol20 is not None else None,
        "volume_confirmed": volume_confirmed,
        "stock_ret20": round(stock_ret20 * 100, 2) if stock_ret20 is not None else None,
        "qqq_ret20": round(qqq_ret20 * 100, 2) if qqq_ret20 is not None else None,
        "relative_strength_confirmed": relative_strength_confirmed,
        "universe_benchmark_confirmed": universe_benchmark_above_ma20,
        "high_conviction_buy": high_conviction_buy,
    }


def assess_sell_signal(df: pd.DataFrame, signal_close: float) -> dict:
    df = prepare_indicators(df)

    if len(df) < max(50, BOX_LOOKBACK_DAYS + 10):
        return {
            "sell_signal_type": "UNKNOWN",
            "exit_reason": "Not enough data",
        }

    latest = df.iloc[-1]
    latest_close = float(latest["close"])
    ma10 = float(latest["MA10"])
    box_bottom = float(latest["BOX_BOTTOM"])
    box_mid = float(latest["BOX_MID"])

    stop_loss_hit = latest_close <= float(signal_close) * (1 - STOP_LOSS_PCT)
    box_breakdown = latest_close < box_bottom
    macd_cross_down = crossed_down_today(df, "MACD", "MACD_SIGNAL")
    close_below_ma10 = latest_close < ma10
    three_down_days = three_consecutive_down_closes(df)

    medium_warnings = {
        "MACD crossed down": macd_cross_down,
        "Close < MA10": close_below_ma10,
        "Three consecutive down closes": three_down_days,
    }

    medium_warning_count = sum(1 for value in medium_warnings.values() if value)

    hard_reasons = []
    if stop_loss_hit:
        hard_reasons.append("Stop loss hit")
    if box_breakdown:
        hard_reasons.append("Box breakdown")

    medium_reasons = [name for name, value in medium_warnings.items() if value]

    if stop_loss_hit:
        sell_signal_type = "SELL SIGNAL"
        exit_reason = "Stop loss hit"
        status = "CLOSED_BY_STOP_LOSS"
    elif box_breakdown:
        sell_signal_type = "SELL SIGNAL"
        exit_reason = "Box breakdown"
        status = "CLOSED_BY_SELL_SIGNAL"
    elif medium_warning_count >= 2:
        sell_signal_type = "SELL SIGNAL"
        exit_reason = "; ".join(medium_reasons)
        status = "CLOSED_BY_SELL_SIGNAL"
    elif medium_warning_count == 1:
        near_ma10 = latest_close <= ma10 * 1.02
        weak_structure = latest_close < box_mid
        if near_ma10 or weak_structure:
            sell_signal_type = "ALMOST SELL"
            exit_reason = "; ".join(medium_reasons)
            status = "OPEN_WITH_ALMOST_SELL"
        else:
            sell_signal_type = "CAUTION HOLD"
            exit_reason = "; ".join(medium_reasons)
            status = "OPEN_WITH_CAUTION"
    else:
        sell_signal_type = "HOLD"
        exit_reason = "No sell warning"
        status = "OPEN"

    return {
        "sell_signal_type": sell_signal_type,
        "latest_date": str(latest["timestamp"]),
        "latest_close": round(latest_close, 2),
        "sell_signal_price": round(latest_close, 2),
        "stop_loss_hit": stop_loss_hit,
        "box_breakdown": box_breakdown,
        "macd_cross_down": macd_cross_down,
        "close_below_ma10": close_below_ma10,
        "three_down_days": three_down_days,
        "medium_warning_count": medium_warning_count,
        "exit_reason": exit_reason,
        "status": status,
    }


def scan_one_symbol(
    symbol: str,
    company_name: str,
    sector: str,
    sub_industry: str,
    df: pd.DataFrame,
    qqq_above_ma20: bool,
    qqq_ret20: float | None,
    universe_benchmark_above_ma20: bool = True,
) -> dict:
    symbol = clean_symbol(symbol)
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
    if pd.isna(latest.get("BOX_TOP")) or pd.isna(latest.get("BOX_BOTTOM")) or pd.isna(latest.get("MA20")):
        return {
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector,
            "sub_industry": sub_industry,
            "final_signal": "ERROR",
            "error": "Not enough indicator data",
        }

    result = assess_buy_signal(
        df=df,
        qqq_above_ma20=qqq_above_ma20,
        qqq_ret20=qqq_ret20,
        universe_benchmark_above_ma20=universe_benchmark_above_ma20,
    )

    return {
        "symbol": symbol,
        "company_name": company_name,
        "sector": sector,
        "sub_industry": sub_industry,
        **result,
    }


def scan_universe(data_client, universe_name: str) -> tuple[pd.DataFrame, dict, dict[str, pd.DataFrame]]:
    universe_df = get_universe_symbols(universe_name)
    universe_symbols = universe_df["Symbol"].tolist()

    benchmark_symbol = UNIVERSE_BENCHMARKS.get(universe_name, MARKET_FILTER_SYMBOL)
    extra_symbols = [MARKET_FILTER_SYMBOL, BROAD_BENCHMARK_SYMBOL, benchmark_symbol]
    symbols_to_download = sorted(set(universe_symbols + extra_symbols))

    all_bars = get_daily_bars_for_symbols(data_client, symbols_to_download)

    prepared_by_symbol: dict[str, pd.DataFrame] = {}

    for symbol in symbols_to_download:
        symbol_df = all_bars[all_bars["symbol"] == symbol].copy()
        if not symbol_df.empty:
            prepared_by_symbol[symbol] = prepare_indicators(symbol_df)

    qqq_df = prepared_by_symbol.get(MARKET_FILTER_SYMBOL)
    if qqq_df is None or len(qqq_df) < 50:
        raise RuntimeError("Not enough QQQ data.")

    qqq_latest = qqq_df.iloc[-1]
    qqq_close = float(qqq_latest["close"])
    qqq_ma20 = float(qqq_latest["MA20"])
    qqq_above_ma20 = qqq_close > qqq_ma20
    qqq_ret20 = _safe_float(qqq_latest.get("RET20"), None)

    benchmark_df = prepared_by_symbol.get(benchmark_symbol, qqq_df)
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
        "qqq_ret20": round(qqq_ret20 * 100, 2) if qqq_ret20 is not None else None,
        "qqq_latest_date": str(qqq_latest["timestamp"]),
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

        symbol_df = prepared_by_symbol.get(symbol)

        if symbol_df is None or symbol_df.empty:
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
            qqq_ret20=qqq_ret20,
            universe_benchmark_above_ma20=universe_benchmark_above_ma20,
        )

        results.append(result)

    return pd.DataFrame(results), market_info, prepared_by_symbol


SIGNALS_TO_TRACK = ["HIGH CONVICTION BUY", "BUY SIGNAL", "ALMOST BUY"]

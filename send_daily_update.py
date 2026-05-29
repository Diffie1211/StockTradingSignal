import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient

from subscription_utils import (
    get_active_subscriptions,
    selected_universes_from_subscription,
    format_manage_link,
    get_secret,
)
from strategy_engine import (
    scan_universe,
    get_daily_bars_for_symbols,
    prepare_indicators,
    assess_sell_signal,
    SIGNALS_TO_TRACK,
    MARKET_FILTER_SYMBOL,
    BROAD_BENCHMARK_SYMBOL,
)
from tracking_utils import (
    record_new_signals,
    get_open_signals,
    update_signal,
    get_signal_history,
    rows_to_dataframe,
    summarize_performance,
)


load_dotenv()

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
# RETURN CALCULATION
# ============================================================

def parse_date(value):
    return pd.to_datetime(value).date()


def trading_rows_after_signal(df: pd.DataFrame, signal_date) -> pd.DataFrame:
    if df.empty:
        return df

    working = df.copy()
    working["date_only"] = pd.to_datetime(working["timestamp"]).dt.date
    return working[working["date_only"] > signal_date].copy().reset_index(drop=True)


def close_on_or_before(df: pd.DataFrame, target_date):
    if df.empty:
        return None

    working = df.copy()
    working["date_only"] = pd.to_datetime(working["timestamp"]).dt.date
    before = working[working["date_only"] <= target_date].copy()

    if before.empty:
        return None

    return float(before.iloc[-1]["close"])


def nday_return_from_signal(df: pd.DataFrame, signal_date, signal_close: float, n: int):
    after = trading_rows_after_signal(df, signal_date)
    if len(after) < n:
        return None

    close_n = float(after.iloc[n - 1]["close"])
    return (close_n / signal_close - 1) * 100


def current_return_from_signal(df: pd.DataFrame, signal_close: float):
    if df.empty:
        return None, None
    latest_close = float(df.iloc[-1]["close"])
    return latest_close, (latest_close / signal_close - 1) * 100


def max_gain_drawdown(df: pd.DataFrame, signal_date, signal_close: float):
    after = trading_rows_after_signal(df, signal_date)
    if after.empty:
        return None, None

    closes = after["close"].astype(float)
    max_gain = (closes.max() / signal_close - 1) * 100
    max_drawdown = (closes.min() / signal_close - 1) * 100
    return max_gain, max_drawdown


def benchmark_returns(benchmark_df: pd.DataFrame, signal_date, horizons: list[int]):
    base_close = close_on_or_before(benchmark_df, signal_date)

    result = {}
    if base_close is None:
        for n in horizons:
            result[n] = None
        return result

    for n in horizons:
        result[n] = nday_return_from_signal(benchmark_df, signal_date, base_close, n)

    return result


def update_open_signal_performance() -> list[dict]:
    open_signals = get_open_signals()

    if not open_signals:
        print("No open signals to update.")
        return []

    symbols = sorted(set([s["ticker"] for s in open_signals] + [MARKET_FILTER_SYMBOL, BROAD_BENCHMARK_SYMBOL]))
    all_bars = get_daily_bars_for_symbols(data_client, symbols)

    prepared_by_symbol = {}
    for symbol in symbols:
        symbol_df = all_bars[all_bars["symbol"] == symbol].copy()
        if not symbol_df.empty:
            prepared_by_symbol[symbol] = prepare_indicators(symbol_df)

    qqq_df = prepared_by_symbol.get(MARKET_FILTER_SYMBOL, pd.DataFrame())
    spy_df = prepared_by_symbol.get(BROAD_BENCHMARK_SYMBOL, pd.DataFrame())

    sell_notices = []
    horizons = [1, 3, 5, 10, 20, 60]

    for signal in open_signals:
        signal_id = signal["id"]
        ticker = signal["ticker"]
        signal_date = parse_date(signal["signal_date"])
        signal_close = float(signal["signal_close"])
        symbol_df = prepared_by_symbol.get(ticker, pd.DataFrame())

        if symbol_df.empty:
            continue

        latest_close, current_return = current_return_from_signal(symbol_df, signal_close)
        max_gain, max_drawdown = max_gain_drawdown(symbol_df, signal_date, signal_close)

        returns = {}
        for n in horizons:
            returns[f"return_{n}d"] = nday_return_from_signal(symbol_df, signal_date, signal_close, n)

        qqq_returns = benchmark_returns(qqq_df, signal_date, horizons)
        spy_returns = benchmark_returns(spy_df, signal_date, horizons)

        payload = {
            "current_close": latest_close,
            "current_return_pct": current_return,
            "max_gain_since_signal": max_gain,
            "max_drawdown_since_signal": max_drawdown,
        }

        for n in horizons:
            payload[f"return_{n}d"] = returns[f"return_{n}d"]
            payload[f"qqq_return_{n}d"] = qqq_returns[n]
            payload[f"spy_return_{n}d"] = spy_returns[n]

            if returns[f"return_{n}d"] is not None and qqq_returns[n] is not None:
                payload[f"excess_vs_qqq_{n}d"] = returns[f"return_{n}d"] - qqq_returns[n]
                payload[f"beat_qqq_{n}d"] = returns[f"return_{n}d"] > qqq_returns[n]

            if returns[f"return_{n}d"] is not None and spy_returns[n] is not None:
                payload[f"excess_vs_spy_{n}d"] = returns[f"return_{n}d"] - spy_returns[n]
                payload[f"beat_spy_{n}d"] = returns[f"return_{n}d"] > spy_returns[n]

        sell = assess_sell_signal(symbol_df, signal_close)
        current_sell_type = sell.get("sell_signal_type", "UNKNOWN")

        payload.update({
            "current_sell_signal_type": current_sell_type,
            "latest_sell_check_date": parse_date(sell.get("latest_date")).isoformat() if sell.get("latest_date") else None,
            "exit_reason": sell.get("exit_reason"),
        })

        previous_notice_type = signal.get("last_notice_sell_signal_type")
        should_notice = False

        if current_sell_type in ["SELL SIGNAL", "ALMOST SELL", "CAUTION HOLD"]:
            if current_sell_type != previous_notice_type:
                should_notice = True
                payload["last_notice_sell_signal_type"] = current_sell_type
                payload["last_notice_sent_at"] = datetime.now(timezone.utc).isoformat()

        if current_sell_type == "CAUTION HOLD" and not signal.get("first_caution_hold_date"):
            payload["first_caution_hold_date"] = parse_date(sell.get("latest_date")).isoformat()

        if current_sell_type == "ALMOST SELL" and not signal.get("first_almost_sell_date"):
            payload["first_almost_sell_date"] = parse_date(sell.get("latest_date")).isoformat()

        if current_sell_type == "SELL SIGNAL":
            sell_date = parse_date(sell.get("latest_date"))
            sell_price = float(sell.get("sell_signal_price"))
            return_to_sell = (sell_price / signal_close - 1) * 100
            days_to_sell = len(trading_rows_after_signal(symbol_df, signal_date))

            payload.update({
                "sell_signal_type": current_sell_type,
                "sell_signal_date": sell_date.isoformat(),
                "sell_signal_price": sell_price,
                "return_to_sell_signal_pct": return_to_sell,
                "days_to_sell_signal": days_to_sell,
                "status": sell.get("status", "CLOSED_BY_SELL_SIGNAL"),
            })
        else:
            payload["status"] = sell.get("status", "OPEN")

            # Expire after 60 trading days if no sell signal appeared.
            after = trading_rows_after_signal(symbol_df, signal_date)
            if len(after) >= 60 and payload["status"] in ["OPEN", "OPEN_WITH_CAUTION", "OPEN_WITH_ALMOST_SELL"]:
                payload["status"] = "EXPIRED_60D"

        updated = update_signal(signal_id, payload)

        if should_notice and updated:
            sell_notices.append(updated)

    return sell_notices


# ============================================================
# EMAIL HTML
# ============================================================

def build_signal_table_html(title: str, rows: list[dict]) -> str:
    if not rows:
        return f"<h3>{title}</h3><p>None today.</p>"

    body = ""
    for row in rows:
        body += f"""
        <tr>
            <td><strong>{row.get('ticker') or row.get('symbol')}</strong></td>
            <td>{row.get('company_name', '')}</td>
            <td>{row.get('signal_type') or row.get('final_signal', '')}</td>
            <td>{row.get('universe', '')}</td>
            <td>{row.get('signal_close') or row.get('latest_close', '')}</td>
            <td>{row.get('buy_score', '')}/{row.get('buy_total', '')}</td>
            <td>{row.get('failed_checks', '')}</td>
        </tr>
        """

    return f"""
    <h3>{title}</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
      <thead>
        <tr>
          <th>Ticker</th><th>Company</th><th>Signal</th><th>Universe</th>
          <th>Signal Close</th><th>Score</th><th>Missing Checks</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
    """


def build_sell_notice_html(rows: list[dict]) -> str:
    if not rows:
        return "<h3>Sell / Caution Notices</h3><p>No new sell notices today.</p>"

    body = ""
    for row in rows:
        body += f"""
        <tr>
            <td><strong>{row.get('ticker')}</strong></td>
            <td>{row.get('company_name', '')}</td>
            <td>{row.get('current_sell_signal_type', '')}</td>
            <td>{row.get('signal_date', '')}</td>
            <td>{row.get('signal_close', '')}</td>
            <td>{row.get('current_close', '')}</td>
            <td>{round(float(row.get('current_return_pct') or 0), 2)}%</td>
            <td>{row.get('exit_reason', '')}</td>
        </tr>
        """

    return f"""
    <h3>Sell / Caution Notices</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
      <thead>
        <tr>
          <th>Ticker</th><th>Company</th><th>Notice</th><th>Buy Signal Date</th>
          <th>Signal Close</th><th>Current Close</th><th>Return</th><th>Reason</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
    """


def build_performance_summary_html() -> str:
    rows = get_signal_history(limit=2000)
    df = rows_to_dataframe(rows)
    summary_df = summarize_performance(df)

    if summary_df.empty:
        return "<h3>Strategy Performance Snapshot</h3><p>Not enough tracked signals yet.</p>"

    body = ""
    for _, row in summary_df.iterrows():
        body += f"""
        <tr>
            <td>{row.get('signal_type', '')}</td>
            <td>{int(row.get('total_signals') or 0)}</td>
            <td>{round(float(row.get('avg_current_return_pct') or 0), 2)}%</td>
            <td>{round(float(row.get('avg_5d_return_pct') or 0), 2)}%</td>
            <td>{round(float(row.get('win_rate_5d_pct') or 0), 2)}%</td>
            <td>{round(float(row.get('beat_qqq_5d_pct') or 0), 2)}%</td>
        </tr>
        """

    return f"""
    <h3>Strategy Performance Snapshot</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
      <thead>
        <tr>
          <th>Signal Type</th><th>Total</th><th>Avg Current Return</th>
          <th>Avg 5D Return</th><th>5D Win Rate</th><th>5D Beat QQQ Rate</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
    """


def build_email_html(subscription: dict, new_signals_by_universe: dict[str, list[dict]], sell_notices: list[dict]) -> str:
    selected_universes = selected_universes_from_subscription(subscription)
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    manage_link = format_manage_link(subscription["token"])

    high_rows = []
    buy_rows = []
    almost_rows = []

    for universe_name in selected_universes:
        for row in new_signals_by_universe.get(universe_name, []):
            if row.get("signal_type") == "HIGH CONVICTION BUY":
                high_rows.append(row)
            elif row.get("signal_type") == "BUY SIGNAL":
                buy_rows.append(row)
            elif row.get("signal_type") == "ALMOST BUY":
                almost_rows.append(row)

    filtered_sell_notices = [
        row for row in sell_notices
        if row.get("universe") in selected_universes
    ]

    return f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.5;">
        <h2>Diffie's Daily Stock Signal Update — {today}</h2>

        <p>
            This email lists new BUY / ALMOST BUY signals and sell/caution notices
            based on Diffie's daily KDJ + MACD + MA20 + QQQ + 7-day Box Theory strategy.
        </p>

        <p>
            This is for manual decision support only. It does not place trades and does not guarantee profit.
        </p>

        {build_signal_table_html('High Conviction BUY', high_rows)}
        {build_signal_table_html('BUY SIGNAL', buy_rows)}
        {build_signal_table_html('ALMOST BUY', almost_rows)}
        {build_sell_notice_html(filtered_sell_notices)}
        {build_performance_summary_html()}

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
        # Still update open signals, so tracking keeps working.
        update_open_signal_performance()
        return

    print(f"Active subscribers: {len(subscribers)}")

    needed_universes = set()
    for subscriber in subscribers:
        for universe_name in selected_universes_from_subscription(subscriber):
            needed_universes.add(universe_name)

    new_signals_by_universe = {}

    for universe_name in sorted(needed_universes):
        print(f"Scanning {universe_name}...")
        results_df, market_info, _prepared = scan_universe(data_client, universe_name)
        saved_rows = record_new_signals(results_df, universe_name)
        new_signals_by_universe[universe_name] = saved_rows
        print(f"Finished {universe_name}: {len(results_df)} rows, saved {len(saved_rows)} new tracked signals")

    print("Updating open signal performance and sell notices...")
    sell_notices = update_open_signal_performance()
    print(f"Sell/caution notices: {len(sell_notices)}")

    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    for subscriber in subscribers:
        email = subscriber["email"]
        html = build_email_html(subscriber, new_signals_by_universe, sell_notices)
        subject = f"Diffie's Daily Stock Signals — {today}"

        print(f"Sending email to {email}...")
        send_email(email, subject, html)

    print("Daily update complete.")


if __name__ == "__main__":
    main()

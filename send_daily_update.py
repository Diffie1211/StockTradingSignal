import os
from datetime import datetime
from zoneinfo import ZoneInfo

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
    SIGNALS_TO_TRACK,
)


load_dotenv()

ALPACA_API_KEY = get_secret("ALPACA_API_KEY")
ALPACA_SECRET_KEY = get_secret("ALPACA_SECRET_KEY")
RESEND_API_KEY = get_secret("RESEND_API_KEY")
RESEND_FROM_EMAIL = get_secret(
    "RESEND_FROM_EMAIL",
    "Diffie's Stock Scanner <onboarding@resend.dev>",
)
APP_PUBLIC_URL = get_secret("APP_PUBLIC_URL", "https://diffie1211-stocktradingsignal.streamlit.app").rstrip("/")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise RuntimeError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY.")

if not RESEND_API_KEY:
    raise RuntimeError("Missing RESEND_API_KEY.")


data_client = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
)


def html_escape(value) -> str:
    value = "" if value is None else str(value)
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_signal_table_html(title: str, rows: list[dict]) -> str:
    if not rows:
        return f"<h3>{title}</h3><p>None today.</p>"

    body = ""

    for row in rows:
        score = f"{row.get('core_pillar_score', row.get('buy_score', ''))}/{row.get('core_pillar_total', row.get('buy_total', ''))}"

        body += f"""
        <tr>
            <td><strong>{html_escape(row.get('symbol', ''))}</strong></td>
            <td>{html_escape(row.get('company_name', ''))}</td>
            <td>{html_escape(row.get('universe', ''))}</td>
            <td>{html_escape(row.get('final_signal', ''))}</td>
            <td>{html_escape(score)}</td>
            <td>{html_escape(row.get('latest_close', ''))}</td>
            <td>{html_escape(row.get('failed_checks', ''))}</td>
        </tr>
        """

    return f"""
    <h3>{title}</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%; font-size: 13px;">
        <thead>
            <tr>
                <th>Symbol</th>
                <th>Company</th>
                <th>List</th>
                <th>Signal</th>
                <th>Core Score</th>
                <th>Close</th>
                <th>Missing Core Pillars</th>
            </tr>
        </thead>
        <tbody>{body}</tbody>
    </table>
    """


def build_copyable_ticker_html(title: str, rows: list[dict]) -> str:
    tickers = []
    for row in rows:
        ticker = row.get("symbol")
        if ticker and ticker not in tickers:
            tickers.append(ticker)

    ticker_text = ", ".join(tickers) if tickers else "None"
    return f"""
    <h3>{title}</h3>
    <div style="font-family: monospace; background: #f6f6f6; border: 1px solid #ddd; border-radius: 8px; padding: 10px;">
        {html_escape(ticker_text)}
    </div>
    """


def build_market_summary_html(market_info_by_universe: dict[str, dict]) -> str:
    if not market_info_by_universe:
        return ""

    rows = ""

    for universe_name, info in market_info_by_universe.items():
        rows += f"""
        <tr>
            <td>{html_escape(universe_name)}</td>
            <td>{html_escape(info.get('benchmark_symbol', ''))}</td>
            <td>{html_escape(info.get('benchmark_close', ''))}</td>
            <td>{html_escape(info.get('benchmark_ma20', ''))}</td>
            <td>{html_escape(info.get('universe_benchmark_above_ma20', ''))}</td>
            <td>{html_escape(info.get('qqq_above_ma20', ''))}</td>
        </tr>
        """

    return f"""
    <h3>Market Condition Summary</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%; font-size: 13px;">
        <thead>
            <tr>
                <th>List</th>
                <th>Benchmark</th>
                <th>Benchmark Close</th>
                <th>Benchmark MA20</th>
                <th>Benchmark > MA20</th>
                <th>QQQ > MA20</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


def build_email_html(
    subscription: dict,
    signals_by_universe: dict[str, list[dict]],
    market_info_by_universe: dict[str, dict],
) -> str:
    selected_universes = selected_universes_from_subscription(subscription)
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    manage_link = format_manage_link(subscription["token"])

    high_rows = []
    buy_rows = []
    almost_rows = []

    for universe_name in selected_universes:
        for row in signals_by_universe.get(universe_name, []):
            row = dict(row)
            row["universe"] = universe_name

            if row.get("final_signal") == "HIGH CONVICTION BUY":
                high_rows.append(row)
            elif row.get("final_signal") == "BUY SIGNAL":
                buy_rows.append(row)
            elif row.get("final_signal") == "ALMOST BUY":
                almost_rows.append(row)

    all_rows = high_rows + buy_rows + almost_rows

    return f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.5; color: #222;">
        <h2>Diffie's Daily Stock Signal Update — {today}</h2>

        <p>
            Today's update includes <strong>HIGH CONVICTION BUY</strong>, <strong>BUY SIGNAL</strong>,
            and <strong>ALMOST BUY</strong> names from your selected lists.
        </p>

        <p>
            The strategy uses independent pillars: market environment, trend, momentum, structure, and risk/extension.
            Volume and relative strength are confirmation tags, not duplicate score points.
        </p>

        {build_market_summary_html(market_info_by_universe)}

        {build_copyable_ticker_html('Copyable ticker list', all_rows)}

        {build_signal_table_html('High Conviction BUY', high_rows)}
        {build_signal_table_html('BUY SIGNAL', buy_rows)}
        {build_signal_table_html('ALMOST BUY', almost_rows)}

        <h3>How to track these later</h3>
        <p>
            Save the ticker list and today's signal date. In the next few days, open the scanner and paste the
            list into the <strong>Signal List Tracker</strong>. It will calculate 1D, 5D, 10D, and 20D returns,
            compare against QQQ/SPY, and check for SELL / ALMOST SELL warnings.
        </p>

        <p>
            <a href="{APP_PUBLIC_URL}" style="background: #111; color: white; padding: 10px 14px; text-decoration: none; border-radius: 6px;">
                Open Diffie's Stock Signal Scanner
            </a>
        </p>

        <hr>

        <p style="font-size: 12px; color: #666;">
            Disclaimer: This email is for education, research, and manual decision support only.
            It is not financial advice, not a trading recommendation, and does not guarantee profit.
            I do not manage anyone's trades or take responsibility for anyone's investment risk.
        </p>

        <p style="font-size: 12px;">
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

    signals_by_universe = {}
    market_info_by_universe = {}

    for universe_name in sorted(needed_universes):
        print(f"Scanning {universe_name}...")
        results_df, market_info, _prepared = scan_universe(data_client, universe_name)
        signal_df = results_df[results_df["final_signal"].isin(SIGNALS_TO_TRACK)].copy()

        signals_by_universe[universe_name] = signal_df.to_dict(orient="records")
        market_info_by_universe[universe_name] = market_info

        print(f"Finished {universe_name}: {len(results_df)} rows, {len(signal_df)} signal rows")

    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    for subscriber in subscribers:
        email = subscriber["email"]
        html = build_email_html(subscriber, signals_by_universe, market_info_by_universe)
        subject = f"Diffie's Daily Stock Signals — {today}"

        print(f"Sending email to {email}...")
        send_email(email, subject, html)

    print("Daily update complete.")


if __name__ == "__main__":
    main()

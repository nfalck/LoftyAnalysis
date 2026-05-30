from urllib.parse import urlparse, quote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException
import time
import random
import pandas as pd
import json
import re
import traceback
from datetime import datetime

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--start-maximized")
chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

driver = webdriver.Chrome(options=chrome_options)
driver.execute_cdp_cmd("Network.enable", {})
wait = WebDriverWait(driver, 20)

# Storage, if something happens or need to cancel script, save progress
overview_rows = []
transaction_rows = []
holder_rows = []
error_rows = []


# Helper scripts
def human_pause(min_s=0.6, max_s=1.8, jitter=0.25):
    """
    Random sleep between min_s and max_s plus a little jitter
    """
    base = random.uniform(min_s, max_s)
    time.sleep(max(0.0, base + random.uniform(-jitter, jitter)))


def page_settle(short=True):
    """
    Pause after navigation/clicks so the network settles
    """
    if short:
        human_pause(1.0, 2.5)
    else:
        human_pause(2.5, 5.5)


def occasional_long_break(i, every=8):
    """
    Take breaks
    """
    if i % every == 0:
        print(f"Taking a longer break after {i} assets")
        human_pause(12, 25, jitter=2)


def light_human_actions():
    """
    Scroll a bit, move mouse, etc
    """
    try:
        # small scroll down/up
        driver.execute_script("window.scrollBy(0, arguments[0]);", random.randint(200, 800))
        human_pause(0.3, 0.9)
        driver.execute_script("window.scrollBy(0, arguments[0]);", -random.randint(150, 500))
        human_pause(0.3, 0.9)
    except Exception:
        pass


def log_asset_error(property_id, asa_id, stage, exc, extra=None):
    err = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "property_id": property_id,
        "asa_id": asa_id,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }
    if extra:
        err.update(extra)
    error_rows.append(err)
    print(f"\nERROR on asa_id={asa_id} (property_id={property_id}) stage='{stage}': {type(exc).__name__}: {exc}\n")


def checkpoint_save(prefix="v4"):
    """Write progress time to time to save work"""
    try:
        if overview_rows:
            pd.DataFrame(overview_rows).to_csv(f"overview_{prefix}_checkpoint.csv", index=False)
            pd.DataFrame(overview_rows).to_parquet(f"overview_{prefix}_checkpoint.parquet", index=False)

        if transaction_rows:
            pd.DataFrame(transaction_rows).to_csv(f"transactions_{prefix}_checkpoint.csv", index=False)
            pd.DataFrame(transaction_rows).to_parquet(f"transactions_{prefix}_checkpoint.parquet", index=False)

        if holder_rows:
            pd.DataFrame(holder_rows).to_csv(f"holders_{prefix}_checkpoint.csv", index=False)
            pd.DataFrame(holder_rows).to_parquet(f"holders_{prefix}_checkpoint.parquet", index=False)

        if error_rows:
            pd.DataFrame(error_rows).to_csv(f"errors_{prefix}_checkpoint.csv", index=False)
    except Exception as e:
        print(f"Checkpoint save failed: {type(e).__name__}: {e}")


def clear_performance_logs():
    try:
        driver.get_log("performance")
    except Exception:
        pass


def asset_id_from_url(url: str) -> str | None:
    try:
        p = urlparse(url)
        parts = [x for x in p.path.split("/") if x]
        if "asset" in parts:
            idx = parts.index("asset")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return parts[-1] if parts else None
    except Exception:
        return None


def parse_circ(text: str) -> int | None:
    if not text:
        return None
    s = text.strip()
    for ch in [" ", "\u00a0", "\u202f", ","]:
        s = s.replace(ch, "")
    if s.isdigit():
        return int(s)

    suffix_multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    suffix = s[-1].upper()
    if suffix in suffix_multipliers:
        try:
            return int(float(s[:-1]) * suffix_multipliers[suffix])
        except ValueError:
            return None
    return None


# Network JSON capture
def _get_json_from_network(substring: str, timeout=20):
    deadline = time.time() + timeout
    target_ids = set()

    while time.time() < deadline:
        try:
            logs = driver.get_log("performance")
        except Exception:
            logs = []

        for entry in logs:
            try:
                msg = json.loads(entry["message"])
            except Exception:
                continue

            message = msg.get("message", {})
            method = message.get("method")
            params = message.get("params", {})

            if method == "Network.responseReceived":
                resp = params.get("response", {})
                url = resp.get("url", "")
                if substring in url:
                    target_ids.add(params.get("requestId"))

            if method == "Network.loadingFinished":
                req_id = params.get("requestId")
                if req_id in target_ids:
                    try:
                        body = driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": req_id})
                        return json.loads(body.get("body", "") or "{}")
                    except WebDriverException:
                        continue
                    except json.JSONDecodeError:
                        continue

        human_pause(0.15, 0.35, jitter=0.05)

    return None


def get_transactions_json(timeout=15):
    return _get_json_from_network("getTransactions", timeout=timeout)


def get_holders_json(timeout=20):
    return _get_json_from_network("getAssetHolders", timeout=timeout)


# Scrapers
def scrape_overview(property_id, asa_id, asset_id):
    w = WebDriverWait(driver, 20)

    created_date_elem = w.until(
        EC.presence_of_element_located((
            By.XPATH,
            "//dt[normalize-space()='Created']/following-sibling::dd"
            "//span[contains(@class,'nowrap')]"
        ))
    )
    created_date = created_date_elem.text.strip()

    all_time_elem = w.until(
        EC.visibility_of_element_located((
            By.XPATH,
            "//dt[normalize-space()='All time txns']/following-sibling::dd[1]"
            "//*[normalize-space()][1]"
        ))
    )
    all_time_txns = int(all_time_elem.text.strip().replace(",", ""))

    w.until(EC.presence_of_all_elements_located((By.TAG_NAME, "dt")))
    dt_elements = driver.find_elements(By.TAG_NAME, "dt")

    circ_dd = None
    for dt in dt_elements:
        if "CIRCULATING" in dt.text.upper():
            circ_dd = dt.find_element(By.XPATH, "./following-sibling::dd[1]")
            break

    circulating = None
    if circ_dd:
        try:
            circ_span = circ_dd.find_element(By.CSS_SELECTOR, "span.ints")
            circ_text = circ_span.text.strip()
        except Exception:
            full_text = circ_dd.text.strip()
            match = re.match(r"([\d.,]+[KMB]?)", full_text, re.IGNORECASE)
            circ_text = match.group(1) if match else full_text
        circulating = parse_circ(circ_text)

    total_holders_elem = w.until(
        EC.visibility_of_element_located((
            By.XPATH,
            "//dt[normalize-space()='Total holders']/following-sibling::dd[1]"
            "//*[contains(@class,'ints')][1]"
        ))
    )
    total_holders = int(total_holders_elem.text.strip().replace(",", ""))

    overview_rows.append({
        "property_id": property_id,
        "asa_id": asa_id,
        "asset_id": asset_id,
        "created_date": created_date,
        "all_time_txns": all_time_txns,
        "circulating_supply": circulating,
        "total_holders": total_holders,
    })

    # short settle
    page_settle(short=True)
    light_human_actions()


def scrape_transactions(property_id, asa_id, asset_id):
    w = WebDriverWait(driver, 20)

    clear_performance_logs()
    tx_tab = w.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, '/txns')]")))
    driver.execute_script("arguments[0].click();", tx_tab)
    page_settle(short=True)

    per_page_select = w.until(EC.element_to_be_clickable((By.ID, "perPage")))
    Select(per_page_select).select_by_value("128")
    page_settle(short=True)

    base_tx_url = f"https://allo.info/asset/{asset_id}/txns"
    page_num = 1
    after_cursor = None

    while True:
        if page_num > 1:
            clear_performance_logs()
            url = f"{base_tx_url}?page={page_num}&after={quote(after_cursor, safe='')}"
            driver.get(url)
            page_settle(short=False)

        tx_json = get_transactions_json(timeout=20)
        if not tx_json:
            print(f"No getTransactions JSON found on page {page_num} for asset_id={asset_id}")
            break

        data = tx_json.get("data", {})
        entries = data.get("entries", [])
        page_info = data.get("page", {})

        for tx in entries:
            amount_raw = tx.get("amount")
            close_amount_raw = tx.get("closeAmount")

            try:
                amount = int(amount_raw) if amount_raw is not None else 0
            except Exception:
                amount = 0

            try:
                close_amount = int(close_amount_raw) if close_amount_raw is not None else 0
            except Exception:
                close_amount = 0

            real_transfer_amount = max(amount, close_amount)

            if amount == 0 and close_amount == 0:
                if tx.get("txType") == "axfer":
                    if tx.get("sender") == tx.get("receiver"):
                        semantic_type = "opt-in"
                    else:
                        semantic_type = "freeze/unfreeze/metadata"
                elif tx.get("txType") == "appl":
                    semantic_type = "application-call"
                else:
                    semantic_type = "zero-value"
            elif close_amount > 0:
                semantic_type = "close-out"
            else:
                semantic_type = "transfer"

            transaction_rows.append({
                "property_id": property_id,
                "asa_id": asa_id,
                "asset_id": tx.get("assetId"),
                "tx_id": tx.get("id"),
                "block": int(tx.get("confirmedRound")) if tx.get("confirmedRound") is not None else None,
                "round_time": tx.get("roundTime"),
                "sender": tx.get("sender"),
                "receiver": tx.get("receiver"),
                "amount": amount,
                "close_amount": close_amount,
                "real_transfer_amount": real_transfer_amount,
                "tx_type": tx.get("txType"),
                "semantic_type": semantic_type,
                "is_inner": tx.get("isInner"),
            })

        # gentle pause per page
        human_pause(1.0, 2.5)
        light_human_actions()

        has_next = page_info.get("hasNext")
        last_cursor = page_info.get("last")
        if not has_next or not last_cursor:
            break

        after_cursor = last_cursor
        page_num += 1


def scrape_holders(property_id, asa_id, asset_id):
    w = WebDriverWait(driver, 20)

    holders_tab = w.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, '/holders')]")))
    clear_performance_logs()
    driver.execute_script("arguments[0].click();", holders_tab)
    page_settle(short=True)

    base_holders_url = f"https://allo.info/asset/{asset_id}/holders"
    page_num = 1
    after_cursor = None

    while True:
        if page_num > 1:
            clear_performance_logs()
            url = f"{base_holders_url}?page={page_num}&after={quote(after_cursor, safe='')}"
            driver.get(url)
            page_settle(short=False)

        holders_json = get_holders_json(timeout=25)
        if not holders_json:
            print(f"No holders JSON found on page {page_num} for asset_id={asset_id}")
            break

        data = holders_json.get("data", {})
        entries = data.get("entries", [])
        page_info = data.get("page", {})

        for h in entries:
            addr = h.get("address")
            balance_raw = h.get("balance")
            try:
                balance = int(balance_raw) if balance_raw is not None else 0
            except Exception:
                balance = 0

            holder_rows.append({
                "property_id": property_id,
                "asa_id": asa_id,
                "asset_id": asset_id,
                "holder_address": addr,
                "tokens_held": balance,
            })

        # gentle pause per page
        human_pause(1.0, 2.8)
        light_human_actions()

        has_next = page_info.get("hasNext")
        last_cursor = page_info.get("last")
        if not has_next or not last_cursor:
            break

        after_cursor = last_cursor
        page_num += 1


def scrape_asset_direct(property_id, asa_id, algorand_address):
    driver.get(algorand_address)
    page_settle(short=False)
    light_human_actions()

    current_url = driver.current_url
    asset_id = asset_id_from_url(current_url)
    if not asset_id:
        raise RuntimeError(f"Could not parse asset_id from URL: {current_url}")

    # Overview
    scrape_overview(property_id, asa_id, asset_id)

    # Transactions
    scrape_transactions(property_id, asa_id, asset_id)

    # Holders
    scrape_holders(property_id, asa_id, asset_id)


CSV_FILE = "properties_v4_with_algorand.csv"

df = pd.read_csv(CSV_FILE)
assets = (
    df[["property_id", "asa_id", "algorand_address"]]
    .dropna(subset=["algorand_address"])
    .to_dict("records")
)

for i, asset in enumerate(assets, start=1):
    property_id = asset["property_id"]
    asa_id = str(asset["asa_id"]).strip() if pd.notna(asset["asa_id"]) else None
    algorand_address = str(asset["algorand_address"]).strip()

    try:
        scrape_asset_direct(property_id, asa_id, algorand_address)

    except Exception as e:
        log_asset_error(property_id, asa_id, stage="scrape_asset_direct", exc=e, extra={"url": algorand_address})

    finally:
        # Save what we have so far no matter what
        checkpoint_save(prefix="v4")

        # Short pause between assets
        human_pause(3.0, 7.5, jitter=1.0)
        occasional_long_break(i, every=8)

# Save
pd.DataFrame(transaction_rows).to_csv("transactions_v7.csv", index=False)
pd.DataFrame(transaction_rows).to_parquet("transactions.parquet", index=False)

pd.DataFrame(overview_rows).to_csv("overview_v7.csv", index=False)
pd.DataFrame(overview_rows).to_parquet("overview.parquet", index=False)

pd.DataFrame(holder_rows).to_csv("holders_v7.csv", index=False)
pd.DataFrame(holder_rows).to_parquet("holders.parquet", index=False)

if error_rows:
    pd.DataFrame(error_rows).to_csv("errors_v7.csv", index=False)

driver.quit()

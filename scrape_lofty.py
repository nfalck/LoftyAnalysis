from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import pandas as pd
from datetime import datetime
import json
import random
import os
import traceback

# Configuration
LINKS_CSV = "lofty_links.csv"

OUT_PROPERTIES_CSV = "properties_v7.csv"
OUT_PRICES_CSV = "prices_v7.csv"
OUT_YIELDS_CSV = "yields_v7.csv"

PAGE_SLEEP_RANGE = (2.0, 4.0)
COOLDOWN_EVERY_N = 6
COOLDOWN_RANGE = (10, 18)
CAPTURE_TIMEOUT = 20

BAD_LISTINGS = {
    "Earl-DAO_--",
    "Universal-Lending-DAO-(ULD)_Sheridan-Wyoming-82801"
}

# Helper scripts
def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    for sep in [",", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, dtype=str)
            if len(df.columns) >= 1:
                return df
        except Exception:
            pass

    return pd.read_csv(path, engine="python", dtype=str)

def pick_link_column(df: pd.DataFrame) -> str:
    candidates = [
        "listing_url", "url", "href", "link", "property_url",
        "Listing URL", "URL", "Link"
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # fallback: first column containing lofty property_deal links
    for c in df.columns:
        vals = df[c].dropna().astype(str)
        if vals.str.contains("/property_deal/", regex=False).any():
            return c

    raise ValueError(
        f"Could not find a link column in lofty_links.csv. Columns: {list(df.columns)}"
    )

def clean_links(df: pd.DataFrame, link_col: str):
    urls = []
    seen = set()
    for raw in df[link_col].dropna().astype(str):
        href = raw.strip()
        if not href:
            continue
        if "/property_deal/" not in href:
            continue
        if any(bad in href for bad in BAD_LISTINGS):
            print("[SKIP bad slug]", href)
            continue
        if href not in seen:
            seen.add(href)
            urls.append(href)
    return urls

def nested_get(dct, *keys, default=None):
    cur = dct
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def get_any(dct, key_candidates, default=None):
    """
    Search for the first matching key in a nested dict/list structure
    """
    def _search(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in key_candidates:
                    return v
                found = _search(v)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _search(item)
                if found is not None:
                    return found
        return None

    found = _search(dct)
    return default if found is None else found

def try_find_element_text(driver, by, selector, timeout=10):
    try:
        elem = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, selector))
        )
        return elem.text
    except Exception:
        return None

def try_click(driver, by, selector, timeout=6):
    try:
        elem = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, selector))
        )
        driver.execute_script("arguments[0].click();", elem)
        return True
    except Exception:
        return False

def capture_network_json(driver, timeout=20):
    """
    Capture
      - get-list-view-data-by-id
      - get-property-yields
    from Chrome performance logs
    """
    deadline = time.time() + timeout

    price_req_ids = set()
    yield_req_ids = set()
    price_json = None
    yield_json = None

    while time.time() < deadline and (price_json is None or yield_json is None):
        try:
            logs = driver.get_log("performance")
        except Exception:
            logs = []

        for entry in logs:
            try:
                msg = json.loads(entry["message"])
                message = msg.get("message", {})
                method = message.get("method")
                params = message.get("params", {})
            except Exception:
                continue

            if method == "Network.responseReceived":
                resp = params.get("response", {})
                url = resp.get("url", "")

                if "get-list-view-data-by-id" in url:
                    price_req_ids.add(params.get("requestId"))

                if "get-property-yields" in url:
                    yield_req_ids.add(params.get("requestId"))

            if method == "Network.loadingFinished":
                req_id = params.get("requestId")

                if req_id in price_req_ids and price_json is None:
                    try:
                        body = driver.execute_cdp_cmd(
                            "Network.getResponseBody",
                            {"requestId": req_id}
                        )
                        price_json = json.loads(body["body"])
                    except Exception:
                        pass

                if req_id in yield_req_ids and yield_json is None:
                    try:
                        body = driver.execute_cdp_cmd(
                            "Network.getResponseBody",
                            {"requestId": req_id}
                        )
                        yield_json = json.loads(body["body"])
                    except Exception:
                        pass

        time.sleep(0.2)

    return price_json, yield_json

def parse_address_parts(h4_text):
    city = state = zip_code = None
    if not h4_text:
        return city, state, zip_code
    try:
        city, rest = h4_text.split(", ", 1)
        state, zip_code = rest.split(" ", 1)
    except Exception:
        pass
    return city, state, zip_code

def parse_sqft_from_dom(driver):
    selectors = [
        "//span[contains(text(), 'sqft')]",
        "//*[contains(text(), 'sqft')]"
    ]
    for xp in selectors:
        try:
            elem = driver.find_element(By.XPATH, xp)
            txt = elem.text.replace("sqft", "").strip()
            clean = txt.split(" ")[0].replace(",", "")
            return int(float(clean))
        except Exception:
            continue
    return None

def parse_description(driver):
    try:
        ul_elem = driver.find_element(
            By.XPATH,
            "//h2[starts-with(normalize-space(), 'Property Details') "
            "   or starts-with(normalize-space(), 'Details')]"
            "/following-sibling::ul[1]"
        )
        li_elems = ul_elem.find_elements(By.TAG_NAME, "li")
        clean_items = []
        for li in li_elems:
            txt = li.text.strip()
            if "View" in txt:
                txt = txt.split("View")[0].strip()
            if txt and not txt.endswith(":") and not txt.startswith("Why Invest"):
                clean_items.append(txt)
        description_text = " ".join(clean_items).strip()
        if description_text:
            return description_text, len(description_text)
    except Exception:
        pass

    # fallback: try property summary text block
    try:
        blocks = driver.find_elements(By.XPATH, "//p")
        texts = [b.text.strip() for b in blocks if b.text and len(b.text.strip()) > 40]
        if texts:
            description_text = " ".join(texts[:5]).strip()
            return description_text, len(description_text)
    except Exception:
        pass

    return None, None

def safe_int(v):
    try:
        if v is None or v == "":
            return None
        return int(float(str(v).replace(",", "")))
    except Exception:
        return v

links_df = safe_read_csv(LINKS_CSV)
link_col = pick_link_column(links_df)
property_urls = clean_links(links_df, link_col)

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--start-maximized")
chrome_options.add_experimental_option("detach", True)
chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

driver = webdriver.Chrome(options=chrome_options)
driver.execute_cdp_cmd("Network.enable", {
    "maxTotalBufferSize": 10000000,
    "maxResourceBufferSize": 5000000
})

property_rows = []
yield_rows = []
price_rows = []
failed_rows = []

for i, listing_link in enumerate(property_urls, start=1):
    print(f"\n[{i}/{len(property_urls)}] {listing_link}")

    if i % COOLDOWN_EVERY_N == 0:
        print("[INFO] Cooldown...")
        time.sleep(random.uniform(*COOLDOWN_RANGE))

    try:
        time.sleep(random.uniform(*PAGE_SLEEP_RANGE))
        try:
            driver.get_log("performance")  # clear the log
        except Exception:
            pass

        driver.get(listing_link)

        # page may still load even if some elements are missing
        try:
            WebDriverWait(driver, 25).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except TimeoutException:
            print("Body not loaded, continuing anyway")

        data = {}
        prop = {}
        try:
            script_elem = driver.find_element(By.ID, "__DATA")
            raw_json = script_elem.get_attribute("textContent")
            data = json.loads(raw_json)
            prop = data.get("property", {}) or {}
        except Exception:
            print("__DATA not found or invalid JSON")

        # Trigger yield and price charts
        try_click(driver, By.XPATH, "//button[normalize-space()='1D']", timeout=5)
        try_click(
            driver,
            By.XPATH,
            "//button[normalize-space()='1D']/following-sibling::button[normalize-space()='All']",
            timeout=5
        )
        time.sleep(1.5)
        try_click(
            driver,
            By.XPATH,
            "//button[@role='tab' and normalize-space()='Yield']",
            timeout=5
        )
        time.sleep(1.5)

        price_json, yields_json = capture_network_json(driver, timeout=CAPTURE_TIMEOUT)

        # DOM
        address = try_find_element_text(driver, By.TAG_NAME, "h1", timeout=6)
        h4_text = try_find_element_text(driver, By.TAG_NAME, "h4", timeout=6)
        city, state, zip_code = parse_address_parts(h4_text)
        sqft = parse_sqft_from_dom(driver)
        description_text, char_length = parse_description(driver)

        property_id = prop.get("id")
        asa_id = prop.get("assetUnit")
        total_tokens = prop.get("tokens")
        year_built = prop.get("year_built")
        utilities = prop.get("utilities")
        property_taxes = prop.get("taxes")
        llc_admin_fee_yearly = prop.get("llc_admin_fee_yearly")
        llc_admin_fee_upfront = prop.get("llc_admin_fee_upfront")
        management_fees = prop.get("management_fees")
        insurance = prop.get("insurance")
        monthly_rent = prop.get("monthly_rent")
        annual_cash_flow = prop.get("cash_flow")
        cap_rate = prop.get("cap_rate")
        projected_annual_return = prop.get("projected_annual_return")
        appreciation = prop.get("appreciation")
        num_of_images = len(prop.get("images", []) or [])
        original_starting_date = prop.get("original_starting_date")
        sellout_date = prop.get("sellout_date")
        isSellerBuyback = prop.get("isSellerBuyBack")
        projected_annual_cash_flow = prop.get("projected_annual_cash_flow")
        projected_rental_yield = prop.get("projected_rental_yield")

        starting_price = nested_get(prop, "trading", "sell", "min", default=None)

        property_type = None
        operating_reserve = None
        total_investment_value = None
        tokens_available = None
        underlying_asset_price = None

        if isinstance(price_json, dict) and "data" in price_json and isinstance(price_json["data"], dict):
            d = price_json["data"]

            property_type = d.get("propertyType", property_type)
            operating_reserve = d.get("operatingReserve", operating_reserve)
            total_investment_value = d.get("marketCap", total_investment_value)
            tokens_available = d.get("tokensAvailable", tokens_available)
            underlying_asset_price = d.get("salePrice", underlying_asset_price)

            # if could not find these datas in the first method
            if projected_annual_cash_flow is None:
                projected_annual_cash_flow = d.get("projected_annual_cash_flow")

            if projected_rental_yield is None:
                projected_rental_yield = d.get("projected_rental_yield")

            if original_starting_date is None:
                original_starting_date = d.get("original_starting_date")

            if sellout_date is None:
                sellout_date = d.get("sellout_date")

            if isSellerBuyback is None:
                isSellerBuyback = d.get("isSellerBuyBack")

            # price history
            try:
                for pt in d.get("chartPoints", []) or []:
                    ts = pt.get("t")
                    price_rows.append({
                        "asa_id": asa_id,
                        "listing_url": listing_link,
                        "timestamp": datetime.utcfromtimestamp(ts / 1000) if ts else None,
                        "price": pt.get("p"),
                        "quantity": pt.get("q"),
                    })
            except Exception as e:
                print("ChartPoints parse failed:", e)

        # yield history
        if isinstance(yields_json, dict) and "data" in yields_json:
            try:
                yield_points = yields_json.get("data", []) or []
                for pt in yield_points:
                    ts = pt.get("t")
                    yield_rows.append({
                        "asa_id": asa_id,
                        "listing_url": listing_link,
                        "date": datetime.utcfromtimestamp(ts / 1000).date() if ts else None,
                        "yield_pct": pt.get("y"),
                        "starting_price": pt.get("min"),
                        "monthly_rent": pt.get("rent"),
                    })

                if original_starting_date is None and len(yield_points) > 0:
                    first_ts = yield_points[0].get("t")
                    if first_ts:
                        original_starting_date = datetime.utcfromtimestamp(first_ts / 1000).date()
            except Exception as e:
                print("Yield parse failed:", e)

        if property_type is None:
            property_type = get_any(prop, ["property_type", "propertyType"], default=None)

        property_rows.append({
            "property_id": property_id,
            "address": address,
            "city": city,
            "state": state,
            "zip_code": zip_code,
            "asa_id": asa_id,
            "property_type": property_type,
            "sqft": sqft,
            "tokens_total": total_tokens,
            "tokens_available": tokens_available,
            "starting_price": starting_price,
            "underlying_asset_price": underlying_asset_price,
            "total_investment_value": total_investment_value,
            "operating_reserve": operating_reserve,
            "year_built": year_built,
            "utilities": utilities,
            "taxes": property_taxes,
            "insurance": insurance,
            "llc_admin_yearly": llc_admin_fee_yearly,
            "llc_admin_upfront": llc_admin_fee_upfront,
            "management_fees": management_fees,
            "monthly_rent": monthly_rent,
            "annual_cash_flow": annual_cash_flow,
            "cap_rate": cap_rate,
            "projected_return": projected_annual_return,
            "appreciation": appreciation,
            "num_images": num_of_images,
            "description": description_text,
            "description_char_len": char_length,
            "original_starting_date": original_starting_date,
            "sellout_date": sellout_date,
            "isSellerBuyBack": isSellerBuyback,
            "projected_annual_cash_flow": projected_annual_cash_flow,
            "projected_rental_yield": projected_rental_yield,
            "listing_url": listing_link,
        })

        print("Scraped", asa_id, property_id)

    except Exception as e:
        print("Listing failed but continuing:", listing_link)
        print("       ", repr(e))
        failed_rows.append({
            "listing_url": listing_link,
            "error": repr(e),
            "traceback": traceback.format_exc(limit=1)
        })
        continue

# SAVE
df_properties = pd.DataFrame(property_rows)
df_prices = pd.DataFrame(price_rows)
df_yields = pd.DataFrame(yield_rows)
df_failed = pd.DataFrame(failed_rows)

# Deduplicate
if not df_properties.empty:
    prop_dedupe_cols = [c for c in ["property_id", "asa_id", "listing_url"] if c in df_properties.columns]
    if prop_dedupe_cols:
        df_properties = df_properties.drop_duplicates(subset=prop_dedupe_cols, keep="first")

if not df_prices.empty:
    price_dedupe_cols = [c for c in ["asa_id", "timestamp", "price", "quantity"] if c in df_prices.columns]
    if price_dedupe_cols:
        df_prices = df_prices.drop_duplicates(subset=price_dedupe_cols, keep="first")

if not df_yields.empty:
    yield_dedupe_cols = [c for c in ["asa_id", "date", "yield_pct", "starting_price", "monthly_rent"] if c in df_yields.columns]
    if yield_dedupe_cols:
        df_yields = df_yields.drop_duplicates(subset=yield_dedupe_cols, keep="first")

df_properties.to_csv(OUT_PROPERTIES_CSV, index=False)
df_prices.to_csv(OUT_PRICES_CSV, index=False)
df_yields.to_csv(OUT_YIELDS_CSV, index=False)

# Failure log
if not df_failed.empty:
    df_failed.to_csv("failed_v7.csv", index=False)
    print(f" - failed_v4.csv: {len(df_failed)} failed listings")

driver.quit()
import os
import json
import time
import requests
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv

from pathlib import Path

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# Config & Credentials
# =============================================================================
AZURE_API_KEY   = os.getenv("AZURE_OPENAI_KEY")
SERPER_API_KEY  = os.getenv("SERPER_API_KEY")
ENDPOINT        = "https://exl-services-resource.cognitiveservices.azure.com/"
# --- Search parameters (loaded from config.json) ---
with open(PROJECT_ROOT / "config.json") as _f:
    _cfg = json.load(_f)

DEPLOYMENT_NAME  = _cfg.get("scraper_deployment", "gpt-4.1-mini")
COMPANY_NAME     = _cfg["company"]
POSITIONS_INPUT  = _cfg["positions"]
NUM_ROWS         = int(_cfg["num_rows"])
SELECTED_COUNTRY = _cfg["country"]

# --- Country mapping ---
COUNTRY_MAP = {
    "United States": {"subdomain": "www", "gl": "us"},
    "Canada":        {"subdomain": "ca",  "gl": "ca"},
    "India":         {"subdomain": "in",  "gl": "in"},
}

country_settings = COUNTRY_MAP[SELECTED_COUNTRY]
subdomain        = country_settings["subdomain"]
gl_code          = country_settings["gl"]
positions        = [pos.strip() for pos in POSITIONS_INPUT.split(",")]

os.makedirs(PROJECT_ROOT / "outputs", exist_ok=True)
OUTPUT_CSV = str(PROJECT_ROOT / "outputs/leads_raw.csv")

print("✅ Config loaded")
print(f"   Company  : {COMPANY_NAME}")
print(f"   Positions: {positions}")
print(f"   Country  : {SELECTED_COUNTRY}")
print(f"   Target   : {NUM_ROWS} rows")


# =============================================================================
# Scrape raw LinkedIn data via Serper
# =============================================================================
def scrape_raw_linkedin_data(company, positions_list, target_count, subdomain, gl_code):
    all_leads    = []
    current_page = 1

    positions_str = " OR ".join([f'intitle:"{pos}"' for pos in positions_list])
    query = f'site:{subdomain}.linkedin.com/in/ "{company}" ({positions_str})'

    print(f"\nExecuting Query: {query}")
    print(f"Searching from Geolocation: {gl_code.upper()}...")
    print(f"Targeting {target_count} rows...\n")

    url = "https://google.serper.dev/search"

    while len(all_leads) < target_count:
        print(f"Pulling Google Page {current_page}...")

        payload = json.dumps({"q": query, "page": current_page, "gl": gl_code})
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

        response = requests.request("POST", url, headers=headers, data=payload)

        if response.status_code != 200:
            print(f"API Error {response.status_code}: {response.text}")
            break

        data            = response.json()
        organic_results = data.get("organic", [])

        if not organic_results:
            print("No more results found. Reached the end of the search.")
            break

        for item in organic_results:
            all_leads.append({
                "Raw Title":   item.get("title", ""),
                "LinkedIn URL": item.get("link", ""),
                "Raw Snippet": item.get("snippet", "").replace("\n", " "),
            })
            if len(all_leads) >= target_count:
                break

        current_page += 1

        if current_page > 20:
            print("Hit safety limit of 20 pages.")
            break

        time.sleep(0.5)

    return pd.DataFrame(all_leads)


# =============================================================================
# Run scraper
# =============================================================================
raw_leads_df = scrape_raw_linkedin_data(COMPANY_NAME, positions, NUM_ROWS, subdomain, gl_code)

if not raw_leads_df.empty:
    print(f"\nSuccessfully scraped {len(raw_leads_df)} raw leads.")
    print(raw_leads_df.to_string())
else:
    print("\nNo leads found. Check your search parameters or API key.")


# =============================================================================
# Azure OpenAI — enrich leads with structured extraction
# =============================================================================
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=ENDPOINT,
    api_key=AZURE_API_KEY,
)


def extract_structured_data(row):
    title   = row["Raw Title"]
    snippet = row["Raw Snippet"]

    system_prompt = """
    You are an expert B2B data extraction tool.
    Analyze the provided LinkedIn search text and extract the following into a strict JSON object:
    - "Clean_Name": The person's exact name.
    - "Clean_Job_Title": Their current job title.
    - "Location": Their city or country.
    - "Department_or_Domain": The specific business unit they work in (e.g., Asset Management, Investment Banking, Global Technology).
    - "Focus_Area_or_Keywords": Any specific projects, skills, or focus areas mentioned (e.g., Public Cloud, Commodity Derivatives, Security & Resiliency).
    - "Education": Any university or college mentioned.

    If any piece of information is completely missing from the text, output "Not specified" for that key. Do not invent information.
    """

    user_prompt = f"Raw Title: {title}\nRaw Snippet: {snippet}"

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        extracted_json = json.loads(response.choices[0].message.content)
        return pd.Series([
            extracted_json.get("Clean_Name",            "Unknown"),
            extracted_json.get("Clean_Job_Title",        "Unknown"),
            extracted_json.get("Location",               "Unknown"),
            extracted_json.get("Department_or_Domain",   "Unknown"),
            extracted_json.get("Focus_Area_or_Keywords", "Unknown"),
            extracted_json.get("Education",              "Unknown"),
        ])
    except Exception as e:
        print(f"Error processing row: {e}")
        return pd.Series(["Error", "Error", "Error", "Error", "Error", "Error"])


print(f"\nSending {len(raw_leads_df)} raw leads to Azure {DEPLOYMENT_NAME} for deep extraction...")

raw_leads_df[["Name", "Job Title", "Location", "Department", "Focus Keywords", "Education"]] = \
    raw_leads_df.apply(extract_structured_data, axis=1)

final_marketing_list = raw_leads_df[[
    "Name", "Job Title", "Department", "Focus Keywords", "Location", "Education", "LinkedIn URL"
]]

# Deduplicate on LinkedIn URL (Serper can return the same profile across pages)
dupes = len(final_marketing_list) - len(final_marketing_list.drop_duplicates(subset=["LinkedIn URL"]))
if dupes > 0:
    print(f"\n⚠️  Removing {dupes} duplicate lead(s)")
final_marketing_list = final_marketing_list.drop_duplicates(subset=["LinkedIn URL"], keep="first")

print(final_marketing_list.to_string())


# =============================================================================
# Save to CSV
# =============================================================================
final_marketing_list.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved {len(final_marketing_list)} leads to {OUTPUT_CSV}")

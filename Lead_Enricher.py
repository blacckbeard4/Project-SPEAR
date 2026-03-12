import os
import json
import time
import difflib
import requests
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Config & Credentials
# =============================================================================
AZURE_API_KEY    = os.getenv("AZURE_OPENAI_KEY")
APIFY_API_TOKEN  = os.getenv("APIFY_API_TOKEN")
SERPER_API_KEY   = os.getenv("SERPER_API_KEY")
ENDPOINT         = "https://exl-services-resource.cognitiveservices.azure.com/"

with open("./config.json") as _f:
    _cfg = json.load(_f)

DEPLOYMENT_NAME    = _cfg.get("scraper_deployment", "gpt-4.1-mini")
COMPANY_NAME       = _cfg["company"]
ENRICHMENT_ENABLED = _cfg.get("enrichment_enabled", True)
SELECTED_COUNTRY   = _cfg["country"]

COUNTRY_MAP = {
    "United States": "us",
    "Canada":        "ca",
    "India":         "in",
}
gl_code = COUNTRY_MAP.get(SELECTED_COUNTRY, "us")

INPUT_CSV  = "./outputs/leads_raw.csv"
OUTPUT_CSV = "./outputs/leads_final.csv"

# Apify actor for LinkedIn profile scraping (no cookies needed)
# Format: username~actorName (tilde separator for Apify API)
APIFY_ACTOR_ID = "supreme_coder~linkedin-profile-scraper"

print("✅ Lead Enricher config loaded")
print(f"   Enrichment enabled: {ENRICHMENT_ENABLED}")
print(f"   Input : {INPUT_CSV}")
print(f"   Output: {OUTPUT_CSV}")


# =============================================================================
# Azure OpenAI client (for GPT extraction from enriched data)
# =============================================================================
gpt_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=ENDPOINT,
    api_key=AZURE_API_KEY,
)


# =============================================================================
# Step 1: Load & Deduplicate
# =============================================================================
def deduplicate_leads(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates by URL normalization and fuzzy name matching."""
    before = len(df)

    # Exact LinkedIn URL dedup
    df = df.drop_duplicates(subset=["LinkedIn URL"], keep="first")

    # Normalize URLs (lowercase, strip trailing slashes)
    df = df.copy()
    df["_norm_url"] = df["LinkedIn URL"].str.lower().str.rstrip("/")
    df = df.drop_duplicates(subset=["_norm_url"], keep="first")
    df = df.drop(columns=["_norm_url"])

    # Fuzzy name matching — flag near-duplicates
    to_drop = set()
    names = df["Name"].tolist()
    indices = df.index.tolist()
    for i in range(len(names)):
        if indices[i] in to_drop:
            continue
        for j in range(i + 1, len(names)):
            if indices[j] in to_drop:
                continue
            ratio = difflib.SequenceMatcher(
                None, str(names[i]).lower(), str(names[j]).lower()
            ).ratio()
            if ratio > 0.85:
                to_drop.add(indices[j])

    if to_drop:
        df = df.drop(index=list(to_drop))

    removed = before - len(df)
    if removed > 0:
        print(f"   Dedup: removed {removed} duplicate(s)")
    else:
        print("   Dedup: no duplicates found")

    return df.reset_index(drop=True)


# =============================================================================
# Step 2: Apify LinkedIn Profile Enrichment
# =============================================================================
def enrich_via_apify(linkedin_url: str) -> dict | None:
    """Call Apify LinkedIn Profile Scraper actor for a single profile URL."""
    if not APIFY_API_TOKEN:
        print("   ⚠️  No APIFY_API_TOKEN set, skipping Apify enrichment")
        return None

    try:
        # Start the actor run synchronously and get dataset items directly
        # This avoids polling — Apify returns results when the run completes
        sync_url = (
            f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID}"
            f"/run-sync-get-dataset-items?token={APIFY_API_TOKEN}"
        )
        run_input = {
            "urls": [{"url": linkedin_url}],
        }
        resp = requests.post(
            sync_url,
            json=run_input,
            headers={"Content-Type": "application/json"},
            timeout=120,  # sync runs can take longer
        )

        if resp.status_code not in (200, 201):
            print(f"   ⚠️  Apify failed ({resp.status_code}): {resp.text[:200]}")
            return None

        # Sync endpoint returns dataset items directly as a JSON array
        items = resp.json()
        if isinstance(items, list) and len(items) > 0:
            return items[0]

        return None

    except Exception as e:
        print(f"   ⚠️  Apify error: {e}")
        return None


def extract_fields_from_apify(profile: dict) -> dict:
    """Map Apify profile JSON to our pipeline fields — extract maximum info.

    Apify actor 'supreme_coder~linkedin-profile-scraper' returns fields like:
    firstName, lastName, headline, geoLocationName, geoCountryName,
    educations, positions, certifications, honors, connectionsCount, etc.
    """
    # Location — Apify provides geoLocationName (e.g., "New York, New York, United States")
    location = (
        profile.get("geoLocationName", "")
        or profile.get("location", "")
        or profile.get("city", "")
        or ""
    )

    # Education — from 'educations' array
    education_entries = []
    edu_list = profile.get("educations", profile.get("education", []))
    if isinstance(edu_list, list):
        for edu in edu_list:
            if not isinstance(edu, dict):
                continue
            school = edu.get("schoolName", "") or edu.get("name", "") or ""
            degree = edu.get("degreeName", "") or edu.get("degree", "") or ""
            field = edu.get("fieldOfStudy", "") or edu.get("field", "") or ""
            parts = [p for p in [school, degree, field] if p]
            if parts:
                education_entries.append(" — ".join(parts))
    education_str = "; ".join(education_entries) if education_entries else ""

    # Skills — from 'skills' array (can be strings or objects)
    skills_raw = profile.get("skills", [])
    skills = []
    if isinstance(skills_raw, list):
        for s in skills_raw[:20]:
            if isinstance(s, str) and s:
                skills.append(s)
            elif isinstance(s, dict):
                name = s.get("name", "") or s.get("skill", "") or ""
                if name:
                    skills.append(name)

    # Summary / About
    summary = profile.get("summary", "") or profile.get("about", "") or ""

    # Positions — from 'positions' array
    positions = profile.get("positions", profile.get("experience", []))
    current_exp_desc = ""
    current_company = ""
    current_title = ""
    experience_summary_parts = []
    if isinstance(positions, list):
        for i, pos in enumerate(positions):
            if not isinstance(pos, dict):
                continue
            title = pos.get("title", "") or ""
            # Company can be nested or direct
            company_data = pos.get("company", {})
            if isinstance(company_data, dict):
                company = company_data.get("name", "") or ""
            else:
                company = pos.get("companyName", "") or str(company_data or "")
            desc = pos.get("description", "") or ""
            location_name = pos.get("locationName", "") or ""

            # Check if current position (no endDate in timePeriod)
            time_period = pos.get("timePeriod", {}) or {}
            is_current = not time_period.get("endDate")

            if is_current and not current_exp_desc:
                current_exp_desc = desc
                current_company = company
                current_title = title

            # Capture top 3 experiences for richer context
            if i < 3 and (title or company):
                entry = f"{title} at {company}" if company else title
                if location_name:
                    entry += f" ({location_name})"
                if desc:
                    entry += f" — {desc[:150]}"
                experience_summary_parts.append(entry)

        if not current_exp_desc and positions:
            first = positions[0]
            if isinstance(first, dict):
                current_exp_desc = first.get("description", "") or ""
                cd = first.get("company", {})
                current_company = cd.get("name", "") if isinstance(cd, dict) else str(cd or "")
                current_title = first.get("title", "") or ""

    experience_summary = "\n".join(experience_summary_parts)

    # Industry
    industry = profile.get("industry", "") or profile.get("industryName", "") or ""

    # Headline
    headline = profile.get("headline", "") or ""

    # Certifications
    certs = []
    for cert in (profile.get("certifications", []) or [])[:5]:
        if isinstance(cert, dict):
            name = cert.get("name", "") or cert.get("title", "") or ""
            authority = cert.get("authority", "") or cert.get("issuer", "") or ""
            if name:
                certs.append(f"{name} ({authority})" if authority else name)

    # Honors / Awards
    honors = []
    for honor in (profile.get("honors", []) or [])[:5]:
        if isinstance(honor, dict):
            title = honor.get("title", "") or honor.get("name", "") or ""
            if title:
                honors.append(title)

    # Languages
    languages = []
    for lang in (profile.get("languages", []) or []):
        if isinstance(lang, dict):
            name = lang.get("name", "") or ""
            if name:
                languages.append(name)
        elif isinstance(lang, str) and lang:
            languages.append(lang)

    # Volunteer
    volunteer_entries = []
    for vol in (profile.get("volunteerExperiences", []) or [])[:3]:
        if isinstance(vol, dict):
            role = vol.get("role", "") or ""
            org = vol.get("companyName", "") or ""
            if role or org:
                volunteer_entries.append(f"{role} at {org}" if org else role)

    # Connections / followers
    connections = profile.get("connectionsCount", "") or ""
    followers = profile.get("followerCount", "") or ""

    return {
        "location": location,
        "education": education_str,
        "skills": ", ".join(skills),
        "summary": summary[:800] if summary else "",
        "industry": industry,
        "headline": headline,
        "current_exp_desc": current_exp_desc[:500] if current_exp_desc else "",
        "current_company": current_company,
        "current_title": current_title,
        "experience_summary": experience_summary[:1000] if experience_summary else "",
        "certifications": ", ".join(certs),
        "honors": ", ".join(honors),
        "languages": ", ".join(languages),
        "volunteer": ", ".join(volunteer_entries),
        "connections": str(connections) if connections else "",
        "followers": str(followers) if followers else "",
    }


# =============================================================================
# Step 2b: GPT extraction of Department & Focus Keywords from enriched data
# =============================================================================
def extract_department_and_keywords(name: str, job_title: str, headline: str,
                                     summary: str, skills: str,
                                     current_exp_desc: str) -> dict:
    """Use GPT to derive Department and Focus Keywords from rich profile data."""
    system_prompt = """You are an expert B2B data extraction tool.
Given a professional's profile information, extract:
- "Department_or_Domain": The specific business unit or domain they work in (e.g., Asset Management, Investment Banking, Global Technology, Risk Analytics). Infer from their headline, experience, and skills.
- "Focus_Area_or_Keywords": Specific projects, technologies, skills, or focus areas (e.g., Python Development, Commodity Derivatives, Cloud Migration, Credit Risk). List 3-5 most relevant ones, comma-separated.

If information is insufficient to determine a field, output "Not specified". Do not invent information.
Output strict JSON with these two keys only."""

    user_prompt = (
        f"Name: {name}\n"
        f"Job Title: {job_title}\n"
        f"Headline: {headline}\n"
        f"Summary: {summary}\n"
        f"Skills: {skills}\n"
        f"Current Role Description: {current_exp_desc}"
    )

    try:
        response = gpt_client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "department": result.get("Department_or_Domain", "Not specified"),
            "focus_keywords": result.get("Focus_Area_or_Keywords", "Not specified"),
        }
    except Exception as e:
        print(f"   ⚠️  GPT extraction error: {e}")
        return {"department": "Not specified", "focus_keywords": "Not specified"}


# =============================================================================
# Step 3: Serper Backfill (for leads where Apify returned no data)
# =============================================================================
def serper_backfill(name: str, company: str, job_title: str) -> dict:
    """Run multiple targeted Google searches to enrich lead data."""
    if not SERPER_API_KEY:
        return {}

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    all_snippets = []

    # Query 1: General person search
    queries = [
        f'"{name}" "{company}" "{job_title}"',
        f'"{name}" "{company}" LinkedIn',
        f'"{name}" "{company}" biography OR profile OR speaker OR alumni',
    ]

    for query in queries:
        try:
            resp = requests.post(
                "https://google.serper.dev/search",
                headers=headers,
                data=json.dumps({"q": query, "gl": gl_code, "num": 5}),
                timeout=15,
            )
            if resp.status_code == 200:
                for r in resp.json().get("organic", []):
                    snippet = r.get("snippet", "")
                    title = r.get("title", "")
                    if snippet:
                        all_snippets.append(f"{title}: {snippet}")
        except Exception:
            continue
        time.sleep(0.3)

    if not all_snippets:
        return {}

    snippets_text = "\n".join(all_snippets[:15])

    # Send to GPT for extraction
    system_prompt = """You are an expert B2B data extraction tool.
Analyze the provided web search snippets about a professional and extract:
- "Location": Their city or country.
- "Department_or_Domain": Their business unit or domain (e.g., Investment Banking, Asset Management, Technology).
- "Focus_Area_or_Keywords": Specific projects, technologies, skills, or focus areas (3-5, comma-separated). Look for mentions of specific initiatives, technologies, trading desks, product areas, etc.
- "Education": Any university, college, or degree mentioned.
- "Headline": Their professional headline or tagline if found.
- "Skills": Any specific technical or professional skills mentioned (comma-separated).

If any field is not found, output "Not specified". Do not invent information.
Output strict JSON with these six keys only."""

    user_prompt = (
        f"Person: {name}, {job_title} at {company}\n\n"
        f"Web search snippets:\n{snippets_text[:3000]}"
    )

    try:
        response = gpt_client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "location": result.get("Location", "Not specified"),
            "department": result.get("Department_or_Domain", "Not specified"),
            "focus_keywords": result.get("Focus_Area_or_Keywords", "Not specified"),
            "education": result.get("Education", "Not specified"),
            "headline": result.get("Headline", "Not specified"),
            "skills": result.get("Skills", "Not specified"),
        }

    except Exception as e:
        print(f"   ⚠️  Serper backfill error: {e}")
        return {}


# =============================================================================
# Step 4: Merge and Output
# =============================================================================
def is_missing(val) -> bool:
    """Check if a field value is effectively missing."""
    if pd.isna(val):
        return True
    s = str(val).strip().lower()
    return s in ("", "not specified", "unknown", "error", "nan", "none")


def pick_best(apify_val: str, serper_val: str, original_val: str) -> str:
    """Pick best available value with priority: apify > serper > original."""
    for val in [apify_val, serper_val, original_val]:
        if val and not is_missing(val):
            return str(val).strip()
    return "Not specified"


def run_enrichment():
    """Main enrichment pipeline."""

    # Load leads
    if not os.path.exists(INPUT_CSV):
        # Fallback: try leads_final.csv if leads_raw.csv doesn't exist yet
        fallback = "./outputs/leads_final.csv"
        if os.path.exists(fallback):
            print(f"   ℹ️  {INPUT_CSV} not found, using {fallback}")
            df = pd.read_csv(fallback)
        else:
            print(f"❌ No input file found at {INPUT_CSV} or {fallback}")
            return
    else:
        df = pd.read_csv(INPUT_CSV)

    print(f"\n📥 Loaded {len(df)} leads")

    # Deduplicate
    df = deduplicate_leads(df)
    print(f"   {len(df)} leads after dedup")

    if not ENRICHMENT_ENABLED:
        print("⏭️  Enrichment disabled in config, saving as-is")
        df.to_csv(OUTPUT_CSV, index=False)
        return

    # Track enrichment stats
    stats = {"apify_success": 0, "serper_backfill": 0, "unchanged": 0}

    # Apify health check — test with first lead, skip all if it fails
    apify_available = False
    if APIFY_API_TOKEN and len(df) > 0:
        first_url = df.iloc[0].get("LinkedIn URL", "")
        if first_url:
            print("\n🔍 Testing Apify availability...")
            test_profile = enrich_via_apify(first_url)
            if test_profile:
                apify_available = True
                print("   ✅ Apify is working")
            else:
                print("   ⚠️  Apify returned no data — skipping Apify for all leads")
                print("   ℹ️  Will use enhanced Serper + GPT backfill instead")

    # Process each lead
    for idx in range(len(df)):
        row = df.iloc[idx]
        name = row.get("Name", "Unknown")
        job_title = row.get("Job Title", "Unknown")
        linkedin_url = row.get("LinkedIn URL", "")

        print(f"\n{'─'*50}")
        print(f"  Enriching {idx+1}/{len(df)}: {name} — {job_title}")

        apify_fields = {}
        gpt_enriched = {}

        # --- Apify enrichment (only if health check passed) ---
        if apify_available and linkedin_url and idx > 0:  # idx 0 already tested
            profile = enrich_via_apify(linkedin_url)
            if profile:
                apify_fields = extract_fields_from_apify(profile)
                print(f"   ✅ Apify: got profile data")
                stats["apify_success"] += 1

                # Use GPT to derive Department and Focus Keywords from rich data
                if apify_fields.get("summary") or apify_fields.get("skills"):
                    gpt_enriched = extract_department_and_keywords(
                        name=name,
                        job_title=job_title,
                        headline=apify_fields.get("headline", ""),
                        summary=apify_fields.get("summary", ""),
                        skills=apify_fields.get("skills", ""),
                        current_exp_desc=apify_fields.get("current_exp_desc", ""),
                    )
            else:
                print(f"   ⚠️  Apify: no data returned")
        elif apify_available and idx == 0 and test_profile:
            # Use the health check result for the first lead
            apify_fields = extract_fields_from_apify(test_profile)
            print(f"   ✅ Apify: using health check data")
            stats["apify_success"] += 1
            if apify_fields.get("summary") or apify_fields.get("skills"):
                gpt_enriched = extract_department_and_keywords(
                    name=name, job_title=job_title,
                    headline=apify_fields.get("headline", ""),
                    summary=apify_fields.get("summary", ""),
                    skills=apify_fields.get("skills", ""),
                    current_exp_desc=apify_fields.get("current_exp_desc", ""),
                )

        # --- Always run Serper backfill for any missing fields ---
        serper_fields = {}
        has_gaps = any([
            is_missing(apify_fields.get("location", "")) and is_missing(row.get("Location", "")),
            is_missing(gpt_enriched.get("department", "")) and is_missing(row.get("Department", "")),
            is_missing(gpt_enriched.get("focus_keywords", "")) and is_missing(row.get("Focus Keywords", "")),
        ])

        if has_gaps:
            print(f"   🔍 Running Serper backfill...")
            serper_fields = serper_backfill(name, COMPANY_NAME, job_title)
            if serper_fields:
                stats["serper_backfill"] += 1

        # --- Merge with priority ---
        df.at[df.index[idx], "Location"] = pick_best(
            apify_fields.get("location", ""),
            serper_fields.get("location", ""),
            row.get("Location", ""),
        )
        df.at[df.index[idx], "Education"] = pick_best(
            apify_fields.get("education", ""),
            serper_fields.get("education", ""),
            row.get("Education", ""),
        )
        df.at[df.index[idx], "Department"] = pick_best(
            gpt_enriched.get("department", ""),
            serper_fields.get("department", ""),
            row.get("Department", ""),
        )
        df.at[df.index[idx], "Focus Keywords"] = pick_best(
            gpt_enriched.get("focus_keywords", ""),
            serper_fields.get("focus_keywords", ""),
            row.get("Focus Keywords", ""),
        )

        # New enriched columns — store all available data
        if apify_fields:
            df.at[df.index[idx], "Headline"] = apify_fields.get("headline", "")
            df.at[df.index[idx], "Summary"] = apify_fields.get("summary", "")
            df.at[df.index[idx], "Industry"] = apify_fields.get("industry", "")
            df.at[df.index[idx], "Skills"] = apify_fields.get("skills", "")
            df.at[df.index[idx], "Certifications"] = apify_fields.get("certifications", "")
            df.at[df.index[idx], "Honors"] = apify_fields.get("honors", "")
            df.at[df.index[idx], "Languages"] = apify_fields.get("languages", "")
            df.at[df.index[idx], "Volunteer"] = apify_fields.get("volunteer", "")
            df.at[df.index[idx], "Experience Summary"] = apify_fields.get("experience_summary", "")
            df.at[df.index[idx], "Connections"] = apify_fields.get("connections", "")
            df.at[df.index[idx], "Followers"] = apify_fields.get("followers", "")
        elif serper_fields:
            # Store Serper-derived headline and skills when Apify unavailable
            headline_val = serper_fields.get("headline", "")
            skills_val = serper_fields.get("skills", "")
            if headline_val and not is_missing(headline_val):
                df.at[df.index[idx], "Headline"] = headline_val
            if skills_val and not is_missing(skills_val):
                df.at[df.index[idx], "Skills"] = skills_val

        if not apify_fields and not serper_fields:
            stats["unchanged"] += 1

    # --- Save enriched output ---
    os.makedirs("./outputs", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    # --- Print enrichment report ---
    print(f"\n{'='*60}")
    print("📊 Enrichment Report")
    print(f"{'='*60}")
    print(f"   Total leads        : {len(df)}")
    print(f"   Apify enriched     : {stats['apify_success']}")
    print(f"   Serper backfilled  : {stats['serper_backfill']}")
    print(f"   Unchanged          : {stats['unchanged']}")

    # Field fill rates
    enriched_cols = [
        "Department", "Focus Keywords", "Location", "Education",
        "Summary", "Skills", "Industry", "Headline",
        "Certifications", "Languages", "Experience Summary",
    ]
    for col in enriched_cols:
        if col in df.columns:
            filled = df[col].apply(lambda x: not is_missing(x)).sum()
            print(f"   {col:22s}: {filled}/{len(df)} filled ({filled/len(df)*100:.0f}%)")

    print(f"\n✅ Enriched leads saved to {OUTPUT_CSV}")


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    run_enrichment()

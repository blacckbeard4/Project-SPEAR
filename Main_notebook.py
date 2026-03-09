import os
import sys
import json
import subprocess
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Config — loaded from config.json (edit that file to change run parameters)
# =============================================================================
with open("./config.json") as _f:
    _cfg = json.load(_f)

COMPANY    = _cfg["company"]
POSITIONS  = _cfg["positions"]
NUM_ROWS   = int(_cfg["num_rows"])
COUNTRY    = _cfg["country"]
REBUILD_DB = bool(_cfg.get("rebuild_db", False))

HERE = Path(__file__).parent

print("🚀 Project Spear — Main Orchestrator")
print(f"\n📋 Run configuration:")
print(f"   Company         : {COMPANY}")
print(f"   Positions       : {POSITIONS}")
print(f"   Target rows     : {NUM_ROWS}")
print(f"   Country         : {COUNTRY}")
print(f"   Rebuild ChromaDB: {REBUILD_DB}")


# =============================================================================
# STEP 1 — Web Scrapper
# =============================================================================
print("\n" + "=" * 60)
print("STEP 1 — Web Scrapper")
print("=" * 60)

subprocess.run(
    [sys.executable, str(HERE / "Web_Scrapper.py")],
    check=True,
)

leads_df       = pd.read_csv("./outputs/leads_final.csv")
scraper_result = f"{len(leads_df)} leads saved"
print(f"\n✅ Scraper finished — {scraper_result}")


# =============================================================================
# STEP 2 — Database Creation (conditional)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2 — ChromaDB Setup")
print("=" * 60)

if REBUILD_DB:
    print("🔄 Rebuild requested — running Database_creation.py...")
    subprocess.run(
        [sys.executable, str(HERE / "Database_creation.py")],
        check=True,
    )
    print("\n✅ Database ready")
else:
    print("⏭️  Skipping DB rebuild (set REBUILD_DB = True to force a rebuild)")
    print("   ChromaDB will be loaded from ./chroma_db/ inside Agent_notebook.py")


# =============================================================================
# STEP 3 — LangGraph Email Agent
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3 — LangGraph Email Agent")
print("=" * 60)

subprocess.run(
    [sys.executable, str(HERE / "Agent_notebook.py")],
    check=True,
)

results_df   = pd.read_csv("./outputs/email_results.csv")
avg_score    = results_df["judge_score"].mean()
agent_result = f"{len(results_df)} emails generated, avg score {avg_score:.2f}/10"
print(f"\n✅ Agent finished — {agent_result}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("🎉 Pipeline complete!")
print("=" * 60)
print(f"   Company  : {COMPANY}")
print(f"   Scraper  : {scraper_result}")
print(f"   Agent    : {agent_result}")
print(f"\n📦 Outputs saved to:")
print(f"   Leads  : ./outputs/leads_final.csv")
print(f"   Emails : ./outputs/email_results.csv")

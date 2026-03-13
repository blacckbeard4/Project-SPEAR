import os
import re
import json
import chromadb
import pandas as pd
from typing import TypedDict
from openai import AzureOpenAI
from chromadb.config import Settings
from langchain_openai import AzureOpenAIEmbeddings
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from pathlib import Path

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# Config & Credentials
# =============================================================================
AZURE_EMB_KEY         = os.getenv("AZURE_EMBEDDING_KEY")
AZURE_EMB_DEPLOYMENT  = "text-embedding-ada-002"
AZURE_EMB_API_VERSION = "2023-05-15"

AZURE_GPT_KEY         = os.getenv("AZURE_OPENAI_KEY")
AZURE_GPT_API_VERSION = "2024-12-01-preview"

AZURE_ENDPOINT = "https://exl-services-resource.cognitiveservices.azure.com/"

CHROMA_LOCAL_DIR = str(PROJECT_ROOT / "chroma_db/")
LEADS_PATH       = str(PROJECT_ROOT / "outputs/leads_final.csv")
OUTPUT_PATH      = str(PROJECT_ROOT / "outputs/email_results.csv")
COLLECTION_NAME  = "exl_case_studies"

SCORE_THRESHOLD = 7.0
MAX_ITERATIONS  = 3
MIN_SEND_SCORE  = 7.0

# --- Config from config.json ---
with open(PROJECT_ROOT / "config.json") as _f:
    _cfg = json.load(_f)
SENDER_NAME                = _cfg.get("sender_name", "Justin Varghese")
SENDER_TITLE               = _cfg.get("sender_title", "EXL Service")
AZURE_GPT_STRATEGY_DEPLOYMENT = _cfg.get("strategy_deployment", "gpt-4.1")
AZURE_GPT_DRAFTER_DEPLOYMENT  = _cfg.get("drafter_deployment", "gpt-4.1")
AZURE_GPT_JUDGE_DEPLOYMENT    = _cfg.get("judge_deployment", "gpt-4.1")

SENIORITY_TIERS = {
    "president": "Frame around enterprise-wide business outcomes: revenue impact, market positioning, and strategic transformation. PREFER dollar amounts and top-line growth metrics IF the case study contains them — otherwise use qualitative enterprise outcomes. Tone: peer-to-peer, strategic.",
    "executive vice president": "Frame around enterprise-wide business outcomes: revenue impact, market positioning, and strategic transformation. PREFER dollar amounts and top-line growth metrics IF the case study contains them — otherwise use qualitative enterprise outcomes. Tone: peer-to-peer, strategic.",
    "senior vice president": "Frame around enterprise-wide business outcomes: revenue impact, market positioning, and strategic transformation. PREFER dollar amounts and top-line growth metrics IF the case study contains them — otherwise use qualitative enterprise outcomes. Tone: peer-to-peer, strategic.",
    "managing director": "Frame around P&L impact, revenue influence, and market differentiation. PREFER dollar figures and enterprise-scale outcomes IF the case study contains them — otherwise use qualitative outcomes like 'improved data quality' or 'cleared backlog.' Never invent dollar amounts. Tone: consultative, strategic.",
    "executive director": "Frame around scalable business outcomes and cross-functional ROI. PREFER percentage improvements and efficiency gains IF the case study contains them — otherwise describe outcomes qualitatively. Tone: consultative.",
    "senior director": "Frame around operational execution and measurable delivery. PREFER time savings, error rate improvements, and FTE metrics IF the case study contains them — otherwise describe outcomes qualitatively. Tone: results-oriented.",
    "vice president": "Frame around operational efficiency, time-to-value, and risk reduction. PREFER percentage improvements, time savings, and process automation metrics IF the case study contains them — otherwise describe outcomes qualitatively. Tone: peer-to-peer, practical.",
    "director": "Frame around hands-on operational metrics: FTE consolidation, cycle time reduction, error rate improvements, and backlog clearance. Only cite numbers that appear in the case study. Tone: results-oriented, practical.",
}


def get_seniority_guidance(job_title: str) -> str:
    title_lower = job_title.lower()
    for key, guidance in SENIORITY_TIERS.items():
        if key in title_lower:
            return guidance
    return SENIORITY_TIERS["vice president"]


print("✅ Config loaded")
print(f"   Strategy model : {AZURE_GPT_STRATEGY_DEPLOYMENT}")
print(f"   Drafter model  : {AZURE_GPT_DRAFTER_DEPLOYMENT}")
print(f"   Judge model    : {AZURE_GPT_JUDGE_DEPLOYMENT}")
print(f"   Score threshold: {SCORE_THRESHOLD}/10")
print(f"   Max iterations : {MAX_ITERATIONS}")


# =============================================================================
# Open ChromaDB + build retrieve()
# =============================================================================
chroma_client = chromadb.PersistentClient(
    path=CHROMA_LOCAL_DIR,
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_collection(name=COLLECTION_NAME)
print(f"✅ Collection ready — {collection.count()} vectors")

_all_meta = collection.get(include=["metadatas"])["metadatas"]
_all_filenames = sorted(set(m["filename"] for m in _all_meta))
print("📚 Case studies indexed in ChromaDB:")
for _fn in _all_filenames:
    print(f"   - {_fn}")
print(f"   Total unique case studies: {len(_all_filenames)}")

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_EMB_DEPLOYMENT,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_EMB_KEY,
    api_version=AZURE_EMB_API_VERSION,
)


def retrieve(query: str, n_results: int = 3) -> list[dict]:
    """Query ChromaDB and return top-n chunks with metadata and relevance score."""
    query_vector = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "filename":        meta["filename"],
            "section":         meta["section"],
            "text":            doc,
            "relevance_score": round(1 - dist, 4),
        })
    return chunks


test = retrieve("financial services risk analytics", n_results=1)
print(f"✅ retrieve() works — top result: {test[0]['filename']} (score: {test[0]['relevance_score']})")


# =============================================================================
# Azure OpenAI GPT client
# =============================================================================
gpt_client = AzureOpenAI(
    api_version=AZURE_GPT_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_GPT_KEY,
)


def call_gpt(system_prompt: str, user_prompt: str, model: str = None) -> str:
    response = gpt_client.chat.completions.create(
        model=model or AZURE_GPT_DRAFTER_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


print("✅ GPT client ready")


# =============================================================================
# LangGraph Shared State
# =============================================================================
class AgentState(TypedDict):
    lead:               dict
    case_study_briefs:  list
    strategy_brief:     str
    email_draft:        str
    judge_score:        float
    judge_feedback:     str
    iterations:         int


# =============================================================================
# Node 1: Retrieval
# =============================================================================
def retrieval_node(state: AgentState) -> dict:
    lead = state["lead"]

    query_parts = [
        lead.get("Job Title", ""),
        lead.get("Department", ""),
        lead.get("Focus Keywords", ""),
        lead.get("Industry", ""),
        str(lead.get("Skills", ""))[:100],
    ]
    query = " ".join(p for p in query_parts if p and p.lower() != "not specified")

    n_results = _cfg.get("retrieval_n_results", 3)
    raw_chunks = retrieve(query, n_results=n_results)

    briefs = []
    for chunk in raw_chunks:
        briefs.append({
            "filename":        chunk["filename"],
            "section":         chunk["section"],
            "summary":         chunk["text"],
            "relevance_score": chunk["relevance_score"],
        })

    unique_files = sorted(set(b["filename"] for b in briefs))
    print(f"   [Retrieval] Query: '{query}'")
    print(f"   [Retrieval] ChromaDB files retrieved: {unique_files}")
    for b in briefs:
        print(f"   → {b['filename']} | {b['section']} | score={b['relevance_score']}")

    return {"case_study_briefs": briefs}


# =============================================================================
# Node 2: Strategy
# =============================================================================
def strategy_node(state: AgentState) -> dict:
    lead   = state["lead"]
    briefs = state["case_study_briefs"]

    briefs_text = "\n\n".join([
        f"Case Study: {b['filename']}\nSection: {b['section']}\nContent: {b['summary']}"
        for b in briefs
    ])
    # Include all enriched fields in lead context
    lead_text = "\n".join([
        f"{k}: {v}" for k, v in lead.items()
        if str(v).strip().lower() not in ("not specified", "nan", "", "none") and pd.notna(v)
    ])
    seniority_guidance = get_seniority_guidance(lead.get("Job Title", ""))

    system_prompt = f"""You are a senior B2B marketing strategist at EXL Service.
Read the lead's profile and relevant EXL case study excerpts,
then produce a concise strategy brief for a personalised outreach email.

Your strategy brief must specify:
1. The ONE primary pain point or business challenge most relevant to this lead.
2. Which case study to lead with and why it resonates with their role. Give the case study a proper readable name (NOT the raw filename).
3. The core value proposition angle (e.g., cost reduction, risk management, digital transformation).
4. Any specific hook from their profile (focus keywords, department, location) to open with.
5. The preferred tone (e.g., consultative, peer-to-peer, analytical).
6. Seniority-appropriate framing: {seniority_guidance}

Keep the brief under 200 words. Be specific, not generic."""

    user_prompt = (
        f"LEAD PROFILE:\n{lead_text}\n\n"
        f"RELEVANT EXL CASE STUDY EXCERPTS:\n{briefs_text}\n\n"
        f"Write the strategy brief now."
    )

    strategy_brief = call_gpt(system_prompt, user_prompt, model=AZURE_GPT_STRATEGY_DEPLOYMENT)
    print(f"   [Strategy] Brief generated ({len(strategy_brief.split())} words)")

    return {"strategy_brief": strategy_brief}


# =============================================================================
# Node 3: Email Drafter
# =============================================================================
def email_drafter_node(state: AgentState) -> dict:
    lead           = state["lead"]
    strategy_brief = state["strategy_brief"]
    iterations     = state.get("iterations", 0)
    judge_feedback = state.get("judge_feedback", "")

    lead_text = "\n".join([f"{k}: {v}" for k, v in lead.items()])
    seniority_guidance = get_seniority_guidance(lead.get("Job Title", ""))

    # Build briefs summary for revision context
    briefs = state.get("case_study_briefs", [])
    briefs_text_summary = "\n".join([
        f"- {b['filename']} / {b['section']}: {b['summary'][:200]}"
        for b in briefs
    ]) if briefs else "No case study excerpts available."

    retry_context = ""
    if iterations > 0 and judge_feedback:
        hallucination_warning = ""
        if "HALLUCINATION" in judge_feedback.upper():
            hallucination_warning = (
                f"\n⚠️ HALLUCINATION DETECTED in previous draft. "
                f"In your revision, do NOT use any percentage or number unless it "
                f"appears exactly in this brief:\n{briefs_text_summary}\n"
                f"If no metric is available, describe the outcome qualitatively.\n"
            )
        retry_context = (
            f"\nIMPORTANT — REVISION REQUEST:\n"
            f"Your previous draft scored below the quality threshold.\n"
            f"Reviewer feedback to address:\n{judge_feedback}\n"
            f"{hallucination_warning}\n"
            f"AVAILABLE CASE STUDY EXCERPTS (only use metrics found here):\n{briefs_text_summary}\n\n"
            f"Fix the issues above while keeping the core message intact.\n"
        )

    first_name = lead.get("Name", "there").split()[0]

    system_prompt = f"""You are {SENDER_NAME}, writing a short outreach email to a senior executive.
Write like a real person — natural, conversational, no marketing jargon.

SENIORITY CONTEXT: {seniority_guidance}

Rules:
- Subject line at top (format: "Subject: ...")
- Start the email body with 'Hi {first_name},' on its own line. Use only their first name. Never open with their title, company, or location.
- The FIRST sentence after the greeting must be an observation about their WORK — not who they are
- Reference ONE case study result naturally (use a readable name, NEVER the raw filename). Example: say "our work with a leading asset manager on master data management" not "exl_master_data_management"
- Connect the case study outcome to THEIR specific situation in 1-2 sentences
- End with a casual, specific CTA (e.g., "Would a 20-minute call next Tuesday work?")
- Total length: 120-170 words (short and punchy — busy execs skim)
- End the email with exactly this two-line sign-off, nothing else:

Best,
{SENDER_NAME}, {SENDER_TITLE}

Do NOT use 'Best regards', do NOT put the name and company on separate lines, do NOT add trailing spaces or markdown line breaks.
The sign-off must be on exactly two lines: 'Best,' then '{SENDER_NAME}, {SENDER_TITLE}'.

OPENER RULE — the first sentence after 'Hi {first_name},' must be ONE of:
a) A specific skill, technology, or initiative from their Focus Keywords or Department in the lead profile
b) A pressure specific to their industry + role combination — not a generic observation any VP could receive
c) A direct hook from their Education or a known challenge of their exact department
If none of the above is possible from the lead data, open with the case study challenge instead: describe what the client faced, not what the recipient is like.

DO NOT — these are hard rules, violating any one fails the email:
- Stack buzzwords like "governance-enabled", "CRM-ready", "data-driven digital transformation"
- Use the phrase "I'd love to..." or "I wanted to share..."
- Repeat the same sentence structure more than once
- Exceed 170 words

METRIC RULES:
- You may ONLY use a metric (%, number, timeframe) if it appears word-for-word in the STRATEGY BRIEF provided below.
- Copy metrics exactly as written — do not paraphrase numbers.
- If the strategy brief contains no explicit metric, describe the outcome qualitatively only: e.g. 'significant cost reduction', 'faster turnaround', 'improved data quality'. No invented numbers.
- If you use any metric not in the strategy brief, the email fails."""

    user_prompt = (
        f"LEAD PROFILE:\n{lead_text}\n\n"
        f"STRATEGY BRIEF:\n{strategy_brief}\n"
        f"{retry_context}"
        f"Write the email now."
    )

    email_draft = call_gpt(system_prompt, user_prompt)
    print(f"   [Email Drafter] Draft written (iteration {iterations + 1})")

    return {"email_draft": email_draft, "iterations": iterations + 1}


# =============================================================================
# Node 4: LLM Judge + routing function
# =============================================================================
JUDGE_RUBRIC = """
Score the email on each criterion. Be STRICT — a score of 8+ requires: a genuinely specific opener tied to real observed work, a metric that appears word-for-word in the provided excerpts, and a CTA with a specific day/time. Most drafts should score 5-7. A 9-10 should be rare.
A score of 7+ means the email could be sent as-is to a real executive.

1. Personalisation (0-3): Does the opening reference something specific about this
   person beyond their name/title/location? Generic "As a [Title] at [Company]..." scores 0-1.
   A truly personalised opener referencing their department, focus area, or a genuine
   observation about their work scores 2-3.
2. Case Study Relevance (0-3): Is the case study clearly relevant to this lead's
   industry/challenge? Is a concrete outcome mentioned? Does it use a readable name
   (NOT a raw filename like "exl_something")? Raw filenames in the email = score 0.
3. Naturalness (0-2): Does this read like a real human wrote it? Buzzword chains
   ("governance-enabled", "CRM-ready", "data-driven digital transformation"),
   formulaic structure, or AI-sounding phrases score 0.
4. CTA & Tone (0-2): Is there one clear low-friction CTA? Is the tone peer-to-peer
   and conversational, not salesy or robotic?

Total: 0-10. Output ONLY valid JSON:
{"score": <float>, "personalisation": <0-3>, "case_study_relevance": <0-3>, "naturalness": <0-2>, "cta_tone": <0-2>, "feedback": "<what to fix, or 'Approved' if >= 7>"}
"""


def judge_node(state: AgentState) -> dict:
    lead        = state["lead"]
    email_draft = state["email_draft"]
    briefs      = state.get("case_study_briefs", [])

    lead_text = "\n".join([f"{k}: {v}" for k, v in lead.items()])

    briefs_text = "\n\n".join([
        f"Case Study: {b['filename']}\nSection: {b['section']}\nContent: {b['summary']}"
        for b in briefs
    ]) if briefs else "No case study excerpts available."

    system_prompt = (
        f"You are a strict quality reviewer for B2B outreach emails.\n{JUDGE_RUBRIC}\n\n"
        f"RETRIEVED CASE STUDY EXCERPTS (the ONLY valid source of metrics):\n{briefs_text}\n\n"
        f"METRIC VALIDATION: Cross-check every number, percentage, and timeframe in the email against the "
        f"RETRIEVED CASE STUDY EXCERPTS above. If any metric in the email does not appear verbatim in the "
        f"excerpts, it is a hallucination. Set case_study_relevance to 0 and add "
        f"'HALLUCINATION DETECTED: [metric]' to feedback."
    )
    user_prompt   = (
        f"LEAD PROFILE:\n{lead_text}\n\n"
        f"EMAIL DRAFT:\n{email_draft}\n\n"
        f"Score this email now."
    )

    raw_output = call_gpt(system_prompt, user_prompt, model=AZURE_GPT_JUDGE_DEPLOYMENT)

    json_str = re.sub(r"```(?:json)?|```", "", raw_output).strip()
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        match  = re.search(r'"score"\s*:\s*([0-9.]+)', json_str)
        score  = float(match.group(1)) if match else 5.0
        result = {"score": score, "feedback": "Could not parse full rubric. Check email draft."}

    score    = float(result.get("score", 5.0))
    feedback = result.get("feedback", "")
    print(f"   [Judge] Score: {score}/10 | Feedback: {feedback}")

    return {"judge_score": score, "judge_feedback": feedback}


def should_retry(state: AgentState) -> str:
    score      = state.get("judge_score", 0.0)
    iterations = state.get("iterations", 0)
    if score >= SCORE_THRESHOLD or iterations >= MAX_ITERATIONS:
        return "end"
    return "retry"


# =============================================================================
# Build LangGraph pipeline
# =============================================================================
graph_builder = StateGraph(AgentState)

graph_builder.add_node("retrieval",     retrieval_node)
graph_builder.add_node("strategy",      strategy_node)
graph_builder.add_node("email_drafter", email_drafter_node)
graph_builder.add_node("judge",         judge_node)

graph_builder.set_entry_point("retrieval")
graph_builder.add_edge("retrieval",     "strategy")
graph_builder.add_edge("strategy",      "email_drafter")
graph_builder.add_edge("email_drafter", "judge")

graph_builder.add_conditional_edges(
    "judge",
    should_retry,
    {"retry": "email_drafter", "end": END},
)

pipeline = graph_builder.compile()
print("✅ LangGraph pipeline compiled")
print("   Flow: retrieval → strategy → email_drafter → judge → (retry | END)")


# =============================================================================
# Load leads
# =============================================================================
leads_df = pd.read_csv(LEADS_PATH)
_before = len(leads_df)
leads_df = leads_df.drop_duplicates(subset=["LinkedIn URL"], keep="first")
_dupes = _before - len(leads_df)
if _dupes > 0:
    print(f"⚠️  Removed {_dupes} duplicate lead(s)")

# Filter out invalid leads (Error/Unknown/empty names from failed scraper runs)
_invalid_values = {"error", "unknown", "not specified", ""}
_valid_mask = leads_df["Name"].apply(
    lambda x: str(x).strip().lower() not in _invalid_values and pd.notna(x)
)
_invalid_count = (~_valid_mask).sum()
if _invalid_count > 0:
    print(f"⚠️  Skipping {_invalid_count} invalid lead(s) (Error/Unknown/empty names)")
leads_df = leads_df[_valid_mask].reset_index(drop=True)

if leads_df.empty:
    print("❌ No valid leads to process. Re-run Web_Scrapper.py to refresh leads_final.csv.")
    exit(0)

print(f"✅ Loaded {len(leads_df)} valid leads")
print(leads_df.head().to_string())


# =============================================================================
# Run pipeline for each lead
# =============================================================================
results = []

for idx, row in leads_df.iterrows():
    lead = row.to_dict()
    print(f"\n{'='*60}")
    print(f"Processing lead {idx + 1}/{len(leads_df)}: {lead.get('Name', 'Unknown')} — {lead.get('Job Title', '')}")
    print(f"{'='*60}")

    initial_state: AgentState = {
        "lead":               lead,
        "case_study_briefs":  [],
        "strategy_brief":     "",
        "email_draft":        "",
        "judge_score":        0.0,
        "judge_feedback":     "",
        "iterations":         0,
    }

    try:
        final_state = pipeline.invoke(initial_state)
        results.append({
            "Name":           lead.get("Name", "Unknown"),
            "Job Title":      lead.get("Job Title", "Unknown"),
            "LinkedIn URL":   lead.get("LinkedIn URL", ""),
            "email_draft":    final_state["email_draft"],
            "judge_score":    final_state["judge_score"],
            "judge_feedback": final_state.get("judge_feedback", ""),
            "iterations":     final_state["iterations"],
        })
        print(f"\n✅ Done — Final score: {final_state['judge_score']}/10")
        print(f"   Iterations: {final_state['iterations']}")

    except Exception as e:
        print(f"❌ Error processing {lead.get('Name', 'Unknown')}: {e}")
        results.append({
            "Name":           lead.get("Name", "Unknown"),
            "Job Title":      lead.get("Job Title", "Unknown"),
            "LinkedIn URL":   lead.get("LinkedIn URL", ""),
            "email_draft":    f"ERROR: {str(e)}",
            "judge_score":    0.0,
            "judge_feedback": f"ERROR: {str(e)}",
            "iterations":     0,
        })

print(f"\n🎉 Pipeline complete — {len(results)} leads processed")


# =============================================================================
# Save results to CSV
# =============================================================================
results_df = pd.DataFrame(results)
results_df["needs_review"] = results_df["judge_score"] < MIN_SEND_SCORE

print(f"\nResults summary:")
print(f"  Total leads processed : {len(results_df)}")
print(f"  Average judge score   : {results_df['judge_score'].mean():.2f}/10")
print(f"  Leads scoring >= {SCORE_THRESHOLD}    : {(results_df['judge_score'] >= SCORE_THRESHOLD).sum()}")
print(f"  Leads needing review  : {results_df['needs_review'].sum()}")

# Full summary table: Name | Score | Iterations | Needs Review
summary = results_df[["Name", "judge_score", "iterations", "needs_review"]].sort_values("judge_score", ascending=False).copy()
summary["Status"] = summary.apply(
    lambda r: f"⚠️ WARNING — below {MIN_SEND_SCORE}" if r["needs_review"] else "✅ Ready to send",
    axis=1,
)
print(f"\n{'='*80}")
print("Full Summary Table")
print(f"{'='*80}")
print(summary[["Name", "judge_score", "iterations", "Status"]].to_string(index=False))

# Flag emails needing review with their judge feedback
flagged = results_df[results_df["needs_review"]].sort_values("judge_score", ascending=True)
if not flagged.empty:
    print(f"\n{'='*80}")
    print(f"⚠️ Emails flagged as needs_review ({len(flagged)} total)")
    print(f"{'='*80}")
    for _, row in flagged.iterrows():
        print(f"\n--- {row['Name']} ({row['Job Title']}) — Score: {row['judge_score']}/10 ---")
        print(f"Judge feedback: {row['judge_feedback']}")
        print("-" * 60)
        print(row["email_draft"])
        print("=" * 60)

# Top 2 highest scoring emails in full
if not results_df.empty:
    top2 = results_df.sort_values("judge_score", ascending=False).head(2)
    print(f"\n{'='*80}")
    print("Top 2 Highest Scoring Emails")
    print(f"{'='*80}")
    for rank, (_, sample) in enumerate(top2.iterrows(), 1):
        print(f"\n--- #{rank} — {sample['Name']} ({sample['Job Title']}) — Score: {sample['judge_score']}/10 ---")
        print("=" * 60)
        print(sample["email_draft"])
        print("=" * 60)

os.makedirs(PROJECT_ROOT / "outputs", exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Results saved to {OUTPUT_PATH}")

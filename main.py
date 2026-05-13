import os, json, re, io, csv, random, sqlite3
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# ── optional AI ──────────────────────────────────────────────────────────────
try:
    import openai
    OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
    AI_AVAILABLE = bool(OPENAI_KEY)
except ImportError:
    AI_AVAILABLE = False

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ── app setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="INDX Ops Copilot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── sqlite ────────────────────────────────────────────────────────────────────
DB = "copilot.db"

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT, vendor TEXT, description TEXT,
        category TEXT, amount REAL, department TEXT,
        transaction_type TEXT, month TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS invoices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vendor TEXT, invoice_date TEXT, invoice_number TEXT,
        total_amount REAL, due_date TEXT, currency TEXT, status TEXT
    )""")
    conn.commit()
    conn.close()

init_db()

# ── category rules ────────────────────────────────────────────────────────────
CATEGORY_RULES = {
    "Marketing":      ["google ads","meta","facebook ads","linkedin ads","instagram","tiktok ads","hubspot","mailchimp"],
    "Cloud & Software":["aws","amazon web","azure","openai","anthropic","github","notion","slack","zoom","figma","vercel","heroku"],
    "Travel":         ["grab","gojek","uber","lyft","taxi","mrt","bus","airbnb","hotel","airline","scoot","sia","airasia"],
    "Payroll":        ["payroll","salary","salaries","wages","cpf","employee"],
    "Contractors":    ["contractor","freelance","consulting","consultant","agency fee","outsource"],
    "Payment Fees":   ["stripe","paypal","braintree","payment gateway","transaction fee"],
    "Rent & Office":  ["rent","wework","the work project","regus","office","coworking"],
    "SaaS":           ["xero","quickbooks","asana","jira","monday","salesforce","hubspot crm","pipedrive"],
    "Legal & Compliance":["lawyer","law firm","audit","accounting fee","compliance","sgx","acra"],
    "Marketing Events":["event","conference","workshop","webinar","meetup","sponsorship"],
}

def auto_categorize(vendor: str, description: str) -> str:
    text = f"{vendor} {description}".lower()
    for cat, keywords in CATEGORY_RULES.items():
        if any(kw in text for kw in keywords):
            return cat
    return "Uncategorized"

# ── data cleaning ─────────────────────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    col_map = {}
    for c in df.columns:
        if "date" in c:        col_map[c] = "date"
        elif "vendor" in c or "supplier" in c: col_map[c] = "vendor"
        elif "desc" in c or "narr" in c:       col_map[c] = "description"
        elif "cat" in c:       col_map[c] = "category"
        elif "amount" in c or "value" in c or "total" in c: col_map[c] = "amount"
        elif "dept" in c or "department" in c: col_map[c] = "department"
        elif "type" in c:      col_map[c] = "transaction_type"
        elif "month" in c:     col_map[c] = "month"
    df = df.rename(columns=col_map)
    for req in ["date","vendor","amount"]:
        if req not in df.columns:
            df[req] = "" if req != "amount" else 0.0
    if "description" not in df.columns:  df["description"] = ""
    if "department" not in df.columns:   df["department"] = "General"
    if "transaction_type" not in df.columns: df["transaction_type"] = "expense"
    df["amount"] = pd.to_numeric(df["amount"].astype(str).str.replace(r"[,$S]","",regex=True), errors="coerce").fillna(0)
    df["vendor"] = df["vendor"].astype(str).str.strip().str.title()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.strftime("%Y-%m").fillna("Unknown")
    if "category" not in df.columns: df["category"] = ""
    mask = df["category"].isna() | (df["category"].astype(str).str.strip() == "")
    df.loc[mask, "category"] = df.loc[mask].apply(
        lambda r: auto_categorize(str(r.get("vendor","")), str(r.get("description",""))), axis=1
    )
    df = df.drop_duplicates(subset=["date","vendor","amount"], keep="first")
    df = df.dropna(subset=["amount"])
    df = df[df["amount"] != 0]
    return df.reset_index(drop=True)

# ── anomaly detection ─────────────────────────────────────────────────────────
def detect_anomalies(df: pd.DataFrame) -> list:
    anomalies = []
    expenses = df[df["transaction_type"] == "expense"].copy()

    # 1. duplicate vendors same amount same month
    dups = expenses.duplicated(subset=["vendor","amount","month"], keep=False)
    if dups.any():
        dup_rows = expenses[dups][["vendor","amount","month"]].drop_duplicates()
        for _, r in dup_rows.iterrows():
            anomalies.append({
                "severity": "high",
                "type": "Duplicate transaction",
                "description": f"Duplicate charge from {r['vendor']} — S${r['amount']:,.0f} in {r['month']}",
                "vendor": r["vendor"],
                "action": "Verify with vendor — possible double billing"
            })

    # 2. large single transaction (>3x category avg)
    for cat, grp in expenses.groupby("category"):
        mean_amt = grp["amount"].mean()
        for _, row in grp.iterrows():
            if row["amount"] > mean_amt * 3 and row["amount"] > 5000:
                anomalies.append({
                    "severity": "high",
                    "type": "Unusually large expense",
                    "description": f"{row['vendor']} charged S${row['amount']:,.0f} — {round(row['amount']/mean_amt,1)}x the {cat} average",
                    "vendor": row["vendor"],
                    "action": "Request invoice and approval trail"
                })

    # 3. category MoM spike >40%
    monthly_cat = expenses.groupby(["month","category"])["amount"].sum().reset_index()
    months = sorted(monthly_cat["month"].unique())
    if len(months) >= 2:
        for cat in monthly_cat["category"].unique():
            cat_data = monthly_cat[monthly_cat["category"]==cat].set_index("month")["amount"]
            for i in range(1, len(months)):
                prev, curr = months[i-1], months[i]
                if prev in cat_data.index and curr in cat_data.index:
                    pv, cv = cat_data[prev], cat_data[curr]
                    if pv > 0 and (cv - pv) / pv > 0.40:
                        pct = round((cv - pv)/pv * 100)
                        anomalies.append({
                            "severity": "medium",
                            "type": "Category spend spike",
                            "description": f"{cat} spend rose {pct}% from {prev} to {curr} (S${pv:,.0f} → S${cv:,.0f})",
                            "vendor": cat,
                            "action": f"Review {cat} invoices for {curr}"
                        })

    # 4. missing category
    uncat = expenses[expenses["category"] == "Uncategorized"]
    if len(uncat) > 0:
        anomalies.append({
            "severity": "low",
            "type": "Uncategorized transactions",
            "description": f"{len(uncat)} transactions could not be auto-categorized",
            "vendor": "Various",
            "action": "Manually review and assign categories"
        })

    # 5. negative revenue
    rev = df[df["transaction_type"] == "revenue"]
    neg_rev = rev[rev["amount"] < 0]
    if len(neg_rev):
        anomalies.append({
            "severity": "high",
            "type": "Negative revenue",
            "description": f"{len(neg_rev)} revenue entries have negative values — possible refunds or data errors",
            "vendor": "Multiple",
            "action": "Confirm these are intentional credits/refunds"
        })

    return anomalies[:12]

# ── KPI computation ───────────────────────────────────────────────────────────
def compute_kpis(df: pd.DataFrame) -> dict:
    revenue = df[df["transaction_type"]=="revenue"]["amount"].sum()
    expenses = df[df["transaction_type"]=="expense"]["amount"].sum()
    net = revenue - expenses
    months = df["month"].nunique() or 1
    burn = expenses / months
    runway = round((revenue * 0.3) / burn, 1) if burn > 0 else 0

    by_cat = df[df["transaction_type"]=="expense"].groupby("category")["amount"].sum().sort_values(ascending=False)
    top_cat = by_cat.index[0] if len(by_cat) else "N/A"
    top_cat_pct = round(by_cat.iloc[0] / expenses * 100) if expenses else 0

    monthly = df.groupby(["month","transaction_type"])["amount"].sum().unstack(fill_value=0).reset_index()
    monthly_list = []
    for _, r in monthly.iterrows():
        monthly_list.append({
            "month": r["month"],
            "revenue": round(r.get("revenue", 0)),
            "expenses": round(r.get("expense", 0))
        })

    by_vendor = df[df["transaction_type"]=="expense"].groupby("vendor")["amount"].sum().sort_values(ascending=False).head(5)
    top_vendors = [{"vendor": v, "amount": round(a)} for v, a in by_vendor.items()]

    by_dept = df[df["transaction_type"]=="expense"].groupby("department")["amount"].sum().sort_values(ascending=False)
    dept_spend = [{"department": d, "amount": round(a)} for d, a in by_dept.items()]

    cat_breakdown = [{"category": c, "amount": round(a)} for c, a in by_cat.items()]

    return {
        "revenue": round(revenue),
        "expenses": round(expenses),
        "net_profit": round(net),
        "burn_rate": round(burn),
        "runway_months": runway,
        "top_category": top_cat,
        "top_category_pct": top_cat_pct,
        "monthly_trend": monthly_list,
        "top_vendors": top_vendors,
        "dept_spend": dept_spend,
        "category_breakdown": cat_breakdown
    }

# ── rule-based insights (offline fallback) ────────────────────────────────────
def generate_rule_insights(kpis: dict, anomalies: list) -> dict:
    rev = kpis["revenue"]
    exp = kpis["expenses"]
    net = kpis["net_profit"]
    burn = kpis["burn_rate"]
    runway = kpis["runway_months"]
    top_cat = kpis["top_category"]
    top_pct = kpis["top_category_pct"]
    trend = kpis["monthly_trend"]
    high_anomalies = [a for a in anomalies if a["severity"] == "high"]

    # MoM revenue change
    mom_rev_str = ""
    if len(trend) >= 2:
        prev_rev = trend[-2]["revenue"]
        curr_rev = trend[-1]["revenue"]
        if prev_rev > 0:
            chg = round((curr_rev - prev_rev) / prev_rev * 100)
            mom_rev_str = f"Revenue {'grew' if chg > 0 else 'declined'} {abs(chg)}% month-on-month."

    # MoM expense change
    mom_exp_str = ""
    if len(trend) >= 2:
        prev_exp = trend[-2]["expenses"]
        curr_exp = trend[-1]["expenses"]
        if prev_exp > 0:
            chg = round((curr_exp - prev_exp) / prev_exp * 100)
            mom_exp_str = f"Expenses {'increased' if chg > 0 else 'decreased'} {abs(chg)}% in the latest month."

    margin = round(net / rev * 100) if rev > 0 else 0

    summary = (
        f"The business recorded S${rev:,} in revenue against S${exp:,} in expenses, "
        f"yielding a net {'profit' if net >= 0 else 'loss'} of S${abs(net):,} "
        f"({margin}% {'margin' if net >= 0 else 'loss rate'}). "
        f"{mom_rev_str} {mom_exp_str}"
    ).strip()

    concerns = []
    if runway < 9:
        concerns.append(f"Cash runway of {runway} months is below the recommended 12-month buffer — prioritize revenue acceleration or cost reduction.")
    if top_pct > 30:
        concerns.append(f"{top_cat} represents {top_pct}% of total spend — review whether this allocation is driving proportionate revenue growth.")
    if len(high_anomalies) > 0:
        concerns.append(f"{len(high_anomalies)} high-severity anomaly/anomalies detected including potential duplicate invoices — reconciliation required before month-end close.")
    if net < 0:
        concerns.append("The business is currently operating at a net loss. Review discretionary spend and validate revenue recognition.")

    actions = [
        f"Audit {top_cat} invoices for the most recent two months — this is your largest cost driver.",
        "Reconcile all flagged duplicate transactions before the next close cycle.",
        f"With a {runway}-month runway, prepare a 90-day cash flow forecast for founder review.",
        "Schedule a vendor review meeting — top 3 vendors account for a disproportionate share of spend.",
    ]
    if runway < 9:
        actions.insert(0, "Immediately model scenarios for extending runway — identify 3 cost lines that can be deferred or renegotiated.")

    variance = []
    for m in kpis["monthly_trend"]:
        if m["expenses"] > 0 and m["revenue"] > 0:
            ratio = round(m["expenses"] / m["revenue"] * 100)
            flag = "⚠" if ratio > 85 else "✓"
            variance.append(f"{flag} {m['month']}: expenses were {ratio}% of revenue (S${m['expenses']:,} / S${m['revenue']:,})")

    return {
        "executive_summary": summary,
        "concerns": concerns,
        "recommended_actions": actions,
        "variance_commentary": variance,
        "mode": "offline"
    }

# ── AI insights (live) ────────────────────────────────────────────────────────
def generate_ai_insights(kpis: dict, anomalies: list) -> dict:
    if not AI_AVAILABLE:
        return generate_rule_insights(kpis, anomalies)
    prompt = f"""You are a CFO-level advisor for a Singapore-based startup. Analyze this financial data and give concise, actionable insights.

KPIs: {json.dumps(kpis, indent=2)}
Anomalies detected: {json.dumps(anomalies, indent=2)}

Return a JSON object with exactly these keys:
- executive_summary: 2-3 sentence summary, CFO tone
- concerns: list of 3-4 specific operational concerns
- recommended_actions: list of 4 specific next actions
- variance_commentary: list of 3-4 month-level commentary strings
- mode: "ai"

Be specific, use numbers from the data, avoid generic advice."""

    try:
        client = openai.OpenAI(api_key=OPENAI_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            response_format={"type":"json_object"},
            max_tokens=800
        )
        result = json.loads(resp.choices[0].message.content)
        result["mode"] = "ai"
        return result
    except Exception as e:
        fallback = generate_rule_insights(kpis, anomalies)
        fallback["mode"] = "offline"
        return fallback

# ── Q&A logic ─────────────────────────────────────────────────────────────────
def answer_question_rule(question: str, kpis: dict, anomalies: list) -> str:
    q = question.lower()
    trend = kpis.get("monthly_trend", [])
    top_vendors = kpis.get("top_vendors", [])
    top_cat = kpis.get("top_category", "N/A")
    runway = kpis.get("runway_months", 0)
    dept = kpis.get("dept_spend", [])

    if "runway" in q or "cash" in q:
        return (f"Current cash runway is estimated at {runway} months based on average burn of "
                f"S${kpis['burn_rate']:,}/month. This is {'below' if runway < 12 else 'within'} the "
                f"recommended 12-month buffer. To extend runway, consider auditing the top 3 expense "
                f"categories and renegotiating any annual contracts up for renewal.")

    if "vendor" in q or "supplier" in q:
        top = ", ".join([f"{v['vendor']} (S${v['amount']:,})" for v in top_vendors[:3]])
        return (f"Your top 3 vendors by spend are: {top}. "
                f"These vendors should be prioritized for contract reviews and payment term negotiations.")

    if "expense" in q and ("increas" in q or "why" in q or "spike" in q):
        spikes = [a for a in anomalies if "spike" in a["type"].lower()]
        if spikes:
            return "Expense increases are driven by: " + "; ".join([a["description"] for a in spikes[:3]])
        return (f"{top_cat} is your largest cost category. "
                f"Review the most recent 2 months of {top_cat} invoices for discretionary spend that can be deferred.")

    if "month" in q and ("close" in q or "reconcil" in q):
        high = [a for a in anomalies if a["severity"] == "high"]
        if high:
            items = "\n".join([f"• {a['description']}" for a in high])
            return f"Before month-end close, prioritize resolving these {len(high)} high-severity issues:\n{items}"
        return "No critical issues flagged. Month-end close looks clean — verify all vendor invoices are matched and submitted."

    if "department" in q or "dept" in q or "overspend" in q:
        if dept:
            top_dept = dept[0]
            return (f"Highest spending department is {top_dept['department']} at S${top_dept['amount']:,}. "
                    f"Compare against approved budget to identify overspend. "
                    f"{'Consider a department spend review.' if top_dept['amount'] > 50000 else 'Spend levels appear within normal range.'}")
        return "No department data available."

    if "invoice" in q or "duplicate" in q or "suspicious" in q:
        dups = [a for a in anomalies if "duplicate" in a["type"].lower()]
        large = [a for a in anomalies if "large" in a["type"].lower()]
        if dups or large:
            items = [a["description"] for a in (dups + large)[:3]]
            return "Suspicious invoice activity detected:\n" + "\n".join(f"• {i}" for i in items)
        return "No suspicious invoices detected in the current dataset."

    if "profit" in q or "margin" in q:
        rev, exp, net = kpis["revenue"], kpis["expenses"], kpis["net_profit"]
        margin = round(net/rev*100) if rev > 0 else 0
        return (f"Net profit is S${net:,} on S${rev:,} revenue — a {margin}% margin. "
                f"{'This is healthy for an early-stage company.' if margin > 15 else 'Margin is thin — identify cost reduction opportunities before the next funding round.'}")

    return (f"Based on the uploaded data: revenue is S${kpis['revenue']:,}, expenses are S${kpis['expenses']:,}, "
            f"and runway is {runway} months. Top cost driver is {top_cat}. "
            f"For more specific analysis, try asking about vendors, departments, anomalies, or month-end close.")

# ── sample data generator ─────────────────────────────────────────────────────
def generate_sample_data() -> list:
    random.seed(42)
    rows = []
    base = datetime(2024, 1, 1)

    monthly_revenue = [85000, 92000, 88000, 97000, 109000, 116000]
    monthly_payroll  = [28000, 28000, 28000, 31000, 31000, 31000]

    for m in range(6):
        month_start = base + timedelta(days=m*30)
        month_str = month_start.strftime("%Y-%m")
        date_str  = month_start.strftime("%Y-%m-%d")

        rows.append({"date":date_str,"vendor":"Client Revenue","description":"Monthly recurring revenue",
                     "category":"Revenue","amount":monthly_revenue[m],"department":"Sales",
                     "transaction_type":"revenue","month":month_str})

        rows.append({"date":date_str,"vendor":"Payroll","description":"Staff salaries + CPF",
                     "category":"Payroll","amount":monthly_payroll[m],"department":"HR",
                     "transaction_type":"expense","month":month_str})

        rows.append({"date":date_str,"vendor":"WeWork","description":"Office rent",
                     "category":"Rent & Office","amount":4500,"department":"Operations",
                     "transaction_type":"expense","month":month_str})

        rows.append({"date":date_str,"vendor":"Amazon Web Services","description":"Cloud infrastructure",
                     "category":"Cloud & Software","amount":round(random.uniform(2200,3800)),
                     "department":"Engineering","transaction_type":"expense","month":month_str})

        rows.append({"date":date_str,"vendor":"Notion","description":"Team workspace subscription",
                     "category":"Cloud & Software","amount":180,"department":"Operations",
                     "transaction_type":"expense","month":month_str})

        mktg_base = [8000, 9500, 9000, 11000, 18500, 22000]
        rows.append({"date":date_str,"vendor":"Meta Ads","description":"Facebook & Instagram campaigns",
                     "category":"Marketing","amount":mktg_base[m],"department":"Marketing",
                     "transaction_type":"expense","month":month_str})

        rows.append({"date":date_str,"vendor":"Google Ads","description":"Search and display advertising",
                     "category":"Marketing","amount":round(random.uniform(3000,5500)),
                     "department":"Marketing","transaction_type":"expense","month":month_str})

        rows.append({"date":date_str,"vendor":"Grab","description":"Business travel",
                     "category":"Travel","amount":round(random.uniform(300,700)),
                     "department":"Sales","transaction_type":"expense","month":month_str})

        rows.append({"date":date_str,"vendor":"Stripe","description":"Payment processing fees",
                     "category":"Payment Fees","amount":round(monthly_revenue[m]*0.029),
                     "department":"Finance","transaction_type":"expense","month":month_str})

    # anomalies
    rows.append({"date":"2024-05-14","vendor":"Meta Ads","description":"Facebook & Instagram campaigns - duplicate",
                 "category":"Marketing","amount":18500,"department":"Marketing",
                 "transaction_type":"expense","month":"2024-05"})

    rows.append({"date":"2024-06-03","vendor":"Unnamed Vendor","description":"Consulting services",
                 "category":"","amount":15000,"department":"Operations",
                 "transaction_type":"expense","month":"2024-06"})

    rows.append({"date":"2024-04-22","vendor":"Singapore Airlines","description":"Business class flights — Tokyo offsite",
                 "category":"Travel","amount":12800,"department":"Sales",
                 "transaction_type":"expense","month":"2024-04"})

    return rows

# ── state store (in-memory for demo) ─────────────────────────────────────────
_store = {"df": None, "kpis": None, "anomalies": None, "invoices": []}

# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/api/load-sample")
async def load_sample():
    rows = generate_sample_data()
    df = pd.DataFrame(rows)
    df = clean_dataframe(df)
    _store["df"] = df
    _store["kpis"] = compute_kpis(df)
    _store["anomalies"] = detect_anomalies(df)
    preview = df.head(20).fillna("").to_dict(orient="records")
    return {"rows": len(df), "preview": preview, "message": f"Loaded {len(df)} transactions across 6 months"}

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    df = clean_dataframe(df)
    _store["df"] = df
    _store["kpis"] = compute_kpis(df)
    _store["anomalies"] = detect_anomalies(df)
    preview = df.head(20).fillna("").to_dict(orient="records")
    return {"rows": len(df), "preview": preview}

@app.get("/api/kpis")
async def get_kpis():
    if _store["kpis"] is None:
        raise HTTPException(400, "No data loaded yet")
    return _store["kpis"]

@app.get("/api/anomalies")
async def get_anomalies():
    if _store["anomalies"] is None:
        raise HTTPException(400, "No data loaded yet")
    return _store["anomalies"]

@app.post("/api/insights")
async def get_insights(payload: dict = {}):
    if _store["kpis"] is None:
        raise HTTPException(400, "No data loaded yet")
    mode = payload.get("mode", "offline")
    if mode == "ai":
        result = generate_ai_insights(_store["kpis"], _store["anomalies"] or [])
    else:
        result = generate_rule_insights(_store["kpis"], _store["anomalies"] or [])
    return result

class QARequest(BaseModel):
    question: str
    mode: Optional[str] = "offline"

@app.post("/api/qa")
async def qa(req: QARequest):
    if _store["kpis"] is None:
        return {"answer": "Please load data first by clicking 'Load sample data' on the Data tab."}
    if req.mode == "ai" and AI_AVAILABLE:
        prompt = (f"You are a CFO advisor for a startup. Answer this question using only the data provided.\n"
                  f"KPIs: {json.dumps(_store['kpis'])}\n"
                  f"Anomalies: {json.dumps(_store['anomalies'])}\n"
                  f"Question: {req.question}\n"
                  f"Answer concisely in 2-4 sentences. Use specific numbers. Do not hallucinate data.")
        try:
            client = openai.OpenAI(api_key=OPENAI_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=300
            )
            return {"answer": resp.choices[0].message.content}
        except:
            pass
    answer = answer_question_rule(req.question, _store["kpis"], _store["anomalies"] or [])
    return {"answer": answer}

@app.post("/api/load-sample-invoices")
async def load_sample_invoices():
    invoices = [
        {"vendor":"Meta Platforms","invoice_date":"2024-06-01","invoice_number":"META-2024-0612",
         "total_amount":18500,"due_date":"2024-06-30","currency":"SGD","status":"flagged"},
        {"vendor":"Meta Platforms","invoice_date":"2024-05-14","invoice_number":"META-2024-0589",
         "total_amount":18500,"due_date":"2024-06-13","currency":"SGD","status":"duplicate"},
        {"vendor":"Amazon Web Services","invoice_date":"2024-06-01","invoice_number":"AWS-9182736",
         "total_amount":3412,"due_date":"2024-06-30","currency":"USD","status":"ok"},
        {"vendor":"WeWork Singapore","invoice_date":"2024-06-01","invoice_number":"WW-SG-4421",
         "total_amount":4500,"due_date":"2024-06-15","currency":"SGD","status":"ok"},
        {"vendor":"Unnamed Vendor","invoice_date":"2024-06-03","invoice_number":"INV-0001",
         "total_amount":15000,"due_date":"2024-06-17","currency":"SGD","status":"flagged"},
        {"vendor":"Singapore Airlines","invoice_date":"2024-04-20","invoice_number":"SIA-8822991",
         "total_amount":12800,"due_date":"2024-05-20","currency":"SGD","status":"flagged"},
    ]
    _store["invoices"] = invoices
    return {"invoices": invoices}

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not PDF_AVAILABLE:
        return {"extracted": {"vendor":"PDF library not installed","invoice_date":"—",
                              "invoice_number":"—","total_amount":0,"due_date":"—","currency":"SGD","status":"ok"},
                "note":"Install pdfplumber for real PDF extraction"}
    content = await file.read()
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        vendor = re.search(r"(?:from|vendor|supplier|bill to)[:\s]+([A-Z][^\n]+)", text, re.I)
        amount = re.search(r"(?:total|amount due|grand total)[:\s]+[\$S]?([\d,]+\.?\d*)", text, re.I)
        inv_num = re.search(r"(?:invoice\s*#?|inv\s*#?)[:\s]*([A-Z0-9\-]+)", text, re.I)
        date    = re.search(r"(?:invoice date|date)[:\s]+(\d{1,2}[\s/\-]\w+[\s/\-]\d{2,4})", text, re.I)
        due     = re.search(r"(?:due date|payment due)[:\s]+(\d{1,2}[\s/\-]\w+[\s/\-]\d{2,4})", text, re.I)
        extracted = {
            "vendor": vendor.group(1).strip() if vendor else "Unknown",
            "invoice_date": date.group(1) if date else "Not found",
            "invoice_number": inv_num.group(1) if inv_num else "Not found",
            "total_amount": float(amount.group(1).replace(",","")) if amount else 0,
            "due_date": due.group(1) if due else "Not found",
            "currency": "SGD",
            "status": "ok"
        }
        _store["invoices"].append(extracted)
        return {"extracted": extracted}
    except Exception as e:
        raise HTTPException(500, f"PDF extraction failed: {e}")

@app.get("/api/status")
async def status():
    return {
        "data_loaded": _store["df"] is not None,
        "ai_available": AI_AVAILABLE,
        "pdf_available": PDF_AVAILABLE,
        "row_count": len(_store["df"]) if _store["df"] is not None else 0
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")

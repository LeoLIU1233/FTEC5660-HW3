#!/usr/bin/env python3
"""FTEC5660 HW3 student starter: an agent that applies 3 of Fisher's 15 points using tools."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sqlite3
import threading
import zlib
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


MODEL_NAME = "deepseek-v4-flash"
INTERVIEW_LIMIT = 10
MAX_SQL_ROWS = 50

# Philip Fisher, "Common Stocks and Uncommon Profits", Part One, Chapter 3.
# For every point, PASS means the company looks good on it. You choose 3
# points and decide yourself what PASS means for each of them.
FISHER_POINTS = {
    1: "Does the company have products or services with sufficient market potential to make possible a sizable increase in sales for at least several years?",
    2: "Does the management have a determination to continue to develop products or processes that will still further increase total sales potentials when the growth potentials of currently attractive product lines have largely been exploited?",
    3: "How effective are the company's research and development efforts in relation to its size?",
    4: "Does the company have an above-average sales organization?",
    5: "Does the company have a worthwhile profit margin?",
    6: "What is the company doing to maintain or improve profit margins?",
    7: "Does the company have outstanding labor and personnel relations?",
    8: "Does the company have outstanding executive relations?",
    9: "Does the company have depth to its management?",
    10: "How good are the company's cost analysis and accounting controls?",
    11: "Are there other aspects of the business, somewhat peculiar to the industry involved, which will give the investor important clues as to how outstanding the company may be in relation to its competition?",
    12: "Does the company have a short-range or long-range outlook in regard to profits?",
    13: "In the foreseeable future will the growth of the company require sufficient equity financing so that the larger number of shares then outstanding will largely cancel the existing stockholders' benefit from this anticipated growth?",
    14: "Does the management talk freely to investors about its affairs when things are going well but 'clam up' when troubles and disappointments occur?",
    15: "Does the company have a management of unquestionable integrity?",
}
VERDICTS = ("PASS", "FAIL")


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


MAX_CONCURRENT_COMPANIES = 4

FINANCIAL_COLUMNS = (
    "year",
    "revenue",
    "revenue_rows",
    "valid_revenue_rows",
    "cost_of_sales",
    "selling_admin",
    "research_development",
    "cost_rows",
    "valid_cost_rows",
    "cost_months",
    "company_industry",
    "median_industry",
    "median_matches",
    "industry_margin_pct",
)

FINANCIAL_QUERY = """
WITH years(year) AS (VALUES ('2024'), ('2025')),
revenue AS (
    SELECT SUBSTR(TRIM(order_date),1,4) AS year,
    SUM(amount) AS revenue, COUNT(*) AS revenue_rows,
    SUM(CASE WHEN TYPEOF(amount) IN ('integer','real') THEN 1 ELSE 0 END)
    AS valid_revenue_rows
    FROM orders
    WHERE SUBSTR(TRIM(order_date),1,4) IN ('2024','2025')
    GROUP BY SUBSTR(TRIM(order_date),1,4)
),
cost_entries AS (
    SELECT SUBSTR(TRIM(month),1,4) AS year,
    cost_of_sales, selling_admin, research_development,
    CASE WHEN SUBSTR(TRIM(month),1,7)
    GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]'
    AND SUBSTR(TRIM(month),6,2) BETWEEN '01' AND '12'
    THEN SUBSTR(TRIM(month),1,7) END AS calendar_month
    FROM monthly_costs
    WHERE SUBSTR(TRIM(month),1,4) IN ('2024','2025')
),
costs AS (
    SELECT year, SUM(cost_of_sales) AS cost_of_sales,
    SUM(selling_admin) AS selling_admin,
    SUM(research_development) AS research_development,
    COUNT(*) AS cost_rows,
    COUNT(DISTINCT calendar_month) AS cost_months,
    SUM(CASE WHEN calendar_month IS NOT NULL
    AND TYPEOF(cost_of_sales) IN ('integer','real')
    AND TYPEOF(selling_admin) IN ('integer','real')
    AND TYPEOF(research_development) IN ('integer','real')
    THEN 1 ELSE 0 END) AS valid_cost_rows
    FROM cost_entries GROUP BY year
),
peers AS (
    SELECT MIN(c.industry) AS company_industry, MIN(i.industry) AS median_industry,
    COUNT(DISTINCT CASE WHEN TYPEOF(i.operating_margin_pct) IN ('integer','real')
    THEN i.operating_margin_pct END) AS median_matches,
    MIN(CASE WHEN TYPEOF(i.operating_margin_pct) IN ('integer','real')
    THEN i.operating_margin_pct END) AS industry_margin_pct
    FROM company AS c LEFT JOIN industry_medians AS i
    ON LOWER(TRIM(c.industry)) = LOWER(TRIM(i.industry))
)
SELECT y.year,
CASE WHEN r.revenue IS NOT NULL THEN PRINTF('%.17g', r.revenue) END AS revenue,
COALESCE(r.revenue_rows,0) AS revenue_rows,
COALESCE(r.valid_revenue_rows,0) AS valid_revenue_rows,
CASE WHEN k.cost_of_sales IS NOT NULL
THEN PRINTF('%.17g', k.cost_of_sales) END AS cost_of_sales,
CASE WHEN k.selling_admin IS NOT NULL
THEN PRINTF('%.17g', k.selling_admin) END AS selling_admin,
CASE WHEN k.research_development IS NOT NULL
THEN PRINTF('%.17g', k.research_development) END AS research_development,
COALESCE(k.cost_rows,0) AS cost_rows,
COALESCE(k.valid_cost_rows,0) AS valid_cost_rows,
COALESCE(k.cost_months,0) AS cost_months,
p.company_industry, p.median_industry, p.median_matches,
CASE WHEN p.industry_margin_pct IS NOT NULL
THEN PRINTF('%.17g', p.industry_margin_pct) END AS industry_margin_pct
FROM years AS y LEFT JOIN revenue AS r ON r.year = y.year
LEFT JOIN costs AS k ON k.year = y.year CROSS JOIN peers AS p
ORDER BY y.year
""".strip()


def _parse_financials(text: str) -> dict[int, dict[str, Any]]:
    """Parse the full-precision financial snapshot returned by sql_query."""
    lines = text.strip().splitlines()
    if len(lines) != 3:
        raise ValueError("Financial snapshot must contain a header and two year rows")
    header = tuple(value.strip() for value in lines[0].split("|"))
    if header != FINANCIAL_COLUMNS:
        raise ValueError("Unexpected financial snapshot header")

    amount_columns = (
        "revenue",
        "cost_of_sales",
        "selling_admin",
        "research_development",
        "industry_margin_pct",
    )
    count_columns = (
        "revenue_rows",
        "valid_revenue_rows",
        "cost_rows",
        "valid_cost_rows",
        "cost_months",
        "median_matches",
    )
    rows: dict[int, dict[str, Any]] = {}
    for line in lines[1:]:
        values = [value.strip() for value in line.split("|")]
        if len(values) != len(FINANCIAL_COLUMNS):
            raise ValueError("Malformed financial snapshot row")
        row: dict[str, Any] = dict(zip(FINANCIAL_COLUMNS, values))
        try:
            year = int(row["year"])
            if year not in (2024, 2025) or year in rows:
                raise ValueError("Financial snapshot years must be exactly 2024 and 2025")
            for key in count_columns:
                row[key] = int(row[key])
                if row[key] < 0:
                    raise ValueError("Financial row counts cannot be negative")
            for key in amount_columns:
                row[key] = None if row[key] in ("None", "NULL", "") else Decimal(row[key])
                if row[key] is not None and not row[key].is_finite():
                    raise ValueError("Financial amounts must be finite")
        except (InvalidOperation, TypeError) as exc:
            raise ValueError("Invalid numeric financial evidence") from exc
        rows[year] = row
    return rows


def _annual_gaps(year: int, row: dict[str, Any]) -> list[str]:
    """Return reasons why an annual operating margin cannot be trusted."""
    gaps = []
    if row["revenue"] is None or row["revenue_rows"] == 0:
        gaps.append(f"FY{year} revenue is missing")
    elif row["revenue"] <= 0:
        gaps.append(f"FY{year} revenue is not positive")
    if row["valid_revenue_rows"] != row["revenue_rows"]:
        gaps.append(
            f"FY{year} valid revenue rows "
            f"{row['valid_revenue_rows']}/{row['revenue_rows']}"
        )
    if row["cost_months"] != 12:
        gaps.append(f"FY{year} valid cost months {row['cost_months']}/12")
    if row["valid_cost_rows"] != row["cost_rows"] or row["cost_rows"] < 12:
        gaps.append(
            f"FY{year} complete cost rows "
            f"{row['valid_cost_rows']}/{row['cost_rows']}"
        )
    if any(
        row[key] is None
        for key in ("cost_of_sales", "selling_admin", "research_development")
    ):
        gaps.append(f"FY{year} a cost-category total is missing")
    return gaps


def _financial_assessment(text: str) -> tuple[str, str, str, str]:
    """Compute Points 5 and 6 from tool-sourced financial data."""
    rows = _parse_financials(text)
    gaps = {year: _annual_gaps(year, row) for year, row in rows.items()}
    margins: dict[int, Decimal] = {}
    for year, row in rows.items():
        if gaps[year]:
            continue
        total_cost = sum(
            row[key]
            for key in ("cost_of_sales", "selling_admin", "research_development")
        )
        margins[year] = Decimal(100) * (row["revenue"] - total_cost) / row["revenue"]

    current = rows[2025]
    median = current["industry_margin_pct"]
    industry = current["company_industry"].strip()
    median_industry = current["median_industry"].strip()
    industry_matches = (
        current["median_matches"] == 1
        and median is not None
        and industry not in ("", "None", "NULL")
        and industry.casefold() == median_industry.casefold()
    )
    if 2025 not in margins:
        verdict5 = "FAIL"
        reason5 = "complete FY2025 operating margin is unavailable"
    elif not industry_matches:
        verdict5 = "FAIL"
        reason5 = "the company's own-industry median is missing or ambiguous"
    else:
        verdict5 = (
            "PASS"
            if margins[2025] > 0 and margins[2025] >= median
            else "FAIL"
        )
        reason5 = (
            f"FY2025 operating margin {margins[2025]:.2f}% versus "
            f"own-industry median {median:.2f}%"
        )

    if 2024 in margins and 2025 in margins:
        verdict6 = "PASS" if margins[2025] >= margins[2024] else "FAIL"
        reason6 = (
            f"FY2024 operating margin {margins[2024]:.2f}% to FY2025 "
            f"{margins[2025]:.2f}%"
        )
    else:
        verdict6 = "FAIL"
        missing = gaps.get(2024, []) + gaps.get(2025, [])
        reason6 = "complete FY2024 and FY2025 operating margins are unavailable"
        if missing:
            reason6 += " (" + "; ".join(missing) + ")"

    return verdict5, verdict6, reason5, reason6


def _parse_json_object(content: Any) -> dict[str, Any]:
    """Extract the JSON object returned by the semantic classifier."""
    if isinstance(content, list):
        content = "\n".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, (str, dict))
        )
    if not isinstance(content, str):
        raise ValueError("model did not return text")
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("model did not return a JSON object")
    return result


def _is_informed_interview(text: str) -> bool:
    """Exclude unsuccessful or cannot-comment interview responses."""
    if not isinstance(text, str) or not text.strip():
        return False
    lowered = text.lower()
    if text.startswith(
        ("Unknown role", "Unknown topic", "Interview limit", "Tool call failed")
    ):
        return False
    return not any(
        marker in lowered
        for marker in ("can't comment", "cannot comment", "can not comment")
    )


def _keyword_labour_verdict(interviews: list[str]) -> str:
    """Deterministic Point 7 fallback when the model call is unavailable."""
    if not interviews:
        return "FAIL"
    joined = " ".join("\n".join(interviews).lower().split())

    signal_groups = (
        (
            4,
            r"\b(?:toxic culture|hostile work environment|harass\w*|"
            r"discriminat\w*|strike threat|mass layoffs?|unsafe workplace)\b",
        ),
        (
            3,
            r"\b(?:constant overtime|endless overtime|constantly overworked|"
            r"overtime is constant|overworked|burnout)\b",
        ),
        (
            2,
            r"\b(?:understaffed|long hours|skip(?:ped|ping)? breaks|"
            r"pressure(?:d)? to|exhausted)\b",
        ),
        (
            2,
            r"\b(?:resign(?:ed|s|ing|ation|ations)?|quitting?|quit|"
            r"attrition|monthly departures?|"
            r"monthly resignations?|left (?:after|because|the company)|"
            r"leave (?:the company|because|for))\b",
        ),
        (
            2,
            r"\b(?:unfair treatment|management (?:does not care|ignores)|"
            r"managers? (?:dismiss|ignore)|complaints? (?:are |were )?"
            r"(?:routinely )?ignored|safety concerns? .*?"
            r"(?:dismissed|ignored)|conditions? .*?unsafe)\b",
        ),
        (
            1,
            r"\b(?:serious dissatisfaction|dissatisfied|unhappy|low morale|"
            r"morale (?:is|was) low|stressful|chronic morale issues|"
            r"bad workplace|bad culture|people are leaving)\b",
        ),
    )
    weights = [
        weight
        for weight, pattern in signal_groups
        if re.search(pattern, joined, re.IGNORECASE)
    ]
    if len(weights) >= 2 or sum(weights) >= 3:
        return "FAIL"
    return "PASS"


def _labour_fallback(
    interviews: list[tuple[str, str]],
) -> tuple[str, str]:
    texts = [text for _, text in interviews]
    verdict = _keyword_labour_verdict(texts)
    if not interviews:
        reason = (
            "no informed employee testimony was available; "
            "conservative FAIL because PASS requires positive evidence"
        )
    elif verdict == "FAIL":
        reason = (
            "employee testimony indicates persistent or serious "
            "labour-relations problems"
        )
    else:
        reason = "informed employee testimony supports healthy labour relations"
    return verdict, reason


async def _assess_labour_with_llm(
    agent: Any,
    interviews: list[tuple[str, str]],
) -> tuple[str, str]:
    """Use DeepSeek as a semantic classifier for Point 7."""
    if agent is None or not interviews:
        return _labour_fallback(interviews)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:
        return _labour_fallback(interviews)

    evidence = "\n\n".join(f"[{role}] {text}" for role, text in interviews)
    system = (
        "You are a conservative Fisher analyst assessing Point 7 only: "
        "outstanding labour and personnel relations. Use only the supplied "
        "employee interview evidence. PASS when informed testimony supports "
        "fair treatment, reasonable working conditions and healthy labour "
        "relations without credible material contradiction. FAIL for "
        "persistently poor treatment, serious dissatisfaction or dysfunctional "
        "relations. Do not use management slogans. Return ONLY JSON with keys "
        "verdict, reason, quote. verdict must be PASS or FAIL; reason is one "
        "short sentence; quote must be an exact substring from one supplied "
        "interview, or an empty string."
    )
    human = f"Employee interview evidence:\n\n{evidence}"
    try:
        response = await asyncio.wait_for(
            agent.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=human)]
            ),
            timeout=75,
        )
        result = _parse_json_object(response.content)
        verdict = str(result.get("verdict", "")).strip().upper()
        if verdict not in VERDICTS:
            raise ValueError("invalid verdict")
        reason = str(result.get("reason", "")).strip()
        if not reason:
            reason = (
                "informed employee testimony was assessed against Fisher Point 7"
            )
        return verdict, reason
    except Exception:
        return _labour_fallback(interviews)


def _sentence(text: str) -> str:
    return text.strip().rstrip(".") + "."


async def _analyze_one_company(
    agent: Any,
    tools: list[Any],
) -> dict[str, Any]:
    """Run Points 5, 6 and 7 for one company through its four LangChain tools."""
    tool_map = {tool.name: tool for tool in tools}

    try:
        finance_text = str(
            await tool_map["sql_query"].ainvoke({"query": FINANCIAL_QUERY})
        )
        verdict5, verdict6, reason5, reason6 = _financial_assessment(finance_text)
    except Exception as exc:
        verdict5 = verdict6 = "FAIL"
        reason5 = reason6 = (
            f"financial data unavailable ({type(exc).__name__})"
        )

    interviews: list[tuple[str, str]] = []
    for role in ("current_employee", "former_employee"):
        try:
            text = str(
                await tool_map["interview"].ainvoke(
                    {"role": role, "topic": "working_conditions"}
                )
            )
        except Exception:
            continue
        if _is_informed_interview(text):
            interviews.append((role, text))

    verdict7, reason7 = await _assess_labour_with_llm(agent, interviews)
    summary = " ".join(
        (
            _sentence(f"Point 5 {verdict5}: {reason5}"),
            _sentence(f"Point 6 {verdict6}: {reason6}"),
            _sentence(f"Point 7 {verdict7}: {reason7}"),
        )
    )
    return {
        "points": {5: verdict5, 6: verdict6, 7: verdict7},
        "summary": summary,
    }


# Choose any 3 of the 15 points, e.g. CHOSEN_POINTS = [1, 7, 15]. Your agent
# gives a PASS or FAIL verdict on these 3 points only, with your own rule for each.
CHOSEN_POINTS: list[int] = [5, 6, 7]


def build_agent() -> Any:
    """Create and return your model or agent once.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env. Each company has its own tools (see ``analyze_companies``), so
    you will probably create one tool-calling agent per company there.
    """
    ### YOUR CODE HERE
    load_env_file()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return None
    try:
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=MODEL_NAME,
            temperature=0,
            max_retries=2,
        )
    except Exception:
        return None


async def analyze_companies(agent: Any, companies: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    """Run your agent and return its verdicts on CHOSEN_POINTS for every company.

    ``companies`` maps each company folder name to its four LangChain tools:

        list_documents()             names and dates of the company's documents
        read_document(name)          the text of one document
        sql_query(query)             one read-only SQLite query on company.db
        interview(role, topic)       one interview answer (INTERVIEW_LIMIT per company)

    Give the tools to a tool-calling agent, e.g.
    ``create_agent(model, tools=companies[name], system_prompt=...)``, or call
    them yourself with ``tool.invoke({...})``. The README explains the
    documents, the database tables and the interview roles and topics.

    Return one dictionary per company folder name, e.g. with
    CHOSEN_POINTS = [1, 7, 15]:

        {
            "hollins_group": {
                "points": {1: "PASS", 7: "FAIL", 15: "PASS"},
                "summary": "Two or three sentences explaining the verdicts.",
            },
            ...
        }

    Each chosen point needs a verdict ("PASS" or "FAIL"; True/False also
    works). Other points are ignored.

    Catch errors per company (e.g. a failed API call) and still return an
    entry for it: an exception here means no results.csv, which scores zero.
    You may analyse companies in parallel (e.g. ``asyncio.gather``).
    """
    ### YOUR CODE HERE
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_COMPANIES)

    async def worker(
        name: str,
        company_tools: list[Any],
    ) -> tuple[str, dict[str, Any]]:
        async with semaphore:
            try:
                result = await _analyze_one_company(agent, company_tools)
            except Exception as exc:
                result = {
                    "points": {5: "FAIL", 6: "FAIL", 7: "FAIL"},
                    "summary": (
                        f"Company analysis failed with {type(exc).__name__}; "
                        "conservative FAIL fallback used."
                    ),
                }
            return name, result

    pairs = await asyncio.gather(
        *(worker(name, company_tools) for name, company_tools in companies.items())
    )
    return dict(pairs)


# Everything below is provided runner/scoring code. No edits are needed.
# Your code must read company data only through these tools: do not open
# company.db or interviews.dat directly.

INTERVIEWS_USED: dict[str, int] = {}


def company_folders(folder: Path) -> list[Path]:
    """Return one sub-folder per company directly inside *folder*, sorted by name."""
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))


def make_tools(company_folder: Path) -> list[Any]:
    """Build the four tools for one company."""
    from langchain_core.tools import tool

    name = company_folder.name
    db_uri = f"{(company_folder / 'company.db').resolve().as_uri()}?mode=ro"
    answers = json.loads(zlib.decompress((company_folder / "interviews.dat").read_bytes()))
    documents = {p.name: p for p in sorted(company_folder.glob("*.md"))}
    lock = threading.Lock()
    INTERVIEWS_USED[name] = 0

    @tool
    def list_documents() -> str:
        """List this company's documents (annual letter, earnings calls, news) with their dates."""
        lines = []
        for doc_name, path in documents.items():
            text = path.read_text(encoding="utf-8").splitlines()
            dated = next((line for line in text if line.startswith("Date:")), "")
            lines.append(f"{doc_name}: {text[0].lstrip('# ')} ({dated or 'undated'})")
        return "\n".join(lines)

    @tool
    def read_document(name: str) -> str:
        """Return the full text of one document named by list_documents, e.g. 'annual_letter_2025.md'."""
        path = documents.get(name.strip())
        if path is None:
            return f"No document named {name!r}. Available: {', '.join(documents)}"
        return path.read_text(encoding="utf-8")

    @tool
    def sql_query(query: str) -> str:
        """Run one read-only SQLite query on this company's database and return at most 50 rows.

        Amounts are in USD thousands. Dates are 'YYYY-MM-DD', months 'YYYY-MM', quarters like '2025Q3'.
        Tables:
          company(name, industry)
          industry_medians(industry, revenue_growth_2y_pct, new_product_share_pct, new_product_sales_per_rd, operating_margin_pct)
          orders(order_id, order_date, product, customer, amount, note)
          products(product, launch_date)
          monthly_costs(month, cost_of_sales, selling_admin, research_development)
          quarterly_cash(quarter, operating_cash_flow, capex, cash_end)
          capex_plan(year, planned_capex)
          returns(month, units_shipped, units_returned)
          insider_trades(trade_date, person, title, side, shares, price)
        Use SUM, GROUP BY and WHERE to compute totals rather than reading raw rows.
        """
        try:
            con = sqlite3.connect(db_uri, uri=True)
            try:
                cursor = con.execute(query)
                columns = [c[0] for c in cursor.description or []]
                rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
            finally:
                con.close()
        except sqlite3.Error as exc:
            return f"SQL error: {exc}"
        if not columns:
            return "Query returned no columns."
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
        out = [" | ".join(columns)] + [" | ".join(fmt(v) for v in row) for row in rows[:MAX_SQL_ROWS]]
        if len(rows) > MAX_SQL_ROWS:
            out.append(f"... more rows not shown (limit {MAX_SQL_ROWS}); aggregate or add LIMIT/OFFSET.")
        if not rows:
            out.append("(no rows)")
        return "\n".join(out)

    @tool
    def interview(role: str, topic: str) -> str:
        """Interview one person about the company (at most 10 interviews per company).

        role: customer, current_employee, former_employee, competitor or supplier.
        topic: sales_team, working_conditions, top_management, decision_making or competitive_edge.
        People who would not know about a topic say so; the README lists who knows what.
        """
        role, topic = role.strip().lower(), topic.strip().lower()
        if role not in answers:
            return f"Unknown role {role!r}. Roles: {', '.join(answers)}. (Not counted.)"
        if topic not in answers[role]:
            return f"Unknown topic {topic!r}. Topics: {', '.join(answers[role])}. (Not counted.)"
        with lock:
            if INTERVIEWS_USED[name] >= INTERVIEW_LIMIT:
                return f"Interview limit reached: {INTERVIEW_LIMIT} interviews per company."
            INTERVIEWS_USED[name] += 1
        return answers[role][topic]

    return [list_documents, read_document, sql_query, interview]


def normalize_verdict(value: Any) -> str | None:
    """Accept PASS/FAIL (any case) or a boolean."""
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    text = str(getattr(value, "content", value)).strip().upper()
    return text if text in VERDICTS else None


def check_chosen_points(chosen: Any) -> list[int]:
    """CHOSEN_POINTS must name 3 different points between 1 and 15."""
    try:
        points = sorted({int(p) for p in chosen})
    except (TypeError, ValueError):
        points = []
    if len(points) != 3 or len(chosen) != 3 or not all(p in FISHER_POINTS for p in points):
        raise SystemExit("Set CHOSEN_POINTS at the top of hw3.py to 3 different point numbers "
                         f"between 1 and 15, e.g. [1, 7, 15] (now: {chosen!r}).")
    return points


def normalize_points(points: Any, chosen: list[int]) -> dict[int, str | None]:
    """Map each chosen point number to PASS/FAIL, or None when missing/invalid."""
    points = points if isinstance(points, dict) else {}
    by_number: dict[int, Any] = {}
    for key, value in points.items():
        try:
            by_number[int(key)] = value
        except (TypeError, ValueError):
            continue
    return {p: normalize_verdict(by_number.get(p)) for p in chosen}


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read the expected 15 point verdicts per company."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def point_errors(points: dict[int, str | None], expected: dict[str, Any] | None) -> tuple[int, str]:
    """Count correct point verdicts and describe the wrong ones."""
    if expected is None:
        return 0, ""
    wrong = []
    for p in points:
        truth = expected["points"][str(p)]
        if points[p] != truth:
            wrong.append(f"{p}: expected {truth}, predicted {points[p] or 'missing'}")
    if all(v is None for v in points.values()):
        return 0, "all point verdicts missing"
    return len(points) - len(wrong), "; ".join(wrong)


def balanced_point_accuracy(
    analyses: dict[str, Any], truth: dict[str, dict[str, Any]], names: list[str], chosen: list[int]
) -> float:
    """Mean of the accuracy on true-PASS points and on true-FAIL points.

    Most verdicts are PASS, so answering PASS everywhere gets a high plain
    accuracy but only 50% balanced accuracy. Grading uses the balanced number.
    """
    hits = {"PASS": [0, 0], "FAIL": [0, 0]}
    for name in names:
        if name not in truth:
            continue
        entry = analyses.get(name)
        points = normalize_points(entry.get("points") if isinstance(entry, dict) else None, chosen)
        for p in chosen:
            expected = truth[name]["points"][str(p)]
            hits[expected][0] += points[p] == expected
            hits[expected][1] += 1
    rates = [h / n for h, n in hits.values() if n]
    return sum(rates) / len(rates) if rates else 0.0


def write_results(
    names: list[str], analyses: dict[str, Any], truth: dict[str, dict[str, Any]], chosen: list[int]
) -> tuple[Path, int]:
    """Write results.csv and return the number of correct point verdicts."""
    output = Path("results.csv")
    correct_points = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["company", "points", "points_correct", "point_errors", "interviews_used", "summary"])
        for name in names:
            entry = analyses.get(name)
            entry = entry if isinstance(entry, dict) else {}
            points = normalize_points(entry.get("points"), chosen)
            expected = truth.get(name)
            n_points, errors = point_errors(points, expected)
            correct_points += n_points
            writer.writerow([
                name,
                " ".join(f"{p}:{points[p] or 'missing'}" for p in chosen),
                f"{n_points}/{len(chosen)}" if expected else "",
                errors,
                INTERVIEWS_USED.get(name, 0),
                str(entry.get("summary", "")).strip(),
            ])
    return output, correct_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW3 on company data folders")
    parser.add_argument(
        "--data-folder",
        required=True,
        type=Path,
        help="folder containing one sub-folder per company",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    folders = company_folders(folder)
    if not folders:
        raise SystemExit(f"no company folders found in {folder}")

    chosen = check_chosen_points(CHOSEN_POINTS)
    load_env_file()
    companies = {path.name: make_tools(path) for path in folders}
    agent = build_agent()
    analyses = await analyze_companies(agent, companies)
    if not isinstance(analyses, dict):
        raise TypeError("analyze_companies() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct_points = write_results(list(companies), analyses, truth, chosen)
    summary = f" Your points: {', '.join(map(str, chosen))}."
    if truth:
        summary += (
            f" Point accuracy: {correct_points}/{len(companies) * len(chosen)}"
            f" (balanced: {balanced_point_accuracy(analyses, truth, list(companies), chosen):.1%})."
        )
    print(f"Processed {len(companies)} compan{'y' if len(companies) == 1 else 'ies'}. Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.data_folder.is_dir():
        raise SystemExit(f"not a folder: {args.data_folder}")
    return asyncio.run(run(args.data_folder))


if __name__ == "__main__":
    raise SystemExit(main())

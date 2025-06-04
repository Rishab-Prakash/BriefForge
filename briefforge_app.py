import streamlit as st
from openai import OpenAI
import os
import re
import tempfile
from xhtml2pdf import pisa

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))

# --- GPT Prompt Logic with Phase 2 Context ---
def generate_swot_with_gpt(industry, geography, goal, company_size, budget, challenge, output_type, past_attempts, target_cac, strategic_asset):
    context = ""
    if past_attempts:
        context += f"\nThe company previously tried: {past_attempts}"
    if target_cac:
        context += f"\nTheir CAC target or pain point is: {target_cac}"
    if strategic_asset:
        context += f"\nThey have a strategic asset: {strategic_asset}"

    prompt = f"""
You are an expert strategy consultant writing a professional market opportunity brief.

The client is a {company_size} company in the {industry} industry, targeting {geography}. 
Their primary goal is: {goal}
Their available budget is: {budget or 'not specified'}
They are currently facing: {challenge}
{context}

Based on this, generate a structured brief that includes:
- A SWOT analysis with 2–3 points per category
- Strategic opportunity areas relevant to this challenge
- A clear, action-oriented plan if output_type requires it

Use Markdown-style section headers (e.g., ## SWOT Analysis) and bold each list item label (e.g., **Innovative Tech**: ...)

Output format: {output_type}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a senior McKinsey-style strategy consultant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content

# --- PDF Export ---
def export_to_pdf(brief_text):
    brief_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', brief_text)
    cleaned_lines = []
    buffer = ""
    for line in brief_text.splitlines():
        if line.strip() == "":
            if buffer:
                cleaned_lines.append(buffer.strip())
                buffer = ""
        elif re.match(r"^\d+\.\s", line) or line.startswith("- ") or line.startswith("* ") or line.startswith("#"):
            if buffer:
                cleaned_lines.append(buffer.strip())
            buffer = line.strip()
        else:
            buffer += " " + line.strip()
    if buffer:
        cleaned_lines.append(buffer.strip())

    html = """
    <html><head><style>
    body {
        font-family: 'Segoe UI', sans-serif;
        padding: 25px;
        font-size: 11pt;
        line-height: 1.25;
        color: #333;
    }
    h1 {
        font-size: 18pt;
        color: #1a365d;
        margin: 0 0 8px 0;
        border-bottom: 2px solid #3182ce;
    }
    h2 {
        font-size: 14pt;
        color: #2c5aa0;
        margin: 10px 0 4px 0;
    }
    h3 {
        font-size: 12pt;
        color: #3182ce;
        margin: 8px 0 3px 0;
    }
    p {
        margin: 2px 0 4px 0;
        text-align: justify;
    }
    ol, ul {
        margin: 2px 0 6px 16px;
    }
    li {
        margin-bottom: 0px;
        line-height: 1.2;
    }
    </style></head><body>
    """

    in_list = False
    sublist_mode = False
    list_type = ""

    for line in cleaned_lines:
        if line.startswith("# "): html += f"<h1>{line[2:]}</h1>\n"
        elif line.startswith("## "): html += f"<h2>{line[3:]}</h2>\n"
        elif line.startswith("### "): html += f"<h3>{line[4:]}</h3>\n"
        elif re.match(r"^\d+\.\s", line):
            if not in_list:
                html += "<ol>\n"
                in_list = True
                list_type = "ol"
            html += f"<li>{line[3:]}</li>\n"
        elif line.startswith("- ") or line.startswith("* "):
            if not sublist_mode:
                html += "<ul>\n"
                sublist_mode = True
            html += f"<li>{line[2:]}</li>\n"
        else:
            if sublist_mode:
                html += "</ul>\n"
                sublist_mode = False
            if in_list:
                html += f"</{list_type}>\n"
                in_list = False
            html += f"<p>{line}</p>\n"

    if sublist_mode: html += "</ul>\n"
    if in_list: html += f"</{list_type}>\n"
    html += "</body></html>"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pisa.CreatePDF(html, dest=tmpfile)
        return tmpfile.name

# --- Streamlit UI ---
st.title("BriefForge – Market Opportunity Brief Generator")
st.markdown("Generate investor-ready strategy briefs in seconds.")

industry = st.text_input("Industry", placeholder="e.g., Fintech, FoodTech, SaaS")
geography = st.text_input("Target Geography", placeholder="e.g., Southeast Asia, North America")
goal = st.text_input("Primary Goal", placeholder="e.g., Expansion, Digital Transformation")
company_size = st.text_input("Company Size", placeholder="e.g., Seed-stage startup, Enterprise, 200 employees")
budget = st.text_input("Budget Constraint (optional)", placeholder="e.g., $250K quarterly")
challenge = st.text_area("Current Strategic Challenge", placeholder="e.g., High churn rate, Low lead conversion, Rising CAC")

# 🆕 Phase 2 Inputs
past_attempts = st.text_area("Previous Strategies Tried (optional)", placeholder="e.g., Tried webinars and influencer campaigns")
target_cac = st.text_input("Target CAC / Key Metric (optional)", placeholder="e.g., CAC must stay below $120")
strategic_asset = st.text_area("Strategic Assets / Advantages (optional)", placeholder="e.g., Access to proprietary hospital data")

output_type = st.selectbox("Preferred Output Type", ["SWOT Only", "SWOT + Action Plan", "Full Strategic Brief"])
use_gpt = st.checkbox("Use GPT-4o for Brief Generation")

if st.button("Generate Brief"):
    if industry and geography and goal and company_size and challenge:
        if use_gpt:
            swot = generate_swot_with_gpt(
                industry, geography, goal, company_size, budget,
                challenge, output_type, past_attempts, target_cac, strategic_asset
            )
        else:
            swot = "⚠️ GPT must be enabled to generate a real strategic brief."

        st.markdown("### 🧾 Market Opportunity Brief")
        for line in swot.split("\n"):
            st.markdown(line)

        st.markdown("### 📄 Download Your Brief as PDF")
        pdf_file = export_to_pdf(swot)
        if pdf_file:
            with open(pdf_file, "rb") as f:
                st.download_button("📥 Download PDF", f, file_name="Market_Opportunity_Brief.pdf")
        else:
            st.error("❌ PDF generation failed.")
    else:
        st.warning("Please fill all required fields.")

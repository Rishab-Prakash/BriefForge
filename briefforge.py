import streamlit as st
from openai import OpenAI
import os
import re
import tempfile
from xhtml2pdf import pisa
import logging
import time
import html
import unicodedata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for freemium model
st.session_state.setdefault("free_briefs_used", 0)

# --- Input Sanitization Functions ---
def sanitize_text_input(text, max_length=None, allow_special_chars=True):
    """Sanitize text input to prevent issues and ensure clean data"""
    if not text:
        return ""

    # Convert to string and strip whitespace
    text = str(text).strip()

    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)

    # Remove null bytes and control characters (except newlines and tabs)
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')

    # HTML escape to prevent injection
    text = html.escape(text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # If special characters not allowed, keep only alphanumeric, spaces, and basic punctuation
    if not allow_special_chars:
        text = re.sub(r'[^\w\s\.,!?()-]', '', text)

    # Truncate if max_length specified
    if max_length and len(text) > max_length:
        text = text[:max_length].strip()

    return text

def sanitize_numeric_input(text):
    """Extract and sanitize numeric values from text"""
    if not text:
        return ""

    text = sanitize_text_input(text)
    # Keep only numbers, decimal points, commas, dollar signs, and basic units
    text = re.sub(r'[^\d\.,\$KMBkmb\s-]', '', text)
    return text.strip()

def validate_no_malicious_content(text):
    """Check for potentially malicious content patterns"""
    if not text:
        return True

    # Convert to lowercase for checking
    text_lower = text.lower()

    # Check for common injection patterns
    malicious_patterns = [
        r'<script[^>]*>',
        r'javascript:',
        r'on\w+\s*=',
        r'eval\s*\(',
        r'exec\s*\(',
        r'system\s*\(',
        r'rm\s+-rf',
        r'drop\s+database',
        r'union\s+select',
        r'insert\s+into',
        r'delete\s+from',
    ]

    for pattern in malicious_patterns:
        if re.search(pattern, text_lower):
            return False

    return True

# --- Enhanced Input Validation ---
def validate_inputs(industry, geography, goal, company_size, challenge, budget="", past_attempts="", target_cac="", strategic_asset=""):
    """Comprehensive input validation with detailed error reporting"""
    errors = []
    warnings = []

    # Required field validation
    required_fields = {
        'Industry': industry,
        'Geography': geography,
        'Goal': goal,
        'Company Size': company_size,
        'Challenge': challenge
    }

    for field_name, field_value in required_fields.items():
        if not field_value or len(field_value.strip()) < 2:
            errors.append(f"{field_name} is required and must be at least 2 characters")

    # Specific field validation
    if industry:
        if len(industry) < 3:
            errors.append("Industry must be at least 3 characters")
        elif len(industry) > 100:
            errors.append("Industry name is too long (max 100 characters)")
        elif not re.match(r'^[a-zA-Z\s\-&,\.]+$', industry):
            errors.append("Industry should contain only letters, spaces, hyphens, ampersands, commas, and periods")

    if geography:
        if len(geography) < 3:
            errors.append("Geography must be at least 3 characters")
        elif len(geography) > 150:
            errors.append("Geography is too long (max 150 characters)")
        elif not re.match(r'^[a-zA-Z\s\-,\.&()]+$', geography):
            errors.append("Geography should contain only letters, spaces, hyphens, commas, periods, ampersands, and parentheses")

    if goal:
        if len(goal) < 5:
            errors.append("Goal must be at least 5 characters")
        elif len(goal) > 200:
            errors.append("Goal is too long (max 200 characters)")

    if company_size:
        if len(company_size) < 3:
            errors.append("Company size must be at least 3 characters")
        elif len(company_size) > 150:
            errors.append("Company size description is too long (max 150 characters)")

    if challenge:
        if len(challenge) < 10:
            errors.append("Challenge description must be at least 10 characters")
        elif len(challenge) > 1500:
            errors.append("Challenge description is too long (max 1500 characters)")
        elif len(challenge.split()) < 3:
            errors.append("Challenge description should contain at least 3 words")

    # Optional field validation
    if budget and len(budget) > 100:
        warnings.append("Budget description is quite long - consider being more concise")

    if past_attempts and len(past_attempts) > 800:
        warnings.append("Previous strategies description is very long - consider summarizing key points")

    if target_cac and len(target_cac) > 150:
        warnings.append("Target CAC/metric description is too long (max 150 characters)")

    if strategic_asset and len(strategic_asset) > 800:
        warnings.append("Strategic assets description is very long - consider summarizing key points")

    # Content quality checks
    all_text = f"{industry} {geography} {goal} {company_size} {challenge} {past_attempts} {strategic_asset}"

    # Check for malicious content
    if not validate_no_malicious_content(all_text):
        errors.append("Input contains potentially unsafe content")

    # Check for reasonable content
    if len(all_text.split()) < 10:
        warnings.append("Your inputs seem quite brief - more detail may help generate better briefs")

    # Check for repeated words (potential spam)
    words = all_text.lower().split()
    if len(words) > 0:
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only check words longer than 3 chars
                word_freq[word] = word_freq.get(word, 0) + 1

        max_freq = max(word_freq.values()) if word_freq else 0
        if max_freq > len(words) * 0.3:  # More than 30% repetition
            warnings.append("Your input contains many repeated words - consider varying your language")

    return errors, warnings

def validate_and_sanitize_all_inputs(industry, geography, goal, company_size, budget, challenge, past_attempts, target_cac, strategic_asset):
    """Sanitize all inputs and return cleaned versions"""

    # Sanitize each input
    clean_inputs = {
        'industry': sanitize_text_input(industry, max_length=100, allow_special_chars=False),
        'geography': sanitize_text_input(geography, max_length=150, allow_special_chars=False),
        'goal': sanitize_text_input(goal, max_length=200),
        'company_size': sanitize_text_input(company_size, max_length=150),
        'budget': sanitize_numeric_input(budget)[:100] if budget else "",
        'challenge': sanitize_text_input(challenge, max_length=1500),
        'past_attempts': sanitize_text_input(past_attempts, max_length=800),
        'target_cac': sanitize_text_input(target_cac, max_length=150),
        'strategic_asset': sanitize_text_input(strategic_asset, max_length=800)
    }

    return clean_inputs

# Enhanced OpenAI client initialization with freemium model
@st.cache_resource(show_spinner=False)
def get_openai_client(user_key=None):
    """Get OpenAI client with freemium model support"""
    try:
        if user_key:
            # Test user-provided key
            client = OpenAI(api_key=user_key)
            client.models.list()  # Test the key
            return client

        # No user key provided - check free usage limit
        if st.session_state.free_briefs_used >= 2:
            st.error("❌ You've used your 2 free briefs. Add your own OpenAI API key above to continue generating unlimited briefs.")
            st.stop()

        # Use fallback app key for free usage
        fallback_key = os.environ.get("OPENAI_KEY")
        if not fallback_key:
            try:
                fallback_key = st.secrets["OPENAI_KEY"]
            except:
                pass

        if not fallback_key:
            st.error("❌ No app API key configured. Please contact support or add your own OpenAI API key.")
            st.stop()

        client = OpenAI(api_key=fallback_key)
        client.models.list()  # Test the key
        return client

    except Exception as e:
        if user_key:
            st.error(f"❌ Invalid API key provided. Please check your OpenAI API key and try again.")
        else:
            st.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
        logger.error(f"OpenAI client initialization failed: {e}")
        st.stop()

# --- GPT Prompt Logic with Phase 2 Context ---
def generate_swot_with_gpt(clean_inputs, output_type, client):
    """Generate SWOT analysis with comprehensive error handling using sanitized inputs"""

    # Build context from sanitized inputs
    context = ""
    if clean_inputs['past_attempts']:
        context += f"\nThe company previously tried: {clean_inputs['past_attempts']}"
    if clean_inputs['target_cac']:
        context += f"\nTheir CAC target or pain point is: {clean_inputs['target_cac']}"
    if clean_inputs['strategic_asset']:
        context += f"\nThey have a strategic asset: {clean_inputs['strategic_asset']}"

    prompt = f"""
You are an expert strategy consultant writing a professional market opportunity brief.

The client is a {clean_inputs['company_size']} company in the {clean_inputs['industry']} industry, targeting {clean_inputs['geography']}. 
Their primary goal is: {clean_inputs['goal']}
Their available budget is: {clean_inputs['budget'] if clean_inputs['budget'] else 'not specified'}
They are currently facing: {clean_inputs['challenge']}
{context}

Based on this, generate a structured brief that includes:
- A SWOT analysis with 2–3 points per category
- Strategic opportunity areas relevant to this challenge
- A clear, action-oriented plan if output_type requires it

Use Markdown-style section headers (e.g., ## SWOT Analysis) and bold each list item label (e.g., **Innovative Tech**: ...)

Output format: {output_type}
"""

    # API call with retry logic
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting GPT API call (attempt {attempt + 1}/{max_retries})")

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a senior McKinsey-style strategy consultant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=2500,  # Increased for better content
                timeout=30  # 30 second timeout
            )

            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from OpenAI API")

            content = response.choices[0].message.content.strip()
            if len(content) < 100:  # Basic quality check
                raise ValueError("Response too short, likely incomplete")

            logger.info("GPT API call successful")
            return content

        except Exception as e:
            logger.warning(f"GPT API call attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                st.warning(f"⚠️ API call failed (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # Final attempt failed
                error_msg = f"Failed to generate brief after {max_retries} attempts. Error: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)

# --- PDF Export with Error Handling ---
def export_to_pdf(brief_text):
    """Export brief to PDF with comprehensive error handling"""

    if not brief_text or not brief_text.strip():
        raise ValueError("Cannot export empty brief to PDF")

    try:
        # Additional sanitization for PDF content
        brief_text = sanitize_text_input(brief_text, allow_special_chars=True)

        # Process markdown to HTML
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

        # Build HTML with additional security
        html_content = """
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
            # Additional HTML escaping for safety
            line = html.escape(line, quote=False)

            if line.startswith("# "): 
                html_content += f"<h1>{line[2:]}</h1>\n"
            elif line.startswith("## "): 
                html_content += f"<h2>{line[3:]}</h2>\n"
            elif line.startswith("### "): 
                html_content += f"<h3>{line[4:]}</h3>\n"
            elif re.match(r"^\d+\.\s", line):
                if not in_list:
                    html_content += "<ol>\n"
                    in_list = True
                    list_type = "ol"
                html_content += f"<li>{line[3:]}</li>\n"
            elif line.startswith("- ") or line.startswith("* "):
                if not sublist_mode:
                    html_content += "<ul>\n"
                    sublist_mode = True
                html_content += f"<li>{line[2:]}</li>\n"
            else:
                if sublist_mode:
                    html_content += "</ul>\n"
                    sublist_mode = False
                if in_list:
                    html_content += f"</{list_type}>\n"
                    in_list = False
                html_content += f"<p>{line}</p>\n"

        if sublist_mode: 
            html_content += "</ul>\n"
        if in_list: 
            html_content += f"</{list_type}>\n"
        html_content += "</body></html>"

        # Create PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            pdf_result = pisa.CreatePDF(html_content, dest=tmpfile)

            if pdf_result.err:
                raise Exception(f"PDF generation failed with {pdf_result.err} errors")

            # Verify file was created and has content
            tmpfile.flush()
            if os.path.getsize(tmpfile.name) == 0:
                raise Exception("Generated PDF file is empty")

            logger.info(f"PDF successfully generated: {tmpfile.name}")
            return tmpfile.name

    except Exception as e:
        error_msg = f"PDF export failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

# --- Streamlit UI ---
st.title("BriefForge – Market Opportunity Brief Generator")
st.markdown("Generate investor-ready strategy briefs in seconds.")

# Freemium model UI
remaining_free = max(0, 2 - st.session_state.free_briefs_used)
if remaining_free > 0:
    st.info(f"🎯 **Free Plan**: {remaining_free} free brief{'s' if remaining_free != 1 else ''} remaining")
else:
    st.warning("🔒 **Free limit reached**: Add your API key below to continue")

# User API key input
user_api_key = st.text_input(
    "🔑 Enter Your OpenAI API Key (Optional)", 
    type="password", 
    help="Add your own API key after 2 free briefs to keep using the app without limits."
)

# Help section for getting API key
with st.expander("Need help getting your own key?"):
    st.markdown("""
    1. Go to [OpenAI platform](https://platform.openai.com/account/api-keys)
    2. Sign up or log in
    3. Click **Create new secret key** and copy it
    4. Paste it above to keep generating briefs after the free limit

    💳 Note: You'll be billed by OpenAI directly — a few cents per use.
    """)

# Initialize OpenAI client with freemium logic
client = get_openai_client(user_api_key)

# Status indicator
if user_api_key:
    st.success("✅ Using your personal OpenAI API key")
else:
    st.success("✅ OpenAI API Connected - Free Plan Active")

# Input validation info
st.info("🛡️ **Security Note**: All inputs are automatically sanitized and validated for your protection.")

# Input fields with enhanced validation feedback
industry = st.text_input(
    "Industry *", 
    placeholder="e.g., Fintech, FoodTech, SaaS", 
    help="Required field - Letters, spaces, and basic punctuation only",
    max_chars=100
)

geography = st.text_input(
    "Target Geography *", 
    placeholder="e.g., Southeast Asia, North America", 
    help="Required field - Geographic regions or countries",
    max_chars=150
)

goal = st.text_input(
    "Primary Goal *", 
    placeholder="e.g., Expansion, Digital Transformation", 
    help="Required field - Main business objective",
    max_chars=200
)

company_size = st.text_input(
    "Company Size *", 
    placeholder="e.g., Seed-stage startup, Enterprise, 200 employees", 
    help="Required field - Company stage or employee count",
    max_chars=150
)

budget = st.text_input(
    "Budget Constraint (optional)", 
    placeholder="e.g., $250K quarterly",
    help="Budget available for this initiative",
    max_chars=100
)

challenge = st.text_area(
    "Current Strategic Challenge *", 
    placeholder="e.g., High churn rate, Low lead conversion, Rising CAC", 
    help="Required field - Describe your main business challenge in detail",
    max_chars=1500,
    height=100
)

# Phase 2 Inputs
st.markdown("### 🔍 Additional Context (Optional)")

past_attempts = st.text_area(
    "Previous Strategies Tried", 
    placeholder="e.g., Tried webinars and influencer campaigns",
    help="What have you already tried that didn't work?",
    max_chars=800
)

target_cac = st.text_input(
    "Target CAC / Key Metric", 
    placeholder="e.g., CAC must stay below $120",
    help="Key performance metrics or constraints",
    max_chars=150
)

strategic_asset = st.text_area(
    "Strategic Assets / Advantages", 
    placeholder="e.g., Access to proprietary hospital data",
    help="What unique advantages does your company have?",
    max_chars=800
)

output_type = st.selectbox("Preferred Output Type", ["SWOT Only", "SWOT + Action Plan", "Full Strategic Brief"])
use_gpt = st.checkbox("Use GPT-4o for Brief Generation", value=True, help="Uncheck to see placeholder text")

# Real-time character counters and validation feedback
col1, col2 = st.columns(2)
with col1:
    if challenge:
        char_count = len(challenge)
        color = "red" if char_count > 1500 else "orange" if char_count > 1200 else "green"
        st.markdown(f"Challenge: <span style='color: {color}'>{char_count}/1500 characters</span>", unsafe_allow_html=True)

with col2:
    total_chars = len(f"{industry}{geography}{goal}{company_size}{challenge}{past_attempts}{strategic_asset}")
    st.caption(f"Total input length: {total_chars} characters")

# Real-time validation preview
if industry and geography and goal and company_size and challenge:
    errors, warnings = validate_inputs(industry, geography, goal, company_size, challenge, budget, past_attempts, target_cac, strategic_asset)

    if warnings:
        with st.expander("⚠️ Input Warnings (Click to view)"):
            for warning in warnings:
                st.warning(f"• {warning}")

if st.button("Generate Brief", type="primary"):
    # Comprehensive validation
    errors, warnings = validate_inputs(industry, geography, goal, company_size, challenge, budget, past_attempts, target_cac, strategic_asset)

    if errors:
        st.error("❌ Please fix the following errors:")
        for error in errors:
            st.error(f"• {error}")
    else:
        # Show warnings if any
        if warnings:
            st.warning("⚠️ Warnings (brief will still be generated):")
            for warning in warnings:
                st.warning(f"• {warning}")

        # Sanitize all inputs
        clean_inputs = validate_and_sanitize_all_inputs(
            industry, geography, goal, company_size, budget, 
            challenge, past_attempts, target_cac, strategic_asset
        )

        if use_gpt:
            try:
                with st.spinner("🤖 Generating your strategic brief... This may take 30-60 seconds."):
                    # Add progress tracking
                    progress_bar = st.progress(0)
                    progress_bar.progress(25)

                    swot = generate_swot_with_gpt(clean_inputs, output_type, client)

                    progress_bar.progress(100)
                    st.success("✅ Brief generated successfully!")

            except Exception as e:
                st.error(f"❌ Failed to generate brief: {str(e)}")
                st.info("💡 Please check your inputs and try again. If the problem persists, try again in a few minutes.")
                logger.error(f"Brief generation failed: {e}")
                swot = None
        else:
            swot = "⚠️ GPT must be enabled to generate a real strategic brief. This is placeholder text."

        # Display and export results
        if swot:
            st.markdown("### 🧾 Market Opportunity Brief")
            st.markdown(swot)

            # Increment usage counter for free users AFTER successful generation
            if swot and not user_api_key:
                st.session_state.free_briefs_used += 1

            st.markdown("### 📄 Download Your Brief as PDF")
            try:
                with st.spinner("📄 Generating PDF..."):
                    pdf_file = export_to_pdf(swot)

                    with open(pdf_file, "rb") as f:
                        st.download_button(
                            "📥 Download PDF", 
                            f, 
                            file_name="Market_Opportunity_Brief.pdf",
                            mime="application/pdf"
                        )
                    st.success("✅ PDF ready for download!")

            except Exception as e:
                st.error(f"❌ PDF generation failed: {str(e)}")
                st.info("💡 You can still copy the text content above manually.")
                logger.error(f"PDF export failed: {e}")

# Footer with enhanced security info
st.markdown("---")
st.markdown("💡 **Tips**: Be specific and detailed for better results • All inputs are automatically sanitized")
st.markdown("🔒 **Security**: Your data is processed securely and not stored permanently")
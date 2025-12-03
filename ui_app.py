import os  #sdfg.,mnb
import time
import threading
from typing import Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

# Import from your deployment/validation module
from sample1 import (
    local_validate,
    auto_deploy_flow,
    verify_credentials,
    LOGS_DIR,
    API_VERSION,
)

# Optional: OpenAI for XML generation
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    OPENAI_AVAILABLE = False

load_dotenv()

st.set_page_config(page_title="Salesforce Flow Builder", page_icon="⚙️", layout="wide")

# --------------------------- Helpers ---------------------------

def generate_xml_with_openai(requirement: str, flow_type_hint: Optional[str] = None) -> Tuple[bool, str]:
    """Generate Flow XML using OpenAI. Returns (ok, xml_or_error)."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not OPENAI_AVAILABLE or not api_key:
        return False, "OpenAI client or OPENAI_API_KEY not configured. Please set OPENAI_API_KEY in your .env."

    client = OpenAI(api_key=api_key)

    flow_type = (
        flow_type_hint
        if flow_type_hint in {"Screen", "AutoLaunched"}
        else ("Screen" if any(k in requirement.lower() for k in ["screen", "user", "input", "display"]) else "AutoLaunched")
    )

    prompt = f"""
Generate valid Salesforce {flow_type} Flow XML for: "{requirement}"

CRITICAL XML VALIDATION - 7-Check Process:
1. Correct tag name (exact case, exists in schema)
2. Required children present (no missing/invented tags)
3. Correct parent placement (proper nesting)
4. Child tag order (name → label → locationX → locationY → others)
5. Position constraints (locationX/locationY on ALL elements)
6. Tag compatibility (follow co-occurrence rules)
7. Exact enum values (PascalCase)

OUTPUT:
- Return ONLY XML (no markdown, no explanations)
- Start: <?xml version="1.0" encoding="UTF-8"?>
- Namespace: xmlns="http://soap.sforce.com/2006/04/metadata"
- 4-space indentation
- Schema-compliant, deployment-ready
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are a Salesforce {flow_type} Flow XML generator. Produce deployment-ready metadata XML."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        xml_text = resp.choices[0].message.content.strip()
        if xml_text.startswith("```xml"):
            xml_text = xml_text.split("```xml", 1)[1].split("```", 1)[0].strip()
        elif xml_text.startswith("```"):
            xml_text = xml_text.split("```", 1)[1].split("```", 1)[0].strip()
        if not xml_text.startswith("<?xml"):
            xml_text = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_text
        return True, xml_text
    except Exception as e:
        return False, f"OpenAI generation failed: {e}"


# --------------------------- UI ---------------------------

st.title("Salesforce Flow XML Generator & Deployer")

with st.sidebar:
    st.header("Environment")
    st.caption(f"API Version: v{API_VERSION}")

# Inputs
creds_ok = verify_credentials()
col1, col2 = st.columns([3, 2])
with col1:
    requirement = st.text_area(
        "Enter your flow requirement",
        height=180,
        placeholder="e.g., Create a screen flow to collect user name and show a thank-you message",
    )
with col2:
    flow_name = st.text_input("Flow API Name (no spaces)", value="MySampleFlow")
    flow_type_hint = st.selectbox("Flow type (hint)", ["AutoDetect", "Screen", "AutoLaunched"], index=0)

st.divider()

generate_btn = st.button("Generate XML", type="primary", disabled=not requirement or not flow_name)

if "xml_text" not in st.session_state:
    st.session_state.xml_text = ""

if generate_btn:
    with st.spinner("Generating XML with OpenAI..."):
        ok, res = generate_xml_with_openai(requirement, None if flow_type_hint == "AutoDetect" else flow_type_hint)
    if ok:
        st.session_state.xml_text = res
        st.success("XML generated successfully")
    else:
        st.error(res)

# Editor/preview
st.subheader("Generated XML")
xml_text = st.text_area("Flow XML", value=st.session_state.xml_text, height=300)
st.session_state.xml_text = xml_text

# Actions row (Detect Type and Save removed)
c1, c2, c3 = st.columns(3)
with c1:
    validate_click = st.button("Local Validate")
with c2:
    sf_validate_click = st.button("Salesforce Validate (Check-Only)")
with c3:
    sf_deploy_click = st.button("Salesforce Deploy")

# Status (Detect Type removed)

# Handlers
if validate_click and st.session_state.xml_text.strip():
    ok, errs = local_validate(st.session_state.xml_text)
    if ok:
        st.success("Local XML structure is valid")
    else:
        st.error("Local XML validation failed:")
        for e in errs:
            st.write(f"- {e}")

if (sf_validate_click or sf_deploy_click) and not creds_ok:
    st.error("Salesforce credentials are not valid. Check your .env configuration.")

def _stream_sf_operation(flow_name: str, xml_text: str, check_only: bool):
    """Run auto_deploy_flow in background and live-stream logs into the UI."""
    # Start background thread
    worker = threading.Thread(target=auto_deploy_flow, args=(flow_name, xml_text), kwargs={"check_only": check_only}, daemon=True)
    worker.start()

    log_placeholder = st.empty()
    status_placeholder = st.empty()
    log_file = os.path.join(LOGS_DIR, f"{flow_name}.log")

    status_placeholder.info("Streaming logs... (this may take a few minutes)")
    last_size = 0
    while worker.is_alive():
        try:
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    data = f.read()
                    # Only update if new content to reduce flicker
                    if len(data) != last_size:
                        log_placeholder.code(data, language="text")
                        last_size = len(data)
        except Exception:
            pass
        time.sleep(1)

    # Final read after thread ends
    final_data = ""
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            final_data = f.read()
            log_placeholder.code(final_data, language="text")
    # Determine final status based on log content to mirror terminal output
    success_marker = " DEPLOYMENT SUCCESS!"
    validation_note = "(Validation only - not actually deployed)"
    failure_markers = [
        " FAILED:",
        " Deployment failed",
        " Deploy start failed",
        " No deploy ID returned",
    ]

    if final_data:
        if success_marker in final_data:
            if validation_note in final_data:
                status_placeholder.success("Validation succeeded (check-only). See logs above.")
            else:
                status_placeholder.success("Deployment succeeded. See logs above.")
        elif any(mark in final_data for mark in failure_markers):
            status_placeholder.error("Deployment failed. See logs above for details.")
        else:
            status_placeholder.info("Completed. See full logs above.")
    else:
        status_placeholder.info("Completed, but no log output was captured.")



if sf_validate_click and creds_ok and st.session_state.xml_text.strip():
    st.info("Starting Salesforce validation (check-only)...")
    _stream_sf_operation(flow_name, st.session_state.xml_text, check_only=True)

if sf_deploy_click and creds_ok and st.session_state.xml_text.strip():
    st.warning("Starting FULL deployment to Salesforce...")
    _stream_sf_operation(flow_name, st.session_state.xml_text, check_only=False)



import os
import io
import time
import json
import zipfile
import requests
import base64
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()

# Try to load Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception as e:
    genai = None
    GEMINI_AVAILABLE = False
    print(f"Warning: Gemini not available - {e}")

API_VERSION = os.getenv("SALESFORCE_API_VERSION", "65.0")
MAX_ITERATIONS = int(os.getenv("MAX_DEPLOY_ITERATIONS", "15"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))
LOGS_DIR = os.getenv("DEPLOY_LOGS_DIR", "deploy_logs")

ACCESS_TOKEN = os.getenv("SALESFORCE_ACCESS_TOKEN", "").strip()
INSTANCE_URL = os.getenv("SALESFORCE_INSTANCE_URL", "").strip()

# OAuth credentials
SF_USERNAME = os.getenv("SALESFORCE_USERNAME", "").strip()
SF_PASSWORD = os.getenv("SALESFORCE_PASSWORD", "").strip()
SF_SECURITY_TOKEN = os.getenv("SALESFORCE_SECURITY_TOKEN", "").strip()
SF_CONSUMER_KEY = os.getenv("SALESFORCE_CONSUMER_KEY", "").strip()
SF_CONSUMER_SECRET = os.getenv("SALESFORCE_CONSUMER_SECRET", "").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini configured successfully")
    except Exception as e:
        print(f" Gemini configuration failed: {e}")
        GEMINI_AVAILABLE = False

def get_fresh_access_token() -> Tuple[Optional[str], Optional[str]]:
    """Get fresh Salesforce access token."""
    if not all([SF_USERNAME, SF_PASSWORD, SF_CONSUMER_KEY, SF_CONSUMER_SECRET]):
        print(" Missing OAuth credentials.")
        return None, None
    
    token_url = "https://login.salesforce.com/services/oauth2/token"
    if "sandbox" in INSTANCE_URL.lower() or "test" in INSTANCE_URL.lower():
        token_url = "https://test.salesforce.com/services/oauth2/token"
    
    payload = {
        "grant_type": "password",
        "client_id": SF_CONSUMER_KEY,
        "client_secret": SF_CONSUMER_SECRET,
        "username": SF_USERNAME,
        "password": SF_PASSWORD + SF_SECURITY_TOKEN
    }
    
    try:
        print(" Getting fresh access token...")
        resp = requests.post(token_url, data=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        access_token = data.get("access_token")
        instance_url = data.get("instance_url")
        
        if access_token and instance_url:
            print(f" Token obtained. Instance: {instance_url}")
            return access_token, instance_url
        return None, None
    except Exception as e:
        print(f"Token error: {e}")
        return None, None

def verify_credentials() -> bool:
    """Verify and refresh credentials if needed."""
    global ACCESS_TOKEN, INSTANCE_URL
    
    if not ACCESS_TOKEN or not INSTANCE_URL:
        ACCESS_TOKEN, INSTANCE_URL = get_fresh_access_token()
        if not ACCESS_TOKEN:
            return False
    
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    test_url = f"{INSTANCE_URL}/services/data/v{API_VERSION}/limits"
    
    try:
        resp = requests.get(test_url, headers=headers, timeout=10)
        if resp.status_code == 401:
            print(" Token expired. Refreshing...")
            ACCESS_TOKEN, INSTANCE_URL = get_fresh_access_token()
            if not ACCESS_TOKEN:
                return False
            headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
            resp = requests.get(test_url, headers=headers, timeout=10)
        
        resp.raise_for_status()
        print(" Salesforce credentials verified")
        return True
    except Exception as e:
        print(f" Verification failed: {e}")
        return False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def timestamp() -> str:
    return datetime.now().isoformat()

def log_event(flow_label: str, content: str):
    ensure_dir(LOGS_DIR)
    fname = os.path.join(LOGS_DIR, f"{flow_label}.log")
    with open(fname, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp()}] {content}\n")
    print(f"[{flow_label}] {content}")

def save_iteration(flow_label: str, iteration: int, xml_text: str, prefix: str = ""):
    ensure_dir(LOGS_DIR)
    fname = os.path.join(LOGS_DIR, f"{flow_label}_{prefix}iter{iteration}.flow-meta.xml")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(xml_text)
    log_event(flow_label, f" Saved {prefix}iteration {iteration}")

def zip_metadata(files_dict: Dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for path, content in files_dict.items():
            z.writestr(path, content)
    buffer.seek(0)
    return buffer.read()

def read_xml_file(xml_path: str) -> str:
    """Read XML from file."""
    with open(xml_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def local_validate(xml_text: str) -> Tuple[bool, List[str]]:
    """Validate XML structure locally."""
    errors = []
    if not xml_text.strip().startswith("<?xml"):
        errors.append("Missing XML declaration (<?xml version=\"1.0\"?>)")
    try:
        ET.fromstring(xml_text)
    except ET.ParseError as e:
        errors.append(f"XML Parse Error: {str(e)}")
    return len(errors) == 0, errors

def detect_flow_type(xml_text: str) -> str:
    """
    Detect flow type from XML content.
    Returns: 'screen', 'autolaunched', 'record-triggered', 'scheduled', 'generic'
    """
    try:
        root = ET.fromstring(xml_text)
        ns = {'md': 'http://soap.sforce.com/2006/04/metadata'}
        
        # Method 1: Check processType
        process_type = root.find('.//md:processType', ns)
        if process_type is not None and process_type.text:
            pt_value = process_type.text.lower()
            if 'autolaunched' in pt_value or 'autolaunchedflow' in pt_value:
                return 'autolaunched'
            elif 'workflow' in pt_value:
                return 'autolaunched'
        
        # Method 2: Check for screen elements (most reliable for screen flows)
        screens = root.findall('.//md:screens', ns)
        if screens:
            return 'screen'
        
        # Method 3: Check start element for trigger type
        start = root.find('.//md:start', ns)
        if start is not None:
            trigger_type = start.find('.//md:triggerType', ns)
            if trigger_type is not None and trigger_type.text:
                tt_value = trigger_type.text.lower()
                if 'recordafter' in tt_value or 'recordbefore' in tt_value:
                    return 'record-triggered'
                elif 'scheduled' in tt_value:
                    return 'scheduled'
            
            # Check for record trigger type
            record_trigger = start.find('.//md:recordTriggerType', ns)
            if record_trigger is not None:
                return 'record-triggered'
        
        # Method 4: Check for typical auto-launched elements
        has_record_lookup = root.find('.//md:recordLookups', ns) is not None
        has_record_update = root.find('.//md:recordUpdates', ns) is not None
        has_record_create = root.find('.//md:recordCreates', ns) is not None
        has_record_delete = root.find('.//md:recordDeletes', ns) is not None
        
        if (has_record_lookup or has_record_update or has_record_create or has_record_delete):
            return 'autolaunched'
        
        # Method 5: Check interviewLabel for clues
        interview_label = root.find('.//md:interviewLabel', ns)
        if interview_label is not None and interview_label.text:
            label = interview_label.text.lower()
            if 'screen' in label or 'user' in label:
                return 'screen'
        
        return 'generic'
        
    except Exception as e:
        print(f" Flow type detection error: {e}")
        return 'generic'

def call_gemini(prompt: str) -> Optional[str]:
    """Call Gemini API with given prompt and return cleaned XML."""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        fixed_xml = response.text.strip()
        
        # Clean up markdown formatting
        if "```xml" in fixed_xml:
            fixed_xml = fixed_xml.split("```xml")[1].split("```")[0].strip()
        elif "```" in fixed_xml:
            fixed_xml = fixed_xml.split("```")[1].split("```")[0].strip()
        
        # Ensure XML declaration
        if not fixed_xml.startswith("<?xml"):
            fixed_xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + fixed_xml
        
        return fixed_xml
        
    except Exception as e:
        print(f" Gemini API call failed: {e}")
        return None

def get_base_validation_context() -> str:
    """Common validation rules for all flow types."""
    return """**CRITICAL: Flow Identity Preservation**

NEVER modify these elements - they define the flow's identity:
- <fullName> - The flow's API name (if present)
- <processMetadataValues> where name="originalFlowName" - The original flow name
- <apiVersion> - DO NOT change the version number
- Preserve all version-related elements exactly as they are

**Basic Flow Structure (ORDER MATTERS):**
```xml
<Flow xmlns="http://soap.sforce.com/2006/04/metadata">
    <apiVersion/>          <!-- MUST BE FIRST -->
    <environments/>        <!-- MUST BE SECOND if present -->
    <interviewLabel/>      <!-- Early in document -->
    <label/>              <!-- Early in document -->
    <!-- Other elements follow -->
</Flow>
```

**Variable Rules:**
- All <variable> elements must be defined before use
- Each variable needs <name>, <dataType>, and <isCollection>
- References must match exactly (case-sensitive)

**General XML Rules:**
- Well-formed XML structure
- All opening tags have matching closing tags
- Proper nesting of elements
- Valid namespace declarations

**What you CAN change:**
- Element ordering (moving tags to correct positions)
- Adding missing required elements with default values
- Fixing field references and data types
- Correcting connector references

**What you CANNOT change:**
- The value inside <fullName> tags
- The value inside originalFlowName
- The <apiVersion> number
- Any version-related metadata
"""

def gemini_fix_screen_flow(xml_text: str, errors: List[str], error_type: str, iteration: int) -> Optional[str]:
    """Fix Screen Flow with screen-specific validation."""
    
    base_context = get_base_validation_context()
    
    screen_context = """
**SALESFORCE FLOW XML - ELEMENT ORDERING AT ROOT LEVEL**

MANDATORY ORDER FOR <Flow> CHILD ELEMENTS:
1. <apiVersion>
2. <assignments> (can have multiple, grouped together)
3. <decisions> (can have multiple, grouped together)
4. <description> (optional)
5. <dynamicChoiceSets> (optional, can have multiple)
6. <formulas> (optional, can have multiple)
7. <interviewLabel> (optional)
8. <label>
9. <processMetadataValues> (optional, can have multiple)
10. <processType>
11. <recordCreates> (optional, can have multiple)
12. <recordUpdates> (optional, can have multiple)
13. <screens> (can have multiple, ALL grouped together)
14. <start>
15. <status>
16. <triggerType> (optional)
17. <variables> (optional, can have multiple)

**EACH <screens> ELEMENT INTERNAL ORDER:**
1. <name>
2. <label>
3. <locationX>
4. <locationY>
5. <allowBack>
6. <allowFinish>
7. <allowPause>
8. <connector> (optional)
9. <fields> or <screenFields> (can have multiple)
10. <showFooter> (optional)
11. <showHeader> (optional)


**LABEL PLACEMENT VALIDATION (STRICT):**
- <label> must appear **only** immediately after <name> inside elements like:
  * <screens>
  * <assignments>
  * <decisions>
  * <recordUpdates>
  * <recordCreates>
- NEVER place <label> inside <connector>, <targetReference>, or <inputAssignments>.
- If a <label> appears inside a <connector>, it must be **REMOVE IMMEDIATELY**.
- Violation Example (INVALID):
    <connector>
        <label>Next Step</label>
        <targetReference>Next_Element</targetReference>
    </connector>
  ✅ Correct Form:
    <connector>
        <targetReference>Next_Element</targetReference>
    </connector>

**COMMON ERRORS:**
- "screens is duplicated" = <screens> elements scattered, not grouped together
- "screenFields invalid at this location" = <fields>/<screenFields> not in position 9
- "onAfter is not a valid FlowTriggerType" = Invalid triggerType value
"""

    trigger_type_context = """
**FLOW TRIGGER TYPE VALIDATION**

VALID VALUES FOR <triggerType>:
- None (for Screen Flows, Autolaunched Flows)
- RecordBeforeSave
- RecordAfterSave
- Scheduled
- PlatformEvent

INVALID VALUES (WILL CAUSE ERROR):
 onAfter
 onBefore  
 BeforeSave
 AfterSave
 RecordChange

**MAPPING INVALID TO VALID VALUES:**
- "onAfter" → "RecordAfterSave"
- "onBefore" → "RecordBeforeSave"
- "BeforeSave" → "RecordBeforeSave"
- "AfterSave" → "RecordAfterSave"

**FOR SCREEN FLOWS:**
Screen Flows should NOT have a <triggerType> element at all.
If you see <triggerType> in a Screen Flow (processType = "Flow"), REMOVE it entirely.

**FOR RECORD-TRIGGERED FLOWS:**
If processType = "AutoLaunchedFlow" or "Workflow":
- Use <triggerType>RecordBeforeSave</triggerType> for before-save triggers
- Use <triggerType>RecordAfterSave</triggerType> for after-save triggers
- Ensure <start> has <triggerType> child matching the Flow-level triggerType
"""

    fix_algorithm = """
**FIX ALGORITHM:**

STEP 1: Extract ALL elements from <Flow> and categorize them

STEP 2: Fix TriggerType Issues
   - Check if <triggerType> exists in the Flow
   - If value is "onAfter", change to "RecordAfterSave"
   - If value is "onBefore", change to "RecordBeforeSave"
   - If processType is "Flow" (Screen Flow), REMOVE <triggerType> entirely
   - Check <start> element for nested <triggerType> and fix similarly

STEP 3: Group Multiple Element Instances
   - Collect ALL <screens> elements → place in position 13
   - Collect ALL <decisions> elements → place in position 3
   - Collect ALL <assignments> elements → place in position 2
   - Collect ALL <variables> elements → place in position 17
   - Group all other repeatable elements in their correct positions

STEP 4: Rebuild <Flow> in MANDATORY ORDER
   Follow the sequence from position 1-17 above

STEP 5: Fix Each <screens> Internal Structure
   For each <screens> element:
   - Check if ALL required elements exist (name, label, locationX, locationY, allowBack, allowFinish, allowPause)
   - If ANY required element is missing, ADD it with default value:
     * <locationX>0</locationX>
     * <locationY>0</locationY>
     * <allowBack>true</allowBack>
     * <allowFinish>true</allowFinish>
     * <allowPause>false</allowPause>
   - Verify child elements are in correct order (positions 1-11)
   - Ensure <fields>/<screenFields> is in position 9

STEP 6: Validate Complete Structure
   - All <screens> grouped together
   - <triggerType> has valid value or is removed
   - All elements in correct positions
"""

    prompt = f"""You are fixing a Salesforce Flow XML file with multiple validation errors.

{base_context}

{screen_context}

{trigger_type_context}

{fix_algorithm}

**CURRENT VALIDATION ERRORS (Iteration {iteration}):**
{chr(10).join(f"- {err}" for err in errors)}

**ERROR TYPE:** {error_type}

**XML TO FIX:**
```xml
{xml_text}
```

**YOUR CRITICAL TASKS (ADDRESS ALL ERRORS LISTED ABOVE):**

1. FIX TRIGGER TYPE ERROR (if present in error list):
   - Find any <triggerType> element
   - If value is "onAfter", replace with "RecordAfterSave"
   - If value is "onBefore", replace with "RecordBeforeSave"
   - If this is a Screen Flow (processType="Flow"), DELETE <triggerType> entirely
   - Check <start> element for nested <triggerType> and fix similarly

2. FIX "screens is duplicated" ERROR:
   - Find ALL <screens> elements
   - Group them together in position 13 (after <recordUpdates>, before <start>)
   - Ensure they're not scattered throughout the XML

3. FIX "screenFields invalid" ERROR:
   - For EACH <screens> element, check internal child order
   - CHECK FOR MISSING REQUIRED FIELDS (name, label, locationX, locationY, allowBack, allowFinish, allowPause)
   - ADD any missing required fields with defaults:
     * If <locationX> is missing: add <locationX>0</locationX>
     * If <locationY> is missing: add <locationY>0</locationY>
     * If <allowBack> is missing: add <allowBack>true</allowBack>
     * If <allowFinish> is missing: add <allowFinish>true</allowFinish>
     * If <allowPause> is missing: add <allowPause>false</allowPause>
   - Move <fields> or <screenFields> to position 9 within that screen
   - Verify all 7 required elements (positions 1-7) exist in correct order

4. REBUILD FLOW STRUCTURE:
   - Extract all direct children of <Flow>
   - Sort them according to the mandatory order
   - Group repeatable elements together
   - Maintain all content and attributes

5. FOR RECORDLOOKUPS:

 Each recordLookups element must have a UNIQUE name within the Flow. 
 If you need multiple record lookups, use different names like: GetAccount, GetContact, LookupOpportunity, etc.
 DO NOT create duplicate recordLookups blocks with the same name. -->

<!-- REQUIRED FIELDS for recordLookups:
     - name: Unique identifier (no spaces, no duplicates)
     - label: Human-readable description
     - locationX and locationY: Position on Flow canvas
     - object: Salesforce object to query (Account, Contact, etc.)
     - getFirstRecordOnly: true or false
     - storeOutputAutomatically: true (recommended) or false -->

**VALIDATION BEFORE RETURNING:**
 <triggerType> has valid value OR is removed (if Screen Flow)
 All <screens> elements are grouped in position 13
 Each <screens> has ALL required fields: name, label, locationX, locationY, allowBack, allowFinish, allowPause
 Each <screens> has <fields>/<screenFields> in position 9
 All Flow-level elements in correct order
 XML is well-formed with proper declaration
 NO <screens> element is missing locationX or locationY

**OUTPUT:**
Return ONLY the complete, valid XML.
No explanations, no markdown, no comments.

{base_context}

{generic_context}

**Current Validation Errors (Iteration {iteration}):**
{chr(10).join(f"- {err}" for err in errors)}

**Error Type:** {error_type}

**XML to Fix:**
```xml
{xml_text}
```

**Instructions:**
1. Perform comprehensive XML validation
2. Check element ordering and nesting
3. Verify reference integrity
4. Fix all reported errors
5. Maintain flow identity unchanged
6. Return ONLY the complete fixed XML without explanation

Return the fixed XML now:"""

    return call_gemini(prompt)


def gemini_fix_autolaunched_flow(xml_text: str, errors: List[str], error_type: str, iteration: int) -> Optional[str]:
    """Fix Auto-Launched Flow with trigger-specific validation."""
    
    base_context = get_base_validation_context()
    
    autolaunched_context = """
**AUTO-LAUNCHED FLOW SPECIFIC RULES:**

1. Process Type (Required for auto-launched):
```xml
<Flow>
    <apiVersion>65.0</apiVersion>
    <processMetadataValues>
        <name>BuilderType</name>
        <value><stringValue>LightningFlowBuilder</stringValue></value>
    </processMetadataValues>
    <processMetadataValues>
        <name>CanvasMode</name>
        <value><stringValue>AUTO_LAYOUT_CANVAS</stringValue></value>
    </processMetadataValues>
    <processType>AutoLaunchedFlow</processType>
    <!-- or Workflow, InvocableProcess -->
</Flow>
```

2. Start Element (Trigger Configuration):
```xml
<start>
    <locationX>0</locationX>
    <locationY>0</locationY>
    <connector>
        <targetReference>FirstElement</targetReference>
    </connector>
    <!-- For Record-Triggered Flows: -->
    <filterLogic>and</filterLogic>
    <filters>
        <field>Status__c</field>
        <operator>EqualTo</operator>
        <value><stringValue>Active</stringValue></value>
    </filters>
    <object>Account</object>
    <recordTriggerType>Create</recordTriggerType>
    <!-- Options: Create, Update, CreateAndUpdate, Delete -->
    <triggerType>RecordAfterSave</triggerType>
    <!-- Options: RecordAfterSave, RecordBeforeSave -->
    
    <!-- For Scheduled Flows: -->
    <scheduledPaths>
        <connector><targetReference>ScheduledAction</targetReference></connector>
        <label>Daily Run</label>
        <offsetNumber>1</offsetNumber>
        <offsetUnit>Days</offsetUnit>
        <timeSource>FlowExecutionTime</timeSource>
    </scheduledPaths>
</start>
```

3. Record Lookups Structure:
```xml
<recordLookups>
    <name>Get_Related_Contact</name>
    <label>Get Related Contact</label>
    <locationX>100</locationX>
    <locationY>100</locationY>
    <assignNullValuesIfNoRecordsFound>false</assignNullValuesIfNoRecordsFound>
    <connector>
        <targetReference>NextElement</targetReference>
    </connector>
    <filterLogic>and</filterLogic>
    <filters>
        <field>AccountId</field>
        <operator>EqualTo</operator>
        <value>
            <elementReference>$Record.Id</elementReference>
        </value>
    </filters>
    <object>Contact</object>
    <outputAssignments>
        <assignToReference>varContactName</assignToReference>
        <field>Name</field>
    </outputAssignments>
    <getFirstRecordOnly>true</getFirstRecordOnly>
    <storeOutputAutomatically>true</storeOutputAutomatically>
</recordLookups>
```

4. Record Updates Structure:
```xml
<recordUpdates>
    <name>Update_Account_Status</name>
    <label>Update Account Status</label>
    <locationX>200</locationX>
    <locationY>200</locationY>
    <connector>
        <targetReference>NextElement</targetReference>
    </connector>
    <filterLogic>and</filterLogic>
    <filters>
        <field>Id</field>
        <operator>EqualTo</operator>
        <value>
            <elementReference>$Record.Id</elementReference>
        </value>
    </filters>
    <inputAssignments>
        <field>Status__c</field>
        <value>
            <stringValue>Updated</stringValue>
        </value>
    </inputAssignments>
    <object>Account</object>
</recordUpdates>
```

5. Record Creates Structure:
```xml
<recordCreates>
    <name>Create_Task</name>
    <label>Create Task</label>
    <locationX>300</locationX>
    <locationY>300</locationY>
    <connector>
        <targetReference>NextElement</targetReference>
    </connector>
    <inputAssignments>
        <field>Subject</field>
        <value>
            <stringValue>Follow up</stringValue>
        </value>
    </inputAssignments>
    <inputAssignments>
        <field>WhoId</field>
        <value>
            <elementReference>$Record.Id</elementReference>
        </value>
    </inputAssignments>
    <object>Task</object>
</recordCreates>
```

6. Decision Elements:
```xml
<decisions>
    <name>Check_Status</name>
    <label>Check Status</label>
    <locationX>400</locationX>
    <locationY>400</locationY>
    <defaultConnector>
        <targetReference>DefaultPath</targetReference>
    </defaultConnector>
    <defaultConnectorLabel>Default</defaultConnectorLabel>
    <rules>
        <name>Status_Is_Active</name>
        <conditionLogic>and</conditionLogic>
        <conditions>
            <leftValueReference>$Record.Status__c</leftValueReference>
            <operator>EqualTo</operator>
            <rightValue>
                <stringValue>Active</stringValue>
            </rightValue>
        </conditions>
        <connector>
            <targetReference>ActivePath</targetReference>
        </connector>
        <label>Status Is Active</label>
    </rules>
</decisions>
```

7. Assignment Elements:
```xml
<assignments>
    <name>Set_Variables</name>
    <label>Set Variables</label>
    <locationX>500</locationX>
    <locationY>500</locationY>
    <assignmentItems>
        <assignToReference>varAccountName</assignToReference>
        <operator>Assign</operator>
        <value>
            <elementReference>$Record.Name</elementReference>
        </value>
    </assignmentItems>
    <connector>
        <targetReference>NextElement</targetReference>
    </connector>
</assignments>
```

8. Record Context Variables:
   - $Record: Current triggering record
   - $Record.Id: ID of triggering record
   - $Record.FieldName__c: Access any field
   - $Record__Prior: Record before update (only for Update triggers)
   - $Record__Prior.FieldName__c: Previous field value

9. Common Auto-Launched Errors:
   - Missing <object> in record operations
   - Invalid <recordTriggerType> (must be: Create, Update, CreateAndUpdate, Delete)
   - Wrong <triggerType> (RecordAfterSave vs RecordBeforeSave)
   - $Record references in flows without record trigger
   - Missing <filterLogic> when multiple filters exist
   - Invalid field references
   - Missing required connector targetReferences
   - Circular reference errors in assignments
"""

    autolaunched_critical = """
 CRITICAL AUTO-LAUNCHED FLOW VALIDATIONS 

**MANDATORY CHECKS:**

1. **Trigger Configuration Validation:**
   - If <object> exists in <start>, must have <recordTriggerType>
   - If <recordTriggerType> exists, must have <triggerType>
   - <triggerType> must be: RecordAfterSave OR RecordBeforeSave
   - <recordTriggerType> must be: Create, Update, CreateAndUpdate, or Delete

2. **Record Operations Validation:**
   - EVERY <recordLookups> MUST have <object> tag
   - EVERY <recordUpdates> MUST have <object> tag
   - EVERY <recordCreates> MUST have <object> tag
   - Object names must be valid Salesforce objects

3. **Filter Logic Validation:**
   - If 2+ <filters>, MUST have <filterLogic>
   - <filterLogic> must be: "and", "or", or custom logic like "1 AND (2 OR 3)"
   - Each filter needs: <field>, <operator>, <value>

4. **Reference Validation:**
   - $Record can only be used if flow has record trigger
   - $Record__Prior can only be used with Update trigger
   - All <elementReference> must point to existing elements
   - All <targetReference> must point to existing elements

5. **Element Ordering:**
   - <start> should be first element after metadata
   - Variables should be defined before use
   - Connectors must create valid flow path

6. FOR RECORDLOOKUPS:

 Each recordLookups element must have a UNIQUE name within the Flow. 
 If you need multiple record lookups, use different names like: GetAccount, GetContact, LookupOpportunity, etc.
 DO NOT create duplicate recordLookups blocks with the same name. -->

<!-- REQUIRED FIELDS for recordLookups:
     - name: Unique identifier (no spaces, no duplicates)
     - label: Human-readable description
     - locationX and locationY: Position on Flow canvas
     - object: Salesforce object to query (Account, Contact, etc.)
     - getFirstRecordOnly: true or false
     - storeOutputAutomatically: true (recommended) or false -->
"""

    prompt = f"""You are fixing a Salesforce AUTO-LAUNCHED FLOW XML.

{base_context}

{autolaunched_context}

{autolaunched_critical}

**Current Validation Errors (Iteration {iteration}):**
{chr(10).join(f"- {err}" for err in errors)}

**Error Type:** {error_type}

**XML to Fix:**
```xml
{xml_text}
```

# **Instructions:**
1. Identify the trigger configuration in <start> element
2. Verify all record operations have required <object> tags
3. Check filter logic consistency
4. Validate all $Record references are appropriate
5. Ensure all connectors point to valid elements
6. Fix all reported errors while preserving business logic
7. Maintain all flow identity elements unchanged
8. Return ONLY the complete, valid XML.

Fixed XML:"""
    return call_gemini(prompt)


def gemini_fix_record_triggered_flow(xml_text: str, errors: List[str], error_type: str, iteration: int) -> Optional[str]:
    """Fix record-triggered flow with specific validation rules."""
    
    base_context = get_base_validation_context()
    
    record_triggered_context = """
**RECORD-TRIGGERED FLOW VALIDATION RULES:**

1. **ProcessType Requirements:**
   - Must be "AutoLaunchedFlow"
   - Record-triggered flows are AutoLaunched with trigger metadata

2. **<start> Element (MANDATORY):**
   Must contain in exact order:
   - <locationX>, <locationY>
   - <doesRequireRecordChangedToMeetCriteria> (true/false)
   - <filterFormula> (optional, for entry criteria)
   - <object> (triggering Salesforce object)
   - <recordTriggerType> (Create | Update | Delete | CreateAndUpdate)
   - <scheduledPaths> OR <connector> (path to first element)
   - <triggerType> (RecordBeforeSave | RecordAfterSave)

3. **Context Variables:**
   - $Record (current record being processed)
   - $Record.FieldName (field access)
   - $Record__Prior (only for Update triggers)
   - $Record.RelatedObject.Field (spanning relationships)

4. **Element Requirements:**
   - ALL elements must have <locationX> and <locationY>
   - recordLookups use <filterLogic>, NOT conditionLogic
   - Decisions use <conditionLogic>
   - All connectors need valid <targetReference>

5. **Common Errors:**
   - Missing <object> in <start>
   - Wrong triggerType (use RecordBeforeSave/RecordAfterSave)
   - Wrong recordTriggerType (use Create/Update/Delete/CreateAndUpdate)
   - Using <filters> instead of <filterFormula> in <start>
   - Missing <doesRequireRecordChangedToMeetCriteria>
   - Invalid $Record context variable references
   - Mixing up filterLogic and conditionLogic

6. **Async Execution Pattern:**
   When using <scheduledPaths>:
```xml
   <scheduledPaths>
       <connector>
           <targetReference>ElementName</targetReference>
       </connector>
       <pathType>AsyncAfterCommit</pathType>
   </scheduledPaths>
```
"""

    prompt = f"""You are fixing a Salesforce Record-Triggered Flow XML.

{base_context}

{record_triggered_context}

**Current Validation Errors (Iteration {iteration}):**
{chr(10).join(f"- {err}" for err in errors)}

**Error Type:** {error_type}

**XML to Fix:**
```xml
{xml_text}
```

**Instructions:**
1. Verify <start> element has all required record-trigger fields
2. Ensure triggerType is RecordBeforeSave or RecordAfterSave
3. Ensure recordTriggerType is Create, Update, Delete, or CreateAndUpdate
4. Check all $Record variable references are valid
5. Verify filterLogic in recordLookups, conditionLogic in decisions
6. Fix all reported errors while maintaining flow logic
7. Return ONLY the complete fixed XML without explanation

Fixed XML:"""
    return call_gemini(prompt)

def gemini_fix_generic_flow(xml_text: str, errors: List[str], error_type: str, iteration: int) -> Optional[str]:
    """Fix any flow type with general validation (fallback)."""
    
    base_context = get_base_validation_context()
    
    generic_context = """
**GENERIC FLOW VALIDATION RULES:**

1. Common Element Ordering:
   - Variables: name, dataType, isCollection, value
   - Formulas: name, dataType, expression
   - Connectors: must have targetReference
   - Decisions: name, label, locationX, locationY, rules, defaultConnector

2. Reference Integrity:
   - All elementReference must point to existing elements
   - All targetReference must point to existing elements
   - Variable references must match variable names exactly

3. Data Type Consistency:
   - String, Number, Boolean, Date, DateTime
   - Collections must set isCollection=true
   - Formula expressions must match result dataType

4. Common Errors:
   - Misordered elements within parent tags
   - Missing required child elements
   - Invalid reference names
   - Malformed XML syntax
   - Namespace issues
"""

    prompt = f"""You are fixing a Salesforce Flow XML (type could not be determined).

{base_context}

{generic_context}

**Current Validation Errors (Iteration {iteration}):**
{chr(10).join(f"- {err}" for err in errors)}

**Error Type:** {error_type}

**XML to Fix:**
```xml
{xml_text}
```

**Instructions:**
1. Perform comprehensive XML validation
2. Check element ordering and nesting
3. Verify reference integrity
4. Fix all reported errors
5. Maintain flow identity unchanged
6. Return ONLY the complete fixed XML without explanation

Return the fixed XML now:"""

    return call_gemini(prompt)

def start_deploy(zip_bytes: bytes, check_only: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """Deploy metadata to Salesforce."""
    url = f"{INSTANCE_URL}/services/Soap/m/{API_VERSION}"
    b64_zip = base64.b64encode(zip_bytes).decode('utf-8')

    soap_body = f"""<?xml version="1.0" encoding="utf-8" ?>
<env:Envelope xmlns:xsd="http://www.w3.org/2001/XMLSchema"
              xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
              xmlns:env="http://schemas.xmlsoap.org/soap/envelope/">
  <env:Header>
    <SessionHeader xmlns="http://soap.sforce.com/2006/04/metadata">
      <sessionId>{ACCESS_TOKEN}</sessionId>
    </SessionHeader>
  </env:Header>
  <env:Body>
    <deploy xmlns="http://soap.sforce.com/2006/04/metadata">
      <ZipFile>{b64_zip}</ZipFile>
      <DeployOptions>
        <allowMissingFiles>false</allowMissingFiles>
        <autoUpdatePackage>false</autoUpdatePackage>
        <checkOnly>{str(check_only).lower()}</checkOnly>
        <rollbackOnError>true</rollbackOnError>
        <singlePackage>true</singlePackage>
      </DeployOptions>
    </deploy>
  </env:Body>
</env:Envelope>"""

    headers = {"Content-Type": "text/xml", "SOAPAction": "deploy"}

    try:
        resp = requests.post(url, headers=headers, data=soap_body.encode("utf-8"), timeout=180)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        ns = {"soapenv": "http://schemas.xmlsoap.org/soap/envelope/",
              "m": "http://soap.sforce.com/2006/04/metadata"}
        deploy_id_elem = root.find(".//m:id", ns)
        
        if deploy_id_elem is not None:
            return True, {"deployId": deploy_id_elem.text}
        return True, {"raw_response": resp.text}
    except Exception as e:
        return False, {"error": str(e)}

def check_deploy_status(deploy_id: str) -> dict:
    """Check deployment status."""
    url = f"{INSTANCE_URL}/services/Soap/m/{API_VERSION}"

    soap_body = f"""<?xml version="1.0" encoding="UTF-8"?>
<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/"
                  xmlns="http://soap.sforce.com/2006/04/metadata">
  <soapenv:Header>
    <SessionHeader>
      <sessionId>{ACCESS_TOKEN}</sessionId>
    </SessionHeader>
  </soapenv:Header>
  <soapenv:Body>
    <checkDeployStatus>
      <asyncProcessId>{deploy_id}</asyncProcessId>
      <includeDetails>true</includeDetails>
    </checkDeployStatus>
  </soapenv:Body>
</soapenv:Envelope>"""

    headers = {"Content-Type": "text/xml", "SOAPAction": "checkDeployStatus"}

    try:
        response = requests.post(url, headers=headers, data=soap_body.encode("utf-8"), timeout=120)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        ns = {"soapenv": "http://schemas.xmlsoap.org/soap/envelope/",
              "m": "http://soap.sforce.com/2006/04/metadata"}

        done = root.findtext(".//m:done", namespaces=ns) == "true"
        success = root.findtext(".//m:success", namespaces=ns) == "true"
        state = root.findtext(".//m:state", namespaces=ns)
        problems = []

        for failure in root.findall(".//m:componentFailures", namespaces=ns):
            file_name = failure.findtext("m:fileName", namespaces=ns)
            problem = failure.findtext("m:problem", namespaces=ns)
            line = failure.findtext("m:lineNumber", namespaces=ns)
            col = failure.findtext("m:columnNumber", namespaces=ns)
            if problem:
                location = f" (Line {line}, Col {col})" if line else ""
                problems.append(f"{file_name or 'Unknown'}{location}: {problem}")

        if not problems:
            for msg in root.findall(".//m:messages", namespaces=ns):
                text = msg.findtext("m:problem", namespaces=ns)
                if text:
                    problems.append(text)

        return {"done": done, "success": success, "state": state, "errors": problems}
    except Exception as e:
        return {"done": False, "success": False, "errors": [str(e)]}

#AUTO-DEPLOY WITH TYPE-SPECIFIC FIX
def auto_deploy_flow(flow_name: str, xml_content: str, check_only: bool = False):
    """
    Automatically deploy a flow with type-specific intelligent error fixing.
    Detects flow type and uses appropriate fix function.
    """
    
    log_event(flow_name, "=" * 60)
    log_event(flow_name, " STARTING AUTO-DEPLOY WITH TYPE-SPECIFIC GEMINI FIX")
    log_event(flow_name, f"Mode: {'VALIDATE ONLY' if check_only else 'FULL DEPLOYMENT'}")
    log_event(flow_name, "=" * 60)
    
    # ========== DETECT FLOW TYPE ==========
    flow_type = detect_flow_type(xml_content)
    log_event(flow_name, f"\n FLOW TYPE DETECTED: {flow_type.upper()}")
    
    # Map flow types to fix functions
    fix_functions = {

        'screen': gemini_fix_screen_flow,

        'autolaunched': gemini_fix_autolaunched_flow,

        'record-triggered': gemini_fix_record_triggered_flow,

        'scheduled': gemini_fix_autolaunched_flow,

        'generic': gemini_fix_generic_flow

    }
    
    fix_function = fix_functions.get(flow_type, gemini_fix_generic_flow)
    log_event(flow_name, f" Using fix function: {fix_function.__name__}")
    log_event(flow_name, "=" * 60)
    
    current_xml = xml_content
    save_iteration(flow_name, 0, current_xml, "original_")
    
    # ========== PHASE 1: FIX XML STRUCTURE ERRORS ==========
    log_event(flow_name, "\n PHASE 1: XML Structure Validation")
    
    for attempt in range(1, MAX_ITERATIONS + 1):
        is_valid, xml_errors = local_validate(current_xml)
        
        if is_valid:
            log_event(flow_name, "XML structure is valid")
            break
        
        log_event(flow_name, f" XML validation failed (Attempt {attempt}/{MAX_ITERATIONS})")
        for idx, err in enumerate(xml_errors, 1):
            log_event(flow_name, f"   Error {idx}: {err}")
        
        if not GEMINI_AVAILABLE:
            log_event(flow_name, " Gemini not available. Cannot auto-fix.")
            return
        
        log_event(flow_name, f" Calling {fix_function.__name__} to fix XML structure...")
        fixed_xml = fix_function(current_xml, xml_errors, "XML_STRUCTURE", attempt)
        
        if not fixed_xml:
            log_event(flow_name, " Gemini returned no fix")
            return
        
        current_xml = fixed_xml
        save_iteration(flow_name, attempt, current_xml, "xml_fix_")
        log_event(flow_name, f" Applied fix attempt {attempt}")
    else:
        log_event(flow_name, " FAILED: Could not fix XML structure after max attempts")
        return
    
    # ========== PHASE 2: SALESFORCE DEPLOYMENT WITH AUTO-FIX ==========
    log_event(flow_name, "\n PHASE 2: Salesforce Deployment")
    
    files_dict = {
        f"flows/{flow_name}.flow-meta.xml": current_xml,
        "package.xml": f"""<?xml version="1.0" encoding="UTF-8"?>
<Package xmlns="http://soap.sforce.com/2006/04/metadata">
<types><members>{flow_name}</members><name>Flow</name></types>
<version>{API_VERSION}</version>
</Package>"""
    }
    
    for deploy_attempt in range(1, MAX_ITERATIONS + 1):
        log_event(flow_name, f"\n Deployment Attempt {deploy_attempt}/{MAX_ITERATIONS}")
        
        zip_bytes = zip_metadata(files_dict)
        ok, resp = start_deploy(zip_bytes, check_only=check_only)
        
        if not ok:
            log_event(flow_name, f" Deploy start failed: {resp.get('error')}")
            return
        
        deploy_id = resp.get("deployId")
        if not deploy_id:
            log_event(flow_name, f" No deploy ID returned")
            return
        
        log_event(flow_name, f" Deploy ID: {deploy_id}")
        log_event(flow_name, " Polling status...")
        
        # Poll for completion
        for poll_count in range(40):  # 40 * 5s = 3+ minutes
            status = check_deploy_status(deploy_id)
            
            if status.get("done"):
                break
            
            if poll_count % 4 == 0:  # Log every 20 seconds
                log_event(flow_name, f"   Still processing... ({poll_count * POLL_INTERVAL}s)")
            
            time.sleep(POLL_INTERVAL)
        
        if status.get("success"):
            log_event(flow_name, "\n" + "=" * 60)
            log_event(flow_name, " DEPLOYMENT SUCCESS!")
            if check_only:
                log_event(flow_name, "   (Validation only - not actually deployed)")
            else:
                log_event(flow_name, f"   Flow '{flow_name}' is now in Salesforce!")
                log_event(flow_name, "    Remember to ACTIVATE the flow in Setup → Flows")
            log_event(flow_name, "=" * 60)
            return
        
        # Deployment failed - extract errors
        sf_errors = status.get("errors", ["Unknown deployment failure"])
        log_event(flow_name, f" Deployment failed (Attempt {deploy_attempt})")
        for idx, err in enumerate(sf_errors, 1):
            log_event(flow_name, f"   Error {idx}: {err}")
        
        if deploy_attempt >= MAX_ITERATIONS:
            log_event(flow_name, "\n FAILED: Max deployment attempts reached")
            return
        
        if not GEMINI_AVAILABLE:
            log_event(flow_name, " Gemini not available. Cannot auto-fix.")
            return
        
        log_event(flow_name, f" Calling {fix_function.__name__} to fix Salesforce errors...")
        fixed_xml = fix_function(
            files_dict[f"flows/{flow_name}.flow-meta.xml"],
            sf_errors,
            "SALESFORCE_DEPLOYMENT",
            deploy_attempt
        )
        
        if not fixed_xml:
            log_event(flow_name, " Gemini returned no fix")
            return
        
        # Validate the fix before retrying
        is_valid, val_errors = local_validate(fixed_xml)
        if not is_valid:
            log_event(flow_name, f" Fix created invalid XML: {val_errors}")
            return
        
        files_dict[f"flows/{flow_name}.flow-meta.xml"] = fixed_xml
        save_iteration(flow_name, deploy_attempt, fixed_xml, "deploy_fix_")
        log_event(flow_name, f" Applied Salesforce fix {deploy_attempt}")
    
    log_event(flow_name, "\n FAILED: Could not deploy after max attempts")

# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    XML_FILE_PATH = os.path.join(os.path.dirname(__file__), "sample_xml10.xml")
    
    # Extract flow name from file path (without .xml extension)
    flow_name = os.path.splitext(os.path.basename(XML_FILE_PATH))[0]
    
    print("\n" + "=" * 70)
    print("  SALESFORCE FLOW AUTO-DEPLOY WITH TYPE-SPECIFIC GEMINI AI FIX")
    print("=" * 70)
    
    try:
        # Verify credentials
        if not verify_credentials():
            print("\n FAILED: Could not verify Salesforce credentials")
            print("\n Required .env variables:")
            print("   SALESFORCE_USERNAME, SALESFORCE_PASSWORD")
            print("   SALESFORCE_SECURITY_TOKEN, SALESFORCE_CONSUMER_KEY")
            print("   SALESFORCE_CONSUMER_SECRET, GEMINI_API_KEY")
            exit(1)
        
        # Check Gemini availability
        if not GEMINI_AVAILABLE:
            print("\n WARNING: Gemini AI not available")
            print("   Auto-fix will not work. Please install: pip install google-generativeai")
            print("   And set GEMINI_API_KEY in .env")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                exit(1)
        
        # Read XML
        print(f"\n Reading XML from: {XML_FILE_PATH}")
        if not os.path.exists(XML_FILE_PATH):
            print(f" File not found: {XML_FILE_PATH}")
            exit(1)
            
        xml_text = read_xml_file(XML_FILE_PATH)
        print(f"   XML size: {len(xml_text)} bytes")
        
        # Preview flow type detection
        detected_type = detect_flow_type(xml_text)
        print(f"   Detected flow type: {detected_type.upper()}")
        
        # Choose mode
        print("\n Choose deployment mode:")
        print("   1. Validate only (check for errors, don't deploy)")
        print("   2. Full deployment (actually deploy to Salesforce)")
        choice = input("\nEnter choice (1 or 2) [default: 2]: ").strip() or "2"
        
        check_only = (choice == "1")
        
        # Start deployment
        print("\n" + "=" * 70)
        auto_deploy_flow(flow_name, xml_text, check_only=check_only)
        
        print("\n" + "=" * 70)
        print(f" Check logs at: {LOGS_DIR}/{flow_name}.log")
        print("=" * 70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n Deployment cancelled by user")
        exit(1)
    except Exception as e:
        print(f"\n CRITICAL ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        exit(1)


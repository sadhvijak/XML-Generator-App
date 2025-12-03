import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context

# Load environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")

client = OpenAI(api_key=api_key)

# Configure LlamaIndex
print(" Initializing embedding model...")
embed_model = HuggingFaceEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1", 
    trust_remote_code=True
)
Settings.embed_model = embed_model
Settings.llm = LlamaOpenAI(model="gpt-4o", api_key=api_key)
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# Initialize ChromaDB
print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_storage")
chroma_collection = chroma_client.get_collection(name="xml_pattern_embeddings")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

print("✓ Connected to ChromaDB with Nomic embeddings")

# Create specialized query engines
print(" Creating specialized query engines...")

element_query_engine = index.as_query_engine(similarity_top_k=5, response_mode="compact")
relationship_query_engine = index.as_query_engine(similarity_top_k=5, response_mode="compact")
sequence_query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")
positional_query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")
child_ordering_query_engine = index.as_query_engine(similarity_top_k=5, response_mode="compact")
xml_examples_query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")
attribute_enum_query_engine = index.as_query_engine(similarity_top_k=5, response_mode="compact")
cooccurrence_query_engine = index.as_query_engine(similarity_top_k=5, response_mode="compact")
context_patterns_query_engine = index.as_query_engine(similarity_top_k=5, response_mode="compact")
general_query_engine = index.as_query_engine(similarity_top_k=10, response_mode="compact")

# Create QueryEngineTool objects
print("  Creating query tools for ReAct agent...")

query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=element_query_engine,
        name="element_templates_tool",
        description=(
            "Retrieves XML element template patterns including Screen Flow specific rules "
            "(fields vs screenFields, fieldText vs label, DisplayText vs Label fieldType). "
            "Input should be a specific element type."
        ),
    ),
    QueryEngineTool.from_defaults(
        query_engine=relationship_query_engine,
        name="parent_child_relationships_tool",
        description=(
            "Retrieves parent-child relationship patterns for XML elements. "
            "Use this for nesting hierarchies. Input should describe the parent element."
        ),
    ),
    QueryEngineTool.from_defaults(
        query_engine=sequence_query_engine,
        name="sequence_templates_tool",
        description=(
            "Retrieves tag ordering and sequence patterns at Flow root level. "
            "Input should specify which container element you need ordering for."
        ),
    ),
    QueryEngineTool.from_defaults(
        query_engine=positional_query_engine,
        name="positional_constraints_tool",
        description=(
            "Retrieves locationX/locationY positioning requirements and spacing rules. "
            "Input should describe the constraint scenario."
        ),
    ),
    QueryEngineTool.from_defaults(
        query_engine=child_ordering_query_engine,
        name="child_ordering_rules_tool",
        description=(
            "Retrieves CANONICAL child element ordering (e.g., name → label → locationX → locationY). "
            "CRITICAL for XML validity. Input should be the parent element name."
        ),
    ),
    QueryEngineTool.from_defaults(
        query_engine=xml_examples_query_engine,
        name="xml_examples_tool",
        description=(
            "Retrieves concrete XML code examples showing proper structure. "
            "Input should be the element name you want examples for."
        ),
    ),
    QueryEngineTool.from_defaults(
        query_engine=attribute_enum_query_engine,
        name="attribute_enums_tool",
        description=(
            "Retrieves valid enum values (EqualTo, DisplayText, Flow, etc.). "
            "Input should be the attribute name or element context."
        ),
    ),
    QueryEngineTool.from_defaults(
        query_engine=cooccurrence_query_engine,
        name="cooccurrence_rules_tool",
        description=(
            "Retrieves element co-occurrence rules (always_with / never_with). "
            "Input should be the element name or combination."
        ),
    ),
    QueryEngineTool.from_defaults(
        query_engine=context_patterns_query_engine,
        name="context_patterns_tool",
        description=(
            "Retrieves context-specific patterns for elements in different locations. "
            "Input should describe the parent/context scenario."
        ),
    ),
    QueryEngineTool.from_defaults(
        query_engine=general_query_engine,
        name="general_patterns_tool",
        description=(
            "Retrieves general Salesforce Flow XML patterns. "
            "Use for broad queries or when unsure which specific pattern type."
        ),
    ),
]

# Create ReAct Agent
print(" Initializing ReAct Agent...")

system_prompt = """You are a Salesforce Flow XML architecture expert. Your job is to gather 
ALL necessary XML patterns and structural information to help generate valid Salesforce Flow XML.

When given a requirement, SYSTEMATICALLY query these pattern types:

1. **Element Templates** - Base structure (CRITICAL: fields vs screenFields, fieldText vs label)
2. **Parent-Child Relationships** - Nesting hierarchies
3. **Child Ordering Rules** - EXACT child tag ordering (name → label → locationX → locationY)
4. **XML Examples** - Concrete working examples
5. **Attribute Enums** - Valid enum values (DisplayText not Label, EqualTo not equals)
6. **Co-occurrence Rules** - Elements that must/cannot appear together
7. **Context Patterns** - Context-specific variations
8. **Sequence Templates** - Flow-level element ordering
9. **Positional Constraints** - locationX/locationY requirements and spacing

CRITICAL SCREEN FLOW RULES TO VERIFY:
- Use <fields> NOT <screenFields>
- Use <fieldText> NOT <label> inside fields
- Use DisplayText NOT Label as fieldType
- locationX/locationY REQUIRED for ALL elements
- Order: name → label → locationX → locationY → other tags

Be thorough - query every relevant tool!
"""

agent = ReActAgent(
    tools=query_engine_tools,
    llm=LlamaOpenAI(model="gpt-4o", api_key=api_key),
    system_prompt=system_prompt,
    verbose=True
)

ctx = Context(agent)

requirement = input("\n Enter your Salesforce Flow requirement:\n> ")

# Simple flow type detection
flow_type = "Screen" if any(keyword in requirement.lower() for keyword in ["screen", "user", "input", "display"]) else "AutoLaunched"
print(f"\n Detected flow type: {flow_type}")

# === Use ReAct Agent to retrieve patterns ===
print(f"\n ReAct Agent analyzing {flow_type} Flow requirement...")
print("=" * 60)

agent_query = f"""
Analyze this Salesforce {flow_type} Flow requirement and gather ALL necessary XML patterns:

REQUIREMENT: {requirement}

Please systematically query ALL relevant pattern types:

1. **element_templates_tool**: Get templates for each element needed
   - For Screen flows: Verify fields vs screenFields, fieldText vs label, DisplayText vs Label
   - For AutoLaunched: Verify recordLookups structure, trigger configuration
   
2. **parent_child_relationships_tool**: Get nesting relationships

3. **child_ordering_rules_tool**: Get EXACT child tag ordering
   - CRITICAL: name → label → locationX → locationY → other tags
   
4. **xml_examples_tool**: Get concrete XML examples

5. **attribute_enums_tool**: Get valid enum values
   - Operators: EqualTo, NotEqualTo (PascalCase)
   - FieldTypes: DisplayText, InputField (PascalCase)
   - ProcessType: Flow, AutoLaunchedFlow
   
6. **cooccurrence_rules_tool**: Check element compatibility

7. **context_patterns_tool**: Get context-specific variations

8. **sequence_templates_tool**: Get Flow-level element ordering

9. **positional_constraints_tool**: Get locationX/locationY requirements

Provide comprehensive analysis with:
- All pattern examples
- Child ordering rules for EVERY parent
- Valid enum values
- XML examples showing correct structure
- Co-occurrence rules
- Flow-type-specific validations

Be thorough!
"""

try:
    import asyncio
    
    async def run_agent():
        handler = agent.run(agent_query, ctx=ctx)
        
        async for ev in handler.stream_events():
            from llama_index.core.agent.workflow import AgentStream, ToolCallResult
            if isinstance(ev, ToolCallResult):
                print(f"\n Tool: {ev.tool_name}")
                print(f"   Input: {str(ev.tool_kwargs)[:100]}...")
            elif isinstance(ev, AgentStream):
                print(ev.delta, end="", flush=True)
        
        return await handler
    
    agent_response = asyncio.run(run_agent())
    pattern_guidance = str(agent_response)
    
    print("\n" + "=" * 60)
    print(" Agent analysis complete\n")
    
except Exception as e:
    print(f" Error running ReAct agent: {e}")
    print("Falling back to direct pattern retrieval...")
    response = general_query_engine.query(requirement)
    pattern_guidance = str(response)

# === Build flow-type-specific validation rules ===
if flow_type == "Screen":
    flow_specific_rules = """
**SCREEN FLOW CRITICAL RULES:**

1. **Tag Names (CRITICAL):**
    Use <fields> NOT <screenFields>
    Use <fieldText> NOT <label> for field text
    Use DisplayText NOT Label for fieldType

2. **ProcessType:**
    Must be "Flow" (NOT "AutoLaunchedFlow")

3. **locationX/locationY (MANDATORY):**
   - EVERY element MUST have both tags
   - Placement: name → label → locationX → locationY → other tags
   - Standard spacing: 176px horizontal, 158px vertical
   - Start: locationX=50, locationY=0

4. **Connector Rules:**
   - Use <connector><targetReference>NextElement</targetReference></connector>
   - NO <label> inside <connector>

5. **Valid FieldTypes:**
   DisplayText, InputField, RadioButtons, DropdownBox, LargeTextArea,
   PasswordField, CheckboxGroup, ComponentInstance

6. CRITICAL: Screen Flows ALWAYS require a <start> element with a connector

The <start> element MUST include:

<locationX>50</locationX>
<locationY>0</locationY>
<connector> with <targetReference> pointing to the FIRST screen

REQUIRED STRUCTURE:
<start>
        <locationX>50</locationX>
        <locationY>0</locationY>
        <connector>
            <targetReference>FirstScreenName</targetReference>
        </connector>
    </start>

7. Element Connection Rules:

    MUST HAVE CONNECTORS (non-terminal elements):

    -<start> - ALWAYS needs connector to first element (CRITICAL)
    -<screens> - Needs connector UNLESS it's the final screen
    -<decisions> - Each outcome path needs connector (except default can end)
    -<assignments> - Always needs connector
    -<recordLookups> - Always needs connector
    -<recordUpdates> - Needs connector unless flow ends
    -<recordCreates> - Needs connector unless flow ends
    -<subflows> - Always needs connector
    -<loops> - Loop body needs connectors

CAN BE UNCONNECTED (terminal elements):

 -Final <screens> element (user clicks Finish, flow ends)
 -<recordDeletes> at flow end
 -Decision outcomes that intentionally end the flow
 -Elements after which flow should terminate

ERROR PATTERNS:

 -"Start element not connected" = Missing <connector> in <start> ← YOUR ERROR
 -"Element X has no path" = Middle element missing connector
 -"Unreachable element" = Element exists but nothing points to it

VALIDATION CHECKLIST:
 -<start> element has <connector> with <targetReference>
 -<targetReference> matches first screen's <name> exactly
 -Every middle element has connector to next element
 -Flow has at least one terminal element (endpoint)

 
EXAMPLES:

**Example 1: Simple Screen with Input Field**
```xml
<screens>
    <name>Screen1</name>
    <label>Enter Information</label>
    <locationX>50</locationX>  <!-- ✅ REQUIRED -->
    <locationY>0</locationY>   <!-- ✅ REQUIRED -->
    <fields>  <!-- ✅ Correct tag name -->
        <name>inputField1</name>
        <dataType>String</dataType>
        <fieldText>Enter your name</fieldText>  <!-- ✅ Use fieldText -->
        <fieldType>InputField</fieldType>
        <isRequired>false</isRequired>
    </fields>
    <allowBack>true</allowBack>
    <allowFinish>true</allowFinish>
    <allowPause>false</allowPause>
</screens>
```

**Example 2: Screen with Display Text**
```xml
<screens>
    <name>WelcomeScreen</name>
    <label>Welcome</label>
    <locationX>50</locationX>   <!-- ✅ REQUIRED -->
    <locationY>158</locationY>  <!-- ✅ REQUIRED (below first screen) -->
    <fields>
        <name>displayText1</name>
        <fieldText>Welcome to the flow!</fieldText>
        <fieldType>DisplayText</fieldType>  <!-- ✅ Use DisplayText, not Label -->
    </fields>
    <allowBack>true</allowBack>
    <allowFinish>true</allowFinish>
    <allowPause>false</allowPause>
</screens>
```

**Example 3: Decision Element**
```xml
<decisions>
    <name>CheckValue</name>
    <label>Check Value</label>
    <locationX>50</locationX>   <!-- ✅ REQUIRED -->
    <locationY>316</locationY>  <!-- ✅ REQUIRED -->
    <defaultConnector>
        <targetReference>DefaultScreen</targetReference>
    </defaultConnector>
    <rules>
        <name>IsValid</name>
        <conditionLogic>and</conditionLogic>
        <conditions>
            <leftValueReference>varName</leftValueReference>
            <operator>EqualTo</operator>
            <rightValue>
                <stringValue>Active</stringValue>
            </rightValue>
        </conditions>
        <connector>
            <targetReference>SuccessScreen</targetReference>
        </connector>
        <label>Is Valid</label>
    </rules>
</decisions>
```

**Example 4: Variable Declaration**
```xml
<variables>
    <name>userName</name>
    <dataType>String</dataType>
    <isInput>false</isInput>
    <isOutput>false</isOutput>
    <locationX>50</locationX>   <!-- ✅ REQUIRED even for variables -->
    <locationY>474</locationY>  <!-- ✅ REQUIRED -->
</variables>
```

**Example 5: Record Create**
```xml
<recordCreates>
    <name>CreateAccount</name>
    <label>Create Account</label>
    <locationX>50</locationX>   <!-- ✅ REQUIRED -->
    <locationY>632</locationY>  <!-- ✅ REQUIRED -->
    <inputAssignments>
        <field>Name</field>
        <value>
            <elementReference>userName</elementReference>
        </value>
    </inputAssignments>
    <object>Account</object>
</recordCreates>
```

**Example 6: Complete Multi-Screen Flow**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<Flow xmlns="http://soap.sforce.com/2006/04/metadata">
    <apiVersion>62.0</apiVersion>
    
    <screens>
        <name>Screen1</name>
        <label>Input Screen</label>
        <locationX>50</locationX>
        <locationY>0</locationY>
        <fields>
            <name>nameInput</name>
            <dataType>String</dataType>
            <fieldText>Enter Name</fieldText>
            <fieldType>InputField</fieldType>
            <isRequired>true</isRequired>
        </fields>
        <allowBack>true</allowBack>
        <allowFinish>true</allowFinish>
        <allowPause>false</allowPause>
        <connector>
            <targetReference>Decision1</targetReference>
        </connector>
    </screens>
    
    <decisions>
        <name>Decision1</name>
        <label>Check Input</label>
        <locationX>50</locationX>
        <locationY>158</locationY>
        <defaultConnector>
            <targetReference>Screen2</targetReference>
        </defaultConnector>
        <rules>
            <name>HasValue</name>
            <conditionLogic>and</conditionLogic>
            <conditions>
                <leftValueReference>nameInput</leftValueReference>
                <operator>IsNull</operator>
                <rightValue>
                    <booleanValue>false</booleanValue>
                </rightValue>
            </conditions>
            <connector>
                <targetReference>Screen3</targetReference>
            </connector>
            <label>Has Value</label>
        </rules>
    </decisions>
    
    <screens>
        <name>Screen2</name>
        <label>Error Screen</label>
        <locationX>226</locationX>
        <locationY>158</locationY>
        <fields>
            <name>errorText</name>
            <fieldText>Please provide a valid input</fieldText>
            <fieldType>DisplayText</fieldType>
        </fields>
        <allowBack>true</allowBack>
        <allowFinish>true</allowFinish>
        <allowPause>false</allowPause>
    </screens>
    
    <screens>
        <name>Screen3</name>
        <label>Success Screen</label>
        <locationX>50</locationX>
        <locationY>316</locationY>
        <fields>
            <name>successText</name>
            <fieldText>Thank you!</fieldText>
            <fieldType>DisplayText</fieldType>
        </fields>
        <allowBack>true</allowBack>
        <allowFinish>true</allowFinish>
        <allowPause>false</allowPause>
    </screens>
    
    <label>Sample Flow</label>
    <processType>Flow</processType>
    <status>Draft</status>
</Flow>
```

SCREEN FLOW STRUCTURE REQUIREMENTS

1. **processType**: Must be "Flow"
2. **runInMode**: "SystemModeWithoutSharing" or "SystemModeWithSharing" (optional)
3. **NO <start> element**: Screen flows don't have triggers
4. **Screen structure**:
   - <screens>
     - <name> (required)
     - <label> (required)
     - **<locationX> (REQUIRED)**
     - **<locationY> (REQUIRED)**
     - <fields> (multiple allowed)
       - <name> (required)
       - <fieldText> (for field label text)
       - <fieldType> (required - use DisplayText NOT Label)
       - <dataType> (for input fields)
       - <isRequired> (optional)
     - <allowBack> (boolean)
     - <allowFinish> (boolean)
     - <allowPause> (boolean)
     - <connector> (for navigation)
"""

else:  # AutoLaunched
    flow_specific_rules = """
**AUTOLAUNCHED FLOW CRITICAL RULES:**

1. **ProcessType:**
    Must be "AutoLaunchedFlow"

2. **<start> Element (REQUIRED):**
   - Must include: locationX, locationY, connector
   - For record-triggered: object, recordTriggerType, triggerType
   - For scheduled: scheduledPaths

3. **Record Operations:**
   - RecordLookups: Use <filterLogic> (NOT conditionLogic)
   - Must have: name, label, object, getFirstRecordOnly, queriedFields
   - NO duplicate names

4. **Decisions:**
   - Use <conditionLogic> inside <rules> (NOT in recordLookups)

5. **System Variables:**
   - $Record, $Record.Id, $Record.FieldName
   - $Record__Prior (Update triggers only)

6. **locationX/locationY (MANDATORY):**
   - Required for ALL elements

**COMMON AUTOLAUNCHED ERRORS TO AVOID:**
❌ Using conditionLogic in recordLookups → Use filterLogic
❌ Missing <queriedFields> in recordLookups
❌ Duplicate recordLookups names
❌ Missing <object> in record operations
❌ Using $Record without record trigger

EXAMPLE:
<recordLookups>
    <name>GetRecord</name>
    <label>Get Record</label>
    <filterLogic>and</filterLogic>  <!-- Use filterLogic -->
    <filters>
        <field>Id</field>
        <operator>EqualTo</operator>
        <value><elementReference>recordId</elementReference></value>
    </filters>
    <object>Account</object>
    <getFirstRecordOnly>true</getFirstRecordOnly>
    <storeOutputAutomatically>true</storeOutputAutomatically>
</recordLookups>

**DECISIONS** (Where conditionLogic IS valid):
<decisions>
    <name>CheckCondition</name>
    <label>Check Condition</label>
    <rules>
        <name>outcome1</name>
        <conditionLogic>and</conditionLogic>  <!-- Correct here -->
        <conditions>
            <leftValueReference>var1</leftValueReference>
            <operator>EqualTo</operator>
            <rightValue><stringValue>Active</stringValue></rightValue>
        </conditions>
        <label>If True</label>
        <connector>
            <targetReference>NextElement</targetReference>
        </connector>
    </rules>
    <defaultConnector>
        <targetReference>DefaultElement</targetReference>
    </defaultConnector>
</decisions>

"""

# === Prepare final generation prompt ===
prompt = f"""
Generate valid Salesforce {flow_type} Flow XML for: "{requirement}"

AGENT-RETRIEVED PATTERNS:
{pattern_guidance}

{flow_specific_rules}

CRITICAL XML VALIDATION - 7-Check Process:

1.  Correct tag name (exact case, exists in schema)
2.  Required children present (no missing/invented tags)
3.  Correct parent placement (proper nesting)
4.  Child tag order (name → label → locationX → locationY → others)
5.  Position constraints (locationX/locationY on ALL elements)
6.  Tag compatibility (follow co-occurrence rules)
7.  Exact enum values (DisplayText not Label, EqualTo not equals)

INTEGRATION RULES:
- Use XML examples as structural templates
- Apply child ordering rules EXACTLY as retrieved
- Use ONLY retrieved valid enum values
- Follow co-occurrence rules
- Apply context-specific patterns
- Respect sequence templates
- Follow positional constraints

MANDATORY CHECKS:
- ALL elements have locationX/locationY
- Correct tag names for flow type (fields vs screenFields)
- Correct field labels (fieldText vs label)
- Valid enum values (PascalCase)
- Proper child element ordering
- No conflicting elements
- Correct processType for flow type
- Start element presence matches flow type

OUTPUT:
- Return ONLY XML (no markdown, no explanations)
- Start: <?xml version="1.0" encoding="UTF-8"?>
- Namespace: xmlns="http://soap.sforce.com/2006/04/metadata"
- 4-space indentation
- Schema-compliant, deployment-ready
"""

print(f"Generating {flow_type} Flow XML ")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": f"You are a Salesforce {flow_type} Flow XML generator. Use ALL provided patterns to ensure valid metadata structure."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.1
)

xml_text = response.choices[0].message.content.strip()

# Remove markdown if present
if xml_text.startswith("```xml"):
    xml_text = xml_text.split("```xml")[1].split("```")[0].strip()
elif xml_text.startswith("```"):
    xml_text = xml_text.split("```")[1].split("```")[0].strip()

print(f"\nGenerated {flow_type} Flow XML:\n")
print(xml_text)

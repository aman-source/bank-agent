# app.py
import os, json, pathlib
from datetime import datetime, date
from typing import Optional, List, Literal, Dict, Any

import pytz
from dateutil.relativedelta import relativedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, field_validator

# LangChain (latest)
from langchain_anthropic import ChatAnthropic
# from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser  # keep as-is if you like
from langchain_core.messages import HumanMessage, AIMessage
import os

# ---------------------------
# Constants / Time handling
# ---------------------------

os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")
IST = pytz.timezone("Asia/Kolkata")

def today_str_ddmmyyyy() -> str:
    return datetime.now(IST).strftime("%d-%m-%Y")

def today_iso() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")

def current_fy_bounds(today: Optional[date] = None):
    if today is None:
        today = datetime.now(IST).date()
    year = today.year
    if today.month >= 4:
        start = date(year, 4, 1)
        end = today
    else:
        start = date(year - 1, 4, 1)
        end = today
    return start.isoformat(), end.isoformat()

# ---------------------------
# Load external prompt files (with fallback defaults)
# ---------------------------
def load_text_or_default(path: str, default_text: str) -> str:
    p = pathlib.Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return default_text.strip()


def extract_text_output(result: Any) -> str:
    """
    Normalize AgentExecutor.invoke(...) result to a plain string.
    Handles str, dict with 'output', BaseMessage, and Anthropic content blocks.
    """
    out = result

    # If dict (common AgentExecutor output)
    if isinstance(out, dict):
        out = out.get("output", out)

    # If BaseMessage (AIMessage/HumanMessage/etc.)
    if hasattr(out, "content"):
        out = out.content  # may be str or list of content blocks

    # If Anthropic-style list of blocks or a generic list
    if isinstance(out, list):
        texts = []
        for part in out:
            # Common Anthropic dict block: {"type": "text", "text": "..."}
            if isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    texts.append(part["text"])
                elif "text" in part:
                    texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
            else:
                # Fallback stringify
                try:
                    texts.append(json.dumps(part, ensure_ascii=False))
                except Exception:
                    texts.append(str(part))
        out = "\n".join([t for t in texts if t])

    # Final guard
    if not isinstance(out, str):
        out = str(out)

    return out.strip()


V1_INTENT_PATH = "/mnt/data/v1UserIntent.txt"
REC_PAY_PATH = "/mnt/data/VoiceRecPay.txt"
REPORTS_PATH = "/mnt/data/VoiceReports.txt"

V1_INTENT_DEFAULT = """
You are an intent classifier for business-related queries. Return JSON only:
{ "userIntent" : "" }
Categories:
1 Monthly Sales Summary
2 Draft Mail
3 Send Mail
4 Reports
5 Receipts/Payments
6 Aliases/Nicknames
7 Edit of Receipt / Payment / Sale
8 Create Account
9 New Entry of Sale
10 Generate e-Invoice
11 Generate e-Way Bill
12 Create Item
13 Greetings
14 Account/Item Summary
15 Delete of Receipt/ Payment/ Sale
Rules:
- If payment/receipt amounts are described, choose 5
- If asking ledger/report/BS/PL/TB/etc, choose 4
- If update existing voucher, choose 7
IMPORTANT: Return valid JSON with only userIntent as a stringified number.
Examples:
Input: "Cash received from Swapna" → { "userIntent": "5" }
Input: "Show me ledger of Ramesh" → { "userIntent": "4" }
""".strip()

REC_PAY_DEFAULT = """
Analyse the userQuery and give the output in the given json format.
Strict JSON only, no backslashes, no extra text. English values, dd-MM-yyyy for date.
Take current date as {0} (IST). If Banking → transactionType=Bank else Cash.
Credited means Receipt, Debited means Payment.
We are the company: "Sita sent 6000" → Receipt; "Durga received 3000" → Payment.
Cash Deposit into Bank → Payment with transactionType=Cash, accountName=Bank name.
Cash Withdrawn from Bank → Receipt with transactionType=Cash, accountName=Bank name.
If editing but unspecified, set isEdit="Mention field names".
If unknown values → leave empty, numbers→0.0.
JSON shape:
{
  "mobileNumber" : "",
  "date" : "",
  "transactionType" : "",
  "gstNumber" : "",
  "totalAmount" : 0.0,
  "purpose" : "",
  "bankName" : "",
  "bankName2" : "",
  "details" : [
     {
       "accountName" : "",
       "accountName2" : "",
       "grossAmount" : 0.0,
       "discount" : 0.0,
       "narration" : "",
       "chequeNumber" : "",
       "chequeDate" : "",
       "transferType" : ""
     }
  ],
  "accuracyLevel" : "",
  "isReceipt": "",
  "isEdit":""
}
userQuery =
""".strip()

REPORTS_DEFAULT = """
Analyze the query and return strictly this JSON (no extra text):
{
  "reportName": "",
  "accountName": "",
  "accountName2": "",
  "fromDate": "",
  "toDate": "",
  "withStock": ""
}
Report codes:
BS, PL, TA, LA, DL, CL, DA, CA, CSR, SPSR, TB, SIL, DSR, ISSDW, ITA
Financial year starts April and ends March.
If no dates, use FY start to today ({0}).
withStock = "Y" only if user asked to exclude closing stock; else "N".
- accountName: Capitalized, English
- accountName2: exact user input form
Return JSON only.
The query to analyze is:
""".strip()

INTENT_PROMPT = load_text_or_default(V1_INTENT_PATH, V1_INTENT_DEFAULT)
REC_PAY_PROMPT = load_text_or_default(REC_PAY_PATH, REC_PAY_DEFAULT)
REPORTS_PROMPT = load_text_or_default(REPORTS_PATH, REPORTS_DEFAULT)

# ---------------------------
# Pydantic Schemas (strict JSON payloads)
# ---------------------------
class TransactionDetail(BaseModel):
    accountName: str = ""
    accountName2: str = ""
    grossAmount: float = 0.0
    discount: float = 0.0
    narration: str = ""
    chequeNumber: str = ""
    chequeDate: str = ""
    transferType: str = ""

class TransactionPayload(BaseModel):
    mobileNumber: str = ""
    date: str = ""  # dd-MM-yyyy
    transactionType: Literal["Cash", "Bank", ""] = ""
    gstNumber: str = ""
    totalAmount: float = 0.0
    purpose: str = ""
    bankName: str = ""
    bankName2: str = ""
    details: List[TransactionDetail] = Field(default_factory=lambda: [TransactionDetail()])
    accuracyLevel: str = ""
    isReceipt: Literal["Receipt", "Payment", ""] = ""
    isEdit: str = ""

    @field_validator("date")
    @classmethod
    def _date_ddmmyyyy(cls, v):
        if not v:
            return v
        datetime.strptime(v, "%d-%m-%Y")
        return v

class ReportPayload(BaseModel):
    reportName: str
    accountName: str
    accountName2: str
    fromDate: str  # yyyy-MM-dd
    toDate: str    # yyyy-MM-dd
    withStock: Literal["Y", "N", ""]

    @field_validator("fromDate", "toDate")
    @classmethod
    def _iso_dates(cls, v):
        if not v:
            return v
        datetime.strptime(v, "%Y-%m-%d")
        return v

# ---------------------------
# Anthropic LLM
# ---------------------------
def make_llm():
    # Requires ANTHROPIC_API_KEY in environment
    return ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=0,
        timeout=60,
        max_tokens=1024,
    )

llm = make_llm()

def fill_placeholder(template: str, value: str) -> str:
    return template.replace("{0}", value)


# ---------------------------
# Tools
# ---------------------------

@tool("IntentClassifier", return_direct=False)
def intent_classifier(user_query: str) -> str:
    """
    Classify the query into one of 1..15 categories as JSON: { "userIntent": "5" }.
    """
    prompt = (
        ChatPromptTemplate.from_messages([
            ("system", "{rules}"),   # <-- only one placeholder
            ("human", "{q}")
        ])
        .partial(rules=INTENT_PROMPT)  # <-- inject entire prompt text here
    )
    chain = prompt | llm | StrOutputParser()
    out = chain.invoke({"q": user_query})
    try:
        data = json.loads(out)
        if not isinstance(data, dict) or "userIntent" not in data:
            raise ValueError("Bad intent JSON")
        return json.dumps({"userIntent": str(data["userIntent"])})
    except Exception:
        return json.dumps({"userIntent": "5"})  # safe default


@tool("ReceiptPaymentParser", return_direct=False)
def receipt_payment_parser(user_query: str) -> str:
    """
    Parse a receipt/payment into strict JSON TransactionPayload.
    """
    today_ddmmyyyy = today_str_ddmmyyyy()
    sys_text = fill_placeholder(REC_PAY_PROMPT, today_str_ddmmyyyy()) # keep your date injection

    prompt = (
        ChatPromptTemplate.from_messages([
            ("system", "{rules}"),   # <-- only one placeholder
            ("human", "{q}")
        ])
        .partial(rules=sys_text)     # <-- inject whole rules blob here
    )

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"q": user_query})
    try:
        data = json.loads(raw)
        validated = TransactionPayload(**data)
        return json.dumps(validated.model_dump())
    except Exception:
        fallback = TransactionPayload(date=today_ddmmyyyy, isEdit="")
        return json.dumps(fallback.model_dump())



@tool("ReportParser", return_direct=False)
def report_parser(user_query: str) -> str:
    """
    Parse a report request into strict JSON ReportPayload.
    """
    today = today_iso()
    sys_text = fill_placeholder(REPORTS_PROMPT, today_iso())

    prompt = (
        ChatPromptTemplate.from_messages([
            ("system", "{rules}"),   # <-- only one placeholder
            ("human", "{q}")
        ])
        .partial(rules=sys_text)     # <-- inject whole rules blob here
    )

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"q": user_query})
    try:
        data = json.loads(raw)
        validated = ReportPayload(**data)
        return json.dumps(validated.model_dump())
    except Exception:
        fy_from, fy_to = current_fy_bounds()
        fallback = ReportPayload(
            reportName="LA",
            accountName="",
            accountName2="",
            fromDate=fy_from,
            toDate=fy_to,
            withStock="N",
        )
        return json.dumps(fallback.model_dump())



TOOLS = [intent_classifier, receipt_payment_parser, report_parser]

# ---------------------------
# Agent Prompt
# ---------------------------
SYSTEM_DIRECTIVE = """
You are a smart bookkeeping assistant.
- First, call IntentClassifier to understand the intent (4=Reports, 5=Receipts/Payments, 7=Edit).
- If intent is 5 or 7: call ReceiptPaymentParser with the user's message (and any prior answers if provided).
- If intent is 4: call ReportParser.
- If the JSON from the parser is missing critical fields (e.g., amount 0.0, accountName empty), ASK A SHORT CLARIFYING QUESTION (plain text).
- Once you have enough info, call the parser tool again and RETURN ONLY THE FINAL STRICT JSON (no extra words).
- Never wrap JSON in code fences. Output must be just JSON when finalized.
"""

def build_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_DIRECTIVE),
        MessagesPlaceholder("chat_history"),     # <— was ("placeholder", "{chat_history}")
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")  # <— was ("placeholder", "{agent_scratchpad}")
    ])
    return AgentExecutor(
        agent=create_tool_calling_agent(llm, TOOLS, prompt),
        tools=TOOLS,
        verbose=False,
        handle_parsing_errors=True,
    )


agent_executor = build_agent()

# ---------------------------
# Simple in-memory session store
# ---------------------------
SESSIONS: Dict[str, List[HumanMessage | AIMessage]] = {}

def history_to_text(history: List[Dict[str, str]]) -> str:
    # Flatten to a simple transcript string for the prompt placeholder
    lines = []
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        else:
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)

# ---------------------------
# FastAPI Models
# ---------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated session id")
    query: str = Field(..., description="User message")
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    type: Literal["chat", "transaction", "report"]
    message: Optional[str] = None     # for clarifying questions / text replies
    payload: Optional[Dict[str, Any]] = None  # final strict JSON

class ResetRequest(BaseModel):
    session_id: str

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(title="Bookkeeping Chatbot (LangChain + Anthropic)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "model": "claude-3-5-sonnet-20240620"}

@app.post("/reset", response_model=dict)
def reset_session(req: ResetRequest):
    SESSIONS.pop(req.session_id, None)
    return {"status": "reset", "session_id": req.session_id}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Initialize or fetch session history as a list of BaseMessages
    history = SESSIONS.setdefault(req.session_id, [])

    # Call agent with prior history, and current input separately
    result = agent_executor.invoke({"input": req.query, "chat_history": history})
    raw_out = extract_text_output(result)

    # Try to parse strict JSON
    try:
        data = json.loads(raw_out)
        if {"reportName", "accountName", "accountName2", "fromDate", "toDate", "withStock"}.issubset(data.keys()):
            # Valid report JSON
            ReportPayload(**data)
            # Append this turn to history
            history.append(HumanMessage(content=req.query))
            history.append(AIMessage(content=json.dumps(data, ensure_ascii=False)))
            return ChatResponse(type="report", payload=data)
        else:
            # Valid transaction JSON
            TransactionPayload(**data)
            history.append(HumanMessage(content=req.query))
            history.append(AIMessage(content=json.dumps(data, ensure_ascii=False)))
            return ChatResponse(type="transaction", payload=data)
    except Exception:
        # Not JSON → likely a clarifying question
        history.append(HumanMessage(content=req.query))
        history.append(AIMessage(content=raw_out))
        return ChatResponse(type="chat", message=raw_out)

@app.post("/seed", response_model=list)
def seed_examples():
    samples = [
        "Cash received from Sita 2000",
        "paid to varun",
        "Vikram account statement"
    ]
    outputs = []
    for s in samples:
        res = agent_executor.invoke({"input": s, "chat_history": []})  # [] not ""
        outputs.append(extract_text_output(res))
    return outputs



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",          # module_name:variable_name
        host="0.0.0.0",     # or "127.0.0.1"
        port=8000,
        reload=True
    )

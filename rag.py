from fastapi import FastAPI, Request, HTTPException, Response, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse 
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles 
from starlette.middleware.sessions import SessionMiddleware
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

import os
import io
import json
import sys
from typing import Optional, Dict, Any, List, Tuple
import tempfile
import speech_recognition as sr
from enum import Enum
import re
from config.models import ModelProvider, MODEL_CONFIGS, REPORT_TEMPLATE
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import anthropic
import groq
import openai
import together
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

# ── Embeddings ────────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ── Qdrant client ─────────────────────────────────────────────────────────────
qdrant_client = QdrantClient(url="http://localhost:6333")

# ── Direct Qdrant search (bypasses broken LangChain wrapper) ──────────────────
def get_relevant_docs(query: str, k: int = 5) -> List[Document]:
    try:
        query_vector = embeddings.embed_query(query)

        results = qdrant_client.search(
            collection_name="financial_docs",
            query_vector=query_vector,
            limit=k
        )

        if not results:
            print("⚠️ No results from Qdrant")
            return []

        docs = []
        for hit in results:
            payload = hit.payload or {}

            content = (
                payload.get("page_content")
                or payload.get("content")
                or payload.get("text")
                or payload.get("document")
                or ""
            )

            if not content.strip():
                continue

            docs.append(Document(
                page_content=content,
                metadata=payload.get("metadata", {})
            ))

        return docs

    except Exception as e:
        print("❌ Qdrant Error:", e)
        return []

# ── Request model ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    conversation_context: Optional[str] = None
    language: Optional[str] = "English"

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your-secret-key-change-this")
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")
# ── LLM config ────────────────────────────────────────────────────────────────
config = {
    'max_new_tokens': 512,
    'context_length': 2048,
    'temperature': 0.7,
    'top_p': 0.95,
    'top_k': 50,
    'stream': False,
    'threads': min(4, int(os.cpu_count() / 2)),
}

# ── Prompt templates ──────────────────────────────────────────────────────────
FINANCIAL_QUERY_PROMPT = """Use the following context to answer the question:
Context: {context}
Question: {query}
Answer: Let me help you with that."""

COMPARISON_PROMPT = """Compare the following based on the context provided:
Context: {context}
Question: {query}
Comparison: Let me compare these for you."""

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\swapn\Downloads\Insurance-RAG-LLM-main\Insurance-RAG-LLM-main\models\mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# ── Initialize LLM and verify Qdrant connection ───────────────────────────────
try:
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="mistral",
        config=config
    )
    print("Successfully loaded local model from:", MODEL_PATH)

 
except Exception as e:
    print("Model loading failed:", e)  

# ── Intent classification ─────────────────────────────────────────────────────
class IntentType(Enum):
    COMPARISON = "comparison"
    PRODUCT_INFO = "product_info"
    COST_ANALYSIS = "cost_analysis"
    COVERAGE_DETAILS = "coverage_details"
    ELIGIBILITY = "eligibility"
    CLAIM_PROCESS = "claim_process"
    STANDARD = "standard"

class IntentClassifier:
    def __init__(self):
        self.intent_patterns = {
            IntentType.COMPARISON: [
                r"compare|versus|vs|difference|better|which is|compare between",
                r"(maxlife|lic).*(maxlife|lic)",
                r"which (policy|plan|insurance) (is|would be) better"
            ],
            IntentType.PRODUCT_INFO: [
                r"what (is|are) .*(policy|plan|insurance|coverage)",
                r"tell me about|explain|describe",
                r"features|benefits|details"
            ],
            IntentType.COST_ANALYSIS: [
                r"cost|price|premium|fee|charge|expensive|cheaper",
                r"how much|payment|monthly|annually",
                r"budget|affordable"
            ],
            IntentType.COVERAGE_DETAILS: [
                r"cover|coverage|protect|benefit|claim",
                r"what (does|do|will) .* cover",
                r"maximum|minimum|limit"
            ],
            IntentType.ELIGIBILITY: [
                r"eligible|qualify|who can|requirement",
                r"criteria|condition|age limit",
                r"can I|should I"
            ],
            IntentType.CLAIM_PROCESS: [
                r"claim|process|procedure|file|submit",
                r"how (to|do|can) I claim",
                r"settlement|payout"
            ]
        }

    def _match_patterns(self, text: str, patterns: List[str]) -> float:
        text = text.lower()
        matches = sum(1 for pattern in patterns if re.search(pattern, text))
        return matches / len(patterns) if matches > 0 else 0

    def classify(self, query: str) -> Tuple[IntentType, float]:
        max_score = 0
        intent = IntentType.STANDARD
        for intent_type, patterns in self.intent_patterns.items():
            score = self._match_patterns(query, patterns)
            if score > max_score:
                max_score = score
                intent = intent_type
        return intent, max_score

intent_classifier = IntentClassifier()

def detect_intent(query: str) -> str:
    intent, confidence = intent_classifier.classify(query)
    print(f"Intent: {intent.value}, Confidence: {confidence:.2f}, Query: {query}")
    if confidence >= 0.3:
        return intent.value
    return "standard"

INTENT_PROMPTS = {
    IntentType.COMPARISON.value: """Compare these financial products based on the context:
Context: {context}
Question: {query}
Focus on key differences in features, costs, and benefits.
Comparison:""",

    IntentType.PRODUCT_INFO.value: """Explain this financial product based on the context:
Context: {context}
Question: {query}
Focus on main features and benefits.
Response:""",

    IntentType.COST_ANALYSIS.value: """Analyze the costs based on the context:
Context: {context}
Question: {query}
Focus on pricing, premiums, and payment terms.
Analysis:""",
}

# ── Model manager ─────────────────────────────────────────────────────────────
class ModelManager:
    def __init__(self):
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        for model_name, cfg in MODEL_CONFIGS.items():
            try:
                if cfg["provider"] == ModelProvider.LOCAL:
                    self.models[model_name] = self._init_local_model(cfg)
                elif cfg["provider"] == ModelProvider.GROQ:
                    self.models[model_name] = self._init_groq_model(cfg)
            except Exception as e:
                print(f"Failed to initialize model {model_name}: {e}")

    def _init_local_model(self, cfg):
        return CTransformers(
            model=cfg["MODEL_PATH"],
            model_type=cfg["model_type"],
            config=cfg["config"]
        )

    def _init_groq_model(self, cfg):
        return groq.Groq(api_key=cfg["api_key"])

    async def generate_response(self, model_name: str, prompt: str, **kwargs):
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")

        cfg = MODEL_CONFIGS[model_name]

        if cfg["provider"] == ModelProvider.LOCAL:
            return model(prompt, **kwargs)

        elif cfg["provider"] == ModelProvider.GROQ:
            response = await model.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=cfg["model_id"],
                **cfg["config"]
            )
            return response.choices[0].message.content

model_manager = ModelManager()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/query_new")
async def process_query_new(
    request: Request,
    body: QueryRequest,
    model_name: str = "local-mistral"
):
    try:
        query = body.query.strip()
        if not query:
            return JSONResponse(content={"query": "", "response": "Please enter a query"})

        try:
            # Direct Qdrant search — no LangChain wrapper
            docs = get_relevant_docs(query, k=5)

            if not docs:
                return JSONResponse(content={
                    "query": query,
                    "response": "I don't have enough information to answer that question."
                })

            context = " ".join([
                doc.page_content.strip()
                for doc in docs[:2]
                if doc.page_content.strip()
            ])

            try:
                intent = detect_intent(query)
                prompt_template = INTENT_PROMPTS.get(intent, FINANCIAL_QUERY_PROMPT)
                full_prompt = prompt_template.format(context=context[:1024], query=query)

                response = await model_manager.generate_response(
                    model_name=model_name,
                    prompt=full_prompt,
                    **MODEL_CONFIGS[model_name]["config"]
                )

                # Store in session
                history = request.session.get("conversation_history", [])
                history.append({"query": query, "response": response})
                request.session["conversation_history"] = history

                if not response or not response.strip():
                    return JSONResponse(content={
                        "query": query,
                        "response": "I understand your question but couldn't generate a proper response. Please try rephrasing."
                    })

                return JSONResponse(content={
                    "query": query,
                    "response": response.strip(),
                    "model": model_name
                })

            except Exception as e:
                print(f"LLM Generation Error: {str(e)}")
                return JSONResponse(content={
                    "query": query,
                    "response": "I encountered an issue while processing your query. Please try again."
                })

        except Exception as e:
            print(f"Document Retrieval Error: {str(e)}")
            return JSONResponse(content={
                "query": query,
                "response": "Error accessing the knowledge base. Please try again."
            })

    except Exception as e:
        print(f"General Error: {str(e)}")
        return JSONResponse(content={
            "query": query if 'query' in locals() else "",
            "response": "An unexpected error occurred. Please try again."
        })


@app.post("/query")
async def query_alias(request: Request, body: QueryRequest):
    return await process_query_new(request, body)


@app.post("/query_voice")
async def query_voice(
    request: Request,
    file: UploadFile = File(...),
    conversation_context: Optional[str] = None,
    language: Optional[str] = "English"
):
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(contents)
            tmp.flush()
            tmp_name = tmp.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_name) as source:
            audio_data = recognizer.record(source)
            language_code = "en-US" if language.lower() == "english" else None
            text = (
                recognizer.recognize_google(audio_data, language=language_code)
                if language_code
                else recognizer.recognize_google(audio_data)
            )

        query_body = QueryRequest(
            query=text,
            conversation_context=conversation_context,
            language=language
        )
        return await process_query_new(request, query_body)

    except Exception as e:
        print(f"Voice query error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice query failed: {str(e)}")


@app.get("/ping")
async def ping():
    return {"message": "pong"}


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


def search_financial_info(query: str) -> List[dict]:
    docs = get_relevant_docs(query)
    return [
        {
            "type": "document",
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown")
        }
        for doc in docs
    ]


@app.post("/generate_report")
async def generate_report(
    request: Request,
    model_name: str = "local-mistral",
    format: str = "pdf"
):
    conversation_history = request.session.get("conversation_history", [])

    report_prompt = REPORT_TEMPLATE.format(
        summary="Summarize our conversation",
        points="Extract key points",
        recommendations="Provide recommendations",
        next_steps="Suggest next steps",
        model_name=model_name,
        date=datetime.now().strftime("%Y-%m-%d")
    )

    report_content = await model_manager.generate_response(
        model_name=model_name,
        prompt=report_prompt
    )

    if format == "pdf":
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("Financial Consultation Report", title_style))
        story.append(Spacer(1, 12))

        for line in report_content.split('\n'):
            if line.strip():
                story.append(Paragraph(line, styles["Normal"]))
                story.append(Spacer(1, 12))

        doc.build(story)
        buffer.seek(0)

        return Response(
            content=buffer.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=financial_report.pdf"}
        )

    return JSONResponse(content={"report": report_content})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
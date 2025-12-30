from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import RateLimitError, APIError

from src.app.config import get_settings
from src.app.knowledge_base import KnowledgeBase
from src.app.pipeline import QAEngine
from src.app.retriever import Retriever
from dotenv import load_dotenv
load_dotenv()
 #开启 uvicorn src.main:app --reload 访问 http://127.0.0.1:8000/ 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings = get_settings()
    kb = KnowledgeBase(settings.data_path)
    retriever = Retriever(kb)
    # Only initialize QAEngine if API key is available
    try:
        app.state.qa_engine = QAEngine(kb=kb, retriever=retriever)
    except ValueError:
        # API key not set, qa_engine will be None
        app.state.qa_engine = None
    yield
    # Shutdown


app = FastAPI(title="Drug Q&A", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    last_updated: list[str]


@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QueryRequest):
    if app.state.qa_engine is None:
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY is required for generation. Please set the DEEPSEEK_API_KEY environment variable."
        )
    try:
        result = app.state.qa_engine.generate_answer(payload.question)
    except (RateLimitError, APIError) as exc:
        # DeepSeek API 错误：检查是否是额度不足或 429 错误
        error_str = str(exc).lower()
        error_code = ''
        
        # 尝试从异常对象获取错误码
        if hasattr(exc, 'response'):
            try:
                if hasattr(exc.response, 'json'):
                    error_data = exc.response.json()
                    error_code = error_data.get('error', {}).get('code', '')
                elif hasattr(exc.response, 'error'):
                    error_code = getattr(exc.response.error, 'code', '')
            except:
                pass
        
        # 检查是否是额度不足错误
        if error_code == 'insufficient_quota' or 'insufficient_quota' in error_str:
            raise HTTPException(
                status_code=429,
                detail="DeepSeek API 额度不足（insufficient_quota）。请在 DeepSeek 平台绑定计费或充值后再试。"
            ) from exc
        
        # 检查是否是 429 错误（速率限制）
        if isinstance(exc, RateLimitError) or '429' in error_str or 'rate limit' in error_str:
            raise HTTPException(
                status_code=429,
                detail="API 请求频率过高，已达到速率限制。请稍后再试。"
            ) from exc
        
        # 其他 API 错误
        raise HTTPException(
            status_code=502,
            detail=f"API 服务错误：{str(exc)}"
        ) from exc
    except ValueError as exc:  # likely missing API key
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        # 其他未预期的错误：检查是否包含额度不足信息
        error_str = str(exc).lower()
        if 'insufficient_quota' in error_str:
            raise HTTPException(
                status_code=429,
                detail="DeepSeek API 额度不足（insufficient_quota）。请在 DeepSeek 平台绑定计费或充值后再试。"
            ) from exc
        raise HTTPException(
            status_code=500,
            detail=f"处理请求时发生错误：{str(exc)}"
        ) from exc
    return result


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def landing_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Drug Q&A</title>
        <style>
            body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; padding: 24px; max-width: 960px; margin: auto; background: #f8fafc; color: #0f172a; }
            h1 { margin-bottom: 0.25rem; }
            p.description { margin-top: 0; color: #475569; }
            .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08); }
            label { display: block; font-weight: 600; margin-bottom: 8px; }
            textarea { width: 100%; min-height: 120px; padding: 12px; font-size: 1rem; border-radius: 8px; border: 1px solid #cbd5e1; resize: vertical; box-sizing: border-box; }
            button { margin-top: 12px; background: #2563eb; color: white; border: none; padding: 12px 16px; border-radius: 8px; cursor: pointer; font-size: 1rem; }
            button:disabled { background: #94a3b8; cursor: not-allowed; }
            .answer { margin-top: 20px; white-space: pre-wrap; }
            .meta { margin-top: 10px; font-size: 0.95rem; color: #475569; }
            .error { color: #b91c1c; margin-top: 10px; }
            .disclaimer { margin-top: 20px; font-size: 0.9rem; color: #475569; background: #f1f5f9; padding: 12px; border-radius: 8px; }
        </style>
    </head>
    <body>
        <h1>药物信息问答</h1>
        <p class="description">提供通用药物信息，仅供参考。请务必咨询医生或药师获得个性化建议。</p>
        <div class="card">
            <label for="question">请输入问题</label>
            <textarea id="question" placeholder="例如：对乙酰氨基酚的常见副作用是什么？"></textarea>
            <button id="askBtn">询问</button>
            <div id="error" class="error" role="alert"></div>
            <div id="answer" class="answer"></div>
            <div id="meta" class="meta"></div>
            <div class="disclaimer">免责声明：本工具不提供个性化医疗建议，信息可能过时或不完整。请咨询专业医疗人员。</div>
        </div>
        <script>
            const btn = document.getElementById("askBtn");
            const questionEl = document.getElementById("question");
            const answerEl = document.getElementById("answer");
            const metaEl = document.getElementById("meta");
            const errorEl = document.getElementById("error");

            async function ask() {
                const question = questionEl.value.trim();
                errorEl.textContent = "";
                answerEl.textContent = "";
                metaEl.textContent = "";
                if (!question) {
                    errorEl.textContent = "请输入问题。";
                    return;
                }
                btn.disabled = true;
                btn.textContent = "处理中...";
                try {
                    const res = await fetch("/ask", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ question })
                    });
                    if (!res.ok) {
                        const detail = await res.json().catch(() => ({}));
                        throw new Error(detail.detail || "请求失败");
                    }
                    const data = await res.json();
                    answerEl.textContent = data.answer || "未返回答案";
                    const sources = (data.sources || []).join(", ");
                    const updated = (data.last_updated || []).join(", ");
                    metaEl.textContent = sources ? `来源: ${sources} | 更新: ${updated}` : "";
                } catch (err) {
                    errorEl.textContent = err.message || "发生错误";
                } finally {
                    btn.disabled = false;
                    btn.textContent = "询问";
                }
            }

            btn.addEventListener("click", ask);
            questionEl.addEventListener("keydown", (ev) => {
                if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey || ev.shiftKey)) {
                    ask();
                }
            });
        </script>
    </body>
    </html>
    """

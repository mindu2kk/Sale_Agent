"""
Factory that wires together all the AI components into a VerificationWorkflow.

Supports both Google Gemini (GOOGLE_API_KEY) and OpenAI (OPENAI_API_KEY).
Gemini is tried first.

Singleton pattern — expensive to init, created once and reused.
"""

import os
import logging
import threading

logger = logging.getLogger(__name__)

_workflow = None
_workflow_lock = threading.Lock()
_workflow_error: Exception | None = None


def get_workflow():
    """Return the singleton VerificationWorkflow. Creates it on first call."""
    global _workflow, _workflow_error
    if _workflow is not None:
        return _workflow
    if _workflow_error is not None:
        raise _workflow_error
    with _workflow_lock:
        # Double-check after acquiring lock
        if _workflow is None and _workflow_error is None:
            try:
                _workflow = _build_workflow()
            except Exception as exc:
                _workflow_error = exc
                raise
    return _workflow


def _patch_react_agent():
    """
    Full compatibility patch for LlamaIndex >= 0.12 where ReActAgent API changed:
    - from_tools() removed
    - .chat() removed (replaced by async .run())
    - response.sources removed (replaced by response.tool_calls)

    Creates a _CompatReActAgent wrapper that restores all three for SalesResearchAgent.
    """
    try:
        import llama_index.core.agent as _agent_module
        from llama_index.core.agent.workflow import ReActAgent as _NewReActAgent

        # Already patched or old version with from_tools — skip
        if hasattr(_agent_module.ReActAgent, "from_tools"):
            return

        logger.info("Patching ReActAgent for new LlamaIndex API...")

        # ── Legacy response wrapper ────────────────────────────────────────────

        class _LegacySource:
            def __init__(self, tool_name, raw_input="", raw_output=""):
                self.tool_name = tool_name
                self.raw_input = raw_input
                self.raw_output = raw_output

        class _LegacyAgentResponse:
            """Wraps new AgentOutput to restore .sources and str() for SalesResearchAgent."""

            def __init__(self, agent_output):
                self._output = agent_output
                # Rebuild .sources from .tool_calls
                sources = []
                for tc in getattr(agent_output, "tool_calls", []) or []:
                    name = (
                        getattr(tc, "tool_name", None)
                        or getattr(tc, "name", None)
                        or "unknown"
                    )
                    sources.append(_LegacySource(
                        tool_name=name,
                        raw_input=str(getattr(tc, "tool_kwargs", {})),
                        raw_output="",
                    ))
                self.sources = sources

            def __str__(self):
                val = getattr(self._output, "response", None)
                if val is None:
                    val = str(self._output)
                return str(val)

        # ── Compat agent ───────────────────────────────────────────────────────

        class _CompatReActAgent:
            """Restores from_tools() and .chat() for legacy SalesResearchAgent code."""

            def __init__(self, tools, llm=None, verbose=False,
                         max_iterations=10, context="", **kwargs):
                self._inner = _NewReActAgent(
                    tools=tools,
                    llm=llm,
                    verbose=verbose,
                    max_iterations=max_iterations,
                    context=context,
                )
                self._LegacyAgentResponse = _LegacyAgentResponse

            @classmethod
            def from_tools(cls, tools, llm=None, verbose=False,
                           max_iterations=10, context="", **kwargs):
                return cls(
                    tools=tools, llm=llm, verbose=verbose,
                    max_iterations=max_iterations, context=context,
                )

            def chat(self, message: str) -> _LegacyAgentResponse:
                """Sync .chat() wrapper around async agent.run()."""
                import asyncio

                async def _run():
                    return await self._inner.run(message)

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        future = asyncio.run_coroutine_threadsafe(_run(), loop)
                        result = future.result(timeout=120)
                    else:
                        result = loop.run_until_complete(_run())
                except RuntimeError:
                    result = asyncio.run(_run())

                return _LegacyAgentResponse(result)

        _agent_module.ReActAgent = _CompatReActAgent  # type: ignore[attr-defined]
        logger.info("ReActAgent patch applied ✓ (from_tools + chat + sources restored)")

    except Exception as exc:
        logger.warning("ReActAgent patch failed: %s", exc)


def _build_llm():
    """Build LLM. Priority: GOOGLE_API_KEY (Gemini) → OPENAI_API_KEY (GPT)."""
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if google_key:
        model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        logger.info("Using Google Gemini: %s", model)
        try:
            from llama_index.llms.google_genai import GoogleGenAI
            llm = GoogleGenAI(model=model, api_key=google_key)
            return llm, model
        except Exception as exc:
            raise RuntimeError(f"Gemini init failed: {exc}") from exc

    if openai_key:
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        logger.info("Using OpenAI: %s", model)
        from llama_index.llms.openai import OpenAI
        llm = OpenAI(model=model, temperature=0.1, api_key=openai_key)
        return llm, model

    raise RuntimeError(
        "No LLM API key found. "
        "Add GOOGLE_API_KEY=... or OPENAI_API_KEY=... to .env"
    )


def _build_workflow():
    """Instantiate and wire all AI components."""
    logger.info("Initialising AI components — this may take ~60s on first run...")

    # Patch FIRST before any imports that use ReActAgent
    _patch_react_agent()

    # ── LLM ───────────────────────────────────────────────────────────────────
    try:
        llm, model_name = _build_llm()
        logger.info("LLM ready: %s", model_name)
    except Exception as exc:
        logger.error("LLM init failed: %s", exc)
        raise RuntimeError(f"LLM init failed: {exc}") from exc

    # ── Config ────────────────────────────────────────────────────────────────
    try:
        from verification.config.config import get_config
        config = get_config()
        config.llm_model_name = model_name
    except Exception as exc:
        logger.error("Config load failed: %s", exc)
        raise RuntimeError(f"Config init failed: {exc}") from exc

    # ── RAG Pipeline ──────────────────────────────────────────────────────────
    try:
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from retriever.hybrid_retriever import HybridRetriever
        from retriever.relevance_checker import RelevanceChecker
        from rag_pipeline import RAGPipeline

        # device="cpu" avoids "Cannot copy out of meta tensor" PyTorch issue
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device="cpu")
        Settings.embed_model = embed_model
        Settings.llm = llm  # prevent LlamaIndex defaulting to OpenAI

        retriever = HybridRetriever(
            docstore_path="./chroma_db/docstore.json",
            chroma_path="./chroma_db",
            embed_model_name="BAAI/bge-m3",
        )
        checker = RelevanceChecker(llm=llm)
        rag_pipeline = RAGPipeline(retriever=retriever, checker=checker)
        logger.info("RAG pipeline ready")
    except Exception as exc:
        logger.error("RAG pipeline init failed: %s", exc)
        raise RuntimeError(f"RAG pipeline init failed: {exc}") from exc

    # ── Agents ────────────────────────────────────────────────────────────────
    try:
        from agent.sales_research_agent import SalesResearchAgent
        from verification.agent.verification_agent import VerificationAgent

        tavily_key = os.getenv("TAVILY_API_KEY")
        research_agent = SalesResearchAgent(
            llm=llm,
            rag_pipeline=rag_pipeline,
            tavily_api_key=tavily_key,
        )
        verification_agent = VerificationAgent(
            llm=llm,
            rag_pipeline=rag_pipeline,
            config=config,
        )
        logger.info("Agents initialised (tavily=%s)", "yes" if tavily_key else "no")
    except Exception as exc:
        logger.error("Agent init failed: %s", exc)
        raise RuntimeError(f"Agent init failed: {exc}") from exc

    # ── Workflow ───────────────────────────────────────────────────────────────
    try:
        from verification.workflow.workflow import VerificationWorkflow
        workflow = VerificationWorkflow(
            research_agent=research_agent,
            verification_agent=verification_agent,
            config=config,
        )
        logger.info("VerificationWorkflow ready ✓  (model=%s)", model_name)
        return workflow
    except Exception as exc:
        logger.error("Workflow build failed: %s", exc)
        raise RuntimeError(f"Workflow build failed: {exc}") from exc

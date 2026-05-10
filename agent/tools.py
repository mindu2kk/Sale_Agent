import json
import os

from llama_index.core.tools import FunctionTool


def build_internal_db_tool(rag_pipeline, cache=None):
    """Factory that creates an internal_db_search FunctionTool with a closure over rag_pipeline."""

    def internal_db_search(query: str) -> str:
        """Tra cứu thông tin sản phẩm, giá, thông số kỹ thuật và chính sách
        bảo hành/đổi trả từ cơ sở dữ liệu nội bộ của công ty.
        Luôn gọi tool này TRƯỚC khi dùng web search."""
        try:
            if cache is not None:
                cached = cache.get(query)
                if cached is not None:
                    import logging
                    logging.getLogger(__name__).debug("Cache hit for query: %s", query)
                    return cached

            result = rag_pipeline.query(query)
            if isinstance(result, str):
                result_json = json.dumps({"status": "NO_MATCH", "message": result})
            else:
                formatted_nodes = []
                for nws in result:
                    formatted_nodes.append({
                        "source": nws.node.metadata.get("source_type", "Unknown"),
                        "product_code": nws.node.metadata.get("product_code", "N/A"),
                        "content": nws.node.text[:500] + "...",
                    })
                result_json = json.dumps(formatted_nodes, ensure_ascii=False)

            if cache is not None:
                cache.set(query, result_json)
            return result_json
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": str(e)})

    return FunctionTool.from_defaults(
        fn=internal_db_search,
        name="internal_db_search",
        description=(
            "Tra cứu thông tin sản phẩm, giá, thông số kỹ thuật và chính sách "
            "bảo hành/đổi trả từ cơ sở dữ liệu nội bộ của công ty. "
            "Luôn gọi tool này TRƯỚC khi dùng web search."
        ),
    )


def build_tavily_tool(tavily_api_key: str | None = None):
    """Factory that creates a tavily_web_search FunctionTool.
    Returns None if no API key is available."""
    api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None

    from tavily import TavilyClient
    client = TavilyClient(api_key=api_key)

    def tavily_web_search(query: str) -> str:
        """Tìm kiếm thông tin thị trường chung, tin tức công nghệ, tỷ giá,
        hoặc xu hướng sản phẩm từ Internet. CHỈ dùng khi Internal_DB_Tool
        hoàn toàn không có thông tin. Dữ liệu có thể cũ hoặc không chính xác về giá."""
        try:
            response = client.search(query, max_results=3)
            results = response.get("results", [])
            return json.dumps(
                [{"title": r.get("title"), "content": r.get("content", "")[:500]} for r in results],
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"status": "ERROR", "message": str(e)})

    return FunctionTool.from_defaults(
        fn=tavily_web_search,
        name="tavily_web_search",
        description=(
            "Tìm kiếm thông tin thị trường chung, tin tức công nghệ, tỷ giá, "
            "hoặc xu hướng sản phẩm từ Internet. CHỈ dùng khi Internal_DB_Tool "
            "hoàn toàn không có thông tin. Dữ liệu có thể cũ hoặc không chính xác về giá."
        ),
    )

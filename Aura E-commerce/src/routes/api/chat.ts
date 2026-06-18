import { createFileRoute } from "@tanstack/react-router";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { generateText } from "ai";
import { z } from "zod";
import { findProduct, getBundleFor, productCatalogPrompt } from "@/lib/products";

const RequestSchema = z.object({
  message: z.string().min(1).max(2000),
});

const SYSTEM_PROMPT = `Bạn là tư vấn viên cao cấp của một cửa hàng điện tử Apple-style tại Việt Nam.
Phong cách: lịch sự, ngắn gọn, tinh tế, dùng "Dạ", "ạ". Tránh dùng emoji.
Trả lời 1-3 câu, không liệt kê dài dòng.

Danh mục sản phẩm hiện có:
${productCatalogPrompt}

QUAN TRỌNG: Bạn PHẢI trả về DUY NHẤT một đối tượng JSON hợp lệ (không kèm markdown, không kèm \`\`\`) theo schema:
{"text": "<câu trả lời tiếng Việt>", "ui_type": "default|product|comparison|checkout|bento", "product_id": "<id|null>", "compare_ids": ["<id>","<id>"] | null}

- ui_type="comparison" khi khách muốn so sánh 2 sản phẩm (vd "so sánh X và Y", "X vs Y"). compare_ids phải là 2 id hợp lệ khác nhau.
- ui_type="checkout" khi khách thể hiện rõ ý định mua/đặt hàng/chốt đơn ("mua ngay", "chốt", "đặt hàng", "lấy con này"). product_id là sản phẩm cần mua.
- ui_type="bento" khi khách hỏi về các tính năng/điểm nổi bật/thông số của 1 sản phẩm cụ thể ("có gì hay", "điểm nổi bật", "thông số", "đặc điểm"). product_id là sản phẩm đó.
- ui_type="product" khi khách hỏi/quan tâm 1 sản phẩm cụ thể. product_id là id sản phẩm phù hợp nhất.
- ui_type="default" cho chào hỏi/câu hỏi chung. product_id=null, compare_ids=null.
- Chỉ chọn id từ danh mục ở trên. Không bịa id.`;

export const Route = createFileRoute("/api/chat")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        let body: unknown;
        try {
          body = await request.json();
        } catch {
          return new Response("Invalid JSON", { status: 400 });
        }
        const parsed = RequestSchema.safeParse(body);
        if (!parsed.success) {
          return new Response("Invalid request", { status: 400 });
        }

        const apiKey = process.env.LOVABLE_API_KEY;
        if (!apiKey) {
          return new Response("Missing LOVABLE_API_KEY", { status: 500 });
        }

        const provider = createOpenAICompatible({
          name: "lovable",
          baseURL: "https://ai.gateway.lovable.dev/v1",
          headers: {
            "Lovable-API-Key": apiKey,
            "X-Lovable-AIG-SDK": "vercel-ai-sdk",
          },
        });

        try {
          const { text } = await generateText({
            model: provider("google/gemini-3-flash-preview"),
            system: SYSTEM_PROMPT,
            prompt: parsed.data.message,
          });

          // Parse JSON from model output, tolerant of code fences.
          const cleaned = text
            .replace(/^```(?:json)?\s*/i, "")
            .replace(/```\s*$/i, "")
            .trim();
          let parsedOut: {
            text?: string;
            ui_type?: "default" | "product" | "comparison" | "checkout" | "bento";
            product_id?: string | null;
            compare_ids?: string[] | null;
          };
          try {
            parsedOut = JSON.parse(cleaned);
          } catch {
            // Fallback: use raw text
            return Response.json({
              text: cleaned || "Dạ, em chưa hiểu rõ ý anh/chị. Anh/chị có thể nói thêm được không ạ?",
              has_product: false,
              product_data: null,
              ui_type: "default",
              compare_ids: null,
            });
          }

          const uiType = parsedOut.ui_type ?? "default";
          const product = parsedOut.product_id ? findProduct(parsedOut.product_id) : undefined;
          const compareIds =
            uiType === "comparison" && Array.isArray(parsedOut.compare_ids)
              ? parsedOut.compare_ids.slice(0, 2)
              : null;
          const validCompare =
            compareIds && compareIds.length === 2 && compareIds.every((id) => findProduct(id))
              ? (compareIds as [string, string])
              : null;

          return Response.json({
            text: parsedOut.text ?? "",
            has_product: Boolean(product) && uiType !== "comparison",
            product_data: product ?? null,
            suggest_bundle: Boolean(product) && uiType === "product",
            bundle: product && uiType === "product" ? getBundleFor(product.id) : null,
            ui_type: uiType,
            compare_ids: validCompare,
          });
        } catch (err) {
          const msg = err instanceof Error ? err.message : "Unknown error";
          if (msg.includes("429")) {
            return new Response("Rate limited", { status: 429 });
          }
          if (msg.includes("402")) {
            return new Response("Credits exhausted", { status: 402 });
          }
          return new Response(`AI error: ${msg}`, { status: 500 });
        }
      },
    },
  },
});
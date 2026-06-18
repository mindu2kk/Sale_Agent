import type { Product } from "@/lib/products";
import { Plus } from "lucide-react";
import { useCopilot } from "@/lib/copilot-store";
import { useNavigate } from "@tanstack/react-router";
import { Spatial3DViewer } from "./Spatial3DViewer";

export function InlineProductCard({ product }: { product: Product }) {
  const triggerCheckout = useCopilot((s) => s.triggerCheckout);
  const navigate = useNavigate();

  const openDetail = () => {
    const go = () => navigate({ to: "/product/$id", params: { id: product.id } });
    if (typeof document !== "undefined" && document.startViewTransition) {
      document.startViewTransition(go);
    } else {
      go();
    }
  };

  return (
    <>
    <div className="copilot-fade-in mt-3 flex items-center gap-3 rounded-xl border border-gray-200 bg-white p-3">
      <button
        type="button"
        onClick={openDetail}
        className="flex h-20 w-20 shrink-0 items-center justify-center rounded-lg bg-[#fbfbfd] transition-transform hover:scale-[1.03]"
        aria-label={`Xem chi tiết ${product.name}`}
      >
        <img
          src={product.image}
          alt={product.name}
          width={80}
          height={80}
          style={{ viewTransitionName: `product-image-${product.id}` }}
          className="h-16 w-16 object-contain"
        />
      </button>
      <div className="min-w-0 flex-1">
        <button
          type="button"
          onClick={openDetail}
          className="truncate text-left text-[14px] font-semibold text-neutral-900 hover:underline"
        >
          {product.name}
        </button>
        <p className="truncate text-[12px] font-light text-neutral-500">{product.category}</p>
        <p className="mt-1 text-[13px] text-neutral-800">{product.price}</p>
      </div>
      <div className="flex flex-col gap-1.5">
        <button
          type="button"
          onClick={() => triggerCheckout(product)}
          className="flex h-9 items-center justify-center rounded-full bg-neutral-900 px-3 text-[12px] font-medium text-white transition-colors hover:bg-neutral-800"
        >
          Mua ngay
        </button>
        <button
          type="button"
          aria-label="Thêm vào giỏ"
          className="flex h-7 items-center justify-center gap-1 rounded-full border border-gray-200 px-3 text-[11px] font-medium text-neutral-700 transition-colors hover:border-neutral-400"
        >
          <Plus className="h-3 w-3" />
          Thêm
        </button>
      </div>
    </div>
    {product.premium ? <Spatial3DViewer product={product} /> : null}
    </>
  );
}
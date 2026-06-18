import { useEffect, useRef } from "react";
import type { Product } from "@/lib/products";
import { useCopilot } from "@/lib/copilot-store";

export function ProductCard({ product }: { product: Product }) {
  const ref = useRef<HTMLElement>(null);
  const setActiveCategory = useCopilot((s) => s.setActiveCategory);

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && entry.intersectionRatio > 0.5) {
          setActiveCategory(product.category);
        }
      },
      { threshold: [0, 0.5, 1] }
    );
    observer.observe(el);
    const onEnter = () => setActiveCategory(product.category);
    el.addEventListener("mouseenter", onEnter);
    return () => {
      observer.disconnect();
      el.removeEventListener("mouseenter", onEnter);
    };
  }, [product.category, setActiveCategory]);

  return (
    <article ref={ref} className="group flex flex-col">
      <div className="relative aspect-square w-full overflow-hidden rounded-2xl bg-[#fbfbfd]">
        <img
          src={product.image}
          alt={product.name}
          loading="lazy"
          width={1024}
          height={1024}
          className="absolute inset-0 h-full w-full object-contain p-10 transition-transform duration-500 group-hover:scale-105"
        />
        <div className="absolute inset-x-0 bottom-0 flex justify-center pb-5 opacity-0 transition-opacity duration-300 group-hover:opacity-100">
          <button className="rounded-full bg-neutral-900 px-5 py-2 text-[12px] font-medium text-white shadow-sm transition-colors hover:bg-neutral-800">
            Mua ngay
          </button>
        </div>
      </div>
      <div className="mt-5 px-1">
        <p className="text-[11px] font-medium uppercase tracking-[0.15em] text-neutral-500">
          {product.category}
        </p>
        <h3 className="mt-1 text-base font-medium text-neutral-900">{product.name}</h3>
        <p className="mt-1 text-sm font-light text-neutral-500">{product.tagline}</p>
        <p className="mt-3 text-sm text-neutral-700">{product.price}</p>
      </div>
    </article>
  );
}
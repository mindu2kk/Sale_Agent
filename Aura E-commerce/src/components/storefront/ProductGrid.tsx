import { products } from "@/lib/products";
import { ProductCard } from "./ProductCard";

export function ProductGrid() {
  return (
    <section id="products" className="bg-white">
      <div className="mx-auto max-w-6xl px-6 py-24">
        <div className="mb-16 flex items-end justify-between">
          <div>
            <p className="text-[13px] font-medium uppercase tracking-[0.2em] text-neutral-500">
              Sản phẩm
            </p>
            <h2 className="mt-3 text-3xl font-light tracking-tight text-neutral-900 md:text-4xl">
              Được chế tác cho mỗi khoảnh khắc.
            </h2>
          </div>
          <a
            href="#"
            className="hidden text-[13px] font-medium text-neutral-900 underline-offset-4 hover:underline md:block"
          >
            Xem tất cả →
          </a>
        </div>
        <div className="grid grid-cols-1 gap-x-10 gap-y-16 md:grid-cols-2 lg:grid-cols-3">
          {products.map((p) => (
            <ProductCard key={p.id} product={p} />
          ))}
        </div>
      </div>
    </section>
  );
}
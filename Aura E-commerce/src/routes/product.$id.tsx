import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { ArrowLeft } from "lucide-react";
import { findProduct } from "@/lib/products";

export const Route = createFileRoute("/product/$id")({
  head: ({ params }) => {
    const p = findProduct(params.id);
    return {
      meta: [
        { title: p ? `${p.name} — Cửa hàng` : "Sản phẩm" },
        { name: "description", content: p?.tagline ?? "Chi tiết sản phẩm" },
      ],
    };
  },
  component: ProductDetail,
  notFoundComponent: () => <p className="p-10">Không tìm thấy sản phẩm</p>,
});

function ProductDetail() {
  const { id } = Route.useParams();
  const product = findProduct(id);
  const navigate = useNavigate();

  if (!product) {
    return (
      <div className="mx-auto max-w-4xl px-6 py-20 text-center">
        <p className="text-neutral-600">Sản phẩm không tồn tại.</p>
        <Link to="/" className="mt-4 inline-block text-sm underline">
          Về trang chủ
        </Link>
      </div>
    );
  }

  const goBack = () => {
    if (document.startViewTransition) {
      document.startViewTransition(() => navigate({ to: "/" }));
    } else {
      navigate({ to: "/" });
    }
  };

  return (
    <main className="min-h-screen bg-white">
      <header className="mx-auto flex max-w-6xl items-center px-6 py-5">
        <button
          type="button"
          onClick={goBack}
          className="inline-flex items-center gap-1.5 rounded-full border border-gray-200 px-3 py-1.5 text-[12px] font-medium text-neutral-700 transition-colors hover:border-neutral-400"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Quay lại
        </button>
      </header>
      <section className="mx-auto grid max-w-6xl grid-cols-1 gap-12 px-6 pb-24 md:grid-cols-2">
        <div className="flex items-center justify-center">
          <img
            src={product.image}
            alt={product.name}
            style={{ viewTransitionName: `product-image-${product.id}` }}
            className="h-[420px] w-[420px] max-w-full object-contain"
          />
        </div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.25, ease: "easeOut" }}
          className="flex flex-col justify-center"
        >
          <p className="text-[12px] font-light uppercase tracking-widest text-neutral-500">
            {product.category}
          </p>
          <h1 className="mt-2 text-4xl font-semibold tracking-tight text-neutral-900">
            {product.name}
          </h1>
          <p className="mt-3 text-[15px] font-light leading-relaxed text-neutral-600">
            {product.tagline}
          </p>
          <p className="mt-6 text-2xl font-semibold text-neutral-900">{product.price}</p>
          <div className="mt-8 flex gap-3">
            <button className="h-12 flex-1 rounded-full bg-neutral-900 text-[14px] font-medium text-white transition-colors hover:bg-neutral-800">
              Mua ngay
            </button>
            <button className="h-12 flex-1 rounded-full border border-gray-200 text-[14px] font-medium text-neutral-900 transition-colors hover:border-neutral-400">
              Thêm vào giỏ
            </button>
          </div>
        </motion.div>
      </section>
    </main>
  );
}
import { createFileRoute } from "@tanstack/react-router";
import { Header } from "@/components/storefront/Header";
import { Hero } from "@/components/storefront/Hero";
import { ProductGrid } from "@/components/storefront/ProductGrid";
import { Footer } from "@/components/storefront/Footer";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Tinh. — Tinh tế trong từng chi tiết" },
      {
        name: "description",
        content:
          "Cửa hàng điện tử cao cấp: Mac, iPhone, iPad, Watch, AirPods. Thiết kế tinh tế, dịch vụ tận tâm.",
      },
      { property: "og:title", content: "Tinh. — Tinh tế trong từng chi tiết" },
      {
        property: "og:description",
        content: "Khám phá bộ sưu tập thiết bị cao cấp được chế tác cho mỗi khoảnh khắc.",
      },
    ],
  }),
  component: Index,
});

function Index() {
  return (
    <div className="min-h-screen bg-white text-neutral-900 antialiased">
      <Header />
      <main>
        <Hero />
        <ProductGrid />
      </main>
      <Footer />
    </div>
  );
}

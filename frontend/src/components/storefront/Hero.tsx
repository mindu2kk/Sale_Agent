import { motion } from 'framer-motion'
import { ArrowDown, ArrowUpRight, Sparkles } from 'lucide-react'
import { useCatalogStore } from '@/stores/catalogStore'
import { useCopilotStore } from '@/stores/copilotStore'

export function Hero() {
  const products = useCatalogStore((state) => state.products)
  const product =
    products
      .filter((item) => item.category === 'Laptop')
      .toSorted(
        (left, right) =>
          Math.abs(left.price_value - 20_000_000) -
          Math.abs(right.price_value - 20_000_000),
      )[0] ?? products[0]
  const openWithPrefill = useCopilotStore((state) => state.openWithPrefill)

  return (
    <section className="hero-section">
      <div className="hero-orb hero-orb-one" />
      <div className="hero-orb hero-orb-two" />
      <div className="mx-auto grid min-h-[calc(100svh-68px)] max-w-[1320px] items-center gap-10 px-5 py-14 lg:grid-cols-[0.92fr_1.08fr] lg:px-8 lg:py-16">
        <motion.div
          className="hero-content"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        >
          <h1 className="hero-title">Chọn nhanh.<br />Hỏi gọn.<br /><em>Mua đúng<span>.</span></em></h1>
          <p className="hero-copy">
            Trợ lý AI hiểu nhu cầu của bạn, so sánh thông minh và gợi ý sản phẩm đáng tiền nhất.
          </p>
          <div className="hero-actions">
            <a href="#collection" className="primary-button">
              Xem bộ sưu tập
              <ArrowDown size={15} />
            </a>
            <button
              className="secondary-button"
              onClick={() => openWithPrefill('Hãy giúp tôi chọn một laptop trong catalog nội bộ.')}
            >
              Mở trợ lý mua sắm
              <ArrowUpRight size={15} />
            </button>
          </div>
          <div className="hero-benefits">
            <span><i>✓</i><strong>Hàng chính hãng</strong><small>Bảo hành toàn quốc</small></span>
            <span><i>↗</i><strong>Giao nhanh 24H</strong><small>Nội thành TP.HCM</small></span>
            <span><i>↻</i><strong>Đổi trả 15 ngày</strong><small>Dễ dàng, nhanh chóng</small></span>
          </div>
        </motion.div>
        <motion.div
          className="hero-stage"
          initial={{ opacity: 0, scale: 0.96, x: 30 }}
          animate={{ opacity: 1, scale: 1, x: 0 }}
          transition={{ duration: 0.85, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="stage-grid" />
          <div className="stage-halo" />
          <div className="stage-wave">
            <span />
            <span />
            <span />
          </div>
          <div className="hero-image-wrap">
            {product ? (
              <img src={product.image_url} alt={product.name} className="hero-image" />
            ) : (
              <div className="catalog-skeleton h-full w-full" />
            )}
          </div>
          <div className="stage-pedestal">
            <span>{product?.brand} {product?.category}</span>
          </div>
          {product && (
            <>
              <div className="stage-product-meta">
                <span>{product.brand} · {product.category}</span>
                <strong>{product.price}</strong>
              </div>
              <button
                className="stage-ask"
                onClick={() => openWithPrefill(`Hãy tư vấn chi tiết sản phẩm mã ${product.code}.`)}
              >
                <Sparkles size={14} />
                Hỏi về SKU {product.code}
              </button>
            </>
          )}
        </motion.div>
      </div>
    </section>
  )
}

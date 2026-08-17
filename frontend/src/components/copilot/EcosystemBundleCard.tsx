import { motion } from 'framer-motion'
import { useCommerceStore } from '@/stores/commerceStore'
import type { Bundle } from '@/types/commerce'

export function EcosystemBundleCard({ bundle }: { bundle: Bundle }) {
  const addToCart = useCommerceStore((s) => s.addToCart)

  return (
    <motion.div
      className="bundle-card"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.24 }}
      whileHover="hover"
    >
      {/* Stacked thumbnails */}
      <div className="bundle-images">
        {bundle.items.slice(0, 3).map((product, index) => (
          <motion.div
            key={product.id}
            className="bundle-image"
            style={{ zIndex: 3 - index, left: `${index * 14}px` }}
            variants={{ hover: { x: (index - 1) * 8 } }}
          >
            {product.image_url
              ? <img src={product.image_url} alt={product.name} />
              : null}
          </motion.div>
        ))}
      </div>

      {/* Info */}
      <div className="min-w-0 flex-1">
        <p style={{ fontSize: '13px', fontWeight: 600, color: '#0F172A', lineHeight: 1.3 }}>{bundle.title}</p>
        <p style={{ fontSize: '11px', color: '#64748B', marginTop: '2px' }}>
          {bundle.items.length} sản phẩm trong bộ
        </p>
      </div>

      <button
        className="bundle-button"
        onClick={() => bundle.items.forEach((p) => addToCart(p))}
        aria-label={`Thêm bộ: ${bundle.title}`}
      >
        Thêm bộ
      </button>
    </motion.div>
  )
}

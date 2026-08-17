import { useState } from 'react'
import { ArrowUpRight, Plus, ShoppingBag } from 'lucide-react'
import { useCommerceStore } from '@/stores/commerceStore'
import type { Product } from '@/types/commerce'

function summarizeSpecs(product: Product) {
  const specs = product.display_specs?.length ? product.display_specs : product.specs
  return specs
    .slice(0, 2)
    .map((spec) => spec.replace(/^[^:：]+[:：]\s*/, '').trim())
    .filter(Boolean)
    .join(' · ')
}

export function InlineProductCard({ product }: { product: Product }) {
  const [imageFailed, setImageFailed] = useState(false)
  const addToCart = useCommerceStore((s) => s.addToCart)
  const openProduct = useCommerceStore((s) => s.openProduct)

  return (
    <div className="inline-product">
      <button
        className="inline-product-image"
        onClick={() => openProduct(product)}
        aria-label={`Xem ${product.name}`}
      >
        {product.image_url && !imageFailed ? (
          <img src={product.image_url} alt={product.name} onError={() => setImageFailed(true)} />
        ) : (
          <ShoppingBag size={18} strokeWidth={1.2} style={{ color: '#CBD5E1' }} />
        )}
      </button>

      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <span className="inline-brand">{product.brand}</span>
          <span className="inline-code">{product.code}</span>
        </div>
        <p className="inline-product-name">{product.name}</p>
        {(product.display_specs?.length || product.specs.length) > 0 && <p className="inline-product-spec">{summarizeSpecs(product)}</p>}
        <p className="inline-product-price">{product.price}</p>
      </div>

      <div className="inline-product-actions">
        <button
          onClick={() => openProduct(product)}
          className="inline-open-button"
          aria-label={`Mở ${product.name}`}
        >
          <ArrowUpRight size={12} />
        </button>
        <button onClick={() => addToCart(product)} className="inline-add-button" aria-label={`Thêm ${product.name} vào giỏ`}>
          <Plus size={12} />
          Thêm
        </button>
      </div>
    </div>
  )
}

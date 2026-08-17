import { useState } from 'react'
import { Check, GitCompareArrows, Heart, Monitor, ShieldCheck, Smartphone, Sparkles } from 'lucide-react'
import { useCommerceStore } from '@/stores/commerceStore'
import type { Product } from '@/types/commerce'

type SpecHighlight = {
  label: string
  value: string
}

function normalizeSpecLabel(label: string) {
  const cleaned = label.toLowerCase()
  if (/cpu|chip|vi xu ly/.test(cleaned)) return 'CPU'
  if (/gpu|card do hoa|do hoa/.test(cleaned)) return 'GPU'
  if (/ram|bo nho/.test(cleaned)) return 'RAM'
  if (/ssd|o cung|luu tru|storage/.test(cleaned)) return 'SSD'
  if (/man hinh|display|screen/.test(cleaned)) return 'Màn hình'
  if (/camera/.test(cleaned)) return 'Camera'
  if (/pin|battery/.test(cleaned)) return 'Pin'
  return label.trim()
}

function splitSpec(spec: string): SpecHighlight | null {
  const separator = spec.includes(':') ? ':' : spec.includes('：') ? '：' : null
  if (!separator) {
    const trimmed = spec.trim()
    return trimmed ? { label: 'Thông số', value: trimmed } : null
  }
  const [rawLabel, ...rest] = spec.split(separator)
  const value = rest.join(separator).trim()
  if (!value) return null
  return {
    label: normalizeSpecLabel(rawLabel.trim()),
    value,
  }
}

function getSpecHighlights(product: Product): SpecHighlight[] {
  const sourceSpecs = product.display_specs?.length ? product.display_specs : product.specs
  const specs = sourceSpecs.map(splitSpec).filter((item): item is SpecHighlight => Boolean(item))
  if (product.display_specs?.length) {
    return specs.slice(0, 4)
  }
  const priority =
    product.category === 'Laptop'
      ? ['CPU', 'GPU', 'RAM', 'SSD', 'Màn hình']
      : ['RAM', 'SSD', 'Camera', 'Màn hình', 'Pin']

  const ordered = priority
    .map((label) => specs.find((item) => item.label === label))
    .filter((item): item is SpecHighlight => Boolean(item))

  const fallback = specs.filter((item) => !ordered.some((chosen) => chosen.label === item.label))
  return [...ordered, ...fallback].slice(0, 4)
}

function getRetailTag(product: Product) {
  const text = product.specs.join(' ').toLowerCase()
  if (/rtx|gtx|radeon rx/.test(text)) return 'Hợp gaming'
  if (/16\s*gb|32\s*gb|lpddr5/.test(text)) return 'Đa nhiệm tốt'
  if (/oled|2\.8k|120hz|amoled/.test(text)) return 'Màn hình nổi bật'
  if (product.price_value <= 20000000) return 'Tầm giá dễ chọn'
  return 'Đáng chú ý'
}

function NoImage({ category }: { category: Product['category'] }) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-2" style={{ color: '#CBD5E1' }}>
      {category === 'Laptop' ? <Monitor size={42} strokeWidth={1} /> : <Smartphone size={42} strokeWidth={1} />}
      <span style={{ fontSize: '11px', color: '#94A3B8' }}>Chưa có ảnh</span>
    </div>
  )
}

export function ProductCard({ product }: { product: Product }) {
  const [imageFailed, setImageFailed] = useState(false)
  const favoriteCodes = useCommerceStore((s) => s.favoriteCodes)
  const compareProducts = useCommerceStore((s) => s.compareProducts)
  const toggleFavorite = useCommerceStore((s) => s.toggleFavorite)
  const toggleCompare = useCommerceStore((s) => s.toggleCompare)
  const openProduct = useCommerceStore((s) => s.openProduct)

  const isFav = favoriteCodes.includes(product.code)
  const isCompared = compareProducts.some((p) => p.code === product.code)
  const highlights = getSpecHighlights(product)
  const retailTag = getRetailTag(product)

  return (
    <article className="product-card" aria-label={product.name}>
      <div
        className="product-card-img-zone"
        role="button"
        onClick={() => openProduct(product)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            openProduct(product)
          }
        }}
        aria-label={`Xem nhanh ${product.name}`}
        tabIndex={0}
      >
        <div className="product-card-hero-glow" aria-hidden="true" />
        <div className="product-card-topline">
          <span className="category-badge category-badge-strong">
            {product.category === 'Laptop' ? 'Laptop' : 'Điện thoại'}
          </span>
          <span className="product-card-trust-badge">
            <ShieldCheck size={12} />
            Catalog thật
          </span>
        </div>

        {product.image_url && !imageFailed ? (
          <img
            className="product-card-img"
            src={product.image_url}
            alt={product.name}
            loading="lazy"
            onError={() => setImageFailed(true)}
          />
        ) : (
          <NoImage category={product.category} />
        )}

        <div className="product-card-img-actions">
          <button
            className="btn-icon h-9 w-9"
            style={{
              backgroundColor: 'rgba(255,255,255,0.94)',
              backdropFilter: 'blur(8px)',
              color: isFav ? '#D70018' : '#64748B',
              boxShadow: '0 10px 22px rgba(15,23,42,0.12)',
            }}
            onClick={(e) => {
              e.stopPropagation()
              toggleFavorite(product.code)
            }}
            aria-label={isFav ? 'Bỏ yêu thích' : 'Thêm vào yêu thích'}
          >
            <Heart size={16} fill={isFav ? 'currentColor' : 'none'} />
          </button>
        </div>

        <div className="product-card-spotlight">
          <Sparkles size={12} />
          {retailTag}
        </div>
      </div>

      <div className="product-card-body">
        <div className="mb-3 flex items-center justify-between gap-3">
          <p
            className="truncate"
            style={{
              fontSize: '11px',
              fontWeight: 700,
              color: '#64748B',
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
            }}
          >
            {product.brand}
          </p>
          <span className="product-card-code">{product.code}</span>
        </div>

        <button
          className="mb-3 w-full text-left"
          style={{
            fontSize: '16px',
            fontWeight: 700,
            lineHeight: 1.35,
            color: '#0F172A',
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
            minHeight: '43px',
          }}
          onClick={() => openProduct(product)}
        >
          {product.name}
        </button>

        {highlights.length > 0 && (
          <div className="product-spec-stack">
            {highlights.slice(0, 3).map((item, index) => (
              <div key={`${product.code}-${item.label}-${item.value}-${index}`} className="product-spec-row">
                <span className="product-spec-label">{item.label}</span>
                <span className="product-spec-value">{item.value}</span>
              </div>
            ))}
            {highlights.length > 3 && (
              <p className="product-spec-more">+{highlights.length - 3} thông số khác</p>
            )}
          </div>
        )}

        <div className="product-price-block">
          <p className="product-price-kicker">Giá đang hiển thị</p>
          <span className="price-main">{product.price}</span>
          <p className="product-price-note">Phù hợp để xem nhanh và so sánh ngay trong catalog.</p>
        </div>

        <div className="mt-auto flex items-center gap-2">
          <button
            className="btn-secondary product-cta-secondary"
            style={
              isCompared
                ? {
                    borderColor: '#FBCDD2',
                    color: '#D70018',
                    backgroundColor: '#FFF1F2',
                  }
                : undefined
            }
            onClick={(e) => {
              e.stopPropagation()
              toggleCompare(product)
            }}
            aria-label={isCompared ? 'Bỏ khỏi so sánh' : 'So sánh sản phẩm'}
          >
            {isCompared ? <Check size={13} /> : <GitCompareArrows size={13} />}
            <span>{isCompared ? 'Đã chọn' : 'So sánh'}</span>
          </button>
          <button
            className="btn-primary product-cta-primary"
            onClick={() => openProduct(product)}
            aria-label={`Xem nhanh ${product.name}`}
          >
            Xem chi tiết
          </button>
        </div>
      </div>
    </article>
  )
}

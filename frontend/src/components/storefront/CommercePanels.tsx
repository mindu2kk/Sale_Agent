import {
  Check,
  CheckCircle2,
  ExternalLink,
  GitCompareArrows,
  Info,
  PackageCheck,
  ShieldCheck,
  ShoppingBag,
  Sparkles,
  Trash2,
  Truck,
  X,
} from 'lucide-react'
import { useCommerceStore } from '@/stores/commerceStore'
import { useCopilotStore } from '@/stores/copilotStore'
import type { Product } from '@/types/commerce'

/* ─── Deterministic AI Insights ─────────────────────────────────────────────
 * Frontend-only, no fabricated claims.
 * Based strictly on spec strings from the real product data.
 */
function deriveInsights(specs: string[]): string[] {
  const text = specs.join(' ')
  const insights: string[] = []

  if (/rtx|gtx|radeon\s*rx/i.test(text)) {
    insights.push('Có GPU rời, phù hợp hơn cho game, đồ họa hoặc tác vụ nặng so với máy chỉ dùng GPU tích hợp.')
  } else if (/intel\s*(uhd|iris|arc|graphics|u[0-9])|celeron/i.test(text)) {
    insights.push('Phù hợp hơn với học tập, văn phòng và tác vụ cơ bản.')
  }

  if (/\b(32|64)\s*gb\s*(ram|lpddr|ddr)/i.test(text)) {
    insights.push('RAM lớn hơn giúp thoải mái hơn khi mở nhiều tab, ứng dụng văn phòng, IDE hoặc đa nhiệm.')
  } else if (/\b16\s*gb\s*(ram|lpddr|ddr)/i.test(text)) {
    insights.push('RAM 16 GB — thoải mái cho đa nhiệm văn phòng và lập trình nhẹ.')
  }

  if (/ssd|nvme/i.test(text)) {
    insights.push('Có SSD, thường giúp mở máy và truy cập ứng dụng nhanh hơn so với ổ cứng truyền thống.')
  }

  return insights
}

/* ─── Spec label/value splitter ─────────────────────────────────────────── */
function splitSpec(spec: string): { label: string; value: string } {
  const SEP = spec.includes(':') ? ':' : spec.includes('：') ? '：' : null
  if (!SEP) return { label: 'Thông tin', value: spec.trim() }
  const idx = spec.indexOf(SEP)
  return {
    label: spec.slice(0, idx).trim(),
    value: spec.slice(idx + 1).trim() || 'Đang cập nhật',
  }
}

/* ═══════════════════════════════════════════════════════════════
   ProductQuickView
═══════════════════════════════════════════════════════════════ */
function ProductQuickView({ product }: { product: Product }) {
  const closeProduct = useCommerceStore((s) => s.closeProduct)
  const addToCart = useCommerceStore((s) => s.addToCart)
  const toggleCompare = useCommerceStore((s) => s.toggleCompare)
  const compareProducts = useCommerceStore((s) => s.compareProducts)
  const openWithPrefill = useCopilotStore((s) => s.openWithPrefill)

  const isCompared = compareProducts.some((p) => p.code === product.code)
  const specRows = product.specs.slice(0, 8).map(splitSpec)
  const insights = deriveInsights(product.specs)

  return (
    <div
      className="modal-overlay"
      role="presentation"
      onMouseDown={closeProduct}
    >
      <article
        className="flex max-h-[90vh] w-full max-w-[900px] overflow-hidden"
        style={{
          backgroundColor: '#FFFFFF',
          borderRadius: '24px',
          boxShadow: '0 24px 64px rgba(15,23,42,0.18)',
        }}
        role="dialog"
        aria-modal="true"
        aria-label={`Chi tiết ${product.name}`}
        onMouseDown={(e) => e.stopPropagation()}
      >
        {/* ── Left: Image pane ── */}
        <section
          className="relative hidden w-[42%] shrink-0 flex-col items-center justify-center md:flex"
          style={{
            background: 'linear-gradient(160deg, #F8FAFC 0%, #EEF2F8 100%)',
            padding: '40px 32px',
          }}
        >
          {product.image_url ? (
            <img
              src={product.image_url}
              alt={product.name}
              style={{ width: '100%', maxWidth: '320px', objectFit: 'contain', maxHeight: '320px' }}
            />
          ) : (
            <div
              className="flex flex-col items-center justify-center gap-3"
              style={{ color: '#CBD5E1' }}
            >
              <ShoppingBag size={56} strokeWidth={0.9} />
              <span style={{ fontSize: '13px', color: '#94A3B8' }}>Chưa có ảnh sản phẩm</span>
            </div>
          )}

          {product.source_url && (
            <a
              href={product.source_url}
              target="_blank"
              rel="noreferrer"
              className="mt-6 flex items-center gap-1"
              style={{ fontSize: '12px', color: '#94A3B8', textDecoration: 'none' }}
            >
              <ExternalLink size={12} /> Xem nguồn
            </a>
          )}

          {/* Trust badges */}
          <div
            className="absolute bottom-5 left-0 right-0 flex items-center justify-center gap-6"
            style={{ fontSize: '11px', color: '#94A3B8' }}
          >
            <span className="flex items-center gap-1"><PackageCheck size={13} style={{ color: '#D70018' }} />Chính hãng</span>
            <span className="flex items-center gap-1"><ShieldCheck size={13} style={{ color: '#D70018' }} />Bảo hành</span>
            <span className="flex items-center gap-1"><Truck size={13} style={{ color: '#D70018' }} />Giao hàng</span>
          </div>
        </section>

        {/* ── Right: Details pane ── */}
        <section className="flex flex-1 flex-col overflow-y-auto scrollbar-thin" style={{ padding: '32px 28px' }}>
          {/* Header */}
          <div className="mb-5 flex items-start justify-between gap-4">
            <div className="min-w-0">
              <p style={{ fontSize: '12px', fontWeight: 500, color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '4px' }}>
                {product.brand} · SKU {product.code}
              </p>
              <h2 style={{ fontSize: '22px', fontWeight: 700, lineHeight: 1.25, color: '#0F172A' }}>
                {product.name}
              </h2>
            </div>
            <button
              className="btn-icon shrink-0"
              onClick={closeProduct}
              aria-label="Đóng"
            >
              <X size={20} />
            </button>
          </div>

          {/* Price */}
          <div className="mb-5">
            <span style={{ fontSize: '32px', fontWeight: 800, color: '#D70018', letterSpacing: '0' }}>
              {product.price}
            </span>
          </div>

          {/* Primary CTA */}
          <button
            className="btn-primary mb-3 w-full"
            style={{ padding: '13px 0', fontSize: '15px' }}
            onClick={() => addToCart(product)}
          >
            <ShoppingBag size={18} style={{ marginRight: '8px' }} />
            Thêm vào giỏ hàng
          </button>

          {/* Secondary CTAs */}
          <div className="mb-6 grid grid-cols-2 gap-2.5">
            <button
              className="btn-secondary"
              style={{ padding: '10px 0', fontSize: '13px' }}
              onClick={() => {
                closeProduct()
                openWithPrefill(`Hãy tư vấn chi tiết sản phẩm mã ${product.code} - ${product.name}`)
              }}
            >
              <Sparkles size={13} style={{ marginRight: '6px', color: '#D70018' }} />
              <span style={{ color: '#D70018' }}>Hỏi AI về máy này</span>
            </button>
            <button
              className="btn-secondary"
              style={isCompared ? { padding: '10px 0', fontSize: '13px', borderColor: '#FBCDD2', color: '#D70018', backgroundColor: '#FFF1F2' } : { padding: '10px 0', fontSize: '13px' }}
              onClick={() => toggleCompare(product)}
            >
              {isCompared
                ? <><Check size={13} style={{ marginRight: '5px' }} />Đã chọn so sánh</>
                : <><GitCompareArrows size={13} style={{ marginRight: '5px' }} />So sánh</>}
            </button>
          </div>

          {/* Specs table */}
          {specRows.length > 0 && (
            <div
              className="mb-5 overflow-hidden rounded-xl border"
              style={{ borderColor: '#EEF2F8' }}
            >
              <div
                className="border-b px-4 py-2.5"
                style={{ backgroundColor: '#F8FAFC', borderColor: '#EEF2F8' }}
              >
                <h3 style={{ fontSize: '13px', fontWeight: 600, color: '#0F172A' }}>Thông số nổi bật</h3>
              </div>
              <div>
                {specRows.map(({ label, value }, i) => (
                  <div
                    key={`${label}-${i}`}
                    className="flex justify-between"
                    style={{
                      padding: '9px 16px',
                      borderBottom: i < specRows.length - 1 ? '1px solid #F1F5F9' : 'none',
                      fontSize: '13px',
                    }}
                  >
                    <span style={{ color: '#64748B', flexShrink: 0, paddingRight: '16px' }}>{label}</span>
                    <span style={{ color: '#0F172A', fontWeight: 500, textAlign: 'right' }}>{value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* AI Insight — deterministic */}
          <div className="ai-insight-box">
            <Info size={15} style={{ color: '#D70018', marginTop: '1px', flexShrink: 0 }} />
            <div>
              <p style={{ fontSize: '12px', fontWeight: 600, color: '#D70018', marginBottom: '4px' }}>
                Gợi ý dựa trên cấu hình thực tế
              </p>
              {insights.length > 0 ? (
                <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                  {insights.map((ins) => (
                    <li key={ins} style={{ fontSize: '13px', color: '#334155', marginBottom: '2px' }}>
                      · {ins}
                    </li>
                  ))}
                </ul>
              ) : (
                <p style={{ fontSize: '13px', color: '#64748B' }}>
                  Chưa đủ dữ liệu để tạo nhận xét nâng cao cho mẫu này.
                </p>
              )}
            </div>
          </div>
        </section>
      </article>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   CartPanel
═══════════════════════════════════════════════════════════════ */
function CartPanel() {
  const cart = useCommerceStore((s) => s.cart)
  const closeCart = useCommerceStore((s) => s.closeCart)
  const removeFromCart = useCommerceStore((s) => s.removeFromCart)
  const openProduct = useCommerceStore((s) => s.openProduct)

  const total = cart.reduce((sum, item) => sum + item.price_value, 0)

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end"
      role="presentation"
      style={{ backgroundColor: 'rgba(15,23,42,0.35)', backdropFilter: 'blur(6px)' }}
      onMouseDown={closeCart}
    >
      <aside
        className="flex h-full w-full max-w-[400px] flex-col"
        style={{
          backgroundColor: '#FFFFFF',
          borderLeft: '1px solid #EEF2F8',
          boxShadow: '-4px 0 32px rgba(15,23,42,0.08)',
        }}
        onMouseDown={(e) => e.stopPropagation()}
        aria-label="Giỏ hàng"
      >
        {/* Header */}
        <header
          className="flex items-center justify-between"
          style={{ padding: '20px 24px', borderBottom: '1px solid #EEF2F8' }}
        >
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#0F172A' }}>Giỏ hàng</h2>
            <p style={{ fontSize: '12px', color: '#94A3B8', marginTop: '2px' }}>{cart.length} sản phẩm</p>
          </div>
          <button className="btn-icon" onClick={closeCart} aria-label="Đóng giỏ hàng">
            <X size={20} />
          </button>
        </header>

        {/* Items */}
        <div className="flex-1 overflow-y-auto scrollbar-thin" style={{ padding: '16px 20px' }}>
          {cart.length > 0 ? (
            cart.map((product) => (
              <article
                key={product.code}
                className="flex items-center gap-3 rounded-xl border"
                style={{ padding: '12px', marginBottom: '10px', borderColor: '#EEF2F8', backgroundColor: '#FAFBFF' }}
              >
                <button
                  className="flex shrink-0 items-center justify-center overflow-hidden rounded-xl"
                  style={{ width: '64px', height: '64px', backgroundColor: '#F1F5F9', padding: '4px' }}
                  onClick={() => openProduct(product)}
                  aria-label={`Xem ${product.name}`}
                >
                  {product.image_url ? (
                    <img className="h-full w-full object-contain" src={product.image_url} alt={product.name} />
                  ) : (
                    <ShoppingBag size={22} strokeWidth={1.2} style={{ color: '#CBD5E1' }} />
                  )}
                </button>
                <div className="min-w-0 flex-1">
                  <p style={{ fontSize: '11px', color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{product.brand}</p>
                  <h4
                    style={{
                      fontSize: '13px', fontWeight: 600, color: '#0F172A',
                      display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
                      lineHeight: 1.3,
                    }}
                  >
                    {product.name}
                  </h4>
                  <p style={{ fontSize: '15px', fontWeight: 700, color: '#D70018', marginTop: '4px' }}>{product.price}</p>
                </div>
                <button
                  className="btn-icon h-8 w-8 shrink-0"
                  style={{ color: '#D70018' }}
                  onClick={() => removeFromCart(product.code)}
                  aria-label={`Xóa ${product.name}`}
                >
                  <Trash2 size={15} />
                </button>
              </article>
            ))
          ) : (
            <div className="flex h-full flex-col items-center justify-center py-24 text-center">
              <div
                className="mb-4 flex items-center justify-center rounded-2xl"
                style={{ width: '64px', height: '64px', backgroundColor: '#F1F5F9' }}
              >
                <ShoppingBag size={28} strokeWidth={1.2} style={{ color: '#CBD5E1' }} />
              </div>
              <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#0F172A', marginBottom: '4px' }}>Giỏ hàng trống</h3>
              <p style={{ fontSize: '13px', color: '#64748B' }}>Thêm sản phẩm để bắt đầu mua sắm.</p>
            </div>
          )}
        </div>

        {/* Footer */}
        {cart.length > 0 && (
          <footer style={{ borderTop: '1px solid #EEF2F8', padding: '20px 24px', backgroundColor: '#F8FAFC' }}>
            <div className="mb-4 flex items-center justify-between">
              <span style={{ fontSize: '14px', color: '#64748B' }}>Tạm tính</span>
              <strong style={{ fontSize: '20px', fontWeight: 700, color: '#D70018' }}>
                {total.toLocaleString('vi-VN')}₫
              </strong>
            </div>
            <button className="btn-primary w-full" style={{ padding: '13px 0', fontSize: '14px' }}>
              <CheckCircle2 size={16} style={{ marginRight: '8px' }} />
              Xác nhận đơn hàng
            </button>
          </footer>
        )}
      </aside>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   CompareTray — floating bottom bar
═══════════════════════════════════════════════════════════════ */
function CompareTray() {
  const products = useCommerceStore((s) => s.compareProducts)
  const closeCompare = useCommerceStore((s) => s.closeCompare)
  const toggleCompare = useCommerceStore((s) => s.toggleCompare)
  const openWithPrefill = useCopilotStore((s) => s.openWithPrefill)

  const canCompare = products.length >= 2

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center"
      role="presentation"
      style={{ backgroundColor: 'rgba(15,23,42,0.25)', backdropFilter: 'blur(4px)' }}
      onMouseDown={closeCompare}
    >
      <aside
        className="compare-tray mx-4 mb-4 w-full"
        onMouseDown={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between border-b"
          style={{ padding: '14px 20px', borderColor: '#EEF2F8', backgroundColor: '#F8FAFC' }}
        >
          <div>
            <h2 style={{ fontSize: '15px', fontWeight: 700, color: '#0F172A' }}>So sánh sản phẩm</h2>
            <p style={{ fontSize: '12px', color: '#64748B', marginTop: '1px' }}>
              {products.length}/3 đã chọn
              {!canCompare && ' — Cần ít nhất 2 sản phẩm'}
            </p>
          </div>
          <button className="btn-icon" onClick={closeCompare} aria-label="Đóng">
            <X size={18} />
          </button>
        </div>

        <div style={{ padding: '16px 20px' }}>
          {products.length > 0 ? (
            <>
              {/* Product columns */}
              <div
                className="grid gap-3 mb-4"
                style={{ gridTemplateColumns: `repeat(${Math.max(products.length, 1)}, 1fr)` }}
              >
                {products.map((product) => (
                  <div
                    key={product.code}
                    className="relative flex flex-col items-center rounded-xl border"
                    style={{ padding: '12px 10px', borderColor: '#EEF2F8', backgroundColor: '#FAFBFF' }}
                  >
                    {/* Remove button */}
                    <button
                      className="btn-icon absolute right-1.5 top-1.5 h-6 w-6"
                      style={{ backgroundColor: 'transparent' }}
                      onClick={() => toggleCompare(product)}
                      aria-label={`Bỏ ${product.name}`}
                    >
                      <X size={12} />
                    </button>

                    {/* Thumb */}
                    <div
                      className="flex items-center justify-center overflow-hidden rounded-lg"
                      style={{ width: '72px', height: '72px', backgroundColor: '#F1F5F9', marginBottom: '8px' }}
                    >
                      {product.image_url ? (
                        <img className="h-full w-full object-contain" src={product.image_url} alt={product.name} />
                      ) : (
                        <ShoppingBag size={24} strokeWidth={1.2} style={{ color: '#CBD5E1' }} />
                      )}
                    </div>

                    <p style={{ fontSize: '10px', color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{product.brand}</p>
                    <p
                      style={{
                        fontSize: '12px', fontWeight: 600, color: '#0F172A', textAlign: 'center', marginTop: '2px',
                        display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
                        lineHeight: 1.3,
                      }}
                    >
                      {product.name}
                    </p>
                    <p style={{ fontSize: '14px', fontWeight: 700, color: '#D70018', marginTop: '6px' }}>{product.price}</p>
                  </div>
                ))}

                {/* Empty slots */}
                {products.length < 3 && (
                  <div
                    className="flex min-h-[140px] flex-col items-center justify-center rounded-xl border border-dashed"
                    style={{ borderColor: '#CBD5E1', color: '#94A3B8' }}
                  >
                    <GitCompareArrows size={24} style={{ opacity: 0.4, marginBottom: '6px' }} />
                    <p style={{ fontSize: '12px', textAlign: 'center', lineHeight: 1.4 }}>
                      Thêm sản phẩm<br />để so sánh
                    </p>
                  </div>
                )}
              </div>

              {/* AI Compare CTA */}
              <button
                className="btn-primary w-full"
                style={{ padding: '12px 0', fontSize: '14px' }}
                disabled={!canCompare}
                onClick={() => {
                  closeCompare()
                  openWithPrefill(
                    `So sánh các sản phẩm: ${products.map((p) => `${p.name} (${p.code})`).join(', ')}`
                  )
                }}
              >
                <Sparkles size={15} style={{ marginRight: '8px' }} />
                {canCompare
                  ? `Nhờ AI phân tích ${products.length} sản phẩm`
                  : 'Cần ít nhất 2 sản phẩm để so sánh'}
              </button>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <GitCompareArrows size={36} style={{ color: '#CBD5E1', marginBottom: '10px' }} />
              <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#0F172A', marginBottom: '4px' }}>Chưa chọn sản phẩm nào</h3>
              <p style={{ fontSize: '13px', color: '#64748B' }}>Bấm "So sánh" trên tối đa 3 sản phẩm bạn quan tâm.</p>
            </div>
          )}
        </div>
      </aside>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   CommercePanels — main export
═══════════════════════════════════════════════════════════════ */
export function CommercePanels() {
  // All state via Zustand selectors — reactive, not getState()
  const selectedProduct = useCommerceStore((s) => s.selectedProduct)
  const detailOpen = useCommerceStore((s) => s.detailOpen)
  const cartOpen = useCommerceStore((s) => s.cartOpen)
  const compareOpen = useCommerceStore((s) => s.compareOpen)

  if (detailOpen && selectedProduct) return <ProductQuickView product={selectedProduct} />
  if (cartOpen) return <CartPanel />
  if (compareOpen) return <CompareTray />
  return null
}

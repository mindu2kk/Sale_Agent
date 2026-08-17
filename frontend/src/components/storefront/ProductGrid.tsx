import { useMemo, useState } from 'react'
import { ChevronDown, LoaderCircle, SlidersHorizontal } from 'lucide-react'
import { useCatalogStore } from '@/stores/catalogStore'
import type { Product } from '@/types/commerce'
import { SmartSuggestionBar } from './SmartSuggestionBar'
import { ProductCard } from './ProductCard'
import { ProductSkeleton } from '../ui/ProductSkeleton'
import { EmptyState } from '../ui/EmptyState'

const CATEGORIES = [
  { label: 'Tất cả', value: '' },
  { label: 'Laptop', value: 'Laptop' },
  { label: 'Điện thoại', value: 'Mobile Phone' },
] as const

const SORT_OPTIONS = [
  { label: 'Phù hợp nhất', value: 'relevance' },
  { label: 'Giá thấp → cao', value: 'price-asc' },
  { label: 'Giá cao → thấp', value: 'price-desc' },
] as const

const PRICE_FILTERS = [
  { label: 'Tất cả giá', value: 'all' },
  { label: 'Dưới 20 triệu', value: 'under-20' },
  { label: '20 - 30 triệu', value: '20-30' },
  { label: 'Trên 30 triệu', value: '30-plus' },
] as const

const USE_CASE_FILTERS = [
  { label: 'Mọi nhu cầu', value: 'all' },
  { label: 'Văn phòng', value: 'office' },
  { label: 'Gaming / đồ họa', value: 'gaming' },
  { label: 'Mỏng nhẹ', value: 'portable' },
] as const

function matchesPrice(product: Product, priceFilter: string) {
  if (priceFilter === 'under-20') return product.price_value < 20000000
  if (priceFilter === '20-30') return product.price_value >= 20000000 && product.price_value <= 30000000
  if (priceFilter === '30-plus') return product.price_value > 30000000
  return true
}

function matchesUseCase(product: Product, useCase: string) {
  if (useCase === 'all') return true
  const text = `${product.name} ${product.specs.join(' ')}`.toLowerCase()
  if (useCase === 'gaming') return /rtx|gtx|radeon rx|gaming/.test(text)
  if (useCase === 'portable') return /lite|air|slim|thin|ultra|14\b|13\b|mỏng|nhẹ/.test(text)
  if (useCase === 'office') return !/rtx|gtx|radeon rx|gaming/.test(text)
  return true
}

export function ProductGrid() {
  const {
    products,
    loading,
    loadingMore,
    hasMore,
    error,
    activeCategory,
    query,
    sort,
    filterByCategory,
    setSort,
    loadMore,
  } = useCatalogStore()

  const [priceFilter, setPriceFilter] = useState<(typeof PRICE_FILTERS)[number]['value']>('all')
  const [useCaseFilter, setUseCaseFilter] = useState<(typeof USE_CASE_FILTERS)[number]['value']>('all')
  const [brandFilter, setBrandFilter] = useState('all')

  const availableBrands = useMemo(
    () => ['all', ...Array.from(new Set(products.map((product) => product.brand))).slice(0, 6)],
    [products],
  )

  const filteredProducts = useMemo(
    () =>
      products.filter((product) => {
        if (!matchesPrice(product, priceFilter)) return false
        if (!matchesUseCase(product, useCaseFilter)) return false
        if (brandFilter !== 'all' && product.brand !== brandFilter) return false
        return true
      }),
    [brandFilter, priceFilter, products, useCaseFilter],
  )

  const sortedProducts = useMemo(() => {
    if (sort === 'relevance') return filteredProducts
    return [...filteredProducts].sort((a, b) =>
      sort === 'price-asc' ? a.price_value - b.price_value : b.price_value - a.price_value,
    )
  }, [filteredProducts, sort])

  const clearLocalFilters = () => {
    setPriceFilter('all')
    setUseCaseFilter('all')
    setBrandFilter('all')
  }

  return (
    <main className="storefront-shell">
      <section className="storefront-toolbar">
        <div>
          <p className="storefront-toolbar-kicker">Catalog đang duyệt</p>
          <h1 className="storefront-toolbar-title">
            {query ? `Kết quả cho "${query}"` : 'Khám phá sản phẩm'}
          </h1>
        </div>
        <div className="storefront-toolbar-stats">
          <div className="storefront-toolbar-pill">
            <span>Đang hiển thị</span>
            <strong>{sortedProducts.length}</strong>
          </div>
          <div className="storefront-toolbar-pill">
            <span>Danh mục</span>
            <strong>{activeCategory || 'Tất cả'}</strong>
          </div>
        </div>
      </section>

      <SmartSuggestionBar />

      <section className="filter-panel" aria-label="Lọc sản phẩm">
        <div className="filter-panel-header">
          <div>
            <p className="filter-panel-kicker">Retail filters</p>
            <h2 className="filter-panel-title">Thu hẹp danh sách theo thứ người mua thực sự quan tâm</h2>
          </div>
          <div className="filter-panel-tools">
            <SlidersHorizontal size={15} />
            <span>{sortedProducts.length} kết quả</span>
          </div>
        </div>

        <div className="filter-group">
          <span className="filter-group-label">Danh mục</span>
          <div className="filter-chip-row">
            {CATEGORIES.map((cat) => (
              <button
                key={cat.value}
                className={`filter-tab ${activeCategory === cat.value ? 'active' : ''}`}
                onClick={() => void filterByCategory(cat.value)}
                aria-pressed={activeCategory === cat.value}
              >
                {cat.label}
              </button>
            ))}
          </div>
        </div>

        <div className="filter-grid">
          <div className="filter-group">
            <span className="filter-group-label">Khoảng giá</span>
            <div className="filter-chip-row">
              {PRICE_FILTERS.map((item) => (
                <button
                  key={item.value}
                  className={`filter-pill ${priceFilter === item.value ? 'active' : ''}`}
                  onClick={() => setPriceFilter(item.value)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          <div className="filter-group">
            <span className="filter-group-label">Nhu cầu</span>
            <div className="filter-chip-row">
              {USE_CASE_FILTERS.map((item) => (
                <button
                  key={item.value}
                  className={`filter-pill ${useCaseFilter === item.value ? 'active' : ''}`}
                  onClick={() => setUseCaseFilter(item.value)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          <div className="filter-group">
            <span className="filter-group-label">Thương hiệu</span>
            <div className="filter-chip-row">
              {availableBrands.map((brand) => (
                <button
                  key={brand}
                  className={`filter-pill ${brandFilter === brand ? 'active' : ''}`}
                  onClick={() => setBrandFilter(brand)}
                >
                  {brand === 'all' ? 'Tất cả hãng' : brand}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="filter-panel-footer">
          <div className="relative">
            <select
              className="filter-pill appearance-none cursor-pointer pr-8"
              value={sort}
              onChange={(e) => setSort(e.target.value as typeof sort)}
              aria-label="Sắp xếp"
              style={{ paddingRight: '32px' }}
            >
              {SORT_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <ChevronDown
              size={13}
              className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2"
              style={{ color: '#94A3B8' }}
            />
          </div>

          <button className="btn-secondary" onClick={clearLocalFilters}>
            Xóa filter cục bộ
          </button>
        </div>
      </section>

      {error && (
        <div
          className="mb-6 rounded-xl border px-4 py-3 text-sm"
          style={{ backgroundColor: '#FFF1F2', borderColor: '#FBCDD2', color: '#D70018' }}
          role="alert"
        >
          {error}
        </div>
      )}

      {!loading && sortedProducts.length === 0 && !error ? (
        <EmptyState
          message="Thử nới khoảng giá, đổi thương hiệu hoặc chuyển nhu cầu để thấy thêm lựa chọn."
          actionLabel="Xóa filter"
          onAction={clearLocalFilters}
        />
      ) : (
        <section className="product-grid-retail" aria-label="Danh sách sản phẩm">
          {loading
            ? Array.from({ length: 8 }, (_, i) => <ProductSkeleton key={i} />)
            : sortedProducts.map((product) => <ProductCard key={product.code} product={product} />)}
        </section>
      )}

      {!loading && products.length > 0 && (
        <div className="mt-12 flex flex-col items-center gap-3">
          {hasMore ? (
            <button
              className="btn-secondary"
              style={{ minWidth: '220px', padding: '11px 24px' }}
              onClick={() => void loadMore()}
              disabled={loadingMore}
            >
              {loadingMore && <LoaderCircle className="mr-2 animate-spin" size={15} />}
              {loadingMore ? 'Đang tải thêm…' : 'Xem thêm sản phẩm'}
            </button>
          ) : (
            <p style={{ fontSize: '13px', color: '#94A3B8' }}>Đã hiển thị toàn bộ {products.length} sản phẩm đã tải.</p>
          )}
        </div>
      )}
    </main>
  )
}

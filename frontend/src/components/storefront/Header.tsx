import { useState } from 'react'
import { GitCompareArrows, Heart, Search, ShoppingBag, Sparkles, X } from 'lucide-react'
import { useCatalogStore } from '@/stores/catalogStore'
import { useCommerceStore } from '@/stores/commerceStore'
import { useCopilotStore } from '@/stores/copilotStore'

export function Header() {
  const [searchValue, setSearchValue] = useState('')
  const search = useCatalogStore((s) => s.search)
  const cart = useCommerceStore((s) => s.cart)
  const favoriteCodes = useCommerceStore((s) => s.favoriteCodes)
  const compareProducts = useCommerceStore((s) => s.compareProducts)
  const openCart = useCommerceStore((s) => s.openCart)
  const openCompare = useCommerceStore((s) => s.openCompare)
  const openCopilot = useCopilotStore((s) => s.open)

  const submitSearch = (event: React.FormEvent) => {
    event.preventDefault()
    if (searchValue.trim()) void search(searchValue.trim())
  }

  const clearSearch = () => {
    setSearchValue('')
    void search('')
  }

  return (
    <header className="top-nav" role="banner">
      <div className="top-utility-strip">
        <div className="top-utility-inner">
          <span>Tech retail by AURA</span>
          <span>Catalog thật · giá thật · tư vấn theo nhu cầu</span>
        </div>
      </div>

      <div className="main-nav-red">
        <div className="main-nav-inner">
          <a href="#" className="aura-brand" aria-label="AURA trang chủ">
            AURA
          </a>

          <form className="search-form" onSubmit={submitSearch} role="search">
            <span className="search-icon" aria-hidden="true">
              <Search size={18} />
            </span>
            <input
              className="search-bar"
              value={searchValue}
              onChange={(event) => setSearchValue(event.target.value)}
              placeholder="Tìm laptop, điện thoại, mã SKU..."
              aria-label="Tìm kiếm sản phẩm"
              type="search"
            />
            {searchValue ? (
              <button type="button" className="search-clear-button" onClick={clearSearch} aria-label="Xóa tìm kiếm">
                <X size={16} />
              </button>
            ) : (
              <span className="search-submit-hint">Enter</span>
            )}
          </form>

          <div className="nav-actions">
            <button
              className="nav-action-button"
              onClick={openCompare}
              aria-label={compareProducts.length > 0 ? `So sánh (${compareProducts.length})` : 'So sánh sản phẩm'}
              title="So sánh"
            >
              <GitCompareArrows size={19} />
              {compareProducts.length > 0 && <span className="nav-action-count">{compareProducts.length}</span>}
            </button>

            <button
              className="nav-action-button"
              aria-label={favoriteCodes.length > 0 ? `Yêu thích (${favoriteCodes.length})` : 'Danh sách yêu thích'}
              title="Yêu thích"
            >
              <Heart size={19} />
              {favoriteCodes.length > 0 && <span className="nav-action-count">{favoriteCodes.length}</span>}
            </button>

            <button
              className="nav-action-button nav-action-cart"
              onClick={openCart}
              aria-label={cart.length > 0 ? `Giỏ hàng (${cart.length})` : 'Giỏ hàng'}
              title="Giỏ hàng"
            >
              <ShoppingBag size={19} />
              {cart.length > 0 && <span className="nav-action-count">{cart.length}</span>}
            </button>

            <button className="nav-ai-button" onClick={openCopilot} aria-label="Mở tư vấn AI">
              <Sparkles size={14} />
              Hỏi AI
            </button>
          </div>
        </div>
      </div>
    </header>
  )
}

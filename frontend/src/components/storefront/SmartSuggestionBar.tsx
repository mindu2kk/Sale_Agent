import { BriefcaseBusiness, Gamepad2, GitCompareArrows, Laptop2, Sparkles, Wallet } from 'lucide-react'
import { useCopilotStore } from '@/stores/copilotStore'
import { useCatalogStore } from '@/stores/catalogStore'
import { useCommerceStore } from '@/stores/commerceStore'

const INTENTS = [
  {
    label: 'Văn phòng',
    icon: BriefcaseBusiness,
    category: 'Laptop',
    query: 'Tư vấn laptop học tập văn phòng',
  },
  {
    label: 'Gaming',
    icon: Gamepad2,
    category: 'Laptop',
    query: 'Tư vấn laptop chơi game, cần GPU rời',
  },
  {
    label: 'Mỏng nhẹ',
    icon: Laptop2,
    category: 'Laptop',
    query: 'Tư vấn laptop mỏng nhẹ, dễ mang theo',
  },
  {
    label: 'Dưới 20 triệu',
    icon: Wallet,
    category: '',
    query: 'Tư vấn sản phẩm công nghệ tốt nhất dưới 20 triệu',
  },
] as const

export function SmartSuggestionBar() {
  const openWithPrefill = useCopilotStore((s) => s.openWithPrefill)
  const filterByCategory = useCatalogStore((s) => s.filterByCategory)
  const openCompare = useCommerceStore((s) => s.openCompare)
  const compareProducts = useCommerceStore((s) => s.compareProducts)

  const handleIntent = (intent: { label: string; category: string; query: string }) => {
    if (intent.category) void filterByCategory(intent.category)
    openWithPrefill(intent.query)
  }

  return (
    <section className="smart-suggestion-bar">
      <div className="smart-suggestion-copy">
        <div className="smart-suggestion-icon">
          <Sparkles size={18} style={{ color: '#D70018' }} />
        </div>
        <div>
          <p className="smart-suggestion-kicker">AI commerce helper</p>
          <h3 className="smart-suggestion-title">Bạn đang tìm máy theo nhu cầu nào?</h3>
          <p className="smart-suggestion-description">
            Chọn lối tư vấn nhanh để AURA lọc đúng hướng trước khi bạn đi sâu vào từng mẫu.
          </p>
        </div>
      </div>

      <div className="smart-suggestion-actions">
        {INTENTS.map((intent) => {
          const Icon = intent.icon
          return (
            <button
              key={intent.label}
              className="suggest-chip suggest-chip-strong"
              onClick={() => handleIntent(intent)}
              aria-label={`Gợi ý: ${intent.label}`}
            >
              <Icon size={14} />
              {intent.label}
            </button>
          )
        })}

        <button
          className="suggest-chip suggest-chip-compare"
          onClick={openCompare}
          aria-label={compareProducts.length >= 2 ? `So sánh ${compareProducts.length} sản phẩm` : 'Mở so sánh'}
        >
          <GitCompareArrows size={14} />
          {compareProducts.length >= 2 ? `So sánh ngay ${compareProducts.length} mẫu` : 'So sánh nhanh'}
        </button>
      </div>
    </section>
  )
}

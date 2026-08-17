import { useEffect } from 'react'
import { CopilotDrawer } from '@/components/copilot/CopilotDrawer'
import { CommercePanels } from '@/components/storefront/CommercePanels'
import { HighlightTooltip } from '@/components/copilot/HighlightTooltip'
import { Footer } from '@/components/storefront/Footer'
import { Header } from '@/components/storefront/Header'
import { ProductGrid } from '@/components/storefront/ProductGrid'
import { useCatalogStore } from '@/stores/catalogStore'

export default function App() {
  const loadFeatured = useCatalogStore((state) => state.loadFeatured)

  useEffect(() => {
    void loadFeatured()
  }, [loadFeatured])

  return (
    <div className="flex min-h-screen flex-col bg-background font-sans text-text-primary">
      <Header />
      <div className="flex-1">
        <ProductGrid />
      </div>
      <Footer />
      <CommercePanels />
      <CopilotDrawer />
      <HighlightTooltip />
    </div>
  )
}

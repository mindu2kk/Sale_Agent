import type { Bundle, Product } from '@/types/commerce'

export function productTone(product: Product): string {
  const palettes: Record<string, string> = {
    Apple: '#f1f1f3',
    Samsung: '#eef2f5',
    Google: '#f0f2ef',
    HP: '#edf1f4',
    Lenovo: '#f3efef',
    Motorola: '#eef2f1',
  }
  return palettes[product.brand] ?? '#f4f4f5'
}

export function buildBundle(products: Product[]): Bundle | undefined {
  if (products.length < 3) return undefined
  return {
    title: 'Nhóm sản phẩm từ catalog AI',
    savings: 'theo tư vấn',
    items: products.slice(0, 3),
  }
}

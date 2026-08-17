import type {
  ChatMessage,
  ChatResponse,
  ConversationState,
  Product,
} from '@/types/commerce'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`
}

export async function sendChatMessage(
  message: string,
  history: ChatMessage[] = [],
  conversationState: ConversationState | null = null,
): Promise<ChatResponse> {
  try {
    const response = await fetch(apiUrl('/api/chat'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        history: history.slice(-12).map((turn) => ({
          role: turn.role,
          text: turn.text,
          product_codes: turn.products?.map((product) => product.code) ?? [],
        })),
        conversation_state: conversationState,
      }),
    })
    if (!response.ok) throw new Error('Không thể kết nối với hệ thống AI lúc này.')
    return response.json() as Promise<ChatResponse>
  } catch {
    throw new Error('Không thể kết nối với hệ thống AI lúc này.')
  }
}

export async function getFeaturedProducts(limit = 6): Promise<Product[]> {
  const response = await fetch(apiUrl(`/api/products/featured?limit=${limit}`))
  if (!response.ok) throw new Error('Không thể tải catalog sản phẩm.')
  const data = (await response.json()) as { products: Product[] }
  return data.products
}

export async function searchProducts(options: {
  query?: string
  category?: string
  limit?: number
  offset?: number
}): Promise<Product[]> {
  const params = new URLSearchParams()
  if (options.query) params.set('q', options.query)
  if (options.category) params.set('category', options.category)
  params.set('limit', String(options.limit ?? 12))
  params.set('offset', String(options.offset ?? 0))
  const response = await fetch(apiUrl(`/api/products?${params}`))
  if (!response.ok) throw new Error('Không thể tìm kiếm catalog sản phẩm.')
  const data = (await response.json()) as { products: Product[] }
  return data.products
}

import { create } from 'zustand'
import { searchProducts } from '@/lib/api'
import type { Product } from '@/types/commerce'

interface CatalogStore {
  products: Product[]
  loading: boolean
  loadingMore: boolean
  hasMore: boolean
  error: string
  activeCategory: string
  query: string
  sort: 'relevance' | 'price-asc' | 'price-desc'
  loadFeatured: () => Promise<void>
  filterByCategory: (category: string) => Promise<void>
  search: (query: string) => Promise<void>
  setSort: (sort: CatalogStore['sort']) => void
  loadMore: () => Promise<void>
}

const PAGE_SIZE = 15

function categoryFromQuery(query: string): string {
  const normalized = query
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
  if (
    ['macbook', 'laptop', 'notebook', 'may tinh xach tay'].some((term) =>
      normalized.includes(term),
    )
  ) {
    return 'Laptop'
  }
  if (
    ['iphone', 'dien thoai', 'smartphone', 'phone'].some((term) =>
      normalized.includes(term),
    )
  ) {
    return 'Mobile Phone'
  }
  return ''
}

export const useCatalogStore = create<CatalogStore>((set) => ({
  products: [],
  loading: false,
  loadingMore: false,
  hasMore: true,
  error: '',
  activeCategory: '',
  query: '',
  sort: 'relevance',
  loadFeatured: async () => {
    set({ loading: true, error: '', activeCategory: '', query: '', hasMore: true })
    try {
      const products = await searchProducts({ limit: PAGE_SIZE })
      set({ products, loading: false, hasMore: products.length === PAGE_SIZE })
    } catch (error) {
      set({
        loading: false,
        error: error instanceof Error ? error.message : 'Không thể tải catalog.',
      })
    }
  },
  filterByCategory: async (category) => {
    set({ loading: true, error: '', activeCategory: category, hasMore: true })
    try {
      const products = await searchProducts({
        query: useCatalogStore.getState().query,
        category,
        limit: PAGE_SIZE,
      })
      set({ products, loading: false, hasMore: products.length === PAGE_SIZE })
    } catch (error) {
      set({
        loading: false,
        error: error instanceof Error ? error.message : 'Không thể tải catalog.',
      })
    }
  },
  search: async (query) => {
    const inferredCategory = categoryFromQuery(query)
    const category = inferredCategory || useCatalogStore.getState().activeCategory
    set({
      loading: true,
      error: '',
      query,
      activeCategory: inferredCategory || useCatalogStore.getState().activeCategory,
      hasMore: true,
    })
    try {
      const products = await searchProducts({ query, category, limit: PAGE_SIZE })
      set({ products, loading: false, hasMore: products.length === PAGE_SIZE })
    } catch (error) {
      set({
        loading: false,
        error: error instanceof Error ? error.message : 'Không thể tìm sản phẩm.',
      })
    }
  },
  setSort: (sort) => set({ sort }),
  loadMore: async () => {
    const state = useCatalogStore.getState()
    if (state.loadingMore || !state.hasMore) return
    set({ loadingMore: true, error: '' })
    try {
      const nextProducts = await searchProducts({
        query: state.query,
        category: state.activeCategory,
        limit: PAGE_SIZE,
        offset: state.products.length,
      })
      set({
        products: [...state.products, ...nextProducts],
        loadingMore: false,
        hasMore: nextProducts.length === PAGE_SIZE,
      })
    } catch (error) {
      set({
        loadingMore: false,
        error: error instanceof Error ? error.message : 'Không thể tải thêm sản phẩm.',
      })
    }
  },
}))

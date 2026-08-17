import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Product } from '@/types/commerce'

interface CommerceStore {
  selectedProduct: Product | null
  detailOpen: boolean
  cartOpen: boolean
  compareOpen: boolean
  cart: Product[]
  favoriteCodes: string[]
  compareProducts: Product[]
  openProduct: (product: Product) => void
  closeProduct: () => void
  openCart: () => void
  closeCart: () => void
  openCompare: () => void
  closeCompare: () => void
  addToCart: (product: Product) => void
  removeFromCart: (code: string) => void
  toggleFavorite: (code: string) => void
  toggleCompare: (product: Product) => void
}

export const useCommerceStore = create<CommerceStore>()(
  persist(
    (set) => ({
      selectedProduct: null,
      detailOpen: false,
      cartOpen: false,
      compareOpen: false,
      cart: [],
      favoriteCodes: [],
      compareProducts: [],
      openProduct: (selectedProduct) =>
        set({ selectedProduct, detailOpen: true, cartOpen: false, compareOpen: false }),
      closeProduct: () => set({ detailOpen: false }),
      openCart: () => set({ cartOpen: true, detailOpen: false, compareOpen: false }),
      closeCart: () => set({ cartOpen: false }),
      openCompare: () => set({ compareOpen: true, detailOpen: false, cartOpen: false }),
      closeCompare: () => set({ compareOpen: false }),
      addToCart: (product) =>
        set((state) => ({
          cart: state.cart.some((item) => item.code === product.code)
            ? state.cart
            : [...state.cart, product],
        })),
      removeFromCart: (code) =>
        set((state) => ({ cart: state.cart.filter((item) => item.code !== code) })),
      toggleFavorite: (code) =>
        set((state) => ({
          favoriteCodes: state.favoriteCodes.includes(code)
            ? state.favoriteCodes.filter((item) => item !== code)
            : [...state.favoriteCodes, code],
        })),
      toggleCompare: (product) =>
        set((state) => {
          if (state.compareProducts.some((item) => item.code === product.code)) {
            return {
              compareProducts: state.compareProducts.filter(
                (item) => item.code !== product.code,
              ),
            }
          }
          if (
            state.compareProducts.length > 0
            && state.compareProducts[0].category !== product.category
          ) {
            return { compareProducts: [product] }
          }
          if (state.compareProducts.length >= 3) {
            return { compareProducts: [...state.compareProducts.slice(1), product] }
          }
          return { compareProducts: [...state.compareProducts, product] }
        }),
    }),
    {
      name: 'aura-commerce',
      partialize: (state) => ({
        cart: state.cart,
        favoriteCodes: state.favoriteCodes,
        compareProducts: state.compareProducts,
      }),
    },
  ),
)

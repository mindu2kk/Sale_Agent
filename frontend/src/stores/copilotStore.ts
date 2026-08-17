import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { buildBundle } from '@/data/products'
import { sendChatMessage } from '@/lib/api'
import type { ChatMessage, ConversationState } from '@/types/commerce'

interface CopilotStore {
  isOpen: boolean
  isLoading: boolean
  input: string
  cartCount: number
  messages: ChatMessage[]
  chatExpiresAt: number | null
  conversationState: ConversationState | null
  open: () => void
  close: () => void
  setInput: (value: string) => void
  openWithPrefill: (value: string) => void
  addToCart: (quantity?: number) => void
  sendMessage: (message?: string) => Promise<void>
  clearExpiredChat: () => void
  refreshChatTtl: () => void
}

const CHAT_TTL_MS = 5 * 60 * 1000
let chatExpiryTimer: ReturnType<typeof setTimeout> | undefined

export const CHAT_CONNECTIVITY_FALLBACK_TEXT =
  'Mình chưa kết nối được máy chủ tư vấn lúc này. Bạn vẫn có thể tiếp tục nhập nhu cầu, hoặc bật backend rồi gửi lại để AURA lọc catalog và so sánh bằng dữ liệu thật.'

const welcomeMessage: ChatMessage = {
  id: 'welcome',
  role: 'assistant',
  text: 'Chào bạn. Hãy gửi mã SKU, mức giá hoặc nhu cầu để mình lọc nhanh cho đúng.',
}

function scheduleChatExpiry(
  expiresAt: number | null,
  clearExpiredChat: () => void,
) {
  if (chatExpiryTimer) clearTimeout(chatExpiryTimer)
  chatExpiryTimer = undefined
  if (!expiresAt) return

  const remaining = expiresAt - Date.now()
  if (remaining <= 0) {
    clearExpiredChat()
    return
  }
  chatExpiryTimer = setTimeout(clearExpiredChat, remaining)
}

export const useCopilotStore = create<CopilotStore>()(
  persist(
    (set, get) => ({
      isOpen: false,
      isLoading: false,
      input: '',
      cartCount: 0,
      messages: [welcomeMessage],
      chatExpiresAt: null,
      conversationState: null,
      open: () => set({ isOpen: true }),
      close: () => set({ isOpen: false }),
      setInput: (input) => set({ input }),
      openWithPrefill: (input) => set({ isOpen: true, input }),
      addToCart: (quantity = 1) =>
        set((state) => ({ cartCount: state.cartCount + quantity })),
      clearExpiredChat: () => {
        if (get().isLoading) {
          get().refreshChatTtl()
          return
        }
        if (chatExpiryTimer) clearTimeout(chatExpiryTimer)
        chatExpiryTimer = undefined
        set({
          input: '',
          messages: [welcomeMessage],
          chatExpiresAt: null,
          conversationState: null,
        })
      },
      refreshChatTtl: () => {
        const chatExpiresAt = Date.now() + CHAT_TTL_MS
        set({ chatExpiresAt })
        scheduleChatExpiry(chatExpiresAt, get().clearExpiredChat)
      },
      sendMessage: async (providedMessage) => {
        const message = (providedMessage ?? get().input).trim()
        if (!message || get().isLoading) return
        const history = get().messages.filter((item) => item.id !== 'welcome')
        get().refreshChatTtl()

        set((state) => ({
          input: '',
          isLoading: true,
          messages: [
            ...state.messages,
            { id: `user-${Date.now()}`, role: 'user', text: message },
          ],
        }))

        try {
          const response = await sendChatMessage(
            message,
            history,
            get().conversationState,
          )
          get().refreshChatTtl()
          set((state) => ({
            isLoading: false,
            conversationState: response.conversation_state,
            messages: [
              ...state.messages,
              {
                id: `assistant-${Date.now()}`,
                role: 'assistant',
                text: response.text,
                products: response.products,
                bundle: response.suggest_bundle
                  ? buildBundle(response.products)
                  : undefined,
                workflowStatus: response.workflow_status,
                aiMode: response.ai_mode,
                sources: response.sources,
                verification: response.verification,
              },
            ],
          }))
        } catch (error) {
          get().refreshChatTtl()
          const text =
            error instanceof Error
              && error.message
              && !/failed to fetch|networkerror|load failed|không thể kết nối/i.test(error.message)
              ? error.message
              : CHAT_CONNECTIVITY_FALLBACK_TEXT
          set((state) => ({
            isLoading: false,
            messages: [
              ...state.messages,
              { id: `error-${Date.now()}`, role: 'assistant', text },
            ],
          }))
        }
      },
    }),
    {
      name: 'sales-copilot-session',
      partialize: (state) => ({
        cartCount: state.cartCount,
        messages: state.messages.slice(-24),
        chatExpiresAt: state.chatExpiresAt,
        conversationState: state.conversationState,
      }),
      merge: (persistedState, currentState) => {
        const persisted = persistedState as Partial<CopilotStore>
        if (
          !persisted.chatExpiresAt
          || persisted.chatExpiresAt <= Date.now()
        ) {
          return currentState
        }
        return { ...currentState, ...persisted }
      },
      onRehydrateStorage: () => (state) => {
        if (state?.chatExpiresAt) {
          scheduleChatExpiry(state.chatExpiresAt, state.clearExpiredChat)
        }
      },
    },
  ),
)

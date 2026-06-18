import { create } from 'zustand'
import type { Message } from '../lib/api'

export type AgentState = 'idle' | 'thinking' | 'streaming' | 'error'

interface ChatStore {
  threadId: string | null
  messages: Message[]
  agentState: AgentState
  streamingContent: string   // token buffer for in-progress assistant message
  errorMessage: string

  // actions
  setThreadId: (id: string) => void
  setMessages: (msgs: Message[]) => void
  addUserMessage: (content: string) => void
  appendToken: (token: string) => void
  finalizeAssistantMessage: () => void
  setAgentState: (s: AgentState) => void
  setError: (msg: string) => void
  clearError: () => void
  reset: () => void
}

export const useChatStore = create<ChatStore>((set, get) => ({
  threadId: null,
  messages: [],
  agentState: 'idle',
  streamingContent: '',
  errorMessage: '',

  setThreadId: (id) => set({ threadId: id }),

  setMessages: (msgs) => set({ messages: msgs }),

  addUserMessage: (content) => {
    const msg: Message = {
      id: `local-${Date.now()}`,
      thread_id: get().threadId ?? '',
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    }
    set((s) => ({ messages: [...s.messages, msg] }))
  },

  appendToken: (token) =>
    set((s) => ({ streamingContent: s.streamingContent + token })),

  finalizeAssistantMessage: () => {
    const content = get().streamingContent
    if (!content) return
    const msg: Message = {
      id: `assistant-${Date.now()}`,
      thread_id: get().threadId ?? '',
      role: 'assistant',
      content,
      created_at: new Date().toISOString(),
    }
    set((s) => ({ messages: [...s.messages, msg], streamingContent: '' }))
  },

  setAgentState: (s) => set({ agentState: s }),

  setError: (msg) => set({ errorMessage: msg, agentState: 'error' }),

  clearError: () => set({ errorMessage: '', agentState: 'idle' }),

  reset: () => set({
    messages: [],
    agentState: 'idle',
    streamingContent: '',
    errorMessage: '',
  }),
}))

import { useEffect, useRef } from 'react'
import { MessageList } from './MessageList'
import { ChatInput } from './ChatInput'
import { useChatStore } from '../stores/chatStore'
import { createThread, getMessages, streamChat } from '../lib/api'

const SESSION_KEY = 'ai_copilot_thread_id'
const STREAM_TIMEOUT_MS = 30_000

export function ChatCore() {
  const store = useChatStore()
  const abortRef = useRef<AbortController | null>(null)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // ── Session init ────────────────────────────────────────────────────────
  useEffect(() => {
    async function init() {
      const savedId = sessionStorage.getItem(SESSION_KEY)

      if (savedId) {
        try {
          const msgs = await getMessages(savedId, 50)
          store.setThreadId(savedId)
          store.setMessages(msgs)
          return
        } catch {
          // Thread not found or error — start fresh
          sessionStorage.removeItem(SESSION_KEY)
        }
      }

      // Create new thread
      try {
        const newId = await createThread()
        store.setThreadId(newId)
        sessionStorage.setItem(SESSION_KEY, newId)
      } catch {
        store.setError('Không thể kết nối đến máy chủ. Hãy kiểm tra backend đang chạy.')
      }
    }

    init()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Send handler ────────────────────────────────────────────────────────
  async function handleSend(message: string) {
    const threadId = store.threadId
    if (!threadId) return

    // Cancel any existing stream
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    // Optimistic: show user message immediately
    store.addUserMessage(message)
    store.setAgentState('thinking')

    // 30s timeout — hide indicator and show error if no first token
    timeoutRef.current = setTimeout(() => {
      if (store.agentState === 'thinking') {
        controller.abort()
        store.setError('Yêu cầu hết thời gian. Vui lòng thử lại.')
      }
    }, STREAM_TIMEOUT_MS)

    await streamChat(
      message,
      threadId,
      // onToken
      (token) => {
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current)
          timeoutRef.current = null
        }
        store.setAgentState('streaming')
        store.appendToken(token)
      },
      // onDone
      (_status) => {
        store.finalizeAssistantMessage()
        store.setAgentState('idle')
      },
      // onError
      (errMsg) => {
        store.finalizeAssistantMessage() // save whatever we got
        store.setError(errMsg)
      },
      controller.signal,
    )

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }
  }

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <>
      {/* Error banner */}
      {store.errorMessage && (
        <div
          style={{
            position: 'fixed', top: 80, left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 50,
            background: 'color-mix(in oklab, oklch(0.4 0.2 25) 80%, transparent)',
            border: '1px solid oklch(0.65 0.27 25)',
            backdropFilter: 'blur(16px)',
            borderRadius: 12,
            padding: '10px 20px',
            color: 'white',
            fontSize: 14,
            display: 'flex', alignItems: 'center', gap: 12,
          }}
        >
          <span>{store.errorMessage}</span>
          <button
            onClick={store.clearError}
            style={{
              background: 'none', border: 'none',
              color: 'rgba(255,255,255,0.7)', cursor: 'pointer',
              fontSize: 18, lineHeight: 1,
            }}
          >
            ×
          </button>
        </div>
      )}

      {/* Message list — scrollable area */}
      <div
        style={{
          position: 'fixed',
          inset: '80px 0 88px',
          maxWidth: 720,
          margin: '0 auto',
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          left: '50%',
          transform: 'translateX(-50%)',
        }}
      >
        <MessageList />
      </div>

      {/* Input bar */}
      <ChatInput onSend={handleSend} />
    </>
  )
}

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowUp,
  Bot,
  ChevronRight,
  GripVertical,
  Maximize2,
  MessagesSquare,
  Mic,
  Minimize2,
  Sparkles,
  X,
} from 'lucide-react'
import { CHAT_CONNECTIVITY_FALLBACK_TEXT, useCopilotStore } from '@/stores/copilotStore'
import { useCommerceStore } from '@/stores/commerceStore'
import { AmbientVoiceVisualizer } from './AmbientVoiceVisualizer'
import { ChatMessageText } from './ChatMessageText'
import { EcosystemBundleCard } from './EcosystemBundleCard'
import { InlineProductCard } from './InlineProductCard'

const QUICK_PROMPTS = [
  {
    label: 'Laptop học tập',
    detail: 'Dưới 20 triệu, bền pin',
    message: 'Laptop học tập dưới 20 triệu',
  },
  {
    label: 'So sánh 2 mẫu',
    detail: 'Chọn máy đáng mua hơn',
    message: 'So sánh laptop gaming tầm trung',
  },
  {
    label: 'Điện thoại camera',
    detail: 'Ảnh đẹp, dễ dùng',
    message: 'Điện thoại chụp ảnh đẹp',
  },
]

const MIN_DRAWER_WIDTH = 400
const MAX_DRAWER_WIDTH = 860
const TECHNICAL_ERROR_PATTERN = /failed to fetch|networkerror|load failed/i

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

export function CopilotDrawer() {
  const {
    isOpen,
    isLoading,
    input,
    messages,
    conversationState,
    open,
    close,
    setInput,
    sendMessage,
  } = useCopilotStore()

  const [isListening, setIsListening] = useState(false)
  const [drawerWidth, setDrawerWidth] = useState(430)
  const [isExpanded, setIsExpanded] = useState(false)
  const [isResizing, setIsResizing] = useState(false)

  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const selectedProduct = useCommerceStore((s) => s.selectedProduct)

  const contextTitle = selectedProduct ? selectedProduct.name : 'Tư vấn tự do theo nhu cầu của bạn'
  const contextMeta = selectedProduct
    ? `${selectedProduct.brand} · ${selectedProduct.code} · ${selectedProduct.price}`
    : conversationState?.category
      ? `Đang hiểu danh mục: ${conversationState.category}`
      : 'Hỏi ngắn, lọc nhanh, so sánh rõ ràng.'
  const contextMode = selectedProduct ? 'Context sản phẩm' : conversationState?.category ? 'Đã nhận diện danh mục' : 'Đang đọc nhu cầu'

  const maxWidth = typeof window === 'undefined'
    ? MAX_DRAWER_WIDTH
    : Math.max(MIN_DRAWER_WIDTH, Math.min(MAX_DRAWER_WIDTH, window.innerWidth - 24))

  const resolvedDrawerWidth = useMemo(() => {
    if (typeof window === 'undefined') return drawerWidth
    return isExpanded ? Math.min(maxWidth, Math.max(620, window.innerWidth - 32)) : clamp(drawerWidth, MIN_DRAWER_WIDTH, maxWidth)
  }, [drawerWidth, isExpanded, maxWidth])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [isLoading, isOpen, messages])

  useEffect(() => {
    const textarea = textareaRef.current
    if (!textarea) return
    textarea.style.height = '0px'
    const nextHeight = Math.min(Math.max(textarea.scrollHeight, 30), 180)
    textarea.style.height = `${nextHeight}px`
  }, [input, isListening])

  useEffect(() => {
    if (!isResizing) return undefined

    const handlePointerMove = (event: PointerEvent) => {
      const nextWidth = window.innerWidth - event.clientX
      setDrawerWidth(clamp(nextWidth, MIN_DRAWER_WIDTH, Math.min(MAX_DRAWER_WIDTH, window.innerWidth - 24)))
    }

    const stopResize = () => setIsResizing(false)

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', stopResize)
    document.body.style.userSelect = 'none'
    document.body.style.cursor = 'ew-resize'

    return () => {
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', stopResize)
      document.body.style.userSelect = ''
      document.body.style.cursor = ''
    }
  }, [isResizing])

  const handleVoiceResult = useCallback(
    (value: string) => {
      setIsListening(false)
      if (value.trim()) setInput(value.trim())
    },
    [setInput],
  )

  const submit = () => void sendMessage()

  return (
    <>
      <button className="ai-float-btn" onClick={open} aria-label="Mở AURA Advisor">
        <Sparkles size={13} style={{ color: '#D70018' }} />
        Hỏi AI
      </button>

      <AnimatePresence>
        {isOpen && (
          <>
            <motion.button
              aria-label="Đóng"
              className="drawer-overlay"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18 }}
              onClick={close}
            />

            <motion.aside
              className={`drawer-content ${isExpanded ? 'drawer-content-expanded' : ''}`}
              style={{ width: `${resolvedDrawerWidth}px` }}
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', stiffness: 320, damping: 34 }}
            >
              <div className="drawer-ambient" aria-hidden="true" />

              <button
                className="drawer-resize-handle"
                onPointerDown={() => {
                  setIsExpanded(false)
                  setIsResizing(true)
                }}
                aria-label="Kéo để mở rộng khung chat"
                type="button"
              >
                <GripVertical size={14} />
              </button>

              <header className="drawer-header">
                <div className="drawer-header-copy">
                  <div className="drawer-header-icon">
                    <Bot size={17} />
                  </div>
                  <div>
                    <p className="drawer-header-kicker">AURA Chat</p>
                    <h2 className="drawer-header-title">Trợ lý mua sắm</h2>
                  </div>
                </div>

                <div className="drawer-header-actions">
                  <button
                    className="drawer-utility-button"
                    onClick={() => setIsExpanded((current) => !current)}
                    aria-label={isExpanded ? 'Thu nhỏ khung chat' : 'Mở rộng khung chat'}
                    type="button"
                  >
                    {isExpanded ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
                    <span>{isExpanded ? 'Thu gọn' : 'Mở rộng'}</span>
                  </button>
                  <button className="btn-icon" onClick={close} aria-label="Đóng">
                    <X size={18} />
                  </button>
                </div>
              </header>

              <section className="drawer-intelligence-strip" aria-label="Trạng thái trợ lý">
                <div className="drawer-intelligence-main">
                  <span className="drawer-live-dot" aria-hidden="true" />
                  <div>
                    <span>{contextMode}</span>
                    <strong>{contextTitle}</strong>
                    <small>{contextMeta}</small>
                  </div>
                </div>
                <div className="drawer-intelligence-meta" aria-hidden="true">
                  <span>Catalog</span>
                  <span>Live</span>
                </div>
              </section>

              <section className="drawer-body">
                <div className="drawer-section-title">
                  <MessagesSquare size={14} />
                  <span>Cuộc trò chuyện</span>
                  <small>{messages.length} tin</small>
                </div>

                <div ref={scrollRef} className="drawer-message-stream scrollbar-thin">
                  {messages.map((message) => {
                    const displayText = message.role === 'assistant' && TECHNICAL_ERROR_PATTERN.test(message.text)
                      ? CHAT_CONNECTIVITY_FALLBACK_TEXT
                      : message.text

                    return (
                    <div key={message.id} className={`drawer-message-row ${message.role === 'user' ? 'is-user' : ''}`}>
                      {message.role === 'assistant' && (
                        <div className="drawer-avatar">
                          <Sparkles size={12} style={{ color: '#D70018' }} />
                        </div>
                      )}

                      <div className={message.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-assistant'}>
                        {message.role === 'assistant' ? (
                          <>
                            <ChatMessageText text={displayText} />
                            {message.products && message.products.length > 0 && (
                              <div className="drawer-recommend-block">
                                <div className="drawer-recommend-heading">Sản phẩm liên quan trong câu trả lời</div>
                                <div className="drawer-recommend-grid">
                                  {message.products.slice(0, 3).map((product) => (
                                    <InlineProductCard key={product.id} product={product} />
                                  ))}
                                </div>
                              </div>
                            )}
                            {message.bundle && (
                              <div style={{ marginTop: '12px' }}>
                                <EcosystemBundleCard bundle={message.bundle} />
                              </div>
                            )}
                          </>
                        ) : (
                          <p className="whitespace-pre-wrap">{displayText}</p>
                        )}
                      </div>
                    </div>
                    )
                  })}

                  {isLoading && (
                    <div className="drawer-message-row">
                      <div className="drawer-avatar">
                        <Sparkles size={12} style={{ color: '#D70018' }} />
                      </div>
                      <div className="drawer-loading-card">
                        {[0, 120, 240].map((delay) => (
                          <span
                            key={delay}
                            className="animate-bounce rounded-full"
                            style={{
                              width: '7px',
                              height: '7px',
                              backgroundColor: '#94A3B8',
                              animationDelay: `${delay}ms`,
                            }}
                          />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </section>

              <section className="drawer-prompt-deck" aria-label="Gợi ý nhanh">
                {selectedProduct ? (
                  <>
                    <button
                      className="drawer-prompt-card drawer-prompt-card-strong"
                      onClick={() => void sendMessage(`Phân tích kỹ mẫu ${selectedProduct.code}`)}
                    >
                      <span>Phân tích kỹ</span>
                      <strong>Điểm mạnh, điểm yếu</strong>
                      <ChevronRight size={14} />
                    </button>
                    <button
                      className="drawer-prompt-card"
                      onClick={() => void sendMessage(`Máy ${selectedProduct.name} có hợp văn phòng không?`)}
                    >
                      <span>Fit nhu cầu</span>
                      <strong>Hợp văn phòng?</strong>
                      <ChevronRight size={14} />
                    </button>
                    <button
                      className="drawer-prompt-card"
                      onClick={() => void sendMessage(`So sánh mẫu ${selectedProduct.code} với các máy cùng tầm giá`)}
                    >
                      <span>Đặt cạnh đối thủ</span>
                      <strong>Cùng tầm giá</strong>
                      <ChevronRight size={14} />
                    </button>
                  </>
                ) : (
                  QUICK_PROMPTS.map((prompt, index) => (
                    <button
                      key={prompt.message}
                      className={`drawer-prompt-card ${index === 0 ? 'drawer-prompt-card-strong' : ''}`}
                      onClick={() => void sendMessage(prompt.message)}
                    >
                      <span>{prompt.detail}</span>
                      <strong>{prompt.label}</strong>
                      <ChevronRight size={14} />
                    </button>
                  ))
                )}
              </section>

              <div className="drawer-input-shell">
                {isListening ? (
                  <AmbientVoiceVisualizer onResult={handleVoiceResult} onCancel={() => setIsListening(false)} />
                ) : (
                  <div className="advisor-input-container">
                    <textarea
                      ref={textareaRef}
                      className="scrollbar-none flex-1 resize-none bg-transparent outline-none"
                      style={{
                        color: '#0F172A',
                        fontSize: '14px',
                        lineHeight: 1.6,
                        minHeight: '30px',
                        maxHeight: '180px',
                        padding: '7px 4px 7px 2px',
                      }}
                      placeholder="Hỏi tiếp về mẫu đang xem, nhu cầu hoặc so sánh..."
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault()
                          submit()
                        }
                      }}
                      rows={1}
                      aria-label="Tin nhắn"
                    />

                    <div className="drawer-input-actions">
                      <button className="btn-icon h-10 w-10" onClick={() => setIsListening(true)} aria-label="Giọng nói">
                        <Mic size={16} />
                      </button>
                      <button className="drawer-send-button" onClick={submit} disabled={!input.trim() || isLoading} aria-label="Gửi">
                        <ArrowUp size={16} />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  )
}

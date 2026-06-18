import { useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { StreamedText } from './StreamedText'
import { TypingIndicator } from './TypingIndicator'
import { useChatStore } from '../stores/chatStore'

export function MessageList() {
  const { messages, streamingContent, agentState } = useChatStore()
  const containerRef = useRef<HTMLDivElement>(null)
  const endRef = useRef<HTMLDivElement>(null)

  // Auto-scroll: only if user hasn't scrolled up
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const isNearBottom =
      container.scrollTop + container.clientHeight >= container.scrollHeight - 60
    if (isNearBottom) {
      endRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, streamingContent])

  const userBubble: React.CSSProperties = {
    maxWidth: '80%',
    padding: '10px 16px',
    borderRadius: 18,
    fontSize: 15,
    lineHeight: 1.55,
    backdropFilter: 'blur(20px)',
    WebkitBackdropFilter: 'blur(20px)',
    background: 'color-mix(in oklab, var(--accent) 30%, transparent)',
    border: '1px solid color-mix(in oklab, var(--accent) 50%, transparent)',
    color: 'white',
    alignSelf: 'flex-end',
  }

  const assistantBubble: React.CSSProperties = {
    maxWidth: '80%',
    padding: '10px 16px',
    borderRadius: 18,
    fontSize: 15,
    lineHeight: 1.55,
    backdropFilter: 'blur(20px)',
    WebkitBackdropFilter: 'blur(20px)',
    background: 'color-mix(in oklab, var(--bg-base) 55%, transparent)',
    border: '1px solid color-mix(in oklab, var(--core-ring) 20%, transparent)',
    color: 'white',
    alignSelf: 'flex-start',
  }

  return (
    <div
      ref={containerRef}
      style={{
        flex: 1,
        overflowY: 'auto',
        padding: '20px 16px 8px',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
      }}
    >
      <AnimatePresence initial={false}>
        {messages.map((m) => (
          <motion.div
            key={m.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            style={{
              display: 'flex',
              justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start',
            }}
          >
            <div style={m.role === 'user' ? userBubble : assistantBubble}>
              {m.role === 'user' ? m.content : <StreamedText text={m.content} />}
            </div>
          </motion.div>
        ))}

        {/* In-progress streaming bubble */}
        {streamingContent && (
          <motion.div
            key="streaming"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            style={{ display: 'flex', justifyContent: 'flex-start' }}
          >
            <div style={assistantBubble}>
              <StreamedText text={streamingContent} />
            </div>
          </motion.div>
        )}

        {/* Typing indicator while waiting for first token */}
        {agentState === 'thinking' && !streamingContent && (
          <motion.div
            key="typing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <TypingIndicator />
          </motion.div>
        )}
      </AnimatePresence>

      <div ref={endRef} />
    </div>
  )
}

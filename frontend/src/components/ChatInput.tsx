import { useRef, useState } from 'react'
import { Send } from 'lucide-react'
import { AIOrb } from './AIOrb'
import { useChatStore } from '../stores/chatStore'

interface ChatInputProps {
  onSend: (message: string) => void
}

export function ChatInput({ onSend }: ChatInputProps) {
  const [text, setText] = useState('')
  const { agentState } = useChatStore()
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const busy = agentState === 'thinking' || agentState === 'streaming'

  const handleSend = () => {
    const trimmed = text.trim()
    if (!trimmed || busy) return
    onSend(trimmed)
    setText('')
    // Reset textarea height
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value)
    // Auto-resize textarea
    const el = e.target
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  }

  return (
    <div
      style={{
        position: 'fixed', bottom: 24, left: '50%',
        transform: 'translateX(-50%)',
        width: 'min(92vw, 640px)',
        zIndex: 40,
      }}
    >
      <div
        className="theme-transition"
        style={{
          display: 'flex', alignItems: 'center', gap: 10,
          borderRadius: 999,
          border: '1px solid color-mix(in oklab, var(--core-ring) 35%, transparent)',
          padding: '8px 10px',
          backdropFilter: 'blur(24px)',
          WebkitBackdropFilter: 'blur(24px)',
          background: 'color-mix(in oklab, var(--bg-base) 70%, transparent)',
          boxShadow: '0 12px 40px -8px color-mix(in oklab, var(--accent) 40%, transparent), inset 0 1px 0 color-mix(in oklab, white 8%, transparent)',
        }}
      >
        <AIOrb state={agentState} />

        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          disabled={busy}
          rows={1}
          aria-label="Nhập tin nhắn"
          placeholder={busy ? 'AI Copilot đang xử lý…' : 'Hỏi AI Copilot về laptop bạn cần…'}
          style={{
            flex: 1, background: 'transparent',
            border: 'none', outline: 'none', resize: 'none',
            color: 'white', fontSize: 15,
            padding: '4px 8px',
            fontFamily: 'inherit',
            lineHeight: 1.5,
            opacity: busy ? 0.6 : 1,
            maxHeight: 120, overflowY: 'auto',
          }}
          // Placeholder color via inline workaround
          onFocus={(e) => (e.target.style.opacity = '1')}
        />

        <button
          type="button"
          onClick={handleSend}
          disabled={busy || !text.trim()}
          aria-label="Gửi tin nhắn"
          style={{
            width: 38, height: 38,
            borderRadius: '50%', border: 'none',
            cursor: busy || !text.trim() ? 'default' : 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
            background: 'linear-gradient(135deg, var(--accent), var(--accent-glow))',
            boxShadow: '0 0 18px var(--accent-glow)',
            opacity: busy || !text.trim() ? 0.4 : 1,
            transition: 'opacity 0.2s',
          }}
        >
          <Send size={16} color="rgba(0,0,0,0.8)" />
        </button>
      </div>
    </div>
  )
}

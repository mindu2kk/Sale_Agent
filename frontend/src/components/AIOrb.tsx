import { motion } from 'framer-motion'
import type { AgentState } from '../stores/chatStore'

interface AIorbProps {
  state: AgentState
}

export function AIOrb({ state }: AIorbProps) {
  return (
    <div style={{ position: 'relative', width: 40, height: 40, flexShrink: 0 }}>
      {/* Core sphere */}
      <motion.div
        style={{
          position: 'absolute', inset: 0,
          borderRadius: '50%',
          background: 'radial-gradient(circle at 30% 30%, var(--accent-glow), var(--accent) 55%, transparent 75%)',
          boxShadow: state === 'error'
            ? '0 0 24px 4px oklch(0.65 0.27 25)'
            : '0 0 24px 4px var(--accent-glow)',
        }}
        animate={
          state === 'idle'
            ? { scale: [1, 1.06, 1] }
            : state === 'streaming'
            ? { scale: [1, 1.1, 0.96, 1.08, 1] }
            : { scale: 1 }
        }
        transition={{
          duration: state === 'streaming' ? 0.5 : 2.4,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />

      {/* Thinking: ring pulse */}
      {state === 'thinking' && (
        <motion.div
          style={{
            position: 'absolute', inset: 0,
            borderRadius: '50%',
            border: '1px solid var(--core-ring)',
          }}
          animate={{ scale: [1, 1.5], opacity: [0.8, 0] }}
          transition={{ duration: 1.2, repeat: Infinity, ease: 'easeOut' }}
        />
      )}

      {/* Streaming: sound bars to the right */}
      {state === 'streaming' && (
        <div
          style={{
            position: 'absolute', right: -28, top: '50%',
            transform: 'translateY(-50%)',
            display: 'flex', alignItems: 'flex-end', gap: 2,
            height: 20, pointerEvents: 'none',
          }}
        >
          {[0, 1, 2, 3, 4].map((i) => (
            <motion.span
              key={i}
              style={{
                width: 2, borderRadius: 2,
                background: 'var(--core-ring)',
                display: 'block',
              }}
              animate={{ height: [4, 16, 6, 20, 4] }}
              transition={{
                duration: 0.8, repeat: Infinity,
                ease: 'easeInOut', delay: i * 0.1,
              }}
            />
          ))}
        </div>
      )}

      {/* Error: static red tint already handled via boxShadow above */}
    </div>
  )
}

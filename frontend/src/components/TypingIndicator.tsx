import { motion } from 'framer-motion'

/** Three animated dots shown while waiting for the first token. */
export function TypingIndicator() {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'flex-start',
      }}
    >
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '12px 18px',
          borderRadius: 18,
          background: 'color-mix(in oklab, var(--bg-base) 55%, transparent)',
          border: '1px solid color-mix(in oklab, var(--core-ring) 20%, transparent)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
        }}
      >
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            style={{
              width: 7, height: 7,
              borderRadius: '50%',
              background: 'var(--core-ring)',
              display: 'block',
            }}
            animate={{ opacity: [0.3, 1, 0.3], y: [0, -4, 0] }}
            transition={{
              duration: 1.2, repeat: Infinity,
              ease: 'easeInOut', delay: i * 0.2,
            }}
          />
        ))}
      </div>
    </div>
  )
}

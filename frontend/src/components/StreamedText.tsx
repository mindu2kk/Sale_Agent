import { motion } from 'framer-motion'

interface StreamedTextProps {
  text: string
}

/**
 * Renders text word-by-word with a blur fade-in animation.
 * Each word slides up and unblurs as it appears — mimics the AI "thinking" aesthetic.
 */
export function StreamedText({ text }: StreamedTextProps) {
  const words = text.split(/(\s+)/)
  return (
    <span style={{ display: 'inline' }}>
      {words.map((w, i) => (
        <motion.span
          key={i}
          initial={{ opacity: 0, y: 6, filter: 'blur(4px)' }}
          animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
          transition={{ duration: 0.3, ease: 'easeOut' }}
          style={{ display: 'inline' }}
        >
          {w}
        </motion.span>
      ))}
    </span>
  )
}

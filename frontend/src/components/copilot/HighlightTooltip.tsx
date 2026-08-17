import { AnimatePresence, motion } from 'framer-motion'
import { Sparkles } from 'lucide-react'
import { useTextSelection } from '@/hooks/useTextSelection'
import { useCopilotStore } from '@/stores/copilotStore'

export function HighlightTooltip() {
  const selection = useTextSelection()
  const openWithPrefill = useCopilotStore((state) => state.openWithPrefill)

  const explainSelection = () => {
    if (!selection) return
    const excerpt =
      selection.text.length > 500 ? `${selection.text.slice(0, 500)}…` : selection.text
    openWithPrefill(`Hãy giải thích cho tôi đoạn này: "${excerpt}"`)
    window.getSelection()?.removeAllRanges()
  }

  return (
    <AnimatePresence>
      {selection && (
        <motion.div
          data-selection-tooltip
          className="fixed z-[70]"
          style={{ left: selection.x, top: selection.y }}
          initial={{ opacity: 0, y: 4, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 4, scale: 0.96 }}
          transition={{ duration: 0.16, ease: 'easeOut' }}
        >
          <button
            onMouseDown={(event) => event.preventDefault()}
            onClick={explainSelection}
            className="selection-button"
          >
            <Sparkles size={14} />
            Giải thích đoạn này
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

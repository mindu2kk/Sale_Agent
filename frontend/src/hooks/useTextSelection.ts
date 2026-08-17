import { useEffect, useState } from 'react'

export interface TextSelection {
  text: string
  x: number
  y: number
}

export function useTextSelection() {
  const [selection, setSelection] = useState<TextSelection | null>(null)

  useEffect(() => {
    const handleMouseUp = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null
      if (target?.closest('[data-copilot-root]')) {
        setSelection(null)
        return
      }

      const selected = window.getSelection()
      const text = selected?.toString().trim() ?? ''
      if (!selected || selected.rangeCount === 0 || text.length <= 10) {
        setSelection(null)
        return
      }

      const rect = selected.getRangeAt(0).getBoundingClientRect()
      setSelection({
        text,
        x: Math.min(Math.max(rect.left + rect.width / 2, 92), window.innerWidth - 92),
        y: Math.max(rect.top, 64),
      })
    }

    const handleMouseDown = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null
      if (!target?.closest('[data-selection-tooltip]')) setSelection(null)
    }

    document.addEventListener('mouseup', handleMouseUp)
    document.addEventListener('mousedown', handleMouseDown)
    return () => {
      document.removeEventListener('mouseup', handleMouseUp)
      document.removeEventListener('mousedown', handleMouseDown)
    }
  }, [])

  return selection
}

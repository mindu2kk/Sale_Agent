import { Sparkles } from "lucide-react";
import { useTextSelection } from "@/hooks/useTextSelection";
import { useCopilot } from "@/lib/copilot-store";
import { motion, AnimatePresence } from "framer-motion";

export function HighlightTooltip() {
  const selection = useTextSelection();
  const { openWithPrefill } = useCopilot();

  const handleClick = () => {
    if (!selection) return;
    
    const snippet = selection.text.length > 280
      ? selection.text.slice(0, 280) + "…"
      : selection.text;
      
    openWithPrefill(`Hãy giải thích cho tôi đoạn này: "${snippet}"`);
    
    // Clear selection
    if (typeof window !== "undefined") {
      window.getSelection()?.removeAllRanges();
    }
  };

  return (
    <AnimatePresence>
      {selection && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.8, y: 10 }}
          transition={{ type: "spring", stiffness: 500, damping: 30 }}
          className="pointer-events-none fixed z-[60]"
          style={{
            left: selection.x,
            top: selection.y,
            transform: "translate(-50%, calc(-100% - 8px))",
          }}
        >
          <button
            type="button"
            onMouseDown={(e) => e.preventDefault()}
            onClick={handleClick}
            className="pointer-events-auto inline-flex items-center gap-1.5 rounded-md bg-neutral-900 px-3 py-1.5 text-xs font-medium text-white shadow-lg transition-colors hover:bg-neutral-800 hover:shadow-xl backdrop-blur-sm border border-neutral-700/50"
          >
            <Sparkles className="h-3.5 w-3.5" />
            Giải thích đoạn này
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
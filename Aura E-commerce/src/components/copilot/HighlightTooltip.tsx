import { Sparkles } from "lucide-react";
import { useTextSelection } from "@/hooks/useTextSelection";
import { useCopilot } from "@/lib/copilot-store";

export function HighlightTooltip() {
  const selection = useTextSelection();
  const openWithPrefill = useCopilot((s) => s.openWithPrefill);

  if (!selection) return null;

  const handleClick = () => {
    const snippet = selection.text.length > 280
      ? selection.text.slice(0, 280) + "…"
      : selection.text;
    openWithPrefill(`Hãy giải thích cho tôi đoạn này: "${snippet}"`);
    if (typeof window !== "undefined") window.getSelection()?.removeAllRanges();
  };

  return (
    <div
      role="tooltip"
      className="copilot-fade-in pointer-events-none fixed z-[60]"
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
        className="pointer-events-auto inline-flex items-center gap-1.5 rounded-md bg-neutral-900 px-3 py-1.5 text-[12px] font-medium text-white shadow-md transition-colors hover:bg-neutral-800"
      >
        <Sparkles className="h-3.5 w-3.5" />
        Giải thích đoạn này
      </button>
    </div>
  );
}
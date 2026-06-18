import { Sparkles } from "lucide-react";
import { useCopilot } from "@/lib/copilot-store";

export function CopilotTrigger() {
  const isOpen = useCopilot((s) => s.isOpen);
  const open = useCopilot((s) => s.open);
  if (isOpen) return null;
  return (
    <button
      onClick={open}
      className="fixed bottom-6 right-6 z-50 flex h-11 items-center gap-2 rounded-full bg-neutral-900 px-5 text-[13px] font-medium text-white shadow-sm transition-all hover:bg-neutral-800 hover:shadow-md"
    >
      <Sparkles className="h-4 w-4" />
      <span>Chat với Chuyên viên</span>
    </button>
  );
}
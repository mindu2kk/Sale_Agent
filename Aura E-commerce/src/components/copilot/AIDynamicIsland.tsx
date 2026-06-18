import { AnimatePresence, motion } from "framer-motion";
import { Loader2, Ticket, X } from "lucide-react";
import { useCopilot } from "@/lib/copilot-store";

export function AIDynamicIsland() {
  const island = useCopilot((s) => s.island);
  const dismiss = useCopilot((s) => s.dismissIsland);

  return (
    <div className="pointer-events-none fixed inset-x-0 top-4 z-[80] flex justify-center">
      <AnimatePresence>
        {island ? (
          <motion.div
            key="island"
            layout
            initial={{ scale: 0.8, opacity: 0, y: -12 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.85, opacity: 0, y: -12 }}
            transition={{ type: "spring", stiffness: 380, damping: 28 }}
            className="pointer-events-auto overflow-hidden rounded-[28px] bg-zinc-900/90 px-4 py-2 text-white shadow-2xl backdrop-blur-xl"
          >
            {island.status === "searching" ? (
              <motion.div
                layout="position"
                className="flex h-7 items-center gap-2 text-[12.5px]"
              >
                <Loader2 className="h-3.5 w-3.5 animate-spin text-white/70" />
                <span className="font-light tracking-tight">{island.label}</span>
                <SoundBars />
              </motion.div>
            ) : (
              <motion.div
                layout="position"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex w-72 items-center gap-3 py-1.5"
              >
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-amber-400/15">
                  <Ticket className="h-4 w-4 text-amber-300" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-[11px] font-light text-white/60">{island.label}</p>
                  <p className="font-mono text-[15px] font-semibold tracking-wider text-amber-300">
                    {island.code}
                  </p>
                  <p className="text-[11px] text-white/70">{island.discount}</p>
                </div>
                <button
                  type="button"
                  onClick={dismiss}
                  className="shrink-0 rounded-full bg-white px-3 py-1.5 text-[11.5px] font-semibold text-neutral-900 transition-colors hover:bg-white/90"
                >
                  Áp dụng
                </button>
                <button
                  type="button"
                  onClick={dismiss}
                  aria-label="Đóng"
                  className="ml-1 flex h-6 w-6 items-center justify-center rounded-full text-white/50 transition-colors hover:bg-white/10 hover:text-white"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </motion.div>
            )}
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}

function SoundBars() {
  return (
    <div className="ml-1 flex items-end gap-[2px]">
      {[0, 1, 2, 3].map((i) => (
        <motion.span
          key={i}
          animate={{ scaleY: [0.4, 1, 0.5, 0.9, 0.3] }}
          transition={{ duration: 1.1, repeat: Infinity, delay: i * 0.12 }}
          className="block h-3 w-[2px] origin-bottom rounded-full bg-white/70"
        />
      ))}
    </div>
  );
}
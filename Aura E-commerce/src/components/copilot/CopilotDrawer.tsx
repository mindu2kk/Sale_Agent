import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import { useCopilot } from "@/lib/copilot-store";
import { ChatMessage, TypingIndicator } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { QuickSuggestions } from "./QuickSuggestions";

export function CopilotDrawer() {
  const isOpen = useCopilot((s) => s.isOpen);
  const close = useCopilot((s) => s.close);
  const messages = useCopilot((s) => s.messages);
  const isLoading = useCopilot((s) => s.isLoading);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  return (
    <>
      <div
        aria-hidden
        onClick={close}
        className={`fixed inset-0 z-40 bg-black/10 transition-opacity ${
          isOpen ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
      />
      <aside
        data-copilot-root
        aria-hidden={!isOpen}
        className={`fixed inset-y-0 right-0 z-50 flex w-full max-w-[400px] flex-col border-l border-gray-100 bg-white transition-transform duration-300 ease-out ${
          isOpen ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-gray-100 px-5">
          <div>
            <p className="text-[14px] font-semibold text-neutral-900">Tư vấn viên trực tuyến</p>
            <p className="text-[11px] font-light text-neutral-500">Phản hồi trong giây lát</p>
          </div>
          <button
            onClick={close}
            aria-label="Đóng"
            className="flex h-8 w-8 items-center justify-center rounded-full text-neutral-500 transition-colors hover:bg-neutral-100 hover:text-neutral-900"
          >
            <X className="h-4 w-4" />
          </button>
        </header>
        <div ref={scrollRef} className="flex-1 space-y-5 overflow-y-auto px-5 py-6">
          {messages.map((m) => (
            <ChatMessage key={m.id} message={m} />
          ))}
          {isLoading ? <TypingIndicator /> : null}
        </div>
        <QuickSuggestions />
        <ChatInput />
      </aside>
    </>
  );
}
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { ArrowUp, Mic } from "lucide-react";
import { useCopilot } from "@/lib/copilot-store";
import { AmbientVoiceVisualizer } from "./AmbientVoiceVisualizer";

export function ChatInput() {
  const [value, setValue] = useState("");
  const [listening, setListening] = useState(false);
  const sendMessage = useCopilot((s) => s.sendMessage);
  const isLoading = useCopilot((s) => s.isLoading);
  const prefillInput = useCopilot((s) => s.prefillInput);
  const setPrefill = useCopilot((s) => s.setPrefill);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (prefillInput) {
      setValue(prefillInput);
      setPrefill(null);
      requestAnimationFrame(() => {
        const el = textareaRef.current;
        if (el) {
          el.focus();
          el.setSelectionRange(el.value.length, el.value.length);
        }
      });
    }
  }, [prefillInput, setPrefill]);

  const submit = async (e?: FormEvent) => {
    e?.preventDefault();
    if (!value.trim() || isLoading) return;
    const text = value;
    setValue("");
    await sendMessage(text);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void submit();
    }
  };

  return (
    <form
      onSubmit={submit}
      className="border-t border-gray-100 bg-transparent px-5 py-4"
    >
      <div className="flex items-end gap-2">
        {listening ? (
          <AmbientVoiceVisualizer
            onResult={(text) => {
              setListening(false);
              if (text) setValue((v) => (v ? `${v} ${text}` : text));
              requestAnimationFrame(() => textareaRef.current?.focus());
            }}
            onCancel={() => setListening(false)}
          />
        ) : (
          <>
            <textarea
              ref={textareaRef}
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder="Nhập câu hỏi của anh/chị…"
              className="flex-1 resize-none rounded-2xl border border-gray-200 bg-white px-4 py-2.5 text-[14px] leading-relaxed text-neutral-900 placeholder:text-neutral-400 focus:border-neutral-400 focus:outline-none"
              style={{ maxHeight: "120px" }}
              disabled={isLoading}
            />
            <button
              type="button"
              onClick={() => setListening(true)}
              disabled={isLoading}
              aria-label="Nói"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-gray-200 bg-white text-neutral-700 transition-colors hover:border-neutral-400 hover:text-neutral-900 disabled:opacity-30"
            >
              <Mic className="h-4 w-4" />
            </button>
            <button
              type="submit"
              disabled={!value.trim() || isLoading}
              aria-label="Gửi"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-neutral-900 text-white transition-opacity hover:bg-neutral-800 disabled:opacity-30"
            >
              <ArrowUp className="h-4 w-4" />
            </button>
          </>
        )}
      </div>
      <p className="mt-2 text-center text-[11px] text-neutral-400">
        Trả lời bởi trợ lý AI — chỉ mang tính tham khảo.
      </p>
    </form>
  );
}
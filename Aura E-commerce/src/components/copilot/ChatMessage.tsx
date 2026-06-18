import type { ChatMessage as ChatMessageType } from "@/lib/copilot-store";
import { InlineProductCard } from "./InlineProductCard";
import { EcosystemBundleCard } from "./EcosystemBundleCard";
import { MagicComparisonMatrix } from "./MagicComparisonMatrix";
import { InChatCheckoutSheet } from "./InChatCheckoutSheet";
import { GenerativeBentoSpecs } from "./GenerativeBentoSpecs";

export function ChatMessage({ message }: { message: ChatMessageType }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="flex max-w-[85%] flex-col items-end gap-2">
          {message.imageDataUrl ? (
            <img
              src={message.imageDataUrl}
              alt="Ảnh đã gửi"
              className="max-h-48 rounded-2xl border border-gray-100 object-cover"
            />
          ) : null}
          <div className="rounded-2xl bg-gray-100 px-4 py-2.5 text-[14px] leading-relaxed text-neutral-900">
            {message.text}
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="flex flex-col">
      <p className="text-[14px] leading-relaxed text-neutral-900">{message.text}</p>
      {message.product ? <InlineProductCard product={message.product} /> : null}
      {message.bundle ? <EcosystemBundleCard bundle={message.bundle} /> : null}
      {message.comparison ? <MagicComparisonMatrix comparison={message.comparison} /> : null}
      {message.checkout ? <InChatCheckoutSheet product={message.checkout} /> : null}
      {message.bento ? <GenerativeBentoSpecs product={message.bento} /> : null}
    </div>
  );
}

export function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 py-1">
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-neutral-400 [animation-delay:-0.3s]" />
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-neutral-400 [animation-delay:-0.15s]" />
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-neutral-400" />
    </div>
  );
}
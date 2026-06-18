import { useState, type FormEvent } from "react";
import { Check, Loader2 } from "lucide-react";
import type { Product } from "@/lib/products";

type Status = "idle" | "loading" | "done";

export function InChatCheckoutSheet({ product }: { product: Product }) {
  const [open, setOpen] = useState(true);
  const [phone, setPhone] = useState("");
  const [address, setAddress] = useState("");
  const [status, setStatus] = useState<Status>("idle");

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (status !== "idle") return;
    setStatus("loading");
    await new Promise((r) => setTimeout(r, 1400));
    setStatus("done");
  };

  return (
    <div className="copilot-fade-in mt-3 overflow-hidden rounded-2xl border border-neutral-100 bg-white">
      {/* Invoice summary */}
      <div className="flex items-center gap-3 border-b border-neutral-100 p-3">
        <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-lg bg-[#fbfbfd]">
          <img src={product.image} alt={product.name} className="h-11 w-11 object-contain" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-[13px] font-semibold text-neutral-900">{product.name}</p>
          <p className="text-[11px] font-light text-neutral-500">Đặt nhanh trong khung chat</p>
        </div>
        <p className="text-[13px] font-semibold text-neutral-900">{product.price}</p>
      </div>

      {!open ? (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="w-full px-4 py-2.5 text-[12px] font-medium text-neutral-700 hover:bg-neutral-50"
        >
          Mở phiếu đặt
        </button>
      ) : (
        <form onSubmit={submit} className="space-y-2.5 p-3">
          <MinimalInput
            value={phone}
            onChange={setPhone}
            placeholder="Số điện thoại"
            type="tel"
            disabled={status !== "idle"}
          />
          <MinimalInput
            value={address}
            onChange={setAddress}
            placeholder="Địa chỉ giao hàng"
            disabled={status !== "idle"}
          />
          <button
            type="submit"
            disabled={!phone || !address || status !== "idle"}
            className={`flex h-11 w-full items-center justify-center gap-2 rounded-full text-[13px] font-medium text-white transition-all duration-300 disabled:opacity-40 ${
              status === "done" ? "bg-emerald-600" : "bg-neutral-900 hover:bg-neutral-800"
            }`}
          >
            {status === "loading" ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : status === "done" ? (
              <>
                <Check className="h-4 w-4" />
                Đã đặt thành công
              </>
            ) : (
              "Xác nhận & Thanh toán"
            )}
          </button>
        </form>
      )}
    </div>
  );
}

function MinimalInput({
  value,
  onChange,
  placeholder,
  type = "text",
  disabled,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  type?: string;
  disabled?: boolean;
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      disabled={disabled}
      className="w-full rounded-lg border border-transparent bg-neutral-50 px-3 py-2.5 text-[13px] text-neutral-900 placeholder:text-neutral-400 transition-colors hover:border-neutral-200 focus:border-neutral-400 focus:bg-white focus:outline-none"
    />
  );
}
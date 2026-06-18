import { useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import type { Comparison, Product } from "@/lib/products";

type SpecRow = { key: keyof Product["specs"]; label: string };
const SPECS: SpecRow[] = [
  { key: "performance", label: "Hiệu năng" },
  { key: "battery", label: "Pin" },
  { key: "portability", label: "Tính di động" },
  { key: "camera", label: "Camera" },
];

export function MagicComparisonMatrix({ comparison }: { comparison: Comparison }) {
  const { a, b, recommendA, recommendB } = comparison;
  const [open, setOpen] = useState(true);
  const [bias, setBias] = useState(50); // 0 = full left, 100 = full right
  const dragging = useRef(false);
  const trackRef = useRef<HTMLDivElement>(null);

  const onDrag = (clientX: number) => {
    const el = trackRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    setBias(pct);
  };

  return (
    <div className="copilot-fade-in mt-3 overflow-hidden rounded-2xl border border-neutral-100 bg-white">
      {/* Header: 2 products + drag divider */}
      <div
        ref={trackRef}
        className="relative grid grid-cols-2 select-none"
        onMouseMove={(e) => dragging.current && onDrag(e.clientX)}
        onMouseUp={() => (dragging.current = false)}
        onMouseLeave={() => (dragging.current = false)}
        onTouchMove={(e) => onDrag(e.touches[0].clientX)}
      >
        <ProductHead product={a} highlight={bias < 50} side="left" />
        <ProductHead product={b} highlight={bias > 50} side="right" />
        {/* Vertical divider handle */}
        <div
          className="absolute inset-y-0 z-10 -translate-x-1/2 cursor-ew-resize"
          style={{ left: `${bias}%` }}
          onMouseDown={() => (dragging.current = true)}
          onTouchStart={() => (dragging.current = true)}
        >
          <div className="h-full w-px bg-neutral-900/60" />
          <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-neutral-200 bg-white px-2 py-1 text-[10px] font-medium text-neutral-700 shadow-sm">
            ⇆
          </div>
        </div>
      </div>
      {/* Recommendation bar */}
      <div className="px-4 pb-3">
        <p className="mb-1 text-[11px] font-light text-neutral-500">Tỷ lệ AI khuyên dùng</p>
        <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-neutral-100">
          <div
            className="h-full bg-neutral-900 transition-all duration-500"
            style={{ width: `${recommendA}%` }}
          />
          <div
            className="h-full bg-neutral-400 transition-all duration-500"
            style={{ width: `${recommendB}%` }}
          />
        </div>
        <div className="mt-1 flex justify-between text-[11px] text-neutral-500">
          <span>{recommendA}%</span>
          <span>{recommendB}%</span>
        </div>
      </div>
      {/* Accordion body */}
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between border-t border-neutral-100 px-4 py-2.5 text-[12px] font-medium text-neutral-700 hover:bg-neutral-50"
      >
        Thông số chi tiết
        <ChevronDown
          className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open ? (
        <div className="space-y-3 border-t border-neutral-100 px-4 py-3">
          {SPECS.map((row) => (
            <SpecBar
              key={row.key}
              label={row.label}
              left={a.specs[row.key]}
              right={b.specs[row.key]}
              bias={bias}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function ProductHead({
  product,
  highlight,
  side,
}: {
  product: Product;
  highlight: boolean;
  side: "left" | "right";
}) {
  return (
    <div
      className={`flex flex-col items-center gap-1 px-3 pb-2 pt-4 text-center transition-opacity ${
        highlight ? "opacity-100" : "opacity-60"
      } ${side === "left" ? "" : ""}`}
    >
      <div className="flex h-20 w-20 items-center justify-center rounded-xl bg-[#fbfbfd]">
        <img src={product.image} alt={product.name} className="h-16 w-16 object-contain" />
      </div>
      <p className="mt-1 line-clamp-1 text-[12px] font-semibold text-neutral-900">
        {product.name}
      </p>
      <p className="text-[11px] text-neutral-500">{product.price}</p>
    </div>
  );
}

function SpecBar({
  label,
  left,
  right,
  bias,
}: {
  label: string;
  left: number;
  right: number;
  bias: number;
}) {
  const leftActive = bias < 50;
  const rightActive = bias > 50;
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px] text-neutral-500">
        <span>{left}</span>
        <span className="font-medium text-neutral-700">{label}</span>
        <span>{right}</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="flex h-1 flex-1 flex-row-reverse overflow-hidden rounded-full bg-neutral-100">
          <div
            className={`h-full transition-all duration-500 ${
              leftActive ? "bg-neutral-900" : "bg-neutral-400"
            }`}
            style={{ width: `${left}%` }}
          />
        </div>
        <div className="flex h-1 flex-1 overflow-hidden rounded-full bg-neutral-100">
          <div
            className={`h-full transition-all duration-500 ${
              rightActive ? "bg-neutral-900" : "bg-neutral-400"
            }`}
            style={{ width: `${right}%` }}
          />
        </div>
      </div>
    </div>
  );
}
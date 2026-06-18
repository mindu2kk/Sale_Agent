import type { Bundle } from "@/lib/products";

export function EcosystemBundleCard({ bundle }: { bundle: Bundle }) {
  const items = bundle.items.slice(0, 3);
  return (
    <div className="copilot-fade-in group/bundle mt-3 flex items-center gap-4 rounded-2xl border border-neutral-100 bg-[#fbfbfd] p-3">
      <div className="relative flex h-16 w-[88px] shrink-0 items-center">
        {items.map((p, i) => (
          <div
            key={p.id}
            className="absolute top-1/2 flex h-14 w-14 -translate-y-1/2 items-center justify-center rounded-full border border-neutral-100 bg-white shadow-sm transition-all duration-300 ease-out"
            style={{
              left: `${i * 18}px`,
              zIndex: items.length - i,
              transform: `translateY(-50%) translateX(var(--spread-${i}, 0px))`,
            }}
          >
            <img
              src={p.image}
              alt={p.name}
              className="h-10 w-10 object-contain"
            />
          </div>
        ))}
        <style>{`
          .group\\/bundle:hover [style*="--spread-0"] { --spread-0: -6px; }
          .group\\/bundle:hover [style*="--spread-1"] { --spread-1: 0px; }
          .group\\/bundle:hover [style*="--spread-2"] { --spread-2: 6px; }
        `}</style>
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-[13px] font-semibold text-neutral-900">Hệ sinh thái tối ưu</p>
        <p className="mt-0.5 truncate text-[12px] font-light text-neutral-500">
          Tiết kiệm {bundle.savings} khi mua kèm
        </p>
      </div>
      <button
        type="button"
        className="shrink-0 rounded-full bg-neutral-900 px-3.5 py-2 text-[12px] font-medium text-white transition-colors hover:bg-neutral-800"
      >
        Mua trọn bộ
      </button>
    </div>
  );
}
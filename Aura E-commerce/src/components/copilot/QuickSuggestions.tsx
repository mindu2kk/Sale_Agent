import { useCopilot } from "@/lib/copilot-store";

const DEFAULT_SUGGESTIONS = [
  "Sản phẩm nào đang bán chạy nhất?",
  "Có chương trình trả góp 0% không?",
  "Cửa hàng có giao hàng tận nơi không?",
];

const CATEGORY_SUGGESTIONS: Record<string, string[]> = {
  Laptop: [
    "Tôi cần laptop làm việc văn phòng",
    "Laptop nào mạnh nhất để dựng video?",
    "So sánh MacBook Pro và Mac Studio",
  ],
  iPhone: [
    "iPhone nào chụp ảnh đẹp nhất?",
    "Giá iPhone 15 Pro bản 256GB?",
    "Có hỗ trợ thu cũ đổi mới không?",
  ],
  iPad: [
    "iPad nào hợp để học tập?",
    "iPad Pro dùng cho thiết kế đồ hoạ thế nào?",
    "Có bán kèm bút Apple Pencil không?",
  ],
  "Âm thanh": [
    "Tai nghe chống ồn tốt nhất hiện nay?",
    "AirPods Pro pin dùng được bao lâu?",
    "Có tai nghe nào dưới 5 triệu không?",
  ],
  "Đồng hồ": [
    "Apple Watch theo dõi sức khoẻ ra sao?",
    "Đồng hồ nào hợp chạy bộ?",
    "Có hỗ trợ eSIM không?",
  ],
  Mac: [
    "Mac Studio cấu hình nào phù hợp dựng 3D?",
    "Mac nào tiết kiệm điện nhất?",
    "Có chương trình ưu đãi cho sinh viên không?",
  ],
};

export function QuickSuggestions() {
  const sendMessage = useCopilot((s) => s.sendMessage);
  const isLoading = useCopilot((s) => s.isLoading);
  const activeCategory = useCopilot((s) => s.activeCategory);

  const suggestions =
    (activeCategory && CATEGORY_SUGGESTIONS[activeCategory]) || DEFAULT_SUGGESTIONS;
  const label = activeCategory ? `Gợi ý cho ${activeCategory}` : "Gợi ý nhanh";

  return (
    <div className="border-t border-gray-100 px-5 pt-3">
      <p className="mb-2 text-[11px] font-light uppercase tracking-wide text-neutral-400">
        {label}
      </p>
      <div key={activeCategory ?? "default"} className="flex flex-wrap gap-2 pb-1 copilot-fade-in">
        {suggestions.map((s) => (
          <button
            key={s}
            type="button"
            disabled={isLoading}
            onClick={() => void sendMessage(s)}
            className="rounded-full border border-gray-200 bg-white px-3 py-1.5 text-[12px] text-neutral-700 transition-colors hover:border-neutral-400 hover:text-neutral-900 disabled:opacity-40"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
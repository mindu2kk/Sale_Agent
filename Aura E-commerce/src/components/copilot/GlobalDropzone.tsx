import { useEffect, useState } from "react";
import { ImageIcon } from "lucide-react";
import { useCopilot } from "@/lib/copilot-store";

export function GlobalDropzone() {
  const [active, setActive] = useState(false);
  const sendImageQuery = useCopilot((s) => s.sendImageQuery);

  useEffect(() => {
    if (typeof window === "undefined") return;
    let depth = 0;
    const hasFiles = (e: DragEvent) =>
      Array.from(e.dataTransfer?.types ?? []).includes("Files");

    const onEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth++;
      setActive(true);
    };
    const onLeave = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) setActive(false);
    };
    const onOver = (e: DragEvent) => {
      if (hasFiles(e)) e.preventDefault();
    };
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      setActive(false);
      const file = e.dataTransfer?.files?.[0];
      if (!file || !file.type.startsWith("image/")) return;
      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = typeof reader.result === "string" ? reader.result : "";
        if (dataUrl) void sendImageQuery(dataUrl);
      };
      reader.readAsDataURL(file);
    };

    window.addEventListener("dragenter", onEnter);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("dragover", onOver);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onEnter);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("drop", onDrop);
    };
  }, [sendImageQuery]);

  if (!active) return null;

  return (
    <div className="copilot-fade-in fixed inset-0 z-[70] flex items-center justify-center bg-black/40 backdrop-blur-md">
      <div className="flex flex-col items-center gap-3 rounded-3xl border border-dashed border-white/60 bg-white/5 px-12 py-10 text-white shadow-[0_0_60px_rgba(255,255,255,0.15)]">
        <ImageIcon className="h-8 w-8 opacity-80" />
        <p className="text-[15px] font-medium">Thả ảnh vào đây để AI phân tích món đồ</p>
        <p className="text-[12px] font-light opacity-70">
          Hỗ trợ JPG, PNG — AI sẽ gợi ý sản phẩm tương ứng
        </p>
      </div>
    </div>
  );
}
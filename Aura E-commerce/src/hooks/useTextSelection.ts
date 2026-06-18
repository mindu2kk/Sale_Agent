import { useEffect, useState } from "react";

export type SelectionSnapshot = {
  text: string;
  x: number;
  y: number;
};

const MIN_LENGTH = 10;

function isInsideCopilot(node: Node | null): boolean {
  let el: Node | null = node;
  while (el && el instanceof Element === false) el = el.parentNode;
  let current = el as Element | null;
  while (current) {
    if (current.hasAttribute?.("data-copilot-root")) return true;
    current = current.parentElement;
  }
  return false;
}

export function useTextSelection(): SelectionSnapshot | null {
  const [selection, setSelection] = useState<SelectionSnapshot | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const update = () => {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || sel.rangeCount === 0) {
        setSelection(null);
        return;
      }
      const text = sel.toString().trim();
      if (text.length < MIN_LENGTH) {
        setSelection(null);
        return;
      }
      const range = sel.getRangeAt(0);
      if (isInsideCopilot(range.commonAncestorContainer)) {
        setSelection(null);
        return;
      }
      const rect = range.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) {
        setSelection(null);
        return;
      }
      setSelection({
        text,
        x: rect.left + rect.width / 2,
        y: rect.top,
      });
    };

    const onMouseUp = () => setTimeout(update, 0);
    const onMouseDown = (e: MouseEvent) => {
      const target = e.target as Node | null;
      if (isInsideCopilot(target)) return;
      setSelection(null);
    };
    const onScroll = () => setSelection(null);

    document.addEventListener("mouseup", onMouseUp);
    document.addEventListener("mousedown", onMouseDown);
    window.addEventListener("scroll", onScroll, true);
    return () => {
      document.removeEventListener("mouseup", onMouseUp);
      document.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("scroll", onScroll, true);
    };
  }, []);

  return selection;
}
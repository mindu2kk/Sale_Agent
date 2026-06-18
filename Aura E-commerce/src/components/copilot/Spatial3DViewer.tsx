import { Suspense, useEffect, useRef, useState } from "react";
import type { Product } from "@/lib/products";

/**
 * Lightweight "spatial" preview: CSS-based 3D rotation driven by drag.
 * Auto-rotates slowly when idle; user drag overrides. Avoids heavy WebGL deps.
 */
export function Spatial3DViewer({ product }: { product: Product }) {
  return (
    <Suspense fallback={<FallbackImage product={product} />}>
      <ViewerInner product={product} />
    </Suspense>
  );
}

function FallbackImage({ product }: { product: Product }) {
  return (
    <div className="copilot-fade-in mt-3 flex h-44 items-center justify-center rounded-3xl bg-gradient-to-b from-neutral-50 to-white">
      <img src={product.image} alt={product.name} className="h-32 w-32 object-contain" />
    </div>
  );
}

function ViewerInner({ product }: { product: Product }) {
  const [angle, setAngle] = useState(0);
  const [active, setActive] = useState(false);
  const dragging = useRef(false);
  const startX = useRef(0);
  const startAngle = useRef(0);

  // Auto-rotate when idle.
  useEffect(() => {
    if (active) return;
    let raf = 0;
    let last = performance.now();
    const tick = (t: number) => {
      const dt = t - last;
      last = t;
      setAngle((a) => a + dt * 0.025);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active]);

  const onPointerDown = (e: React.PointerEvent) => {
    dragging.current = true;
    setActive(true);
    startX.current = e.clientX;
    startAngle.current = angle;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragging.current) return;
    const delta = e.clientX - startX.current;
    setAngle(startAngle.current + delta * 0.6);
  };
  const onPointerUp = () => {
    dragging.current = false;
    // Resume auto-rotate after 2s of inactivity.
    window.setTimeout(() => setActive(false), 2000);
  };

  return (
    <div
      className="copilot-fade-in relative mt-3 flex h-48 cursor-grab items-center justify-center overflow-hidden rounded-3xl bg-gradient-to-b from-neutral-50/80 to-white active:cursor-grabbing"
      style={{ perspective: "900px" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      {/* Floor shadow */}
      <div className="absolute bottom-5 left-1/2 h-3 w-32 -translate-x-1/2 rounded-full bg-black/15 blur-md" />
      <img
        src={product.image}
        alt={product.name}
        draggable={false}
        className="h-36 w-36 select-none object-contain transition-transform duration-100"
        style={{
          transform: `rotateY(${angle}deg)`,
          transformStyle: "preserve-3d",
          filter: "drop-shadow(0 14px 22px rgba(0,0,0,0.18))",
        }}
      />
      <p className="pointer-events-none absolute bottom-2 right-3 text-[10px] font-light text-neutral-400">
        Kéo để xoay 360°
      </p>
    </div>
  );
}
import { motion } from "framer-motion";
import { Camera, Cpu, Feather, Sparkles, Zap } from "lucide-react";
import type { Product } from "@/lib/products";

type Cell = {
  area: string;
  title: string;
  value: string;
  icon: React.ReactNode;
  variant?: "image" | "fluid" | "pulse" | "default";
  fill?: number; // 0-100 for fluid
  image?: string;
};

export function GenerativeBentoSpecs({ product }: { product: Product }) {
  const cells: Cell[] = [
    {
      area: "col-span-2 row-span-2",
      title: "Camera",
      value: `${Math.max(12, product.specs.camera * 2)}MP`,
      icon: <Camera className="h-4 w-4" />,
      variant: "image",
      image: product.image,
    },
    {
      area: "col-span-2 row-span-1",
      title: "Pin",
      value: `${1200 + product.specs.battery * 40} mAh`,
      icon: <Zap className="h-4 w-4" />,
      variant: "fluid",
      fill: product.specs.battery,
    },
    {
      area: "col-span-1 row-span-1",
      title: "Hiệu năng",
      value: `${product.specs.performance}`,
      icon: <Cpu className="h-4 w-4" />,
    },
    {
      area: "col-span-1 row-span-1",
      title: "Titanium",
      value: "Khung siêu nhẹ",
      icon: <Feather className="h-4 w-4" />,
      variant: "pulse",
    },
  ];

  return (
    <motion.div
      initial="hidden"
      animate="show"
      variants={{ show: { transition: { staggerChildren: 0.1 } } }}
      className="mt-3 rounded-3xl border border-gray-200/50 bg-[#fbfbfd] p-3"
    >
      <div className="mb-2 flex items-center gap-1.5 px-1">
        <Sparkles className="h-3 w-3 text-neutral-400" />
        <p className="text-[11px] font-light text-neutral-500">
          Điểm nổi bật · {product.name}
        </p>
      </div>
      <div className="grid grid-cols-4 grid-rows-[repeat(2,minmax(72px,1fr))] gap-2">
        {cells.map((c, i) => (
          <BentoCell key={i} cell={c} />
        ))}
      </div>
    </motion.div>
  );
}

function BentoCell({ cell }: { cell: Cell }) {
  return (
    <motion.div
      variants={{
        hidden: { opacity: 0, y: 12, scale: 0.95 },
        show: { opacity: 1, y: 0, scale: 1 },
      }}
      transition={{ type: "spring", stiffness: 280, damping: 24 }}
      whileHover={{ scale: 1.02 }}
      className={`group/cell relative overflow-hidden rounded-2xl border border-gray-200/60 bg-white p-3 ${cell.area}`}
    >
      {/* Glass sweep reflection on hover */}
      <span className="pointer-events-none absolute inset-y-0 -left-1/2 w-1/2 -skew-x-12 bg-gradient-to-r from-transparent via-white/70 to-transparent opacity-0 transition-all duration-700 group-hover/cell:left-[120%] group-hover/cell:opacity-100" />

      <div className="flex items-center gap-1.5 text-neutral-500">
        {cell.icon}
        <span className="text-[10.5px] font-medium uppercase tracking-wide">
          {cell.title}
        </span>
      </div>

      {cell.variant === "image" && cell.image ? (
        <div className="mt-2 flex h-full flex-col">
          <div className="flex flex-1 items-center justify-center">
            <img src={cell.image} alt="" className="h-20 w-20 object-contain" />
          </div>
          <p className="mt-1 text-[26px] font-bold leading-none tracking-tight text-neutral-900">
            {cell.value}
          </p>
        </div>
      ) : cell.variant === "fluid" ? (
        <div className="relative mt-2 flex h-full flex-col justify-end">
          <p className="relative z-10 text-[20px] font-bold leading-none tracking-tight text-neutral-900">
            {cell.value}
          </p>
          <FluidFill fill={cell.fill ?? 50} />
        </div>
      ) : cell.variant === "pulse" ? (
        <div className="mt-2 flex items-end gap-1.5">
          <motion.span
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="h-2 w-2 rounded-full bg-amber-400"
          />
          <p className="text-[12px] font-semibold text-neutral-900">{cell.value}</p>
        </div>
      ) : (
        <p className="mt-2 text-[22px] font-bold leading-none tracking-tight text-neutral-900">
          {cell.value}
        </p>
      )}
    </motion.div>
  );
}

function FluidFill({ fill }: { fill: number }) {
  return (
    <div className="absolute inset-x-0 bottom-0 overflow-hidden rounded-b-2xl">
      <motion.div
        initial={{ y: "100%" }}
        animate={{ y: `${100 - fill}%` }}
        transition={{ type: "spring", stiffness: 80, damping: 18, delay: 0.2 }}
        className="h-[60px] bg-gradient-to-t from-blue-500/70 via-blue-400/50 to-blue-300/30"
      >
        <svg
          viewBox="0 0 120 12"
          preserveAspectRatio="none"
          className="absolute -top-2 left-0 h-3 w-full text-blue-400/60"
        >
          <motion.path
            animate={{ d: ["M0 6 Q30 0 60 6 T120 6 V12 H0Z", "M0 6 Q30 12 60 6 T120 6 V12 H0Z", "M0 6 Q30 0 60 6 T120 6 V12 H0Z"] }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
            fill="currentColor"
          />
        </svg>
      </motion.div>
    </div>
  );
}
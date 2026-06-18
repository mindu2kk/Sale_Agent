import { Link } from "@tanstack/react-router";
import { ShoppingBag, Search } from "lucide-react";

const nav = ["Mac", "iPad", "iPhone", "Watch", "AirPods", "Phụ kiện"];

export function Header() {
  return (
    <header className="sticky top-0 z-40 border-b border-gray-100 bg-white/70 backdrop-blur-md">
      <div className="mx-auto flex h-12 max-w-6xl items-center justify-between px-6">
        <Link to="/" className="text-sm font-semibold tracking-tight text-neutral-900">
          Tinh.
        </Link>
        <nav className="hidden items-center gap-8 md:flex">
          {nav.map((item) => (
            <a
              key={item}
              href="#"
              className="text-[13px] font-normal text-neutral-700 transition-colors hover:text-neutral-900"
            >
              {item}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-2">
          <button
            aria-label="Tìm kiếm"
            className="hidden h-9 w-9 items-center justify-center rounded-full text-neutral-700 transition-colors hover:bg-neutral-100 md:flex"
          >
            <Search className="h-4 w-4" />
          </button>
          <button
            aria-label="Giỏ hàng"
            className="flex h-9 items-center gap-2 rounded-full bg-neutral-900 px-4 text-[13px] font-medium text-white transition-colors hover:bg-neutral-800"
          >
            <ShoppingBag className="h-3.5 w-3.5" />
            <span>Giỏ hàng</span>
          </button>
        </div>
      </div>
    </header>
  );
}
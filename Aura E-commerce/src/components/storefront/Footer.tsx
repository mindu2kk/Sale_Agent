export function Footer() {
  return (
    <footer className="border-t border-gray-100 bg-white">
      <div className="mx-auto max-w-6xl px-6 py-12">
        <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
          {[
            { title: "Mua sắm", items: ["Mac", "iPad", "iPhone", "Watch"] },
            { title: "Dịch vụ", items: ["Bảo hành", "Giao hàng", "Trả góp"] },
            { title: "Về chúng tôi", items: ["Câu chuyện", "Cửa hàng", "Liên hệ"] },
            { title: "Hỗ trợ", items: ["FAQ", "Chính sách", "Đổi trả"] },
          ].map((col) => (
            <div key={col.title}>
              <h4 className="text-[12px] font-semibold uppercase tracking-[0.15em] text-neutral-900">
                {col.title}
              </h4>
              <ul className="mt-4 space-y-3">
                {col.items.map((i) => (
                  <li key={i}>
                    <a href="#" className="text-[13px] font-light text-neutral-600 hover:text-neutral-900">
                      {i}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <div className="mt-12 border-t border-gray-100 pt-6 text-[12px] font-light text-neutral-500">
          © {new Date().getFullYear()} Tinh. Mọi quyền được bảo lưu.
        </div>
      </div>
    </footer>
  );
}
export function Footer() {
  return (
    <footer className="mt-20 border-t border-border bg-surface">
      <div className="mx-auto flex max-w-7xl flex-col gap-8 px-4 py-12 sm:flex-row sm:items-end sm:justify-between sm:px-6 lg:px-8">
        <div>
          <div className="mb-2 text-xl font-bold tracking-tight text-text-primary">AURA</div>
          <p className="text-sm text-text-secondary">Mua sắm công nghệ cao cấp, tin cậy.</p>
        </div>
        <div className="flex gap-6 text-sm font-medium text-text-secondary">
          <a href="#" className="hover:text-primary">Giao hàng</a>
          <a href="#" className="hover:text-primary">Bảo hành</a>
          <a href="#" className="hover:text-primary">Quyền riêng tư</a>
        </div>
      </div>
    </footer>
  )
}

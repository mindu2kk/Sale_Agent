import { FolderSearch } from 'lucide-react'

interface EmptyStateProps {
  title?: string
  message?: string
  actionLabel?: string
  onAction?: () => void
}

export function EmptyState({
  title = 'Chưa tìm thấy sản phẩm phù hợp',
  message = 'Hãy thử thay đổi bộ lọc hoặc từ khóa tìm kiếm để khám phá thêm.',
  actionLabel = 'Xóa bộ lọc',
  onAction,
}: EmptyStateProps) {
  return (
    <div className="flex w-full flex-col items-center justify-center rounded-2xl border border-dashed border-border py-16 text-center">
      <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-surface-subtle text-text-muted">
        <FolderSearch size={32} />
      </div>
      <h3 className="mb-2 text-lg font-semibold text-text-primary">{title}</h3>
      <p className="mb-6 max-w-md text-sm text-text-secondary">{message}</p>
      {onAction && (
        <button className="btn-secondary" onClick={onAction}>
          {actionLabel}
        </button>
      )}
    </div>
  )
}

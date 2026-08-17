export function ProductSkeleton() {
  return (
    <div className="product-card animate-pulse" aria-hidden="true">
      {/* Image zone — 4:3 ratio matching product-card-img-zone */}
      <div
        style={{
          aspectRatio: '4 / 3',
          borderRadius: '20px 20px 0 0',
          backgroundColor: '#EEF2F8',
        }}
      />

      {/* Card body */}
      <div style={{ padding: '14px 16px 16px' }} className="flex flex-col gap-2">
        {/* Category */}
        <div className="flex items-center">
          <div className="h-5 w-16 rounded-full" style={{ backgroundColor: '#EEF2F8' }} />
        </div>

        {/* Brand */}
        <div className="h-3 w-24 rounded" style={{ backgroundColor: '#EEF2F8' }} />

        {/* Name — 2 lines */}
        <div className="h-4 w-full rounded" style={{ backgroundColor: '#EEF2F8' }} />
        <div className="h-4 w-4/5 rounded" style={{ backgroundColor: '#EEF2F8' }} />

        {/* Spec chips */}
        <div className="flex gap-1.5">
          <div className="h-5 w-14 rounded-md" style={{ backgroundColor: '#EEF2F8' }} />
          <div className="h-5 w-16 rounded-md" style={{ backgroundColor: '#EEF2F8' }} />
          <div className="h-5 w-12 rounded-md" style={{ backgroundColor: '#EEF2F8' }} />
        </div>

        {/* Price */}
        <div className="h-7 w-32 rounded" style={{ backgroundColor: '#EEF2F8', marginTop: '4px' }} />

        {/* CTA */}
        <div className="flex items-center gap-2" style={{ marginTop: '4px' }}>
          <div className="h-9 flex-1 rounded-full" style={{ backgroundColor: '#EEF2F8' }} />
          <div className="h-9 flex-1 rounded-full" style={{ backgroundColor: '#FFF1F2' }} />
        </div>
      </div>
    </div>
  )
}

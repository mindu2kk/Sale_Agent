import { motion } from 'framer-motion'
import { ArrowUpRight, ShieldCheck, Sparkles } from 'lucide-react'
import { useCatalogStore } from '@/stores/catalogStore'

export function Editorial() {
  const product = useCatalogStore((state) =>
    state.products.find((item) => item.category === 'Laptop'),
  )

  if (!product) return null

  return (
    <section className="editorial-section">
      <div className="mx-auto grid max-w-[1320px] gap-12 px-5 py-24 lg:grid-cols-[1.05fr_0.95fr] lg:px-8 lg:py-32">
        <motion.div
          className="editorial-image"
          initial={{ opacity: 0, x: -24 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
        >
          <img src={product.image_url} alt={product.name} />
          <div className="editorial-floating-card">
            <Sparkles size={15} />
            <span>Chọn một đoạn bất kỳ</span>
            <ArrowUpRight size={14} />
          </div>
        </motion.div>
        <div className="flex flex-col justify-center">
          <p className="section-index">02 / Hỏi ngay tại chỗ</p>
          <h2 className="editorial-title">Đừng tự giải mã<br /><em>một rừng thông số.</em></h2>
          <p className="editorial-copy">
            Bôi đen thông số, chính sách hay mô tả đang đọc. Trợ lý sẽ tiếp tục từ đúng ngữ cảnh đó,
            không bắt bạn kể lại từ đầu.
          </p>
          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            <div className="editorial-note">
              <Sparkles size={18} />
              <div>
                <strong>Hỏi đúng đoạn</strong>
                <span>Tooltip nhỏ gọn xuất hiện ngay trên vùng text bạn vừa bôi đen.</span>
              </div>
            </div>
            <div className="editorial-note">
              <ShieldCheck size={18} />
              <div>
                <strong>Trả lời gọn rõ</strong>
                <span>Giá, cấu hình và gợi ý mua sắm được trả về theo ngữ cảnh đang xem.</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

import heroDevice from "@/assets/hero-device.png";

export function Hero() {
  return (
    <section className="bg-[#f5f5f7]">
      <div className="mx-auto flex min-h-[90vh] max-w-6xl flex-col items-center justify-start px-6 pt-20 pb-10 text-center">
        <p className="text-[13px] font-medium uppercase tracking-[0.2em] text-neutral-500">
          Bộ sưu tập mới
        </p>
        <h1 className="mt-4 max-w-3xl text-5xl font-light leading-[1.05] tracking-tight text-neutral-900 md:text-6xl lg:text-7xl">
          Tương lai.
          <br />
          Trong tầm tay.
        </h1>
        <p className="mt-5 max-w-md text-base font-light text-neutral-600">
          Khám phá những thiết kế được chế tác từ vật liệu cao cấp và công nghệ tinh tế nhất.
        </p>
        <div className="mt-6 flex items-center gap-6 text-[13px]">
          <a href="#products" className="font-medium text-neutral-900 underline-offset-4 hover:underline">
            Tìm hiểu thêm
          </a>
          <span className="text-neutral-300">·</span>
          <a href="#products" className="font-medium text-neutral-900 underline-offset-4 hover:underline">
            Mua ngay
          </a>
        </div>
        <div className="mt-4 w-full max-w-4xl">
          <img
            src={heroDevice}
            alt="Thiết bị nổi bật"
            width={1600}
            height={1024}
            className="mx-auto h-[460px] w-full object-contain md:h-[520px]"
          />
        </div>
      </div>
    </section>
  );
}
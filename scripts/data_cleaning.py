import pandas as pd
import re
import os
import glob

def clean_real_csv_data():
    """Nhiệm vụ 2.2: Tiền xử lý, gộp cột và làm sạch CSV theo đúng schema thực tế"""
    print("⏳ Đang làm sạch dataset thực tế...")
    
    # Đọc file CSV thực tế của bạn (Nhớ đổi tên file cho khớp nhé)
    raw_csv_path = 'data/product_catalog_raw.csv' 
    
    if not os.path.exists(raw_csv_path):
        print(f"⚠️ Không tìm thấy {raw_csv_path}. Vui lòng chép file vào thư mục data.")
        return

    # Đọc dữ liệu
    df = pd.read_csv(raw_csv_path, encoding='utf-8')
    
    # 1. BẢO MẬT: Xóa bỏ các cột chứa dữ liệu cá nhân (PII) và cột không cần thiết cho Catalog
    columns_to_drop = ['Customer Name', 'Customer Location', 'Quantity Sold', 'Inward Date', 'Dispatch Date']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # 2. XỬ LÝ NHIỄU: Thay thế chuỗi 'N/A' thành chuỗi rỗng để dễ gộp văn bản
    df = df.replace('N/A', '')
    
    # 3. CHUẨN HÓA GIÁ: Ép kiểu về số, nhân 100, format dấu chấm và thêm VNĐ
    # Chuyển cột Price về dạng số (đề phòng có khoảng trắng hoặc lỗi parse)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    # Nhân 100 và format (VD: 78570 -> 7857000 -> 7,857,000 -> 7.857.000 VNĐ)
    df['Price'] = df['Price'].apply(lambda x: f"{int(x) * 100:,} VNĐ".replace(',', '.') if pd.notnull(x) else "Liên hệ")

    # 4. DATA SERIALIZATION: Gộp thông số thành câu văn tự nhiên
    def build_product_description(row):
        # Thông tin cơ bản
        desc = f"Sản phẩm {row['Product']} thương hiệu {row['Brand']}, mã sản phẩm {row['Product Code']}. "
        desc += f"Giá bán hiện tại là {row['Price']}. "
        
        # Ghép thông số kỹ thuật
        specs = []
        if row['Core Specification']: specs.append(f"Core {row['Core Specification']}")
        if row['Processor Specification']: specs.append(f"Chip {row['Processor Specification']}")
        if row['RAM']: specs.append(f"RAM {row['RAM']}")
        if row['ROM']: specs.append(f"Bộ nhớ trong {row['ROM']}")
        if row['SSD']: specs.append(f"Ổ cứng SSD {row['SSD']}")
        
        if specs:
            desc += "Cấu hình kỹ thuật bao gồm: " + ", ".join(specs) + "."
            
        return desc

    # Tạo cột LLM_Context
    df['LLM_Context'] = df.apply(build_product_description, axis=1)
    
    # Giữ lại các cột quan trọng nhất và xuất file
    final_df = df[['Product Code', 'Product', 'Brand', 'Price', 'LLM_Context']]
    
    clean_csv_path = 'data/product_catalog_clean.csv'
    final_df.to_csv(clean_csv_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ Đã chuẩn hóa CSV. Xem cột 'LLM_Context' tại: {clean_csv_path}")


def process_and_anonymize_scripts():
    """Nhiệm vụ 2.3: Ẩn danh hóa PII trong các file Kịch bản Sales (TXT)"""
    print("\n⏳ Đang quét và ẩn danh hóa dữ liệu TXT/PDF (PII Masking)...")
    
    script_files = glob.glob('data/Sales_Scripts/*.txt')
    if not script_files:
        print("⚠️ Không tìm thấy file .txt nào trong data/Sales_Scripts/ để quét ẩn danh.")
        return
        
    for file_path in script_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Regex che PII
        content = re.sub(r'(0[3|5|7|8|9])+([0-9]{8})\b', '[PHONE_REDACTED]', content)
        content = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[EMAIL_REDACTED]', content)
        content = re.sub(r'\b\d{9,14}\b', '[BANK_ACCOUNT_REDACTED]', content)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
            
        print(f"   🔒 Đã che dữ liệu nhạy cảm trong: {os.path.basename(file_path)}")
        
    print("✅ Hoàn tất bảo mật dữ liệu văn bản phi cấu trúc.")


if __name__ == "__main__":
    print("=== KHỞI CHẠY BATCH 2: DATA CLEANING & SERIALIZATION ===")
    clean_real_csv_data()
    process_and_anonymize_scripts()
    print("=== HOÀN TẤT BATCH 2 ===")
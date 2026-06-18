export interface Product {
  id: string
  name: string
  brand: string
  category: 'Mobile Phone' | 'Laptop' | 'Tablet' | 'Headphones' | 'Accessories'
  price: string
  priceVnd: number
  image?: string
  specs?: {
    chip?: string
    ram?: string
    storage?: string
    display?: string
    battery?: string
    [key: string]: any
  }
}

export interface BundleItem {
  product: Product
  discount?: number
  reason?: string
}

export interface EcosystemBundle {
  mainProduct: Product
  accessories: BundleItem[]
  totalOriginalPrice: number
  totalBundlePrice: number
  savings: number
  savingsPercent: number
  theme: 'mobile' | 'productivity' | 'gaming' | 'creative'
}

// Sample data for demo - in real app this comes from backend
export const SAMPLE_PRODUCTS: Product[] = [
  {
    id: 'iphone-15-pro',
    name: 'iPhone 15 Pro',
    brand: 'Apple',
    category: 'Mobile Phone',
    price: '29.990.000 VNĐ',
    priceVnd: 29990000,
    image: 'https://source.unsplash.com/400x400/?iphone,15,pro',
    specs: {
      chip: 'A17 Pro',
      ram: '8GB',
      storage: '128GB',
      display: '6.1" Super Retina XDR'
    }
  },
  {
    id: 'airpods-pro',
    name: 'AirPods Pro (2nd gen)',
    brand: 'Apple',
    category: 'Headphones',
    price: '6.190.000 VNĐ',
    priceVnd: 6190000,
    image: 'https://source.unsplash.com/400x400/?airpods,pro',
  },
  {
    id: 'magsafe-charger',
    name: 'MagSafe Charger',
    brand: 'Apple',
    category: 'Accessories',
    price: '1.290.000 VNĐ',
    priceVnd: 1290000,
    image: 'https://source.unsplash.com/400x400/?magsafe,charger',
  },
  {
    id: 'clear-case',
    name: 'iPhone 15 Pro Clear Case',
    brand: 'Apple',
    category: 'Accessories',
    price: '1.490.000 VNĐ',
    priceVnd: 1490000,
    image: 'https://source.unsplash.com/400x400/?iphone,case,clear',
  }
]

export function generateEcosystemBundle(mainProduct: Product): EcosystemBundle {
  // Simple bundle logic - in real app this comes from backend
  const accessories: BundleItem[] = []
  
  if (mainProduct.category === 'Mobile Phone') {
    // Mobile ecosystem
    if (mainProduct.brand === 'Apple') {
      accessories.push(
        {
          product: SAMPLE_PRODUCTS.find(p => p.id === 'airpods-pro')!,
          discount: 10,
          reason: 'Hoàn hảo với iPhone'
        },
        {
          product: SAMPLE_PRODUCTS.find(p => p.id === 'magsafe-charger')!,
          discount: 15,
          reason: 'Sạc không dây tiện lợi'
        },
        {
          product: SAMPLE_PRODUCTS.find(p => p.id === 'clear-case')!,
          discount: 20,
          reason: 'Bảo vệ tối ưu'
        }
      )
    }
  }
  
  const totalOriginal = mainProduct.priceVnd + accessories.reduce((sum, item) => sum + item.product.priceVnd, 0)
  const totalBundle = mainProduct.priceVnd + accessories.reduce((sum, item) => {
    const discountAmount = item.product.priceVnd * (item.discount || 0) / 100
    return sum + item.product.priceVnd - discountAmount
  }, 0)
  
  return {
    mainProduct,
    accessories,
    totalOriginalPrice: totalOriginal,
    totalBundlePrice: totalBundle,
    savings: totalOriginal - totalBundle,
    savingsPercent: Math.round(((totalOriginal - totalBundle) / totalOriginal) * 100),
    theme: 'mobile'
  }
}
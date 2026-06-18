import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { EcosystemBundleCard } from "./EcosystemBundleCard"
import { getSampleProducts, generateEcosystemBundles } from "@/lib/product-adapter"
import { getProducts } from "@/lib/csv-loader"
import { useState, useEffect } from "react"
import { EcosystemBundle } from "@/types/product"
import { RefreshCw, Sparkles } from "lucide-react"

export function DemoContent() {
  const [bundles, setBundles] = useState<EcosystemBundle[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [currentBundleIndex, setCurrentBundleIndex] = useState(0)
  
  // Load real product data and generate bundles
  useEffect(() => {
    loadBundles()
  }, [])
  
  const loadBundles = async () => {
    setIsLoading(true)
    try {
      const products = await getProducts()
      const generatedBundles = generateEcosystemBundles(products)
      
      if (generatedBundles.length > 0) {
        setBundles(generatedBundles)
      } else {
        // Fallback to sample data
        const sampleProducts = getSampleProducts()
        const sampleBundles = generateEcosystemBundles(sampleProducts)
        setBundles(sampleBundles)
      }
    } catch (error) {
      console.error('Error loading bundles:', error)
      // Fallback to sample data
      const sampleProducts = getSampleProducts()
      const sampleBundles = generateEcosystemBundles(sampleProducts)
      setBundles(sampleBundles)
    } finally {
      setIsLoading(false)
    }
  }
  
  const nextBundle = () => {
    setCurrentBundleIndex((prev) => (prev + 1) % bundles.length)
  }
  
  const currentBundle = bundles[currentBundleIndex]
  
  return (
    <div className="fixed bottom-4 left-4 z-40 space-y-4">
      {/* Feature 1: Contextual Highlight */}
      <Card className="w-96 bg-card/80 backdrop-blur">
        <CardHeader>
          <CardTitle>Demo - Contextual Highlight</CardTitle>
          <CardDescription>
            Bôi đen đoạn text bất kỳ dưới đây để test tính năng
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <p>
            <strong>iPhone 15 Pro Max</strong> được trang bị chip Apple A17 Pro mạnh mẽ, 
            hỗ trợ quay video 4K ProRes và có khả năng zoom optical 5x. 
            Giá bán chính thức tại Việt Nam là 34.990.000 VNĐ cho bản 256GB.
          </p>
          
          <p>
            <strong>Samsung Galaxy S24 Ultra</strong> sử dụng chip Snapdragon 8 Gen 3, 
            có bút S Pen tích hợp và camera chính 200MP. Sản phẩm có mức giá 
            khởi điểm 31.490.000 VNĐ cho phiên bản 256GB.
          </p>
          
          <p>
            <strong>Chính sách bảo hành:</strong> Tất cả sản phẩm Apple được bảo hành 
            12 tháng tại các trung tâm chính thức. Samsung cung cấp bảo hành 
            24 tháng cho dòng Galaxy S series cao cấp.
          </p>
          
          <div className="text-xs text-muted-foreground mt-4 p-2 bg-muted rounded">
            💡 <strong>Hướng dẫn:</strong> Bôi đen bất kỳ đoạn text nào ở trên 
            (ít nhất 10 ký tự) để thấy tooltip "Giải thích đoạn này" xuất hiện!
          </div>
        </CardContent>
      </Card>
      
      {/* Feature 2: Ecosystem Bundle */}
      <div className="w-80">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-purple-500" />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Feature 2: Ecosystem Bundle
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              onClick={nextBundle}
              disabled={isLoading || bundles.length <= 1}
              variant="outline"
              size="sm"
              className="text-xs"
            >
              <RefreshCw className="h-3 w-3 mr-1" />
              Đổi Bundle ({currentBundleIndex + 1}/{bundles.length})
            </Button>
          </div>
        </div>
        
        {isLoading ? (
          <Card className="p-6">
            <div className="flex items-center justify-center space-x-2">
              <RefreshCw className="h-4 w-4 animate-spin" />
              <span className="text-sm text-muted-foreground">
                Đang tạo bundle từ data thật...
              </span>
            </div>
          </Card>
        ) : currentBundle ? (
          <EcosystemBundleCard bundle={currentBundle} />
        ) : (
          <Card className="p-6">
            <div className="text-center text-sm text-muted-foreground">
              Không tìm thấy bundle phù hợp
            </div>
          </Card>
        )}
        
        {bundles.length > 0 && (
          <div className="text-xs text-center text-muted-foreground mt-2">
            📦 Đã tạo {bundles.length} bundle từ {bundles.reduce((sum, b) => sum + b.accessories.length, 0)} sản phẩm thật
          </div>
        )}
      </div>
    </div>
  )
}
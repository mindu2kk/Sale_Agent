import { motion, AnimatePresence } from "framer-motion";
import { ShoppingCart, Sparkles, Gift, ArrowRight } from "lucide-react";
import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { EcosystemBundle, BundleItem } from "@/types/product";
import { useCopilot } from "@/lib/copilot-store";

interface EcosystemBundleCardProps {
  bundle: EcosystemBundle;
  className?: string;
}

export function EcosystemBundleCard({ bundle, className }: EcosystemBundleCardProps) {
  const [isHovered, setIsHovered] = useState(false);
  const { addMessage } = useCopilot();

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('vi-VN').format(price) + ' VNĐ';
  };

  const handleAddToChat = () => {
    const bundleText = `Bundle Ecosystem: ${bundle.mainProduct.name} + ${bundle.accessories.length} phụ kiện với giá ${formatPrice(bundle.totalBundlePrice)} (tiết kiệm ${bundle.savingsPercent}%)`;
    
    addMessage({
      role: 'assistant',
      content: `🎁 **Gợi ý Bundle Ecosystem cho bạn:**

**${bundle.mainProduct.name}** + ${bundle.accessories.length} phụ kiện

${bundle.accessories.map(item => 
  `• ${item.product.name} - ${formatPrice(item.product.priceVnd)} (-${item.discount}%) *${item.reason}*`
).join('\n')}

**Tổng tiền gốc:** ${formatPrice(bundle.totalOriginalPrice)}
**Giá Bundle:** ${formatPrice(bundle.totalBundlePrice)}
**Tiết kiệm:** ${formatPrice(bundle.savings)} (${bundle.savingsPercent}%)

Bạn có muốn tôi giải thích chi tiết về bundle này không?`,
      ui_type: 'bento',
      bundle: [bundle.mainProduct, ...bundle.accessories.map(a => a.product)]
    });
  };

  return (
    <motion.div
      className={className}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
    >
      <Card className="overflow-hidden bg-gradient-to-br from-purple-50 to-blue-50 border-purple-200 dark:from-purple-950/20 dark:to-blue-950/20 dark:border-purple-800">
        <CardContent className="p-6">
          {/* Header */}
          <div className="flex items-center gap-3 mb-4">
            <div className="flex items-center gap-2">
              <Gift className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              <h3 className="font-semibold text-gray-900 dark:text-white">
                Ecosystem Bundle
              </h3>
            </div>
            <div className="flex items-center gap-1 px-2 py-1 bg-green-100 dark:bg-green-900/30 rounded-full">
              <Sparkles className="h-3 w-3 text-green-600 dark:text-green-400" />
              <span className="text-xs font-medium text-green-700 dark:text-green-300">
                Tiết kiệm {bundle.savingsPercent}%
              </span>
            </div>
          </div>

          {/* Products Stack */}
          <div className="relative h-32 mb-4 flex items-center justify-center">
            {/* Main Product - Always in center */}
            <motion.div
              className="absolute z-10 w-24 h-24 rounded-xl bg-white dark:bg-gray-800 shadow-lg border-2 border-purple-200 dark:border-purple-600 overflow-hidden"
              animate={{
                scale: isHovered ? 1.1 : 1,
                rotate: isHovered ? -5 : 0,
              }}
              transition={{ type: "spring", stiffness: 300, damping: 20 }}
            >
              <img
                src={bundle.mainProduct.image || `https://images.unsplash.com/400x400/?${encodeURIComponent(bundle.mainProduct.name)}`}
                alt={bundle.mainProduct.name}
                className="w-full h-full object-cover"
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.src = `https://images.unsplash.com/400x400/?smartphone,${bundle.mainProduct.brand.toLowerCase()}`;
                }}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-purple-900/20 to-transparent" />
            </motion.div>

            {/* Accessories - Spread on hover */}
            {bundle.accessories.map((item, index) => {
              const angle = (index * (360 / bundle.accessories.length)) * (Math.PI / 180);
              const radius = isHovered ? 50 : 20;
              const x = Math.cos(angle) * radius;
              const y = Math.sin(angle) * radius;

              return (
                <motion.div
                  key={item.product.id}
                  className="absolute w-16 h-16 rounded-lg bg-white dark:bg-gray-800 shadow-md border border-gray-200 dark:border-gray-700 overflow-hidden"
                  animate={{
                    x: isHovered ? x : 0,
                    y: isHovered ? y : 0,
                    scale: isHovered ? 1 : 0.7,
                    rotate: isHovered ? index * 15 - 30 : 0,
                    opacity: isHovered ? 1 : 0.8,
                  }}
                  transition={{ 
                    type: "spring", 
                    stiffness: 300, 
                    damping: 20,
                    delay: isHovered ? index * 0.05 : 0 
                  }}
                  style={{ zIndex: 5 - index }}
                >
                  <img
                    src={item.product.image || `https://images.unsplash.com/300x300/?${encodeURIComponent(item.product.name)}`}
                    alt={item.product.name}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      target.src = `https://images.unsplash.com/300x300/?accessory,${item.product.category.toLowerCase()}`;
                    }}
                  />
                  {item.discount && (
                    <div className="absolute -top-1 -right-1 bg-red-500 text-white text-xs px-1 rounded-full">
                      -{item.discount}%
                    </div>
                  )}
                </motion.div>
              );
            })}

            {/* Connection Lines (visible on hover) */}
            <AnimatePresence>
              {isHovered && bundle.accessories.map((_, index) => {
                const angle = (index * (360 / bundle.accessories.length)) * (Math.PI / 180);
                const length = 40;
                const x = Math.cos(angle) * length / 2;
                const y = Math.sin(angle) * length / 2;
                
                return (
                  <motion.div
                    key={`line-${index}`}
                    className="absolute w-0.5 bg-gradient-to-r from-purple-400 to-blue-400 origin-center"
                    style={{
                      height: length,
                      left: '50%',
                      top: '50%',
                      transformOrigin: 'center bottom',
                      transform: `translate(-50%, -${length/2}px) rotate(${angle * (180/Math.PI) - 90}deg)`,
                    }}
                    initial={{ scaleY: 0, opacity: 0 }}
                    animate={{ scaleY: 1, opacity: 0.6 }}
                    exit={{ scaleY: 0, opacity: 0 }}
                    transition={{ delay: index * 0.05 }}
                  />
                );
              })}
            </AnimatePresence>
          </div>

          {/* Product Info */}
          <div className="text-center mb-4">
            <h4 className="font-medium text-gray-900 dark:text-white mb-1">
              {bundle.mainProduct.name}
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              + {bundle.accessories.length} phụ kiện hoàn hảo
            </p>
          </div>

          {/* Pricing */}
          <div className="space-y-2 mb-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Giá gốc:</span>
              <span className="text-sm line-through text-gray-500">
                {formatPrice(bundle.totalOriginalPrice)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="font-medium text-gray-900 dark:text-white">Bundle:</span>
              <span className="font-bold text-purple-600 dark:text-purple-400">
                {formatPrice(bundle.totalBundlePrice)}
              </span>
            </div>
            <div className="flex justify-between items-center text-green-600 dark:text-green-400">
              <span className="text-sm">Tiết kiệm:</span>
              <span className="font-medium">
                {formatPrice(bundle.savings)}
              </span>
            </div>
          </div>

          {/* Actions */}
          <div className="space-y-2">
            <Button
              onClick={handleAddToChat}
              className="w-full bg-purple-600 hover:bg-purple-700 text-white"
              size="sm"
            >
              <Sparkles className="h-4 w-4 mr-2" />
              Tư vấn Bundle này
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
            
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ 
                height: isHovered ? 'auto' : 0, 
                opacity: isHovered ? 1 : 0 
              }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <Button
                variant="outline"
                className="w-full border-purple-200 dark:border-purple-800 text-purple-600 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-950/20"
                size="sm"
              >
                <ShoppingCart className="h-4 w-4 mr-2" />
                Thêm vào giỏ hàng
              </Button>
            </motion.div>
          </div>

          {/* Hover hint */}
          <AnimatePresence>
            {!isHovered && (
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-xs text-center text-gray-500 dark:text-gray-400 mt-2"
              >
                Hover để xem chi tiết các sản phẩm
              </motion.p>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>
    </motion.div>
  );
}
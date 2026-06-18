import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useCopilot } from "@/lib/copilot-store";
import { Send, Sparkles } from "lucide-react";
import { useState } from "react";

export function CopilotDrawer() {
  const { isOpen, closeDrawer, messages, addMessage, isLoading } = useCopilot();
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (!input.trim() || isLoading) return;
    
    addMessage({
      role: 'user',
      content: input.trim()
    });
    
    setInput("");
    
    // TODO: Call actual API
    setTimeout(() => {
      addMessage({
        role: 'assistant',
        content: `Tôi đã nhận được yêu cầu: "${input}". Đây là câu trả lời demo. Backend API sẽ được tích hợp trong bước tiếp theo.`
      });
    }, 1000);
  };

  return (
    <Dialog open={isOpen} onOpenChange={closeDrawer}>
      <DialogContent className="sm:max-w-2xl max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-purple-400" />
            AI Sales Copilot
          </DialogTitle>
        </DialogHeader>
        
        {/* Messages */}
        <div className="flex-1 overflow-y-auto space-y-4 py-4 min-h-[300px]">
          {messages.length === 0 ? (
            <div className="text-center text-muted-foreground py-8">
              <Sparkles className="h-12 w-12 mx-auto mb-4 text-purple-400/50" />
              <p>Chào bạn! Tôi có thể giúp gì cho bạn về sản phẩm điện tử?</p>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] p-3 rounded-lg ${
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted'
                  }`}
                >
                  <p className="text-sm">{message.content}</p>
                  <p className="text-xs opacity-70 mt-1">
                    {message.timestamp.toLocaleTimeString('vi-VN', {
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </p>
                </div>
              </div>
            ))
          )}
        </div>
        
        {/* Input */}
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Hỏi về sản phẩm, giá cả, thông số kỹ thuật..."
            className="flex-1 px-3 py-2 text-sm border rounded-md bg-background"
            disabled={isLoading}
          />
          <Button 
            onClick={handleSend} 
            disabled={!input.trim() || isLoading}
            size="sm"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
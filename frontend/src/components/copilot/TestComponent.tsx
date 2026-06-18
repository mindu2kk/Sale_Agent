import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useCopilot } from "@/lib/copilot-store"

export function TestComponent() {
  const { isOpen, toggleDrawer, addMessage } = useCopilot()
  
  const handleTest = () => {
    addMessage({
      role: 'user',
      content: 'Test message from new architecture!'
    })
  }
  
  return (
    <Card className="w-80 fixed bottom-4 right-4 z-50">
      <CardHeader>
        <CardTitle className="text-lg">Foundation Test</CardTitle>
        <CardDescription>
          Testing Tailwind + Zustand + shadcn/ui
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <Button onClick={toggleDrawer} variant="outline" className="w-full">
          Toggle Copilot: {isOpen ? 'Open' : 'Closed'}
        </Button>
        <Button onClick={handleTest} className="w-full">
          Add Test Message
        </Button>
      </CardContent>
    </Card>
  )
}
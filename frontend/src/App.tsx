import { TestComponent } from './components/copilot/TestComponent'
import { HighlightTooltip } from './components/copilot/HighlightTooltip'
import { CopilotDrawer } from './components/copilot/CopilotDrawer'
import { DemoContent } from './components/copilot/DemoContent'

export default function App() {
  return (
    <main className="relative w-screen h-screen overflow-hidden bg-gray-900 text-white">
      {/* Simple header */}
      <header className="fixed top-0 left-0 right-0 z-30 flex items-center justify-center p-6">
        <div className="text-center">
          <div className="text-xs uppercase tracking-widest text-purple-400">
            AI Sales Copilot
          </div>
          <h1 className="mt-2 text-2xl font-semibold text-white">
            Cỗ máy kể chuyện
          </h1>
        </div>
      </header>

      {/* Main content area */}
      <div className="pt-24 pb-8 px-4">
        <div className="max-w-2xl mx-auto text-center">
          <p className="text-gray-400 mb-8">
            Đang test Feature 2: Ecosystem Bundle Card
          </p>
          <div className="text-green-400">
            ✅ Tailwind CSS loaded successfully!
          </div>
        </div>
      </div>
      
      {/* Premium Features */}
      <HighlightTooltip />
      <CopilotDrawer />
      
      {/* Demo & Test Components */}
      <DemoContent />
      <TestComponent />
    </main>
  )
}

import { FluidMeshBackground } from './components/FluidMeshBackground'
import { ChatCore } from './components/ChatCore'

export default function App() {
  return (
    <main
      style={{
        position: 'relative',
        width: '100vw',
        height: '100vh',
        overflow: 'hidden',
        background: 'var(--bg-base)',
      }}
    >
      {/* Animated background */}
      <FluidMeshBackground />

      {/* Header */}
      <header
        style={{
          position: 'fixed', top: 0, left: 0, right: 0,
          zIndex: 30,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '20px 24px 16px',
          pointerEvents: 'none',
        }}
      >
        <div style={{ textAlign: 'center' }}>
          <div
            style={{
              fontSize: 10,
              textTransform: 'uppercase',
              letterSpacing: '0.4em',
              color: 'var(--core-ring)',
            }}
          >
            AI Sales Copilot
          </div>
          <h1
            style={{
              marginTop: 6,
              fontSize: 'clamp(18px, 4vw, 26px)',
              fontWeight: 600,
              color: 'white',
            }}
          >
            Cỗ máy kể chuyện
          </h1>
        </div>
      </header>

      {/* Chat interface */}
      <ChatCore />
    </main>
  )
}

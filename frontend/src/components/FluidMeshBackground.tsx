import { motion } from 'framer-motion'

export function FluidMeshBackground() {
  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: -1, overflow: 'hidden',
        pointerEvents: 'none',
      }}
    >
      {/* Blob 1 */}
      <motion.div
        style={{
          position: 'absolute',
          top: '-25%', left: '-25%',
          width: '70vmax', height: '70vmax',
          borderRadius: '50%',
          background: 'var(--mesh-1)',
          opacity: 0.6,
          filter: 'blur(80px)',
        }}
        animate={{ x: [0, 80, -40, 0], y: [0, -60, 40, 0] }}
        transition={{ duration: 22, repeat: Infinity, ease: 'easeInOut' }}
      />
      {/* Blob 2 */}
      <motion.div
        style={{
          position: 'absolute',
          top: '33%', right: '-25%',
          width: '60vmax', height: '60vmax',
          borderRadius: '50%',
          background: 'var(--mesh-2)',
          opacity: 0.5,
          filter: 'blur(80px)',
        }}
        animate={{ x: [0, -100, 60, 0], y: [0, 80, -50, 0] }}
        transition={{ duration: 26, repeat: Infinity, ease: 'easeInOut' }}
      />
      {/* Blob 3 */}
      <motion.div
        style={{
          position: 'absolute',
          bottom: '-25%', left: '25%',
          width: '65vmax', height: '65vmax',
          borderRadius: '50%',
          background: 'var(--mesh-3)',
          opacity: 0.45,
          filter: 'blur(80px)',
        }}
        animate={{ x: [0, 60, -80, 0], y: [0, -40, 60, 0] }}
        transition={{ duration: 30, repeat: Infinity, ease: 'easeInOut' }}
      />
      {/* Blob 4 */}
      <motion.div
        style={{
          position: 'absolute',
          top: '25%', left: '33%',
          width: '50vmax', height: '50vmax',
          borderRadius: '50%',
          background: 'var(--mesh-4)',
          opacity: 0.4,
          filter: 'blur(80px)',
        }}
        animate={{ x: [0, -60, 80, 0], y: [0, 50, -70, 0] }}
        transition={{ duration: 28, repeat: Infinity, ease: 'easeInOut' }}
      />
      {/* Subtle grain overlay */}
      <div
        style={{
          position: 'absolute', inset: 0,
          opacity: 0.04,
          mixBlendMode: 'overlay',
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.7'/%3E%3C/svg%3E")`,
        }}
      />
    </div>
  )
}

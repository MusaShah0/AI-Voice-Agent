import { useEffect, useRef } from 'react'

export default function LiveOrb({ size = 28, interactive = false, className = '' }) {
  const orbRef = useRef(null)

  useEffect(() => {
    if (!interactive) return undefined

    const scene = orbRef.current?.closest('.orb-scene')
    if (!scene) return undefined

    const updateGaze = (event) => {
      const bounds = scene.getBoundingClientRect()
      const orb = orbRef.current
      if (!orb || event.clientX < bounds.left || event.clientX > bounds.right ||
          event.clientY < bounds.top || event.clientY > bounds.bottom) {
        orb?.style.setProperty('--look-x', '0%')
        orb?.style.setProperty('--look-y', '0%')
        return
      }

      const x = Math.max(-1, Math.min(1, (event.clientX - bounds.left - bounds.width / 2) / (bounds.width / 2)))
      const y = Math.max(-1, Math.min(1, (event.clientY - bounds.top - bounds.height / 2) / (bounds.height / 2)))
      orb.style.setProperty('--look-x', `${x * 14}%`)
      orb.style.setProperty('--look-y', `${y * 12}%`)
    }

    window.addEventListener('pointermove', updateGaze, { passive: true })
    return () => window.removeEventListener('pointermove', updateGaze)
  }, [interactive])

  return (
    <span
      ref={orbRef}
      className={`live-orb ${className}`.trim()}
      style={{ width: size, height: size }}
      aria-hidden="true"
    >
      <span className="live-orb-face">
        <span className="live-orb-eyes">
          <i />
          <i />
        </span>
      </span>
    </span>
  )
}

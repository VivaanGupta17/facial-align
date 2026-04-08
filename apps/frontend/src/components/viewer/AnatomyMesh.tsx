import { useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import type { StructureLabel } from '../../types/medical'

interface AnatomyMeshProps {
  label: StructureLabel
  color: string
  opacity: number
  wireframe: boolean
  selected: boolean
  visible: boolean
  /** Position offset for displaced fragments */
  position?: [number, number, number]
  rotation?: [number, number, number]
  onClick?: () => void
  onHover?: (hovered: boolean) => void
  // Shape variant — maps anatomical structure to placeholder geometry
  shapeVariant?: 'mandible' | 'maxilla' | 'zygoma' | 'orbit' | 'teeth' | 'bone' | 'fragment'
}

/**
 * Individual anatomy mesh component.
 *
 * Currently renders placeholder geometry (BoxGeometry, etc.) to demonstrate
 * the viewer works. When real mesh loading is implemented:
 * 1. Use useGLTF() from @react-three/drei to load the GLB mesh
 * 2. Replace the placeholder mesh with <primitive object={gltf.scene} />
 * 3. Apply color via meshStandardMaterial with .color.set(color)
 * 4. Apply opacity via .material.opacity and .material.transparent
 */
export default function AnatomyMesh({
  label,
  color,
  opacity,
  wireframe,
  selected,
  visible,
  position = [0, 0, 0],
  rotation = [0, 0, 0],
  onClick,
  onHover,
  shapeVariant = 'bone',
}: AnatomyMeshProps) {
  const meshRef = useRef<THREE.Mesh>(null!)
  const [hovered, setHovered] = useState(false)

  // Subtle animation on selection
  useFrame((_, delta) => {
    if (!meshRef.current) return
    if (selected) {
      meshRef.current.rotation.y += delta * 0.2
    }
  })

  if (!visible) return null

  // Convert hex color to Three.js color
  const threeColor = new THREE.Color(color)

  // Hover / selected appearance
  const displayColor = hovered
    ? threeColor.clone().lerp(new THREE.Color('#ffffff'), 0.25)
    : selected
    ? threeColor.clone().lerp(new THREE.Color('#22d3ee'), 0.4)
    : threeColor

  const emissiveColor = selected
    ? new THREE.Color('#06b6d4')
    : hovered
    ? new THREE.Color('#94a3b8')
    : new THREE.Color('#000000')

  const emissiveIntensity = selected ? 0.15 : hovered ? 0.08 : 0

  // Placeholder geometry selection based on anatomical structure
  const PlaceholderGeometry = () => {
    switch (shapeVariant) {
      case 'mandible':
        // U-shaped jawbone approximation — use a wide elongated box
        return (
          <group>
            {/* Main body */}
            <mesh>
              <boxGeometry args={[6, 1.5, 1.2]} />
              <meshStandardMaterial color={displayColor} opacity={opacity} transparent wireframe={wireframe} emissive={emissiveColor} emissiveIntensity={emissiveIntensity} />
            </mesh>
            {/* Left ramus */}
            <mesh position={[-2.8, 2, 0]}>
              <boxGeometry args={[0.8, 3, 1.0]} />
              <meshStandardMaterial color={displayColor} opacity={opacity} transparent wireframe={wireframe} emissive={emissiveColor} emissiveIntensity={emissiveIntensity} />
            </mesh>
            {/* Right ramus */}
            <mesh position={[2.8, 2, 0]}>
              <boxGeometry args={[0.8, 3, 1.0]} />
              <meshStandardMaterial color={displayColor} opacity={opacity} transparent wireframe={wireframe} emissive={emissiveColor} emissiveIntensity={emissiveIntensity} />
            </mesh>
          </group>
        )
      case 'maxilla':
        return (
          <mesh>
            <boxGeometry args={[5.5, 1.2, 2.5]} />
            <meshStandardMaterial color={displayColor} opacity={opacity} transparent wireframe={wireframe} emissive={emissiveColor} emissiveIntensity={emissiveIntensity} />
          </mesh>
        )
      case 'zygoma':
        return (
          <mesh>
            <octahedronGeometry args={[1.5, 0]} />
            <meshStandardMaterial color={displayColor} opacity={opacity} transparent wireframe={wireframe} emissive={emissiveColor} emissiveIntensity={emissiveIntensity} />
          </mesh>
        )
      case 'orbit':
        return (
          <mesh>
            <torusGeometry args={[1.8, 0.4, 8, 20]} />
            <meshStandardMaterial color={displayColor} opacity={opacity} transparent wireframe={wireframe} emissive={emissiveColor} emissiveIntensity={emissiveIntensity} />
          </mesh>
        )
      case 'teeth':
        return (
          <group>
            {Array.from({ length: 14 }).map((_, i) => (
              <mesh key={i} position={[(i - 6.5) * 0.4, 0, i < 7 ? 0 : 0.2]}>
                <boxGeometry args={[0.3, 0.8, 0.3]} />
                <meshStandardMaterial color={displayColor} opacity={opacity} transparent wireframe={wireframe} emissive={emissiveColor} emissiveIntensity={emissiveIntensity} />
              </mesh>
            ))}
          </group>
        )
      case 'fragment':
        return (
          <mesh>
            <icosahedronGeometry args={[1, 0]} />
            <meshStandardMaterial color={displayColor} opacity={opacity} transparent wireframe={wireframe} emissive={emissiveColor} emissiveIntensity={emissiveIntensity} />
          </mesh>
        )
      case 'bone':
      default:
        return (
          <mesh>
            <dodecahedronGeometry args={[1.5, 0]} />
            <meshStandardMaterial color={displayColor} opacity={opacity} transparent wireframe={wireframe} emissive={emissiveColor} emissiveIntensity={emissiveIntensity} />
          </mesh>
        )
    }
  }

  return (
    <group
      ref={meshRef as any}
      position={position}
      rotation={rotation.map(r => (r * Math.PI) / 180) as [number, number, number]}
      onClick={(e) => { e.stopPropagation(); onClick?.() }}
      onPointerOver={(e) => { e.stopPropagation(); setHovered(true); onHover?.(true); document.body.style.cursor = 'pointer' }}
      onPointerOut={() => { setHovered(false); onHover?.(false); document.body.style.cursor = 'auto' }}
      data-testid={`anatomy-mesh-${label}`}
    >
      <PlaceholderGeometry />
    </group>
  )
}

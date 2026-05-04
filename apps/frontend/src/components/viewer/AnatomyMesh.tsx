import { useEffect, useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
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
  /** URL to GLB mesh file — when provided, loads real mesh instead of placeholder */
  meshUri?: string
}

/**
 * Individual anatomy mesh component.
 *
 * When meshUri is provided, it attempts to load a real GLB mesh.
 * If the mesh is unavailable or invalid, it falls back to placeholder geometry
 * instead of crashing the entire viewer.
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
  meshUri,
}: AnatomyMeshProps) {
  const meshRef = useRef<THREE.Mesh>(null!)
  const [hovered, setHovered] = useState(false)
  const [loadedScene, setLoadedScene] = useState<THREE.Group | null>(null)

  // Subtle animation on selection
  useFrame((_, delta) => {
    if (!meshRef.current) return
    if (selected) {
      meshRef.current.rotation.y += delta * 0.2
    }
  })

  useEffect(() => {
    if (!meshUri) {
      setLoadedScene(null)
      return
    }

    let cancelled = false
    const loader = new GLTFLoader()

    loader.load(
      meshUri,
      (gltf) => {
        if (!cancelled) {
          setLoadedScene(gltf.scene)
        }
      },
      undefined,
      () => {
        if (!cancelled) {
          setLoadedScene(null)
        }
      },
    )

    return () => {
      cancelled = true
    }
  }, [meshUri])

  const { displayColor, emissiveColor, emissiveIntensity } = useMemo(() => {
    const baseColor = new THREE.Color(color)
    const nextDisplayColor = hovered
      ? baseColor.clone().lerp(new THREE.Color('#ffffff'), 0.25)
      : selected
      ? baseColor.clone().lerp(new THREE.Color('#22d3ee'), 0.4)
      : baseColor

    const nextEmissiveColor = selected
      ? new THREE.Color('#06b6d4')
      : hovered
      ? new THREE.Color('#94a3b8')
      : new THREE.Color('#000000')

    return {
      displayColor: nextDisplayColor,
      emissiveColor: nextEmissiveColor,
      emissiveIntensity: selected ? 0.15 : hovered ? 0.08 : 0,
    }
  }, [color, hovered, selected])

  const renderedScene = useMemo(() => {
    if (!loadedScene) return null

    const clone = loadedScene.clone(true)
    clone.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.material = new THREE.MeshStandardMaterial({
          color: displayColor,
          opacity,
          transparent: opacity < 1,
          wireframe,
          emissive: emissiveColor,
          emissiveIntensity,
          side: THREE.DoubleSide,
        })
      }
    })

    return clone
  }, [displayColor, emissiveColor, emissiveIntensity, loadedScene, opacity, wireframe])

  if (!visible) return null

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
      name={label}
      position={position}
      rotation={rotation.map(r => (r * Math.PI) / 180) as [number, number, number]}
      onClick={(e) => { e.stopPropagation(); onClick?.() }}
      onPointerOver={(e) => { e.stopPropagation(); setHovered(true); onHover?.(true); document.body.style.cursor = 'pointer' }}
      onPointerOut={() => { setHovered(false); onHover?.(false); document.body.style.cursor = 'auto' }}
    >
      {renderedScene ? (
        <primitive object={renderedScene} />
      ) : (
        <PlaceholderGeometry />
      )}
    </group>
  )
}

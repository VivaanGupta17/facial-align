# Facial Align — Frontend

React/TypeScript frontend for the AI-native craniofacial surgical planning platform.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | React 18 + TypeScript |
| Build | Vite 5 |
| Styling | Tailwind CSS 3 (dark mode, custom medical palette) |
| State | Zustand (case, viewer, planning stores) |
| Data fetching | TanStack Query v5 |
| 3D viewer | React Three Fiber + @react-three/drei |
| Icons | lucide-react |
| Charts | recharts |
| Router | react-router-dom v6 |

## File Structure

```
src/
├── App.tsx                          # Root router
├── main.tsx                         # Entry point + QueryClient
├── styles/globals.css               # Tailwind base + CSS vars
├── types/medical.ts                 # All TypeScript types (472 lines)
├── lib/
│   ├── api.ts                       # Typed API client (mock-first)
│   └── mockData.ts                  # Realistic surgical mock data
├── hooks/
│   ├── useCases.ts                  # TanStack Query hooks — cases
│   ├── useSegmentation.ts           # TanStack Query hooks — segmentation
│   └── usePlanning.ts               # TanStack Query hooks — planning + review
├── stores/
│   ├── caseStore.ts                 # Active case state
│   ├── viewerStore.ts               # 3D viewer state (visibility, tools, camera)
│   └── planningStore.ts             # Planning state (transforms, history, undo/redo)
├── components/
│   ├── layout/
│   │   ├── AppShell.tsx             # Fixed sidebar + topbar layout
│   │   ├── Sidebar.tsx              # Navigation with SVG logo
│   │   └── TopBar.tsx               # Breadcrumbs + GPU status + user menu
│   ├── common/
│   │   ├── StatusBadge.tsx          # Color-coded case status badges
│   │   ├── ConfidenceBar.tsx        # Progress bar + ring for AI confidence
│   │   ├── MetricCard.tsx           # Metric display cards + inline metrics
│   │   ├── LoadingOverlay.tsx       # Skeleton, spinner, empty/error states
│   │   └── RecommendationCard.tsx   # AI recommendation with accept/reject
│   ├── viewer/
│   │   ├── Viewer3D.tsx             # Main Three.js canvas + structures panel
│   │   ├── AnatomyMesh.tsx          # Individual anatomy mesh (placeholder geometry)
│   │   ├── FragmentControls.tsx     # Transform sliders + AI suggestion + undo/redo
│   │   └── ViewerToolbar.tsx        # View mode, measurement, screenshot tools
│   └── planning/
│       ├── SegmentationReview.tsx   # Segmentation results + per-structure actions
│       ├── ReductionWorkspace.tsx   # Split 3D + planning panel
│       ├── OcclusionWorkspace.tsx   # Dental metrics + constraint toggles
│       └── SurgeonReview.tsx        # Approval checklist + signature
└── pages/
    ├── DashboardPage.tsx            # Stats + recent cases + system health
    ├── CaseListPage.tsx             # Filterable case table
    ├── CaseDetailPage.tsx           # Tabbed case workspace
    └── UploadPage.tsx               # 4-step DICOM upload wizard
```

## Running Locally

```bash
cd apps/frontend
npm install
npm run dev
```

App runs at http://localhost:5173

## Design System

- **Backgrounds**: `#0f172a` (bg), `#1e293b` (surface), `#233044` (surface-2)
- **Accent**: `#06b6d4` / `#22d3ee` (teal/cyan — AI elements)
- **Success**: `#10b981` (green — approved/confirmed)
- **Warning**: `#f59e0b` (amber — flagged/pending)
- **Error**: `#ef4444` (red — rejected/critical)
- **Typography**: Inter (sans), JetBrains Mono (measurements/coordinates)

## Mock Data

All API calls return mock data with simulated delays. The mock includes:
- 6 realistic surgical cases across all status stages
- Segmentation results with 8 anatomical structures + confidence scores
- Reduction plan with fragment transforms, occlusal metrics, constraint validations
- Surgeon review checklist
- Live system health data (GPU utilization, queue depth, model versions)

To connect the real backend, update `src/lib/api.ts` — each function has a comment
indicating where the real fetch call replaces the mock.

## 3D Viewer Notes

The 3D viewer uses placeholder geometry (BoxGeometry, OctahedronGeometry, etc.)
to demonstrate the viewer works. To load real meshes:

1. Use `useGLTF(meshUri)` from `@react-three/drei` in `AnatomyMesh.tsx`
2. Replace `<PlaceholderGeometry />` with `<primitive object={gltf.scene} />`
3. Apply per-structure color/opacity via `mesh.material.color.set(color)` and `.opacity`

## Testing

All components have `data-testid` attributes for E2E testing.

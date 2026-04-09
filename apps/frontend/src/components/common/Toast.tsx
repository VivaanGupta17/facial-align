import { useEffect, useState } from 'react'
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react'
import { useToastStore, type Toast } from '../../stores/toastStore'

const TOAST_CONFIG: Record<Toast['type'], { icon: React.ReactNode; bg: string; border: string; text: string }> = {
  success: {
    icon: <CheckCircle size={16} />,
    bg: 'bg-emerald-950/90',
    border: 'border-emerald-800',
    text: 'text-emerald-400',
  },
  error: {
    icon: <AlertCircle size={16} />,
    bg: 'bg-red-950/90',
    border: 'border-red-800',
    text: 'text-red-400',
  },
  warning: {
    icon: <AlertTriangle size={16} />,
    bg: 'bg-amber-950/90',
    border: 'border-amber-800',
    text: 'text-amber-400',
  },
  info: {
    icon: <Info size={16} />,
    bg: 'bg-cyan-950/90',
    border: 'border-cyan-800',
    text: 'text-cyan-400',
  },
}

function ToastItem({ toast }: { toast: Toast }) {
  const { removeToast } = useToastStore()
  const config = TOAST_CONFIG[toast.type]
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    requestAnimationFrame(() => setVisible(true))
  }, [])

  const handleDismiss = () => {
    setVisible(false)
    setTimeout(() => removeToast(toast.id), 200)
  }

  return (
    <div
      className={`flex items-start gap-2.5 px-3.5 py-2.5 rounded-lg border backdrop-blur-sm shadow-lg transition-all duration-200 max-w-sm ${config.bg} ${config.border} ${
        visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'
      }`}
      data-testid={`toast-${toast.type}`}
      role="alert"
    >
      <span className={`shrink-0 mt-0.5 ${config.text}`}>{config.icon}</span>
      <p className="flex-1 text-sm text-slate-200 leading-snug">{toast.message}</p>
      <button
        onClick={handleDismiss}
        className="shrink-0 text-slate-500 hover:text-slate-300 transition-colors"
        aria-label="Dismiss"
        data-testid="toast-dismiss"
      >
        <X size={14} />
      </button>
    </div>
  )
}

export default function ToastContainer() {
  const { toasts } = useToastStore()

  if (toasts.length === 0) return null

  return (
    <div
      className="fixed bottom-4 right-4 z-[9999] flex flex-col-reverse gap-2"
      data-testid="toast-container"
    >
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} />
      ))}
    </div>
  )
}

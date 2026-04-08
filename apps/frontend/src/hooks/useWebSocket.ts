/**
 * WebSocket hook for real-time job status updates in Facial Align.
 *
 * Features:
 *  - Auto-reconnect with exponential backoff (configurable)
 *  - Typed message protocol (SEGMENTATION_PROGRESS, REDUCTION_COMPLETE, MESH_READY, etc.)
 *  - Connection state machine
 *  - Integrates with Zustand stores to update segmentation / planning state
 *  - Cleans up automatically on unmount
 */

import { useEffect, useRef, useCallback, useState } from 'react'
import { useCaseStore } from '../stores/caseStore'
import { usePlanningStore } from '../stores/planningStore'

// =============================================================================
// Message types (mirror backend WebSocket protocol)
// =============================================================================

export type WsMessageType =
  | 'SEGMENTATION_PROGRESS'
  | 'SEGMENTATION_COMPLETE'
  | 'SEGMENTATION_FAILED'
  | 'REDUCTION_COMPLETE'
  | 'REDUCTION_FAILED'
  | 'MESH_READY'
  | 'MESH_FAILED'
  | 'JOB_QUEUED'
  | 'JOB_STARTED'
  | 'JOB_FAILED'
  | 'CASE_STATUS_CHANGED'
  | 'PING'
  | 'PONG'
  | 'CONNECT_ACK'
  | 'ERROR'

export interface WsBaseMessage {
  type: WsMessageType
  jobId?: string
  caseId?: string
  timestamp: string
  requestId?: string
}

export interface WsSegmentationProgressMessage extends WsBaseMessage {
  type: 'SEGMENTATION_PROGRESS'
  progress: number      // 0–100
  stage: string         // e.g. "preprocessing", "inference", "postprocessing"
  caseId: string
}

export interface WsSegmentationCompleteMessage extends WsBaseMessage {
  type: 'SEGMENTATION_COMPLETE'
  caseId: string
  resultId: string
  structureCount: number
  overallConfidence: number
}

export interface WsSegmentationFailedMessage extends WsBaseMessage {
  type: 'SEGMENTATION_FAILED'
  caseId: string
  errorCode: string
  errorMessage: string
}

export interface WsReductionCompleteMessage extends WsBaseMessage {
  type: 'REDUCTION_COMPLETE'
  caseId: string
  planId: string
  aiConfidence: number
}

export interface WsMeshReadyMessage extends WsBaseMessage {
  type: 'MESH_READY'
  caseId: string
  structureLabel: string
  meshUri: string
}

export interface WsCaseStatusChangedMessage extends WsBaseMessage {
  type: 'CASE_STATUS_CHANGED'
  caseId: string
  oldStatus: string
  newStatus: string
}

export interface WsErrorMessage extends WsBaseMessage {
  type: 'ERROR'
  errorCode: string
  errorMessage: string
}

export type WsMessage =
  | WsBaseMessage
  | WsSegmentationProgressMessage
  | WsSegmentationCompleteMessage
  | WsSegmentationFailedMessage
  | WsReductionCompleteMessage
  | WsMeshReadyMessage
  | WsCaseStatusChangedMessage
  | WsErrorMessage

// =============================================================================
// Connection state
// =============================================================================

export type WsConnectionState =
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'disconnected'
  | 'error'

// =============================================================================
// Configuration
// =============================================================================

interface WebSocketConfig {
  /** WebSocket server URL (defaults to ws://same-host/ws) */
  url?: string
  /** Maximum reconnection attempts before giving up (0 = unlimited) */
  maxReconnectAttempts?: number
  /** Initial reconnect delay in milliseconds */
  reconnectDelay?: number
  /** Maximum reconnect delay (exponential backoff cap) */
  maxReconnectDelay?: number
  /** Backoff multiplier */
  backoffMultiplier?: number
  /** Ping interval in ms (0 = disabled) */
  pingIntervalMs?: number
  /** Message handlers outside of store integration */
  onMessage?: (msg: WsMessage) => void
  /** Called when connection state changes */
  onStateChange?: (state: WsConnectionState) => void
}

// =============================================================================
// Internal state
// =============================================================================

interface WebSocketState {
  connectionState: WsConnectionState
  reconnectAttempts: number
  lastError: string | null
  latencyMs: number | null
  messagesReceived: number
}

// =============================================================================
// Hook
// =============================================================================

interface UseWebSocketReturn {
  connectionState: WsConnectionState
  reconnectAttempts: number
  lastError: string | null
  latencyMs: number | null
  messagesReceived: number
  send: (msg: Omit<WsBaseMessage, 'timestamp'>) => boolean
  disconnect: () => void
  reconnect: () => void
}

const WS_URL = typeof window !== 'undefined'
  ? `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws`
  : 'ws://localhost:8000/ws'

export function useWebSocket(config: WebSocketConfig = {}): UseWebSocketReturn {
  const {
    url = WS_URL,
    maxReconnectAttempts = 10,
    reconnectDelay = 1000,
    maxReconnectDelay = 30_000,
    backoffMultiplier = 1.5,
    pingIntervalMs = 30_000,
    onMessage,
    onStateChange,
  } = config

  const [state, setState] = useState<WebSocketState>({
    connectionState: 'idle',
    reconnectAttempts: 0,
    lastError: null,
    latencyMs: null,
    messagesReceived: 0,
  })

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pingTimestampRef = useRef<number | null>(null)
  const mountedRef = useRef(true)
  const reconnectAttemptsRef = useRef(0)
  const currentDelayRef = useRef(reconnectDelay)

  // Zustand store actions
  const { setCaseError } = useCaseStore()
  const { setGenerating } = usePlanningStore()

  const setConnectionState = useCallback((connectionState: WsConnectionState) => {
    if (!mountedRef.current) return
    setState(prev => ({ ...prev, connectionState }))
    onStateChange?.(connectionState)
  }, [onStateChange])

  const handleMessage = useCallback((event: MessageEvent) => {
    let msg: WsMessage
    try {
      msg = JSON.parse(event.data) as WsMessage
    } catch {
      console.warn('[useWebSocket] Failed to parse message:', event.data)
      return
    }

    if (!mountedRef.current) return

    setState(prev => ({ ...prev, messagesReceived: prev.messagesReceived + 1 }))

    // Handle latency pong
    if (msg.type === 'PONG' && pingTimestampRef.current != null) {
      const latency = Date.now() - pingTimestampRef.current
      pingTimestampRef.current = null
      setState(prev => ({ ...prev, latencyMs: latency }))
      return
    }

    // Integrate with Zustand stores
    dispatchToStores(msg, { setCaseError, setGenerating })

    // User-supplied handler
    onMessage?.(msg)
  }, [onMessage, setCaseError, setGenerating])

  const startPing = useCallback(() => {
    if (!pingIntervalMs) return
    pingTimerRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        pingTimestampRef.current = Date.now()
        wsRef.current.send(JSON.stringify({ type: 'PING', timestamp: new Date().toISOString() }))
      }
    }, pingIntervalMs)
  }, [pingIntervalMs])

  const stopPing = useCallback(() => {
    if (pingTimerRef.current) {
      clearInterval(pingTimerRef.current)
      pingTimerRef.current = null
    }
  }, [])

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return

    setConnectionState(reconnectAttemptsRef.current > 0 ? 'reconnecting' : 'connecting')

    let ws: WebSocket
    try {
      ws = new WebSocket(url)
    } catch (err) {
      setConnectionState('error')
      setState(prev => ({ ...prev, lastError: 'Failed to create WebSocket connection' }))
      return
    }

    wsRef.current = ws

    ws.onopen = () => {
      if (!mountedRef.current) return
      reconnectAttemptsRef.current = 0
      currentDelayRef.current = reconnectDelay
      setState(prev => ({
        ...prev,
        connectionState: 'connected',
        reconnectAttempts: 0,
        lastError: null,
      }))
      onStateChange?.('connected')
      startPing()
    }

    ws.onmessage = handleMessage

    ws.onerror = () => {
      if (!mountedRef.current) return
      setState(prev => ({ ...prev, lastError: 'WebSocket connection error' }))
    }

    ws.onclose = (event) => {
      if (!mountedRef.current) return
      stopPing()
      wsRef.current = null

      // Clean close — don't reconnect
      if (event.wasClean && event.code === 1000) {
        setConnectionState('disconnected')
        return
      }

      // Abnormal close — schedule reconnect
      if (maxReconnectAttempts === 0 || reconnectAttemptsRef.current < maxReconnectAttempts) {
        reconnectAttemptsRef.current++
        const delay = Math.min(currentDelayRef.current, maxReconnectDelay)
        currentDelayRef.current = Math.min(currentDelayRef.current * backoffMultiplier, maxReconnectDelay)

        setState(prev => ({
          ...prev,
          connectionState: 'reconnecting',
          reconnectAttempts: reconnectAttemptsRef.current,
          lastError: `Disconnected (code ${event.code}). Reconnecting in ${Math.round(delay / 1000)}s…`,
        }))
        onStateChange?.('reconnecting')

        reconnectTimerRef.current = setTimeout(connect, delay)
      } else {
        setConnectionState('error')
        setState(prev => ({
          ...prev,
          lastError: `Gave up reconnecting after ${reconnectAttemptsRef.current} attempt(s).`,
        }))
      }
    }
  }, [url, reconnectDelay, maxReconnectDelay, maxReconnectAttempts, backoffMultiplier,
      startPing, stopPing, handleMessage, setConnectionState, onStateChange])

  const disconnect = useCallback(() => {
    mountedRef.current = false
    if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
    stopPing()
    wsRef.current?.close(1000, 'Client disconnect')
    wsRef.current = null
    setState(prev => ({ ...prev, connectionState: 'disconnected' }))
  }, [stopPing])

  const reconnect = useCallback(() => {
    mountedRef.current = true
    reconnectAttemptsRef.current = 0
    currentDelayRef.current = reconnectDelay
    if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
    connect()
  }, [connect, reconnectDelay])

  const send = useCallback((msg: Omit<WsBaseMessage, 'timestamp'>): boolean => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return false
    wsRef.current.send(JSON.stringify({ ...msg, timestamp: new Date().toISOString() }))
    return true
  }, [])

  // Connect on mount
  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      stopPing()
      wsRef.current?.close(1000, 'Unmount')
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    connectionState: state.connectionState,
    reconnectAttempts: state.reconnectAttempts,
    lastError: state.lastError,
    latencyMs: state.latencyMs,
    messagesReceived: state.messagesReceived,
    send,
    disconnect,
    reconnect,
  }
}

// =============================================================================
// Focused hook for job progress updates
// =============================================================================

interface UseJobProgressOptions {
  jobId: string
  onProgress?: (progress: number, stage: string) => void
  onComplete?: (resultId?: string) => void
  onFailed?: (errorCode: string, errorMessage: string) => void
}

/**
 * Convenience hook — subscribe to progress/completion events for a single job.
 */
export function useJobProgress({
  jobId,
  onProgress,
  onComplete,
  onFailed,
}: UseJobProgressOptions) {
  const { connectionState, send } = useWebSocket({
    onMessage: useCallback((msg: WsMessage) => {
      if (msg.jobId !== jobId && (msg as WsSegmentationCompleteMessage).caseId == null) return

      if (msg.type === 'SEGMENTATION_PROGRESS') {
        const m = msg as WsSegmentationProgressMessage
        onProgress?.(m.progress, m.stage)
      } else if (msg.type === 'SEGMENTATION_COMPLETE') {
        const m = msg as WsSegmentationCompleteMessage
        onComplete?.(m.resultId)
      } else if (msg.type === 'SEGMENTATION_FAILED' || msg.type === 'JOB_FAILED') {
        const m = msg as WsSegmentationFailedMessage
        onFailed?.(m.errorCode ?? 'JOB_FAILED', m.errorMessage ?? 'Job failed')
      } else if (msg.type === 'REDUCTION_COMPLETE') {
        const m = msg as WsReductionCompleteMessage
        onComplete?.(m.planId)
      }
    }, [jobId, onProgress, onComplete, onFailed]),
  })

  // Subscribe to job events
  useEffect(() => {
    if (connectionState === 'connected') {
      send({ type: 'JOB_QUEUED', jobId })
    }
  }, [connectionState, jobId, send])

  return { connectionState }
}

// =============================================================================
// Store integration dispatcher
// =============================================================================

function dispatchToStores(
  msg: WsMessage,
  actions: {
    setCaseError: (error: string | null) => void
    setGenerating: (v: boolean) => void
  }
) {
  switch (msg.type) {
    case 'SEGMENTATION_FAILED':
    case 'JOB_FAILED': {
      const m = msg as WsSegmentationFailedMessage
      actions.setCaseError(m.errorMessage ?? 'A background job failed.')
      break
    }
    case 'REDUCTION_COMPLETE': {
      actions.setGenerating(false)
      break
    }
    case 'REDUCTION_FAILED': {
      actions.setGenerating(false)
      const m = msg as WsSegmentationFailedMessage
      actions.setCaseError(m.errorMessage ?? 'Reduction planning failed.')
      break
    }
    case 'CASE_STATUS_CHANGED': {
      // Could trigger a React Query invalidation here — left to the consumer
      break
    }
    default:
      break
  }
}

// =============================================================================
// Connection status badge utility
// =============================================================================

export function getConnectionStatusLabel(state: WsConnectionState): string {
  switch (state) {
    case 'idle':         return 'Not connected'
    case 'connecting':   return 'Connecting…'
    case 'connected':    return 'Connected'
    case 'reconnecting': return 'Reconnecting…'
    case 'disconnected': return 'Disconnected'
    case 'error':        return 'Connection error'
  }
}

export function getConnectionStatusColor(state: WsConnectionState): string {
  switch (state) {
    case 'connected':    return 'text-emerald-400'
    case 'connecting':
    case 'reconnecting': return 'text-amber-400'
    case 'error':        return 'text-red-400'
    default:             return 'text-slate-500'
  }
}

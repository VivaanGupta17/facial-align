/**
 * Error classes, boundary utilities, and user-friendly message formatting
 * for the Facial Align platform.
 *
 * Backend error codes mirror app/core/exceptions.py error_code strings.
 */

// =============================================================================
// Custom Error Classes
// =============================================================================

/** Base error class for all Facial Align frontend errors. */
export class FacialAlignError extends Error {
  readonly code: string
  readonly details?: Record<string, unknown>
  readonly timestamp: string

  constructor(message: string, code = 'INTERNAL_ERROR', details?: Record<string, unknown>) {
    super(message)
    this.name = 'FacialAlignError'
    this.code = code
    this.details = details
    this.timestamp = new Date().toISOString()
    // Preserve prototype chain in compiled ES5
    Object.setPrototypeOf(this, new.target.prototype)
  }
}

/** HTTP/REST API error — carries HTTP status code. */
export class ApiError extends FacialAlignError {
  readonly status: number

  constructor(
    message: string,
    status: number,
    code = 'API_ERROR',
    details?: Record<string, unknown>
  ) {
    super(message, code, details)
    this.name = 'ApiError'
    this.status = status
  }

  get isNotFound() { return this.status === 404 }
  get isUnauthorized() { return this.status === 401 }
  get isForbidden() { return this.status === 403 }
  get isConflict() { return this.status === 409 }
  get isUnprocessable() { return this.status === 422 }
  get isServerError() { return this.status >= 500 }
  get isClientError() { return this.status >= 400 && this.status < 500 }
}

/** Network-level failure — fetch rejected, no HTTP response received. */
export class NetworkError extends FacialAlignError {
  readonly cause?: Error

  constructor(message = 'Unable to connect to the server. Check your network connection.', cause?: Error) {
    super(message, 'NETWORK_ERROR')
    this.name = 'NetworkError'
    this.cause = cause
  }
}

/** Client-side input validation failure. */
export class ValidationError extends FacialAlignError {
  readonly field?: string
  readonly fieldErrors?: Record<string, string>

  constructor(message: string, field?: string, fieldErrors?: Record<string, string>) {
    super(message, 'VALIDATION_ERROR')
    this.name = 'ValidationError'
    this.field = field
    this.fieldErrors = fieldErrors
  }
}

/** Authentication failure — token missing, expired, or invalid. */
export class AuthenticationError extends FacialAlignError {
  constructor(message = 'Your session has expired. Please sign in again.') {
    super(message, 'AUTHENTICATION_ERROR')
    this.name = 'AuthenticationError'
  }
}

/** Authorization failure — user lacks required permissions. */
export class AuthorizationError extends FacialAlignError {
  constructor(message = 'You do not have permission to perform this action.') {
    super(message, 'FORBIDDEN')
    this.name = 'AuthorizationError'
  }
}

// =============================================================================
// Backend Error Code → User-Friendly Message Map
// Mirrors all error_code strings from app/core/exceptions.py
// =============================================================================

export const ERROR_CODE_MESSAGES: Readonly<Record<string, string>> = {
  // Generic
  INTERNAL_ERROR: 'An unexpected server error occurred. Please try again.',
  NOT_FOUND: 'The requested resource was not found.',
  CONFLICT: 'This operation conflicts with existing data.',
  FORBIDDEN: 'You do not have permission to perform this action.',
  INSUFFICIENT_PERMISSIONS: 'Your account lacks the required permissions for this action.',

  // Entity not found
  PATIENT_NOT_FOUND: 'Patient record not found.',
  STUDY_NOT_FOUND: 'DICOM study not found.',
  CASE_NOT_FOUND: 'Surgical case not found.',
  PLAN_NOT_FOUND: 'Reduction plan not found.',
  SEGMENTATION_NOT_FOUND: 'Segmentation results not found for this case.',

  // Validation
  VALIDATION_ERROR: 'One or more input values are invalid. Please check the form.',
  INVALID_TRANSFORM: 'The 3D transform matrix is invalid — rotation must be orthonormal.',
  INVALID_OCCLUSAL_CONSTRAINT: 'The occlusal constraints are geometrically infeasible with the current fragment positions.',

  // Conflict
  DUPLICATE_STUDY: 'This DICOM study has already been uploaded. Duplicate studies are not permitted.',
  INVALID_STATUS_TRANSITION: 'This case cannot be moved to the requested status from its current state.',

  // DICOM
  DICOM_ERROR: 'An error occurred while processing the DICOM data.',
  DICOM_PARSE_ERROR: 'Failed to parse the DICOM file. Ensure the file is a valid, uncompressed CT series.',
  DICOM_VALIDATION_ERROR: 'The DICOM study does not meet the required quality standards for surgical planning.',
  DICOM_DEIDENTIFICATION_ERROR: 'Failed to de-identify the DICOM study. Contact your administrator.',
  INSUFFICIENT_ANATOMICAL_COVERAGE: 'The CT volume does not cover the full craniofacial region required for planning. Please acquire a full facial CT.',
  SLICE_THICKNESS_TOO_LARGE: 'CT slice thickness exceeds the 1.5 mm limit for surgical planning. Acquire a thinner-slice series.',

  // Segmentation
  SEGMENTATION_ERROR: 'The segmentation pipeline encountered an error.',
  MODEL_LOAD_ERROR: 'The AI segmentation model is temporarily unavailable. Please try again in a few minutes.',
  INFERENCE_ERROR: 'AI inference failed. The CT data may be too noisy or incomplete.',
  MODEL_NOT_AVAILABLE: 'The requested AI model is not available in the registry.',
  SEGMENTATION_POSTPROCESSING_ERROR: 'Segmentation post-processing failed. The mask may be incomplete.',
  LOW_CONFIDENCE_SEGMENTATION: 'Segmentation confidence is below the clinical threshold. Manual review is required.',

  // Mesh
  MESH_ERROR: 'A mesh processing error occurred.',
  MESH_EXTRACTION_ERROR: 'Failed to extract the 3D surface mesh from the segmentation mask.',
  MESH_SIMPLIFICATION_ERROR: 'Mesh decimation failed. The surface geometry may be too complex.',
  EMPTY_MASK: 'The segmentation mask is empty — no voxels were labeled for this structure.',
  MESH_QUALITY_ERROR: 'The extracted mesh does not meet quality thresholds (manifold, self-intersections).',

  // Registration
  REGISTRATION_ERROR: 'A 3D registration error occurred.',
  ICP_CONVERGENCE_ERROR: 'ICP registration did not converge. The fragments may be too far apart or have insufficient overlap.',
  REGISTRATION_DIVERGENCE: 'Registration produced an unacceptably high residual error.',
  INSUFFICIENT_OVERLAP: 'The source and target meshes have insufficient overlap to register.',

  // Reduction planning
  REDUCTION_PLANNING_ERROR: 'An error occurred during reduction planning.',
  FRAGMENT_PROCESSING_ERROR: 'Error processing fracture fragment geometry.',
  CONSTRAINT_VIOLATION: 'The planned reduction violates one or more anatomical or occlusal constraints.',
  SYMMETRY_THRESHOLD_EXCEEDED: 'The planned reduction exceeds the bilateral symmetry threshold.',

  // Occlusion
  OCCLUSION_ERROR: 'An occlusion analysis error occurred.',
  DENTAL_ARCH_ERROR: 'Error processing dental arch geometry.',
  OCCLUSION_METRIC_ERROR: 'Failed to compute occlusal metric.',

  // Storage
  STORAGE_ERROR: 'A file storage error occurred.',
  FILE_NOT_FOUND: 'The requested file could not be found in storage.',
  STORAGE_QUOTA_EXCEEDED: 'Storage quota exceeded. Contact your administrator to increase capacity.',

  // Tasks
  TASK_ERROR: 'A background job error occurred.',
  TASK_NOT_FOUND: 'The background job could not be found.',
  TASK_TIMEOUT: 'The background job timed out. The server may be under heavy load — try again later.',

  // Network / client-side
  NETWORK_ERROR: 'Unable to connect to the server. Check your network connection.',
  AUTHENTICATION_ERROR: 'Your session has expired. Please sign in again.',
}

// =============================================================================
// User-Friendly Error Formatter
// =============================================================================

export interface FormattedError {
  title: string
  description: string
  code: string
  recoverable: boolean
  actionLabel?: string
}

/**
 * Convert any caught error into a consistently formatted object
 * suitable for rendering in a UI component.
 */
export function formatError(error: unknown): FormattedError {
  if (error instanceof AuthenticationError) {
    return {
      title: 'Session Expired',
      description: error.message,
      code: error.code,
      recoverable: true,
      actionLabel: 'Sign In',
    }
  }

  if (error instanceof AuthorizationError) {
    return {
      title: 'Access Denied',
      description: error.message,
      code: error.code,
      recoverable: false,
    }
  }

  if (error instanceof NetworkError) {
    return {
      title: 'Connection Error',
      description: error.message,
      code: error.code,
      recoverable: true,
      actionLabel: 'Retry',
    }
  }

  if (error instanceof ValidationError) {
    return {
      title: 'Validation Error',
      description: error.message,
      code: error.code,
      recoverable: true,
      actionLabel: 'Review Input',
    }
  }

  if (error instanceof ApiError) {
    const knownMessage = ERROR_CODE_MESSAGES[error.code]
    return {
      title: getErrorTitle(error),
      description: knownMessage ?? error.message,
      code: error.code,
      recoverable: !error.isServerError || error.status === 503,
      actionLabel: error.isServerError ? 'Retry' : undefined,
    }
  }

  if (error instanceof FacialAlignError) {
    const knownMessage = ERROR_CODE_MESSAGES[error.code]
    return {
      title: 'Error',
      description: knownMessage ?? error.message,
      code: error.code,
      recoverable: true,
      actionLabel: 'Retry',
    }
  }

  if (error instanceof Error) {
    return {
      title: 'Unexpected Error',
      description: error.message || 'An unexpected error occurred.',
      code: 'UNKNOWN_ERROR',
      recoverable: true,
      actionLabel: 'Retry',
    }
  }

  return {
    title: 'Unknown Error',
    description: 'An unknown error occurred.',
    code: 'UNKNOWN_ERROR',
    recoverable: true,
    actionLabel: 'Retry',
  }
}

function getErrorTitle(error: ApiError): string {
  if (error.isNotFound) return 'Not Found'
  if (error.isUnauthorized) return 'Session Expired'
  if (error.isForbidden) return 'Access Denied'
  if (error.isConflict) return 'Conflict'
  if (error.isUnprocessable) return 'Invalid Data'
  if (error.isServerError) return 'Server Error'
  return 'Error'
}

// =============================================================================
// Parse API Response Errors
// =============================================================================

/**
 * Parse a failed fetch response into a typed ApiError.
 * Expects backend JSON shape: { error: string, message: string, context?: {} }
 */
export async function parseApiError(response: Response): Promise<ApiError> {
  let code = 'API_ERROR'
  let message = `HTTP ${response.status}`
  let details: Record<string, unknown> | undefined

  try {
    const body = await response.json()
    // FastAPI exception handler wraps in { detail: { error, message, context? } }
    const detail = body?.detail ?? body
    code = detail?.error ?? code
    message = detail?.message ?? message
    details = detail?.context
  } catch {
    // Body not JSON — leave defaults
  }

  return new ApiError(message, response.status, code, details)
}

// =============================================================================
// Error Boundary Context Helpers
// =============================================================================

export interface ErrorContext {
  component?: string
  action?: string
  caseId?: string
  planId?: string
  fragmentId?: string
  extra?: Record<string, unknown>
}

/**
 * Log an error with structured context for debugging.
 * In production, replace console.error with your telemetry SDK.
 */
export function logError(error: unknown, context: ErrorContext = {}): void {
  const formatted = formatError(error)
  console.error('[FacialAlign Error]', {
    code: formatted.code,
    title: formatted.title,
    description: formatted.description,
    context,
    raw: error,
    timestamp: new Date().toISOString(),
  })
}

/**
 * Type-guard: is this a retryable error?
 */
export function isRetryable(error: unknown): boolean {
  if (error instanceof NetworkError) return true
  if (error instanceof ApiError) return error.isServerError || error.status === 429
  return false
}

/**
 * Type-guard: does this error require re-authentication?
 */
export function requiresReauth(error: unknown): boolean {
  if (error instanceof AuthenticationError) return true
  if (error instanceof ApiError) return error.isUnauthorized
  return false
}

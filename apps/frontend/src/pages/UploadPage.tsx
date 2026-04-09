import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, FileText, ChevronRight, Check, AlertCircle, X, Plus, AlertTriangle } from 'lucide-react'
import { studiesApi, casesApi } from '../lib/api'
import { Spinner } from '../components/common/LoadingOverlay'
import { useToastStore } from '../stores/toastStore'

type Step = 1 | 2 | 3 | 4

const STEPS = [
  { n: 1 as Step, label: 'Upload DICOM' },
  { n: 2 as Step, label: 'Review Metadata' },
  { n: 3 as Step, label: 'Case Setup' },
  { n: 4 as Step, label: 'Processing' },
]

interface DicomTag {
  tag: string
  name: string
  value: string
  sensitive: boolean
}

export default function UploadPage() {
  const navigate = useNavigate()
  const { addToast } = useToastStore()
  const [step, setStep] = useState<Step>(1)
  const [files, setFiles] = useState<File[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadCancelled, setUploadCancelled] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [studyId, setStudyId] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [dicomTags, setDicomTags] = useState<DicomTag[]>([])
  const [metadataLoading, setMetadataLoading] = useState(false)
  const [caseType, setCaseType] = useState('mandible_fracture')
  const [notes, setNotes] = useState('')
  const [createdCaseNumber, setCreatedCaseNumber] = useState<string | null>(null)
  const [createdCaseId, setCreatedCaseId] = useState<string | null>(null)
  // Chunked upload state
  const [isChunked, setIsChunked] = useState(false)
  const [chunkCurrent, setChunkCurrent] = useState(0)
  const [chunkTotal, setChunkTotal] = useState(0)
  const [uploadSpeed, setUploadSpeed] = useState(0)
  const [uploadEta, setUploadEta] = useState(0)

  // Study type for multi-study support
  const [studyType, setStudyType] = useState('pre_op')
  const [studyLabel, setStudyLabel] = useState('')

  const fileInputRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const droppedFiles = Array.from(e.dataTransfer.files)
    setFiles(prev => [...prev, ...droppedFiles])
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(prev => [...prev, ...Array.from(e.target.files!)])
    }
  }

  const removeFile = (i: number) => setFiles(prev => prev.filter((_, idx) => idx !== i))

  const totalSizeMb = files.reduce((s, f) => s + f.size, 0) / (1024 * 1024)

  const handleCancelUpload = () => {
    abortRef.current?.abort()
    setUploadCancelled(true)
    setIsUploading(false)
    setUploadProgress(0)
    addToast({ type: 'info', message: 'Upload cancelled' })
  }

  const CHUNKED_THRESHOLD = 100 * 1024 * 1024 // 100 MB

  const fmtSpeed = (bps: number) => {
    if (bps > 1e6) return `${(bps / 1e6).toFixed(1)} MB/s`
    if (bps > 1e3) return `${(bps / 1e3).toFixed(0)} KB/s`
    return `${bps.toFixed(0)} B/s`
  }
  const fmtEta = (sec: number) => {
    if (sec < 60) return `${Math.ceil(sec)}s`
    if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.ceil(sec % 60)}s`
    return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`
  }

  const handleUpload = async () => {
    if (files.length === 0) return
    setIsUploading(true)
    setUploadError(null)
    setUploadCancelled(false)
    abortRef.current = new AbortController()
    const useChunked = files[0].size > CHUNKED_THRESHOLD
    setIsChunked(useChunked)
    try {
      let result: { jobId: string; studyId: string }
      if (useChunked) {
        result = await studiesApi.uploadChunked(files[0], 'auto-mrn', {
          signal: abortRef.current.signal,
          caseType: caseType,
          onProgress: (received, total, speedBps, etaSec) => {
            setChunkCurrent(received)
            setChunkTotal(total)
            setUploadSpeed(speedBps)
            setUploadEta(etaSec)
            setUploadProgress(Math.round((received / total) * 100))
          },
        })
      } else {
        result = await studiesApi.upload(files[0], setUploadProgress)
      }
      if (uploadCancelled) return
      setJobId(result.jobId)
      setStudyId(result.studyId)

      // Fetch real metadata from the study
      setMetadataLoading(true)
      try {
        const metadata = await studiesApi.getMetadata(result.studyId)
        const tags: DicomTag[] = Object.entries(metadata).map(([key, val]) => {
          const sensitive = ['patientName', 'patientBirthDate', 'patient_name', 'patient_birth_date'].includes(key)
          return {
            tag: key,
            name: key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ').trim(),
            value: sensitive ? 'STRIPPED — Anonymized' : String(val ?? ''),
            sensitive,
          }
        })
        setDicomTags(tags)
      } catch {
        setDicomTags([])
      } finally {
        setMetadataLoading(false)
      }

      setStep(2)
    } catch (err) {
      if (!uploadCancelled) {
        setUploadError('Upload failed. Please check your connection and try again.')
      }
    } finally {
      setIsUploading(false)
    }
  }

  const handleCreateCase = async () => {
    try {
      const newCase = await casesApi.create({
        caseType: caseType as any,
        studyId: studyId ?? undefined,
      })
      setCreatedCaseNumber(newCase.caseNumber)
      setCreatedCaseId(newCase.id)
      addToast({ type: 'success', message: `Case ${newCase.caseNumber} created successfully` })
      setStep(4)
    } catch (err) {
      addToast({ type: 'error', message: 'Failed to create case. Please try again.' })
    }
  }

  return (
    <div className="p-6 max-w-4xl mx-auto animate-fade-in" data-testid="upload-page">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-xl font-bold text-slate-100">Upload DICOM Study</h1>
        <p className="text-sm text-slate-400 mt-0.5">Import CT or CBCT series to create a new surgical planning case</p>
      </div>

      {/* Step indicator */}
      <div className="flex items-center gap-0 mb-8" data-testid="step-indicator">
        {STEPS.map((s, i) => (
          <div key={s.n} className="flex items-center">
            <div className="flex items-center gap-2">
              <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${
                step > s.n ? 'bg-emerald-600 text-white' : step === s.n ? 'bg-cyan-500 text-slate-900' : 'bg-slate-700 text-slate-400'
              }`}>
                {step > s.n ? <Check size={14} /> : s.n}
              </div>
              <span className={`text-sm font-medium ${step === s.n ? 'text-slate-100' : step > s.n ? 'text-emerald-400' : 'text-slate-500'}`}>
                {s.label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <div className={`flex-1 h-px mx-4 w-16 ${step > s.n ? 'bg-emerald-600' : 'bg-slate-700'}`} />
            )}
          </div>
        ))}
      </div>

      {/* Step 1: Upload */}
      {step === 1 && (
        <div className="space-y-4 animate-fade-in" data-testid="step-1">
          {/* Drop zone */}
          <div
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
            onDragLeave={() => setIsDragging(false)}
            className={`border-2 border-dashed rounded-xl p-12 text-center transition-all ${
              isDragging
                ? 'border-cyan-500 bg-cyan-950/30'
                : 'border-slate-600 hover:border-slate-500 bg-slate-800/50'
            }`}
            data-testid="drop-zone"
          >
            <div className="flex flex-col items-center gap-4">
              <div className={`w-16 h-16 rounded-full flex items-center justify-center ${isDragging ? 'bg-cyan-900' : 'bg-slate-700'}`}>
                <Upload size={28} className={isDragging ? 'text-cyan-400' : 'text-slate-400'} />
              </div>
              <div>
                <p className="text-base font-semibold text-slate-200">Drop DICOM files or folder here</p>
                <p className="text-sm text-slate-400 mt-1">
                  Supports .dcm, .dicom, or ZIP archives. CT and CBCT modalities supported.
                </p>
              </div>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="btn-secondary"
              >
                Browse Files
              </button>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".dcm,.dicom,.zip"
                className="hidden"
                onChange={handleFileSelect}
                data-testid="file-input"
              />
            </div>
          </div>

          {/* File list */}
          {files.length > 0 && (
            <div className="bg-slate-800 border border-slate-700 rounded-lg" data-testid="file-list">
              <div className="panel-header">
                <h3 className="panel-title">Selected Files ({files.length})</h3>
                <span className="text-xs font-mono text-slate-400">{totalSizeMb.toFixed(1)} MB</span>
              </div>
              <div className="divide-y divide-slate-700 max-h-48 overflow-y-auto">
                {files.map((f, i) => (
                  <div key={i} className="flex items-center gap-3 px-4 py-2.5" data-testid={`file-item-${i}`}>
                    <FileText size={14} className="text-slate-400 shrink-0" />
                    <span className="flex-1 text-sm text-slate-300 font-mono truncate">{f.name}</span>
                    <span className="text-xs text-slate-500 font-mono shrink-0">{(f.size / 1024).toFixed(0)} KB</span>
                    <button onClick={() => removeFile(i)} className="text-slate-500 hover:text-red-400 transition-colors shrink-0">
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* File size warning */}
          {totalSizeMb > 500 && !isUploading && (
            <div className="flex items-center gap-2 p-3 bg-amber-950/60 border border-amber-800 rounded-lg text-sm text-amber-400" data-testid="file-size-warning">
              <AlertTriangle size={15} />
              File is very large ({totalSizeMb.toFixed(0)} MB). Upload may take several minutes.
            </div>
          )}

          {/* Upload progress */}
          {isUploading && (
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-4" data-testid="upload-progress">
              <div className="flex items-center gap-3 mb-2">
                <Spinner size={16} />
                <span className="flex-1 text-sm text-slate-300">
                  {isChunked
                    ? `Uploading chunk ${chunkCurrent} of ${chunkTotal}...`
                    : 'Uploading and validating DICOM data...'}
                </span>
                <button
                  onClick={handleCancelUpload}
                  className="flex items-center gap-1 text-xs text-red-400 hover:text-red-300 transition-colors"
                  data-testid="cancel-upload-btn"
                >
                  <X size={13} /> Cancel
                </button>
              </div>
              <div className="h-2 bg-slate-700 rounded-full">
                <div
                  className="h-full bg-cyan-500 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <div className="flex justify-between mt-1">
                <span className="text-xs text-slate-500 font-mono">
                  {isChunked && uploadSpeed > 0 && `${fmtSpeed(uploadSpeed)} — ETA ${fmtEta(uploadEta)}`}
                </span>
                <span className="text-xs text-slate-500 font-mono">{Math.round(uploadProgress)}%</span>
              </div>
            </div>
          )}

          {uploadError && (
            <div className="flex items-center gap-2 p-3 bg-red-950 border border-red-800 rounded-lg text-sm text-red-400" data-testid="upload-error">
              <AlertCircle size={15} />
              {uploadError}
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-3">
            {files.length === 0 && (
              <button
                onClick={() => { setFiles([new File([], 'DEMO_CT_HEAD_412_slices.zip')]); }}
                className="btn-ghost text-cyan-400 text-sm"
                data-testid="demo-data-btn"
              >
                + Load demo data
              </button>
            )}
            <button
              onClick={handleUpload}
              disabled={files.length === 0 || isUploading}
              className="flex items-center gap-2 btn-primary disabled:opacity-50"
              data-testid="upload-btn"
            >
              {isUploading ? <Spinner size={14} /> : <Upload size={15} />}
              Upload & Process
              <ChevronRight size={15} />
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Metadata Review */}
      {step === 2 && (
        <div className="space-y-4 animate-fade-in" data-testid="step-2">
          <div className="bg-slate-800 border border-slate-700 rounded-lg">
            <div className="panel-header">
              <h3 className="panel-title">Study Metadata</h3>
              <div className="flex items-center gap-3">
                {studyId && <span className="text-2xs font-mono text-slate-500">Study: {studyId.slice(0, 8)}...</span>}
                {jobId && <span className="text-2xs font-mono text-slate-500">Job: {jobId.slice(0, 8)}...</span>}
              </div>
            </div>
            {metadataLoading ? (
              <div className="flex items-center justify-center p-8">
                <Spinner size={20} />
                <span className="ml-3 text-sm text-slate-400">Loading metadata...</span>
              </div>
            ) : dicomTags.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="data-table w-full">
                  <thead>
                    <tr>
                      <th className="w-48">Attribute</th>
                      <th>Value</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {dicomTags.map(tag => (
                      <tr key={tag.tag} data-testid={`dicom-tag-${tag.tag}`}>
                        <td><span className="text-sm text-slate-300">{tag.name}</span></td>
                        <td>
                          <span className={`text-sm font-mono ${tag.sensitive ? 'text-amber-400' : 'text-slate-200'}`}>
                            {tag.value}
                          </span>
                        </td>
                        <td>
                          {tag.sensitive && (
                            <span className="flex items-center gap-1 text-2xs text-amber-400">
                              <AlertCircle size={10} /> PHI Removed
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="p-8 text-center text-sm text-slate-500">
                No additional metadata available for this study.
              </div>
            )}
          </div>

          <div className="flex items-center gap-2 p-3 bg-emerald-950 border border-emerald-800 rounded-lg text-sm text-emerald-400" data-testid="phi-notice">
            <Check size={15} />
            All PHI fields have been automatically stripped. Only clinical metadata retained.
          </div>

          <div className="flex justify-between">
            <button onClick={() => setStep(1)} className="btn-secondary">← Back</button>
            <button onClick={() => setStep(3)} className="flex items-center gap-2 btn-primary" data-testid="step2-next">
              Continue to Case Setup <ChevronRight size={15} />
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Case Creation Form */}
      {step === 3 && (
        <div className="space-y-4 animate-fade-in" data-testid="step-3">
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 space-y-5">
            <h3 className="text-sm font-semibold text-slate-200">Case Configuration</h3>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label-sm block mb-1.5">Case Type *</label>
                <select
                  value={caseType}
                  onChange={e => setCaseType(e.target.value)}
                  className="select-base"
                  data-testid="case-type-select"
                >
                  <option value="mandible_fracture">Mandible Fracture</option>
                  <option value="midface_fracture">Midface Fracture</option>
                  <option value="panfacial_fracture">Panfacial Fracture</option>
                  <option value="orbital_fracture">Orbital Fracture</option>
                  <option value="frontal_sinus_fracture">Frontal Sinus Fracture</option>
                  <option value="orthognathic">Orthognathic Surgery</option>
                  <option value="tumor_resection">Tumor Resection</option>
                  <option value="reconstruction">Reconstruction</option>
                </select>
              </div>

              <div>
                <label className="label-sm block mb-1.5">Target Surgery Date</label>
                <input type="datetime-local" className="input-base" data-testid="scheduled-date" />
              </div>
            </div>

            {/* Study Classification (multi-study support) */}
            <div className="border border-slate-700 rounded-md p-3 bg-slate-900">
              <p className="text-xs font-semibold text-slate-300 mb-2">Study Classification</p>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label-sm block mb-1.5">Study Type</label>
                  <select
                    value={studyType}
                    onChange={e => setStudyType(e.target.value)}
                    className="select-base"
                    data-testid="study-type-select"
                  >
                    <option value="pre_op">Pre-operative</option>
                    <option value="post_op">Post-operative</option>
                    <option value="follow_up">Follow-up</option>
                    <option value="intra_op">Intra-operative</option>
                  </select>
                </div>
                <div>
                  <label className="label-sm block mb-1.5">Study Label (optional)</label>
                  <input
                    type="text"
                    value={studyLabel}
                    onChange={e => setStudyLabel(e.target.value)}
                    placeholder="e.g., Initial CT, 6-week follow-up"
                    className="input-base"
                    data-testid="study-label-input"
                  />
                </div>
              </div>
            </div>

            <div>
              <label className="label-sm block mb-1.5">Clinical Notes</label>
              <textarea
                value={notes}
                onChange={e => setNotes(e.target.value)}
                rows={4}
                placeholder="Add clinical notes, fracture description, relevant history..."
                className="input-base resize-none"
                data-testid="clinical-notes"
              />
            </div>

            <div className="border border-slate-700 rounded-md p-3 bg-slate-900">
              <p className="text-xs font-semibold text-slate-300 mb-2">AI Processing Options</p>
              <div className="space-y-2">
                {[
                  { label: 'Run auto-segmentation after upload', id: 'auto-seg', checked: true },
                  { label: 'Generate initial reduction plan', id: 'auto-plan', checked: true },
                  { label: 'Send notification when complete', id: 'notify', checked: true },
                ].map(opt => (
                  <label key={opt.id} className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer" data-testid={`opt-${opt.id}`}>
                    <input
                      type="checkbox"
                      defaultChecked={opt.checked}
                      className="rounded border-slate-600 bg-slate-800 text-cyan-500"
                    />
                    {opt.label}
                  </label>
                ))}
              </div>
            </div>
          </div>

          <div className="flex justify-between">
            <button onClick={() => setStep(2)} className="btn-secondary">← Back</button>
            <button
              onClick={handleCreateCase}
              className="flex items-center gap-2 btn-primary"
              data-testid="create-case-btn"
            >
              <Plus size={15} />
              Create Case & Start Processing
            </button>
          </div>
        </div>
      )}

      {/* Step 4: Confirmation */}
      {step === 4 && (
        <div className="animate-fade-in" data-testid="step-4">
          <div className="text-center py-12 space-y-6">
            <div className="w-20 h-20 rounded-full bg-emerald-950 border-2 border-emerald-600 flex items-center justify-center mx-auto">
              <Check size={36} className="text-emerald-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-slate-100">Case Created Successfully</h2>
              <p className="text-slate-400 mt-2">AI segmentation has been queued — estimated completion in 12–18 minutes.</p>
            </div>

            <div className="max-w-sm mx-auto bg-slate-800 border border-slate-700 rounded-lg p-4 text-left space-y-2">
              {[
                { label: 'Case Number', value: createdCaseNumber ?? 'Pending...' },
                { label: 'Status', value: 'Segmentation Queued' },
                { label: 'Case Type', value: caseType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) },
              ].map(r => (
                <div key={r.label} className="flex justify-between text-sm">
                  <span className="text-slate-400">{r.label}</span>
                  <span className="text-slate-100 font-mono">{r.value}</span>
                </div>
              ))}
            </div>

            <div className="flex gap-3 justify-center">
              <button onClick={() => navigate('/dashboard')} className="btn-secondary">
                Dashboard
              </button>
              {createdCaseId ? (
                <button onClick={() => navigate(`/cases/${createdCaseId}`)} className="btn-primary" data-testid="view-case-btn">
                  View Case →
                </button>
              ) : (
                <button onClick={() => navigate('/cases')} className="btn-primary" data-testid="view-cases-btn">
                  View All Cases →
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

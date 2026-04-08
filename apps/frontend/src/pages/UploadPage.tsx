import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, FileText, ChevronRight, Check, AlertCircle, X, Plus } from 'lucide-react'
import { studiesApi } from '../lib/api'
import { Spinner } from '../components/common/LoadingOverlay'

type Step = 1 | 2 | 3 | 4

const STEPS = [
  { n: 1 as Step, label: 'Upload DICOM' },
  { n: 2 as Step, label: 'Review Metadata' },
  { n: 3 as Step, label: 'Case Setup' },
  { n: 4 as Step, label: 'Processing' },
]

const MOCK_DICOM_TAGS = [
  { tag: '(0010,0020)', name: 'Patient ID', value: 'Anonymized — FA-2024-XXXX', sensitive: false },
  { tag: '(0008,0060)', name: 'Modality', value: 'CT', sensitive: false },
  { tag: '(0008,103E)', name: 'Series Description', value: 'CT Maxillofacial w/o contrast', sensitive: false },
  { tag: '(0018,0050)', name: 'Slice Thickness', value: '0.625 mm', sensitive: false },
  { tag: '(0028,0030)', name: 'Pixel Spacing', value: '0.488 \\ 0.488 mm', sensitive: false },
  { tag: '(0028,0010)', name: 'Rows', value: '512', sensitive: false },
  { tag: '(0028,0011)', name: 'Columns', value: '512', sensitive: false },
  { tag: '(0020,0013)', name: 'Instance Number', value: '1 – 412 (412 slices)', sensitive: false },
  { tag: '(0018,0080)', name: 'kVp', value: '120 kV', sensitive: false },
  { tag: '(0018,1152)', name: 'Exposure', value: '280 mAs', sensitive: false },
  { tag: '(0008,0022)', name: 'Acquisition Date', value: '2024-11-15', sensitive: false },
  { tag: '(0008,0023)', name: 'Content Date', value: '2024-11-15', sensitive: false },
  { tag: '(0010,0030)', name: 'Patient Birth Date', value: '⚠ STRIPPED — Not stored', sensitive: true },
  { tag: '(0010,0010)', name: 'Patient Name', value: '⚠ STRIPPED — Anonymized', sensitive: true },
  { tag: '(0008,0090)', name: 'Referring Physician', value: 'Dr. Chen', sensitive: false },
  { tag: '(0008,0080)', name: 'Institution Name', value: 'Metro General Hospital', sensitive: false },
]

export default function UploadPage() {
  const navigate = useNavigate()
  const [step, setStep] = useState<Step>(1)
  const [files, setFiles] = useState<File[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [caseType, setCaseType] = useState('mandible_fracture')
  const [notes, setNotes] = useState('')
  const [surgeon, setSurgeon] = useState('Dr. Emily Chen')
  const [priority, setPriority] = useState('routine')
  const fileInputRef = useRef<HTMLInputElement>(null)

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

  const handleUpload = async () => {
    if (files.length === 0) return
    setIsUploading(true)
    setUploadError(null)
    try {
      const result = await studiesApi.upload(files[0], setUploadProgress)
      setJobId(result.jobId)
      setStep(2)
    } catch (err) {
      setUploadError('Upload failed. Please check your connection and try again.')
    } finally {
      setIsUploading(false)
    }
  }

  const handleCreateCase = async () => {
    setStep(4)
    // Simulate processing initiated
    await new Promise(r => setTimeout(r, 1000))
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

          {/* Upload progress */}
          {isUploading && (
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-4" data-testid="upload-progress">
              <div className="flex items-center gap-3 mb-2">
                <Spinner size={16} />
                <span className="text-sm text-slate-300">Uploading and validating DICOM data...</span>
              </div>
              <div className="h-2 bg-slate-700 rounded-full">
                <div
                  className="h-full bg-cyan-500 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="text-xs text-slate-500 mt-1 font-mono text-right">{Math.round(uploadProgress)}%</p>
            </div>
          )}

          {uploadError && (
            <div className="flex items-center gap-2 p-3 bg-red-950 border border-red-800 rounded-lg text-sm text-red-400" data-testid="upload-error">
              <AlertCircle size={15} />
              {uploadError}
            </div>
          )}

          {/* DEMO mode: allow proceeding with no files */}
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
              <h3 className="panel-title">Parsed DICOM Tags</h3>
              <span className="text-2xs font-mono text-slate-500">Job ID: {jobId}</span>
            </div>
            <div className="overflow-x-auto">
              <table className="data-table w-full">
                <thead>
                  <tr>
                    <th className="w-36">Tag</th>
                    <th>Attribute Name</th>
                    <th>Value</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {MOCK_DICOM_TAGS.map(tag => (
                    <tr key={tag.tag} data-testid={`dicom-tag-${tag.tag}`}>
                      <td><span className="font-mono text-xs text-slate-400">{tag.tag}</span></td>
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
                <label className="label-sm block mb-1.5">Priority *</label>
                <select
                  value={priority}
                  onChange={e => setPriority(e.target.value)}
                  className="select-base"
                  data-testid="priority-select"
                >
                  <option value="routine">Routine</option>
                  <option value="urgent">Urgent</option>
                  <option value="stat">STAT</option>
                </select>
              </div>

              <div>
                <label className="label-sm block mb-1.5">Primary Surgeon *</label>
                <select
                  value={surgeon}
                  onChange={e => setSurgeon(e.target.value)}
                  className="select-base"
                  data-testid="surgeon-select"
                >
                  <option>Dr. Emily Chen</option>
                  <option>Dr. Marcus Reid</option>
                  <option>Dr. Aisha Okonkwo</option>
                </select>
              </div>

              <div>
                <label className="label-sm block mb-1.5">Reviewer</label>
                <select className="select-base" data-testid="reviewer-select">
                  <option>Dr. Marcus Reid</option>
                  <option>Dr. Emily Chen</option>
                  <option>Dr. Aisha Okonkwo</option>
                </select>
              </div>
            </div>

            <div>
              <label className="label-sm block mb-1.5">Scheduled Surgery Date</label>
              <input type="datetime-local" className="input-base w-full max-w-xs" data-testid="scheduled-date" />
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
                { label: 'Case Number', value: 'FA-2024-0062' },
                { label: 'Status', value: 'Segmentation Queued' },
                { label: 'Primary Surgeon', value: surgeon },
                { label: 'Case Type', value: caseType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) },
                { label: 'Queue Position', value: '#3' },
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
              <button onClick={() => navigate('/cases')} className="btn-primary" data-testid="view-cases-btn">
                View All Cases →
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

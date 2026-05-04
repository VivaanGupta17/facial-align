import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'

import { caseStudiesApi, casesApi, segmentationApi, studiesApi } from '../lib/api'
import { useToastStore } from '../stores/toastStore'
import UploadPage from './UploadPage'

describe('UploadPage', () => {
  it('requires a patient MRN before upload begins', async () => {
    render(
      <MemoryRouter>
        <UploadPage />
      </MemoryRouter>
    )

    const file = new File(['dicom'], 'scan.dcm', { type: 'application/dicom' })
    const fileInput = screen.getByTestId('file-input') as HTMLInputElement
    fireEvent.change(fileInput, { target: { files: [file] } })

    expect(screen.getByTestId('upload-btn')).toBeDisabled()
    expect(screen.getByText('Patient MRN required')).toBeInTheDocument()
  })

  it('uploads a study, loads metadata, and advances into case setup', async () => {
    vi.spyOn(studiesApi, 'upload').mockResolvedValue({
      jobId: 'job-001',
      studyId: 'study-001',
      patientId: 'patient-001',
    })
    vi.spyOn(studiesApi, 'getMetadata').mockResolvedValue({
      studyDescription: 'Head CT',
      institutionName: 'Test Hospital',
    })

    render(
      <MemoryRouter>
        <UploadPage />
      </MemoryRouter>
    )

    const file = new File(['dicom'], 'scan.dcm', { type: 'application/dicom' })
    const fileInput = screen.getByTestId('file-input') as HTMLInputElement
    fireEvent.change(fileInput, { target: { files: [file] } })
    fireEvent.change(screen.getByTestId('patient-mrn-input'), { target: { value: 'MRN-001' } })

    fireEvent.click(screen.getByTestId('upload-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('step-2')).toBeInTheDocument()
    })
    expect(screen.getByText(/STRIPPED|Head CT/)).toBeInTheDocument()
  })

  it('creates a case and queues segmentation from the uploaded study', async () => {
    useToastStore.setState({ toasts: [] })
    vi.spyOn(studiesApi, 'upload').mockResolvedValue({
      jobId: 'job-001',
      studyId: 'study-001',
      patientId: 'patient-001',
    })
    vi.spyOn(studiesApi, 'getMetadata').mockResolvedValue({ studyDescription: 'Head CT' })
    vi.spyOn(casesApi, 'create').mockResolvedValue({
      id: 'case-001',
      caseNumber: 'FA-2026-CASE01',
      patientId: 'patient-001',
      studyId: 'study-001',
      caseType: 'mandible_fracture',
      status: 'processing',
      surgeonId: 'surgeon-001',
      reviewerId: null,
      fractureClassification: null,
      plannedProcedure: null,
      diagnosisCodes: null,
      targetSurgeryDate: null,
      teamIds: [],
      currentTaskId: null,
      lastError: null,
      createdAt: '2026-05-03T12:00:00.000Z',
      updatedAt: '2026-05-03T12:00:00.000Z',
      approvedAt: null,
      createdBy: 'surgeon-001',
      latestSegmentation: null,
      latestPlan: null,
      segmentationCount: 0,
      planCount: 0,
      allowedTransitions: [],
      studies: [],
    })
    vi.spyOn(caseStudiesApi, 'list').mockResolvedValue([])
    vi.spyOn(segmentationApi, 'submitJob').mockResolvedValue({
      jobId: 'seg-job-001',
      segmentationId: 'seg-001',
    })

    render(
      <MemoryRouter>
        <UploadPage />
      </MemoryRouter>
    )

    const file = new File(['dicom'], 'scan.dcm', { type: 'application/dicom' })
    const fileInput = screen.getByTestId('file-input') as HTMLInputElement
    fireEvent.change(fileInput, { target: { files: [file] } })
    fireEvent.change(screen.getByTestId('patient-mrn-input'), { target: { value: 'MRN-001' } })
    fireEvent.click(screen.getByTestId('upload-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('step-2')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId('step2-next'))
    await waitFor(() => {
      expect(screen.getByTestId('step-3')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId('create-case-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('step-4')).toBeInTheDocument()
    })
    expect(await screen.findByText('FA-2026-CASE01')).toBeInTheDocument()
  })
})

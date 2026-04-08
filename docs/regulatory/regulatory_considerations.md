# Regulatory and Privacy Considerations — Facial Align

**Version:** 1.0  
**Status:** Research-phase documentation — not a legal opinion  
**Audience:** Founding team, clinical collaborators, future regulatory counsel  
**Last Updated:** 2025

> **Disclaimer:** This document is an engineering and product planning reference. It does not constitute legal or regulatory advice. All regulatory submissions must involve qualified regulatory counsel and a Regulatory Affairs professional.

---

## Table of Contents

1. [FDA SaMD Classification](#1-fda-samd-classification)
2. [Intended Use Statement](#2-intended-use-statement)
3. [Predicate Devices](#3-predicate-devices)
4. [HIPAA Technical Safeguards](#4-hipaa-technical-safeguards)
5. [De-identification Requirements](#5-de-identification-requirements)
6. [Audit Logging Requirements](#6-audit-logging-requirements)
7. [Data Retention Policies](#7-data-retention-policies)
8. [Business Associate Agreement Considerations](#8-business-associate-agreement-considerations)
9. [International Regulatory Notes](#9-international-regulatory-notes)
10. [Regulatory Readiness Roadmap](#10-regulatory-readiness-roadmap)

---

## 1. FDA SaMD Classification

### Classification Framework

Facial Align is Software as a Medical Device (SaMD) under the FDA's Digital Health Center of Excellence framework. The classification follows the International Medical Device Regulators Forum (IMDRF) SaMD Risk Framework and the FDA's corresponding guidance.

**IMDRF SaMD Risk Matrix:**

| Healthcare situation | Critical | Serious | Non-serious |
|---------------------|---------|---------|------------|
| Treat / diagnose | IV | III | II |
| Drive clinical management | III | II | I |
| Inform clinical management | II | I | I |

**Facial Align's classification:**

- **Healthcare situation:** "Serious" — Craniofacial surgical planning directly affects outcomes in a surgical procedure with irreversible anatomical consequences
- **Purpose:** "Drive clinical management" — The system produces a surgical plan that the surgeon uses to execute the procedure
- **Result:** **IMDRF Category III** → FDA **Class II**, 510(k) pathway

**Critical distinction:** The system stays Class II (not Class III) because it operates with **physician-in-the-loop** (PITL). The software produces a plan that the surgeon reviews and explicitly approves. No autonomous action is taken on the patient. If the design were changed to autonomously execute any aspect of surgery (robotics integration without surgeon override capability), classification would likely shift to Class III.

### FDA Device Class

- **Class II** — Moderate risk; requires 510(k) premarket notification
- **Regulation:** 21 CFR Part 892 (Radiology Devices), Subpart F — Therapeutic Radiation Dosimetry Systems
  - More precisely: surgical planning software typically clears under **892.2050** (Image-Intensified Fluoroscopic X-Ray System) or under a newer SaMD-specific classification
  - Exact product code determined by predicate selection and Pre-Sub meeting with FDA
- **Exempt status:** FDA's "clinical decision support" exemption (21st Century Cures Act) does NOT apply because the software performs analysis of medical image data and does NOT allow independent review of the underlying data by the clinician before acting on the recommendation

### 510(k) Pathway

**Standard 510(k):** Demonstrate substantial equivalence to a predicate device in intended use and technological characteristics. Expected timeline: 12–18 months from submission to clearance.

**De Novo pathway (alternative):** If no suitable predicate exists, a De Novo classification request creates a new device type. Higher burden but creates a cleared predicate for competitors.

**Pre-Submission (Q-Sub) process (recommended):**
1. Prepare Pre-Sub package: draft intended use, proposed predicate, proposed test plan
2. Submit to FDA for written feedback (FDA responds within 90 days)
3. Receive feedback on predicate acceptability and clinical testing requirements
4. Incorporate feedback before starting formal 510(k) preparation

**Recommended timing:** Q-Sub should occur during Phase 2 (months 12–24), before finalizing the clinical validation study design.

---

## 2. Intended Use Statement

The intended use statement is the most critical regulatory document — it defines scope, user population, and clinical context, and it constrains every other regulatory requirement.

**Draft Intended Use Statement (v0.1 — for internal planning purposes only):**

> Facial Align is a software system intended to assist qualified craniomaxillofacial surgeons in preoperative virtual surgical planning for craniomaxillofacial procedures. The system processes preoperative CT imaging data to produce: (1) segmented 3D anatomical models of craniofacial skeletal structures; (2) automated identification of cephalometric landmarks; and (3) candidate surgical plans consisting of proposed bone segment movement vectors. All system outputs are intended for review, modification, and approval by the treating surgeon prior to use in clinical decision-making. The system is not intended for autonomous surgical planning, intraoperative guidance, or use without physician oversight.

**Intended use must exclude:**
- Real-time intraoperative guidance (this would likely require separate clearance)
- Pediatric patients under 12 years of age (until validated in this population)
- Pathological conditions outside the training distribution (active infection, metabolic bone disease affecting density calibration)
- Standalone use without surgeon review (the PITL requirement is structural, not optional)

**Indications for Use (draft):**

> Facial Align is indicated for use by qualified craniomaxillofacial surgeons as a preoperative planning aid for:
> - Mandibular and maxillofacial fracture reconstruction
> - Orthognathic surgery (LeFort I, bilateral sagittal split osteotomy, genioplasty)
> - Mandibular reconstruction following tumor ablation
> For patients aged 18 years and older with CT imaging of diagnostic quality (slice thickness ≤ 1.5mm).

---

## 3. Predicate Devices

A predicate is a legally marketed device to which substantial equivalence must be demonstrated. The predicate selection strategy determines the regulatory path and the required evidence package.

### Primary Predicate Candidates

**3D Systems VSP Orthognathics** (K152010 and subsequent modifications)
- Cleared indication: Computer-aided design and manufacturing of surgical guides and splints for orthognathic surgery
- Relevance: Same intended use (VSP for CMF surgery), same image input type (CT DICOM)
- Difference: 3D Systems VSP is a service + software combination; Facial Align is pure software
- Predicate strength: Strong for orthognathic use case; weaker for trauma

**Materialise ProPlan CMF** (K203038 or K213049)
- Cleared indication: Software for virtual surgical planning in CMF and orthopedic surgery
- Relevance: Direct software-only predicate; covers CMF indication
- Predicate strength: Strong — covers intended use and technological characteristics

**Brainlab Elements for CMF** (if cleared — check current status)
- Cleared indication: Surgical planning and navigation for CMF
- Relevance: Covers CMF planning; adds navigation functionality
- Predicate strength: Good for core planning functions; navigation components not relevant to Facial Align Phase 1–2

### Split Predicate Strategy

If no single predicate covers all intended uses, FDA permits a split predicate: one predicate establishes intended use equivalence; a second establishes technological characteristics equivalence. Example:
- Predicate A (Materialise ProPlan CMF): establishes intended use equivalence for VSP
- Predicate B (an AI-based segmentation tool, e.g., Imbio Lung Density Analysis): establishes equivalence for AI-driven image analysis with PITL design

### Predicate Selection Considerations

| Factor | Recommendation |
|--------|---------------|
| Scope | Select a predicate whose cleared indication matches or contains Facial Align's intended use |
| Technology | Prefer a software-only predicate over a combined hardware+software predicate |
| AI | If predicate is non-AI, FDA may require additional testing for the AI component per AI/ML guidance |
| Failure modes | Predicate's failure mode documentation informs Facial Align's risk analysis |

---

## 4. HIPAA Technical Safeguards

HIPAA's Security Rule (45 CFR §§ 164.302–164.318) requires covered entities and business associates to implement technical safeguards for electronic Protected Health Information (ePHI). When Facial Align is used in a clinical setting, it processes ePHI and must comply.

### Access Controls (§ 164.312(a)(1))

| Requirement | Implementation |
|-------------|---------------|
| Unique user identification | Each user has a unique UUID; no shared accounts |
| Emergency access procedure | Admin recovery procedure documented; break-glass logs maintained |
| Automatic logoff | 30-minute session timeout; configurable per institution |
| Encryption and decryption | All ePHI encrypted at rest (AES-256) and in transit (TLS 1.3) |

### Audit Controls (§ 164.312(b))

- All ePHI access logged to append-only `audit_log` table
- Log entries include: timestamp, user_id, action type, resource type/ID, IP address, user agent
- Logs cannot be modified or deleted by application code (Postgres row-level security enforced)
- Log retention: minimum 6 years (HIPAA requirement)
- Log monitoring: alerts on anomalous access patterns (high-volume reads, off-hours access)

### Integrity Controls (§ 164.312(c)(1))

- All MinIO objects have SHA-256 checksums stored at write time and verified at read
- Database transactions use row-level versioning (updated_at, version columns)
- NIfTI and STL files include embedded hash in metadata sidecar

### Transmission Security (§ 164.312(e)(1))

- TLS 1.3 enforced on all external connections; TLS 1.2 minimum internal
- No ePHI transmitted via email, SMS, or unencrypted channels
- MinIO presigned URLs include HTTPS enforcement; expiry 15 minutes
- No ePHI in URL query parameters, HTTP headers, or server logs

### Workforce Controls (non-technical, but platform-supported)

- Authentication logs available for HR/compliance review
- Role assignment requires admin action (no self-elevation)
- Account deprovisioning: immediate session invalidation on role change or termination

---

## 5. De-identification Requirements

HIPAA permits two methods for de-identification:

**Method 1 — Expert Determination:** A qualified expert certifies using statistical or scientific principles that re-identification risk is very small.

**Method 2 — Safe Harbor:** Remove all 18 specified identifiers. This is the method Facial Align implements.

### The 18 HIPAA Safe Harbor Identifiers to Remove

| # | Identifier | DICOM Tag | Facial Align Action |
|---|-----------|-----------|---------------------|
| 1 | Names | PatientName, ReferringPhysicianName | Remove / replace with hash |
| 2 | Geographic subdivisions | PatientAddress, Institution | Remove |
| 3 | Dates (except year) | StudyDate, SeriesDate, BirthDate | Date-shift by random ±1–365 days per patient |
| 4 | Phone numbers | PhoneNumber (if present) | Remove |
| 5 | Fax numbers | — | Remove all contact fields |
| 6 | Email addresses | — | Remove |
| 7 | Social security numbers | — | Remove |
| 8 | Medical record numbers | PatientID | Hash (SHA-256 with salt) |
| 9 | Health plan beneficiary numbers | — | Remove |
| 10 | Account numbers | AccessionNumber | Hash |
| 11 | Certificate/license numbers | — | Remove |
| 12 | Vehicle identifiers | — | Not in DICOM; N/A |
| 13 | Device identifiers | DeviceSerialNumber | Remove |
| 14 | Web URLs | — | Remove |
| 15 | IP addresses | — | Not in DICOM; N/A |
| 16 | Biometric identifiers | — | Check pixel data for face images |
| 17 | Full-face photographs | PixelData (secondary captures) | Detect and redact or exclude |
| 18 | Unique identifying numbers | StudyInstanceUID, SeriesInstanceUID | Re-generate (preserve internal mapping) |

### Implementation

```python
# De-identification recipe (pydicom/deid)
# apps/backend/app/services/dicom/deidentify.py

DEID_RECIPE = """
FORMAT DICOM

%header

REMOVE PatientName
REPLACE PatientID hash
REMOVE PatientBirthDate
REPLACE StudyDate date_shift
REPLACE SeriesDate date_shift
REMOVE InstitutionName
REMOVE ReferringPhysicianName
REPLACE AccessionNumber hash
REPLACE StudyInstanceUID duid    # deterministic UID from internal case UUID
REPLACE SeriesInstanceUID duid
REPLACE SOPInstanceUID duid
REMOVE ALL_PRIVATE
"""
```

### Date Shifting

Dates are shifted by a consistent offset per patient (so relative timing between studies is preserved):
- Offset: random integer in range [−365, +365] days
- Offset is stored in encrypted `DateShiftConfig` table (keyed by patient internal UUID)
- Age preservation: year of birth is preserved; month/day removed (only birth year stored)

### Pixel Data PHI Detection

Standard header de-identification does not address burned-in text in pixel data (common in scout images, secondary captures, and some CBCT systems). Facial Align implements:
1. Check for Secondary Capture SOP class (these frequently contain burned-in text)
2. Run vision model detection on secondary captures and scout series
3. Apply pixel blackout to detected text regions before storage
4. Flag series for manual review if detection confidence is low

### Verification Protocol

Before any de-identified dataset is used for model training or research:
1. Run DICOM PS 3.15 Annex E compliance verification script
2. Inspect 5% random sample of de-identified files for residual PHI
3. Run name/date detection regex over all text-type tags
4. Log verification outcome to `DatasetVerification` table

---

## 6. Audit Logging Requirements

### What Must Be Logged

Per HIPAA §164.312(b) and standard security best practices, every event that involves ePHI access, creation, modification, or deletion must generate an audit entry.

**Mandatory audit events:**

| Category | Events |
|---------|--------|
| Authentication | Login success, login failure, logout, token refresh, session timeout |
| Case lifecycle | Case created, case viewed, case status changed, case archived |
| PHI access | Patient record viewed, DICOM study downloaded, de-identified export |
| Plan lifecycle | Plan candidate generated, plan reviewed, plan modified, plan approved, plan rejected |
| Segmentation | Segmentation output viewed, segmentation corrected, segmentation accepted |
| User management | User created, role changed, user deactivated |
| Export | STL export, PDF report generated, navigation DICOM generated |
| Administrative | Model version promoted, system configuration changed |

### Log Entry Schema

```json
{
  "id": "bigint (auto-increment)",
  "timestamp": "2025-03-15T14:22:31.441Z",
  "user_id": "uuid or null (system events)",
  "user_role": "SURGEON",
  "action": "PLAN_APPROVED",
  "resource_type": "surgical_plan",
  "resource_id": "uuid",
  "case_id": "uuid (if applicable)",
  "ip_address": "192.168.1.x",
  "user_agent": "Mozilla/5.0...",
  "session_id": "uuid",
  "details": {
    "plan_version": 2,
    "confidence": 0.91,
    "modified_from_ai": true,
    "modification_count": 3
  },
  "success": true,
  "error_code": null
}
```

### Log Storage and Integrity

- **Database:** PostgreSQL `audit_log` table, range-partitioned by month
- **Retention:** 6 years minimum (HIPAA); 10 years recommended (aligns with medical record retention)
- **Immutability:** Table grants: INSERT only for application role; no UPDATE or DELETE
- **Backup:** Audit logs are included in daily database backups; separate backup stream with 90-day retention
- **Forwarding:** In Phase 2+, forward to SIEM (Splunk, Datadog) for alerting and compliance reporting
- **Integrity verification:** Weekly hash chain verification of audit log entries

---

## 7. Data Retention Policies

### DICOM and Clinical Data

| Data Type | Minimum Retention | Notes |
|-----------|------------------|-------|
| Raw DICOM studies | 10 years | Aligns with HIPAA 6-year + medical record standards |
| De-identified training data | Indefinite | No PHI; no retention obligation; valuable for model training |
| Surgical planning reports (PDF) | 10 years | Part of medical record |
| Mesh files (STL/GLTF) | 10 years or case lifetime | Associated with surgical plan record |
| Post-operative comparison data | 10 years | Part of outcome tracking record |

### Operational Data

| Data Type | Retention | Notes |
|-----------|-----------|-------|
| Audit logs | 6 years minimum | HIPAA §164.316(b)(2) |
| Application logs | 2 years | Operational debugging; no PHI |
| Job results (Redis) | 24 hours | Short-lived; results in Postgres are permanent |
| Session tokens | Until expiry | Revoked tokens kept in revocation list 30 days |
| Backup data | 30 days (daily), 1 year (monthly), 7 years (annual) | |

### Patient Data Deletion (Right to Erasure)

HIPAA does not require data deletion on patient request (unlike GDPR). However, if operating in the EU or under GDPR:
- Implement `case.deletion_requested` flag
- De-identification (already implemented) satisfies GDPR erasure for research data
- Personal information in the `patients` table can be zeroed on request while preserving case structure

---

## 8. Business Associate Agreement Considerations

A **Business Associate Agreement (BAA)** is required when Facial Align processes ePHI on behalf of a HIPAA-covered entity (hospital, clinic). When the platform is deployed as a cloud service, the BAA chain includes:

```
Covered Entity (hospital)
    └── Business Associate Agreement with Facial Align (software vendor)
            └── BAA with Cloud Provider (AWS/GCP/Azure) — sign before launch
            └── BAA with any sub-processors (email service, analytics, monitoring)
```

### Cloud Provider BAA

AWS, GCP, and Azure all offer HIPAA BAA coverage for specific services. Key points:
- **AWS:** Signs BAA for HIPAA-eligible services (S3, RDS, EC2, EKS, CloudTrail). Not all AWS services are covered — review the HIPAA Eligible Services Reference before adding services.
- **GCP:** HIPAA BAA available; Google Workspace included if used for internal communication
- **Azure:** HIPAA BAA for Azure services; Defender for Cloud provides compliance reporting

**Action required before Phase 2 clinical deployment:** Execute BAA with chosen cloud provider before any ePHI enters the environment.

### BAA Template for Clinical Partners

When a hospital or academic medical center uses Facial Align:
1. Hospital signs Facial Align's standard BAA
2. BAA specifies: permitted uses of ePHI, safeguards in place, breach notification procedures (60 days), data destruction on termination
3. BAA must be executed before any test cases containing real patient data

### Research Exception

IRB-approved research with a waiver of HIPAA authorization may allow use of de-identified data without a formal BAA. Confirm with institutional legal counsel. Facial Align's de-identification pipeline (Safe Harbor compliant) supports this research pathway.

---

## 9. International Regulatory Notes

### European Union — MDR / CE Marking

Under the EU Medical Device Regulation (MDR) 2017/745:
- **Classification:** Rule 11 — Software intended to provide information used to make decisions with diagnosis or therapeutic purposes: **Class IIb** (higher than FDA Class II equivalent)
- **Conformity assessment:** Class IIb requires involvement of a Notified Body (independent third-party auditor)
- **Key standards:**
  - IEC 62304 (Software lifecycle processes) — required
  - IEC 62366 (Usability engineering) — required
  - ISO 14971 (Risk management) — required
  - ISO 13485 (Quality management system) — required
- **IVDR:** Does not apply (Facial Align is not an in vitro diagnostic device)
- **Timeline:** CE marking via MDR is typically 18–36 months; significantly longer than FDA

**Strategic recommendation:** Pursue FDA 510(k) first (Class II pathway, lower burden); use FDA clearance as evidence in EU MDR submission.

### United Kingdom — UKCA

Post-Brexit, UK uses its own UKCA marking process (mirroring EU MDR with some differences). FDA and CE marking do not automatically transfer. UK is not a Phase 1 or 2 priority.

### Canada — Health Canada (MDEL / MDSAP)

- **Classification:** Class III Medical Device (software with significant risk)
- **Pathway:** Premarket review required; Health Canada typically defers to FDA clearance evidence
- **MDSAP:** Medical Device Single Audit Program — a MDSAP audit certificate is recognized by Health Canada, FDA (for inspections), and ANVISA (Brazil)

### Australia — TGA

- **Classification:** Likely Class IIb under AIMD Regulations
- **Pathway:** TGA accepts FDA 510(k) and CE marking as supporting evidence, but requires its own application
- **Phase 3 priority**

### Japan — PMDA

- **Classification:** Controlled medical device (class II or III depending on AI interpretation)
- **Pathway:** Requires Japanese PMDA application; significant translation burden
- **Phase 3 priority; consider partnership with Japanese distributor**

---

## 10. Regulatory Readiness Roadmap

### Phase 1 (Current — Research)

- [ ] IRB approval at primary academic partner institution
- [ ] Data use agreement (DUA) for retrospective case data
- [ ] De-identification pipeline implemented and verified
- [ ] Begin SaMD documentation practice (design history file structure)
- [ ] Identify regulatory counsel

### Phase 2 (Clinical Research Platform)

- [ ] Execute BAA with cloud provider (before first clinical case)
- [ ] Execute BAAs with all clinical site partners
- [ ] Submit Pre-Submission Q-Sub to FDA (request meeting to discuss predicate and testing plan)
- [ ] Begin IEC 62304 software lifecycle documentation
- [ ] Begin ISO 14971 risk management file
- [ ] Preregistered clinical validation study protocol (IDE not required for 510(k) exempt studies)
- [ ] 510(k) predicate identified and documented

### Phase 3 (Commercial / Regulatory Submission)

- [ ] 510(k) submission package complete
- [ ] Clinical validation study data enrolled (≥ 50 cases with postop follow-up)
- [ ] Predetermined Change Control Plan (PCCP) drafted and filed with 510(k)
- [ ] Post-market surveillance plan defined
- [ ] Begin EU MDR pre-submission discussions with Notified Body

### Documents Required for 510(k) Submission

| Document | Standard | Status |
|---------|---------|--------|
| 510(k) summary | FDA guidance | Phase 3 |
| Device description | — | Phase 3 |
| Substantial equivalence comparison | — | Phase 3 |
| Performance testing summary | FDA AI/ML guidance | Phase 3 |
| Software documentation (IEC 62304) | IEC 62304 | Phase 2 start |
| Risk management file (ISO 14971) | ISO 14971 | Phase 2 start |
| Usability study (IEC 62366) | IEC 62366 | Phase 2 |
| Cybersecurity documentation | FDA 2023 guidance | Phase 2 |
| Clinical validation data | FDA De Novo / 510(k) guidance | Phase 2–3 |
| Labeling | 21 CFR 801 | Phase 3 |
| PCCP (for AI updates) | FDA PCCP guidance | Phase 3 |

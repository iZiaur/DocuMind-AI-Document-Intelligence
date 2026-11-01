"""
Synthetic documents for DocuMind AI Stage 5 Evaluation.
These documents represent realistic enterprise policies and engineering guidelines
created purely for automated benchmarking without any proprietary or PII data.
"""

from typing import Dict, List, Any

SAMPLE_DOCUMENTS: Dict[str, Dict[str, Any]] = {
    "Information Security & Incident Response Policy": {
        "title": "Information Security & Incident Response Policy",
        "filename": "InfoSec_Incident_Response_Policy.pdf",
        "pages": [
            # Page 1: Severity classification & escalation
            (
                "DocuMind Information Security & Incident Response Policy. "
                "Section 1: Incident Severity Classification and Escalation Procedures. "
                "Security incidents are categorized into three severity levels. "
                "Critical Severity-1 incidents involve active data breaches, system compromise, or unauthorized administrative access. "
                "Severity-1 incidents require executive notification within 15 minutes of detection and activation of the executive war room. "
                "Major Severity-2 incidents involve service degradation or targeted phishing attempts; these require an incident lead assignment within 30 minutes. "
                "Moderate Severity-3 incidents involve isolated malware alerts or non-exploited configuration defects, requiring ticket triage within 4 business hours."
            ),
            # Page 2: Access control & passwords
            (
                "DocuMind Information Security & Incident Response Policy. "
                "Section 2: Access Control, Authentication, and Account Security. "
                "Multi-factor authentication (MFA) is strictly mandatory for all employees accessing corporate email, source code repositories, and production cloud infrastructure. "
                "User passwords must be at least 14 characters long, contain uppercase, lowercase, numbers, and symbols, and must be rotated every 90 days. "
                "Quarterly access reviews must be performed by department heads to revoke unnecessary permissions and offboard departed personnel. "
                "Privileged administrative accounts must never be shared and require session recording during production maintenance windows."
            ),
            # Page 3: Data retention & backups
            (
                "DocuMind Information Security & Incident Response Policy. "
                "Section 3: Data Retention, Cryptographic Protection, and Disaster Recovery. "
                "All customer data stored at rest must be encrypted using AES-256 GCM. Data in transit must use TLS 1.3. "
                "System audit logs and compliance records must be retained for exactly seven years in immutable write-once-read-many (WORM) cloud storage. "
                "Automated database snapshots and backups are taken daily and retained for 30 days. "
                "Comprehensive monthly database backup archives are retained for 1 year. Disaster recovery failover drills are conducted semiannually."
            ),
            # Page 4: Remediation & post-mortem
            (
                "DocuMind Information Security & Incident Response Policy. "
                "Section 4: Remediation, Root Cause Analysis, and External Disclosures. "
                "For all Severity-1 and Severity-2 incidents, the incident response commander must conduct a blameless post-mortem. "
                "The formalized Root Cause Analysis (RCA) document must be submitted to executive leadership and compliance within 72 hours of incident resolution. "
                "External disclosures, regulatory filings, or public press releases regarding any security incident strictly require prior written approval from the General Counsel. "
                "Identified security vulnerability remediation tickets must be patched within 14 calendar days for critical vulnerabilities."
            ),
        ],
    },
    "Corporate Travel & Expense Reimbursement Policy": {
        "title": "Corporate Travel & Expense Reimbursement Policy",
        "filename": "Corporate_Travel_Expense_Policy.pdf",
        "pages": [
            # Page 1: Flights & airfare
            (
                "DocuMind Corporate Travel & Expense Reimbursement Policy. "
                "Section 1: Airfare Booking Rules, Class of Service, and Advance Reservations. "
                "All domestic flights must be booked in economy class at least 14 days in advance of travel through the designated corporate booking platform. "
                "Business class travel is permitted only for international flights where the continuous scheduled flight duration exceeds 8 hours. "
                "Any single flight reservation or itinerary exceeding $1,500 total cost requires prior written approval from a Vice President before purchase. "
                "Flight cancellation fees and change fees are reimbursable only when resulting from documented business emergency cancellations."
            ),
            # Page 2: Meals & lodging
            (
                "DocuMind Corporate Travel & Expense Reimbursement Policy. "
                "Section 2: Daily Per Diem, Meals, Lodging, and Non-Reimbursable Items. "
                "The maximum allowable domestic per diem for meals is capped at $75 per day, broken down into $20 for breakfast, $25 for lunch, and $30 for dinner. "
                "Alcoholic beverages are strictly non-reimbursable under standard individual meal expenses. "
                "Hotel lodging rates are capped at $250 per night excluding taxes in standard regions, and capped at $350 per night in designated high-cost metropolitan cities (e.g., New York, San Francisco, London). "
                "Personal entertainment, hotel minibar charges, in-room movie rentals, and gym fees are not eligible for corporate reimbursement."
            ),
            # Page 3: Submission & mileage
            (
                "DocuMind Corporate Travel & Expense Reimbursement Policy. "
                "Section 3: Expense Report Submission Timelines, Receipts, and Mileage Reimbursement. "
                "All expense reports must be submitted through the finance portal within 30 calendar days of expense incurrence or travel completion. "
                "Itemized receipts showing date, vendor, item description, and total amount are mandatory for any expense exceeding $25. "
                "Business use of a personal vehicle is reimbursed at the standard rate of 67 cents per mile driven for business purposes. "
                "Verbal approvals are strictly prohibited for any expense claim regardless of amount; all approvals must be documented digitally in the portal."
            ),
        ],
    },
    "Engineering & Cloud Deployment Guide": {
        "title": "Engineering & Cloud Deployment Guide",
        "filename": "Engineering_Cloud_Deployment_Guide.pdf",
        "pages": [
            # Page 1: Staging & production deployment
            (
                "DocuMind Engineering & Cloud Deployment Guide. "
                "Section 1: Continuous Integration, Pull Request Guidelines, and Deployment Windows. "
                "Every pull request requires passing automated unit and integration tests and must receive at least two peer code reviews before merge. "
                "Merged code must deploy to the staging environment and undergo a mandatory staging soak time of at least 2 hours with zero error alerts before production rollout. "
                "Routine production deployments are scheduled Tuesday through Thursday between 9:00 AM and 3:00 PM. "
                "Production deployments are strictly prohibited on Fridays after 2:00 PM local time to protect weekend system availability."
            ),
            # Page 2: Database migrations
            (
                "DocuMind Engineering & Cloud Deployment Guide. "
                "Section 2: Database Schema Migrations, Rollback Procedures, and Lock Avoidance. "
                "All database schema migrations must be backwards-compatible with existing running service versions and execute in under 60 seconds. "
                "Table migrations must not hold exclusive table locks during active traffic hours. "
                "Destructive schema operations (such as dropping database columns or tables) require a two-sprint phased deprecation plan. "
                "Automated rollback scripts must be verified and tested in the staging environment prior to deploying schema migrations to production."
            ),
            # Page 3: Alerting & on-call
            (
                "DocuMind Engineering & Cloud Deployment Guide. "
                "Section 3: Observability, Service Level Objectives (SLOs), and On-Call Escalation. "
                "API service performance targets an SLO of 99.9% uptime. Latency alerts trigger when P99 response time exceeds 500ms for three consecutive 1-minute intervals. "
                "The designated primary on-call engineer must acknowledge high-priority production alerts within 10 minutes of notification. "
                "If a high-priority alert is not acknowledged within 15 minutes, the monitoring system automatically escalates notification to the secondary engineer and Engineering Manager. "
                "All incident timeline notes must be logged in the centralized incident management portal."
            ),
        ],
    },
}

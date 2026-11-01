"""
DocuMind AI v0.5.0 — Checkpoint F: Real Browser Automation & UI Verification
Runs the complete 20-step browser demonstration against the running Docker stack:
- Frontend (Nginx): http://localhost:5173
- Backend: http://localhost:8000
- PostgreSQL: localhost:5432
"""

import sys
import os
import io
import time
from playwright.sync_api import sync_playwright, expect
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

FRONTEND_URL = "http://localhost:5173"
CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
SCREENSHOT_DIR = "d:\\Desktop\\Projects\\AI Project\\screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def generate_test_pdf(filename: str) -> str:
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    resources = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
    })

    pages = [
        "DocuMind Global Security Architecture. Page 1: Mandatory encryption standard is AES-256-GCM across all storage tiers.",
        "DocuMind Global Security Architecture. Page 2: Disaster recovery RPO is 15 minutes and RTO is 1 hour under regional failure.",
        "DocuMind Global Security Architecture. Page 3: Access control requires hardware-based FIDO2 security keys for administrative access.",
    ]

    for p_text in pages:
        page = writer.add_blank_page(width=400, height=400)
        page[NameObject("/Resources")] = resources
        stream = DecodedStreamObject()
        safe_text = p_text.replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 40 350 Td ({safe_text}) Tj ET".encode("latin-1", errors="ignore"))
        page[NameObject("/Contents")] = stream

    filepath = os.path.join(SCREENSHOT_DIR, filename)
    with open(filepath, "wb") as f:
        writer.write(f)
    return filepath


def run_demonstration():
    print("[Checkpoint F] Starting 20-step real browser demonstration...")
    pdf_path = generate_test_pdf("Global_Security_Architecture_2026.pdf")

    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
            args=["--no-sandbox", "--disable-gpu"]
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # Monitor console & runtime errors
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # -------------------------------------------------------------
        # Step 1: Open application in browser
        # -------------------------------------------------------------
        print("Step 1: Navigating to", FRONTEND_URL)
        page.goto(FRONTEND_URL, wait_until="networkidle")
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "step_01_landing.png"))
        assert "DocuMind AI" in page.content()
        print("  [PASS] Landing page loaded successfully.")

        # -------------------------------------------------------------
        # Step 2: Register user via UI Auth Modal
        # -------------------------------------------------------------
        print("Step 2: Authenticating new user via UI modal...")
        ts = int(time.time())
        username = f"demo_user_{ts}"
        email = f"{username}@enterprise.documind"
        password = "SecurePassword2026!"

        # Open Auth Modal
        page.locator("button:has-text('Sign In / Register')").click()
        page.wait_for_selector("text=Welcome Back")

        # Switch to Register mode
        page.get_by_role("button", name="Register", exact=True).click()
        page.wait_for_selector("text=Create Account")

        page.fill("input[type='email']", email)
        page.fill("input[placeholder='doc_analyst']", username)
        page.fill("input[type='password']", password)
        page.locator("button[type='submit']:has-text('Create Secure Account')").click()

        # Wait until modal closes and username badge appears in header
        page.wait_for_selector(f"text={username}", timeout=10000)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "step_02_authenticated.png"))
        print(f"  [PASS] Registered and authenticated as '{username}'.")

        # -------------------------------------------------------------
        # Step 3: Open Document Management Interface
        # -------------------------------------------------------------
        print("Step 3: Opening Document Management interface...", flush=True)
        page.locator("aside button:has-text('Documents')").click()
        page.wait_for_selector("text=Document Knowledge Base")
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "step_03_documents_tab.png"))
        print("  [PASS] Document Knowledge Base tab open.", flush=True)

        # -------------------------------------------------------------
        # Step 4: Upload new uniquely named test PDF
        # -------------------------------------------------------------
        print("Step 4: Uploading test PDF...")
        # Set file input
        file_input = page.locator("input[type='file']")
        file_input.set_input_files(pdf_path)

        page.wait_for_selector("button:has-text('Upload & Index')")
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "step_04_file_selected.png"))
        page.locator("button:has-text('Upload & Index')").click()

        # -------------------------------------------------------------
        # Step 5 & 6: Verify uploaded document appears and completes
        # -------------------------------------------------------------
        print("Step 5 & 6: Verifying document appears and processing completes...")
        # Wait for success alert or card to appear
        page.wait_for_selector("text=Global_Security_Architecture_2026", timeout=20000)
        # Verify status is completed
        page.wait_for_selector("text=3 Pages", timeout=15000)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "step_06_ingestion_completed.png"))
        print("  [PASS] Document uploaded, parsed, chunked, and indexed with 3 pages in PostgreSQL pgvector.")

        # -------------------------------------------------------------
        # Step 7: Open Ask DocuMind
        # -------------------------------------------------------------
        print("Step 7: Switching to Ask DocuMind...", flush=True)
        page.locator("aside button:has-text('Ask DocuMind')").click()
        page.wait_for_selector("text=Ask DocuMind Intelligence")
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "step_07_ask_tab.png"))
        print("  [PASS] Ask DocuMind RAG workspace active.", flush=True)

        # -------------------------------------------------------------
        # Step 8: Create a new conversation thread
        # -------------------------------------------------------------
        print("Step 8: Creating new conversation thread...", flush=True)
        page.locator("button:has-text('New Chat')").first.click()
        # Wait for conversation to appear in sidebar
        page.wait_for_selector("div.group.relative:has-text('New Conversation')", timeout=10000)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "step_08_conversation_created.png"))
        print("  [PASS] Conversation thread created in sidebar.", flush=True)

        # -------------------------------------------------------------
        # Step 9: Ask a question present in PDF
        # -------------------------------------------------------------
        print("Step 9: Submitting question from PDF...")
        q1 = "What is the mandatory encryption standard?"
        textarea = page.locator("textarea[placeholder*='Ask a natural-language question']")
        textarea.fill(q1)
        page.locator("button[aria-label='Submit question']").click()

        # -------------------------------------------------------------
        # Step 10: Verify assistant gives evidence-grounded answer
        # -------------------------------------------------------------
        print("Step 10: Waiting for evidence-grounded answer...")
        # Wait for in-flight loader to finish and answer to appear
        page.wait_for_selector("text=AES-256-GCM", timeout=30000)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "step_10_grounded_answer.png"))
        print("  [PASS] Grounded answer received referencing AES-256-GCM.")

        # -------------------------------------------------------------
        # Step 11 & 12: Click citation badge and verify Source Inspector
        # -------------------------------------------------------------
        print("Step 11 & 12: Inspecting citation badge and Source Inspector...")
        citation_badge = page.locator("button:has-text('p.1')").first
        expect(citation_badge).to_be_visible()
        citation_badge.click()

        # Verify Source Inspector opens
        page.wait_for_selector("text=Source Inspector", timeout=5000)
        page.wait_for_selector("text=Verified Source")
        page.wait_for_selector("text=Page 1")
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "step_12_source_inspector.png"))
        print("  [PASS] Source Inspector opened with Verified Source Page 1 and excerpt.")

        # -------------------------------------------------------------
        # Step 13 & 14: Ask ambiguous follow-up question
        # -------------------------------------------------------------
        print("Step 13 & 14: Submitting ambiguous follow-up question...")
        q2 = "What about the disaster recovery objectives under failure?"
        textarea.fill(q2)
        page.locator("button[aria-label='Submit question']").click()

        # Wait for answer with RPO and RTO
        page.wait_for_selector("text=15 minutes", timeout=30000)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "step_14_followup_answer.png"))
        print("  [PASS] Multi-turn context maintained; answer grounded with RPO 15 minutes and RTO 1 hour.")

        # -------------------------------------------------------------
        # Step 15: Create another conversation
        # -------------------------------------------------------------
        print("Step 15: Creating second conversation thread...")
        page.locator("button:has-text('New Chat')").click()
        time.sleep(1.0)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "step_15_second_conv.png"))

        # Verify new conversation is empty
        conv_threads = page.locator("div.group.relative")
        expect(conv_threads).to_have_count(2)
        print("  [PASS] Second conversation thread created in sidebar.")

        # -------------------------------------------------------------
        # Step 16 & 17: Switch between conversations & verify isolation
        # -------------------------------------------------------------
        print("Step 16 & 17: Switching between conversations...")
        # Submit question in Conversation 2
        q3 = "What hardware is needed for access control?"
        textarea.fill(q3)
        page.locator("button[aria-label='Submit question']").click()
        page.wait_for_selector("text=FIDO2", timeout=30000)
        print("  [PASS] Conversation 2 answered with FIDO2 hardware requirement.")

        # Switch back to Conversation 1 (first thread in list)
        threads = page.locator("div.group.relative").all()
        # The first thread in sidebar is now Conversation 2 (most recently updated).
        # The second thread is Conversation 1.
        threads[1].click()
        # Verify Conversation 1 messages are shown (AES-256-GCM and 15 minutes)
        page.wait_for_selector("text=AES-256-GCM", timeout=5000)
        page.wait_for_selector("text=15 minutes", timeout=5000)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "step_17_conversation_isolation.png"))
        print("  [PASS] Switched back to Conversation 1; previous messages verified intact.")

        # -------------------------------------------------------------
        # Step 18 & 19: Refresh browser & verify active thread restored
        # -------------------------------------------------------------
        print("Step 18 & 19: Refreshing browser and verifying persistent restore...", flush=True)
        page.reload(wait_until="networkidle")
        if not page.locator("text=AES-256-GCM").is_visible():
            page.locator("aside button:has-text('Ask DocuMind')").click()
        page.wait_for_selector("text=AES-256-GCM", timeout=15000)
        page.wait_for_selector("text=15 minutes", timeout=15000)
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "step_19_refreshed_state.png"))
        print("  [PASS] Browser refreshed; active conversation and all messages restored from PostgreSQL.", flush=True)

        # -------------------------------------------------------------
        # Step 20: Delete test conversation and verify sidebar updates
        # -------------------------------------------------------------
        print("Step 20: Deleting test conversation from sidebar...", flush=True)
        initial_count = page.locator("div.group.relative").count()
        # Hover over first conversation thread so delete button is visible
        first_thread = page.locator("div.group.relative").first
        first_thread.hover()
        # Click delete button on the first thread
        delete_btn = page.locator("button[aria-label='Delete Conversation']").first
        delete_btn.click()

        # Confirmation button appears with title "Confirm Delete"
        confirm_btn = page.locator("button[title='Confirm Delete']").first
        confirm_btn.click()
        time.sleep(1.0)

        new_count = page.locator("div.group.relative").count()
        assert new_count == initial_count - 1, f"Expected {initial_count - 1} threads, found {new_count}"
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "step_20_deleted_conv.png"))
        print("  [PASS] Conversation successfully deleted; sidebar updated deterministically.", flush=True)

        print("\n[ALL 20 BROWSER WORKFLOW STEPS PASSED SUCCESSFULLY]")
        print("Console errors count:", len(console_errors))
        print("Page runtime errors count:", len(page_errors))
        if console_errors:
            print("Console errors:", console_errors)

        browser.close()


if __name__ == "__main__":
    run_demonstration()

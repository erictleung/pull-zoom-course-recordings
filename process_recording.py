import os
from pathlib import Path
import re
import sys
from typing import Optional
from playwright.sync_api import sync_playwright
from transformers import pipeline

# Directory to persist Zoom session cookies and login state
USER_DATA_DIR = Path(__file__).resolve().parent / "zoom_user_data"

# Hugging Face summarizer (lazy-loaded)
_summarizer = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        print("\nLoading Hugging Face summarization model (distilbart-cnn-12-6)...")
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return _summarizer


def clean_vtt(vtt_content: str) -> str:
    """Strips timestamps, cue metadata, and WebVTT headers."""
    cleaned = re.sub(r"^WEBVTT.*?\n\n", "", vtt_content, flags=re.DOTALL)
    cleaned = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}.*?\n", "", cleaned)
    cleaned = re.sub(r"^\d+\s*\n", "", cleaned, flags=re.MULTILINE)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return " ".join(lines)


def summarize_transcript(transcript: str, max_chunk_words: int = 700) -> str:
    """Chunks long transcripts and extracts key bullet points."""
    summarizer = get_summarizer()
    words = transcript.split()
    chunks = [" ".join(words[i : i + max_chunk_words]) for i in range(0, len(words), max_chunk_words)]

    print(f"Generating summary from {len(chunks)} text chunks...")
    bullets = []
    for chunk in chunks[:4]:
        res = summarizer(chunk, max_length=100, min_length=30, do_sample=False, truncation=True)
        bullets.append(f"• {res[0]['summary_text'].strip()}")

    return "\n".join(bullets)


def generate_course_announcement(
    topic: str,
    share_url: str,
    transcript: Optional[str],
) -> str:
    if transcript:
        summary = summarize_transcript(transcript)
        preview = transcript[:450] + ("..." if len(transcript) > 450 else "")
    else:
        summary = "• No transcript found or transcript is still processing."
        preview = "N/A"

    return f"""
==================== COPY BELOW THIS LINE ====================
Hi everyone,

The recording and transcript for "{topic}" are now available.

📌 Access Details:
• Recording & Materials: {share_url}

📝 Key Takeaways & Highlights:
{summary}

📄 Transcript Excerpt:
"{preview}"

Let me know if you run into any trouble accessing the materials!
==================== COPY ABOVE THIS LINE ====================
"""


def run_scraper():
    with sync_playwright() as p:
        # Launch persistent browser context (stores cookies/SSO state locally)
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(USER_DATA_DIR),
            headless=False,  # Set to False so you can see the UI and handle SSO
            args=["--start-maximized"],
            no_viewport=True,
        )
        page = context.new_page()

        print("Navigating to Zoom Cloud Recordings...")
        page.goto("https://zoom.us/recording")

        # Check if user needs to authenticate (SSO / Login page)
        if "signin" in page.url or "login" in page.url:
            print("\n*** ACTION REQUIRED ***")
            print("Please log into Zoom (via SSO or password) in the opened browser window.")
            print("Once you reach your Cloud Recordings page, come back here and press Enter.")
            input("Press Enter after you are logged in...")

        # Wait for recordings table to load
        page.wait_for_selector("table, .recording-list", timeout=30000)

        # Extract meeting rows from the recording dashboard
        rows = page.locator("tbody tr").all()
        if not rows:
            print("No recording rows found on the page.")
            context.close()
            return

        print("\n--- Recent Cloud Recordings ---")
        recordings = []
        for i, row in enumerate(rows[:8], start=1):
            text = row.inner_text().split("\n")
            topic = text[0].strip() if text else "Untitled Meeting"
            date_info = " | ".join(text[1:3]) if len(text) > 2 else "Recent"
            recordings.append({"row": row, "topic": topic})
            print(f"[{i}] {topic} ({date_info})")

        choice = input(f"\nSelect a recording (1-{len(recordings)}) or 'q' to quit: ").strip()
        if choice.lower() == "q" or not choice.isdigit() or not (1 <= int(choice) <= len(recordings)):
            context.close()
            return

        selected_item = recordings[int(choice) - 1]
        selected_row = selected_item["row"]
        topic_title = selected_item["topic"]

        # Click into the recording detail page
        link = selected_row.locator("a").first
        link.click()
        page.wait_for_load_state("networkidle")

        # Extract the Shareable URL
        share_url = "N/A"
        try:
            # Look for the share button or link field in the recording details view
            share_btn = page.locator("button:has-text('Copy shareable link'), button:has-text('Share')").first
            if share_btn.is_visible():
                share_btn.click()
                page.wait_for_timeout(500)
                # Some interfaces copy directly to clipboard, others display a dialog with an input
                dialog_input = page.locator("input[readonly], input.share-link").first
                if dialog_input.is_visible():
                    share_url = dialog_input.input_value()
                else:
                    share_url = page.url
        except Exception:
            share_url = page.url

        # Locate and download the Audio Transcript (.vtt) file
        raw_transcript = None
        vtt_candidate = page.locator("tr:has-text('Audio Transcript'), tr:has-text('Transcript')").first

        if vtt_candidate.is_visible():
            download_btn = vtt_candidate.locator("button, a").filter(has_text=re.compile(r"Download|Export", re.I)).first
            if download_btn.is_visible():
                print("Downloading transcript...")
                with page.expect_download(timeout=10000) as download_info:
                    download_btn.click()
                download = download_info.value
                dest_path = Path("temp_transcript.vtt")
                download.save_as(str(dest_path))

                if dest_path.exists():
                    raw_transcript = clean_vtt(dest_path.read_text(encoding="utf-8", errors="ignore"))
                    dest_path.unlink()  # Clean up temp file
        else:
            print("No standalone transcript download element detected on page.")

        context.close()

        # Build message and summarize
        announcement = generate_course_announcement(topic_title, share_url, raw_transcript)
        print(announcement)

        with open("latest_course_message.txt", "w", encoding="utf-8") as f:
            f.write(announcement)
        print("Saved announcement to latest_course_message.txt")


if __name__ == "__main__":
    run_scraper()

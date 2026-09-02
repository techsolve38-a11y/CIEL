"""
Gmail — read-only, metadata only (subject/sender/date), NOT full email
bodies. This is a deliberate privacy choice, same reasoning as Drive
using metadata-only scope: CIEL gets enough to be useful (recent senders,
subjects, timing) without ingesting full message content into memory.
Full content access is a real, separate upgrade if you want it later —
worth deciding deliberately, not defaulting into it.
"""

from google_auth import get_service


def get_gmail_service():
    return get_service("gmail", "v1")


def get_recent_messages(service, max_results=10):
    """Fetch recent message metadata only — format='metadata' asks Gmail
    to return headers (Subject, From, Date) without the message body."""
    results = service.users().messages().list(
        userId="me", maxResults=max_results, labelIds=["INBOX"],
    ).execute()
    message_refs = results.get("messages", [])

    messages = []
    for ref in message_refs:
        msg = service.users().messages().get(
            userId="me", id=ref["id"], format="metadata",
            metadataHeaders=["Subject", "From", "Date"],
        ).execute()
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        messages.append({
            "id": msg["id"],
            "subject": headers.get("Subject", "(no subject)"),
            "from": headers.get("From", "(unknown sender)"),
            "date": headers.get("Date", ""),
            "snippet": msg.get("snippet", ""),
        })
    return messages


if __name__ == "__main__":
    service = get_gmail_service()
    messages = get_recent_messages(service, max_results=10)
    for m in messages:
        print(f"  [{m['date']}] {m['from']}: {m['subject']}")

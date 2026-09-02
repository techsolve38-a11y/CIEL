"""
Google Drive — metadata only (file names, types, modified dates), NOT
file content. Uses drive.metadata.readonly, a narrower scope than full
drive.readonly, on purpose — CIEL can see what files exist and when they
changed without being able to read their actual contents. Reading actual
document content would need a broader scope and separate, deliberate
decision later.
"""

from google_auth import get_service


def get_drive_service():
    return get_service("drive", "v3")


def get_recent_files(service, max_results=15):
    results = service.files().list(
        pageSize=max_results,
        orderBy="modifiedTime desc",
        fields="files(id, name, mimeType, modifiedTime)",
    ).execute()
    return results.get("files", [])


if __name__ == "__main__":
    service = get_drive_service()
    files = get_recent_files(service, max_results=15)
    for f in files:
        print(f"  [{f['modifiedTime']}] {f['name']} ({f['mimeType']})")

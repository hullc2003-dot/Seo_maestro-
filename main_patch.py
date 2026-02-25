# --- PATCH: Contact UI signaling ---
CONTACT_UI_URL = "http://localhost:8000/status"

async def signal_ui(status: str):
    """POST a simple status message to the UI endpoint."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(CONTACT_UI_URL, json={"status": status}) as resp:
                if resp.status != 200:
                    logger.error(f"[UI] Signal failed with HTTP {resp.status}")
    except Exception as exc:  # pragma: no cover
        logger.error(f"[UI] Exception while signaling UI: {exc}")

async def commit_and_notify(R, path, msg, content, client, headers):
    """
    Wrapper around the existing commit logic.
    After a successful save (response code 200/201) it notifies the UI.
    """
    # Call the original commit implementation (assumed to be `commit_file`).
    result = await commit_file(R, path, msg, content, client, headers)
    if isinstance(result, str) and result.startswith("Saved_"):
        await signal_ui("Operational milestone reached: file saved")
    return result
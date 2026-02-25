# --- PATCH: Intellectual Honesty Functions ---
def fn_5_chk(threshold: float = 0.8) -> bool:
    """
    Validate current alignment against the given threshold.
    Returns True if alignment >= threshold, else False.
    """
    try:
        # `alignment_score` should be maintained elsewhere in the agent.
        score = globals().get("alignment_score", 1.0)
        return float(score) >= threshold
    except Exception:
        logger.exception("[fn_5_chk] Alignment check failed")
        return False


def fn_7_mut(new_rule: str) -> None:
    """
    Mutate the agent's operational rules when alignment is insufficient.
    Appends `new_rule` to the global `operational_rules` list and notifies the UI.
    """
    global operational_rules
    if not isinstance(operational_rules, list):
        operational_rules = []
    operational_rules.append(new_rule)
    logger.info("[fn_7_mut] Added new rule: %s", new_rule)
    # Signal the UI that rules have been mutated
    asyncio.create_task(signal_ui("Operational rules mutated"))
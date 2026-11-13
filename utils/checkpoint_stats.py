"""Remove THOP bookkeeping keys from checkpoint state dictionaries."""


def without_profiling_stats(state_dict, expected_keys=()):
    """Copy a state_dict, preserving metadata and tensor references.

    Only unexpected THOP counters are removed. Other unexpected keys remain
    visible to strict model loading, so architecture mismatches still fail.
    """
    result = state_dict.copy()
    if hasattr(state_dict, "_metadata"):
        result._metadata = state_dict._metadata
    for key in list(result):
        if (key.rsplit(".", 1)[-1] in {"total_ops", "total_params"}
                and key not in expected_keys):
            del result[key]
    return result

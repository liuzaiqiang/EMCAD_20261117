"""Filter THOP bookkeeping without changing model parameters or live buffers."""


def without_profiling_stats(state_dict, expected_keys=()):
    """Return a shallow copy, preserving state_dict metadata and tensor references.

    When loading, expected_keys protects any fields explicitly required by the
    target model. All other unexpected keys remain for strict loading to reject.
    """
    result = state_dict.copy()
    if hasattr(state_dict, "_metadata"):
        result._metadata = state_dict._metadata
    for key in list(result):
        if (key.rsplit(".", 1)[-1] in {"total_ops", "total_params"}
                and key not in expected_keys):
            del result[key]
    return result

def expected_calibration_error(confidences, correct, n_bins=10):

    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if not confidences:
        raise ValueError("empty input")

    total = len(confidences)
    bins = [[] for _ in range(n_bins)]

    for c, ok in zip(confidences, correct):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, ok))

    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        accuracy = sum(1 for _, ok in b if ok) / len(b)
        ece += (len(b) / total) * abs(avg_conf - accuracy)

    return ece
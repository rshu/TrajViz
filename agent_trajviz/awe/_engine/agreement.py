"""Inter-rater agreement: Cohen's / Fleiss' kappa, onset metrics, and
detector-vs-human-consensus — the numbers that back the reliability claim.

Crucially, compute the HUMAN-HUMAN kappa first (the ceiling): if expert
annotators only agree to kappa=k on root-cause, the detector matching humans at
~k is at ceiling, and that bounds what any automated method can claim."""

from __future__ import annotations

from collections import Counter


def cohen_kappa(a: list, b: list) -> float | None:
    """Two raters, paired categorical labels."""
    pairs = [(x, y) for x, y in zip(a, b, strict=False) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return None
    cats = {x for p in pairs for x in p}
    po = sum(x == y for x, y in pairs) / n
    ca = Counter(x for x, _ in pairs)
    cb = Counter(y for _, y in pairs)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(items: list[dict]) -> float | None:
    """items[i] = {category: count}; every item rated by the same n>=2 raters."""
    items = [it for it in items if sum(it.values()) >= 2]
    if not items:
        return None
    n = sum(items[0].values())
    if any(sum(it.values()) != n for it in items):
        return None  # requires equal rater count per item
    cats = sorted({c for it in items for c in it})
    N = len(items)
    Pi = [(sum(it.get(c, 0) ** 2 for c in cats) - n) / (n * (n - 1)) for it in items]
    Pbar = sum(Pi) / N
    pj = {c: sum(it.get(c, 0) for it in items) / (N * n) for c in cats}
    Pe = sum(v**2 for v in pj.values())
    return 1.0 if Pe == 1 else (Pbar - Pe) / (1 - Pe)


def onset_metrics(a: list, b: list) -> dict:
    pairs = [(x, y) for x, y in zip(a, b, strict=False) if x is not None and y is not None]
    if not pairs:
        return {"n": 0, "exact_match": None, "mean_abs_dist": None}
    return {
        "n": len(pairs),
        "exact_match": sum(x == y for x, y in pairs) / len(pairs),
        "mean_abs_dist": sum(abs(x - y) for x, y in pairs) / len(pairs),
    }


def consensus(annotators: dict, case_ids: list, field: str) -> dict:
    """Majority label per case across annotators (ties -> first seen)."""
    out = {}
    for cid in case_ids:
        votes = Counter()
        for labels in annotators.values():
            v = labels.get(cid, {}).get(field)
            if v is not None:
                votes[v] += 1
        out[cid] = votes.most_common(1)[0][0] if votes else None
    return out


def detector_vs_consensus(detector: dict, annotators: dict, case_ids: list, field: str) -> dict:
    cons = consensus(annotators, case_ids, field)
    det = [detector.get(c, {}).get(field) for c in case_ids]
    con = [cons.get(c) for c in case_ids]
    paired = [(d, c) for d, c in zip(det, con, strict=False) if d is not None and c is not None]
    acc = sum(d == c for d, c in paired) / len(paired) if paired else None
    return {"n": len(paired), "accuracy": acc, "kappa": cohen_kappa(det, con)}


def human_human(annotators: dict, case_ids: list, field: str) -> dict:
    """Pairwise mean Cohen kappa + Fleiss kappa across all annotators."""
    names = list(annotators)
    kappas = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = [annotators[names[i]].get(c, {}).get(field) for c in case_ids]
            b = [annotators[names[j]].get(c, {}).get(field) for c in case_ids]
            k = cohen_kappa(a, b)
            if k is not None:
                kappas.append(k)
    items = []
    for c in case_ids:
        votes = Counter(
            annotators[nm].get(c, {}).get(field) for nm in names if annotators[nm].get(c, {}).get(field) is not None
        )
        if votes:
            items.append(dict(votes))
    return {
        "pairwise_mean_kappa": sum(kappas) / len(kappas) if kappas else None,
        "fleiss_kappa": fleiss_kappa(items),
        "n_pairs": len(kappas),
    }

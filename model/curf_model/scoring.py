"""Shared time/frequency matching for raw-model and display-policy evaluation."""


def match(s, e, time_fraction=.25):
    if "stream" in e and s.get("s", 0) != e["stream"]:
        return 0.0
    if min(s["bw"], e["bw"]) <= 0 or max(s["bw"], e["bw"]) > 4 * min(s["bw"], e["bw"]):
        return 0.0
    flo = max(s["f"] - s["bw"]/2, e["fc"] - e["bw"]/2)
    fhi = min(s["f"] + s["bw"]/2, e["fc"] + e["bw"]/2)
    overlap = max(0.0, fhi - flo)
    if overlap < .5 * min(s["bw"], e["bw"]):
        return 0.0
    dt = max(0.0, min(s["t1"], e["t1"]) - max(s["t0"], e["t0"]))
    if dt < time_fraction * min(s["t1"] - s["t0"], e["t1"] - e["t0"]) or dt <= 0:
        return 0.0
    return overlap / (s["bw"] + e["bw"] - overlap) * dt / (s["t1"] - s["t0"])

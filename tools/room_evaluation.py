"""M4 scoring: complete finite feeds, RX isolation, protocol guesses, and separate rejection statistics."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from signal_policy import SignalPolicy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"model"))
from curf_model.scoring import match

PROTOCOLS = {"WiFi（推测）":{"wifi"}, "WiFi DSSS（推测）":{"wifi"}, "蓝牙 / BLE（推测）":{"ble","bluetooth"},
             "Zigbee（推测）":{"zigbee"}, "4G LTE（推测）":{"lte"}, "LTE / NR（推测）":{"lte","nr"},
             "5G NR（推测）":{"nr"}, "GSM（推测）":{"gsm"}}


def config_hash(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def strict_match(r,e):
    # A merged component or an off-centre skirt is not a correct carrier label.
    if abs(r["f"]-e.get("carrier_fc",e["fc"])) > max(5000,.25*e.get("carrier_bw",e["bw"])):
        return 0.
    # Classification may use a valid partial packet at the hop boundary. It does
    # not claim to have decoded a complete packet. Require actual time overlap.
    return match(r,e,time_fraction=.05)


def evaluate_case(directory, config, write_feed=True):
    directory=Path(directory)
    truth=json.loads((directory/"truth.json").read_text())
    if truth.get("version") != 2 or any("stream" not in e or "protocol" not in e for e in truth["truth"]):
        raise ValueError("M4 truth requires RX and protocol labels")
    p=SignalPolicy(config)
    counts=Counter()
    seen=set()
    errors=[]
    meta,ended=False,False
    by_stream=defaultdict(list)
    carriers={c["source_id"]:c for c in truth["carriers"]}
    for e in truth["truth"]:
        carrier=carriers.get(e["source_id"],{})
        by_stream[e["stream"]].append({**e,"carrier_fc":e.get("carrier_fc",carrier.get("fc",e["fc"])),
                                      "carrier_bw":e.get("carrier_bw",carrier.get("bw",e["bw"]))})
    output=(directory/"display.feed").open("w") if write_feed else None
    try:
        with (directory/"raw.feed").open() as f:
            for line in f:
                row=json.loads(line)
                if "meta" in row:
                    if meta:
                        raise ValueError("M4 expects one finite run")
                    meta=True
                    if row["meta"]["freq"] != truth["freq"] or row["meta"]["streams"] != truth["streams"]:
                        raise ValueError("RF centre or receiver mapping does not match truth")
                elif not meta:
                    raise ValueError("missing feed metadata")
                if ended:
                    raise ValueError("records after end")
                row=p.process(row)
                ended="end" in row
                if output:
                    output.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")
                if "sig" not in row:
                    continue
                r,d=row["sig"],row["display"]
                counts["records"]+=1
                status=d["status"]
                if status not in ("unknown","stable","held"):
                    counts[status]+=1
                    continue
                candidates=[(strict_match(r,e),i,e) for i,e in enumerate(by_stream[r["s"]])]
                value,_,e=max(candidates,key=lambda x:x[0],default=(0,-1,None))
                expected=bool(e and value>0 and e.get("expected_named"))
                if status=="unknown":
                    counts["unknown"]+=1
                    counts["positive_unknown" if expected else "negative_unknown"]+=1
                    continue
                counts["named"]+=1
                counts["family_"+d["family"]]+=1
                bad = value<=0 or e["family"] != d["family"] or not e.get("expected_named")
                reason="unmatched" if value<=0 else "wrong_family" if e["family"] != d["family"] else "negative_named" if not e.get("expected_named") else None
                if d.get("protocol"):
                    counts["protocol_named"]+=1
                    compatible=PROTOCOLS.get(d["protocol"],set())
                    if value<=0 or e["protocol"] not in compatible:
                        counts["protocol_wrong"]+=1
                        bad=True
                        reason=reason or "wrong_protocol"
                if bad:
                    counts["errors"]+=1
                    counts[reason]+=1
                    if len(errors)<100:
                        errors.append({"reason":reason,"sig":r,"display":d,"truth":e if value>0 else None})
                else:
                    counts["correct"]+=1
                    counts["positive_named"]+=1
                    seen.add(e["source_id"])
        if not ended:
            raise ValueError("M4 feed incomplete: missing end record")
    finally:
        if output:
            output.close()
    runtime=json.loads((directory/"runtime.json").read_text())
    real_checks={"zero_dropped":runtime["dropped_frames"]==0,"zero_gaps":runtime["gap_frames"]==0,
                 "no_skips":all(runtime["stage2"][k]==0 for k in ("skipped_busy","skipped_ring","skipped_capacity")),
                 "no_clipping":truth["generation"]["clipped_fraction"]==0,
                 "duration":runtime["wall_s"]>=truth["secs"]-.02}
    high=[c for c in truth["carriers"] if c["expected_named"] and c["snr_db"]>=14]
    return {"seed":truth["seed"],"profile":truth["profile"],"kind":truth["kind"],"snr_db":truth["nominal_snr_db"],
            "streams":truth["streams"],"counts":dict(counts),"error_fraction":counts["errors"]/counts["named"] if counts["named"] else None,
            "unknown_fraction":counts["unknown"]/(counts["unknown"]+counts["named"]) if counts["unknown"]+counts["named"] else None,
            "high_snr_carriers":len(high),"high_snr_named":sum(c["source_id"] in seen for c in high),
            "checks":real_checks,"errors":errors,"policy_sha256":config_hash(config)}


def summarize(results, split, min_named=100):
    total=Counter()
    groups=defaultdict(Counter)
    for r in results:
        c=Counter(r["counts"])
        total.update(c)
        groups[r["profile"]].update(c)
    def stats(c):
        return {"named":c["named"],"errors":c["errors"],"error_fraction":c["errors"]/c["named"] if c["named"] else None,
                "unknown":c["unknown"],"unknown_fraction":c["unknown"]/(c["named"]+c["unknown"]) if c["named"]+c["unknown"] else None,
                "positive_unknown_fraction":c["positive_unknown"]/(c["positive_unknown"]+c["positive_named"]) if c["positive_unknown"]+c["positive_named"] else None,
                "protocol_named":c["protocol_named"],"protocol_wrong":c["protocol_wrong"],
                "named_by_family":{k[7:]:v for k,v in c.items() if k.startswith("family_")}}
    checks={"complete_runs":all(all(r["checks"].values()) for r in results),
            "total_error_below_2pct":total["named"]>0 and total["errors"]/total["named"]<.02,
            "each_band_below_2pct":all(c["named"]>0 and c["errors"]/c["named"]<.02 for c in groups.values()),
            "enough_named_per_band":len(groups)==4 and all(c["named"]>=min_named for c in groups.values()),
            "no_named_noise":all(r["counts"].get("named",0)==0 for r in results if r["kind"]=="noise")}
    return {"split":split,"passed":all(checks.values()),"checks":checks,"totals":stats(total),
            "by_band":{k:stats(c) for k,c in groups.items()},"cases":len(results),"min_named_per_band":min_named,
            "high_snr_carrier_coverage":{"named":sum(r["high_snr_named"] for r in results),"total":sum(r["high_snr_carriers"] for r in results)},
            "metric":"named signal-result modulation labels plus any displayed protocol guess; wrong/unmatched labels count once",
            "limits":"Scene seeds and validation slices are separated; underlying clean source pool is shared. Correlated model windows are not independent statistical trials."}

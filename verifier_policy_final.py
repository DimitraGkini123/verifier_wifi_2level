# verifier_policy.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Tuple
from collections import deque


# ===================== Labels that match YOUR models =====================

class GateLabel(str, Enum):
    SAFE = "SAFE"
    COMPROMISED = "COMPROMISED"
    UNCERTAIN = "UNCERTAIN"


class WorkloadLabel(str, Enum):
    LIGHT = "LIGHT"
    MEDIUM = "MEDIUM"
    HEAVY = "HEAVY"
    UNKNOWN = "UNKNOWN"   # used when gate != SAFE


class Attack2BLabel(str, Enum):
    INJECTION     = "INJECTION"
    INTERRUPTION  = "INTERRUPTION"
    UNKNOWN       = "UNKNOWN"   # used when gate != COMPROMISED


class AttestKind(str, Enum):
    NONE = "NONE"
    PARTIAL = "PARTIAL"
    FULL = "FULL"


# ===================== Decisions / State =====================

@dataclass
class PolicyDecision:
    do_get_windows: bool = True
    get_windows_max: int = 6

    attest_kind: AttestKind = AttestKind.NONE
    coverage: float = 0.0

    quarantine: bool = False
    require_recheck: bool = False

    reason: str = ""


@dataclass
class InferenceSummary:
    majority: str
    confidence: float
    n: int


@dataclass
class DevicePolicyState:
    stable_gate: GateLabel = GateLabel.UNCERTAIN
    stable_workload: WorkloadLabel = WorkloadLabel.UNKNOWN
    stable_attack2b: Attack2BLabel = Attack2BLabel.UNKNOWN

    last_gate: GateLabel = GateLabel.UNCERTAIN
    last_gate_conf: float = 0.0

    last_workload: WorkloadLabel = WorkloadLabel.UNKNOWN
    last_workload_conf: float = 0.0

    last_attack2b: Attack2BLabel = Attack2BLabel.UNKNOWN
    last_attack2b_conf: float = 0.0

    last_reason: str = "init"

    pending_gate: Optional[GateLabel] = None
    pending_gate_count: int = 0

    pending_workload: Optional[WorkloadLabel] = None
    pending_workload_count: int = 0

    pending_attack2b: Optional[Attack2BLabel] = None
    pending_attack2b_count: int = 0

    next_get_windows_ts: float = 0.0
    next_attest_ts: float = 0.0

    attest_cooldown_s: float = 0.0

    wl_hist: deque = field(default_factory=lambda: deque(maxlen=3))


# ===================== Policy Config =====================

@dataclass
class PolicyConfig:
    # GET_WINDOWS rates by GATE
    get_windows_period_s: Dict[GateLabel, float] = field(default_factory=lambda: {
        GateLabel.SAFE: 1.0,
        GateLabel.UNCERTAIN: 0.4,
        GateLabel.COMPROMISED: 0.4,
    })
    get_windows_max: Dict[GateLabel, int] = field(default_factory=lambda: {
        GateLabel.SAFE: 8,
        GateLabel.UNCERTAIN: 14,
        GateLabel.COMPROMISED: 14,
    })

    # ATTEST scheduling by GATE
    attest_period_s: Dict[GateLabel, float] = field(default_factory=lambda: {
        GateLabel.SAFE: 25.0,
        GateLabel.UNCERTAIN: 6.0,
        GateLabel.COMPROMISED: 4.0,
    })
    attest_kind: Dict[GateLabel, AttestKind] = field(default_factory=lambda: {
        GateLabel.SAFE: AttestKind.PARTIAL,
        GateLabel.UNCERTAIN: AttestKind.PARTIAL,
        GateLabel.COMPROMISED: AttestKind.FULL,
    })

    partial_coverage_by_workload: Dict[WorkloadLabel, float] = field(default_factory=lambda: {
        WorkloadLabel.LIGHT: 0.60,
        WorkloadLabel.MEDIUM: 0.40,
        WorkloadLabel.HEAVY: 0.20,
        WorkloadLabel.UNKNOWN: 0.50,
    })

    compromised_attest_kind_by_attack: Dict[Attack2BLabel, AttestKind] = field(default_factory=lambda: {
        Attack2BLabel.INJECTION:    AttestKind.FULL,
        Attack2BLabel.INTERRUPTION: AttestKind.PARTIAL,
        Attack2BLabel.UNKNOWN:      AttestKind.FULL,
    })

    compromised_attest_period_s_by_attack: Dict[Attack2BLabel, float] = field(default_factory=lambda: {
        Attack2BLabel.INJECTION:    20.0,
        Attack2BLabel.INTERRUPTION: 6.0,
        Attack2BLabel.UNKNOWN:      10.0,
    })

    compromised_partial_coverage_by_attack: Dict[Attack2BLabel, float] = field(default_factory=lambda: {
        Attack2BLabel.INTERRUPTION: 0.80,
        Attack2BLabel.UNKNOWN:      0.70,
        Attack2BLabel.INJECTION:    0.0,
    })

    uncertain_coverage_min: float = 0.70
    min_attest_cooldown_s: float = 1.0
    quarantine_on_compromised: bool = False
    attack2b_min_majority_frac: float = 0.70

    # =========================
    # NEW: Asymmetric gate batch decision
    # =========================
    gate_safe_min_frac: float = 0.95     # SAFE only if >=95% SAFE in batch
    gate_comp_min_frac: float = 0.10    # COMPROMISED if >=10% COMP in batch
    gate_comp_min_count: int = 6         # OR if >=2 COMP windows in batch
    gate_asym_enabled: bool = True


# ===================== Policy Engine =====================

class PolicyEngine:
    def __init__(
        self,
        hysteresis_gate_n: int = 2,
        hysteresis_workload_n: int = 2,
        hysteresis_attack2b_n: int = 2,
        enable_get_windows: bool = True,
        config: Optional[PolicyConfig] = None,
    ):
        self.h_gate = max(1, int(hysteresis_gate_n))
        self.h_wl = max(1, int(hysteresis_workload_n))
        self.h_atk = max(1, int(hysteresis_attack2b_n))
        self.enable_get_windows = bool(enable_get_windows)
        self.cfg = config or PolicyConfig()
        self.devices: Dict[str, DevicePolicyState] = {}
        self.device_cfg: Dict[str, PolicyConfig] = {}

    def _st(self, dev: str) -> DevicePolicyState:
        st = self.devices.get(dev)
        if st is None:
            st = DevicePolicyState()
            self.devices[dev] = st
        return st

    def set_device_config(self, dev: str, cfg: PolicyConfig) -> None:
        self.device_cfg[dev] = cfg
        st = self._st(dev)
        st.next_get_windows_ts = 0.0
        st.next_attest_ts = 0.0

    @staticmethod
    def _majority_str(labels: List[str], default: str) -> InferenceSummary:
        if not labels:
            return InferenceSummary(majority=default, confidence=0.0, n=0)
        counts: Dict[str, int] = {}
        for lb in labels:
            counts[lb] = counts.get(lb, 0) + 1
        maj = max(counts.items(), key=lambda kv: kv[1])[0]
        conf = counts[maj] / float(len(labels))
        return InferenceSummary(majority=maj, confidence=conf, n=len(labels))

    @staticmethod
    def _majority_enum(vals, default_enum):
        if not vals:
            return default_enum, 0.0
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        maj = max(counts.items(), key=lambda kv: kv[1])[0]
        conf = counts[maj] / float(len(vals))
        return maj, float(conf)

    def _update_stable(
        self,
        *,
        st: DevicePolicyState,
        level_name: str,
        new_majority,
        new_conf: float,
        hysteresis_n: int,
        stable_attr: str,
        pending_attr: str,
        pending_count_attr: str,
    ) -> Tuple[bool, str]:
        stable = getattr(st, stable_attr)
        pending = getattr(st, pending_attr)
        pending_count = getattr(st, pending_count_attr)

        if new_majority == stable:
            setattr(st, pending_attr, None)
            setattr(st, pending_count_attr, 0)
            return False, f"{level_name}:majority_matches_stable"

        if pending != new_majority:
            setattr(st, pending_attr, new_majority)
            setattr(st, pending_count_attr, 1)
            return False, f"{level_name}:pending_start:{new_majority.value}"
        else:
            pending_count += 1
            setattr(st, pending_count_attr, pending_count)
            if pending_count >= hysteresis_n:
                old = stable
                setattr(st, stable_attr, pending or new_majority)
                setattr(st, pending_attr, None)
                setattr(st, pending_count_attr, 0)
                return True, f"{level_name}:stable_changed:{old.value}->{new_majority.value}"
            return False, f"{level_name}:pending_count:{pending_count}/{hysteresis_n}"

    # =====================
    # NEW: Asymmetric gate collapse
    # =====================
    def _asym_gate_majority(self, gate_labels: List[GateLabel], cfg: PolicyConfig) -> Tuple[GateLabel, float, str]:
        """
        Return (maj_enum, conf_proxy, reason)
        conf_proxy is not a model prob; it's "how dominant" the chosen decision is.
        """
        if not gate_labels:
            return GateLabel.UNCERTAIN, 0.0, "asym:empty"

        n = len(gate_labels)
        safe_cnt = sum(1 for g in gate_labels if g == GateLabel.SAFE)
        comp_cnt = sum(1 for g in gate_labels if g == GateLabel.COMPROMISED)
        safe_frac = safe_cnt / float(n)
        comp_frac = comp_cnt / float(n)

        # 1) COMPROMISED if "few but persistent"
        if (comp_cnt >= int(cfg.gate_comp_min_count)) or (comp_frac >= float(cfg.gate_comp_min_frac)):
            return GateLabel.COMPROMISED, float(comp_frac), f"asym:comp cnt={comp_cnt}/{n} frac={comp_frac:.3f}"

        # 2) SAFE only if almost all safe
        if safe_frac >= float(cfg.gate_safe_min_frac):
            return GateLabel.SAFE, float(safe_frac), f"asym:safe frac={safe_frac:.3f}"

        # 3) otherwise UNCERTAIN
        return GateLabel.UNCERTAIN, float(max(safe_frac, comp_frac)), f"asym:uncertain safe={safe_frac:.3f} comp={comp_frac:.3f}"

    # ===================== Inference hooks =====================

    def on_gate_batch(self, dev: str, gate_labels: List[GateLabel], now: float) -> InferenceSummary:
        st = self._st(dev)
        cfg = self.device_cfg.get(dev, self.cfg)

        if bool(getattr(cfg, "gate_asym_enabled", True)):
            maj, conf_proxy, asym_reason = self._asym_gate_majority(gate_labels, cfg)
            summ = InferenceSummary(majority=maj.value, confidence=float(conf_proxy), n=len(gate_labels))
            maj_enum = maj
            reason_prefix = asym_reason
        else:
            summ = self._majority_str([g.value for g in gate_labels], default=GateLabel.UNCERTAIN.value)
            maj_enum = GateLabel(summ.majority)
            reason_prefix = "maj:plain"

        st.last_gate = maj_enum
        st.last_gate_conf = float(summ.confidence)

        changed, reason = self._update_stable(
            st=st,
            level_name="gate",
            new_majority=maj_enum,
            new_conf=float(summ.confidence),
            hysteresis_n=self.h_gate,
            stable_attr="stable_gate",
            pending_attr="pending_gate",
            pending_count_attr="pending_gate_count",
        )

        if changed:
            st.next_get_windows_ts = 0.0
            st.next_attest_ts = 0.0

            if st.stable_gate != GateLabel.SAFE:
                st.stable_workload = WorkloadLabel.UNKNOWN
                st.pending_workload = None
                st.pending_workload_count = 0
                st.wl_hist.clear()

            if st.stable_gate != GateLabel.COMPROMISED:
                st.stable_attack2b = Attack2BLabel.UNKNOWN
                st.pending_attack2b = None
                st.pending_attack2b_count = 0

        st.last_reason = f"{reason_prefix} | {reason}"
        return InferenceSummary(majority=maj_enum.value, confidence=float(summ.confidence), n=summ.n)

    def on_workload_batch(self, dev: str, workload_labels: List[WorkloadLabel], now: float) -> InferenceSummary:
        st = self._st(dev)
        if st.stable_gate != GateLabel.SAFE:
            st.last_workload = WorkloadLabel.UNKNOWN
            st.last_workload_conf = 0.0
            st.last_reason = "workload:ignored_gate_not_safe"
            st.wl_hist.clear()
            return InferenceSummary(majority=WorkloadLabel.UNKNOWN.value, confidence=0.0, n=0)

        summ_raw = self._majority_str([w.value for w in workload_labels], default=WorkloadLabel.UNKNOWN.value)
        raw_maj = WorkloadLabel(summ_raw.majority)
        raw_conf = float(summ_raw.confidence)

        if raw_maj != WorkloadLabel.UNKNOWN:
            st.wl_hist.append(raw_maj)

        smooth_maj, smooth_conf = self._majority_enum(list(st.wl_hist), default_enum=raw_maj)

        st.last_workload = smooth_maj
        st.last_workload_conf = float(smooth_conf)

        changed, reason = self._update_stable(
            st=st,
            level_name="workload",
            new_majority=smooth_maj,
            new_conf=float(smooth_conf),
            hysteresis_n=self.h_wl,
            stable_attr="stable_workload",
            pending_attr="pending_workload",
            pending_count_attr="pending_workload_count",
        )

        if changed:
            st.next_get_windows_ts = 0.0
            st.next_attest_ts = 0.0

        hist_str = ",".join([v.value for v in list(st.wl_hist)])
        st.last_reason = f"{reason} | wl_raw={raw_maj.value}({raw_conf:.2f}) wl_smooth={smooth_maj.value}({smooth_conf:.2f}) hist=[{hist_str}]"

        return InferenceSummary(majority=smooth_maj.value, confidence=float(smooth_conf), n=summ_raw.n)

    def on_attack2b_batch(self, dev: str, attack_labels: List[Attack2BLabel], now: float) -> InferenceSummary:
        st = self._st(dev)
        if st.stable_gate != GateLabel.COMPROMISED:
            st.last_attack2b = Attack2BLabel.UNKNOWN
            st.last_attack2b_conf = 0.0
            st.last_reason = "attack2b:ignored_gate_not_compromised"
            return InferenceSummary(majority=Attack2BLabel.UNKNOWN.value, confidence=0.0, n=0)

        cfg = self.device_cfg.get(dev, self.cfg)

        summ = self._majority_str([a.value for a in attack_labels], default=Attack2BLabel.UNKNOWN.value)
        maj = Attack2BLabel(summ.majority)

        st.last_attack2b = maj
        st.last_attack2b_conf = float(summ.confidence)

        min_frac = float(getattr(cfg, "attack2b_min_majority_frac", 0.0) or 0.0)
        if float(summ.confidence) < min_frac:
            st.last_reason = f"attack2b:below_majority_frac:{summ.confidence:.3f}<{min_frac:.3f}"
            return InferenceSummary(majority=maj.value, confidence=float(summ.confidence), n=summ.n)

        changed, reason = self._update_stable(
            st=st,
            level_name="attack2b",
            new_majority=maj,
            new_conf=float(summ.confidence),
            hysteresis_n=self.h_atk,
            stable_attr="stable_attack2b",
            pending_attr="pending_attack2b",
            pending_count_attr="pending_attack2b_count",
        )

        if changed:
            st.next_get_windows_ts = 0.0
            st.next_attest_ts = 0.0

        st.last_reason = reason
        return InferenceSummary(majority=maj.value, confidence=float(summ.confidence), n=summ.n)

    # ===================== Scheduling tick =====================

    def tick(self, dev: str, now: float) -> PolicyDecision:
        st = self._st(dev)
        cfg = self.device_cfg.get(dev, self.cfg)

        gate = st.stable_gate
        wl = st.stable_workload
        atk = st.stable_attack2b

        do_get = False
        gw_max = cfg.get_windows_max[gate]
        if self.enable_get_windows and now >= st.next_get_windows_ts:
            do_get = True
            st.next_get_windows_ts = now + float(cfg.get_windows_period_s[gate])

        kind = AttestKind.NONE
        cov = 0.0
        quarantine = False
        require_recheck = False

        if now >= st.next_attest_ts and now >= st.attest_cooldown_s:
            kind = cfg.attest_kind[gate]
            cov = 0.0

            if gate == GateLabel.COMPROMISED:
                kind = cfg.compromised_attest_kind_by_attack.get(atk, AttestKind.FULL)
                period = float(cfg.compromised_attest_period_s_by_attack.get(atk, cfg.attest_period_s[gate]))

                if kind == AttestKind.PARTIAL:
                    cov = float(cfg.compromised_partial_coverage_by_attack.get(atk, 0.7))
                    cov = max(0.0, min(1.0, cov))

                st.next_attest_ts = now + period
                st.attest_cooldown_s = now + float(cfg.min_attest_cooldown_s)

            else:
                if kind == AttestKind.PARTIAL:
                    base_cov = float(cfg.partial_coverage_by_workload.get(wl, 0.5))
                    base_cov = max(0.0, min(1.0, base_cov))
                    if gate == GateLabel.UNCERTAIN:
                        cov = max(base_cov, float(cfg.uncertain_coverage_min))
                        require_recheck = True
                    else:
                        cov = base_cov

                st.next_attest_ts = now + float(cfg.attest_period_s[gate])
                st.attest_cooldown_s = now + float(cfg.min_attest_cooldown_s)

        if gate == GateLabel.COMPROMISED and cfg.quarantine_on_compromised:
            quarantine = True

        reason = st.last_reason + f" | stable_gate={gate.value} stable_wl={wl.value} stable_attack={atk.value}"

        return PolicyDecision(
            do_get_windows=do_get,
            get_windows_max=gw_max,
            attest_kind=kind,
            coverage=cov,
            quarantine=quarantine,
            require_recheck=require_recheck,
            reason=reason,
        )

    def get_stable_labels(self, dev: str) -> Tuple[GateLabel, WorkloadLabel, Attack2BLabel]:
        st = self._st(dev)
        return st.stable_gate, st.stable_workload, st.stable_attack2b

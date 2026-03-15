# verifier_policy_server_final.py
##FINAL VERSION OF SERVER
import os
from pathlib import Path
import asyncio
import json
import secrets
import time
import random
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import math
import traceback
import contextlib
import logging

import numpy as np
import joblib
from collections import deque

from utils import save_json_atomic, jdump, sha256, ts_ms, now_s, unhex
# from lru_blocks import DeviceLRUBlocks
from lru_blocks_rand import DeviceLRUBlocks
from policy_2level_lr import Device2StagePolicy

from verifier_policy_final import (
    PolicyEngine,
    PolicyConfig,  # <<< NEW
    GateLabel, WorkloadLabel, Attack2BLabel,
    AttestKind as PLAttestKind,
)

LRU_STATE_PATH = "lru_state.json"

HOST = "0.0.0.0"
PORT = 4242
GOLDEN_PATH = "golden.json"

# NEW: budget config file
BUDGET_CFG_PATH = "budget_config_2level.json"
# LOG_DIR = "logs_2level"
# LOG_DIR = "logs_security_critical"
LOG_DIR = "logs_final_inj"

TRUST_UNKNOWN = "UNKNOWN"
TRUST_TRUSTED = "TRUSTED"
TRUST_UNTRUSTED = "UNTRUSTED"

AUTO_PROVISION_ON_REGISTER = True
AUTO_PROVISION_BLOCKS_TOO = True   # αν θες και blocks
AUTO_PROVISION_FORCE = False       # ΜΗΝ το κάνεις True εκτός αν θες overwrite
AUTO_PROVISION_DELAY_S = 0.3

# quarantine configs
QUARANTINE_ON_FULL_FAILS = 2        # trigger after 2 consecutive FULL failures
QUARANTINE_RECHECK_S = 40.0         # how often to attempt recovery FULL while quarantined

# ---------------- ML / Policy config ----------------
# Keep legacy 2-stage policy (for WL/ATK & fallback gate)
GATE_PATH = "models/gate_compromised_lr_uncertain.joblib"
# NEW: gate model (HGB, W-window aggregation)
GATE_HGB_PATH = "models/level1.joblib"

WL_PATH   = "models/level2a.joblib"
ATK_PATH  = "models/level2b.joblib"
ML_ENABLE = True

# Policy hysteresis: πόσες φορές πρέπει να δεις majority διαφορετικό για να αλλάξεις stable label
POLICY_HYSTERESIS_N = 2

# Safety jitter (μικρό) για να μη συγχρονίζονται πολλά devices (αν έχεις πολλά)
LOOP_TICK_S = 0.20
JITTER_S = 0.05

# Initial full attestation on connect?
DO_INITIAL_FULL_ATTEST = True


# =========================
# Budgeting (token bucket)
# =========================

FULL_COST_UNITS = 100  # 100 budget = full hash

@dataclass
class BudgetConfig:
    per_min_units: int = 50
    cap_units: int = 50
    min_k: int = 1  # minimum partial k allowed (degrade not below this)

@dataclass
class DeviceBudgetState:
    tokens: float = 0.0
    last_ts: float = 0.0

class BudgetManager:
    """
    Token bucket per device:
      - refill continuously at per_min_units / 60 per sec
      - cap at cap_units
      - spend immediately when scheduling to avoid races
    """

    def __init__(self, default_cfg: BudgetConfig, per_device_cfg: Optional[Dict[str, BudgetConfig]] = None):
        self.default_cfg = default_cfg
        self.per_device_cfg = per_device_cfg or {}
        self.st: Dict[str, DeviceBudgetState] = {}

    def _cfg(self, dev: str) -> BudgetConfig:
        return self.per_device_cfg.get(dev, self.default_cfg)

    def _state(self, dev: str, now: float) -> DeviceBudgetState:
        st = self.st.get(dev)
        if st is None:
            cfg = self._cfg(dev)
            st = DeviceBudgetState(tokens=float(cfg.cap_units), last_ts=now)
            self.st[dev] = st
        return st

    def _refill(self, dev: str, now: float) -> None:
        cfg = self._cfg(dev)
        st = self._state(dev, now)
        dt = max(0.0, now - st.last_ts)
        rate = float(cfg.per_min_units) / 60.0  # units per second
        st.tokens = min(float(cfg.cap_units), st.tokens + dt * rate)
        st.last_ts = now

    def tokens_now(self, dev: str, now: float) -> float:
        self._refill(dev, now)
        return self.st[dev].tokens

    def spend(self, dev: str, cost: float, now: float) -> bool:
        self._refill(dev, now)
        st = self.st[dev]
        if st.tokens + 1e-9 >= cost:
            st.tokens -= float(cost)
            return True
        return False

    def refund(self, dev: str, cost: float, now: float) -> None:
        self._refill(dev, now)
        cfg = self._cfg(dev)
        st = self.st[dev]
        st.tokens = min(float(cfg.cap_units), st.tokens + float(cost))

    # ----- cost model -----
    @staticmethod
    def cost_full() -> int:
        return FULL_COST_UNITS

    @staticmethod
    def cost_partial(k: int, block_count: int) -> int:
        """
        Proportional cost:
          FULL = 100 units
          PARTIAL(k) ≈ 100 * k / block_count
        Minimum 1 unit to avoid "free" attest.
        """
        bc = max(1, int(block_count))
        kk = max(1, min(int(k), bc))
        return max(1, int(round(FULL_COST_UNITS * (kk / float(bc)))))

    def fit_plan(
        self,
        dev: str,
        now: float,
        ideal_kind: str,
        ideal_k: int,
        block_count: int,
        min_k: int = 1
    ) -> Tuple[str, int, int, str]:
        """
        Decide what can be executed given current tokens.
        Returns: (kind, k, cost_units, reason)
          kind in {"NONE","PARTIAL","FULL"}
        """
        toks = self.tokens_now(dev, now)

        # Normalize limits
        bc = max(1, int(block_count))  # used only for costing; caller should still check real bc>0 for partial
        mk = max(1, min(int(min_k), bc))

        if ideal_kind == "FULL":
            c = self.cost_full()
            if toks >= c:
                return ("FULL", 0, c, "budget_ok_full")

            # Degrade FULL -> PARTIAL (largest k that fits, but not below min_k)
            for kk in range(bc, mk - 1, -1):
                cc = self.cost_partial(kk, bc)
                if toks >= cc:
                    return ("PARTIAL", kk, cc, f"degrade_full_to_partial:k={kk}")
            return ("NONE", 0, 0, "budget_insufficient_for_any_attest")

        if ideal_kind == "PARTIAL":
            kk0 = max(mk, min(int(ideal_k), bc))
            for kk in range(kk0, mk - 1, -1):
                cc = self.cost_partial(kk, bc)
                if toks >= cc:
                    if kk == kk0:
                        return ("PARTIAL", kk, cc, "budget_ok_partial")
                    return ("PARTIAL", kk, cc, f"degrade_partial:k={kk0}->{kk}")
            return ("NONE", 0, 0, "budget_insufficient_for_partial")

        return ("NONE", 0, 0, "policy_none")


# ==========================================================
# NEW: Online Gate inference with HGB over rolling W windows
# ==========================================================
def _safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return a / (b + eps)

class OnlineGateHGB:
    """
    NO-BASELINE version (always-on):
      - Builds base features from each window
      - Keeps a rolling deque of last W feature vectors
      - When deque is full -> aggregates (mean/std/min/max per feature) -> HGB predict_proba
      - Trust gating: only infer when trust_state == "TRUSTED"
      - reset() clears only rolling window (no baseline state exists)
    """

    def __init__(self, blob: dict):
        
        self.model = blob["model"]
        self.W = int(blob["W"])
        self.thr = float(blob.get("thr", 0.25))

        self.agg_features = list(blob["agg_features"])
        self.base_features = list(blob["base_features"])

        norm_cfg = blob.get("norm_cfg", {}) or {}
        self.eps = float(norm_cfg.get("eps", 1e-9))

        # rolling inference window
        self.win_hist = deque(maxlen=self.W)

    def reset(self, reason: str = "reset"):
        # NO baseline: just clear rolling history so next decision needs W fresh windows
        self.win_hist.clear()

    def _base_vec_from_window(self, w: dict) -> list[float]:
        dC = float(w.get("dC", 0.0) or 0.0)
        dL = float(w.get("dL", 0.0) or 0.0)
        dP = float(w.get("dP", 0.0) or 0.0)
        dE = float(w.get("dE", 0.0) or 0.0)
        dF = float(w.get("dF", 0.0) or 0.0)
        cyc_per_us = float(w.get("cyc_per_us", 0.0) or 0.0)

        feats = {
            "cyc_per_us": cyc_per_us,
            "lsu_per_cyc": _safe_div(dL, dC, self.eps),
            "cpi_per_cyc": _safe_div(dP, dC, self.eps),
            "exc_per_cyc": _safe_div(dE, dC, self.eps),
            "fold_per_cyc": _safe_div(dF, dC, self.eps),
        }
        return [float(feats[f]) for f in self.base_features]

    def update(self, w: dict, now: float) -> Optional[dict]:

        #if trust_state != "TRUSTED":
        #    return None

        x = np.asarray(self._base_vec_from_window(w), dtype=np.float32)

        self.win_hist.append(x)
        if len(self.win_hist) < self.W:
            return None

        XW = np.stack(self.win_hist, axis=0)

        feat_dict = {}
        for j, f in enumerate(self.base_features):
            feat_dict[f"{f}_mean_W{self.W}"] = float(XW[:, j].mean())
            feat_dict[f"{f}_std_W{self.W}"]  = float(XW[:, j].std(ddof=0))
            feat_dict[f"{f}_max_W{self.W}"]  = float(XW[:, j].max())
            feat_dict[f"{f}_min_W{self.W}"]  = float(XW[:, j].min())

        xagg = np.asarray([feat_dict[c] for c in self.agg_features], dtype=np.float32).reshape(1, -1)

        p = float(self.model.predict_proba(xagg)[0, 1])
        pred = int(p >= self.thr)
        conf = float(max(p, 1.0 - p))
        return {"p": p, "pred": pred, "thr": self.thr, "conf": conf}



# =========================
# Server structures
# =========================

@dataclass
class PendingReq:
    fut: asyncio.Future
    sent_msg: dict

@dataclass
class DeviceConn:
    device_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    pending: Dict[str, PendingReq] = field(default_factory=dict)
    last_seen_ts: float = field(default_factory=lambda: time.time())

    trust_state: str = TRUST_UNKNOWN
    attest_fail_streak: int = 0
    last_attest_ok_ts: float = 0.0
    last_attest_fail_ts: float = 0.0

    # quarantine!!
    full_hash_fail_streak: int = 0          # counts consecutive FULL failures (final result)
    quarantined: bool = False
    quarantine_since_ts: float = 0.0
    last_quarantine_check_ts: float = 0.0   # throttle full retries while quarantined

    # NEW: baseline arm marker (after INITIAL FULL ok)
    #baseline_armed: bool = False

    def is_alive(self) -> bool:
        return not self.writer.is_closing()


class VerifierPolicyServer:
    def __init__(self, golden_db: dict):
        # Golden DB
        self.golden = golden_db
        # cache last batch majorities (from GET_WINDOWS ML batch)
        self.last_batch_majority: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("verifier")

        # Devices
        self.devices: Dict[str, DeviceConn] = {}
        self.selected_device: Optional[str] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Per-device window cursor ("since")
        self.last_seen: Dict[str, int] = {}

        # Files
        self.windows_fp: Dict[str, Any] = {}
        self.events_fp: Dict[str, Any] = {}
        self.attest_fp: Dict[str, Any] = {}

        # Attestation locks (per device)
        self.attest_locks: Dict[str, asyncio.Lock] = {}

        # LRU blocks for partial
        self.block_lru: Dict[str, DeviceLRUBlocks] = {}
        self._load_lru_state()

        # ML model (existing 2-stage policy used for WL/ATK and fallback gate)
        self.lr_policy = Device2StagePolicy(GATE_PATH, WL_PATH, ATK_PATH) if ML_ENABLE else None

        # NEW: HGB Gate model (W-window aggregation)
        self.gate_hgb_blob = None
        self.gate_hgb_engines: Dict[str, OnlineGateHGB] = {}
        if ML_ENABLE:
            self.gate_hgb_blob = joblib.load(GATE_HGB_PATH)

        # Policy engine per server
        self.policy = PolicyEngine(
            hysteresis_gate_n=POLICY_HYSTERESIS_N,
            hysteresis_workload_n=POLICY_HYSTERESIS_N,
            hysteresis_attack2b_n=POLICY_HYSTERESIS_N,
            enable_get_windows=True,
        )

        # Per-device policy task
        self.policy_tasks: Dict[str, asyncio.Task] = {}

        # Budget (from budget_config.json)
        per_dev_budget, per_dev_min_k, dev_level, level_cov = self._load_budget_config(BUDGET_CFG_PATH)
        self.device_min_k: Dict[str, int] = per_dev_min_k
        self.device_level = dev_level

        self.budget = BudgetManager(
            default_cfg=BudgetConfig(per_min_units=50, cap_units=50, min_k=1),
            per_device_cfg=per_dev_budget
        )

        # ---- Device capabilities from HELLO ----
        self.device_caps_path = "device_caps.json"
        self.device_caps: Dict[str, Dict[str, int]] = {}
        self._load_device_caps()

        # ----------------- NEW: per-level PolicyConfig (detection speed) -----------------
        def _mk_policy_cfg_for_level(level_name: str) -> PolicyConfig:
            cfg = PolicyConfig()

            # Tune GET_WINDOWS for faster ML detection (affects how fast W windows accumulate)
            if level_name == "security_critical":
                cfg.get_windows_period_s = {
                    GateLabel.SAFE: 0.7,
                    GateLabel.UNCERTAIN: 0.4,
                    GateLabel.COMPROMISED: 0.4,
                }
                cfg.get_windows_max = {
                    GateLabel.SAFE: 20,
                    GateLabel.UNCERTAIN: 50,
                    GateLabel.COMPROMISED: 50,
                }
                # Optional: attest a bit faster too (not strictly ML detection, but response speed)
                cfg.attest_period_s = {
                    GateLabel.SAFE: 10.0,
                    GateLabel.UNCERTAIN: 3.0,
                    GateLabel.COMPROMISED: 5.0,
                }
                cfg.min_attest_cooldown_s = 0.5

            elif level_name == "availability_critical":
                cfg.get_windows_period_s = {
                    GateLabel.SAFE: 1.5,
                    GateLabel.UNCERTAIN: 1.2,
                    GateLabel.COMPROMISED: 1.2,
                }
                cfg.get_windows_max = {
                    GateLabel.SAFE: 15,
                    GateLabel.UNCERTAIN: 20,
                    GateLabel.COMPROMISED: 20,
                }
                cfg.attest_period_s = {
                    GateLabel.SAFE: 30.0,
                    GateLabel.UNCERTAIN: 10.0,
                    GateLabel.COMPROMISED: 10.0,
                }
                # keep attest as-is (your defaults) unless you want faster reaction too
            else:
                cfg.get_windows_period_s = {
                    GateLabel.SAFE: 1.0,
                    GateLabel.UNCERTAIN: 0.5,
                    GateLabel.COMPROMISED: 0.7,
                }
                cfg.get_windows_max = {
                    GateLabel.SAFE: 20,
                    GateLabel.UNCERTAIN: 30,
                    GateLabel.COMPROMISED: 30,
                }
                cfg.attest_period_s = {
                    GateLabel.SAFE: 20.0,
                    GateLabel.UNCERTAIN: 5.0,
                    GateLabel.COMPROMISED: 10.0,
                }
                pass

            return cfg

        self.policy_cfg_by_level: Dict[str, PolicyConfig] = {
            "normal": _mk_policy_cfg_for_level("normal"),
            "availability_critical": _mk_policy_cfg_for_level("availability_critical"),
            "security_critical": _mk_policy_cfg_for_level("security_critical"),
        }

        ##helpers for batched
    def _chunk(self, xs: List[int], n: int) -> List[List[int]]:
        return [xs[i:i+n] for i in range(0, len(xs), n)]

    async def _attest_partial_once_indices(
        self, dev: str, indices: List[int], nonce: str, timeout: float = 12.0
    ) -> dict:
        if not indices:
            return {"type": "ERROR", "reason": "empty_indices"}

        resp = await self.send_request_timed(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "PARTIAL_BLOCKS",
            "region": "fw",
            "nonce": nonce,
            "indices": indices
        }, timeout=timeout)

        if isinstance(resp, dict):
            resp["_indices"] = indices
            resp["_k"] = len(indices)
        return resp
    
    async def attest_partial_batched_once(self, dev: str, k: int, timeout: float = 12.0) -> dict:
        bc = self.get_block_count(dev)
        if bc <= 0:
            return {"type": "ERROR", "reason": "no_golden_blocks"}

        k = max(1, min(int(k), bc))
        lru = self._get_block_lru(dev)
        if lru is None:
            return {"type": "ERROR", "reason": "no_golden_blocks"}

        # πάρε ΟΛΑ τα indices upfront
        indices_all = lru.pick(k, pool_frac=0.25, pool_min=32)

        # chunk size από caps (HELLO)
        caps = self._caps(dev)
        max_req = int(caps.get("max_req_blocks", 32) or 32)
        batch = max(1, max_req)  # ή min(max_req, 32) αν θες hard cap

        chunks = self._chunk(indices_all, batch)

        # ίδιο nonce σε όλα τα chunks (ίδιο session binding)
        nonce = secrets.token_hex(8)

        t0 = time.perf_counter()
        req_bytes_sum = 0
        resp_bytes_sum = 0

        last_resp: dict = {"type": "ERROR", "reason": "no_chunks"}

        for bi, chunk in enumerate(chunks):
            r = await self._attest_partial_once_indices(dev, chunk, nonce=nonce, timeout=timeout)

            if isinstance(r, dict):
                req_bytes_sum += int(r.get("_req_bytes", 0) or 0)
                resp_bytes_sum += int(r.get("_resp_bytes", 0) or 0)

            last_resp = r if isinstance(r, dict) else {"type": "ERROR", "reason": "bad_resp"}

            # EARLY STOP: αν αποτύχει ένα batch, κόβεις εδώ
            if not bool(last_resp.get("verify_ok", False)):
                last_resp["_batched"] = True
                last_resp["_batch_size"] = batch
                last_resp["_batch_index"] = bi
                last_resp["_batch_count"] = len(chunks)
                last_resp["_indices_all"] = indices_all
                last_resp["_k_total"] = k
                last_resp["_nonce"] = nonce

                # totals μέχρι το fail (χρήσιμο)
                rtt_ms = (time.perf_counter() - t0) * 1000.0
                last_resp["_rtt_ms_total"] = round(rtt_ms, 2)
                last_resp["_req_bytes_total"] = req_bytes_sum
                last_resp["_resp_bytes_total"] = resp_bytes_sum
                return last_resp

        # αν όλα ΟΚ:
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        ok_resp = dict(last_resp)
        ok_resp["verify_ok"] = True
        ok_resp["verify_reason"] = "ok"
        ok_resp["_batched"] = True
        ok_resp["_batch_size"] = batch
        ok_resp["_batch_count"] = len(chunks)
        ok_resp["_indices_all"] = indices_all
        ok_resp["_k_total"] = k
        ok_resp["_nonce"] = nonce
        ok_resp["_rtt_ms_total"] = round(rtt_ms, 2)
        ok_resp["_req_bytes_total"] = req_bytes_sum
        ok_resp["_resp_bytes_total"] = resp_bytes_sum
        return ok_resp

    def _load_device_caps(self):
        try:
            with open(self.device_caps_path, "r", encoding="utf-8") as f:
                self.device_caps = json.load(f) or {}
        except FileNotFoundError:
            self.device_caps = {}
        except Exception:
            self.device_caps = {}

    def _save_device_caps(self):
        try:
            save_json_atomic(self.device_caps_path, self.device_caps)
        except Exception:
            pass

    def _caps(self, dev: str) -> Dict[str, int]:
        c = self.device_caps.get(dev) or {}
        fw_blocks_n = int(c.get("fw_blocks_n", 0) or 0)
        max_req_blocks = int(c.get("max_req_blocks", 32) or 32)
        if max_req_blocks <= 0:
            max_req_blocks = 32
        return {"fw_blocks_n": fw_blocks_n, "max_req_blocks": max_req_blocks}

    def has_golden_full(self, device_id: str, region: str = "fw") -> bool:
        try:
            _ = self.golden[device_id][region]["sha256"]
            return True
        except Exception:
            return False

    async def auto_provision_on_register(self, dev: str):
        await asyncio.sleep(AUTO_PROVISION_DELAY_S)

        if dev not in self.devices:
            return

        fp_evt = self.events_fp.get(dev)
        if fp_evt:
            self._jwrite(fp_evt, {"ts_ms": ts_ms(), "device": dev, "event": "auto_provision_check_start"})

        # ---- FULL golden ----
        if not self.has_golden_full(dev, region="fw"):
            if fp_evt:
                self._jwrite(fp_evt, {"ts_ms": ts_ms(), "device": dev, "event": "auto_provision_full_start"})

            if AUTO_PROVISION_FORCE:
                resp = await self.force_provision_golden_full(dev, region="fw")
            else:
                resp = await self.provision_golden_full(dev, region="fw")

            if fp_evt:
                self._jwrite(fp_evt, {"ts_ms": ts_ms(), "device": dev, "event": "auto_provision_full_done", "resp": resp})

            if not isinstance(resp, dict) or resp.get("type") != "OK":
                return

        # ---- BLOCKS golden (optional) ----
        if AUTO_PROVISION_BLOCKS_TOO and (not self.has_golden_blocks(dev)):
            if fp_evt:
                self._jwrite(fp_evt, {"ts_ms": ts_ms(), "device": dev, "event": "auto_provision_blocks_start"})

            resp2 = await self.provision_golden_blocks(dev, force=AUTO_PROVISION_FORCE)

            if fp_evt:
                self._jwrite(fp_evt, {"ts_ms": ts_ms(), "device": dev, "event": "auto_provision_blocks_done", "resp": resp2})

        # ---- sanity FULL attest μετά το provisioning ----
        if fp_evt:
            self._jwrite(fp_evt, {"ts_ms": ts_ms(), "device": dev, "event": "auto_provision_sanity_full_attest_start"})

        resp3 = await self.attest_full_and_log(dev, trigger="AUTO_PROVISION_SANITY", ml={"policy_reason": "auto_provision_sanity"})

        if fp_evt:
            self._jwrite(fp_evt, {
                "ts_ms": ts_ms(),
                "device": dev,
                "event": "auto_provision_sanity_full_attest_done",
                "verify_ok": (resp3.get("verify_ok") if isinstance(resp3, dict) else False),
                "verify_reason": (resp3.get("verify_reason") if isinstance(resp3, dict) else "bad_resp")
            })

    async def provision_golden_full(self, dev: str, region: str = "fw") -> dict:
        if self.has_golden_full(dev, region):
            return {"type": "ERROR", "reason": "golden_already_exists_refusing_overwrite", "device": dev, "region": region}

        nonce = secrets.token_hex(8)
        resp = await self.send_request(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "FULL_HASH_PROVER",
            "region": region,
            "nonce": nonce
        }, timeout=12.0)

        fw_hex = resp.get("fw_hash_hex")
        if not fw_hex:
            return {"type": "ERROR", "reason": "missing_fw_hash_hex_in_response", "resp": resp}

        self.set_golden_full_hash(dev, region, fw_hex)
        return {"type": "OK", "event": "golden_provisioned", "device": dev, "region": region, "fw_hash_hex": fw_hex}

    # ----------------- quarantine -----------------
    def _enter_quarantine(self, dev: str, reason: str):
        dc = self.devices.get(dev)
        if not dc:
            return
        if dc.quarantined:
            return

        dc.quarantined = True
        ge = self.gate_hgb_engines.get(dev)
        if ge is not None:
            ge.reset(reason="enter_quarantine")

        dc.quarantine_since_ts = time.time()
        dc.last_quarantine_check_ts = 0.0
        dc.trust_state = TRUST_UNTRUSTED
        dc.full_hash_fail_streak = max(dc.full_hash_fail_streak, QUARANTINE_ON_FULL_FAILS)

        fp_evt = self.events_fp.get(dev)
        if fp_evt:
            self._jwrite(fp_evt, {
                "ts_ms": ts_ms(),
                "device": dev,
                "event": "quarantine_entered",
                "reason": reason,
                "full_hash_fail_streak": dc.full_hash_fail_streak,
            })

    def _exit_quarantine(self, dev: str, reason: str = "full_hash_recovered"):
        dc = self.devices.get(dev)
        if not dc:
            return
        if not dc.quarantined:
            return

        dc.quarantined = False
        dc.quarantine_since_ts = 0.0
        dc.last_quarantine_check_ts = 0.0
        dc.full_hash_fail_streak = 0

        fp_evt = self.events_fp.get(dev)
        if fp_evt:
            self._jwrite(fp_evt, {
                "ts_ms": ts_ms(),
                "device": dev,
                "event": "quarantine_exited",
                "reason": reason,
            })

    async def _quarantine_periodic_recheck(self, dev: str):
        dc = self.devices.get(dev)
        if not dc or not dc.quarantined:
            return

        now = time.time()
        if (now - float(dc.last_quarantine_check_ts or 0.0)) < QUARANTINE_RECHECK_S:
            return

        dc.last_quarantine_check_ts = now

        fp_evt = self.events_fp.get(dev)
        if fp_evt:
            self._jwrite(fp_evt, {
                "ts_ms": ts_ms(),
                "device": dev,
                "event": "quarantine_recheck_start",
                "interval_s": QUARANTINE_RECHECK_S,
            })

        resp = await self.attest_full_and_log(dev, trigger="QUARANTINE_RECHECK", ml={
            "policy_reason": "quarantine_recheck",
        })

        ok = bool(isinstance(resp, dict) and resp.get("verify_ok", False))

        if ok:
            self._exit_quarantine(dev, reason="recheck_full_ok")

            try:
                if hasattr(self.policy, "reset_device"):
                    self.policy.reset_device(dev)
                else:
                    if hasattr(self.policy, "devices") and isinstance(self.policy.devices, dict):
                        self.policy.devices.pop(dev, None)
            except Exception:
                pass

            dc2 = self.devices.get(dev)
            if dc2:
                dc2.trust_state = TRUST_TRUSTED
                dc2.full_hash_fail_streak = 0

            if fp_evt:
                self._jwrite(fp_evt, {
                    "ts_ms": ts_ms(),
                    "device": dev,
                    "event": "quarantine_recheck_ok_exit",
                })
        else:
            if fp_evt:
                self._jwrite(fp_evt, {
                    "ts_ms": ts_ms(),
                    "device": dev,
                    "event": "quarantine_recheck_failed_stay",
                    "verify_reason": (resp.get("verify_reason") if isinstance(resp, dict) else "bad_resp"),
                })

    # ----------------- budget config loader -----------------
    def _load_budget_config(
        self, path: str
    ) -> Tuple[Dict[str, BudgetConfig], Dict[str, int], Dict[str, str], Dict[str, float]]:
        per_device_budget: Dict[str, BudgetConfig] = {}
        per_device_min_k: Dict[str, int] = {}
        device_level: Dict[str, str] = {}
        level_cov: Dict[str, float] = {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except FileNotFoundError:
            print(f"[{now_s()}] [BUDGET] No {path} found. Using defaults.")
            return per_device_budget, per_device_min_k, device_level, level_cov
        except Exception as e:
            print(f"[{now_s()}] [BUDGET] Failed to load {path}: {e}. Using defaults.")
            return per_device_budget, per_device_min_k, device_level, level_cov

        levels = cfg.get("levels", {}) or {}
        devices = cfg.get("devices", {}) or {}

        for lvl_name, lvl_cfg in levels.items():
            try:
                if isinstance(lvl_cfg, dict) and "coverage" in lvl_cfg:
                    level_cov[str(lvl_name)] = float(lvl_cfg["coverage"])
            except Exception:
                pass

        for dev, level_name in devices.items():
            lvl = levels.get(level_name, {}) or {}
            per_min = int(lvl.get("per_min_units", 50))
            cap = int(lvl.get("cap_units", per_min))
            min_k = int(lvl.get("min_k", 1))

            per_device_budget[dev] = BudgetConfig(per_min_units=per_min, cap_units=cap, min_k=max(1, min_k))
            per_device_min_k[dev] = max(1, min_k)
            device_level[dev] = str(level_name)

            if str(level_name) not in level_cov:
                level_cov[str(level_name)] = 1.0

        print(f"[{now_s()}] [BUDGET] loaded {len(per_device_budget)} device budgets from {path}")
        return per_device_budget, per_device_min_k, device_level, level_cov

    # ----------------- file helpers -----------------
    def _open_files_for(self, dev: str):
        if dev in self.windows_fp:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        wpath = str(Path(LOG_DIR) / f"windows_{dev}_{stamp}.jsonl")
        epath = str(Path(LOG_DIR) / f"events_{dev}_{stamp}.jsonl")
        apath = str(Path(LOG_DIR) / f"attest_{dev}_{stamp}.jsonl")
        self.windows_fp[dev] = open(wpath, "a", encoding="utf-8", buffering=1)
        self.events_fp[dev] = open(epath, "a", encoding="utf-8", buffering=1)
        self.attest_fp[dev] = open(apath, "a", encoding="utf-8", buffering=1)
        print(f"[{now_s()}] files for {dev}: windows={wpath} events={epath} attest={apath}")

    @staticmethod
    def _jwrite(fp, obj: dict):
        fp.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        fp.flush()

    def _close_files_for(self, dev: str):
        for dct in (self.windows_fp, self.events_fp, self.attest_fp):
            fp = dct.pop(dev, None)
            try:
                if fp:
                    fp.close()
            except Exception:
                pass

    async def force_provision_golden_full(self, dev: str, region: str = "fw") -> dict:
        nonce = secrets.token_hex(8)
        resp = await self.send_request(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "FULL_HASH_PROVER",
            "region": region,
            "nonce": nonce
        }, timeout=12.0)

        fw_hex = resp.get("fw_hash_hex")
        if not fw_hex:
            return {"type": "ERROR", "reason": "missing_fw_hash_hex_in_response", "resp": resp}

        self.set_golden_full_hash(dev, region, fw_hex)

        return {"type": "OK", "event": "golden_overwritten", "device": dev, "region": region, "fw_hash_hex": fw_hex}

    def set_golden_full_hash(self, device_id: str, region: str, fw_hash_hex: str):
        if device_id not in self.golden:
            self.golden[device_id] = {}
        if region not in self.golden[device_id]:
            self.golden[device_id][region] = {}
        self.golden[device_id][region]["sha256"] = fw_hash_hex.lower()
        save_json_atomic(GOLDEN_PATH, self.golden)

    def set_golden_blocks(self, device_id: str, block_size: int, hashes_hex: list[str], force: bool = False):
        if device_id not in self.golden:
            self.golden[device_id] = {}

        if (not force) and self.has_golden_blocks(device_id):
            raise RuntimeError("golden_blocks_already_exist_refusing_overwrite")

        self.golden[device_id]["blocks"] = {
            "block_size": int(block_size),
            "block_count": int(len(hashes_hex)),
            "hashes": [h.lower() for h in hashes_hex],
        }
        save_json_atomic(GOLDEN_PATH, self.golden)

    async def provision_golden_blocks(self, dev: str, force: bool = False) -> dict:
        if self.has_golden_blocks(dev) and not force:
            return {"type": "ERROR", "reason": "golden_blocks_already_exists_refusing_overwrite", "device": dev}

        caps = self._caps(dev)
        max_req = int(caps.get("max_req_blocks", 32) or 32)
        if max_req <= 0:
            max_req = 32

        nonce = secrets.token_hex(8)
        probe = await self.send_request(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "PARTIAL_BLOCKS",
            "region": "fw",
            "nonce": nonce,
            "indices": [0]
        }, timeout=12.0)

        if probe.get("type") != "ATTEST_RESPONSE" or probe.get("mode") != "PARTIAL_BLOCKS":
            return {"type": "ERROR", "reason": "bad_probe_response", "resp": probe}

        block_count = int(probe.get("block_count", 0) or 0)

        block_size = int(probe.get("block_size", 0) or 0)
        if block_size <= 0:
            blocks0 = probe.get("blocks", []) or []
            if blocks0 and isinstance(blocks0, list):
                block_size = int(blocks0[0].get("len", 0) or 0)

        if block_size <= 0 or block_count <= 0:
            return {"type": "ERROR", "reason": "missing_block_meta", "resp": probe}

        fw_blocks_n_hello = int(caps.get("fw_blocks_n", 0) or 0)
        if fw_blocks_n_hello and fw_blocks_n_hello != block_count:
            fp_evt = self.events_fp.get(dev)
            if fp_evt:
                self._jwrite(fp_evt, {
                    "ts_ms": ts_ms(),
                    "device": dev,
                    "event": "blocks_meta_mismatch",
                    "hello_fw_blocks_n": fw_blocks_n_hello,
                    "prover_block_count": block_count
                })

        got: Dict[int, str] = {}

        for start in range(0, block_count, max_req):
            end = min(block_count, start + max_req)
            chunk = list(range(start, end))

            nonce = secrets.token_hex(8)
            resp = await self.send_request(dev, {
                "type": "ATTEST_REQUEST",
                "mode": "PARTIAL_BLOCKS",
                "region": "fw",
                "nonce": nonce,
                "indices": chunk
            }, timeout=20.0)

            if resp.get("type") != "ATTEST_RESPONSE" or resp.get("mode") != "PARTIAL_BLOCKS":
                return {"type": "ERROR", "reason": "bad_blocks_response", "resp": resp, "range": [start, end]}

            blocks = resp.get("blocks", []) or []
            for b in blocks:
                if "index" in b and "hash_hex" in b:
                    got[int(b["index"])] = b["hash_hex"]

            fp_evt = self.events_fp.get(dev)
            if fp_evt:
                self._jwrite(fp_evt, {
                    "ts_ms": ts_ms(),
                    "device": dev,
                    "event": "provision_blocks_progress",
                    "got": len(got),
                    "need": block_count,
                    "last_range": [start, end],
                    "max_req_blocks": max_req
                })

        if len(got) < block_count:
            return {"type": "ERROR", "reason": "missing_some_blocks", "got": len(got), "need": block_count}

        hashes = [got[i] for i in range(block_count)]

        try:
            self.set_golden_blocks(dev, block_size, hashes, force=force)
        except RuntimeError as e:
            return {"type": "ERROR", "reason": str(e)}

        return {
            "type": "OK",
            "event": "golden_blocks_provisioned" if not force else "golden_blocks_overwritten",
            "device": dev,
            "block_size": block_size,
            "block_count": block_count,
            "max_req_blocks": max_req
        }

    # ----------------- LRU persistence -----------------
    def _load_lru_state(self):
        try:
            with open(LRU_STATE_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for dev, st in raw.items():
                    if isinstance(st, dict):
                        self.block_lru[dev] = DeviceLRUBlocks.from_state(st)
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def _save_lru_state(self):
        try:
            blob = {dev: lru.export_state() for dev, lru in self.block_lru.items()}
            save_json_atomic(LRU_STATE_PATH, blob)
        except Exception as e:
            print(f"[{now_s()}] [LRU] FAILED saving to {Path(LRU_STATE_PATH).resolve()}: {e}")
            traceback.print_exc()

    def _get_block_lru(self, dev: str) -> Optional[DeviceLRUBlocks]:
        bc = self.get_block_count(dev)
        if bc <= 0:
            return None
        lru = self.block_lru.get(dev)
        if lru is None:
            seed = (hash(dev) & 0xFFFFFFFF)
            lru = DeviceLRUBlocks.fresh(bc, seed=seed)
           #lru = DeviceLRUBlocks.fresh(bc)  # determ
            self.block_lru[dev] = lru
            self._save_lru_state()
        else:
            lru.ensure_size(bc)
        return lru

    # ----------------- locks -----------------
    def _attest_lock(self, dev: str) -> asyncio.Lock:
        lk = self.attest_locks.get(dev)
        if lk is None:
            lk = asyncio.Lock()
            self.attest_locks[dev] = lk
        return lk

    # ----------------- golden access -----------------
    def golden_full_hash(self, device_id: str, region: str = "fw") -> Optional[bytes]:
        try:
            return unhex(self.golden[device_id][region]["sha256"])
        except Exception:
            return None

    def has_golden_blocks(self, device_id: str) -> bool:
        try:
            _ = self.golden[device_id]["blocks"]["hashes"]
            return True
        except Exception:
            return False

    def get_block_count(self, device_id: str) -> int:
        try:
            return int(self.golden[device_id]["blocks"]["block_count"])
        except Exception:
            return 0

    def golden_block_hash(self, device_id: str, index: int) -> Optional[bytes]:
        try:
            return unhex(self.golden[device_id]["blocks"]["hashes"][index])
        except Exception:
            return None

    # ----------------- logging attest -----------------
    def log_attest_event(
        self,
        dev: str,
        kind: str,
        k: Optional[int],
        indices: Optional[List[int]],
        resp: dict,
        trust_before: str,
        trust_after: str,
        trigger: str = "POLICY",
        ml: Optional[dict] = None,
    ):
        fp = self.attest_fp.get(dev)
        if not fp:
            return

        # --- summarize indices ---
        idx_list = indices if isinstance(indices, list) else None
        idx_count = len(idx_list) if idx_list is not None else None

        # small sample only (so logs don't explode)
        idx_sample = None
        if idx_list is not None:
            if len(idx_list) <= 16:
                idx_sample = idx_list
            else:
                idx_sample = idx_list[:8] + ["..."] + idx_list[-8:]

        self._jwrite(fp, {
            "ts_ms": ts_ms(),
            "device": dev,
            "event": "attest",
            "attest_kind": kind,
            "trigger": trigger,
            "ml": ml,
            "k": k,

            # ✅ instead of dumping full indices
            "indices_count": idx_count,
            "indices_sample": idx_sample,

            "trust_before": trust_before,
            "trust_after": trust_after,
            "verify_ok": resp.get("verify_ok"),
            "verify_reason": resp.get("verify_reason", resp.get("reason")),
            "rtt_ms": resp.get("_rtt_ms"),
            "req_bytes": resp.get("_req_bytes"),
            "resp_bytes": resp.get("_resp_bytes"),
            "budget_tokens_after": resp.get("_budget_tokens_after"),
            "budget_cost_units": resp.get("_budget_cost_units"),

            # ✅ extra batch metadata if present
            "batched": resp.get("_batched"),
            "batch_size": resp.get("_batch_size"),
            "batch_index": resp.get("_batch_index"),
            "batch_count": resp.get("_batch_count"),
            "k_total": resp.get("_k_total"),
        })

    # ----------------- RX loop -----------------
    async def rx_loop(self, dc: DeviceConn):
        while True:
            line = await dc.reader.readline()
            if not line:
                return
            dc.last_seen_ts = time.time()
            try:
                msg = json.loads(line.decode("utf-8"))
            except Exception:
                print(f"[{now_s()}] [RX raw] {dc.device_id}: {line!r}")
                continue

            req_id = msg.get("req_id")
            if req_id and req_id in dc.pending:
                pending = dc.pending.pop(req_id)
                verified_msg = self.verify_if_needed(dc.device_id, pending.sent_msg, msg)
                if not pending.fut.done():
                    pending.fut.set_result(verified_msg)
            else:
                print(f"[{now_s()}] [RX] {dc.device_id}: {msg}")

    # ----------------- request send -----------------
    async def send_request(self, device_id: str, msg: dict, timeout: float = 5.0) -> dict:
        dc = self.devices.get(device_id)
        if not dc or not dc.is_alive():
            return {"type": "ERROR", "reason": "device_not_connected"}

        req_id = secrets.token_hex(8)
        msg = dict(msg)
        msg["req_id"] = req_id

        fut = self.loop.create_future()
        dc.pending[req_id] = PendingReq(fut=fut, sent_msg=msg)

        dc.writer.write(jdump(msg))
        await dc.writer.drain()

        try:
            resp = await asyncio.wait_for(fut, timeout=timeout)
            return resp
        except asyncio.TimeoutError:
            dc.pending.pop(req_id, None)
            return {"type": "ERROR", "reason": "timeout_waiting_response", "req_id": req_id}

    async def send_request_timed(self, device_id: str, msg: dict, timeout: float = 5.0) -> dict:
        req_line = jdump({**msg, "req_id": "0000000000000000"})
        req_bytes = len(req_line)
        t0 = time.perf_counter()
        resp = await self.send_request(device_id, msg, timeout=timeout)
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        resp_bytes = len(jdump(resp)) if isinstance(resp, dict) else 0
        if isinstance(resp, dict):
            resp["_rtt_ms"] = round(rtt_ms, 2)
            resp["_req_bytes"] = int(req_bytes)
            resp["_resp_bytes"] = int(resp_bytes)
        return resp

    # ----------------- verification -----------------
    def verify_if_needed(self, device_id: str, sent: dict, received: dict) -> dict:
        mode = sent.get("mode")
        rtype = received.get("type")

        if rtype not in ("ATTEST_RESPONSE", "PONG", "WINDOWS"):
            return received

        # FULL
        if mode == "FULL_HASH_PROVER" and rtype == "ATTEST_RESPONSE":
            golden = self.golden_full_hash(device_id, region=sent.get("region", "fw"))
            if golden is None:
                received["verify_ok"] = False
                received["verify_reason"] = "missing_golden_full_hash"
                return received

            nonce_hex = sent.get("nonce")
            if "response_hex" in received and nonce_hex:
                nonce = unhex(nonce_hex)
                expected = sha256(nonce + golden)
                got = unhex(received["response_hex"])
                ok = (got == expected)
                received["verify_ok"] = ok
                received["verify_reason"] = "nonce_bound_match" if ok else "nonce_bound_mismatch"
                return received

            received["verify_ok"] = False
            received["verify_reason"] = "missing_hash_fields"
            return received

        # PARTIAL
        if mode == "PARTIAL_BLOCKS" and rtype == "ATTEST_RESPONSE":
            if not self.has_golden_blocks(device_id):
                received["verify_ok"] = False
                received["verify_reason"] = "missing_golden_blocks"
                return received

            nonce_hex = sent.get("nonce")
            nonce = unhex(nonce_hex) if nonce_hex else None

            blocks = received.get("blocks", []) or []
            all_ok = True
            reasons = []

            for b in blocks:
                idx = b.get("index")
                if idx is None:
                    all_ok = False
                    reasons.append("block_missing_index")
                    continue

                golden_b = self.golden_block_hash(device_id, int(idx))
                if golden_b is None:
                    all_ok = False
                    reasons.append(f"missing_golden_block_{idx}")
                    continue

                ok = False
                if nonce is not None and "response_hex" in b:
                    expected = sha256(nonce + golden_b)
                    got = unhex(b["response_hex"])
                    ok = (got == expected)
                elif "hash_hex" in b:
                    ok = (unhex(b["hash_hex"]) == golden_b)

                if not ok:
                    all_ok = False
                    reasons.append(f"block_{idx}_mismatch")

            received["verify_ok"] = all_ok
            received["verify_reason"] = "ok" if all_ok else ",".join(reasons)
            return received

        return received

    # ----------------- trust update -----------------
    def _update_trust_from_attest(self, dev: str, resp: dict, attempt: int):
        dc = self.devices.get(dev)
        if not dc:
            return

        reason = resp.get("verify_reason", resp.get("reason", "unknown"))

        if reason in ("missing_golden_full_hash", "missing_golden_blocks", "no_golden_blocks"):
            dc.trust_state = TRUST_UNKNOWN
            dc.attest_fail_streak = 0
            return

        ok = bool(resp.get("verify_ok", False))
        if ok:
            dc.trust_state = TRUST_TRUSTED
            dc.attest_fail_streak = 0
            dc.last_attest_ok_ts = time.time()
        else:
            dc.attest_fail_streak += 1
            dc.last_attest_fail_ts = time.time()
            if dc.attest_fail_streak >= 2:
                dc.trust_state = TRUST_UNTRUSTED

    # ----------------- attest actions -----------------
    async def attest_full_once(self, dev: str, timeout: float = 8.0) -> dict:
        nonce = secrets.token_hex(8)
        resp = await self.send_request_timed(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "FULL_HASH_PROVER",
            "region": "fw",
            "nonce": nonce
        }, timeout=timeout)
        return resp

    async def attest_full_with_retry(self, dev: str) -> dict:
        resp1 = await self.attest_full_once(dev, timeout=8.0)
        self._update_trust_from_attest(dev, resp1 if isinstance(resp1, dict) else {}, attempt=1)
        if resp1.get("verify_ok", False):
            return resp1
        await asyncio.sleep(1.0)
        resp2 = await self.attest_full_once(dev, timeout=8.0)
        self._update_trust_from_attest(dev, resp2 if isinstance(resp2, dict) else {}, attempt=2)
        return resp2

    async def attest_full_and_log(self, dev: str, trigger: str = "POLICY", ml: Optional[dict] = None) -> dict:
        async with self._attest_lock(dev):
            dc = self.devices.get(dev)
            if not dc:
                return {"type": "ERROR", "reason": "device_not_connected"}

            trust_before = dc.trust_state
            resp = await self.attest_full_with_retry(dev)
            trust_after = dc.trust_state if dev in self.devices else TRUST_UNKNOWN

            if isinstance(resp, dict):
                ok = bool(resp.get("verify_ok", False))

                if ok:
                    dc.full_hash_fail_streak = 0
                    if getattr(dc, "quarantined", False):
                        self._exit_quarantine(dev, reason="full_hash_ok")

                else:
                    # RESET HGB baseline on any FULL failure
                    ge = self.gate_hgb_engines.get(dev)
                    if ge is not None:
                        ge.reset(reason="full_attest_fail")
                    
                    dc.full_hash_fail_streak = int(getattr(dc, "full_hash_fail_streak", 0)) + 1
                    
                    # ✅ NEW: Immediate recheck on FIRST mismatch (before entering quarantine)
                    if dc.full_hash_fail_streak == 1:
                        self.logger.warning(f"[{dev}] FULL mismatch detected (streak=1), scheduling immediate recheck...")
                        
                        # Log the first failure
                        self.log_attest_event(
                            dev=dev, kind="FULL", k=None, indices=None,
                            resp=resp if isinstance(resp, dict) else {"type": "ERROR", "reason": "bad_resp"},
                            trust_before=trust_before, trust_after=trust_after,
                            trigger=trigger, ml=ml
                        )
                        
                        # Small delay to allow device to recover/stabilize
                        await asyncio.sleep(0.5)
                        
                        # Immediate retry
                        self.logger.info(f"[{dev}] Performing immediate FULL recheck...")
                        resp_retry = await self.attest_full_with_retry(dev)
                        trust_after = dc.trust_state if dev in self.devices else TRUST_UNKNOWN
                        
                        if isinstance(resp_retry, dict):
                            ok_retry = bool(resp_retry.get("verify_ok", False))
                            
                            if ok_retry:
                                # Recovery successful!
                                dc.full_hash_fail_streak = 0
                                self.logger.info(f"[{dev}]  FULL recheck SUCCESS - recovered from mismatch")
                                
                                self.log_attest_event(
                                    dev=dev, kind="FULL", k=None, indices=None,
                                    resp=resp_retry,
                                    trust_before=TRUST_UNTRUSTED, trust_after=trust_after,
                                    trigger="MISMATCH_RECHECK", 
                                    ml={"policy_reason": "mismatch_recheck_success"}
                                )
                                
                                # Override response with successful retry
                                resp = resp_retry
                            else:
                                # Retry also failed - NOW we have 2 consecutive failures
                                dc.full_hash_fail_streak = 2
                                self.logger.error(f"[{dev}] ✗ FULL recheck FAILED - 2 consecutive failures, entering quarantine")
                                
                                self.log_attest_event(
                                    dev=dev, kind="FULL", k=None, indices=None,
                                    resp=resp_retry,
                                    trust_before=TRUST_UNTRUSTED, trust_after=trust_after,
                                    trigger="MISMATCH_RECHECK", 
                                    ml={"policy_reason": "mismatch_recheck_failed"}
                                )
                                
                                # Use retry response for final processing
                                resp = resp_retry
                    
                    # Enter quarantine if we have 2+ consecutive failures
                    if dc.full_hash_fail_streak >= QUARANTINE_ON_FULL_FAILS:
                        self._enter_quarantine(
                            dev,
                            reason=f"full_hash_failed_{dc.full_hash_fail_streak}x:{resp.get('verify_reason')}"
                        )


            if isinstance(resp, dict) and ml and (not ml.get("budget_bypass", False)):
                resp["_budget_cost_units"] = ml.get("budget_cost_units")
                resp["_budget_tokens_after"] = ml.get("budget_tokens_after")

            self.log_attest_event(
                dev=dev, kind="FULL", k=None, indices=None,
                resp=resp if isinstance(resp, dict) else {"type": "ERROR", "reason": "bad_resp"},
                trust_before=trust_before, trust_after=trust_after,
                trigger=trigger, ml=ml
            )
            return resp

    async def _attest_full_inner_and_log(
        self,
        dev: str,
        trigger: str = "POLICY",
        ml: Optional[dict] = None,
        force_quarantine_on_fail: bool = False,
    ) -> dict:
        dc = self.devices.get(dev)
        if not dc:
            return {"type": "ERROR", "reason": "device_not_connected"}

        trust_before = dc.trust_state
        resp = await self.attest_full_with_retry(dev)
        trust_after = dc.trust_state if dev in self.devices else TRUST_UNKNOWN

        if isinstance(resp, dict):
            ok = bool(resp.get("verify_ok", False))
            if ok:
                dc.full_hash_fail_streak = 0
                if getattr(dc, "quarantined", False):
                    self._exit_quarantine(dev, reason="full_hash_ok")
            else:
                # RESET HGB baseline on any FULL failure
                ge = self.gate_hgb_engines.get(dev)
                if ge is not None:
                    ge.reset(reason="full_attest_fail")

                dc.full_hash_fail_streak = int(getattr(dc, "full_hash_fail_streak", 0)) + 1
                if force_quarantine_on_fail:
                    self._enter_quarantine(dev, reason=f"escalated_full_failed:{resp.get('verify_reason')}")
                else:
                    if dc.full_hash_fail_streak >= QUARANTINE_ON_FULL_FAILS:
                        self._enter_quarantine(
                            dev,
                            reason=f"full_hash_failed_{dc.full_hash_fail_streak}x:{resp.get('verify_reason')}"
                        )

        if isinstance(resp, dict) and ml and (not ml.get("budget_bypass", False)):
            resp["_budget_cost_units"] = ml.get("budget_cost_units")
            resp["_budget_tokens_after"] = ml.get("budget_tokens_after")

        self.log_attest_event(
            dev=dev, kind="FULL", k=None, indices=None,
            resp=resp if isinstance(resp, dict) else {"type": "ERROR", "reason": "bad_resp"},
            trust_before=trust_before, trust_after=trust_after,
            trigger=trigger, ml=ml
        )
        return resp

    async def attest_partial_once(self, dev: str, k: int, timeout: float = 12.0) -> dict:
        bc = self.get_block_count(dev)
        if bc <= 0:
            return {"type": "ERROR", "reason": "no_golden_blocks"}

        k = max(1, min(int(k), bc))
        lru = self._get_block_lru(dev)
        if lru is None:
            return {"type": "ERROR", "reason": "no_golden_blocks"}

        indices = lru.pick(k, pool_frac=0.25, pool_min=32)
        #indices = sorted(lru.pick(k))  # determ
        nonce = secrets.token_hex(8)

        resp = await self.send_request_timed(dev, {
            "type": "ATTEST_REQUEST",
            "mode": "PARTIAL_BLOCKS",
            "region": "fw",
            "nonce": nonce,
            "indices": indices
        }, timeout=timeout)

        if isinstance(resp, dict):
            resp["_k"] = k
            resp["_indices"] = indices
        return resp

    async def attest_partial_and_log(self, dev: str, k: int, trigger: str = "POLICY", ml: Optional[dict] = None) -> dict:
        async with self._attest_lock(dev):
            dc = self.devices.get(dev)
            if not dc:
                return {"type": "ERROR", "reason": "device_not_connected"}

            trust_before = dc.trust_state

            resp = await self.attest_partial_batched_once(dev, k=k, timeout=12.0)
            self._update_trust_from_attest(dev, resp if isinstance(resp, dict) else {}, attempt=1)

            # FIX: reset HGB baseline ONLY on PARTIAL FAIL (όχι στο success)
            if isinstance(resp, dict) and (not bool(resp.get("verify_ok", False))):
                ge = self.gate_hgb_engines.get(dev)
                if ge is not None:
                    ge.reset(reason="partial_attest_fail")
            else:
                # success -> touch LRU
                idxs = resp.get("_indices") if isinstance(resp, dict) else []
                lru = self._get_block_lru(dev)
                if lru is not None and idxs:
                    lru.touch(idxs)
                    self._save_lru_state()

            trust_after = dc.trust_state if dev in self.devices else TRUST_UNKNOWN
            indices = resp.get("_indices_all") if isinstance(resp, dict) else None
            kk = resp.get("_k_total") if isinstance(resp, dict) else k

            if isinstance(resp, dict) and ml:
                resp["_budget_cost_units"] = ml.get("budget_cost_units")
                resp["_budget_tokens_after"] = ml.get("budget_tokens_after")

            self.log_attest_event(
                dev=dev, kind="PARTIAL", k=kk, indices=indices,
                resp=resp if isinstance(resp, dict) else {"type": "ERROR", "reason": "bad_resp"},
                trust_before=trust_before, trust_after=trust_after,
                trigger=trigger, ml=ml
            )

            if isinstance(resp, dict) and not bool(resp.get("verify_ok", False)):
                fp_evt = self.events_fp.get(dev)
                if fp_evt:
                    self._jwrite(fp_evt, {
                        "ts_ms": ts_ms(),
                        "device": dev,
                        "event": "partial_failed_escalate_full",
                        "partial_reason": resp.get("verify_reason", resp.get("reason")),
                    })

                ml2 = dict(ml or {})
                ml2["escalation_from"] = "PARTIAL_FAIL"

                ml2["budget_bypass"] = True
                for kk2 in ("budget_reason", "budget_cost_units", "budget_tokens_after", "budget_tokens_before",
                           "budget_fit_kind", "budget_fit_k"):
                    ml2.pop(kk2, None)

                full_resp = await self._attest_full_inner_and_log(
                    dev=dev,
                    trigger="ESCALATE_FROM_PARTIAL",
                    ml=ml2,
                    force_quarantine_on_fail=True,
                )
                return full_resp

            return resp

    async def policy_loop(self, dev: str):
        while dev in self.devices and (not self.has_golden_full(dev, "fw")):
            await asyncio.sleep(0.2)

        if dev not in self.devices:
            return

        self._open_files_for(dev)
        fp_evt = self.events_fp.get(dev)

        if DO_INITIAL_FULL_ATTEST and self.has_golden_full(dev, "fw"):
            await asyncio.sleep(0.5)
            if dev in self.devices and self.has_golden_full(dev, "fw"):
                if fp_evt:
                    self._jwrite(fp_evt, {"ts_ms": ts_ms(), "device": dev, "event": "initial_full_attest_start"})
                await self.attest_full_and_log(dev, trigger="INITIAL")

        while True:
            if dev not in self.devices:
                return

            now = time.time()
            dc = self.devices.get(dev)
            if not dc:
                return

            decision = self.policy.tick(dev, now=now)

            # =========================
            # QUARANTINE GATE
            # =========================
            if getattr(dc, "quarantined", False):
                if fp_evt:
                    if decision.do_get_windows:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "quarantine_skip_get_windows",
                            "policy_reason": decision.reason
                        })
                    if decision.attest_kind != PLAttestKind.NONE:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "quarantine_skip_policy_attest",
                            "policy_attest_kind": str(decision.attest_kind),
                            "policy_reason": decision.reason
                        })

                await self._quarantine_periodic_recheck(dev)
                await asyncio.sleep(LOOP_TICK_S + random.random() * JITTER_S)
                continue

            # =========================
            # 1) GET_WINDOWS (if due)
            # =========================
            if decision.do_get_windows:
                lk = self._attest_lock(dev)
                if lk.locked():
                    if fp_evt:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "policy_skip_get_windows_attest_inflight",
                            "reason": decision.reason
                        })
                else:
                    since = int(self.last_seen.get(dev, 0))
                    t0 = time.perf_counter()

                    resp = await self.send_request(
                        dev,
                        {
                            "type": "GET_WINDOWS",
                            "since": since,
                            "max": int(decision.get_windows_max),
                        },
                        timeout=8.0
                    )

                    rtt_ms = int((time.perf_counter() - t0) * 1000)

                    if fp_evt:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "policy_get_windows_done",
                            "since": since,
                            "max": int(decision.get_windows_max),
                            "rtt_ms": rtt_ms,
                            "resp_type": resp.get("type"),
                            "reason": decision.reason
                        })

                    if resp.get("type") == "WINDOWS":
                        windows = resp.get("windows", []) or []

                        to_id = resp.get("to", None)
                        if to_id is None and windows:
                            try:
                                to_id = int(windows[-1].get("window_id"))
                            except Exception:
                                to_id = None
                        if to_id is not None:
                            try:
                                self.last_seen[dev] = int(to_id) + 1
                            except Exception:
                                pass

                        fp_win = self.windows_fp.get(dev)
                        dc2 = self.devices.get(dev)
                        trust = dc2.trust_state if dc2 else TRUST_UNKNOWN
                        tnow = time.time()

                        if fp_win:
                            for w in windows:
                                self._jwrite(fp_win, {
                                    "ts": tnow,
                                    "device_id_str": dev,
                                    "trust_state": trust,
                                    "trusted_for_decision": (trust == TRUST_TRUSTED),
                                    **w
                                })

                        # 2) ML inference on windows -> update policy engine
                        if ML_ENABLE and windows:
                            gate_labels: List[GateLabel] = []
                            wl_labels: List[WorkloadLabel] = []
                            atk_labels: List[Attack2BLabel] = []

                            gate_counts: Dict[str, int] = {}
                            wl_counts: Dict[str, int] = {}
                            atk_counts: Dict[str, int] = {}

                            gate_confs: List[float] = []
                            wl_confs: List[float] = []
                            atk_confs: List[float] = []

                            ok_cnt = 0
                            # ---- DEBUG: gate source accounting ----
                            gate_src_counts = {"HGB": 0, "LR_FALLBACK": 0, "HGB_WARMUP": 0, "DROPPED": 0}
                            hgb_p_vals: List[float] = []
                            hgb_thr = None
                            hgb_W = None

                            raw_atk_seen = []

                            def majority_from_counts(counts: dict, default: str = "UNKNOWN"):
                                if not counts:
                                    return default, 0.0, 0
                                total = sum(int(v) for v in counts.values())
                                if total <= 0:
                                    return default, 0.0, 0
                                maj = max(counts.items(), key=lambda kv: int(kv[1]))[0]
                                frac = float(counts[maj]) / float(total)
                                return maj, frac, total

                            for w in windows:
                                # old pipeline for WL/ATK and fallback
                                pr = self.lr_policy.predict(dev, w) if self.lr_policy is not None else {"ok": False}

                                g_final = "UNCERTAIN"
                                g_conf = None
                                g_src = "DROPPED"  # will change

                                # --- HGB gate ---
                                ge = self.gate_hgb_engines.get(dev)
                                if ge is not None:
                                    hgb_thr = getattr(ge, "thr", None)
                                    hgb_W = getattr(ge, "W", None)
                                    gout = ge.update(w, now=time.time())

                                    if isinstance(gout, dict):
                                        g_final = "COMPROMISED" if int(gout["pred"]) == 1 else "SAFE"
                                        g_conf = float(gout.get("conf")) if gout.get("conf") is not None else None
                                        g_src = "HGB"
                                        try:
                                            hgb_p_vals.append(float(gout.get("p")))
                                        except Exception:
                                            pass
                                    else:
                                        # HGB exists but not ready yet (armed? warmup? W fill?)
                                        g_src = "HGB_WARMUP"

                                # --- fallback to LR only if HGB not ready ---
                                if g_final == "UNCERTAIN":
                                    if isinstance(pr, dict) and pr.get("ok"):
                                        g_tmp = (pr.get("gate_label") or "UNCERTAIN").upper()
                                        if g_tmp in ("SAFE", "COMPROMISED", "UNCERTAIN"):
                                            g_final = g_tmp
                                            try:
                                                g_conf = float(pr.get("gate_conf")) if pr.get("gate_conf") is not None else None
                                            except Exception:
                                                g_conf = None
                                            g_src = "LR_FALLBACK"

                                # if still uncertain and no LR -> drop this window from gate batch
                                # --- IMPORTANT FIX: never drop windows from gate batch ---
                                # If HGB isn't ready (warmup) and LR doesn't give an answer, keep this window as UNCERTAIN.
                                if g_final == "UNCERTAIN" and (g_src in ("DROPPED", "HGB_WARMUP")):
                                    gate_src_counts["DROPPED" if g_src == "DROPPED" else "HGB_WARMUP"] += 1

                                    # Keep it in batch as UNCERTAIN (prevents fake majorities).
                                    g_enum = GateLabel.UNCERTAIN
                                    gate_labels.append(g_enum)
                                    gate_counts["UNCERTAIN"] = gate_counts.get("UNCERTAIN", 0) + 1
                                    ok_cnt += 1
                                    continue

                                # count source
                                if g_src in gate_src_counts:
                                    gate_src_counts[g_src] += 1
                                else:
                                    gate_src_counts[g_src] = 1

                                ok_cnt += 1

                                g_enum = GateLabel(g_final)
                                gate_labels.append(g_enum)
                                gate_counts[g_final] = gate_counts.get(g_final, 0) + 1
                                if g_conf is not None:
                                    gate_confs.append(float(g_conf))

                                if g_enum == GateLabel.SAFE:
                                    wl = "UNKNOWN"
                                    wc = None
                                    if isinstance(pr, dict) and pr.get("ok"):
                                        wl = (pr.get("workload_label") or "UNKNOWN").upper()
                                        if wl not in ("LIGHT", "MEDIUM", "HEAVY", "UNKNOWN"):
                                            wl = "UNKNOWN"
                                        wc = pr.get("workload_conf")
                                    wl_enum = WorkloadLabel(wl)
                                    wl_labels.append(wl_enum)
                                    wl_counts[wl] = wl_counts.get(wl, 0) + 1
                                    if wc is not None:
                                        try:
                                            wl_confs.append(float(wc))
                                        except Exception:
                                            pass

                                if g_enum == GateLabel.COMPROMISED:
                                    pr_atk = self.lr_policy.predict_attack2b_only(dev, w)

                                    raw_atk = pr_atk.get("attack2b_label") if pr_atk.get("ok") else None
                                    atk = (raw_atk or "UNKNOWN").upper()
                                    if atk not in ("INJECTION", "INTERRUPTION", "UNKNOWN"):
                                        atk = "UNKNOWN"

                                    ac = pr_atk.get("attack2b_conf")

                                    raw_atk_seen.append(raw_atk)

                                    atk_enum = Attack2BLabel(atk)
                                    atk_labels.append(atk_enum)
                                    atk_counts[atk] = atk_counts.get(atk, 0) + 1
                                    if ac is not None:
                                        try:
                                            atk_confs.append(float(ac))
                                        except Exception:
                                            pass

                            atk_maj, atk_maj_frac, atk_n = majority_from_counts(atk_counts, default="UNKNOWN")

                            gate_summ = self.policy.on_gate_batch(dev, gate_labels, now=time.time())
                            wl_summ = self.policy.on_workload_batch(dev, wl_labels, now=time.time())
                            atk_summ = self.policy.on_attack2b_batch(dev, atk_labels, now=time.time())

                            # store batch majority so ATTEST can log it
                            self.last_batch_majority[dev] = {
                                "gate_majority": gate_summ.majority,
                                "gate_frac": float(gate_summ.confidence) if gate_summ.n > 0 else None,
                                "wl_majority": wl_summ.majority,
                                "wl_frac": float(wl_summ.confidence) if wl_summ.n > 0 else None,
                                "attack2b_majority": atk_summ.majority,
                                "attack2b_frac": float(atk_summ.confidence) if atk_summ.n > 0 else None,
                                "attack2b_n": int(atk_summ.n),
                                "ts": time.time(),
                            }

                            decision = self.policy.tick(dev, now=time.time())
                            st = self.policy.devices.get(dev)

                            def _avg(xs):
                                return (sum(xs) / len(xs)) if xs else None

                            if fp_evt:
                                self._jwrite(fp_evt, {
                                    "ts_ms": ts_ms(),
                                    "device": dev,
                                    "event": "ml_inference_batch_2stage",
                                    "n_windows": len(windows),
                                    "n_ok": ok_cnt,
                                    "gate_counts": gate_counts,
                                    "workload_counts": wl_counts,
                                    "attack2b_counts": atk_counts,
                                    "gate_majority": gate_summ.majority,
                                    "gate_majority_frac": round(float(gate_summ.confidence), 3),
                                    "workload_majority": wl_summ.majority,
                                    "workload_majority_frac": round(float(wl_summ.confidence), 3) if wl_summ.n > 0 else None,
                                    "attack2b_majority": atk_summ.majority,
                                    "attack2b_majority_frac": round(float(atk_summ.confidence), 3) if atk_summ.n > 0 else None,
                                    "stable_gate": st.stable_gate.value if st else None,
                                    "stable_workload": st.stable_workload.value if st else None,
                                    "stable_attack2b": st.stable_attack2b.value if st else None,
                                    "policy_reason": st.last_reason if st else None,
                                    "gate_conf_avg": round(float(_avg(gate_confs)), 3) if _avg(gate_confs) is not None else None,
                                    "wl_conf_avg": round(float(_avg(wl_confs)), 3) if _avg(wl_confs) is not None else None,
                                    "atk_conf_avg": round(float(_avg(atk_confs)), 3) if _avg(atk_confs) is not None else None,
                                    "window_id_range": {
                                        "from": windows[0].get("window_id") if windows else None,
                                        "to": windows[-1].get("window_id") if windows else None,
                                    },
                                    "hgb": {
                                        "W": hgb_W,
                                        "thr": hgb_thr,
                                        "p_avg": round(float(sum(hgb_p_vals)/len(hgb_p_vals)), 4) if hgb_p_vals else None,
                                        "p_min": round(float(min(hgb_p_vals)), 4) if hgb_p_vals else None,
                                        "p_max": round(float(max(hgb_p_vals)), 4) if hgb_p_vals else None,
                                    },
                                    "gate_src_counts": gate_src_counts,
                                    "attack2b_majority_from_counts": atk_maj,
                                    "attack2b_majority_frac_from_counts": round(atk_maj_frac, 3),
                                })

            # =========================
            # 3) ATTEST (policy-driven) + BUDGET GATE
            # =========================
            if decision.attest_kind != PLAttestKind.NONE:
                st = self.policy.devices.get(dev)
                bm = self.last_batch_majority.get(dev, {})

                ml_meta = {
                    "stable_gate": st.stable_gate.value if st else None,
                    "stable_workload": st.stable_workload.value if st else None,
                    "stable_attack2b": st.stable_attack2b.value if st else None,

                    "last_gate": st.last_gate.value if st else None,
                    "last_gate_conf": round(float(st.last_gate_conf), 3) if st else None,
                    "policy_reason": st.last_reason if st else None,

                    # Majority from last ML batch (not "last_*")
                    "attack2b_majority": bm.get("attack2b_majority"),
                    "attack2b_majority_frac": round(float(bm["attack2b_frac"]), 3) if bm.get("attack2b_frac") is not None else None,
                    "attack2b_majority_n": bm.get("attack2b_n"),

                    "policy_decision_quarantine": bool(getattr(decision, "quarantine", False)),
                    "policy_decision_require_recheck": bool(getattr(decision, "require_recheck", False)),
                }

                # ---- HARD OVERRIDE: COMPROMISED + INJECTION => FULL immediately (no thresholds) ----
                
                bypass_budget = False
                try:
                    if st and st.stable_gate == GateLabel.COMPROMISED and st.stable_attack2b == Attack2BLabel.INJECTION:
                        bypass_budget = True
                except Exception:
                    bypass_budget = False

                if bypass_budget:
                    ml2 = dict(ml_meta)
                    ml2["budget_bypass"] = True
                    ml2["budget_reason"] = "bypass_budget_compromised_injection"

                    if fp_evt:
                        self._jwrite(fp_evt, {
                            "ts_ms": ts_ms(),
                            "device": dev,
                            "event": "bypass_full_request",
                            "reason": "COMPROMISED+INJECTION => FULL (no thresholds)",
                        })

                    try:
                        await self.attest_full_and_log(dev, trigger="POLICY_BYPASS_BUDGET", ml=ml2)
                    except Exception as e:
                        if fp_evt:
                            self._jwrite(fp_evt, {
                                "ts_ms": ts_ms(),
                                "device": dev,
                                "event": "bypass_full_failed",
                                "err": repr(e),
                            })

                    # cooldown ώστε να μην ξαναβαράει FULL κάθε tick
                    try:
                        st2 = self.policy.devices.get(dev)
                        if st2:
                            st2.next_attest_ts = time.time() + 2.0
                            st2.attest_cooldown_s = time.time() + 1.0
                    except Exception:
                        pass

                    await asyncio.sleep(LOOP_TICK_S + random.random() * JITTER_S)
                    continue


                if decision.attest_kind == PLAttestKind.FULL:
                    ideal_kind = "FULL"
                    ideal_k = 0
                    cov_used = None
                elif decision.attest_kind == PLAttestKind.PARTIAL:
                    ideal_kind = "PARTIAL"
                    real_bc = self.get_block_count(dev)
                    cov_used = float(getattr(decision, "coverage", 0.0) or 0.0)
                    cov_used = max(0.0, min(1.0, cov_used))
                    if real_bc > 0:
                        ideal_k = max(1, min(real_bc, int(math.ceil(cov_used * real_bc))))
                    else:
                        ideal_k = 0
                else:
                    ideal_kind = "NONE"
                    ideal_k = 0
                    cov_used = None

                real_bc = self.get_block_count(dev)
                min_k = int(self.device_min_k.get(dev, 1))
                now2 = time.time()

                tokens_before = self.budget.tokens_now(dev, now2)

                fit_kind, fit_k, cost_units, budget_reason = self.budget.fit_plan(
                    dev=dev,
                    now=now2,
                    ideal_kind=ideal_kind,
                    ideal_k=ideal_k,
                    block_count=max(1, real_bc if real_bc > 0 else 1),
                    min_k=min_k
                )

                if fit_kind == "PARTIAL" and real_bc <= 0:
                    fit_kind, fit_k, cost_units, budget_reason = ("NONE", 0, 0, "no_golden_blocks_for_partial")

                level = getattr(self, "device_level", {}).get(dev, "normal")

                target_cov = cov_used if (ideal_kind == "PARTIAL") else None
                ideal_cov = (ideal_k / real_bc) if (ideal_kind == "PARTIAL" and real_bc > 0) else None
                final_cov = (fit_k / real_bc) if (fit_kind == "PARTIAL" and real_bc > 0) else None

                if fp_evt:
                    self._jwrite(fp_evt, {
                        "ts_ms": ts_ms(),
                        "device": dev,
                        "event": "budget_gate",
                        "device_level": level,
                        "ideal": {
                            "kind": ideal_kind,
                            "k": ideal_k,
                            "target_coverage": round(float(target_cov), 3) if target_cov is not None else None,
                            "ideal_coverage": round(float(ideal_cov), 3) if ideal_cov is not None else None,
                        },
                        "fit": {
                            "kind": fit_kind,
                            "k": fit_k,
                            "cost_units": cost_units,
                            "final_coverage": round(float(final_cov), 3) if final_cov is not None else None,
                        },
                        "min_k": min_k,
                        "block_count": real_bc,
                        "tokens_before": round(tokens_before, 3),
                        "reason": budget_reason,
                    })

                if fit_kind != "NONE":
                    ok_spend = self.budget.spend(dev, cost_units, now2)
                    if not ok_spend:
                        if fp_evt:
                            self._jwrite(fp_evt, {
                                "ts_ms": ts_ms(),
                                "device": dev,
                                "event": "budget_spend_failed_after_fit",
                                "cost_units": cost_units
                            })
                    else:
                        tokens_after = self.budget.tokens_now(dev, now2)

                        ml_meta2 = dict(ml_meta)
                        ml_meta2["budget_reason"] = budget_reason
                        ml_meta2["budget_cost_units"] = int(cost_units)
                        ml_meta2["budget_tokens_after"] = round(tokens_after, 3)
                        ml_meta2["budget_tokens_before"] = round(tokens_before, 3)
                        ml_meta2["budget_fit_kind"] = fit_kind
                        ml_meta2["budget_fit_k"] = int(fit_k)
                        ml_meta2["device_level"] = level
                        ml_meta2["target_coverage"] = round(float(target_cov), 3) if target_cov is not None else None
                        ml_meta2["ideal_coverage"] = round(float(ideal_cov), 3) if ideal_cov is not None else None
                        ml_meta2["final_coverage"] = round(float(final_cov), 3) if final_cov is not None else None

                        if fit_kind == "FULL":
                            asyncio.create_task(self.attest_full_and_log(dev, trigger="POLICY", ml=ml_meta2))
                        else:
                            asyncio.create_task(self.attest_partial_and_log(dev, k=int(fit_k), trigger="POLICY", ml=ml_meta2))

            await asyncio.sleep(LOOP_TICK_S + random.random() * JITTER_S)

    # ----------------- client handler -----------------
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        print(f"[{now_s()}] [+] Connection from {peer}")

        async def _safe_close(w: asyncio.StreamWriter):
            try:
                w.close()
            except Exception:
                return
            with contextlib.suppress(Exception):
                await asyncio.wait_for(w.wait_closed(), timeout=1.0)

        try:
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        except asyncio.TimeoutError:
            print(f"[{now_s()}] [-] No HELLO from {peer}, closing")
            await _safe_close(writer)
            return

        if not line:
            await _safe_close(writer)
            return

        try:
            hello = json.loads(line.decode("utf-8"))
        except Exception:
            print(f"[{now_s()}] [-] Bad HELLO JSON from {peer}: {line!r}")
            await _safe_close(writer)
            return

        if hello.get("type") != "HELLO" or "device_id" not in hello:
            print(f"[{now_s()}] [-] Expected HELLO with device_id, got: {hello}")
            await _safe_close(writer)
            return

        device_id = hello["device_id"]

        fw_blocks_n = int(hello.get("fw_blocks_n", 0) or 0)
        max_req_blocks = int(hello.get("max_req_blocks", 32) or 32)

        self.device_caps[device_id] = {
            "fw_blocks_n": fw_blocks_n,
            "max_req_blocks": max_req_blocks,
        }
        self._save_device_caps()

        dc = DeviceConn(device_id=device_id, reader=reader, writer=writer)
        self.devices[device_id] = dc

        # ----------------- NEW: attach per-device policy cfg based on budget "device_level" -----------------
        lvl = getattr(self, "device_level", {}).get(device_id, "normal")
        pcfg = getattr(self, "policy_cfg_by_level", {}).get(lvl) or self.policy_cfg_by_level["normal"]
        try:
            self.policy.set_device_config(device_id, pcfg)
        except Exception:
            pass

        if self.selected_device is None:
            self.selected_device = device_id

        # init per-device gate engine
        if ML_ENABLE and self.gate_hgb_blob is not None:
            self.gate_hgb_engines[device_id] = OnlineGateHGB(self.gate_hgb_blob)


        self._open_files_for(device_id)
        fp_evt = self.events_fp.get(device_id)
        if fp_evt:
            self._jwrite(fp_evt, {"ts_ms": ts_ms(), "device": device_id, "event": "device_registered"})

        if AUTO_PROVISION_ON_REGISTER:
            asyncio.create_task(self.auto_provision_on_register(device_id))

        if device_id not in self.policy_tasks or self.policy_tasks[device_id].done():
            self.policy_tasks[device_id] = asyncio.create_task(self.policy_loop(device_id))
            print(f"[{now_s()}] [POLICY] started for {device_id}")

        try:
            await self.rx_loop(dc)
        finally:
            task = self.policy_tasks.get(device_id)
            if task and not task.done():
                task.cancel()
            self.policy_tasks.pop(device_id, None)

            self._close_files_for(device_id)

            # cleanup gate engine
            self.gate_hgb_engines.pop(device_id, None)

            if self.devices.get(device_id) is dc:
                del self.devices[device_id]
                if self.selected_device == device_id:
                    self.selected_device = next(iter(self.devices), None)

            await _safe_close(writer)
            print(f"[{now_s()}] [x] Disconnected device_id={device_id}")

    # ----------------- optional CLI -----------------
    def cli_thread(self):
        print("\nCLI commands:")
        print("  list")
        print("  use <device_id>")
        print("  ping")
        print("  budget")
        print("  force_provision_golden  (overwrite golden!)")
        print("  force_provision_blocks (overwrite blocks!)")
        print("  quit\n")

        while True:
            try:
                cmd = input("verifier_policy> ").strip()
            except EOFError:
                cmd = "quit"

            if cmd == "":
                continue

            if cmd == "quit":
                print("bye.")
                self.loop.call_soon_threadsafe(self.loop.stop)
                return

            if cmd == "list":
                devs = list(self.devices.keys())
                print(f"connected: {devs} | selected={self.selected_device}")
                continue

            if cmd.startswith("use "):
                _, dev = cmd.split(" ", 1)
                dev = dev.strip()
                if dev in self.devices:
                    self.selected_device = dev
                    print(f"selected={dev}")
                else:
                    print("no such device connected")
                continue

            if cmd == "budget":
                devs = list(self.devices.keys())
                now = time.time()
                for d in devs:
                    t = self.budget.tokens_now(d, now)
                    mk = self.device_min_k.get(d, 1)
                    bc = self.get_block_count(d)
                    print(f"{d}: tokens={t:.2f}, min_k={mk}, block_count={bc}")
                continue

            dev = self.selected_device
            if not dev:
                print("no device selected/connected")
                continue

            if cmd == "ping":
                coro = self.send_request(dev, {"type": "PING"})
            elif cmd == "force_provision_golden":
                coro = self.force_provision_golden_full(dev, region="fw")
            elif cmd == "force_provision_blocks":
                coro = self.provision_golden_blocks(dev, force=True)
            else:
                print("unknown command")
                continue

            fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
            try:
                resp = fut.result(timeout=8.0)
                print(f"[{now_s()}] [RESP] {resp}")
            except Exception as e:
                print("error waiting:", e)


async def main():
    try:
        with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
            golden = json.load(f)
    except FileNotFoundError:
        golden = {}

    srv = VerifierPolicyServer(golden)
    srv.loop = asyncio.get_running_loop()

    t = threading.Thread(target=srv.cli_thread, daemon=True)
    t.start()

    server = await asyncio.start_server(srv.handle_client, HOST, PORT)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"[{now_s()}] Verifier POLICY listening on {addrs}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
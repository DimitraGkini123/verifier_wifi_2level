# device_policy_2stage_FIXED.py
import joblib
import numpy as np
from typing import Dict, Any, Optional
from collections import deque

# ---- Make custom transformer visible for joblib unpickling ----
import __main__
from ml_utils import PerDeviceStandardScaler
__main__.PerDeviceStandardScaler = PerDeviceStandardScaler


class Device2StagePolicy:
    WORKLOAD_MAP = {0: "LIGHT", 1: "MEDIUM", 2: "HEAVY"}
    ATTACK2B_MAP = {0: "INJECTION", 1: "INTERRUPTION"}

    EPS = 1e-9

    def __init__(self, gate_path: str, workload_path: str, attack2b_path: str):
        self.gate = joblib.load(gate_path)
        self.wl   = joblib.load(workload_path)
        self.atk  = joblib.load(attack2b_path)

        # Gate model expects aggregated features
        self.gate_W = int(self.gate.get("W", 5))
        self.gate_agg_features = list(self.gate.get("agg_features", []))
        self.gate_base_features = list(self.gate.get("base_features", []))
        self.gate_model = self.gate["model"]
        
        # Other models use raw features
        self.wl_features   = list(self.wl["features"])
        self.atk_features  = list(self.atk["features"])
        self.wl_model   = self.wl["model"]
        self.atk_model  = self.atk["model"]

        # Gate thresholds
        pol = self.gate.get("policy", {}) if isinstance(self.gate, dict) else {}
        self.low_thr  = float(pol.get("low_thr", 0.5))
        self.high_thr = float(pol.get("high_thr", 0.5))
        
        # Use threshold from training if available
        if "thr" in self.gate:
            self.high_thr = float(self.gate["thr"])

        self.device_col = str(self.gate.get("device_col", "device_id"))
        
        # Attack labels
        self.atk_labels = None
        try:
            if isinstance(self.atk, dict):
                lbls = self.atk.get("labels")
                if isinstance(lbls, dict) and lbls:
                    self.atk_labels = {int(k): str(v).upper() for k, v in lbls.items()}
        except Exception:
            self.atk_labels = None

        # ===== CRITICAL: Window buffer for aggregation =====
        # Store last W windows per device for gate aggregation
        self.window_buffers = {}  # device_id -> deque of dicts
        
        # Gate prediction history for majority voting
        self.gate_history = {}  # device_id -> deque of gate_labels
        self.gate_history_size = 3  # Use last 3 predictions for majority

    # ---------- helpers ----------
    def _device_key(self, device_id: str) -> int:
        digits = ""
        for ch in reversed(str(device_id)):
            if ch.isdigit():
                digits = ch + digits
            elif digits:
                break
        if not digits:
            raise ValueError(f"Cannot parse numeric id from device_id={device_id!r}")
        return int(digits)

    @staticmethod
    def _append_device_col(X_feat: np.ndarray, dev_int: int) -> np.ndarray:
        dev_col = np.array([[float(dev_int)]], dtype=np.float32)
        return np.concatenate([X_feat, dev_col], axis=1)

    @staticmethod
    def _as_list(p: Optional[np.ndarray]) -> Optional[list]:
        if p is None:
            return None
        return [float(v) for v in p.tolist()]

    @staticmethod
    def _max_conf_from_proba(p: Optional[np.ndarray]) -> Optional[float]:
        if p is None:
            return None
        try:
            return float(np.max(p))
        except Exception:
            return None

    def _gate_label_from_prob(self, p_comp: float) -> str:
        if p_comp <= self.low_thr:
            return "SAFE"
        if p_comp >= self.high_thr:
            return "COMPROMISED"
        return "UNCERTAIN"
    
    def _get_stable_gate_label(self, device_id: str, current_label: str) -> str:
        """
        Apply majority voting over last N predictions to reduce flickering.
        Returns the stable label.
        """
        # Initialize history for this device
        if device_id not in self.gate_history:
            self.gate_history[device_id] = deque(maxlen=self.gate_history_size)
        
        # Add current prediction
        self.gate_history[device_id].append(current_label)
        
        # Need at least 2 samples for stability
        if len(self.gate_history[device_id]) < 2:
            return current_label
        
        # Count votes
        votes = list(self.gate_history[device_id])
        safe_count = votes.count("SAFE")
        comp_count = votes.count("COMPROMISED")
        uncertain_count = votes.count("UNCERTAIN")
        
        # Majority wins
        max_count = max(safe_count, comp_count, uncertain_count)
        
        if comp_count == max_count and comp_count >= 2:
            return "COMPROMISED"
        elif safe_count == max_count and safe_count >= 2:
            return "SAFE"
        else:
            return "UNCERTAIN"

    def _pipeline_classes(self, pipe):
        """Get classes_ from the final estimator inside a sklearn Pipeline."""
        try:
            if hasattr(pipe, "named_steps"):
                last = list(pipe.named_steps.values())[-1]
                return getattr(last, "classes_", None)
        except Exception:
            pass
        return getattr(pipe, "classes_", None)

    def _prob_of_class(self, proba_row: np.ndarray, classes: Optional[np.ndarray], target_class: int) -> float:
        """Return P(y==target_class)."""
        try:
            if classes is not None:
                idx = int(np.where(classes == target_class)[0][0])
                return float(proba_row[idx])
        except Exception:
            pass
        # fallback: assume binary [P(0), P(1)]
        if len(proba_row) >= 2:
            return float(proba_row[1])
        return float(proba_row[0])

    # ---------- CORRECTED: feature enrichment (matches training exactly) ----------
    def _compute_base_features(self, window: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute base ratio features that match training.
        Returns dict with features like: cyc_per_us, lsu_per_cyc, cpi_per_cyc, etc.
        """
        # Read raw counters
        dC = float(window.get("dC", 0))
        dL = float(window.get("dL", 0))
        dP = float(window.get("dP", 0))
        dE = float(window.get("dE", 0))
        dF = float(window.get("dF", 0))
        dT = float(window.get("dT", 1.0))  # avoid div by zero
        
        # EXACTLY as in training add_ratio_features()
        base = {}
        
        #base["cyc_per_us"] = float(window.get("cyc_per_us", dC / (dT + self.EPS)))
        #base["lsu_per_cyc"] = dL / (dC + 1.0)  # safe_div logic
        #base["cpi_per_cyc"] = dP / (dC + 1.0)
        #ase["exc_per_cyc"] = dE / (dC + 1.0)
        #base["fold_per_cyc"] = dF / (dC + 1.0)
        base["cyc_per_us"]  = float(window.get("cyc_per_us", dC / (dT + self.EPS)))
        base["lsu_per_cyc"] = dL / (dC + self.EPS)
        base["cpi_per_cyc"] = dP / (dC + self.EPS)
        base["exc_per_cyc"] = dE / (dC + self.EPS)
        base["fold_per_cyc"]= dF / (dC + self.EPS)
        
        return base

    def _aggregate_window_buffer(self, device_id: str, base_features: Dict[str, float]) -> Optional[np.ndarray]:
        """
        Maintain sliding window buffer and compute aggregated features (mean, std, max, min)
        matching the training pipeline.
        
        Returns None if buffer not full yet, otherwise returns aggregated feature vector.
        """
        # Initialize buffer for this device if needed
        if device_id not in self.window_buffers:
            self.window_buffers[device_id] = deque(maxlen=self.gate_W)
        
        # Add current window to buffer
        self.window_buffers[device_id].append(base_features)
        
        # Need W windows for aggregation
        if len(self.window_buffers[device_id]) < self.gate_W:
            return None
        
        # Compute aggregated statistics
        buffer = self.window_buffers[device_id]
        agg_features = {}
        
        # For each base feature, compute mean/std/max/min
        for feat_name in self.gate_base_features:
            values = [w[feat_name] for w in buffer]
            arr = np.array(values, dtype=np.float32)
            
            agg_features[f"{feat_name}_mean_W{self.gate_W}"] = float(np.mean(arr))
            agg_features[f"{feat_name}_std_W{self.gate_W}"] = float(np.std(arr, ddof=0))
            agg_features[f"{feat_name}_max_W{self.gate_W}"] = float(np.max(arr))
            agg_features[f"{feat_name}_min_W{self.gate_W}"] = float(np.min(arr))
        
        # Build feature vector in correct order
        X = np.array([[agg_features[f] for f in self.gate_agg_features]], dtype=np.float32)
        return X

    def _enrich_window_for_wl_atk(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legacy enrichment for workload and attack models (if they use raw + engineered features).
        Keep this for backwards compatibility with your existing models.
        """
        w = dict(window)

        dC = float(w.get("dC", 0))
        dL = float(w.get("dL", 0))
        dP = float(w.get("dP", 0))
        dE = float(w.get("dE", 0))
        dF = float(w.get("dF", 0))
        dS = float(w.get("dS", 0))
        dT = float(w.get("dT", 1.0))

        # Ratios
        w["lsu_per_cyc"]  = dL / (dC + 1.0)
        w["cpi_per_cyc"]  = dP / (dC + 1.0)
        w["exc_per_cyc"]  = dE / (dC + 1.0)
        w["fold_per_cyc"] = dF / (dC + 1.0)

        w["lsu_per_us2"]  = dL / (dT + self.EPS)
        w["cpi_per_us2"]  = dP / (dT + self.EPS)
        w["exc_per_us2"]  = dE / (dT + self.EPS)

        w["cyc_per_lsu2"] = dC / (dL + 1.0)

        # log1p
        for k in ["dC","dL","dP","dE","dF","dS","dT"]:
            v = float(w.get(k, 0))
            if v < 0:
                v = 0.0
            w[f"log1p_{k}"] = float(np.log1p(v))

        return w

    def _row_from_window(self, features: list, window: Dict[str, Any]) -> np.ndarray:
        return np.array([[float(window[f]) for f in features]], dtype=np.float32)
    
    def predict_attack2b_only(self, device_id: str, window: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run ONLY the attack2b model (no gate check). Use this when an external gate (HGB)
        has already decided COMPROMISED.
        """
        try:
            dev_int = self._device_key(device_id)
            w = self._enrich_window_for_wl_atk(window)

            Xa = self._append_device_col(self._row_from_window(self.atk_features, w), dev_int)
            pred_a = int(self.atk_model.predict(Xa)[0])
            pa = self.atk_model.predict_proba(Xa)[0] if hasattr(self.atk_model, "predict_proba") else None

            if self.atk_labels is not None:
                atk_label = self.atk_labels.get(pred_a, "UNKNOWN")
            else:
                atk_label = self.ATTACK2B_MAP.get(pred_a, "UNKNOWN")

            return {
                "ok": True,
                "reason": "ok_attack2b_only",
                "device_int": dev_int,
                "attack2b_pred": pred_a,
                "attack2b_label": atk_label,
                "attack2b_proba": self._as_list(pa),
                "attack2b_conf": self._max_conf_from_proba(pa),
            }

        except Exception as e:
            return {
                "ok": False,
                "reason": f"predict_attack2b_only_error:{e}",
                "attack2b_label": None,
                "attack2b_conf": None,
            }

    # ---------- main predict (FIXED) ----------
    def predict(self, device_id: str, window: Dict[str, Any]) -> Dict[str, Any]:
        """
        CORRECTED: Now uses proper aggregation for gate model.
        """
        try:
            dev_int = self._device_key(device_id)

            # ----- GATE: Compute base features and aggregate -----
            base_feats = self._compute_base_features(window)
            Xg = self._aggregate_window_buffer(device_id, base_feats)
            
            # Not enough windows yet - return UNCERTAIN
            if Xg is None:
                return {
                    "ok": True,
                    "reason": f"warming_up_buffer_{len(self.window_buffers.get(device_id, []))}/{self.gate_W}",
                    "device_int": dev_int,
                    "gate_label": "UNCERTAIN",
                    "p_compromised": 0.5,
                    "gate_conf": 0.0,
                    "gate_proba": None,
                    "gate_low_thr": float(self.low_thr),
                    "gate_high_thr": float(self.high_thr),
                }

            if not hasattr(self.gate_model, "predict_proba"):
                return {
                    "ok": False,
                    "reason": "gate_model_has_no_predict_proba",
                    "device_int": dev_int,
                    "gate_label": "UNCERTAIN",
                    "p_compromised": 0.5,
                    "gate_conf": 0.0,
                    "gate_proba": None,
                    "gate_low_thr": float(self.low_thr),
                    "gate_high_thr": float(self.high_thr),
                }

            pg = self.gate_model.predict_proba(Xg)[0]
            classes_g = self._pipeline_classes(self.gate_model)

            p_comp = self._prob_of_class(pg, classes_g, 1)
            gate_label_raw = self._gate_label_from_prob(p_comp)
            gate_label = self._get_stable_gate_label(device_id, gate_label_raw)
            gate_conf = float(np.max(pg))

            out: Dict[str, Any] = {
                "ok": True,
                "reason": "ok",
                "device_int": dev_int,
                "gate_label": gate_label,
                "gate_label_raw": gate_label_raw,  # For debugging
                "p_compromised": float(p_comp),
                "gate_low_thr": float(self.low_thr),
                "gate_high_thr": float(self.high_thr),
                "gate_proba": self._as_list(pg),
                "gate_conf": float(gate_conf),
            }

            # ----- SAFE -> WORKLOAD -----
            if gate_label == "SAFE":
                w = self._enrich_window_for_wl_atk(window)
                Xw = self._append_device_col(self._row_from_window(self.wl_features, w), dev_int)
                pred_w = int(self.wl_model.predict(Xw)[0])
                pw = self.wl_model.predict_proba(Xw)[0] if hasattr(self.wl_model, "predict_proba") else None

                out.update({
                    "workload_pred": pred_w,
                    "workload_label": self.WORKLOAD_MAP.get(pred_w, "UNKNOWN"),
                    "workload_proba": self._as_list(pw),
                    "workload_conf": self._max_conf_from_proba(pw),
                })
                return out

            # ----- COMPROMISED -> ATTACK2B -----
            if gate_label == "COMPROMISED":
                w = self._enrich_window_for_wl_atk(window)
                Xa = self._append_device_col(self._row_from_window(self.atk_features, w), dev_int)
                pred_a = int(self.atk_model.predict(Xa)[0])
                pa = self.atk_model.predict_proba(Xa)[0] if hasattr(self.atk_model, "predict_proba") else None

                if self.atk_labels is not None:
                    atk_label = self.atk_labels.get(pred_a, "UNKNOWN")
                else:
                    atk_label = self.ATTACK2B_MAP.get(pred_a, "UNKNOWN")

                out.update({
                    "attack2b_pred": pred_a,
                    "attack2b_label": atk_label,
                    "attack2b_proba": self._as_list(pa),
                    "attack2b_conf": self._max_conf_from_proba(pa),
                })
                return out

            return out  # UNCERTAIN

        except Exception as e:
            return {
                "ok": False,
                "reason": f"predict_error:{e}",
                "gate_label": "UNCERTAIN",
                "p_compromised": 0.5,
                "gate_conf": 0.0,
                "gate_proba": None,
            }
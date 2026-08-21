import sys
sys.dont_write_bytecode = True


class SafetyGateResult(object):
    def __init__(self, state="NORMAL", scale=1.0, stop=False, reason="", risk=0.0):
        self.state = state
        self.scale = float(scale)
        self.stop = bool(stop)
        self.reason = reason
        self.risk = float(risk)


class SafetyGate(object):
    def __init__(self, rho_warn=1.8, rho_stop=2.8, min_scale=0.2, enabled=True):
        self.rho_warn = float(rho_warn)
        self.rho_stop = float(rho_stop)
        self.min_scale = float(min_scale)
        self.enabled = bool(enabled)

    def evaluate(self, risk, forbidden=False, extra_reason=""):
        risk = float(risk)
        if not self.enabled:
            return SafetyGateResult("NORMAL", 1.0, False, "disabled", risk)
        if forbidden:
            return SafetyGateResult(
                "STOP", 0.0, True, extra_reason or "forbidden_zone", risk)
        if risk >= self.rho_stop:
            return SafetyGateResult("STOP", 0.0, True, "risk_stop", risk)
        if risk >= self.rho_warn:
            denom = max(self.rho_stop - self.rho_warn, 1e-6)
            alpha = min(1.0, max(0.0, (risk - self.rho_warn) / denom))
            scale = 1.0 - alpha * (1.0 - self.min_scale)
            return SafetyGateResult("SLOW", scale, False, "risk_slow", risk)
        return SafetyGateResult("NORMAL", 1.0, False, "", risk)

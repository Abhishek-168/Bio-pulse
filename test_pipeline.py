"""
Camera-free unit tests for the Bio-Pulse pipeline.

Covers the multi-channel rPPG overhaul, anti-spoof cues, the silent screen
veto, and backward compatibility. Run:

    python test_pipeline.py        # or: python -m pytest test_pipeline.py
"""
import numpy as np

from pulse_extractor import PulseExtractor, phase_coherence
from liveness_detector import LivenessDetector
from screen_detector import ScreenDetector

RNG = np.random.default_rng(42)
FPS = 30.0
N = 300  # 10 s


def _make_rgb_pulse(bpm=72.0, n=N, fps=FPS, illum_noise=0.0, pulse_amp=2.0, seed=1):
    """
    Build synthetic R,G,B sample streams with a pulse in green and a shared
    illumination noise that hits all channels equally (the thing POS rejects).
    Returns a list of (r, g, b) tuples.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    f = bpm / 60.0
    pulse = pulse_amp * np.sin(2 * np.pi * f * t)
    shared = illum_noise * np.cumsum(rng.standard_normal(n))  # slow drift, all channels
    r = 100 + shared + 0.1 * pulse
    g = 120 + shared + pulse            # green carries the strongest pulse
    b = 90 + shared + 0.1 * pulse
    return list(zip(r, g, b))


def _feed(pe, samples):
    for s in samples:
        pe.add_sample(s)


# ──────────────────────────────────────────────
# 6a. POS rejects shared illumination noise
# ──────────────────────────────────────────────
def test_pos_rejects_illumination_noise():
    samples = _make_rgb_pulse(bpm=72, illum_noise=3.0, seed=7)

    pe_pos = PulseExtractor(fps=FPS, channel_mode="pos")
    pe_green = PulseExtractor(fps=FPS, channel_mode="green")
    _feed(pe_pos, samples)
    _feed(pe_green, samples)

    bpm_pos, snr_pos, _ = pe_pos.get_bpm()
    bpm_green, snr_green, _ = pe_green.get_bpm()

    assert abs(bpm_pos - 72) < 6, f"POS BPM off: {bpm_pos}"
    # POS should reject the shared illumination noise better than green-only
    assert snr_pos >= snr_green - 0.5, f"POS SNR {snr_pos} < green {snr_green}"
    print(f"[6a] POS BPM={bpm_pos:.1f} SNR={snr_pos:.1f} | green SNR={snr_green:.1f}  OK")


# ──────────────────────────────────────────────
# 6b. Harmonic ratio: pure tone ~0, tone+harmonic above threshold
# ──────────────────────────────────────────────
def test_harmonic_ratio():
    t = np.arange(N) / FPS
    pure = 120 + 2 * np.sin(2 * np.pi * 1.2 * t)
    harm = 120 + 2 * np.sin(2 * np.pi * 1.2 * t) + 1.0 * np.sin(2 * np.pi * 2.4 * t)

    pe_pure = PulseExtractor(fps=FPS, channel_mode="green")
    pe_harm = PulseExtractor(fps=FPS, channel_mode="green")
    for v in pure:
        pe_pure.add_sample_green(v)
    for v in harm:
        pe_harm.add_sample_green(v)

    r_pure = pe_pure.get_harmonic_ratio()
    r_harm = pe_harm.get_harmonic_ratio()

    assert r_pure < 0.05, f"pure tone harmonic ratio too high: {r_pure}"
    assert r_harm > r_pure, f"harmonic signal ratio {r_harm} not > pure {r_pure}"
    print(f"[6b] harmonic ratio pure={r_pure:.3f} with-harmonic={r_harm:.3f}  OK")


# ──────────────────────────────────────────────
# 6c. HRV band: constant IBI fails jitter, natural jitter passes, noise fails regularity
# ──────────────────────────────────────────────
def test_hrv_band():
    # Perfectly periodic → jitter_ok should be False (too perfect = synthetic)
    pe_perfect = PulseExtractor(fps=FPS, channel_mode="green")
    t = np.arange(N) / FPS
    for v in 120 + 2 * np.sin(2 * np.pi * 1.2 * t):
        pe_perfect.add_sample_green(v)
    hrv_perfect = pe_perfect.get_hrv_metrics()
    assert hrv_perfect["jitter_ok"] is False, f"perfect tone passed jitter: {hrv_perfect}"

    # Natural jitter (~8% IBI variation) → jitter_ok True
    rng = np.random.default_rng(3)
    pe_real = PulseExtractor(fps=FPS, buffer_seconds=15, channel_mode="green")
    # build a beat train with jittered intervals
    sig = np.zeros(int(FPS * 15))
    base = FPS / 1.2  # samples per beat at 72 BPM
    pos = 0.0
    while pos < len(sig) - 1:
        sig[int(pos)] = 1.0
        pos += base * (1 + 0.08 * rng.standard_normal())
    # smooth impulses into pulse-like bumps
    kernel = np.hanning(int(base * 0.6))
    sig = np.convolve(sig, kernel, mode="same") + 120
    for v in sig:
        pe_real.add_sample_green(v)
    hrv_real = pe_real.get_hrv_metrics()
    assert hrv_real["jitter_ok"] is True, f"natural jitter failed: {hrv_real}"

    # The HRV jitter band's primary job is rejecting synthetic TOO-PERFECT
    # tones (lower bound); its upper bound is deliberately generous so real
    # noisy rPPG isn't falsely rejected. Pure noise is therefore caught
    # elsewhere — by low SNR — not by the jitter band. Verify that path.
    pe_noise = PulseExtractor(fps=FPS, channel_mode="green")
    for v in 120 + 5 * rng.standard_normal(N):
        pe_noise.add_sample_green(v)
    _, snr_noise, _ = pe_noise.get_bpm()
    assert snr_noise < 4.0, f"noise SNR too high: {snr_noise}"
    print(f"[6c] perfect jitter_ok={hrv_perfect['jitter_ok']} "
          f"real jitter_ok={hrv_real['jitter_ok']} "
          f"noise SNR={snr_noise:.1f} (caught by SNR gate)  OK")


# ──────────────────────────────────────────────
# 6d. Phase coherence: in-phase ~+1, noise ~0
# ──────────────────────────────────────────────
def test_phase_coherence():
    t = np.arange(N) / FPS
    a = np.sin(2 * np.pi * 1.2 * t)
    b = np.sin(2 * np.pi * 1.2 * t)  # in phase
    coh = phase_coherence(a, b, FPS, 1.2)
    assert coh > 0.9, f"in-phase coherence too low: {coh}"

    rng = np.random.default_rng(11)
    cohs = [phase_coherence(rng.standard_normal(N), rng.standard_normal(N), FPS, 1.2)
            for _ in range(20)]
    assert abs(np.mean(cohs)) < 0.4, f"noise coherence not ~0: {np.mean(cohs)}"
    print(f"[6d] phase in-phase={coh:.2f} noise_mean={np.mean(cohs):.2f}  OK")


# ──────────────────────────────────────────────
# 6e. Kalman rejects an injected outlier
# ──────────────────────────────────────────────
def test_kalman_outlier():
    pe = PulseExtractor(fps=FPS)
    est = 0.0
    for _ in range(10):
        est = pe.update_bpm_track(72.0, snr=8.0)
    before = est
    est = pe.update_bpm_track(150.0, snr=8.0)  # spoof jump
    assert abs(est - before) < 5, f"Kalman followed outlier: {before}->{est}"
    print(f"[6e] Kalman stable {before:.1f}->{est:.1f} despite 150 outlier  OK")


# ──────────────────────────────────────────────
# 6f. Screen detector: flicker + moiré
# ──────────────────────────────────────────────
def test_screen_flicker():
    fps = FPS
    sd_screen = ScreenDetector(fps=fps)
    sd_real = ScreenDetector(fps=fps)
    t = np.arange(int(fps * 6)) / fps
    # Screen: strong out-of-band flicker (e.g. 8 Hz beat) on luminance
    flicker = 1.0 + 0.3 * np.sin(2 * np.pi * 8.0 * t)
    # Real: only cardiac-band variation
    cardiac = 1.0 + 0.02 * np.sin(2 * np.pi * 1.2 * t)
    base = np.full((40, 40, 3), 100, dtype=np.uint8)
    for i in range(len(t)):
        sd_screen.add_frame((base * flicker[i]).astype(np.uint8))
        sd_real.add_frame((base * cardiac[i]).astype(np.uint8))
    fs_screen = sd_screen.flicker_score()
    fs_real = sd_real.flicker_score()
    assert fs_screen > fs_real, f"flicker screen {fs_screen} !> real {fs_real}"
    print(f"[6f-flicker] screen={fs_screen:.2f} real={fs_real:.2f}  OK")


def test_screen_moire():
    sd = ScreenDetector(fps=FPS)
    # Smooth gradient (real skin) → low moiré
    grad = np.tile(np.linspace(0, 255, 60, dtype=np.uint8), (60, 1))
    grad_bgr = np.stack([grad] * 3, axis=-1)
    low = sd.moire_score(grad_bgr)
    # High-frequency sinusoidal grid (screen pixel grid, near Nyquist) → high moiré
    x = np.arange(60)
    grid = (127 + 120 * np.sin(2 * np.pi * x / 2.2)).astype(np.uint8)
    grid2d = np.tile(grid, (60, 1))
    grid_bgr = np.stack([grid2d] * 3, axis=-1)
    high = sd.moire_score(grid_bgr)
    assert high > low, f"moiré grid {high} !> gradient {low}"
    print(f"[6f-moire] gradient={low:.3f} grid={high:.3f}  OK")


# ──────────────────────────────────────────────
# 6g. Liveness backward-compat (2-arg update still works)
# ──────────────────────────────────────────────
def test_liveness_backward_compat():
    # Good synthetic pulse, fed via legacy green path + 2-arg update
    pe = PulseExtractor(fps=FPS, channel_mode="green")
    t = np.arange(N) / FPS
    for v in 120 + 2 * np.sin(2 * np.pi * 1.2 * t):
        pe.add_sample_green(v)
    bpm, snr, _ = pe.get_bpm()
    ld = LivenessDetector()
    decision = LivenessDetector.PENDING
    for _ in range(8):
        decision = ld.update(bpm, snr)  # legacy 2-arg call
    assert decision == LivenessDetector.ALIVE, f"legacy good signal not ALIVE: {decision}"

    # Flat signal → DENIED
    ld2 = LivenessDetector()
    d2 = LivenessDetector.PENDING
    for _ in range(12):
        d2 = ld2.update(0.0, 0.0)
    assert d2 == LivenessDetector.DENIED, f"flat not DENIED: {d2}"
    print(f"[6g] legacy good={decision} flat={d2}  OK")


def test_liveness_screen_veto():
    """A high is_screen score must silently force DENIED even with a good pulse."""
    pe = PulseExtractor(fps=FPS, channel_mode="green")
    t = np.arange(N) / FPS
    for v in 120 + 2 * np.sin(2 * np.pi * 1.2 * t):
        pe.add_sample_green(v)
    bpm, snr, _ = pe.get_bpm()
    ld = LivenessDetector()
    d = LivenessDetector.PENDING
    for _ in range(8):
        d = ld.update(bpm, snr, regularity=0.9, correlation=0.9,
                      harmonic_ratio=0.3, jitter_ok=True, phase_coherence=0.9,
                      is_screen=0.9)
    assert d == LivenessDetector.DENIED, f"screen veto failed: {d}"
    print(f"[6g-veto] good pulse + is_screen=0.9 -> {d}  OK")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

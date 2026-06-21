"""Quick smoke test for PulseExtractor + LivenessDetector pipeline."""
import numpy as np
from pulse_extractor import PulseExtractor
from liveness_detector import LivenessDetector

pe = PulseExtractor(fps=30.0)
ld = LivenessDetector()

# Simulate a 72 BPM (1.2 Hz) pulse signal for 10 seconds
t = np.arange(300) / 30.0
fake_signal = 120 + 2 * np.sin(2 * np.pi * 1.2 * t)

for s in fake_signal:
    pe.add_sample(s)

bpm, snr, filt = pe.get_bpm()
print(f"Simulated 72 BPM (1.2 Hz) signal")
print(f"Detected BPM: {bpm:.1f}")
print(f"SNR: {snr:.1f} dB")
print(f"Filtered signal length: {len(filt)}")

# Feed enough readings for liveness decision
for _ in range(6):
    decision = ld.update(bpm, snr)

status = ld.get_status()
print(f"Decision: {status['decision']}")
print(f"Avg BPM: {status['avg_bpm']:.1f}")
print(f"Avg SNR: {status['avg_snr']:.1f}")
print()

# Test with flat signal (photo/spoof scenario)
pe2 = PulseExtractor(fps=30.0)
ld2 = LivenessDetector()
flat_signal = np.full(300, 120.0)  # constant = no pulse

for s in flat_signal:
    pe2.add_sample(s)

bpm2, snr2, filt2 = pe2.get_bpm()
print(f"Flat signal (spoof scenario)")
print(f"Detected BPM: {bpm2:.1f}")
print(f"SNR: {snr2:.1f} dB")

for _ in range(12):  # Need 10+ failed readings for DENIED
    decision2 = ld2.update(bpm2, snr2)

status2 = ld2.get_status()
print(f"Decision: {status2['decision']}")

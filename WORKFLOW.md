# Bio-Pulse Authenticator — Project Workflow

## 1. End-to-end pipeline (per frame)

```mermaid
flowchart TD
    A["Webcam capture (cv2.VideoCapture)"] --> B["Flip frame + convert BGR→RGB"]
    B --> C["MediaPipe FaceLandmarker.detect"]
    C --> D{Face found?}

    D -- No --> E["face_lost_frames++"]
    E --> F{"lost > 1s?"}
    F -- Yes --> G["Reset: pulse / liveness / screen / challenge / session"]
    F -- No --> Z
    G --> Z["Render dashboard + show verdict"]

    D -- Yes --> H["Compute forehead ROI + cheek ROI"]
    H --> I["Extract per-ROI R,G,B means (BGR→RGB mapped)"]

    I --> J["PulseExtractor.add_sample (forehead & cheek)"]

    subgraph SIG["Signal extraction (PulseExtractor)"]
        J --> K["POS multi-channel projection"]
        K --> L["Standardize (z-score)"]
        L --> M["Smoothness-priors detrend (Tarvainen, from heartbeat-js)"]
        M --> N["Butterworth bandpass 0.75–3 Hz"]
        N --> O["FFT → BPM + SNR"]
        O --> P["Kalman BPM smoothing (outlier-gated)"]
        N --> Q["Peak regularity + HRV (jitter band)"]
        N --> R["Harmonic ratio (2nd harmonic)"]
    end

    subgraph CUE["Anti-spoof cues"]
        I --> S["ScreenDetector: moiré × flicker × glare → is_screen (SILENT)"]
        N --> T["Cross-ROI magnitude correlation (lag + sign robust)"]
        N --> U["Cross-ROI phase coherence at pulse freq"]
        C --> V["ChallengeResponse: 3-step series — blink / head-turn / mouth / smile / brows / nod (ACTIVE)"]
    end

    P --> W["LivenessDetector.update"]
    Q --> W
    R --> W
    S --> W
    T --> W
    U --> W

    W --> X["Per-frame raw verdict: ALIVE / DENIED / PENDING"]
    V --> Y["Challenge gate"]
    X --> Y
    Y --> SA["SessionAnalyzer (10–12s)"]
    SA --> Z
```

## 2. LivenessDetector decision logic (per frame)

```mermaid
flowchart TD
    A["update(bpm, snr, regularity, correlation, harmonic, jitter, phase, is_screen)"] --> B{"avg is_screen ≥ 0.5?"}
    B -- Yes --> DENY["DENIED (silent screen veto)"]
    B -- No --> C{"bpm ≤ 0 or snr ≤ 0?"}
    C -- Yes --> PEND1["PENDING (count failures → DENIED if persistent)"]
    C -- No --> D["Append to rolling history (window=8)"]
    D --> E{"enough readings?"}
    E -- No --> PEND2["PENDING"]
    E -- Yes --> F["Evaluate gates"]

    F --> G{"BPM in 45–180? (HARD)"}
    F --> H{"cross-ROI OK: correlation OR phase? (HARD)"}
    F --> I["Quality VOTE (7 cues): SNR, consistency, regularity, correlation, phase, harmonic, jitter"]

    G -- No --> DENY2["DENIED"]
    H -- No --> DENY2
    I --> J{"votes ≥ 50% (≥4/7)?"}
    J -- No --> DENY2
    G -- Yes --> K
    H -- Yes --> K
    J -- Yes --> K{"all hard gates + majority?"}
    K -- Yes --> ALIVE["ALIVE"]
    K -- No --> DENY2
```

## 3. Session + challenge → final locked verdict

```mermaid
stateDiagram-v2
    [*] --> Warmup: face acquired
    Warmup --> Analyzing: pulse buffer ready (~3s)
    note right of Warmup
        Buffer fills, no BPM yet → PENDING
    end note

    Analyzing --> Analyzing: ALIVE frame accrues / challenge prompt shown
    Analyzing --> Challenge_Reissue: challenge timed out
    Challenge_Reissue --> Analyzing: new random 3-step series

    Analyzing --> GRANTED: ≥2s cumulative ALIVE frames (early grant)
    Analyzing --> DENIED: window elapsed (~12s) without enough ALIVE

    GRANTED --> [*]: locked (stable, no flicker)
    DENIED --> [*]: locked

    note right of GRANTED
        ALIVE requires: live pulse +
        cross-ROI agreement +
        majority quality vote +
        passed challenge
    end note
```

## 4. Optional backend service (BE — Node/Express + Redis)

```mermaid
flowchart LR
    A["Client POST /verify (video bytes)"] --> B["Express server"]
    B --> C["Enqueue job → Redis list (video:queue)"]
    C --> D["SSE stream: status = enqueued"]
    D --> E["Poll Redis job:status"]
    E --> F{"status done/error?"}
    F -- No --> E
    F -- Yes --> G["Stream final status + close"]
```

## Key idea
Appearance-based detectors ask *"does this look real?"*. Bio-Pulse asks *"is this body alive?"* — proving a real, frequency-correct heartbeat (rPPG) that is spatially consistent across face regions, while an attacker must simultaneously beat the pulse checks, cross-ROI blood-flow consistency, an active liveness challenge, and a silent screen-replay detector — all on a commodity webcam.

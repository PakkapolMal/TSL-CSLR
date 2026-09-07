# Phase 1 Review

| video | frames | duration (s) | left hand % | right hand % | pose % | excluded spans | excluded frames | signs | sentences | signs/s | mean sign dur (s) | median sign dur (s) | sign dur CV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| house_29_w25 | 7503 | 250.1 | 92.5% | 92.5% | 98.0% | 20 | 147 | 283 | 38 | 1.13 | 0.33 | 0.27 | 0.83 |
| house2_190766_w1 | 17181 | 572.7 | 98.3% | 98.1% | 99.3% | 2 | 116 | 783 | 51 | 1.37 | 0.28 | 0.23 | 0.70 |
| senate_181266_w0 | 28716 | 957.2 | 85.1% | 78.9% | 99.6% | 1 | 120 | 949 | 19 | 0.99 | 0.47 | 0.40 | 0.74 |

## Qualitative review (human)

| Question | Why it matters | Answer |
|---|---|---|
| Do sentence boundaries land on visible pauses / hand drops / body shifts? | If yes, Phase 2 has a viable fallback tier. | |
| Are sign boundaries plausible, or is it splitting at a near-constant rate? | Cross-check against the CV computed above. | |
| Does behaviour differ noticeably between videos (= between signers)? | Predicts how hard signer-independence will be. | |
| Where does it fail — fingerspelling, classifiers, fast sequences? | Directly informs the Phase 2 annotation convention. | |

**Overall judgement (human):** sign-level usable / sentence-level only / neither — _blank, fill in after ELAN review_.

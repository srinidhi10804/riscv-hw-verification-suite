
---

## Modules

### 1. Int Multiplier (`int_multiplier/`)

This module provides an integer multiplier implementation along with verification scripts and AI-based comparison for correctness.

**Files:**

- `int_multiplier.bsv` – BSV implementation of the integer multiplier.  
- `verifier.py` – Python script for verifying the multiplier functionality.  
- `bsv_ai_validator_comparison.py` – AI-based comparison between BSV output and reference model.  
- `ai_test_results.json` – Stores results of AI-based test runs.  
- `edge_case_comparison.json` – Contains test results for edge cases.

---

### 2. SRT Radix-2 Divider (`srt_radix2_divider/`)

This module contains the SRT Radix-2 divider implementation along with verification scripts and AI-based validation.

**Files:**

- `srt_radix2_divider.bsv` – BSV implementation of the SRT Radix-2 divider.  
- `divider_test_cases.py` – Python test cases for functional verification.  
- `srt_radix2_divider.py` – Python reference model for the divider.  
- `bsv_ai_valid_radix2.py` – AI-based validation for BSV outputs.  
- `divider_edge_case_comparison.json` – Edge case verification results.

### 3. SRT Radix-4 Divider (`srt_radix4_divider/`)

This module contains the SRT Radix-4 divider implementation along with verification scripts and AI-based validation.

**Files:**

- `srt_radix4_divider.bsv` – BSV implementation of the SRT Radix-4 divider.  
- `srt_radix4_divider.py` – Python reference model for the divider.  
- `divider_test_cases.json` – JSON file containing test cases.  
- `divider_test_vectors.txt` – Test vectors for functional verification.  
- `bsv_ai_valid_radix4.py` – AI-based validation for BSV outputs.  
- `divider_edge_case_comparison_radix4.json` – Edge case verification results.

---
4. SRT Radix-4 Divider (64-bit) (srt_radix4_divider_64/)

This module provides an AI-assisted verification setup for a 64-bit SRT Radix-4 Divider, integrating intelligent test generation and edge-case validation using Python and BSV.

Files:

- srt_radix4_divider_64.bsv – BSV implementation of the 64-bit SRT Radix-4 divider.
- srt_radix4_divider_64.py – Python model for divider verification.
- bsv_ai_valid_radix64.py – AI-based validation of BSV outputs.
- srt_radix4_64_ai_vs_bsv.py – AI vs BSV coverage comparison script.
- divider_test_results.json – Stores test results from AI-generated cases.
- divider_edge_case_comparison.json – Records AI–BSV edge-case comparison outcomes.
---
5. Carry Look-Ahead Adder (130-bit) (CLA_130/)

This module implements and verifies a 130-bit Carry Look-Ahead Adder (CLA) using AI-assisted test generation and coverage analysis.

Files:
- CLA_130.bsv – BSV implementation of the 130-bit CLA.
- cla_130_ai.py – Python AI-based test generation module.
- cla_130_ai_vs_bsv.py – AI vs BSV comparison and coverage evaluator.
- cla_edge_case_comparison.json – Edge case coverage results.
- cla_results_20251027_045300.json – AI test case output data.
- cla_summary_20251027_045300.txt – Summary of verification and AI coverage statistics.
---
6. Lead Zero Detect 64 (lead_zero_detect64/)

This module implements a 64-bit Leading Zero Detector (LZD) in Bluespec SystemVerilog (BSV) and provides AI-assisted verification using Python scripts.
It identifies the number of leading zeros in a 64-bit input word, useful for normalization steps in division, floating-point, and arithmetic units.

Files:
- lead_zero_detect64.bsv – BSV implementation of the 64-bit Leading Zero Detector.
- lead_zero_detect64.py – Python reference model for validating BSV results.
- lzd_bsv_validator.py – AI-assisted validation script comparing BSV and Python model outputs.
- lzd_test_results.json – JSON file storing test results for normal and random cases.
- lzd_edge_case_comparison.json – Edge case verification data comparing AI-predicted and BSV outputs.
---
## How to Use

1. Clone the repository:

```bash
git clone https://github.com/srinidhi10804/riscv-hw-verification-suite.git
cd riscv-hw-verification-suite


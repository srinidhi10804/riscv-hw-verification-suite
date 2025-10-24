
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

---

## How to Use

1. Clone the repository:

```bash
git clone https://github.com/srinidhi10804/riscv-hw-verification-suite.git
cd riscv-hw-verification-suite


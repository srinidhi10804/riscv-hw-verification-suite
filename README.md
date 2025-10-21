Non-Restoring Divider (BSV)

This project implements a 64-bit non-restoring divider in Bluespec SystemVerilog (BSV). The divider supports:

Signed and unsigned division

32-bit and 64-bit operations

Returning either quotient or remainder depending on the requested operation

Handling special cases like division by zero and signed overflow

The design is iterative, performing one division step per clock cycle, and exposes a clean interface:

ma_start(dividend, divisor, opcode, funct3) — starts a division operation

mav_result() — returns the result (valid + quotient/remainder)

ma_set_flush() — resets the divider if needed

The BSV testbench runs through a fixed set of test cases using pre-defined dividends and divisors, including their two’s complements, for various operations (DIV, REM, DIVU, REMU, DIVW, REMW, DIVUW, REMUW). It prints cycle-by-cycle results to verify the functionality.

Limitations of BSV Testbench

While the BSV testbench covers basic cases, it has limited coverage:

Only a few fixed dividend/divisor values are tested

Edge cases like large negative numbers, random inputs, and extreme values are not fully tested

Special cases are manually hardcoded, limiting scalability

Python Verification Script

To supplement the BSV testbench, a Python script:

Generates random 32-bit and 64-bit dividends and divisors

Computes expected results for all operation types (DIV, REM, DIVU, REMU, etc.)

Compares Python results with BSV outputs for mismatches

Provides coverage for edge cases like negative numbers, zero divisor, and signed overflow

This combination ensures the design is functionally correct across a much broader range of inputs than the BSV testbench alone.

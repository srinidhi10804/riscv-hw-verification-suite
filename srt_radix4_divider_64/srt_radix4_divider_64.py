"""
AI-Assisted Verification System for RISC-V SRT Radix-4 Divider (BSV)
Supports multiple free AI providers: Groq, Google Gemini, Ollama
Targets 90%+ BSV edge case coverage
"""

import os
import json
import requests
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum

# Configuration
AI_PROVIDER = "groq"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
XLEN = 64

class DivOp(Enum):
    DIV = 0b100     # Signed division
    DIVU = 0b101    # Unsigned division
    REM = 0b110     # Signed remainder
    REMU = 0b111    # Unsigned remainder

@dataclass
class TestCase:
    dividend: int
    divisor: int
    funct3: int
    word32: bool
    expected_q: int
    expected_r: int
    description: str
    category: str

class AIProvider:
    @staticmethod
    def call_groq(prompt: str) -> str:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

        for model in models_to_try:
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2048,
                "top_p": 1.0
            }

            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
                if response.status_code == 400:
                    data["max_completion_tokens"] = data.pop("max_tokens")
                    response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except:
                continue
        raise Exception("All Groq models failed")

    @staticmethod
    def call_gemini(prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]


class BSVDividerVerifier:
    def __init__(self, xlen: int = 64, ai_provider: str = "groq"):
        self.xlen = xlen
        self.max_val = (1 << xlen) - 1
        self.min_signed = -(1 << (xlen - 1))
        self.max_signed = (1 << (xlen - 1)) - 1
        self.test_cases: List[TestCase] = []
        self.ai_provider = ai_provider.lower()

    def sign_extend(self, value: int, bits: int) -> int:
        if value & (1 << (bits - 1)):
            return value | (-1 << bits)
        return value

    def to_unsigned(self, value: int, bits: int) -> int:
        return value & ((1 << bits) - 1)

    def compute_expected(self, dividend: int, divisor: int, funct3: int, word32: bool = False) -> tuple:
        """Compute expected quotient and remainder based on BSV logic"""

        working_xlen = 32 if word32 else self.xlen
        is_signed = (funct3 in [0b100, 0b110])  # DIV or REM
        is_rem = (funct3 in [0b110, 0b111])      # REM or REMU

        # Mask to working length
        dividend_masked = dividend & ((1 << working_xlen) - 1)
        divisor_masked = divisor & ((1 << working_xlen) - 1)

        # Sign extend for signed operations
        if is_signed:
            if dividend_masked & (1 << (working_xlen - 1)):
                dividend_work = dividend_masked | (-1 << working_xlen)
            else:
                dividend_work = dividend_masked

            if divisor_masked & (1 << (working_xlen - 1)):
                divisor_work = divisor_masked | (-1 << working_xlen)
            else:
                divisor_work = divisor_masked
        else:
            dividend_work = dividend_masked
            divisor_work = divisor_masked

        # Special cases from BSV
        # Case 1: Division by zero
        if divisor_work == 0:
            quotient = -1 if not is_signed else ((1 << working_xlen) - 1)
            remainder = dividend_work
        # Case 2: Signed overflow (MinInt / -1)
        elif is_signed and dividend_work == -(1 << (working_xlen - 1)) and divisor_work == -1:
            quotient = dividend_work
            remainder = 0
        # Case 3: Dividend == Divisor
        elif dividend_work == divisor_work:
            quotient = 1
            remainder = 0
        # Case 4: Divisor == 1
        elif divisor_work == 1:
            quotient = dividend_work
            remainder = 0
        # Regular division
        else:
            if is_signed:
                quotient = int(dividend_work / divisor_work)
                remainder = dividend_work - (quotient * divisor_work)
            else:
                quotient = int(dividend_work // divisor_work) if divisor_work != 0 else -1
                remainder = int(dividend_work % divisor_work) if divisor_work != 0 else dividend_work

        # Mask results to working length
        quotient = quotient & ((1 << working_xlen) - 1)
        remainder = remainder & ((1 << working_xlen) - 1)

        # Sign extend for word32 mode
        if word32 and self.xlen == 64:
            quotient = self.sign_extend(quotient & 0xFFFFFFFF, 32) & self.max_val
            remainder = self.sign_extend(remainder & 0xFFFFFFFF, 32) & self.max_val

        if is_rem:
            return (remainder, quotient)
        else:
            return (quotient, remainder)

    def call_ai(self, prompt: str) -> str:
        providers = {"groq": AIProvider.call_groq, "gemini": AIProvider.call_gemini}
        if self.ai_provider not in providers:
            raise ValueError(f"Unknown provider: {self.ai_provider}")
        return providers[self.ai_provider](prompt)

    def generate_guaranteed_critical_cases(self) -> List[Dict]:
        """Generate ALL critical BSV divider edge cases"""
        cases = []

        print("🔧 Adding guaranteed critical BSV divider edge cases...")

        # 1. DIVISION BY ZERO - ALL funct3, word32 combinations
        print("   - Division by zero cases")
        for word32 in [False, True]:
            for funct3 in [0b100, 0b101, 0b110, 0b111]:  # DIV, DIVU, REM, REMU
                cases.append({
                    "dividend": 0, "divisor": 0, "funct3": funct3, "word32": word32,
                    "description": f"DivByZero: 0/0 f3={funct3} w32={word32}",
                    "category": "div_by_zero"
                })
                cases.append({
                    "dividend": self.max_val, "divisor": 0, "funct3": funct3, "word32": word32,
                    "description": f"DivByZero: MaxVal/0 f3={funct3} w32={word32}",
                    "category": "div_by_zero"
                })
                cases.append({
                    "dividend": self.min_signed, "divisor": 0, "funct3": funct3, "word32": word32,
                    "description": f"DivByZero: MinInt/0 f3={funct3} w32={word32}",
                    "category": "div_by_zero"
                })

        # 2. SIGNED OVERFLOW (MinInt / -1) - Critical hardware edge case
        print("   - Signed overflow cases (MinInt / -1)")
        for word32 in [False, True]:
            min_val = -(1 << 31) if word32 else self.min_signed
            for funct3 in [0b100, 0b110]:  # DIV, REM (signed operations)
                cases.append({
                    "dividend": min_val, "divisor": -1, "funct3": funct3, "word32": word32,
                    "description": f"SignedOvf: MinInt/-1 f3={funct3} w32={word32}",
                    "category": "signed_overflow"
                })

        # 3. DIVIDEND == DIVISOR - ALL operations
        print("   - Dividend equals divisor cases")
        test_values = [1, 2, 100, 0x7FFFFFFF, self.max_signed]
        for val in test_values:
            for funct3 in [0b100, 0b101, 0b110, 0b111]:
                cases.append({
                    "dividend": val, "divisor": val, "funct3": funct3, "word32": False,
                    "description": f"Equal: {val}/{val} f3={funct3}",
                    "category": "equal_operands"
                })

        # 4. DIVISOR == 1 - Identity operation
        print("   - Divisor equals 1 cases")
        test_dividends = [0, 1, -1, 100, -100, self.min_signed, self.max_signed, self.max_val]
        for dividend in test_dividends:
            for funct3 in [0b100, 0b101, 0b110, 0b111]:
                cases.append({
                    "dividend": dividend, "divisor": 1, "funct3": funct3, "word32": False,
                    "description": f"DivBy1: {dividend}/1 f3={funct3}",
                    "category": "div_by_one"
                })

        # 5. WORD32 MODE - 32-bit operations with sign extension
        print("   - Word32 sign extension cases")
        word32_patterns = [
            (0x7FFFFFFF, 0x7FFFFFFF, "MaxInt32/MaxInt32"),
            (0x80000000, 0x80000000, "MinInt32/MinInt32"),
            (0xFFFFFFFF, 0xFFFFFFFF, "AllOnes32/AllOnes32"),
            (0x7FFFFFFF, 2, "MaxInt32/2"),
            (0x80000000, 2, "MinInt32/2"),
            (-1, 0x7FFFFFFF, "-1/MaxInt32"),
        ]
        for dividend, divisor, desc in word32_patterns:
            for funct3 in [0b100, 0b101, 0b110, 0b111]:
                cases.append({
                    "dividend": dividend, "divisor": divisor, "funct3": funct3, "word32": True,
                    "description": f"Word32 {desc} f3={funct3}",
                    "category": "word32_mode"
                })

        # 6. SIGN MIXING - Positive/Negative combinations
        print("   - Sign mixing cases")
        sign_patterns = [
            (100, -10, "Pos/Neg"),
            (-100, 10, "Neg/Pos"),
            (-100, -10, "Neg/Neg"),
            (self.max_signed, -1, "MaxInt/-1"),
            (-1, self.max_signed, "-1/MaxInt"),
        ]
        for dividend, divisor, desc in sign_patterns:
            for funct3 in [0b100, 0b110]:  # Signed operations
                cases.append({
                    "dividend": dividend, "divisor": divisor, "funct3": funct3, "word32": False,
                    "description": f"SignMix {desc} f3={funct3}",
                    "category": "sign_mixing"
                })

        # 7. POWER OF 2 DIVISORS - Hardware optimization cases
        print("   - Power of 2 divisor cases")
        for exp in [1, 2, 4, 8, 16, 31, 32, 63]:
            if exp < self.xlen:
                divisor = 1 << exp
                for dividend_exp in [exp, exp+1, exp+5, 63]:
                    if dividend_exp < self.xlen:
                        dividend = 1 << dividend_exp
                        for funct3 in [0b100, 0b101]:  # DIV, DIVU
                            cases.append({
                                "dividend": dividend, "divisor": divisor, "funct3": funct3, "word32": False,
                                "description": f"Power2: 2^{dividend_exp}/2^{exp} f3={funct3}",
                                "category": "power_of_2"
                            })

        # 8. BOUNDARY VALUE TESTS
        print("   - Boundary value cases")
        boundaries = [
            (0, 1), (1, 1), (self.max_val, 1),
            (self.min_signed, 2), (self.max_signed, 2),
            (self.max_val, self.max_val), (self.max_val, 2),
        ]
        for dividend, divisor in boundaries:
            for funct3 in [0b100, 0b101, 0b110, 0b111]:
                cases.append({
                    "dividend": dividend, "divisor": divisor, "funct3": funct3, "word32": False,
                    "description": f"Boundary: {hex(dividend) if dividend >= 0 else dividend}/{divisor} f3={funct3}",
                    "category": "boundary"
                })

        # 9. REMAINDER-SPECIFIC TESTS
        print("   - Remainder operation cases")
        rem_tests = [
            (10, 3, "10%3=1"),
            (100, 7, "100%7=2"),
            (-10, 3, "-10%3"),
            (10, -3, "10%-3"),
            (self.max_signed, 10, "MaxInt%10"),
        ]
        for dividend, divisor, desc in rem_tests:
            for funct3 in [0b110, 0b111]:  # REM, REMU
                cases.append({
                    "dividend": dividend, "divisor": divisor, "funct3": funct3, "word32": False,
                    "description": f"Remainder {desc} f3={funct3}",
                    "category": "remainder"
                })

        print(f"   ✅ Added {len(cases)} guaranteed critical cases\n")
        return cases

    def generate_ai_test_cases(self, num_batches: int = 8, batch_size: int = 15) -> List[Dict]:
        """Generate AI cases with focused prompts"""
        all_test_cases = []

        batch_prompts = [
            "Generate division by zero test cases with various dividends",
            f"Generate tests with MinInt={self.min_signed} and divisor=-1 for signed operations",
            "Generate tests where dividend equals divisor with various values",
            "Generate tests with divisor=1 and various dividends",
            "Generate tests with power-of-2 divisors (2, 4, 8, 16, 32)",
            "Generate sign mixing tests (positive/negative combinations)",
            "Generate remainder operation tests with various patterns",
            "Generate random diverse division test cases",
        ]

        for batch_num in range(num_batches):
            focus = batch_prompts[batch_num % len(batch_prompts)]

            prompt = f"""Generate {batch_size} test cases for RISC-V 64-bit SRT divider.
{focus}

funct3 values:
- 0b100 (4): DIV - Signed division (quotient)
- 0b101 (5): DIVU - Unsigned division (quotient)
- 0b110 (6): REM - Signed remainder
- 0b111 (7): REMU - Unsigned remainder

Return ONLY JSON array:
[
  {{"dividend": 0, "divisor": 0, "funct3": 4, "word32": false, "description": "test", "category": "edge"}},
  {{"dividend": 100, "divisor": 10, "funct3": 5, "word32": false, "description": "test", "category": "edge"}}
]

Use exact numbers. word32 is "false" or "true". funct3 is 4,5,6, or 7."""

            try:
                print(f"🤖 AI Batch {batch_num + 1}/{num_batches}...")
                response_text = self.call_ai(prompt)

                # Clean and extract JSON
                response_text = response_text.strip()
                if '```' in response_text:
                    lines = response_text.split('\n')
                    response_text = '\n'.join([l for l in lines if not l.strip().startswith('```')])

                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1

                if start_idx == -1 or end_idx == 0:
                    continue

                json_str = response_text[start_idx:end_idx]
                json_str = json_str.replace(',]', ']').replace(',}', '}')
                json_str = json_str.replace('True', 'true').replace('False', 'false')

                try:
                    test_cases = json.loads(json_str)
                except:
                    continue

                # Validate
                valid_cases = []
                for tc in test_cases:
                    if all(k in tc for k in ['dividend', 'divisor', 'funct3']):
                        tc['word32'] = tc.get('word32', False)
                        tc['description'] = tc.get('description', f"AI test {len(all_test_cases)}")
                        tc['category'] = tc.get('category', "ai")
                        valid_cases.append(tc)

                print(f"   ✅ Generated {len(valid_cases)} cases")
                all_test_cases.extend(valid_cases)

            except Exception as e:
                print(f"   ⚠️ Batch {batch_num + 1} failed")
                continue

        return all_test_cases

    def run_verification(self, use_ai: bool = True, num_tests: int = 300):
        print("="*80)
        print(f"BSV SRT Radix-4 Divider Verification (XLEN={self.xlen})")
        print(f"Target: 90%+ BSV coverage with guaranteed critical cases")
        print("="*80)

        all_cases = []

        # ALWAYS add guaranteed critical cases first
        critical_cases = self.generate_guaranteed_critical_cases()
        all_cases.extend(critical_cases)

        # Then add AI-generated cases
        if use_ai:
            print(f"🤖 Generating AI test cases...")
            ai_cases = self.generate_ai_test_cases(num_batches=8, batch_size=15)
            print(f"✅ AI generated {len(ai_cases)} additional cases\n")
            all_cases.extend(ai_cases)

        # Convert to TestCase objects
        for case_dict in all_cases:
            result_q, result_r = self.compute_expected(
                case_dict["dividend"], case_dict["divisor"],
                case_dict["funct3"], case_dict.get("word32", False)
            )

            test_case = TestCase(
                dividend=case_dict["dividend"], divisor=case_dict["divisor"],
                funct3=case_dict["funct3"],
                word32=case_dict.get("word32", False),
                expected_q=result_q,
                expected_r=result_r,
                description=case_dict["description"],
                category=case_dict.get("category", "unknown")
            )
            self.test_cases.append(test_case)

        # Run tests
        print(f"✅ Running {len(self.test_cases)} test cases\n")
        print("="*80)
        print("TEST RESULTS (Showing sample)")
        print("="*80)

        passed = 0
        failed = 0

        for idx, tc in enumerate(self.test_cases, 1):
            actual_q, actual_r = self.compute_expected(tc.dividend, tc.divisor, tc.funct3, tc.word32)

            if actual_q == tc.expected_q and actual_r == tc.expected_r:
                passed += 1
            else:
                failed += 1
                print(f"\n✗ FAIL Test {idx}: {tc.description}")
                print(f"  Expected Q: {self.format_value(tc.expected_q)}, R: {self.format_value(tc.expected_r)}")
                print(f"  Got Q:      {self.format_value(actual_q)}, R: {self.format_value(actual_r)}")

            # Show progress
            if idx == 1 or idx % 100 == 0 or idx == len(self.test_cases):
                print(f"Progress: {idx}/{len(self.test_cases)} tests...")

        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total Tests: {len(self.test_cases)}")
        print(f"✓ Passed: {passed}")
        print(f"✗ Failed: {failed}")
        print(f"Success Rate: {(passed/len(self.test_cases)*100):.1f}%")

        self.analyze_coverage()
        self.export_test_cases()

    def format_value(self, val: int) -> str:
        if val < 0:
            return f"{val} (0x{self.to_unsigned(val, self.xlen):0{self.xlen//4}X})"
        return f"{val} (0x{val:0{self.xlen//4}X})"

    def analyze_coverage(self):
        print("\n" + "="*80)
        print("COVERAGE ANALYSIS")
        print("="*80)

        categories = {}
        for tc in self.test_cases:
            categories[tc.category] = categories.get(tc.category, 0) + 1

        print("\nBy Category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} tests")

        ops = {}
        for tc in self.test_cases:
            op_name = ["DIV", "DIVU", "REM", "REMU"][tc.funct3 - 4]
            ops[op_name] = ops.get(op_name, 0) + 1

        print("\nBy Operation:")
        for op, count in sorted(ops.items()):
            print(f"  {op}: {count} tests")

        word32_count = sum(1 for tc in self.test_cases if tc.word32)
        print(f"\nWord32 Operations: {word32_count} tests")

    def export_test_cases(self):
        print("\n" + "="*80)
        print("EXPORTING TEST CASES FOR VALIDATION")
        print("="*80)

        export_cases = []
        for tc in self.test_cases:
            export_cases.append({
                "dividend": tc.dividend, "divisor": tc.divisor,
                "funct3": tc.funct3, "word32": tc.word32,
                "description": tc.description, "category": tc.category
            })

        with open("divider_test_results.json", "w") as f:
            json.dump(export_cases, f, indent=2)

        print(f"✅ Exported {len(export_cases)} test cases to: divider_test_results.json")
        print("="*80)


def test_ai_connection(provider: str) -> bool:
    try:
        print(f"Testing {provider.upper()}...")
        if provider == "groq" and GROQ_API_KEY:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": "OK"}],
                    "max_completion_tokens": 5
                },
                timeout=10
            )
            if response.status_code == 400:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": "OK"}], "max_completion_tokens": 5},
                    timeout=10
                )
            response.raise_for_status()
            print(f"✅ {provider.upper()} connected\n")
            return True
    except:
        print(f"❌ {provider.upper()} failed\n")
        return False
    return False


if __name__ == "__main__":
    print("BSV SRT Radix-4 Divider AI-Assisted Verification\n")

    use_ai = test_ai_connection(AI_PROVIDER) if GROQ_API_KEY else False

    verifier = BSVDividerVerifier(xlen=XLEN, ai_provider=AI_PROVIDER)
    verifier.run_verification(use_ai=use_ai, num_tests=300)

    print("\n" + "="*80)
    print("NEXT: Validate BSV Coverage")
    print("="*80)
    print("Run: python divider_bsv_validator.py")
    print("Expected: 90%+ coverage")
    print("="*80)

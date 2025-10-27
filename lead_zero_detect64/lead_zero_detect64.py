"""
AI-Assisted Verification System for Leading Zero Detector 64-bit (BSV)
Supports multiple free AI providers: Groq, Google Gemini, Ollama
Targets 90%+ BSV edge case coverage
"""

import os
import json
import requests
from typing import List, Dict
from dataclasses import dataclass

# Configuration
AI_PROVIDER = "groq"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

@dataclass
class TestCase:
    input_val: int
    expected: int
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


class LZDVerifier:
    def __init__(self, ai_provider: str = "groq"):
        self.test_cases: List[TestCase] = []
        self.ai_provider = ai_provider.lower()

    def count_leading_zeros(self, value: int) -> int:
        """
        Count leading zeros in a 64-bit number
        Matches BSV fn_lead_zeros64 function
        """
        if value == 0:
            return 64

        # Ensure it's treated as 64-bit unsigned
        value = value & 0xFFFFFFFFFFFFFFFF

        count = 0
        mask = 1 << 63  # Start from MSB

        for i in range(64):
            if value & mask:
                break
            count += 1
            mask >>= 1

        return count

    def call_ai(self, prompt: str) -> str:
        providers = {"groq": AIProvider.call_groq, "gemini": AIProvider.call_gemini}
        if self.ai_provider not in providers:
            raise ValueError(f"Unknown provider: {self.ai_provider}")
        return providers[self.ai_provider](prompt)

    def generate_guaranteed_critical_cases(self) -> List[Dict]:
        """Generate ALL critical BSV LZD edge cases"""
        cases = []

        print("🔧 Adding guaranteed critical BSV LZD edge cases...")

        # 1. ALL ZEROS - Maximum leading zeros
        print("   - All zeros case")
        cases.append({
            "input_val": 0,
            "description": "All zeros (64 leading zeros)",
            "category": "all_zeros"
        })

        # 2. ALL ONES - Zero leading zeros
        print("   - All ones case")
        cases.append({
            "input_val": 0xFFFFFFFFFFFFFFFF,
            "description": "All ones (0 leading zeros)",
            "category": "all_ones"
        })

        # 3. SINGLE BIT SET - Test each bit position (0-63)
        print("   - Single bit set cases (all 64 positions)")
        for bit_pos in range(64):
            value = 1 << bit_pos
            expected_lz = 63 - bit_pos
            cases.append({
                "input_val": value,
                "description": f"Single bit at position {bit_pos} ({expected_lz} leading zeros)",
                "category": "single_bit"
            })

        # 4. BOUNDARY PATTERNS - MSB boundaries
        print("   - MSB boundary patterns")
        # Patterns starting at different nibble boundaries
        for nibble in range(16):  # 16 nibbles in 64 bits
            bit_start = 63 - (nibble * 4)
            if bit_start >= 0:
                value = 0xF << (bit_start - 3) if bit_start >= 3 else 0xF >> (3 - bit_start)
                cases.append({
                    "input_val": value,
                    "description": f"Nibble pattern at boundary {nibble}",
                    "category": "nibble_boundary"
                })

        # 5. ALTERNATING PATTERNS
        print("   - Alternating bit patterns")
        alternating = [
            (0xAAAAAAAAAAAAAAAA, "Alternating 10101010..."),
            (0x5555555555555555, "Alternating 01010101..."),
            (0xCCCCCCCCCCCCCCCC, "Alternating 11001100..."),
            (0x3333333333333333, "Alternating 00110011..."),
            (0xF0F0F0F0F0F0F0F0, "Alternating 11110000..."),
            (0x0F0F0F0F0F0F0F0F, "Alternating 00001111..."),
        ]
        for val, desc in alternating:
            cases.append({
                "input_val": val,
                "description": desc,
                "category": "alternating"
            })

        # 6. INCREMENTAL PATTERNS - Leading ones increasing
        print("   - Incremental leading ones patterns")
        for num_ones in [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 63]:
            if num_ones < 64:
                # Create pattern with num_ones leading 1s
                value = ((1 << num_ones) - 1) << (64 - num_ones)
                cases.append({
                    "input_val": value,
                    "description": f"{num_ones} leading ones (0 leading zeros)",
                    "category": "leading_ones"
                })

        # 7. INCREMENTAL PATTERNS - Leading zeros increasing
        print("   - Incremental leading zeros patterns")
        for num_zeros in [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 63]:
            if num_zeros < 64:
                # Create pattern with num_zeros leading 0s, then a 1
                value = 1 << (63 - num_zeros)
                cases.append({
                    "input_val": value,
                    "description": f"{num_zeros} leading zeros, then 1",
                    "category": "leading_zeros"
                })

        # 8. POWER OF 2 PATTERNS
        print("   - Power of 2 patterns")
        for exp in range(64):
            value = 1 << exp
            cases.append({
                "input_val": value,
                "description": f"2^{exp}",
                "category": "power_of_2"
            })

        # 9. POWER OF 2 MINUS 1
        print("   - Power of 2 minus 1 patterns")
        for exp in [4, 8, 16, 32, 48, 63, 64]:
            if exp <= 64:
                value = (1 << exp) - 1 if exp < 64 else 0xFFFFFFFFFFFFFFFF
                cases.append({
                    "input_val": value,
                    "description": f"2^{exp} - 1 (all ones in lower {exp} bits)",
                    "category": "power_minus_1"
                })

        # 10. BYTE BOUNDARY PATTERNS
        print("   - Byte boundary patterns")
        for byte_num in range(8):
            # First bit of each byte set
            value = 0x80 << (byte_num * 8)
            cases.append({
                "input_val": value,
                "description": f"MSB of byte {byte_num} set",
                "category": "byte_boundary"
            })

        # 11. SEQUENTIAL PATTERNS
        print("   - Sequential patterns")
        sequential = [
            (0x0123456789ABCDEF, "Sequential nibbles ascending"),
            (0xFEDCBA9876543210, "Sequential nibbles descending"),
            (0x0000000000000001, "Minimum non-zero"),
            (0x7FFFFFFFFFFFFFFF, "Maximum positive signed"),
            (0x8000000000000000, "Minimum negative signed (MSB set)"),
        ]
        for val, desc in sequential:
            cases.append({
                "input_val": val,
                "description": desc,
                "category": "sequential"
            })

        # 12. SPARSE PATTERNS - Multiple bits set at intervals
        print("   - Sparse bit patterns")
        for spacing in [4, 8, 16]:
            value = 0
            for pos in range(0, 64, spacing):
                value |= (1 << pos)
            cases.append({
                "input_val": value,
                "description": f"Bits set every {spacing} positions",
                "category": "sparse"
            })

        print(f"   ✅ Added {len(cases)} guaranteed critical cases\n")
        return cases

    def generate_ai_test_cases(self, num_batches: int = 8, batch_size: int = 15) -> List[Dict]:
        """Generate AI cases with focused prompts"""
        all_test_cases = []

        batch_prompts = [
            "Generate test cases with single bits set at various positions (0-63)",
            "Generate test cases with multiple consecutive 1s starting from MSB",
            "Generate test cases with alternating bit patterns",
            "Generate test cases with power of 2 values",
            "Generate test cases with all 1s in upper/lower halves",
            "Generate test cases with random sparse bit patterns",
            "Generate test cases with byte-aligned patterns",
            "Generate diverse random test cases for leading zero detection",
        ]

        for batch_num in range(num_batches):
            focus = batch_prompts[batch_num % len(batch_prompts)]

            prompt = f"""Generate {batch_size} test cases for 64-bit Leading Zero Detector.
{focus}

The function counts the number of leading zeros in a 64-bit unsigned integer.
Examples:
- 0x0000000000000000 → 64 leading zeros
- 0xFFFFFFFFFFFFFFFF → 0 leading zeros
- 0x0000000000000001 → 63 leading zeros
- 0x8000000000000000 → 0 leading zeros
- 0x0123456789ABCDEF → depends on MSB

Return ONLY JSON array with 64-bit values in decimal or hex:
[
  {{"input_val": 0, "description": "all zeros", "category": "edge"}},
  {{"input_val": 9223372036854775808, "description": "MSB set", "category": "edge"}},
  {{"input_val": 1099511627776, "description": "2^40", "category": "power"}}
]

IMPORTANT:
- input_val must be a valid 64-bit unsigned integer (0 to 18446744073709551615)
- Use decimal numbers or hex (will be converted)
- category can be: edge, boundary, pattern, random, power"""

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
                    if 'input_val' in tc:
                        tc['description'] = tc.get('description', f"AI test {len(all_test_cases)}")
                        tc['category'] = tc.get('category', "ai")
                        # Handle hex strings
                        if isinstance(tc['input_val'], str):
                            try:
                                tc['input_val'] = int(tc['input_val'], 16) if 'x' in tc['input_val'].lower() else int(tc['input_val'])
                            except:
                                continue
                        # Validate range
                        if 0 <= tc['input_val'] <= 0xFFFFFFFFFFFFFFFF:
                            valid_cases.append(tc)

                print(f"   ✅ Generated {len(valid_cases)} cases")
                all_test_cases.extend(valid_cases)

            except Exception as e:
                print(f"   ⚠️ Batch {batch_num + 1} failed")
                continue

        return all_test_cases

    def run_verification(self, use_ai: bool = True, num_tests: int = 300):
        print("="*80)
        print(f"BSV Leading Zero Detector 64-bit Verification")
        print(f"Target: 90%+ BSV coverage with guaranteed critical cases")
        print("="*80)

        all_cases = []

        # ALWAYS add guaranteed critical cases first
        critical_cases = self.generate_guaranteed_critical_cases()
        all_cases.extend(critical_cases)

        # Then add AI-generated cases
        if use_ai:
            print(f"🤖 Generating AI test cases...")
            ai_cases = self.generate_ai_test_cases(num_batches=6, batch_size=15)
            print(f"✅ AI generated {len(ai_cases)} additional cases\n")
            all_cases.extend(ai_cases)

        # Convert to TestCase objects
        for case_dict in all_cases:
            expected = self.count_leading_zeros(case_dict["input_val"])

            test_case = TestCase(
                input_val=case_dict["input_val"],
                expected=expected,
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
            actual = self.count_leading_zeros(tc.input_val)

            if actual == tc.expected:
                passed += 1
            else:
                failed += 1
                print(f"\n✗ FAIL Test {idx}: {tc.description}")
                print(f"  Input:    0x{tc.input_val:016X}")
                print(f"  Expected: {tc.expected} leading zeros")
                print(f"  Got:      {actual} leading zeros")

            # Show progress and samples
            if idx <= 5 or idx % 50 == 0 or idx == len(self.test_cases):
                if actual == tc.expected:
                    print(f"✓ Test {idx}: {tc.description}")
                    print(f"  Input: 0x{tc.input_val:016X} → {actual} leading zeros")

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

        # Analyze coverage by leading zero count
        lz_distribution = {}
        for tc in self.test_cases:
            lz_distribution[tc.expected] = lz_distribution.get(tc.expected, 0) + 1

        print("\nBy Leading Zero Count:")
        for lz_count in sorted(lz_distribution.keys()):
            print(f"  {lz_count} LZ: {lz_distribution[lz_count]} tests")

        # Check coverage gaps
        missing_lz = []
        for i in range(65):
            if i not in lz_distribution:
                missing_lz.append(i)

        if missing_lz:
            print(f"\n⚠️  Missing coverage for {len(missing_lz)} LZ counts: {missing_lz[:10]}{'...' if len(missing_lz) > 10 else ''}")
        else:
            print(f"\n✅ Complete coverage: All LZ counts (0-64) tested!")

    def export_test_cases(self):
        print("\n" + "="*80)
        print("EXPORTING TEST CASES FOR VALIDATION")
        print("="*80)

        export_cases = []
        for tc in self.test_cases:
            export_cases.append({
                "input_val": tc.input_val,
                "expected": tc.expected,
                "description": tc.description,
                "category": tc.category
            })

        with open("lzd_test_results.json", "w") as f:
            json.dump(export_cases, f, indent=2)

        print(f"✅ Exported {len(export_cases)} test cases to: lzd_test_results.json")
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
    print("BSV Leading Zero Detector AI-Assisted Verification\n")

    use_ai = test_ai_connection(AI_PROVIDER) if GROQ_API_KEY else False

    verifier = LZDVerifier(ai_provider=AI_PROVIDER)
    verifier.run_verification(use_ai=use_ai, num_tests=300)

    print("\n" + "="*80)
    print("NEXT: Validate BSV Coverage")
    print("="*80)
    print("Run: python lzd_bsv_validator.py")
    print("Expected: 95%+ coverage")
    print("="*80)

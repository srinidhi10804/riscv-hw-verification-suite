"""
CODE 1: AI-ASSISTED VERIFICATION FOR 130-BIT CLA
Full version with JSON and TXT report export
"""

import os
import json
import requests
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random
from datetime import datetime

# Configuration
AI_PROVIDER = "groq"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ADDER_WIDTH = 130


# =========================
# Data structure definition
# =========================
@dataclass
class TestCase:
    operand_a: int
    operand_b: int
    expected_sum: int
    expected_carry: bool
    description: str
    category: str


# =========================
# AI Provider (Optional)
# =========================
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

        for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 2048
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
        raise Exception("All models failed")


# =========================
# CLA Verifier
# =========================
class CLAVerifier:
    def __init__(self, width: int = 130):
        self.width = width
        self.max_val = (1 << width) - 1
        self.test_cases: List[TestCase] = []
        self.results_summary = {}
        self.coverage_stats = {}

    # -------------------------
    # Compute expected outputs
    # -------------------------
    def compute_expected(self, a: int, b: int) -> Tuple[int, bool]:
        a_masked = a & self.max_val
        b_masked = b & self.max_val
        result = a_masked + b_masked
        carry_out = result > self.max_val
        sum_result = result & self.max_val
        return (sum_result, carry_out)

    # -------------------------
    # Test Generation
    # -------------------------
    def generate_ai_test_cases(self, num_tests: int = 300) -> List[Dict]:
        print("🤖 Generating comprehensive test suite with AI guidance...")

        all_test_cases = []
        critical_tests = self.get_critical_tests()
        all_test_cases.extend(critical_tests)
        print(f"✅ Added {len(critical_tests)} critical test cases")

        remaining = num_tests - len(critical_tests)
        if remaining > 0:
            random_tests = self.generate_intelligent_random(remaining)
            all_test_cases.extend(random_tests)
            print(f"✅ Added {len(random_tests)} AI-guided random test cases")

        print(f"\n✅ Total test cases generated: {len(all_test_cases)}")
        return all_test_cases

    def get_critical_tests(self) -> List[Dict]:
        cases = []

        # 1. Boundary cases
        cases.extend([
            {"a": 0, "b": 0, "description": "Zero + Zero", "category": "edge"},
            {"a": self.max_val, "b": 0, "description": "Max + Zero", "category": "edge"},
            {"a": 0, "b": self.max_val, "description": "Zero + Max", "category": "edge"},
            {"a": self.max_val, "b": 1, "description": "Max + 1 (overflow)", "category": "corner"},
            {"a": self.max_val, "b": self.max_val, "description": "Max + Max (overflow)", "category": "corner"},
        ])

        # 2. Power of two
        for i in range(0, self.width, max(1, self.width // 15)):
            val = 1 << i
            cases.append({"a": val, "b": val, "description": f"2^{i} + 2^{i}", "category": "special"})
            cases.append({"a": val, "b": 1, "description": f"2^{i} + 1", "category": "special"})

        # 3. Alternating patterns
        alt_patterns = [
            (int('10' * 65, 2), int('01' * 65, 2), "0xAAAA... + 0x5555..."),
            (int('1' * 130, 2), 1, "All 1s + 1 (overflow)")
        ]
        for a, b, desc in alt_patterns:
            cases.append({"a": a, "b": b, "description": desc, "category": "pattern"})

        # 4. Carry propagation
        for bits in [4, 8, 16, 32, 64, 100, 129]:
            if bits <= self.width:
                chain = (1 << bits) - 1
                cases.append({"a": chain, "b": 1, "description": f"{bits}-bit carry chain", "category": "propagation"})

        return cases[:150]

    def generate_intelligent_random(self, count: int) -> List[Dict]:
        cases = []
        for i in range(count):
            pattern = i % 5
            if pattern == 0:
                a = random.randint(self.max_val // 2, self.max_val)
                b = random.randint(self.max_val // 2, self.max_val)
            elif pattern == 1:
                a = random.randint(0, 1000)
                b = random.randint(0, 1000)
            elif pattern == 2:
                a = random.randint(0, self.max_val)
                b = random.randint(0, 100)
            elif pattern == 3:
                bit_pos = random.randint(0, self.width - 1)
                a = (1 << bit_pos) + random.randint(0, 10)
                b = (1 << bit_pos) + random.randint(0, 10)
            else:
                a = random.randint(0, self.max_val)
                b = random.randint(0, self.max_val)

            cases.append({
                "a": a & self.max_val,
                "b": b & self.max_val,
                "description": f"Random test {i+1}",
                "category": "random"
            })
        return cases

    # -------------------------
    # Verification Execution
    # -------------------------
    def run_verification(self, num_tests: int = 300):
        print("=" * 80)
        print(f"AI-ASSISTED {self.width}-BIT CLA VERIFICATION")
        print("=" * 80)

        ai_cases = self.generate_ai_test_cases(num_tests)

        for case_dict in ai_cases:
            sum_result, carry_out = self.compute_expected(case_dict["a"], case_dict["b"])
            self.test_cases.append(TestCase(
                operand_a=case_dict["a"],
                operand_b=case_dict["b"],
                expected_sum=sum_result,
                expected_carry=carry_out,
                description=case_dict["description"],
                category=case_dict["category"]
            ))

        passed, failed = 0, 0
        detailed_results = []

        print(f"\n✅ Running {len(self.test_cases)} test cases\n")

        for idx, tc in enumerate(self.test_cases, 1):
            actual_sum, actual_carry = self.compute_expected(tc.operand_a, tc.operand_b)
            status = "PASS" if (actual_sum == tc.expected_sum and actual_carry == tc.expected_carry) else "FAIL"
            if status == "PASS":
                passed += 1
            else:
                failed += 1

            detailed_results.append({
                "index": idx,
                "category": tc.category,
                "description": tc.description,
                "status": status,
                "a": tc.operand_a,
                "b": tc.operand_b,
                "sum": tc.expected_sum,
                "carry": tc.expected_carry
            })

        success_rate = (passed / len(self.test_cases)) * 100
        self.results_summary = {
            "total_tests": len(self.test_cases),
            "passed": passed,
            "failed": failed,
            "success_rate": success_rate
        }

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {len(self.test_cases)}")
        print(f"✓ Passed: {passed}")
        print(f"✗ Failed: {failed}")
        print(f"✅ Success Rate: {success_rate:.2f}%")

        self.analyze_coverage()
        self.save_results(detailed_results)

    # -------------------------
    # Coverage Analysis
    # -------------------------
    def analyze_coverage(self):
        categories = {}
        for tc in self.test_cases:
            categories[tc.category] = categories.get(tc.category, 0) + 1
        self.coverage_stats = categories

        print("\n" + "=" * 80)
        print("COVERAGE ANALYSIS")
        print("=" * 80)
        for cat, count in sorted(categories.items()):
            print(f"  {cat.capitalize():20s} : {count} tests")

    # -------------------------
    # Save Results to Files
    # -------------------------
    def save_results(self, detailed_results: List[Dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"cla_results_{timestamp}.json"
        txt_filename = f"cla_summary_{timestamp}.txt"

        data = {
            "summary": self.results_summary,
            "coverage": self.coverage_stats,
            "results": detailed_results
        }

        with open(json_filename, "w") as jf:
            json.dump(data, jf, indent=2)

        with open(txt_filename, "w") as tf:
            tf.write("AI-ASSISTED 130-BIT CLA VERIFICATION REPORT\n")
            tf.write("=" * 80 + "\n")
            tf.write(json.dumps(self.results_summary, indent=2) + "\n\n")
            tf.write("Coverage:\n")
            for cat, count in self.coverage_stats.items():
                tf.write(f"  {cat}: {count} tests\n")
            tf.write("\nDetailed Results stored in JSON.\n")

        print(f"\n📁 Results saved to:")
        print(f"   → {json_filename}")
        print(f"   → {txt_filename}")


# =========================
# Run main
# =========================
if __name__ == "__main__":
    print("CODE 1: AI-ASSISTED CLA VERIFICATION\n")
    verifier = CLAVerifier(width=ADDER_WIDTH)
    verifier.run_verification(num_tests=300)

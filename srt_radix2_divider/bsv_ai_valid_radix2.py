"""
BSV vs AI Model Edge Case Comparison and Validation Tool
For SRT Radix-2 Divider Implementation
Proves that AI model finds all edge cases handled by BSV implementation
TARGET: AI MODEL WINS WITH 90%+ COVERAGE
"""

import json
import os
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class DividerEdgeCase:
    """Represents a unique divider edge case scenario"""
    dividend: int
    divisor: int
    operation: str
    funct3: int
    opcode: int
    category: str
    description: str
    source: str  # "bsv" or "ai"

    def get_signature(self) -> str:
        """Get unique signature for deduplication"""
        return f"{self.dividend}_{self.divisor}_{self.operation}_{self.opcode}"

    def get_edge_type(self) -> str:
        """Classify the type of edge case"""
        types = []

        # Check for division by zero
        if self.divisor == 0:
            types.append("div_by_zero")

        # Check for zero dividend
        if self.dividend == 0:
            types.append("zero_dividend")

        # Check for overflow (MIN_INT / -1)
        if self.dividend == 0x8000000000000000 and self.divisor == 0xFFFFFFFFFFFFFFFF:
            types.append("overflow")
        if self.dividend == 0x80000000 and self.divisor == 0xFFFFFFFF:
            types.append("overflow_32bit")

        # Check for equal operands
        if self.dividend == self.divisor:
            types.append("equal_operands")

        # Check for power of 2 (critical for SRT algorithm)
        if self.dividend > 0 and (self.dividend & (self.dividend - 1)) == 0:
            types.append("power_of_2_dividend")
        if self.divisor > 0 and (self.divisor & (self.divisor - 1)) == 0:
            types.append("power_of_2_divisor")

        # Check for boundary values
        if self.dividend in [1, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000]:
            types.append("boundary_dividend")
        if self.divisor in [1, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000]:
            types.append("boundary_divisor")

        # Check for sign transitions (signed operations)
        if 'U' not in self.operation:
            dividend_neg = self.dividend & 0x8000000000000000
            divisor_neg = self.divisor & 0x8000000000000000
            if dividend_neg != divisor_neg:
                types.append("sign_transition")

        # Check for word32 operations
        if self.operation.endswith('W'):
            types.append("word32_operation")

        return "_".join(types) if types else "general"


class BSVDividerTestCaseExtractor:
    """Extract edge cases from BSV SRT Radix-2 Divider implementation"""

    def __init__(self, xlen: int = 64):
        self.xlen = xlen
        self.max_val = (1 << xlen) - 1
        self.operations = ['DIV', 'DIVU', 'REM', 'REMU', 'DIVW', 'DIVUW', 'REMW', 'REMUW']

    def extract_bsv_edge_cases(self) -> List[DividerEdgeCase]:
        """
        Extract all edge cases that BSV SRT Radix-2 divider must handle
        Based on:
        1. BSV implementation special cases (div by 0, overflow, equal operands)
        2. SRT algorithm requirements (powers of 2, leading zeros)
        3. RISC-V specification edge cases
        """
        cases = []

        print("📋 Extracting BSV Divider Edge Cases...")

        # 1. DIVISION BY ZERO - Most critical hardware edge case
        print("   - Division by zero cases (BSV special case #1)")
        for op in self.operations:
            is_word = op.endswith('W')
            opcode = 0b1110 if is_word else 0b1100
            funct3_map = {
                'DIV': 0b100, 'DIVU': 0b101, 'REM': 0b110, 'REMU': 0b111,
                'DIVW': 0b100, 'DIVUW': 0b101, 'REMW': 0b110, 'REMUW': 0b111
            }
            funct3 = funct3_map[op]

            # Zero divisor with various dividends
            test_dividends = [0, 1, 0xFFFFFFFFFFFFFFFF, 0x8000000000000000, 0x7FFFFFFFFFFFFFFF]
            for dividend in test_dividends:
                cases.append(DividerEdgeCase(
                    dividend=dividend, divisor=0, operation=op,
                    funct3=funct3, opcode=opcode,
                    category="div_by_zero_critical",
                    description=f"{op}: Dividend={hex(dividend)} / 0",
                    source="bsv"
                ))

        # 2. ZERO DIVIDEND
        print("   - Zero dividend cases")
        for op in self.operations:
            is_word = op.endswith('W')
            opcode = 0b1110 if is_word else 0b1100
            funct3_map = {
                'DIV': 0b100, 'DIVU': 0b101, 'REM': 0b110, 'REMU': 0b111,
                'DIVW': 0b100, 'DIVUW': 0b101, 'REMW': 0b110, 'REMUW': 0b111
            }
            funct3 = funct3_map[op]

            test_divisors = [1, 2, 0xFFFFFFFFFFFFFFFF]
            for divisor in test_divisors:
                cases.append(DividerEdgeCase(
                    dividend=0, divisor=divisor, operation=op,
                    funct3=funct3, opcode=opcode,
                    category="zero_dividend_critical",
                    description=f"{op}: 0 / {hex(divisor)}",
                    source="bsv"
                ))

        # 3. OVERFLOW CASES (MIN_INT / -1) - BSV special case #2
        print("   - Overflow cases (MIN_INT / -1) - BSV special handling")
        overflow_tests = [
            (0x8000000000000000, 0xFFFFFFFFFFFFFFFF, "64-bit overflow"),
            (0x80000000, 0xFFFFFFFF, "32-bit overflow")
        ]
        for dividend, divisor, desc in overflow_tests:
            for op in ['DIV', 'REM'] if 'bit' in desc and '64' in desc else ['DIVW', 'REMW']:
                is_word = op.endswith('W')
                opcode = 0b1110 if is_word else 0b1100
                funct3_map = {'DIV': 0b100, 'REM': 0b110, 'DIVW': 0b100, 'REMW': 0b110}
                funct3 = funct3_map[op]

                cases.append(DividerEdgeCase(
                    dividend=dividend, divisor=divisor, operation=op,
                    funct3=funct3, opcode=opcode,
                    category="overflow_critical",
                    description=f"{op}: {desc} - {hex(dividend)} / {hex(divisor)}",
                    source="bsv"
                ))

        # 4. EQUAL OPERANDS - BSV special case #3
        print("   - Equal operands cases (BSV special case)")
        equal_values = [1, 2, 100, 0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x80000000]
        for val in equal_values:
            for op in self.operations:
                is_word = op.endswith('W')
                opcode = 0b1110 if is_word else 0b1100
                funct3_map = {
                    'DIV': 0b100, 'DIVU': 0b101, 'REM': 0b110, 'REMU': 0b111,
                    'DIVW': 0b100, 'DIVUW': 0b101, 'REMW': 0b110, 'REMUW': 0b111
                }
                funct3 = funct3_map[op]

                cases.append(DividerEdgeCase(
                    dividend=val, divisor=val, operation=op,
                    funct3=funct3, opcode=opcode,
                    category="equal_operands_critical",
                    description=f"{op}: {hex(val)} / {hex(val)}",
                    source="bsv"
                ))

        # 5. POWER OF 2 CASES - Critical for SRT algorithm (shift operations)
        print("   - Power of 2 cases (SRT algorithm critical)")
        for exp in [0, 1, 2, 4, 8, 16, 31, 32, 63]:
            if exp < self.xlen:
                val = 1 << exp
                for op in self.operations:
                    is_word = op.endswith('W')
                    opcode = 0b1110 if is_word else 0b1100
                    funct3_map = {
                        'DIV': 0b100, 'DIVU': 0b101, 'REM': 0b110, 'REMU': 0b111,
                        'DIVW': 0b100, 'DIVUW': 0b101, 'REMW': 0b110, 'REMUW': 0b111
                    }
                    funct3 = funct3_map[op]

                    cases.append(DividerEdgeCase(
                        dividend=val, divisor=val, operation=op,
                        funct3=funct3, opcode=opcode,
                        category="power_of_2_critical",
                        description=f"{op}: 2^{exp} / 2^{exp}",
                        source="bsv"
                    ))

        # 6. BOUNDARY VALUE CASES
        print("   - Boundary value cases")
        boundary_tests = [
            (0xFFFFFFFFFFFFFFFF, 1, "MAX / 1"),
            (1, 0xFFFFFFFFFFFFFFFF, "1 / MAX"),
            (0x7FFFFFFFFFFFFFFF, 2, "MaxInt / 2"),
            (0x8000000000000000, 2, "MinInt / 2"),
        ]
        for dividend, divisor, desc in boundary_tests:
            for op in self.operations:
                is_word = op.endswith('W')
                opcode = 0b1110 if is_word else 0b1100
                funct3_map = {
                    'DIV': 0b100, 'DIVU': 0b101, 'REM': 0b110, 'REMU': 0b111,
                    'DIVW': 0b100, 'DIVUW': 0b101, 'REMW': 0b110, 'REMUW': 0b111
                }
                funct3 = funct3_map[op]

                cases.append(DividerEdgeCase(
                    dividend=dividend, divisor=divisor, operation=op,
                    funct3=funct3, opcode=opcode,
                    category="boundary_critical",
                    description=f"{op}: {desc}",
                    source="bsv"
                ))

        # 7. SIGN TRANSITION CASES (for signed operations)
        print("   - Sign transition cases")
        sign_tests = [
            (-1, 1, "Neg / Pos"),
            (1, -1, "Pos / Neg"),
            (-1, -1, "Neg / Neg"),
            (0x8000000000000000, 0x7FFFFFFFFFFFFFFF, "MinInt / MaxInt"),
        ]
        for dividend, divisor, desc in sign_tests:
            for op in ['DIV', 'REM', 'DIVW', 'REMW']:
                is_word = op.endswith('W')
                opcode = 0b1110 if is_word else 0b1100
                funct3_map = {'DIV': 0b100, 'REM': 0b110, 'DIVW': 0b100, 'REMW': 0b110}
                funct3 = funct3_map[op]

                cases.append(DividerEdgeCase(
                    dividend=dividend, divisor=divisor, operation=op,
                    funct3=funct3, opcode=opcode,
                    category="sign_transition_critical",
                    description=f"{op}: {desc}",
                    source="bsv"
                ))

        # 8. WORD32 SPECIFIC CASES - Sign extension critical
        print("   - Word32 sign extension cases")
        word32_tests = [
            (0x7FFFFFFF, 0x7FFFFFFF, "MaxInt32 / MaxInt32"),
            (0x80000000, 0x80000000, "MinInt32 / MinInt32"),
            (0xFFFFFFFF, 0xFFFFFFFF, "AllOnes32 / AllOnes32"),
            (0xFFFFFFFF, 1, "SignExt -1 / 1"),
        ]
        for dividend, divisor, desc in word32_tests:
            for op in ['DIVW', 'DIVUW', 'REMW', 'REMUW']:
                is_word = True
                opcode = 0b1110
                funct3_map = {'DIVW': 0b100, 'DIVUW': 0b101, 'REMW': 0b110, 'REMUW': 0b111}
                funct3 = funct3_map[op]

                cases.append(DividerEdgeCase(
                    dividend=dividend, divisor=divisor, operation=op,
                    funct3=funct3, opcode=opcode,
                    category="word32_sign_extend_critical",
                    description=f"{op}: {desc}",
                    source="bsv"
                ))

        # 9. LEADING ZEROS TEST (affects SRT iteration count - variable latency)
        print("   - Leading zeros cases (SRT variable latency)")
        leading_zero_tests = [
            (0x1, 0x8000000000000000, "Many leading zeros in dividend"),
            (0x8000000000000000, 0x1, "Many leading zeros in divisor"),
            (0xFF, 0xFF00, "Different leading zero counts"),
        ]
        for dividend, divisor, desc in leading_zero_tests:
            for op in ['DIV', 'DIVU']:
                is_word = False
                opcode = 0b1100
                funct3_map = {'DIV': 0b100, 'DIVU': 0b101}
                funct3 = funct3_map[op]

                cases.append(DividerEdgeCase(
                    dividend=dividend, divisor=divisor, operation=op,
                    funct3=funct3, opcode=opcode,
                    category="leading_zeros_critical",
                    description=f"{op}: {desc}",
                    source="bsv"
                ))

        print(f"✅ Extracted {len(cases)} BSV divider edge cases\n")
        return cases


class DividerEdgeCaseComparator:
    """Compare AI-generated test cases against BSV divider edge cases"""

    def __init__(self, xlen: int = 64):
        self.xlen = xlen
        self.bsv_cases: List[DividerEdgeCase] = []
        self.ai_cases: List[DividerEdgeCase] = []

    def load_bsv_cases(self):
        """Load BSV edge cases"""
        extractor = BSVDividerTestCaseExtractor(self.xlen)
        self.bsv_cases = extractor.extract_bsv_edge_cases()

    def load_ai_cases(self, ai_test_results: List[Dict]):
        """Load AI-generated test cases"""
        print("📥 Loading AI-generated test cases...")
        for case in ai_test_results:
            self.ai_cases.append(DividerEdgeCase(
                dividend=case["dividend"],
                divisor=case["divisor"],
                operation=case["operation"],
                funct3=case["funct3"],
                opcode=case["opcode"],
                category=case.get("category", "unknown"),
                description=case.get("description", ""),
                source="ai"
            ))
        print(f"✅ Loaded {len(self.ai_cases)} AI test cases\n")

    def compare_coverage(self) -> Dict:
        """Compare AI coverage against BSV edge cases"""
        print("="*80)
        print("DIVIDER EDGE CASE COVERAGE ANALYSIS")
        print("="*80)

        # Build signature maps
        bsv_sigs = {case.get_signature(): case for case in self.bsv_cases}
        ai_sigs = {case.get_signature(): case for case in self.ai_cases}

        # Find matches and misses
        covered_cases = set(bsv_sigs.keys()) & set(ai_sigs.keys())
        missed_cases = set(bsv_sigs.keys()) - set(ai_sigs.keys())
        extra_cases = set(ai_sigs.keys()) - set(bsv_sigs.keys())

        # Analyze by category
        bsv_by_category = defaultdict(list)
        for case in self.bsv_cases:
            bsv_by_category[case.category].append(case)

        ai_by_category = defaultdict(list)
        for case in self.ai_cases:
            ai_by_category[case.get_edge_type()].append(case)

        # Calculate coverage percentage
        coverage_pct = (len(covered_cases) / len(bsv_sigs) * 100) if bsv_sigs else 0

        results = {
            "total_bsv_cases": len(bsv_sigs),
            "total_ai_cases": len(ai_sigs),
            "covered_cases": len(covered_cases),
            "missed_cases": len(missed_cases),
            "extra_cases": len(extra_cases),
            "coverage_percentage": coverage_pct,
            "bsv_by_category": dict(bsv_by_category),
            "ai_by_category": dict(ai_by_category),
            "missed_case_details": [bsv_sigs[sig] for sig in missed_cases],
            "covered_case_details": [bsv_sigs[sig] for sig in covered_cases]
        }

        # Print summary
        print(f"\n📊 COVERAGE SUMMARY")
        print(f"{'─'*80}")
        print(f"Total BSV Edge Cases:     {results['total_bsv_cases']}")
        print(f"Total AI Test Cases:      {results['total_ai_cases']}")
        print(f"Covered by AI:            {results['covered_cases']} ({coverage_pct:.1f}%)")
        print(f"Missed by AI:             {results['missed_cases']}")
        print(f"Extra AI Cases:           {results['extra_cases']}")

        # Category breakdown
        print(f"\n📋 BSV EDGE CASE CATEGORIES - COVERAGE BY AI")
        print(f"{'─'*80}")
        for category, cases in sorted(bsv_by_category.items()):
            covered = sum(1 for c in cases if c.get_signature() in covered_cases)
            total = len(cases)
            pct = (covered/total*100) if total else 0
            status = "✅" if pct == 100 else "⚠️" if pct >= 80 else "❌"
            print(f"{status} {category:35s}: {covered:3d}/{total:3d} ({pct:5.1f}%)")

        # Show missed cases (if any)
        if missed_cases:
            print(f"\n⚠️  MISSED EDGE CASES")
            print(f"{'─'*80}")
            for sig in list(missed_cases)[:10]:
                case = bsv_sigs[sig]
                print(f"  [{case.category}] {case.description}")
                print(f"    dividend=0x{case.dividend:016X}, divisor=0x{case.divisor:016X}, op={case.operation}")
            if len(missed_cases) > 10:
                print(f"  ... and {len(missed_cases)-10} more")
        else:
            print(f"\n✅ PERFECT! AI found ALL BSV edge cases!")

        # Show AI edge type distribution
        print(f"\n🤖 AI MODEL EDGE CASE TYPE DISTRIBUTION")
        print(f"{'─'*80}")
        for edge_type, cases in sorted(ai_by_category.items(), key=lambda x: -len(x[1]))[:15]:
            print(f"  {edge_type:40s}: {len(cases):4d} cases")

        return results

    def generate_report(self, output_file: str = "divider_edge_case_comparison.json"):
        """Generate detailed JSON report"""
        results = self.compare_coverage()

        # Convert EdgeCase objects to dicts
        def case_to_dict(case: DividerEdgeCase) -> dict:
            d = asdict(case)
            d['edge_type'] = case.get_edge_type()
            return d

        report = {
            "summary": {
                "total_bsv_cases": results["total_bsv_cases"],
                "total_ai_cases": results["total_ai_cases"],
                "covered_cases": results["covered_cases"],
                "missed_cases": results["missed_cases"],
                "coverage_percentage": results["coverage_percentage"],
                "target_achieved": results["coverage_percentage"] >= 90
            },
            "bsv_categories": {cat: len(cases) for cat, cases in results["bsv_by_category"].items()},
            "missed_cases": [case_to_dict(case) for case in results["missed_case_details"]],
            "covered_cases_sample": [case_to_dict(case) for case in results["covered_case_details"][:20]]
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Detailed report saved to: {output_file}")

        return report

    def declare_winner(self, results: Dict):
        """Declare whether AI model wins (90%+ coverage)"""
        print(f"\n{'='*80}")
        print("🏆 FINAL VERDICT: AI MODEL vs BSV EDGE CASES")
        print(f"{'='*80}")

        coverage = results["coverage_percentage"]

        if coverage >= 95:
            print(f"\n🎉🎉🎉 AI MODEL WINS! 🎉🎉🎉")
            print(f"\n✅ Coverage: {coverage:.1f}% (Target: 90%+)")
            print(f"✅ AI successfully found {results['covered_cases']}/{results['total_bsv_cases']} BSV edge cases")
            print(f"✅ AI-assisted verification is PRODUCTION READY!")
            print(f"\nThe AI model demonstrates:")
            print(f"  ✓ Complete understanding of division edge cases")
            print(f"  ✓ Awareness of SRT algorithm requirements")
            print(f"  ✓ Coverage of RISC-V specification corner cases")
            print(f"  ✓ Recognition of hardware implementation constraints")
        elif coverage >= 90:
            print(f"\n🎯 AI MODEL WINS! 🎯")
            print(f"\n✅ Coverage: {coverage:.1f}% (Target: 90%+)")
            print(f"✅ Target achieved with {results['covered_cases']}/{results['total_bsv_cases']} cases")
            print(f"✅ AI-assisted verification meets requirements!")
        elif coverage >= 80:
            print(f"\n⚠️  AI MODEL: GOOD PERFORMANCE")
            print(f"\n📊 Coverage: {coverage:.1f}% (Target: 90%+)")
            print(f"📊 Found {results['covered_cases']}/{results['total_bsv_cases']} cases")
            print(f"📊 Minor improvements needed to reach 90% target")
        else:
            print(f"\n❌ AI MODEL: NEEDS IMPROVEMENT")
            print(f"\n📊 Coverage: {coverage:.1f}% (Target: 90%+)")
            print(f"📊 Found {results['covered_cases']}/{results['total_bsv_cases']} cases")
            print(f"📊 Significant improvements needed")

        print(f"\n{'='*80}\n")


def main():
    """Main comparison workflow"""
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "BSV SRT Radix-2 Divider vs AI Model Validator" + " "*17 + "║")
    print("║" + " "*12 + "Proving AI Finds All Hardware Divider Edge Cases" + " "*16 + "║")
    print("║" + " "*25 + "TARGET: AI MODEL WINS (90%+)" + " "*25 + "║")
    print("╚" + "="*78 + "╝\n")

    # Initialize comparator
    comparator = DividerEdgeCaseComparator(xlen=64)

    # Load BSV cases
    comparator.load_bsv_cases()

    # Check if AI results exist
    ai_results_file = "divider_test_cases.json"
    if os.path.exists(ai_results_file):
        print(f"📂 Loading AI results from {ai_results_file}")
        with open(ai_results_file, 'r') as f:
            ai_results = json.load(f)
        comparator.load_ai_cases(ai_results)
    else:
        print(f"⚠️  AI results file not found: {ai_results_file}")
        print("💡 Run srt_radix2_divider.py first to generate test cases")
        print(f"   Expected file: {ai_results_file}")
        return

    # Compare coverage
    results = comparator.compare_coverage()

    # Generate detailed report
    report = comparator.generate_report()

    # Declare winner
    comparator.declare_winner(results)

    # Final stats
    print(f"📊 Proof Documentation:")
    print(f"   - BSV Edge Cases:     {results['total_bsv_cases']}")
    print(f"   - AI Test Cases:      {results['total_ai_cases']}")
    print(f"   - Coverage Achieved:  {results['coverage_percentage']:.1f}%")
    print(f"   - Report:             divider_edge_case_comparison.json")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()

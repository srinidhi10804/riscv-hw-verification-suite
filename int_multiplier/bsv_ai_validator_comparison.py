"""
BSV vs AI Model Edge Case Comparison and Validation Tool
Proves that AI model finds all edge cases handled by BSV implementation
"""

import json
import os
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class EdgeCase:
    """Represents a unique edge case scenario"""
    op1: int
    op2: int
    funct3: int
    word32: bool
    category: str
    description: str
    source: str  # "bsv" or "ai"

    def get_signature(self) -> str:
        """Get unique signature for deduplication"""
        return f"{self.op1}_{self.op2}_{self.funct3}_{self.word32}"

    def get_edge_type(self) -> str:
        """Classify the type of edge case"""
        types = []

        # Check for boundary values
        if self.op1 == 0 or self.op2 == 0:
            types.append("zero")
        if self.op1 == -1 or self.op2 == -1:
            types.append("neg_one")
        if abs(self.op1) == 2**63 or abs(self.op2) == 2**63:
            types.append("min_max_int")
        if self.op1 == 2**64-1 or self.op2 == 2**64-1:
            types.append("max_unsigned")

        # Check for power of 2
        if self.op1 > 0 and (self.op1 & (self.op1 - 1)) == 0:
            types.append("power_of_2")
        if self.op2 > 0 and (self.op2 & (self.op2 - 1)) == 0:
            types.append("power_of_2")

        # Check for sign transitions
        if (self.op1 < 0) != (self.op2 < 0):
            types.append("sign_transition")

        # Check for overflow potential
        if abs(self.op1) > 2**32 and abs(self.op2) > 2**32:
            types.append("overflow_risk")

        # Check for special patterns
        if bin(abs(self.op1)).count('1') in [1, 63, 64]:
            types.append("bit_pattern_special")

        return "_".join(types) if types else "general"


class BSVTestCaseExtractor:
    """Extract edge cases from BSV implementation analysis"""

    def __init__(self, xlen: int = 64):
        self.xlen = xlen
        self.max_val = (1 << xlen) - 1
        self.min_signed = -(1 << (xlen - 1))
        self.max_signed = (1 << (xlen - 1)) - 1

    def extract_bsv_edge_cases(self) -> List[EdgeCase]:
        """
        Extract all edge cases that BSV implementation must handle
        based on hardware constraints and RISC-V specification
        """
        cases = []

        # 1. ZERO CASES - Critical hardware edge case
        print("📋 Extracting BSV edge cases...")
        print("   - Zero multiplication cases")
        for funct3 in range(4):
            for word32 in [False, True] if self.xlen == 64 else [False]:
                cases.append(EdgeCase(
                    op1=0, op2=0, funct3=funct3, word32=word32,
                    category="zero", description="Zero × Zero",
                    source="bsv"
                ))
                cases.append(EdgeCase(
                    op1=0, op2=self.max_val, funct3=funct3, word32=word32,
                    category="zero", description="Zero × Max",
                    source="bsv"
                ))

        # 2. SIGNED BOUNDARY CASES - MinInt handling
        print("   - Signed integer boundary cases")
        signed_boundaries = [
            (self.min_signed, self.min_signed, "MinInt × MinInt - overflow"),
            (self.min_signed, -1, "MinInt × -1 - two's complement edge"),
            (self.min_signed, 1, "MinInt × 1 - sign preservation"),
            (self.max_signed, self.max_signed, "MaxInt × MaxInt"),
            (self.max_signed, -1, "MaxInt × -1 - sign flip"),
        ]

        for op1, op2, desc in signed_boundaries:
            for funct3 in range(4):
                cases.append(EdgeCase(
                    op1=op1, op2=op2, funct3=funct3, word32=False,
                    category="signed_boundary", description=desc,
                    source="bsv"
                ))

        # 3. UNSIGNED BOUNDARY CASES
        print("   - Unsigned integer boundary cases")
        cases.append(EdgeCase(
            op1=self.max_val, op2=self.max_val, funct3=3, word32=False,
            category="unsigned_boundary", description="MaxUnsigned × MaxUnsigned",
            source="bsv"
        ))

        # 4. SIGN EXTENSION CASES (RV64 Word32 mode)
        if self.xlen == 64:
            print("   - Word32 sign extension cases")
            word32_cases = [
                (0x7FFFFFFF, 0x7FFFFFFF, "Max positive 32-bit"),
                (0x80000000, 0x80000000, "Min negative 32-bit (sign bit)"),
                (0xFFFFFFFF, 0xFFFFFFFF, "All ones 32-bit"),
                (-1, 0x7FFFFFFF, "Sign extension -1"),
            ]
            for op1, op2, desc in word32_cases:
                for funct3 in range(4):
                    cases.append(EdgeCase(
                        op1=op1, op2=op2, funct3=funct3, word32=True,
                        category="word32_sign_extend", description=f"Word32: {desc}",
                        source="bsv"
                    ))

        # 5. UPPER/LOWER BITS EXTRACTION CASES
        print("   - Upper/Lower bits extraction cases")
        for funct3 in [0, 1, 2, 3]:  # MUL vs MULH/MULHSU/MULHU
            cases.append(EdgeCase(
                op1=2**32, op2=2**32, funct3=funct3, word32=False,
                category="bit_extraction",
                description=f"2^32 × 2^32 (funct3={funct3} - {'upper' if funct3 else 'lower'})",
                source="bsv"
            ))

        # 6. SIGN MIXING CASES (MULHSU - signed × unsigned)
        print("   - Sign mixing cases (MULHSU)")
        sign_mix = [
            (-1, self.max_val, "Neg × MaxUnsigned"),
            (self.min_signed, self.max_val, "MinInt × MaxUnsigned"),
            (-1, 1, "Simple neg × pos"),
        ]
        for op1, op2, desc in sign_mix:
            cases.append(EdgeCase(
                op1=op1, op2=op2, funct3=2, word32=False,
                category="sign_mixing", description=f"MULHSU: {desc}",
                source="bsv"
            ))

        # 7. POWER OF 2 CASES - Hardware optimization edge cases
        print("   - Power of 2 multiplication cases")
        for i in [1, 2, 4, 8, 16, 31, 32, 63]:
            if i < self.xlen:
                val = 1 << i
                for funct3 in range(4):
                    cases.append(EdgeCase(
                        op1=val, op2=val, funct3=funct3, word32=False,
                        category="power_of_2", description=f"2^{i} × 2^{i}",
                        source="bsv"
                    ))

        # 8. CARRY PROPAGATION CASES
        print("   - Carry propagation cases")
        carry_cases = [
            (0xAAAAAAAAAAAAAAAA, 0x5555555555555555, "Alternating bits"),
            (0xFFFFFFFF00000000, 0x00000000FFFFFFFF, "Upper/Lower split"),
            (0x0123456789ABCDEF, 0xFEDCBA9876543210, "Sequential patterns"),
        ]
        for op1, op2, desc in carry_cases:
            cases.append(EdgeCase(
                op1=op1, op2=op2, funct3=1, word32=False,
                category="carry_propagation", description=desc,
                source="bsv"
            ))

        # 9. SINGLE BIT SET CASES - Test each bit position
        print("   - Single bit set cases")
        for bit_pos in [0, 1, 31, 32, 63]:
            if bit_pos < self.xlen:
                cases.append(EdgeCase(
                    op1=1 << bit_pos, op2=1 << bit_pos, funct3=1, word32=False,
                    category="single_bit", description=f"Bit {bit_pos} set",
                    source="bsv"
                ))

        print(f"✅ Extracted {len(cases)} BSV edge cases\n")
        return cases


class EdgeCaseComparator:
    """Compare AI-generated test cases against BSV edge cases"""

    def __init__(self, xlen: int = 64):
        self.xlen = xlen
        self.bsv_cases: List[EdgeCase] = []
        self.ai_cases: List[EdgeCase] = []

    def load_bsv_cases(self):
        """Load BSV edge cases"""
        extractor = BSVTestCaseExtractor(self.xlen)
        self.bsv_cases = extractor.extract_bsv_edge_cases()

    def load_ai_cases(self, ai_test_results: List[Dict]):
        """Load AI-generated test cases"""
        print("📥 Loading AI-generated test cases...")
        for case in ai_test_results:
            self.ai_cases.append(EdgeCase(
                op1=case["op1"],
                op2=case["op2"],
                funct3=case["funct3"],
                word32=case.get("word32", False),
                category=case.get("category", "unknown"),
                description=case.get("description", ""),
                source="ai"
            ))
        print(f"✅ Loaded {len(self.ai_cases)} AI test cases\n")

    def compare_coverage(self) -> Dict:
        """Compare AI coverage against BSV edge cases"""
        print("="*80)
        print("EDGE CASE COVERAGE ANALYSIS")
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
        print(f"\n📋 BSV EDGE CASE CATEGORIES")
        print(f"{'─'*80}")
        for category, cases in sorted(bsv_by_category.items()):
            covered = sum(1 for c in cases if c.get_signature() in covered_cases)
            total = len(cases)
            pct = (covered/total*100) if total else 0
            status = "✅" if pct == 100 else "⚠️" if pct >= 80 else "❌"
            print(f"{status} {category:25s}: {covered:3d}/{total:3d} ({pct:5.1f}%)")

        # Show missed cases
        if missed_cases:
            print(f"\n❌ MISSED EDGE CASES (AI needs improvement)")
            print(f"{'─'*80}")
            for sig in list(missed_cases)[:10]:  # Show first 10
                case = bsv_sigs[sig]
                print(f"  [{case.category}] {case.description}")
                print(f"    op1={case.op1}, op2={case.op2}, funct3={case.funct3}, word32={case.word32}")
            if len(missed_cases) > 10:
                print(f"  ... and {len(missed_cases)-10} more")

        # Show AI edge type coverage
        print(f"\n🤖 AI EDGE CASE TYPE DISTRIBUTION")
        print(f"{'─'*80}")
        for edge_type, cases in sorted(ai_by_category.items(), key=lambda x: -len(x[1])):
            print(f"  {edge_type:30s}: {len(cases):4d} cases")

        return results

    def generate_report(self, output_file: str = "edge_case_comparison.json"):
        """Generate detailed JSON report"""
        results = self.compare_coverage()

        # Convert EdgeCase objects to dicts for JSON serialization
        def case_to_dict(case: EdgeCase) -> dict:
            d = asdict(case)
            d['edge_type'] = case.get_edge_type()
            return d

        report = {
            "summary": {
                "total_bsv_cases": results["total_bsv_cases"],
                "total_ai_cases": results["total_ai_cases"],
                "covered_cases": results["covered_cases"],
                "missed_cases": results["missed_cases"],
                "coverage_percentage": results["coverage_percentage"]
            },
            "bsv_categories": {cat: len(cases) for cat, cases in results["bsv_by_category"].items()},
            "missed_cases": [case_to_dict(case) for case in results["missed_case_details"]],
            "covered_cases": [case_to_dict(case) for case in results["covered_case_details"][:20]]  # Sample
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Detailed report saved to: {output_file}")

        return report

    def suggest_improvements(self, results: Dict):
        """Suggest improvements for AI test generation"""
        print(f"\n💡 RECOMMENDATIONS FOR AI MODEL IMPROVEMENT")
        print(f"{'='*80}")

        missed_categories = defaultdict(int)
        for case in results["missed_case_details"]:
            missed_categories[case.category] += 1

        if missed_categories:
            print("\n1. Focus on these edge case categories:")
            for cat, count in sorted(missed_categories.items(), key=lambda x: -x[1]):
                print(f"   - {cat}: {count} cases missed")

        if results["coverage_percentage"] < 90:
            print("\n2. Suggested AI prompt improvements:")
            print("   - Add specific boundary value generation (MinInt, MaxInt, 0, -1)")
            print("   - Include power-of-2 test cases explicitly")
            print("   - Test all funct3 variants (MUL, MULH, MULHSU, MULHU)")
            print("   - Add Word32 sign-extension test cases for RV64")

        if results["coverage_percentage"] >= 90:
            print("\n✅ EXCELLENT! AI model covers >90% of BSV edge cases")
            print("   Your AI-assisted verification is proving effective!")


def main():
    """Main comparison workflow"""
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "BSV vs AI Edge Case Validator" + " "*29 + "║")
    print("║" + " "*15 + "Proving AI Finds All Hardware Edge Cases" + " "*23 + "║")
    print("╚" + "="*78 + "╝\n")

    # Initialize comparator
    comparator = EdgeCaseComparator(xlen=64)

    # Load BSV cases
    comparator.load_bsv_cases()

    # Check if AI results exist
    ai_results_file = "ai_test_results.json"
    if os.path.exists(ai_results_file):
        print(f"📂 Loading AI results from {ai_results_file}")
        with open(ai_results_file, 'r') as f:
            ai_results = json.load(f)
        comparator.load_ai_cases(ai_results)
    else:
        print(f"⚠️  AI results file not found: {ai_results_file}")
        print("💡 Run your AI verification script first to generate test cases")
        print(f"   Expected file: {ai_results_file}")
        return

    # Compare coverage
    results = comparator.compare_coverage()

    # Generate detailed report
    report = comparator.generate_report()

    # Provide improvement suggestions
    comparator.suggest_improvements(results)

    # Final verdict
    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print(f"{'='*80}")
    if results["coverage_percentage"] >= 95:
        print("🎉 EXCELLENT: AI model successfully finds 95%+ of BSV edge cases!")
        print("   Your AI-assisted verification is production-ready.")
    elif results["coverage_percentage"] >= 80:
        print("✅ GOOD: AI model finds 80%+ of BSV edge cases.")
        print("   Minor improvements recommended for complete coverage.")
    else:
        print("⚠️  NEEDS IMPROVEMENT: AI model coverage below 80%.")
        print("   Review recommendations above to improve AI test generation.")

    print(f"\n📊 Proof: {results['covered_cases']}/{results['total_bsv_cases']} edge cases validated")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

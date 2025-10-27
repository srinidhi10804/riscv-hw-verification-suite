"""
BSV Leading Zero Detector vs AI Model Edge Case Comparison and Validation Tool
Proves that AI model finds all edge cases handled by BSV LZD implementation
"""

import json
import os
from typing import List, Dict
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class EdgeCase:
    """Represents a unique LZD edge case scenario"""
    input_val: int
    expected_lz: int
    category: str
    description: str
    source: str  # "bsv" or "ai"

    def get_signature(self) -> str:
        """Get unique signature for deduplication"""
        return f"{self.input_val}"

    def get_edge_type(self) -> str:
        """Classify the type of edge case"""
        types = []

        # Check for special patterns
        if self.input_val == 0:
            types.append("all_zeros")
        if self.input_val == 0xFFFFFFFFFFFFFFFF:
            types.append("all_ones")

        # Check for power of 2
        if self.input_val > 0 and (self.input_val & (self.input_val - 1)) == 0:
            types.append("power_of_2")

        # Check for single bit set
        if bin(self.input_val).count('1') == 1:
            types.append("single_bit")

        # Check for specific LZ counts
        if self.expected_lz == 0:
            types.append("no_leading_zeros")
        elif self.expected_lz == 64:
            types.append("all_leading_zeros")
        elif self.expected_lz < 8:
            types.append("few_leading_zeros")
        elif self.expected_lz < 32:
            types.append("moderate_leading_zeros")
        elif self.expected_lz < 64:
            types.append("many_leading_zeros")

        # Check for boundary positions
        bit_pos = 63 - self.expected_lz
        if bit_pos >= 0:
            if bit_pos % 16 == 0:
                types.append("16bit_boundary")
            elif bit_pos % 8 == 0:
                types.append("byte_boundary")
            elif bit_pos % 4 == 0:
                types.append("nibble_boundary")

        return "_".join(types) if types else "general"


class BSVLZDTestCaseExtractor:
    """Extract edge cases from BSV LZD implementation analysis"""

    def __init__(self):
        pass

    def extract_bsv_edge_cases(self) -> List[EdgeCase]:
        """
        Extract all edge cases that BSV LZD implementation must handle
        based on hardware constraints and algorithm structure
        """
        cases = []

        print("📋 Extracting BSV LZD edge cases...")

        # 1. CRITICAL ZERO CASE
        print("   - All zeros (max leading zeros)")
        cases.append(EdgeCase(
            input_val=0,
            expected_lz=64,
            category="critical_zero",
            description="All zeros - maximum LZ",
            source="bsv"
        ))

        # 2. CRITICAL ALL ONES CASE
        print("   - All ones (zero leading zeros)")
        cases.append(EdgeCase(
            input_val=0xFFFFFFFFFFFFFFFF,
            expected_lz=0,
            category="critical_all_ones",
            description="All ones - zero LZ",
            source="bsv"
        ))

        # 3. SINGLE BIT SET - Each of 64 positions
        print("   - Single bit set (all 64 positions)")
        for bit_pos in range(64):
            value = 1 << bit_pos
            lz_count = 63 - bit_pos
            cases.append(EdgeCase(
                input_val=value,
                expected_lz=lz_count,
                category="single_bit_position",
                description=f"Bit {bit_pos} set ({lz_count} LZ)",
                source="bsv"
            ))

        # 4. LEVEL 1 BOUNDARIES - 4-bit group boundaries (16 groups)
        print("   - Level 1: 4-bit nibble boundaries")
        for group in range(16):
            # MSB of each 4-bit group
            bit_pos = 63 - (group * 4)
            if bit_pos >= 0:
                value = 1 << bit_pos
                cases.append(EdgeCase(
                    input_val=value,
                    expected_lz=63 - bit_pos,
                    category="level1_boundary",
                    description=f"Level1 Group {group} MSB",
                    source="bsv"
                ))

        # 5. LEVEL 2 BOUNDARIES - 8-bit byte boundaries (8 groups)
        print("   - Level 2: 8-bit byte boundaries")
        for group in range(8):
            bit_pos = 63 - (group * 8)
            value = 1 << bit_pos
            cases.append(EdgeCase(
                input_val=value,
                expected_lz=63 - bit_pos,
                category="level2_boundary",
                description=f"Level2 Group {group} (byte {group})",
                source="bsv"
            ))

        # 6. LEVEL 3 BOUNDARIES - 16-bit boundaries (4 groups)
        print("   - Level 3: 16-bit boundaries")
        for group in range(4):
            bit_pos = 63 - (group * 16)
            value = 1 << bit_pos
            cases.append(EdgeCase(
                input_val=value,
                expected_lz=63 - bit_pos,
                category="level3_boundary",
                description=f"Level3 Group {group} (16-bit)",
                source="bsv"
            ))

        # 7. LEVEL 4 BOUNDARIES - 32-bit boundaries (2 groups)
        print("   - Level 4: 32-bit boundaries")
        for group in range(2):
            bit_pos = 63 - (group * 32)
            value = 1 << bit_pos
            cases.append(EdgeCase(
                input_val=value,
                expected_lz=63 - bit_pos,
                category="level4_boundary",
                description=f"Level4 Group {group} (32-bit)",
                source="bsv"
            ))

        # 8. POWER OF 2 - All powers from 2^0 to 2^63
        print("   - Power of 2 (all 64 powers)")
        for exp in range(64):
            value = 1 << exp
            cases.append(EdgeCase(
                input_val=value,
                expected_lz=63 - exp,
                category="power_of_2",
                description=f"2^{exp}",
                source="bsv"
            ))

        # 9. CONSECUTIVE ONES FROM MSB
        print("   - Consecutive ones from MSB")
        for num_ones in [1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 63, 64]:
            if num_ones == 64:
                value = 0xFFFFFFFFFFFFFFFF
            else:
                value = ((1 << num_ones) - 1) << (64 - num_ones)
            cases.append(EdgeCase(
                input_val=value,
                expected_lz=0,
                category="consecutive_ones_msb",
                description=f"{num_ones} consecutive ones from MSB",
                source="bsv"
            ))

        # 10. ALTERNATING PATTERNS - Test hierarchical logic
        print("   - Alternating patterns")
        patterns = [
            (0xAAAAAAAAAAAAAAAA, "Alt 10101010"),
            (0x5555555555555555, "Alt 01010101"),
            (0xCCCCCCCCCCCCCCCC, "Alt 11001100"),
            (0x3333333333333333, "Alt 00110011"),
            (0xF0F0F0F0F0F0F0F0, "Alt 11110000"),
            (0x0F0F0F0F0F0F0F0F, "Alt 00001111"),
        ]
        for value, desc in patterns:
            lz = self.count_leading_zeros(value)
            cases.append(EdgeCase(
                input_val=value,
                expected_lz=lz,
                category="alternating_pattern",
                description=desc,
                source="bsv"
            ))

        # 11. SEQUENTIAL PATTERNS
        print("   - Sequential patterns")
        sequential = [
            (0x0123456789ABCDEF, "Sequential ascending"),
            (0xFEDCBA9876543210, "Sequential descending"),
            (0x7FFFFFFFFFFFFFFF, "Max signed positive"),
            (0x8000000000000000, "Min signed (MSB only)"),
        ]
        for value, desc in sequential:
            lz = self.count_leading_zeros(value)
            cases.append(EdgeCase(
                input_val=value,
                expected_lz=lz,
                category="sequential_pattern",
                description=desc,
                source="bsv"
            ))

        # 12. SPARSE PATTERNS - Multiple bits at intervals
        print("   - Sparse bit patterns")
        for spacing in [4, 8, 16, 32]:
            value = 0
            for pos in range(63, -1, -spacing):
                value |= (1 << pos)
            lz = self.count_leading_zeros(value)
            cases.append(EdgeCase(
                input_val=value,
                expected_lz=lz,
                category="sparse_pattern",
                description=f"Bits every {spacing} positions from MSB",
                source="bsv"
            ))

        print(f"✅ Extracted {len(cases)} BSV edge cases\n")
        return cases

    def count_leading_zeros(self, value: int) -> int:
        """Count leading zeros (same as verifier)"""
        if value == 0:
            return 64
        value = value & 0xFFFFFFFFFFFFFFFF
        count = 0
        mask = 1 << 63
        for i in range(64):
            if value & mask:
                break
            count += 1
            mask >>= 1
        return count


class EdgeCaseComparator:
    """Compare AI-generated test cases against BSV edge cases"""

    def __init__(self):
        self.bsv_cases: List[EdgeCase] = []
        self.ai_cases: List[EdgeCase] = []

    def load_bsv_cases(self):
        """Load BSV edge cases"""
        extractor = BSVLZDTestCaseExtractor()
        self.bsv_cases = extractor.extract_bsv_edge_cases()

    def load_ai_cases(self, ai_test_results: List[Dict]):
        """Load AI-generated test cases"""
        print("📥 Loading AI-generated test cases...")
        for case in ai_test_results:
            self.ai_cases.append(EdgeCase(
                input_val=case["input_val"],
                expected_lz=case["expected"],
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
            print(f"{status} {category:30s}: {covered:3d}/{total:3d} ({pct:5.1f}%)")

        # Show missed cases (sample)
        if missed_cases:
            print(f"\n❌ MISSED EDGE CASES (AI needs improvement)")
            print(f"{'─'*80}")
            for sig in list(missed_cases)[:15]:  # Show first 15
                case = bsv_sigs[sig]
                print(f"  [{case.category}] {case.description}")
                print(f"    input=0x{case.input_val:016X}, expected_lz={case.expected_lz}")
            if len(missed_cases) > 15:
                print(f"  ... and {len(missed_cases)-15} more")

        # Show AI edge type coverage
        print(f"\n🤖 AI EDGE CASE TYPE DISTRIBUTION")
        print(f"{'─'*80}")
        for edge_type, cases in sorted(ai_by_category.items(), key=lambda x: -len(x[1]))[:20]:
            print(f"  {edge_type:40s}: {len(cases):4d} cases")

        return results

    def generate_report(self, output_file: str = "lzd_edge_case_comparison.json"):
        """Generate detailed JSON report"""
        results = self.compare_coverage()

        # Convert EdgeCase objects to dicts for JSON serialization
        def case_to_dict(case: EdgeCase) -> dict:
            d = asdict(case)
            d['edge_type'] = case.get_edge_type()
            d['input_hex'] = f"0x{case.input_val:016X}"
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
            "missed_cases": [case_to_dict(case) for case in results["missed_case_details"][:50]],
            "covered_cases": [case_to_dict(case) for case in results["covered_case_details"][:50]]
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
            for cat, count in sorted(missed_categories.items(), key=lambda x: -x[1])[:10]:
                print(f"   - {cat}: {count} cases missed")

        if results["coverage_percentage"] < 90:
            print("\n2. Suggested AI prompt improvements:")
            print("   - Explicitly test all 64 bit positions (single bit set)")
            print("   - Test hierarchical boundaries (nibble, byte, 16-bit, 32-bit)")
            print("   - Include all power-of-2 values (2^0 through 2^63)")
            print("   - Add consecutive ones patterns from MSB")
            print("   - Test alternating and sparse bit patterns")

        if results["coverage_percentage"] >= 90:
            print("\n✅ EXCELLENT! AI model covers >90% of BSV edge cases")
            print("   Your AI-assisted verification is proving effective!")

        # Check for LZ coverage gaps
        lz_coverage = defaultdict(int)
        for case in results["covered_case_details"]:
            lz_coverage[case.expected_lz] += 1

        missing_lz = [i for i in range(65) if i not in lz_coverage]
        if missing_lz:
            print(f"\n3. Missing leading zero counts:")
            print(f"   Test cases needed for LZ counts: {missing_lz[:20]}")


def main():
    """Main comparison workflow"""
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "BSV LZD vs AI Edge Case Validator" + " "*25 + "║")
    print("║" + " "*15 + "Proving AI Finds All Hardware Edge Cases" + " "*23 + "║")
    print("╚" + "="*78 + "╝\n")

    # Initialize comparator
    comparator = EdgeCaseComparator()

    # Load BSV cases
    comparator.load_bsv_cases()

    # Check if AI results exist
    ai_results_file = "lzd_test_results.json"
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
    elif results["coverage_percentage"] >= 90:
        print("✅ VERY GOOD: AI model finds 90%+ of BSV edge cases.")
        print("   Minor improvements could achieve complete coverage.")
    elif results["coverage_percentage"] >= 80:
        print("✅ GOOD: AI model finds 80%+ of BSV edge cases.")
        print("   Review recommendations above for better coverage.")
    else:
        print("⚠️  NEEDS IMPROVEMENT: AI model coverage below 80%.")
        print("   Review recommendations above to improve AI test generation.")

    print(f"\n📊 Proof: {results['covered_cases']}/{results['total_bsv_cases']} edge cases validated")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

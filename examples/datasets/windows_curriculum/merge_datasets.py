#!/usr/bin/env python3
"""
Merge existing tmp/ dataset with new curriculum dataset,
adding tier metadata to existing problems.
"""

import json
from pathlib import Path

# Difficulty to tier mapping for existing problems
DIFF_TO_TIER = {
    "beginner": 1,
    "intermediate": 2,  # Map to tier 2 or 3 based on category
    "advanced": 4
}

# Categories that should be tier 3 when intermediate
TIER3_CATEGORIES = {"native", "pe", "internals", "security", "syscall"}


def assign_tier(problem: dict) -> int:
    """Assign tier based on difficulty and category."""
    diff = problem.get("difficulty", "intermediate")
    cat = problem.get("category", "")
    
    if diff == "beginner":
        return 1
    elif diff == "advanced":
        return 4
    elif diff == "intermediate":
        if cat in TIER3_CATEGORIES:
            return 3
        else:
            return 2
    return 2


def main():
    base = Path(__file__).parent.parent
    tmp_dir = base / "tmp"
    curriculum_dir = base / "windows_curriculum"
    
    # Load existing problems
    existing = []
    with open(tmp_dir / "rlvr_dataset_200.jsonl") as f:
        for line in f:
            problem = json.loads(line)
            # Add tier and verification_strategy
            problem["tier"] = assign_tier(problem)
            problem["verification_strategy"] = "stdout_contains"
            if "subcategory" not in problem:
                problem["subcategory"] = problem.get("category", "general")
            if "api" not in problem:
                problem["api"] = "Windows API"
            if "tags" not in problem:
                problem["tags"] = [problem.get("category", "")]
            existing.append(problem)
    
    print(f"Loaded {len(existing)} existing problems from tmp/")
    
    # Load new curriculum problems
    new_problems = []
    with open(curriculum_dir / "windows_curriculum_rlvr.jsonl") as f:
        for line in f:
            new_problems.append(json.loads(line))
    
    print(f"Loaded {len(new_problems)} new curriculum problems")
    
    # Merge - existing first (they're well-tested), then new
    merged = existing + new_problems
    
    # Re-sort by tier
    merged.sort(key=lambda x: (x.get("tier", 2), x.get("category", ""), x.get("difficulty", "")))
    
    # Reassign IDs
    for i, p in enumerate(merged):
        p["id"] = f"win_{p['tier']}_{i:04d}"
    
    # Save merged RLVR dataset
    output_rlvr = curriculum_dir / "windows_systems_full_rlvr.jsonl"
    with open(output_rlvr, "w") as f:
        for p in merged:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(merged)} problems to {output_rlvr}")
    
    # Generate SFT version
    output_sft = curriculum_dir / "windows_systems_full_sft.jsonl"
    with open(output_sft, "w") as f:
        for p in merged:
            sft = {
                "text": f"""<|im_start|>system
You are an expert Windows systems programmer.
<|im_end|>
<|im_start|>user
{p['prompt']}
<|im_end|>
<|im_start|>assistant
{p['solution']}
<|im_end|>""",
                "metadata": {
                    "source": "windows_systems_full",
                    "tier": p["tier"],
                    "category": p["category"],
                    "difficulty": p["difficulty"]
                }
            }
            f.write(json.dumps(sft) + "\n")
    print(f"Wrote {len(merged)} SFT examples to {output_sft}")
    
    # Generate curriculum order
    curriculum = {
        "total": len(merged),
        "tiers": {}
    }
    for p in merged:
        tier = f"tier_{p['tier']}"
        if tier not in curriculum["tiers"]:
            curriculum["tiers"][tier] = {"count": 0, "categories": {}}
        curriculum["tiers"][tier]["count"] += 1
        cat = p["category"]
        if cat not in curriculum["tiers"][tier]["categories"]:
            curriculum["tiers"][tier]["categories"][cat] = 0
        curriculum["tiers"][tier]["categories"][cat] += 1
    
    with open(curriculum_dir / "curriculum_order_full.json", "w") as f:
        json.dump(curriculum, f, indent=2)
    
    # Stats
    print(f"\n=== MERGED DATASET STATS ===")
    print(f"Total: {len(merged)}")
    
    tier_counts = {}
    for p in merged:
        tier_counts[p["tier"]] = tier_counts.get(p["tier"], 0) + 1
    print("\nBy Tier:")
    for t in sorted(tier_counts.keys()):
        print(f"  Tier {t}: {tier_counts[t]}")
    
    cat_counts = {}
    for p in merged:
        cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1
    print("\nBy Category:")
    for c, n in sorted(cat_counts.items()):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()


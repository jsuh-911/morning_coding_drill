"""Write the synthetic biomarker dataset to disk.

The drill itself generates data in memory via morning_drill.generators;
this script exists for inspecting the dataset outside Python (SQL, CLI drills).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from morning_drill.generators import generate_biomarker_dataset


def main():
    output_dir = ROOT / "datasets" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_biomarker_dataset()
    output = output_dir / "biomarker_study.csv"
    df.to_csv(output, index=False)

    print(f"Created {output}")
    print(f"Rows: {len(df):,}")


if __name__ == "__main__":
    main()

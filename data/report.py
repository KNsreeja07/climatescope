import pandas as pd
from pathlib import Path

INPUT = r"data\GlobalWeatherRepository.csv"
REPORT = Path("reports/initial_EDA_report.md")

def main():
    df = pd.read_csv(INPUT, low_memory=False)

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("# Initial EDA Report — Global Weather Repository\n\n")
        f.write(f"## Shape\n- Rows: {df.shape[0]}\n- Columns: {df.shape[1]}\n\n")
        f.write("## Columns & Types\n")
        f.write(str(df.dtypes) + "\n\n")
        f.write("## Missingness (top 10)\n")
        f.write(str(df.isna().mean().sort_values(ascending=False).head(10)) + "\n")

    print("EDA report written to", REPORT)

if __name__ == "__main__":
    main()

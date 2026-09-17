"""
W2D3 - Pandas Series Practice
Creates pandas Series objects and practices indexing, filtering,
and value access.
"""

import pandas as pd


def main():
    # --- Series from a list (default integer index) ---
    scores = pd.Series([88, 92, 79, 95, 67])
    print("Series from a list:\n", scores)

    # --- Series with a custom index (like a labeled dictionary) ---
    city_temps = pd.Series(
        [72, 68, 85, 90],
        index=["Columbus", "Chicago", "Austin", "Phoenix"],
    )
    print("\nSeries with custom index:\n", city_temps)

    # --- Indexing ---
    print("\nAccess by position (scores[0]):", scores.iloc[0])
    print("Access by label (city_temps['Austin']):", city_temps["Austin"])
    print("Slice by position (scores[1:3]):\n", scores.iloc[1:3])

    # --- Filtering ---
    print("\nCities above 75 degrees:\n", city_temps[city_temps > 75])
    print("\nScores >= 80:\n", scores[scores >= 80])

    # --- Value access / summary ---
    print("\nAll values:", city_temps.values)
    print("All index labels:", list(city_temps.index))
    print("Mean temperature:", city_temps.mean())
    print("Max score:", scores.max())


if __name__ == "__main__":
    main()

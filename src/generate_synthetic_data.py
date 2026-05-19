import os
import numpy as np
import pandas as pd


def generate_air_medical_data(output_path: str, seed: int = 42):
    np.random.seed(seed)

    dates = pd.date_range(start="2022-01-01", end="2025-12-31", freq="D")

    regions = [
        "Mountain West",
        "Midwest",
        "Southeast",
        "Northeast",
        "Southwest",
    ]

    rows = []

    for date in dates:
        for region in regions:
            day_of_week = date.dayofweek
            month = date.month
            weekend_flag = 1 if day_of_week >= 5 else 0

            seasonal_effect = 2 * np.sin((month / 12) * 2 * np.pi)

            holiday_flag = 1 if (
                (date.month == 7 and date.day == 4)
                or (date.month == 12 and date.day in [24, 25, 31])
                or (date.month == 1 and date.day == 1)
            ) else 0

            rurality_index = {
                "Mountain West": 0.85,
                "Midwest": 0.65,
                "Southeast": 0.55,
                "Northeast": 0.35,
                "Southwest": 0.75,
            }[region]

            base_temp = {
                "Mountain West": 45,
                "Midwest": 50,
                "Southeast": 68,
                "Northeast": 48,
                "Southwest": 75,
            }[region]

            avg_temperature = base_temp + 20 * np.sin((month / 12) * 2 * np.pi) + np.random.normal(0, 6)
            precipitation = max(0, np.random.normal(0.15, 0.25))
            wind_speed = max(0, np.random.normal(10, 5))
            visibility = max(0.5, np.random.normal(8, 2))

            trauma_index = np.random.normal(5, 1.5)
            hospital_transfer_volume = np.random.poisson(18 + rurality_index * 8)

            weather_risk = (
                precipitation * 3
                + max(0, wind_speed - 15) * 0.15
                + max(0, 5 - visibility) * 0.8
            )

            demand = (
                8
                + hospital_transfer_volume * 0.35
                + trauma_index * 1.2
                + weekend_flag * 2
                + holiday_flag * 4
                + rurality_index * 5
                + seasonal_effect
                - weather_risk * 0.8
                + np.random.normal(0, 2)
            )

            daily_transport_requests = max(0, round(demand))

            rows.append({
                "date": date,
                "region": region,
                "day_of_week": day_of_week,
                "month": month,
                "weekend_flag": weekend_flag,
                "holiday_flag": holiday_flag,
                "avg_temperature": round(avg_temperature, 2),
                "precipitation": round(precipitation, 2),
                "wind_speed": round(wind_speed, 2),
                "visibility": round(visibility, 2),
                "trauma_index": round(trauma_index, 2),
                "hospital_transfer_volume": hospital_transfer_volume,
                "rurality_index": rurality_index,
                "daily_transport_requests": daily_transport_requests,
            })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df):,} rows to {output_path}")


if __name__ == "__main__":
    generate_air_medical_data("data/raw/air_medical_transport_demand.csv")
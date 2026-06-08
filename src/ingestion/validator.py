import pandas as pd
from pydantic import BaseModel, field_validator, ValidationError
from typing import Optional

FAKE_MMSI = {"987654321", "123456789", "000000000"}

class PositionRecord(BaseModel):
    mmsi: str
    msg_type: int
    sentence_type: str
    speed: float
    lon: float
    lat: float
    course: float
    heading: int

    @field_validator("mmsi")
    @classmethod
    def mmsi_valid(cls, v):
        if not v.isdigit() or len(v) != 9:
            raise ValueError(f"Invalid MMSI: {v}")
        return v

    @field_validator("lat")
    @classmethod
    def lat_range(cls, v):
        if not (-90 <= v <= 90):
            raise ValueError(f"Lat out of range: {v}")
        return v

    @field_validator("lon")
    @classmethod
    def lon_range(cls, v):
        if not (-180 <= v <= 180):
            raise ValueError(f"Lon out of range: {v}")
        return v

    @field_validator("speed")
    @classmethod
    def speed_range(cls, v):
        if not (0 <= v <= 102.2):
            raise ValueError(f"Speed out of range: {v}")
        return v

    @field_validator("course")
    @classmethod
    def course_range(cls, v):
        if not (0 <= v <= 360):
            raise ValueError(f"Course out of range: {v}")
        return v

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mmsi"] = df["mmsi"].astype(str).str.zfill(9)

    valid_flags = []
    error_msgs = []

    for _, row in df.iterrows():
        try:
            PositionRecord(
                mmsi=str(row["mmsi"]),
                msg_type=int(row["msg_type"]),
                sentence_type=str(row["sentence_type"]),
                speed=float(row["speed"]) if pd.notna(row["speed"]) else 0.0,
                lon=float(row["lon"]) if pd.notna(row["lon"]) else 0.0,
                lat=float(row["lat"]) if pd.notna(row["lat"]) else 0.0,
                course=float(row["course"]) if pd.notna(row["course"]) else 0.0,
                heading=int(row["heading"]) if pd.notna(row["heading"]) else 511,
            )
            valid_flags.append(True)
            error_msgs.append(None)
        except ValidationError as e:
            valid_flags.append(False)
            error_msgs.append(str(e.errors()[0]["msg"]))

    df["is_valid"] = valid_flags
    df["validation_error"] = error_msgs
    df["flag_fake_mmsi"] = df["mmsi"].isin(FAKE_MMSI)

    return df

if __name__ == "__main__":
    from loader import load_raw
    df = load_raw()
    pos_df = df[df["msg_type"].isin([1, 3])].copy()
    result = validate_dataframe(pos_df)
    print(f"Valid rows    : {result['is_valid'].sum():,}")
    print(f"Invalid rows  : {(~result['is_valid']).sum():,}")
    print(f"Fake MMSI rows: {result['flag_fake_mmsi'].sum():,}")
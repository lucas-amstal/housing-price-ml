from pydantic import BaseModel, Field


class HousingFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group (tens of thousands of USD)")
    HouseAge: float = Field(..., description="Median house age in block group (years)")
    AveRooms: float = Field(..., description="Average number of rooms per household")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average number of household members")
    Latitude: float
    Longitude: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "MedInc": 8.3,
                "HouseAge": 41,
                "AveRooms": 6.98,
                "AveBedrms": 1.02,
                "Population": 322,
                "AveOccup": 2.55,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_median_house_value_100k: float

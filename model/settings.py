from pydantic import BaseModel
from zoneinfo import ZoneInfo
from datetime import datetime, time
from typing import List


class Settings(BaseModel):

    mag_seven: List[str] = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
    onnx_model_name: str = "mag_seven_onnxmodel"
    skmodel_name: str = "mag_seven_skmodel"
    mlflow_experiment_name: str = "magnificent_7_experiment"
    
    ny_tz: ZoneInfo = ZoneInfo("America/New_York")
    start_date: datetime = datetime(2022,1,1, tzinfo=ny_tz)
    end_date: datetime = datetime.now(ZoneInfo("America/New_York")).replace(hour=0, minute=0, second=0, microsecond=0)
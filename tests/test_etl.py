import pandas as pd
from src.etl import clean_orders


def test_clean_orders():
    data = {
        "olist_orders_dataset": pd.DataFrame({
            "order_id": ["o1"],
            "customer_id": ["c1"],
            "order_status": ["delivered"],
            "order_purchase_timestamp": ["2018-01-01 10:00:00"],
            "order_delivered_customer_date": ["2018-01-10 10:00:00"],
            "order_estimated_delivery_date": ["2018-01-08 10:00:00"],
        })
    }

    result = clean_orders(data)

    assert "delivery_delay" in result.columns
    assert result["delivery_delay"].iloc[0] == 2
    assert pd.api.types.is_datetime64_any_dtype(result["order_purchase_timestamp"])

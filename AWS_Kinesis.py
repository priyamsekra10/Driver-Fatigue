try:
    import re
    import os
    import json
    import boto3
    import datetime
    import uuid
    from datetime import datetime
    import json
    from faker import Faker
    import random
    import faker
except Exception as e:
    print("Error : {} ".format(e))
 
 
def get_referrer():
 
    x = random.randint(1, 5)
    x = x * 50
    y = x + 30
    data = {}
    data["user_id"] = random.randint(x, y)
    data["device_id"] = random.choice(
        ["mobile", "computer", "tablet", "mobile", "computer"]
    )
    data["client_event"] = random.choice(
        [
            "beer_vitrine_nav",
            "beer_checkout",
            "beer_product_detail",
            "beer_products",
            "beer_selection",
            "beer_cart",
        ]
    )
    now = datetime.now()
    str_now = now.isoformat()
    data["client_timestamp"] = str_now
    return data
 
 
def main():
 
    AWS_ACCESS_KEY = "XXX"
    AWS_SECRET_KEY = "XXXX+XXXXXX"
    AWS_REGION_NAME = "us-east-1"
 
    for i in range(1, 120):
        data = get_referrer()
        print(data)
 
        client = boto3.client(
            "firehose",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION_NAME,
        )
 
        response = client.put_record(
            DeliveryStreamName='elastic-search-delivery-streams',
            Record={
                'Data': json.dumps(data)
            }
        )
 
        print(response)
 
 
main()
 
"""
References :
https://aws.amazon.com/blogs/big-data/create-real-time-clickstream-sessions-and-run-analytics-with-amazon-kinesis-data-analytics-aws-glue-and-amazon-athena/
customer_id=!{partitionKeyFromQuery:customer_id}/
"""
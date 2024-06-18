import boto3
import datetime
from dotenv import load_dotenv
import os

load_dotenv()


def update_driver_behavior(driver_id, behavior_data):
    # Initialize a session using Amazon DynamoDB with credentials
    dynamodb = boto3.resource(
        'dynamodb',
        aws_access_key_id=os.getenv('ACCESS_KEY_ID'),
        aws_secret_access_key= os.getenv('SECRET_ACCESS_KEY'),
        region_name='ap-south-1'
    )

    # Select your DynamoDB table
    table = dynamodb.Table('DriverBehavior')

    # Generate a timestamp
    timestamp = datetime.datetime.utcnow().isoformat()

    # Insert data into the table
    response = table.put_item(
       Item={
            'DriverId': driver_id,
            'Timestamp': timestamp,
            'BehaviorData': behavior_data
        }
    )


driver_id = "123pri"  
update_driver_behavior(driver_id, "Drowsy")



#     return response

# # Example usage
# if __name__ == "__main__":
#     driver_id = 'Priyam_Sekra'
#     behavior_data = 'Drink and drive'
#     response = update_driver_behavior(driver_id, behavior_data)
#     print("Data inserted successfully:", response)

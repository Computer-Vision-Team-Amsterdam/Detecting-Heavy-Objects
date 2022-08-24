import pandas as pd
import psycopg2
from psycopg2 import Error

connection = None
try:
    # Connect to an existing database
    connection = psycopg2.connect(user="",
                                  password="",
                                  host="",
                                  port="5432",
                                  database="postprocessing-output")

    # Create a cursor to perform database operations
    cursor = connection.cursor()
    # Print PostgreSQL details
    print("PostgreSQL server information")

    # Executing a SQL query
    with open("prioritized_objects.csv", "r") as f:
        next(f)  # Skip the header row.
        cursor.copy_from(f, 'containers', sep=';')

    connection.commit()

except (Exception, Error) as error:
    print("Error while connecting to PostgreSQL", error)
finally:
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")

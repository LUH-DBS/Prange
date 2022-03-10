import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()
db_params = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
}

with psycopg2.connect(**db_params) as connection:

    cursor = connection.cursor()

    query = "SELECT * FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema'"

    outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)

    with open('./test.csv', 'w') as outfile:
        cursor.copy_expert(outputquery, outfile)

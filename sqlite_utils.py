import sqlite3
import time

from concurrent.futures import ThreadPoolExecutor


def create_extended_table(m, n, db_name="test_intersections.db", new_column_low=0, new_column_high=100):
    """
    Extend the dimension of an existing table by appending one column and creating a new table.
    """
    import sqlite3
    import random

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Define table names
    old_table_name = f"intersections_m{m}_n{n-1}"
    new_table_name = f"intersections_m{m}_n{n}"

    # Check if the old table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (old_table_name,))
    if not cursor.fetchone():
        print(f"Table {old_table_name} does not exist. Cannot create {new_table_name}.")
        conn.close()
        return

    # Fetch records from the old table
    cursor.execute(f"SELECT * FROM {old_table_name}")
    records = cursor.fetchall()
    num_columns = len(records[0])  # Including ID and constant

    # Append a new column to each record
    extended_records = [
        (*record[:-1], random.randint(new_column_low, new_column_high) - random.randint(new_column_low, new_column_high), record[-1])
        for record in records
    ]

    # Define new table schema
    new_columns = ", ".join([f"c{i+1} INTEGER" for i in range(num_columns - 2 + 1)])  # Adding one more column
    cursor.execute(f"""
        CREATE TABLE {new_table_name} (
            id INTEGER PRIMARY KEY,
            {new_columns},
            constant INTEGER
        )
    """)

    # Insert extended records into the new table
    cursor.executemany(
        f"INSERT INTO {new_table_name} VALUES ({', '.join(['?' for _ in range(num_columns + 1)])})",
        extended_records
    )

    # Create index on ID for faster queries
    cursor.execute(f"CREATE INDEX idx_{new_table_name}_id ON {new_table_name}(id)")

    conn.commit()
    conn.close()
    print(f"New table {new_table_name} created successfully.")



def save_functions_to_sqlite(records, m, n, db_name="test_intersections.db",  table_name = "", constant_low=-150, constant_high=-50):
    """
    Save records to an SQLite database in a table named dynamically based on m and n.
    If the table exists, update the constants.
    """
    import sqlite3
    import random

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Define the dynamic table name
    # table_name = f"intersections_m{m}_n{n}"
    table_name = f"{table_name}_m{m}_n{n}"
    print(f"Table name: {table_name}")

    # Check if the table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    table_exists = cursor.fetchone()

    if table_exists:
        print(f"Table {table_name} exists. Updating constants.")
        # Update constants for existing records
        cursor.execute(f"SELECT id FROM {table_name}")
        record_ids = [row[0] for row in cursor.fetchall()]

        for record_id in record_ids:
            new_constant = random.randint(constant_low, constant_high) - random.randint(constant_low, constant_high)
            cursor.execute(f"UPDATE {table_name} SET constant = ? WHERE id = ?", (new_constant, record_id))

    else:
        print(f"Table {table_name} does not exist. Creating a new table.")
        # Create table
        num_coefficients = len(records[0]) - 2  # Number of coefficients in each record
        columns = ", ".join([f"c{i+1} INTEGER" for i in range(num_coefficients)])
        cursor.execute(f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY,
                {columns},
                constant float
            )
        """)

        # Insert records
        cursor.executemany(
            f"INSERT INTO {table_name} VALUES ({', '.join(['?' for _ in range(num_coefficients + 2)])})",
            records
        )

        # Create index on ID for faster queries
        cursor.execute(f"CREATE INDEX idx_{table_name}_id ON {table_name}(id)")

    conn.commit()
    conn.close()



def read_from_sqlite(m, n, db_name="test_intersections.db", record_id=None, conn=None):
    """
    Read records from a dynamically named SQLite table based on m and n.
    Optionally filter by ID. Use an existing connection if provided.
    Returns a single record (without the index) as a tuple or a list of tuples.
    """
    # Use the provided connection, or create a new one
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(db_name)
        close_conn = True

    cursor = conn.cursor()
    table_name = f"intersections_m{m}_n{n}"

    # Query records
    if record_id is not None:
        cursor.execute(f"SELECT * FROM {table_name} WHERE id = ?", (record_id,))
        result = cursor.fetchone()  # Fetch a single record
        result = tuple(result[1:]) if result else None  # Skip the index (position 0)
    else:
        cursor.execute(f"SELECT * FROM {table_name}")
        result = [tuple(row[1:]) for row in cursor.fetchall()]  # Skip the index for all rows

    # Close the connection if it was created in this function
    if close_conn:
        conn.close()

    return result



def get_all_ids(m, n, db_name="test_intersections.db"):
    """
    Fetch all IDs from the specified table.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    table_name = f"intersections_m{m}_n{n}"
    try:
        cursor.execute(f"SELECT id FROM {table_name}")
        ids = [row[0] for row in cursor.fetchall()]
        return ids
    except sqlite3.OperationalError as e:
        print(f"Error fetching IDs from table {table_name}: {e}")
        return []
    finally:
        conn.close()


class SQLiteReader:
    """
    Class to read records from SQLite and store them in a class variable.
    """
    records = []  # Class variable to store all fetched records
    init_constraint = []
    time_get_record_by_id = 0.0

    # @classmethod
    # def read_all_from_sqlite(cls, m, n, db_name="test_intersections.db", conn=None):
    #     """
    #     Fetch all records from the table and save them to the class variable.
    #     Skips the index column.
    #     """
    #     close_conn = False
    #     if conn is None:
    #         conn = sqlite3.connect(db_name)
    #         close_conn = True
    #
    #     cursor = conn.cursor()
    #     table_name = f"intersections_m{m}_n{n}"
    #
    #     try:
    #         cursor.execute(f"SELECT * FROM {table_name}")
    #         cls.records = [tuple(row[1:]) for row in cursor.fetchall()]  # Skip index column
    #         print(f"Records loaded from table {table_name}.")
    #     except sqlite3.OperationalError as e:
    #         print(f"Error reading table {table_name}: {e}")
    #         cls.records = []  # Reset records if there's an error
    #     finally:
    #         if close_conn:
    #             conn.close()

    @classmethod
    def read_all_from_sqlite(cls, m, n, size=None, db_name="test_intersections.db", conn=None):
        """
        Fetch a limited number of records from the table and save them to the class variable.
        Skips the index column.

        :param m: Integer, first parameter defining the table name.
        :param n: Integer, second parameter defining the table name.
        :param size: Integer, number of records to fetch. If None, fetch all.
        :param db_name: String, name of the database.
        :param conn: SQLite connection object. If None, a new connection is created.
        """
        close_conn = False
        if conn is None:
            conn = sqlite3.connect(db_name)
            close_conn = True

        cursor = conn.cursor()
        table_name = f"intersections_m{m}_n{n}"
        print(f"Reading records from table {table_name}.")

        try:
            query = f"SELECT * FROM {table_name} LIMIT ?" if size else f"SELECT * FROM {table_name}"
            cursor.execute(query, (size,)) if size else cursor.execute(query)

            cls.records = [tuple(row[1:]) for row in cursor.fetchall()]  # Skip index column
            print(f"Loaded {len(cls.records)} records from table {table_name}.")
        except sqlite3.OperationalError as e:
            print(f"Error reading table {table_name}: {e}")
            cls.records = []  # Reset records if there's an error
        finally:
            if close_conn:
                conn.close()

    @classmethod
    def get_records(cls):
        """
        Return the records stored in the class variable.
        """
        return cls.records

    @classmethod
    def get_record_by_id(cls, record_id):
        """
        Retrieve a record by ID (1-based index).
        Returns None if the ID is out of range.
        """
        start_time = time.time()
        if not cls.records:
            print("No records loaded. Call read_all_from_sqlite first.")
            cls.time_get_record_by_id += time.time() - start_time
            return None

        try:
            # SQLite IDs usually start from 1, so subtract 1 for list indexing
            cls.time_get_record_by_id += time.time() - start_time
            return cls.records[record_id - 1]
        except IndexError:
            print(f"Record with ID {record_id} does not exist.")
            return None

    # @classmethod
    # def find_matching_records(cls, records):
    #     """
    #     Finds matching records in total_records for unique combinations of the first element in the input list.
    #
    #     Parameters:
    #         records (list): A list of records to process, e.g., [1, 2, 3, 4, 5].
    #         total_records (list): A list of tuples representing the total records.
    #
    #     Returns:
    #         list: Matching records from total_records.
    #     """
    #     from itertools import combinations
    #
    #     # Step 1: Compute unique combinations for the first element
    #     first_element = records[0]
    #     combinations_list = [(first_element, other) for other in records[1:]]
    #
    #     # Step 2: Find matches in total_records
    #     matching_records_id = []
    #     for comb in combinations_list:
    #         for idx, total_record in enumerate(cls.records):
    #             # Check if the last two elements of total_record match the current combination
    #             if (total_record[-2:] == comb) or (total_record[-2:] == comb[::-1]):
    #                 matching_records_id.append(idx + 1)
    #
    #     return matching_records_id

    @classmethod
    def find_matching_records(cls, records):
        # print(records)
        """
        Finds matching records in total_records for unique combinations of the first element in the input list.

        Parameters:
            records (list): A list of records to process, e.g., [1, 2, 3, 4, 5].

        Returns:
            list: Matching record indices from total_records.
        """
        # Step 1: Compute unique combinations for the first element
        first_element = records[0]
        combinations_list = [(first_element, other) for other in records[1:]]

        # Step 2: Function to check for matches in a chunk of records
        def find_matches(start, end):
            matching_indices = []
            for idx in range(start, end):
                total_record = cls.records[idx]
                if (total_record[-2:] in combinations_list) or (total_record[-2:] in [t[::-1] for t in combinations_list]):
                    matching_indices.append(idx + 1)
            return matching_indices

        # Multi-threaded execution
        num_threads = min(16, len(cls.records))  # Use up to 8 threads or limit based on records
        chunk_size = (len(cls.records) + num_threads - 1) // num_threads
        tasks = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(0, len(cls.records), chunk_size):
                tasks.append(executor.submit(find_matches, i, min(i + chunk_size, len(cls.records))))

            # Collect results
            matching_records_id = []
            for future in tasks:
                matching_records_id.extend(future.result())

        return matching_records_id


def save_intersections_to_sqlite(records, m, n, db_name="test_intersections.db", table_name="", constant_low=0, constant_high=0):
    """
    Save records with the format (id, c1, c2, ..., fi, fj) to an SQLite database.
    The table schema and columns are determined dynamically based on the first record.
    """
    import sqlite3

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Define the dynamic table name
    table_name = f"{table_name}_m{m}_n{n}"
    print(f"Table name: {table_name}")

    # Extract the number of coefficient columns dynamically
    num_coefficients = len(records[0]) - 4  # Subtract `id`, `fi`, and `fj`
    coefficient_columns = ", ".join([f"c{i+1} INTEGER" for i in range(num_coefficients)])

    # Check if the table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    table_exists = cursor.fetchone()

    if not table_exists:
        print(f"Table {table_name} does not exist. Creating a new table.")

        # Create table with dynamic schema
        cursor.execute(f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY,
                {coefficient_columns},
                constant INTEGER,
                fi INTEGER,
                fj INTEGER
            )
        """)

    else:
        print(f"Table {table_name} exists. Inserting new records.")

    # Prepare the insert query dynamically
    placeholders = ", ".join(["?" for _ in range(len(records[0]))])  # Match the number of columns
    insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"

    # Insert all records
    cursor.executemany(insert_query, records)

    # Create an index for `fi` and `fj` columns (optional but useful for querying)
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_fi_fj ON {table_name}(fi, fj)")

    conn.commit()
    conn.close()
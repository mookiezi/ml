import psycopg2
import os
from datetime import datetime

# --- Configuration ---
user = 'postgres',
host = 'localhost',
database = 'maniac',
password = 'Km38!xgWZr$Bq7LJ2@tuEvn*pfDcsY5#oAhMRNw^Vz',

def fetch_messages_with_timestamps(guild_id=None, author_id=None):
    """
    Fetches message content and timestamps from the database,
    optionally filtered by guild and author, ordered by time.
    Returns a list of tuples: (timestamp, author_id, content).
    """
    conn = None
    try:
        conn = psycopg2.connect(host=host, database=database,
                                user=user, password=password)
        cur = conn.cursor()

        query = "SELECT created_at, author_id, content FROM messages WHERE guild_id = %s"
        conditions = []
        params = [guild_id]  # guild_id is mandatory

        if author_id:
            query += " AND author_id = %s"
            params.append(author_id)

        query += " ORDER BY created_at ASC"  # Order by time for sequential analysis

        cur.execute(query, params)
        messages = cur.fetchall()
        return messages

    except psycopg2.Error as e:
        print(f"Error fetching messages: {e}")
        return []
    finally:
        if conn:
            conn.close()


def analyze_user_timing(messages):
    """
    Analyzes the time difference between consecutive messages from different users.
    Returns a list of time differences (in seconds).
    """
    timing_differences = []
    previous_message_time = None
    previous_author_id = None

    for timestamp, author_id, content in messages:
        if previous_message_time and author_id != previous_author_id:
            time_difference = (timestamp - previous_message_time).total_seconds()
            timing_differences.append(time_difference)

        previous_message_time = timestamp
        previous_author_id = author_id

    return timing_differences


def write_timing_to_log(timing_data, log_file):
    """
    Writes the analyzed timing data to a text log file.
    """
    with open(log_file, 'w') as f:
        f.write("Observed User-to-User Message Timings (in seconds):\n")
        for timing in timing_data:
            f.write(f"{timing:.4f}\n")
        f.write(f"\nTotal Timings Observed: {len(timing_data)}\n")


if __name__ == "__main__":
    # Get log file path from user
    log_file_path = input(
        "Enter the desired path for the log file (e.g., 'timings.txt' or 'data/user_timings.log'): ")

    # Ensure .txt extension
    if not log_file_path.lower().endswith(".txt"):
        log_file_path = os.path.splitext(log_file_path)[0] + ".txt"  # Add .txt if missing

    # Get guild_id from user
    guild_id = input("Enter the guild ID: ")

    # Get author_id from user
    author_id = input("Enter the author ID (optional, press Enter to skip): ")

    messages = fetch_messages_with_timestamps(guild_id=guild_id, author_id=author_id)

    if messages:
        user_timings = analyze_user_timing(messages)
        write_timing_to_log(user_timings, log_file_path)
        print(f"User-to-user message timings have been logged to '{log_file_path}'")
    else:
        print("No messages found in the database.")

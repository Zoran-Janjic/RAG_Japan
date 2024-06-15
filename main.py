from appUI import create_ui
from dotenv import load_dotenv
import os


def main():
    # Load the .env file
    load_dotenv()
    db_host = os.getenv("OPENAI_API_KEY")
    print(f"OPENAI_API_KEY: {db_host}")
    # create_ui()


if __name__ == "__main__":
    main()
from appUI import create_ui
from dotenv import load_dotenv


def main():
    # Load the .env file
    load_dotenv()
    create_ui()


if __name__ == "__main__":
    main()

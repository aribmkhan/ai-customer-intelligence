import os

from anthropic import Anthropic
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


def test_claude_connection():
    """Test the connection to the Claude API."""

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY was not found. Check your .env file."
        )

    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": (
                    "Reply with exactly: "
                    "Claude API connection successful!"
                ),
            }
        ],
    )

    print(response.content[0].text)


if __name__ == "__main__":
    test_claude_connection()
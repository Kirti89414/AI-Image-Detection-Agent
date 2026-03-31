from dotenv import load_dotenv
import os
import base64
from groq import Groq

load_dotenv()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_mime_type(image_path):
    ext = image_path.lower().split(".")[-1]

    if ext in ["jpg", "jpeg"]:
        return "image/jpeg"
    elif ext == "png":
        return "image/png"
    elif ext == "webp":
        return "image/webp"
    else:
        raise ValueError("Unsupported image format. Use jpg, jpeg, png, or webp.")


def ask_image_agent(image_path, question):
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type = get_mime_type(image_path)
    base64_image = encode_image(image_path)

    client = Groq(api_key=groq_api_key)

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert image recognition AI agent. "
                    "Analyze the image carefully and answer accurately. "
                    "If the image is unclear, say so honestly."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def main():
    image_path = input("Enter image path: ").strip().strip('"')

    print("\nImage Agent is ready!")
    print("Ask questions about the image.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        try:
            answer = ask_image_agent(image_path, question)
            print("\nAnswer:")
            print(answer)
            print("-" * 80)
        except Exception as e:
            print(f"\nError: {e}")
            print("-" * 80)


if __name__ == "__main__":
    main()
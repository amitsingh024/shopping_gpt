from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_response(user_query, products, system_prompt):

    context = products.to_string(index=False)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""
                User Query:
                {user_query}

                Product Data:
                {context}
                """
            }
        ],
        temperature=0.4
    )

    return response.choices[0].message.content
"""
This notebook was used to add the questions colummn for the corresponding movie quote in my dataset using the groq api, the dataset finally had around 360 samples
"""

from dotenv import load_dotenv
load_dotenv()
from langchain.chains import LLMChain
from pydantic import BaseModel
from typing import List ,Dict
import instructor
from groq import Groq
import json
class ResponseModel(BaseModel):
    question: str


client = instructor.from_groq(Groq(api_key="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")
, mode=instructor.Mode.JSON)
output_schema="""{"question":"question?"}"""
def make_question(text):
  messages = [
          {
    "role": "system",
    "content": f"""
    Please analyze the given list of movie quotes: {text}
    You are a creative and precise assistant whose task is to generate a well-formed question
    for each quote such that the quote itself would be the correct answer to that question.
    The question should be natural, contextually accurate, and clearly lead to the given quote as the response.
    Return the output strictly following this JSON format: {output_schema}
    """
}
      ]
  response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=ResponseModel,
        messages=messages,
        temperature=0.9,
        max_tokens=1000,
    )

  return json.loads(response.json())["question"]

l=make_question("It's only by facing our fears that we can discover our true potential.")
print(l)

import pandas as pd


df = pd.read_csv("movie_quotes.csv")

# Ensure the 'quote' column exists
if 'quote' not in df.columns:
    raise ValueError("Expected a column named 'quote' in the CSV file.")


def safe_make_question(quote):
    try:
        return make_question(quote)
    except Exception as e:
        return f"Error: {e}"

df['question'] = df['quote'].apply(safe_make_question)

output_file = "movie_quotes_with_questions.csv"
df.to_csv(output_file, index=False)

print(f"Successfully saved new file: {output_file}")


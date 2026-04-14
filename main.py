from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

api_key = os.getenv("API_KEY")

client = OpenAI(api_key = api_key,base_url="https://api.groq.com/openai/v1")



app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask",methods = ["POST"])
def answer():
    question = request.form.get("question")

    answer = client.responses.create(
        input = [
            {"role":"system" , "content": "act as a helpful ai assistant"},
            {"role":"user","content":question}
        ],
        model = "openai/gpt-oss-20b",
        temperature = 0.7,
        max_output_tokens = 512

    )

    answer = answer.output_text.strip()

    return jsonify({"response":answer}), 200

@app.route("/summarize",methods = ["POST"])
def summarize():
    email_text = request.form.get("email")
    prompt = f"summarize the following email in 2-3 sentences: {email_text}"

    response = client.responses.create(
        model = "openai/gpt-oss-20b",
        input = [
            {"role":"system","content":"Act like an expert email assistant"},
            {"role":"user","content":prompt}
        ],
        temperature = 0.3,
        max_output_tokens = 512
    )

    summary = response.output_text.strip()

    return jsonify({"response":summary})

if __name__ == "__main__":
    app.run(debug = True)
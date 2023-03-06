import os


import openai
import json


class VeriQA:
    def __init__(self, model="gpt-3.5-turbo", veri_description="veri.json"):
        try:
            self.openai_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise Exception("Please set the OPENAI_API_KEY environment variable")

        self.model = model  # 0.002$ per 1000 tokens

        if not os.path.exists(veri_description):
            raise Exception("Please download veri.json from Veri")
        veri_info = json.load(open(veri_description))
        text = "\n".join([qa["question"] + "\n" + qa["answer"] for qa in veri_info])
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(model)
            print("Total:", len(enc.encode(text)), "tokens")
        except ImportError:
            print("Please install tiktoken to check the number of tokens")
        self.veri = text

    def __call__(self, question, log=False):
        question = question.strip()
        if not question.endswith("?"):
            question += "?"

        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are VeriBot, a large language model trained by OpenAI. "
                    "Answer as concisely and succinctly in two to three sentences.",
                },
                {
                    "role": "user",
                    "content": f'Given this document """{self.veri}""", {question} Answer as concisely and succinctly in two to three sentences.',
                },
            ],
            temperature=0.2,
            max_tokens=650,
        )

        answer = [
            m["message"]["content"]
            for m in completion["choices"]
            if m["message"]["role"] == "assistant"
        ][0]

        if log:
            print("Question:", question)
            print("Answer  :", answer)
        return answer


bot = VeriQA()
print(bot("What is metabolic health?"))

import os
from random import choice

from locust import FastHttpUser, task, between


QUESTIONS = [
    "Tôi bị sốt và ho khan thì nên làm gì?",
    "Triệu chứng đau đầu, chóng mặt, mệt mỏi có thể là bệnh gì?",
    "Tôi bị đau bụng và buồn nôn sau khi ăn, nguyên nhân thường gặp là gì?",
    "Làm sao để phân biệt cảm cúm và cảm lạnh?",
    "Tôi có các nốt đỏ trên da và ngứa, nên theo dõi những gì?",
]


class ChatApiUser(FastHttpUser):
    host = os.getenv("LOCUST_HOST", "http://localhost:3400")
    wait_time = between(1, 3)

    def on_start(self):
        self.chat_history = []

    @task
    def chat(self):
        payload = {
            "question": choice(QUESTIONS),
            "chat_history": self.chat_history,
        }

        with self.client.post("/chat", json=payload, catch_response=True, name="/chat") as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status code: {response.status_code}")
                return

            try:
                data = response.json()
            except ValueError:
                response.failure("Response is not valid JSON")
                return

            answer = data.get("answer")
            chat_history = data.get("chat_history")

            if not isinstance(answer, str) or not answer.strip():
                response.failure("Missing or empty answer in response")
                return

            if not isinstance(chat_history, list):
                response.failure("chat_history is not a list")
                return

            self.chat_history = chat_history
            response.success()
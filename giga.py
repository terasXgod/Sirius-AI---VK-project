from dotenv import load_dotenv
import os

from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models.gigachat import GigaChat

load_dotenv()

api_key = os.getenv('api_key')



if __name__ == "__main__":
    giga = GigaChat(credentials=api_key, verify_ssl_certs=False, streaming=True)
    message = [
        SystemMessage(
            content=f"мне необходимо суммаризировать новость, помоги мне в этом!\n"
        )
    ]
    message.append(HumanMessage(content=''))
    res = giga(message)
    message.append(res)
    # Ответ сервиса
    print("Bot: ", res.content)
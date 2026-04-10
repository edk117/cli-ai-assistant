import json
import os
import time

from anthropic import Anthropic
from anthropic._exceptions import (
    APIConnectionError as AnthropicConnectionError,
    APIStatusError as AnthropicStatusError,
    APITimeoutError as AnthropicTimeoutError,
)
from dotenv import load_dotenv
from openai import OpenAI
from openai._exceptions import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

# Загрузка конфигурации
load_dotenv()

HISTORY_FILE = "history.json"
SYSTEM_PROMPT = "Ты полезный ассистент."

anthropic_client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://api.proxyapi.ru/anthropic",
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.proxyapi.ru/openai/v1",
)

# История диалога: сохранение и загрузка
def load_history() -> list:
    """Загружает историю из JSON-файла."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_history(history: list) -> None:
    """Сохраняет историю в JSON-файл."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def clear_history_file() -> None:
    """Удаляет файл истории."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)


# Безопасный запрос с повтором
MAX_RETRIES = 3
RETRY_DELAY = 5


def safe_request(func, *args, **kwargs):
    """Вызывает API с повтором при сетевых ошибках и таймаутах."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except (APITimeoutError, AnthropicTimeoutError):
            if attempt < MAX_RETRIES:
                print(f"\n⏳ Сервер не отвечает. Попытка {attempt}/{MAX_RETRIES}, повтор через {RETRY_DELAY}с…")
                time.sleep(RETRY_DELAY)
            else:
                print(f"\n❌ Сервер не ответил после {MAX_RETRIES} попыток.\n")
                return None
        except (APIConnectionError, AnthropicConnectionError):
            if attempt < MAX_RETRIES:
                print(f"\n⚠️  Нет соединения. Попытка {attempt}/{MAX_RETRIES}, повтор через {RETRY_DELAY}с…")
                time.sleep(RETRY_DELAY)
            else:
                print(f"\n❌ Проверьте интернет-соединение.\n")
                return None
        except RateLimitError:
            if attempt < MAX_RETRIES:
                print(f"\n🚦 Превышен лимит запросов. Ждём 30с…")
                time.sleep(30)
            else:
                print(f"\n❌ Слишком много запросов, попробуйте позже.\n")
                return None
        except (APIStatusError, AnthropicStatusError) as e:
            if e.status_code == 401:
                print(f"\n🔑 Неверный API-ключ. Проверьте .env.\n")
            elif e.status_code == 404:
                print(f"\n🔍 Модель не найдена. Проверьте название модели.\n")
            else:
                print(f"\n❌ Ошибка сервера ({e.status_code}): {e.message}\n")
            return None
        except Exception as e:
            print(f"\n❌ Неизвестная ошибка: {e}\n")
            return None

    return None


# Обёртка с обработкой ошибок и таймаутами


def chat_thinking(user_message: str, history: list) -> tuple[str, list]:
    """Думающая модель с отображением reasoning."""
    history.append({"role": "user", "content": user_message})

    response = safe_request(
        anthropic_client.messages.create,
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=history,
        thinking={"type": "enabled", "budget_tokens": 1024},
        timeout=60,
    )

    if response is None:
        history.pop()
        return "", history

    # Извлекаем reasoning и ответ
    reasoning_text = ""
    answer_text = ""
    for block in response.content:
        if block.type == "thinking":
            reasoning_text = block.thinking
        elif block.type == "text":
            answer_text = block.text

    # Показываем reasoning и метрики токенов
    if reasoning_text:
        usage = response.usage
        print(f"\n{'='*50}")
        print("🧠 REASONING:")
        print(f"   Входные токены:  {usage.input_tokens}")
        print(f"   Выходные токены: {usage.output_tokens}")
        print(f"   Всего токенов:   {usage.input_tokens + usage.output_tokens}")
        print(f"{'='*50}")
        print(reasoning_text)
        print(f"{'='*50}\n")

    history.append({"role": "assistant", "content": answer_text})
    return answer_text, history


def chat_normal(user_message: str, history: list) -> tuple[str, list]:
    """Обычная модель через OpenAI Chat Completions API."""
    history.append({"role": "user", "content": user_message})

    response = safe_request(
        openai_client.chat.completions.create,
        model="gpt-5.4",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        timeout=60,
    )

    if response is None:
        history.pop()
        return "", history

    assistant_message = response.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_message})
    return assistant_message, history


# Главное меню


if __name__ == "__main__":
    print("=" * 40)
    print("  Chat Assistant — ProxyAPI")
    print("=" * 40)
    print()

    # Загрузка сохранённой истории
    conversation_history = load_history()
    if conversation_history:
        msg_count = len(conversation_history)
        print(f"📂 Загружена история: {msg_count} сообщений")
    else:
        print("🆕 Новый диалог")

    # Выбор режима
    print()
    print("Выберите модель:")
    print("  1 — думающая модель (claude-sonnet-4-5)")
    print("  2 — обычная модель (gpt-5.4)")
    print()

    choice = input("Ваш выбор (1 или 2): ").strip()
    mode = "thinking" if choice == "1" else "normal"

    model_name = "claude-sonnet-4-5 🧠" if mode == "thinking" else "gpt-5.4"
    print(f"\n✅ Режим: {model_name}")
    print("┌─────────────────────────────────────────────────┐")
    print("│ Команды:                                        │")
    print("│   exit / quit  — выйти и сохранить историю      │")
    print("│   clear        — очистить историю              │")
    print("│   Enter пустой — пропуск                       │")
    print("└─────────────────────────────────────────────────┘\n")

    while True:
        user_input = input("Вы: ").strip()

        if user_input.lower() in ("exit", "quit"):
            save_history(conversation_history)
            print(f"\n💾 История сохранена ({len(conversation_history)} сообщений)")
            print("До свидания!")
            break

        if user_input.lower() == "clear":
            conversation_history.clear()
            clear_history_file()
            print("🗑️  История очищена.\n")
            continue

        if not user_input:
            continue

        reply = (
            chat_thinking(user_input, conversation_history)
            if mode == "thinking"
            else chat_normal(user_input, conversation_history)
        )

        answer, _ = reply
        if answer:
            print(f"\nАссистент: {answer}\n")

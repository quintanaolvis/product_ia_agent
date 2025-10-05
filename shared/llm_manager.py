import os
import requests
from typing import Dict, Any, List
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from logging import getLogger
from dotenv import load_dotenv


load_dotenv()

logger = getLogger(__name__)


PROMPT_BASE_SYSTEM = "test"


class LLMManager:
    def __init__(self):
        """
        Inicializa el LLMManager y configura la sesión HTTP con reintentos automáticos.
        Los reintentos aplican para errores de red y códigos de estado 429, 500, 502, 503, 504.
        """
        self.api_key = os.environ["OPEN_ROUTERS_API_KEY"]
        self.api_url = os.environ["OPEN_ROUTERS_API_URL"]
        self.model = os.environ["OPEN_ROUTERS_MODEL"]
        self.prompt_system = PROMPT_BASE_SYSTEM

        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def _build_messages(
        self, prompt: str, session_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Construye la lista de mensajes para enviar al modelo.
        Envia el contexto de la sesión y evita duplicados.
        """
        messages = [{"role": "system", "content": self.prompt_system}]

        session_messages = session_data.get("messages", [])
        last_user_content = None
        for msg in session_messages:
            role = msg["role"]
            content = (msg.get("content") or "").strip()
            if not content:
                continue

            role = "user" if role == "user" else "assistant"

            if role == "user" and content == prompt:
                last_user_content = content
                continue

            messages.append({"role": role, "content": content})
            if role == "user":
                last_user_content = content

        if not last_user_content or last_user_content != prompt:
            messages.append({"role": "user", "content": prompt})

        return messages

    def invoke(self, prompt: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Llama a la API de LLM con el prompt y el historial de la sesión.
        Incluye herramientas (tools) para que el modelo pueda invocar funciones.
        """
        if not prompt.strip():
            raise ValueError("El prompt no puede estar vacío.")

        try:
            messages = self._build_messages(prompt, session_data)

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "temperature": 0.3,
                "tools": [],
                "tool_choice": "auto",
            }

            logger.info(f"Llamando a LLM con {len(messages)}")

            response = self.session.post(self.api_url, json=payload, timeout=30)

            if response.status_code != 200:
                error_msg = f"API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            result = response.json()
            logger.info("Respuesta recibida exitosamente de LLM")
            return result

        except requests.exceptions.Timeout:
            error_msg = "La solicitud a LLM excedió el tiempo de espera."
            logger.error(error_msg)
            raise TimeoutError(error_msg) from None

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Error de conexión con LLM: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from None

        except Exception as e:
            logger.exception("Error inesperado al llamar a LLM")
            raise e


# ---------------------------------------------
# Singleton seguro (thread-safe si es necesario)
# ---------------------------------------------
_llm_manager = None


def get_llm_manager() -> LLMManager:
    """Devuelve una instancia única del LLMManager (singleton)."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager

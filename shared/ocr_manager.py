import ast
import json
import base64

from shared.utils import (
    clean_data,
    download_image_url,
    get_image_mime_type,
    get_prompt,
)
from logging import getLogger
from shared.llm_manager import get_llm_manager

logger = getLogger(__name__)


def get_text_from_image(content, prompt=str):
    """
    Extrae texto de una imagen usando el modelo de OCR y el prompt definido.
    """

    mime_type = get_image_mime_type(content)
    if not mime_type:
        return {
            "success": False,
            "message": "Formato de imagen no soportado. Por favor, envía una imagen JPG o PNG.",
            "ocr_context": "Formato de imagen no soportado",
            "image": {},
            "text": "No se pudo extraer texto",
        }

    try:
        base64_image = base64.b64encode(content).decode("utf-8")
    except Exception as e:
        return {
            "success": False,
            "message": "Error al procesar la imagen. Por favor, verifica el contenido.",
            "ocr_context": "Error al procesar la imagen",
            "image": {},
            "text": "No se pudo extraer texto",
        }

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                },
            ],
        }
    ]

    try:
        session_data = {
            "user_id": "default_user",
            "messages": messages,
        }
        llm_manager = get_llm_manager()
        response_from_llm = llm_manager.invoke(session_data=session_data, prompt=prompt)

        # Extraer el texto del dict antes de limpiar
        if isinstance(response_from_llm, dict):
            # Ajusta la ruta según la estructura real de tu respuesta
            response_text = (
                response_from_llm.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        else:
            response_text = response_from_llm

        cleaned_content = clean_data(response_text)

        try:
            final_response = json.loads(cleaned_content)
        except json.JSONDecodeError:
            final_response = ast.literal_eval(cleaned_content)
        except ValueError as ve:
            return {
                "success": False,
                "message": "Hubo un error al procesar la imagen, por favor intenta de nuevo.",
                "ocr_context": "No se pudo procesar la imagen",
                "image": {},
                "text": "No se pudo extraer texto",
            }

        # Aseguramos que la clave "text" esté presente
        if isinstance(final_response, dict):
            final_response.setdefault(
                "text", final_response.get("ocr_context", "No se pudo extraer texto")
            )
            return final_response
        else:
            return {
                "success": True,
                "message": "OCR procesado correctamente.",
                "ocr_context": "",
                "image": {},
                "text": str(final_response),
            }

    except ConnectionError:
        logger.error("Error de conexión con el modelo")
        return {
            "success": False,
            "message": "No se pudo conectar al servicio. Por favor, intenta de nuevo.",
            "ocr_context": "Error de conexión con el modelo",
            "image": {},
            "text": "No se pudo extraer texto",
        }
    except ValueError as ve:
        logger.error(f"[ERROR_VALUE]: {str(ve)}")
        return {
            "success": False,
            "message": "Hubo un error al procesar la imagen, por favor intenta de nuevo.",
            "ocr_context": "No se pudo procesar la imagen",
            "image": {},
            "text": "No se pudo extraer texto",
        }
    except Exception as e:
        logger.exception(f"Error inesperado al procesar imagen: {str(e)}")
        return {
            "success": False,
            "message": "Hubo un error al procesar la imagen, por favor intenta de nuevo.",
            "ocr_context": "No se pudo procesar la imagen",
            "image": {},
            "text": "No se pudo extraer texto",
        }


def invoke_ocr(media_url):
    """
    Invoca el proceso de OCR para una imagen dada.
    """
    try:
        image_content = download_image_url(media_url)
        mime_type = get_image_mime_type(image_content)
        if not mime_type:
            logger.info(f"Formato de imagen no soportado mime type: {mime_type}")
            return {
                "success": False,
                "message": "Formato de imagen no soportado. Por favor, envía una imagen JPG o PNG.",
                "ocr_context": "Formato de imagen no soportado.",
                "image": {},
            }
        prompt = get_prompt("OCR_PROMPT")
        body = get_text_from_image(image_content, prompt)
        return body
    except Exception as e:
        logger.exception(f"Error procesando imagen para OCR: {str(e)}")
        return {
            "success": False,
            "message": "Hubo un error procesando la imagen, por favor intentalo de nuevo.",
            "ocr_context": "No se pudo procesar la imagen",
            "image": {},
        }

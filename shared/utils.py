import os
import re
import imghdr
import requests

from logging import getLogger

logger = getLogger(__name__)


def download_image_url(url: str) -> bytes:
    """Descarga una imagen desde una URL y retorna su contenido en bytes."""
    response = requests.get(url, headers={"Content-Type": "application/json"})
    if response.status_code == 200:
        return response.content
    else:
        raise ValueError(
            f"[OCR_DOWNLOAD_ERROR]: {response.status_code} {response.text}"
        )


def get_image_mime_type(content) -> str:
    """Determina el tipo MIME de la imagen."""
    image_type = imghdr.what(None, h=content)
    print(f"Tipo de imagen detectado: {image_type}")
    if image_type in ["jpg", "jpeg", "png", "webp"]:
        return f"image/{image_type}"
    return None


def clean_data(content):
    """Limpia la respuesta eliminando backticks y bloques de código."""
    content = re.sub(r"^```[\w]*\n|```$", "", content, flags=re.MULTILINE)
    return content.strip()


def get_prompt(prompt_key=None) -> str | None:
    """
    Retorna el contenido del prompt almacenado en una variable de entorno
    """

    try:
        storage_url = os.environ.get(prompt_key)
        if not storage_url:
            raise ValueError(f"Environment variable '{storage_url}' is not set.")

        with open(storage_url, "r", encoding="utf-8-sig") as file:
            return file.read()

        return None
    except Exception as e:
        logger.error(f"Error retrieving prompt: {e}")
        return None


def semantize_products(products):
    """
    Convierte la lista de productos en un texto semantizado para el prompt clasificador.
    """
    result = []
    for idx, product in enumerate(products, 1):
        texto = (
            f"Producto {idx}: "
            f"Marca: {product.get('brand', 'N/A')}, "
            f"Nombre: {product.get('displayName', 'N/A')}, "
            f"Vendedor: {product.get('sellerName', 'N/A')}, "
            f"Precio: S/{product.get('price', 'N/A')}, "
            f"Envío: {product.get('shipping', 'N/A')}, "
            f"Reseñas: {product.get('reviewData', 'N/A')}, "
            f"Imagen: {product.get('picture_url', 'N/A')}, "
            f"Recomendaciones del OCR IA de la Imagen: {product.get('recomendations_from_ocr', 'N/A')}"
        )
        result.append(texto)
    return "\n".join(result)


def get_single_falabella_product(product_url: str) -> dict:
    """
    Obtiene los datos de un producto de Falabella Perú usando el productId extraído de la URL.
    """
    import requests
    import re

    # Extraer el productId de la URL
    match = re.search(r"/(\d+)$", product_url)
    if not match:
        raise ValueError("No se pudo extraer el productId de la URL proporcionada.")
    product_id = match.group(1)

    endpoint = f"https://www.falabella.com.pe/s/browse/v1/sponsored-products/pe?productId={product_id}"
    headers = {
        "accept": "*/*",
        "accept-language": "es,en-US;q=0.9,en;q=0.8",
        "cache-control": "no-cache",
        "origin": "https://sodimac.falabella.com.pe",
        "pragma": "no-cache",
        "priority": "u=1, i",
        "referer": "https://sodimac.falabella.com.pe/",
        "refererurl": product_url,
        "sec-ch-ua": '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    }

    response = requests.get(endpoint, headers=headers)
    if response.status_code != 200:
        raise ValueError(
            f"Error al obtener el producto: {response.status_code} {response.text}"
        )

    # La respuesta tiene la estructura: {"success": bool, "sponsoredProducts": [...]}
    data = response.json()

    # Validar que la respuesta sea exitosa
    if not data.get("success", False):
        raise ValueError(
            f"La API respondió con success=False: {data.get('errors', [])}"
        )

    # Extraer el primer producto de sponsoredProducts
    sponsored_products = data.get("sponsoredProducts", [])
    if not sponsored_products:
        raise ValueError(f"No se encontraron productos para el productId: {product_id}")

    return sponsored_products[0]


def extract_falabella_product_data(raw_product: dict) -> dict:
    """
    Extrae y normaliza los campos principales de un producto Falabella para el flujo del agente.
    """
    price = None
    if "prices" in raw_product and raw_product["prices"]:
        # Primero buscar el precio tipo 3 (precio actual sin cruzar)
        for p in raw_product["prices"]:
            if p.get("type") == 3 and not p.get("crossed", False):
                price_str = p.get("originalPrice", "")
                # Limpiar el string: remover comas y convertir a float
                price = float(price_str.replace(",", "")) if price_str else None
                break

        # Si no encontró precio tipo 3, buscar precio tipo 1 (oportunidad única)
        if price is None:
            for p in raw_product["prices"]:
                if p.get("type") == 1 and not p.get("crossed", False):
                    price_str = p.get("originalPrice", "")
                    price = float(price_str.replace(",", "")) if price_str else None
                    break

        # Si aún no hay precio, tomar el primero que no esté cruzado
        if price is None:
            for p in raw_product["prices"]:
                if not p.get("crossed", False):
                    price_str = p.get("originalPrice", "")
                    price = float(price_str.replace(",", "")) if price_str else None
                    break

    picture_url = None
    if "mediaUrls" in raw_product and raw_product["mediaUrls"]:
        picture_url = raw_product["mediaUrls"][0]

    review_data = None
    if "totalReviews" in raw_product:
        review_data = raw_product["totalReviews"]
    elif "rating" in raw_product and raw_product["rating"] > 0:
        review_data = raw_product["rating"]

    return {
        "brand": raw_product.get("brand", "N/A"),
        "displayName": raw_product.get("displayName", "N/A"),
        "sellerName": raw_product.get("sellerName", "N/A"),
        "price": price,
        "shipping": "N/A",
        "reviewData": review_data,
        "picture_url": picture_url,
        "url": raw_product.get("url", "N/A"),
    }

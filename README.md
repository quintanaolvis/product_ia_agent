# product_ia_agent

**product_ia_agent** es una API basada en FastAPI que permite analizar productos de Falabella Perú, extraer información relevante mediante OCR y LLM, y generar recomendaciones para optimizar la presentación de productos en e-commerce.

## Estructura del proyecto

```
.env
.gitignore
api.py
requirements.txt
prompts/
    classifier_prompt.md
    ocr_prompt.md
shared/
    llm_manager.py
    ocr_manager.py
    utils.py
```

## Instalación

1. Clona el repositorio.
2. Instala las dependencias:

```sh
pip install -r requirements.txt
```

3. Configura las variables de entorno en `.env` (ya incluido en el repo).

## Uso

Inicia el servidor FastAPI con:

```sh
uvicorn api:app --reload
```

### Endpoint principal

- **POST /agent/query**

  Envía la URL de un producto de Falabella Perú y recibe recomendaciones de optimización.

  **Ejemplo de request:**

  ```json
  {
    "product_url": "https://www.falabella.com.pe/falabella-pe/product/12345678"
  }
  ```

  **Respuesta:**

  ```json
  {
    "message": { ... }
  }
  ```

## ¿Cómo funciona?

- Extrae datos del producto usando la URL.
- Descarga la imagen y realiza OCR con un modelo LLM.
- Evalúa la calidad de la información del producto y genera propuestas de mejora usando prompts personalizados.

---

Para dudas o mejoras, abre un issue o contacta al equipo.

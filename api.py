import json
from fastapi import FastAPI
from pydantic import BaseModel
from shared.utils import (
    extract_falabella_product_data,
    get_single_falabella_product,
    get_prompt,
    semantize_products,
)
from shared.ocr_manager import invoke_ocr
from shared.llm_manager import get_llm_manager

app = FastAPI()


class QueryRequest(BaseModel):
    product_url: str


@app.post("/agent/query")
def agent_query(request: QueryRequest):
    product_url = request.product_url
    raw_product = get_single_falabella_product(product_url)
    product = extract_falabella_product_data(raw_product)
    if not product.get("picture_url"):
        product["recomendations_from_ocr"] = "El producto no tiene imagen disponible"
    else:
        ocr_result = invoke_ocr(product["picture_url"])
        product["recomendations_from_ocr"] = ocr_result.get(
            "text", "No se pudo extraer texto"
        )
    product_to_evaluate = semantize_products([product])
    pre_prompt = get_prompt("CLASSIFIER_PROMPT")
    prompt = pre_prompt.format(products_to_evaluate=product_to_evaluate)
    llm_manager = get_llm_manager()
    session_data = {
        "user_id": "default_user",
        "messages": [{"role": "user", "content": prompt}],
    }
    response_from_llm = llm_manager.invoke(prompt, session_data=session_data)
    content_str = response_from_llm["choices"][0]["message"]["content"]
    try:
        content_json = json.loads(content_str)
        return {"response": content_json.get("response", content_str)}
    except Exception:
        return {"response": content_str}

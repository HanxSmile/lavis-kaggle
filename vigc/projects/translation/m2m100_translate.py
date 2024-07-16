import logging
import os

from fastapi import FastAPI
from typing import Dict, List
from pydantic import BaseModel

from easynmt import EasyNMT, models
import time
from utils import preproc

# Logging config
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TranslateRequest(BaseModel):
    parameter: Dict


class TranslateResponse(BaseModel):
    trans_results: List[str]


model = EasyNMT(translator=models.AutoModel("/app/m2m100_1.2B"))

app = FastAPI()


@app.get("/ready")
def ready():
    return {"status": "OK"}


@app.post("/v1/translate")
def translate(request: TranslateRequest) -> TranslateResponse:
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    texts = request.parameter["text"]
    source_lang = request.parameter["from"]
    target_lang = request.parameter["to"]
    translated_texts = []

    for text in texts:
        logger.debug(f"Received request with text of length {len(text)}...")
        start = time.time()
        try:
            # translated_text = convert(model.translate(text, source_lang='vi', target_lang='zh'), 'zh-hans')
            # translated_text = translated_text.replace(',', '，').replace('。 ', '。').replace('!', '！').replace('?', '？')
            translated_text = model.translate(preproc(text), source_lang=source_lang, target_lang=target_lang)
            translated_texts.append(translated_text)

            end = time.time()
            logger.debug(f'Translated to a text with a length of {len(translated_text)}')
            logger.debug('程序运行时间为: %s Seconds' % (end - start))


        except Exception as e:
            logger.error(f"Translation failed due to {e}.")
            logger.error(f"Empty return for debugging only.")
            translated_texts.append('')

    return TranslateResponse(
        trans_results=translated_texts
    )

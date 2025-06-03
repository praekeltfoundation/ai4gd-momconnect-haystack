from os import environ
from dotenv import load_dotenv

from fastapi import FastAPI

load_dotenv()
API_TOKEN = environ["API_TOKEN"]

app = FastAPI()


@app.get("/health")
def health():
    return {"health": "ok"}

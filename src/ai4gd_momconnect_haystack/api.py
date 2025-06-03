from os import environ
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException

load_dotenv()
API_TOKEN = environ["API_TOKEN"]

app = FastAPI()


@app.get("/health")
def health():
    return {"health": "ok"}


def verify_token(authorization: Annotated[str, Header()]):
    """
    Verify the API token from the Authorization header.
    """
    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() != "token":
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication format. Expected 'Token <token>'",
        )

    if credential != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
        )

    return credential


@app.post("/v1/onboarding")
def onboarding(token: str = Depends(verify_token)):
    return {}

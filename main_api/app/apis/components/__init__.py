from typing import Any, Dict, Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator

from app import settings

router = APIRouter()


class ComponentsInput(BaseModel):
    component: str
    route: str
    method: str
    payload: Optional[Dict[str, Any]] = None

    @validator("component", each_item=False)
    def component_name(cls, v):
        valid_component_names = settings.VALID_COMPONENT_NAMES
        if v not in valid_component_names:
            raise HTTPException(
                status_code=422,
                detail=f"Valid component names: {valid_component_names}",
            )
        return v


@router.post("/components")
def endpoint(input_data: ComponentsInput):
    "Directly communicates with components"
    if input_data.component == "models_api":
        component_url = settings.models_api_url
    elif input_data.component == "transformers_api":
        component_url = settings.transformers_api_url
    url = "{url}{route}".format(url=component_url, route=input_data.route)
    if input_data.method == "GET":
        r = requests.get(url=url, params=input_data.payload)
        r.raise_for_status()
        return r.json()
    elif input_data.method == "POST":
        r = requests.post(url=url, json=input_data.payload)
        r.raise_for_status()
        return r.json()

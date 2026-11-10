"""Observation-bound visual selections. A selection is not a verified object pose."""

import hashlib
from typing import Annotated

from pydantic import Field, StrictInt

from .contracts import Model
from .data import dumps
from .perception import local_geometry

BoundingBox = Annotated[list[StrictInt], Field(min_length=4, max_length=4)]


class GroundTarget(Model):
    target: str = Field(min_length=1, max_length=300)
    camera: str
    observation_step: StrictInt = Field(ge=0)
    bbox: BoundingBox
    confidence: float = Field(ge=0, le=1)


class GroundingQuery(Model):
    grounding_id: str = Field(pattern=r"^[0-9a-f]{32}$")


def ground_target(obs, request, source):
    if request.observation_step != obs.step or request.camera not in obs.rgb:
        raise ValueError("Grounding needs a matching observation step and camera")
    height, width = obs.rgb[request.camera].shape[:2]
    x0, y0, x1, y1 = request.bbox
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError("Grounding bbox outside original image")
    record = dict(
        **request.model_dump(),
        episode_id=obs.episode_id,
        sim_time=obs.sim_time,
        source=source,
        original_size=[width, height],
        image_sha256=hashlib.sha256(obs.rgb[request.camera].tobytes()).hexdigest(),
        note="Visual selection; confidence is self-reported, not pose or contact verification",
    )
    record["grounding_id"] = hashlib.sha256(dumps(record).encode()).hexdigest()[:32]
    return record


def grounded_geometry(obs, grounding):
    if grounding["episode_id"] != obs.episode_id or grounding["observation_step"] != obs.step:
        raise ValueError("Stale grounding: reobserve and reselect after motion")
    if hashlib.sha256(obs.rgb[grounding["camera"]].tobytes()).hexdigest() != grounding["image_sha256"]:
        raise ValueError("Grounding image identity changed")
    return {
        **local_geometry(obs, grounding["camera"], list(grounding["bbox"])),
        "grounding_id": grounding["grounding_id"],
        "observation_step": obs.step,
    }

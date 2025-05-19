import os

import uvicorn
from fastapi import UploadFile, File, Depends, HTTPException, Form, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
import io
import numpy as np
import cv2
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
from contextlib import asynccontextmanager
import tensorflow as tf

from fastapi.middleware.cors import CORSMiddleware
from encoder import Encoder


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ⏳ Частина ДО yield — тут ініціалізуються ресурси
    if tf.test.gpu_device_name():
        print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    encoder = Encoder()
    app.state.encoder = encoder

    yield  # ⬅ Після цього FastAPI починає обробляти HTTP-запити


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_encoder(request: Request) -> Encoder:
    return request.app.state.encoder


def validate_image_file(file: UploadFile = File(...)) -> UploadFile:
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid file type")
    valid_mime_types = {"image/jpeg", "image/png"}

    if file.content_type not in valid_mime_types:
        raise HTTPException(status_code=400, detail="Invalid MIME type")
    return file


def validate_and_convert_to_nparray(file_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Corrupted or invalid image file")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@app.post("/add-fragments/")
async def add_fragments(file: UploadFile = Depends(validate_image_file),
                        encoder: Encoder = Depends(get_encoder),
                        background_tasks: BackgroundTasks = BackgroundTasks):
    file_bytes = await file.read()

    # Validate and load image
    np_image = validate_and_convert_to_nparray(file_bytes)

    print("Image received successfully")

    fragments_count = encoder.add_fragments_from_img(np_image)
    background_tasks.add_task(encoder.db.build_tree)
    if fragments_count:
        return {
            "status": "Successfully added fragments into base",
            "added_fragments_count": fragments_count,
            "db_fragments_count": encoder.db.get_db_size()
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to add fragments into base")


@app.post("/encode-image/")
async def encode_image(file: UploadFile = Depends(validate_image_file),
                       encoder: Encoder = Depends(get_encoder),
                       background_tasks: BackgroundTasks = BackgroundTasks):
    file_bytes = await file.read()

    image = validate_and_convert_to_nparray(file_bytes)

    print("Image received successfully")

    # Обробка
    try:
        encoded_image = encoder.encode(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")
    # Фоново оновлюємо дерево й додаємо нові фрагменти в БД
    background_tasks.add_task(encoder.db.update_fragments)
    return StreamingResponse(
        io.BytesIO(encoded_image),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{file.filename.split('.')[0]}.bin"'},
    )


class ImageSize(BaseModel):
    width: int = Form()
    height: int = Form()


def get_image_size(
        width: int = Form(...),
        height: int = Form(...)) -> ImageSize:
    return ImageSize(width=width, height=height)


@app.post("/decode-image/")
async def decode_image(
        compressed_img: UploadFile = File(...),
        img_size: ImageSize = Depends(get_image_size),
        encoder: Encoder = Depends(get_encoder)
):
    try:
        # width, height = img_size.width, img_size.height
        compressed_img = await compressed_img.read()

        # Декодуємо отримані дані
        decoded_image = encoder.decode(compressed_img, (img_size.height, img_size.width))

        # Конвертуємо отримане зображення в формат, який можна відправити як відповідь
        decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', decoded_image)

        image_bytes = buffer.tobytes()

        image_io = io.BytesIO(image_bytes)
        image_io.seek(0)

        # Повертаємо як стрімінгову відповідь
        return StreamingResponse(
            image_io,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=decoded_image.png"}
        )

    except ValueError as ve:
        if "could not broadcast input array from shape" in str(ve):
            raise HTTPException(status_code=400, detail="Image decoding failed: Invalid image dimensions")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decoding failed: {str(e)}")


@app.post("/change-similarity-threshold/{threshold}")
def change_similarity_threshold(threshold: float, encoder: Encoder = Depends(get_encoder)):
    # Validate the threshold range
    if not (0 <= threshold <= 1):
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1.")

    # Set the similarity threshold in the encoder
    encoder.set_similarity_threshold(threshold)

    # Return a JSON response
    return {"message": f"Similarity threshold changed to {threshold}"}


@app.post("/get-ssim/")
async def get_ssim_metric(
    original_img: UploadFile = File(...),
    decoded_img: UploadFile = File(...),
    encoder: Encoder = Depends(get_encoder)
):
    # Валідація типів
    validate_image_file(original_img)
    validate_image_file(decoded_img)

    # Читання байтів
    original_bytes = await original_img.read()
    decoded_bytes = await decoded_img.read()

    # Перетворення у NumPy масив
    original_np = validate_and_convert_to_nparray(original_bytes)
    decoded_np = validate_and_convert_to_nparray(decoded_bytes)

    # Обчислення SSIM
    try:
        ssim_value = encoder.get_ssim(original_np, decoded_np)
    except ValueError as ve:
        if "Input images must have the same dimensions" in str(ve):
            raise HTTPException(status_code=400, detail="SSIM calculation failed: Images have different dimensions")

    return {
        "status": "success",
        "ssim": ssim_value
    }

@app.get("/health/")
def health():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port)

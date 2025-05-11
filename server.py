import uvicorn
from fastapi import UploadFile, File, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
from PIL import Image
import io
import numpy as np
import cv2
from encoder import Encoder

app = FastAPI()

def validate_image_file(file: UploadFile = File(...)) -> UploadFile:
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid file type")
    valid_mime_types = {"image/jpeg", "image/png"}

    if file.content_type not in valid_mime_types:
        raise HTTPException(status_code=400, detail="Invalid MIME type")
    return file

@app.post("/encode-image/")
async def upload_file(file: UploadFile = Depends(validate_image_file)):
    file_bytes = await file.read()

    # Перевірка, чи це справді зображення
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  # Перевіряє, чи файл дійсно є зображенням
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted or invalid image file")

    image_array = np.frombuffer(file_bytes, dtype=np.uint8)

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Image received successfully")

    # Обробка
    try:
        encoder = Encoder()
        encoded_image = encoder.encode(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")

    return StreamingResponse(
        io.BytesIO(encoded_image),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{file.filename.split('.')[0]}.bin"'},
    )

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
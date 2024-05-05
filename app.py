from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import uvicorn
from PIL import Image

from services.image import ImageService

app = FastAPI()


@app.post("/try-on/")
async def create_tryon(
    human_img: UploadFile,
    cloth_img: UploadFile,
):
    with open("cloth.jpg", "wb") as cloth:
        cloth.write(await cloth_img.read())
    cloth_img = Image.open("cloth.jpg").convert("RGB")

    with open("human.jpg", "wb") as human:
        human.write(await human_img.read())
    human_img = Image.open("human.jpg").convert("RGB")

    human_img = human_img.copy()
    layer = human_img.copy()
    composite = human_img.copy()

    human_dict = {
        "background": human_img,
        "layers": [layer],
        "composite": composite,
    }

    result = ImageService.get_images(human_dict, cloth_img)
    result.save("res.jpg")

    return FileResponse("res.jpg")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

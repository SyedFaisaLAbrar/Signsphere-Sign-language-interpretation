from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import psl_script as psl_script

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server origin
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    # Placeholder: Perform processing on the video
    # Save or analyze the video as needed
    content = await file.read()
    with open("uploaded_video.webm", "wb") as f:
        f.write(content)
    # Perform your ML model prediction or processing here
    prediction = psl_script.main("uploaded_video.webm")
    return JSONResponse(content={"message": f"Video processed successfully. Action ( {prediction} )"}, status_code=200)

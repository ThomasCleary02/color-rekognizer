import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from detect_color import RGBColorAnalyzer
from mangum import Mangum
import numpy as np
import base64
import cv2
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure FastAPI with root_path
stage = os.environ.get('STAGE', None)
root_path = f"/{stage}" if stage else ""
app = FastAPI(root_path=root_path)

analyzer = RGBColorAnalyzer()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Received request: {request.method} {request.url}")
    logger.debug(f"Headers: {request.headers}")
    response = await call_next(request)
    logger.debug(f"Response status: {response.status_code}")
    return response

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "The Lambda function is running correctly"}

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...), isBase64Encoded: bool = False):
    try:
        logger.info(f"Received file: {file.filename}")
        
        contents = await file.read()

        # Decode the Base64 image data if needed
        if isBase64Encoded:
            contents = base64.b64decode(contents)
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        logger.debug(f"File size: {len(contents)} bytes")
        
        nparr = np.frombuffer(contents, np.uint8)
        logger.debug(f"Numpy array shape: {nparr.shape}")
        
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        logger.debug(f"Decoded image shape: {img.shape}")
        
        results = analyzer.analyze_image(img, num_colors=3)
        
        formatted_results = {}
        for i, (rgb, percentage) in enumerate(results.items(), 1):
            rgb_tuple = tuple(numpy_to_python(x) for x in rgb)
            complement_tuple = RGBColorAnalyzer.find_complement(rgb_tuple)
            
            formatted_results[f"color{i}"] = {
                "color": {
                    "rgb": list(rgb_tuple),
                    "hex": RGBColorAnalyzer.rgb_to_hex(rgb_tuple),
                },
                "compliment": {
                    "rgb": list(complement_tuple),
                    "hex": RGBColorAnalyzer.rgb_to_hex(complement_tuple),
                },
                "percentage": f"{numpy_to_python(percentage)}%",
            }
        
        logger.info("Image analysis completed successfully")
        return JSONResponse(content=formatted_results)
    
    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        await file.close()

# Modify the Mangum handler configuration
handler = Mangum(app, lifespan="off")
logger.info("Mangum handler initialized")
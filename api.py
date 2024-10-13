from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from detect_color import RGBColorAnalyzer
from mangum import Mangum
import numpy as np
import logging
import cv2
import os


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure FastAPI with root_path
stage = os.environ.get('STAGE', None)
root_path = f"/{stage}" if stage else ""
app = FastAPI(root_path=root_path)

app = FastAPI()
analyzer = RGBColorAnalyzer()

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>API Documentation</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f0f0f0;
            }
            .button {
                display: inline-block;
                padding: 10px 20px;
                font-size: 18px;
                cursor: pointer;
                text-align: center;
                text-decoration: none;
                outline: none;
                color: #fff;
                background-color: #4CAF50;
                border: none;
                border-radius: 15px;
                box-shadow: 0 9px #999;
            }
            .button:hover {background-color: #3e8e41}
            .button:active {
                background-color: #3e8e41;
                box-shadow: 0 5px #666;
                transform: translateY(4px);
            }
        </style>
    </head>
    <body>
        <a href="/docs" class="button">Click here for docs</a>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/analyze_image/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        
        contents = await file.read()
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
                    "rgb": list(rgb_tuple),  # Convert to list for JSON serialization
                    "hex": RGBColorAnalyzer.rgb_to_hex(rgb_tuple),
                },
                "compliment": {  # Note: 'compliment' is used here as per your request, though 'complement' is the correct spelling
                    "rgb": list(complement_tuple),  # Convert to list for JSON serialization
                    "hex": RGBColorAnalyzer.rgb_to_hex(complement_tuple),
                },
                "percentage": f"{numpy_to_python(percentage)}%",  # Add percentage sign
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

handler = Mangum(app, lifespan="off")
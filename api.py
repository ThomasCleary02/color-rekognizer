# File: api.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import logging
from detect_color import RGBColorAnalyzer

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
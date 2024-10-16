from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure FastAPI with root_path
stage = os.environ.get('STAGE', None)
root_path = f"/{stage}" if stage else ""
app = FastAPI(root_path=root_path)

analyzer = RGBColorAnalyzer()

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
        
        img = Image.open(io.BytesIO(contents))
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        logger.debug(f"Decoded image size: {img.size}")
        
        results = analyzer.analyze_image(img, num_colors=3)
        
        formatted_results = {}
        for i, (rgb, percentage) in enumerate(results.items(), 1):
            complement_tuple = RGBColorAnalyzer.find_complement(rgb)
            
            formatted_results[f"color{i}"] = {
                "color": {
                    "rgb": list(rgb),
                    "hex": RGBColorAnalyzer.rgb_to_hex(rgb),
                },
                "compliment": {
                    "rgb": list(complement_tuple),
                    "hex": RGBColorAnalyzer.rgb_to_hex(complement_tuple),
                },
                "percentage": f"{percentage}%",
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
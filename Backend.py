from fastapi import FastAPI, File, UploadFile
from PIL import Image
import pytesseract
from transformers import pipeline
from io import BytesIO

app = FastAPI()

# Load HuggingFace Model (Offline or Free via API)
qa_pipeline = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

@app.post("/evaluate/")
async def evaluate_product(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    
    # OCR - Extract text from the image
    extracted_text = pytesseract.image_to_string(image)

    # Generative AI Prompt
    prompt = f"""
    User Information:
    - Age: 32
    - Gender: Male
    - Weight: 190 lbs
    - Height: 193 cm
    - Heritage: Indian (Asian)
    - Work Style: Regular 9-5 schedule
    - Meals per Day: 2 meals
    - Calorie Intake Goal: 2000 kcal/day
    - Sleep Duration: 6-7 hours/night
    - Travel Frequency: Monthly leisure trips
    - Physical Activity: 5+ days per week
    - Diet: Non-vegetarian
    - Health Goals: Maintain an active lifestyle and build muscle for good physique
    - Health Data Integration: not available

    Analyze the following product label based on this user's health profile:
    {extracted_text}

    Provide the following:
    1. Health Score (0–10) with explanation (based on calories, sugar, fats, sodium, and micronutrients compared to user goals).
    2. Highlights: Benefits (e.g., high protein, fiber, vitamins, etc.).
    3. Warnings: List any concerns (e.g., added sugars, allergens, preservatives).
    4. Suitability for Travel: Assess whether the product is convenient and nutritious for frequent travel.
    5. Recommendations: Suggest 3 healthier travel-friendly alternatives for similar use cases.
    """

    # Generate AI-based evaluation
    response = qa_pipeline(prompt, max_length=1000)[0]["generated_text"]
    return {"evaluation": response}

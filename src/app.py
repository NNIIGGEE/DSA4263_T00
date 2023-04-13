from fastapi import FastAPI
from src import classifier_router

app = FastAPI()
app.include_router(classifier_router.router , prefix='/src')


@app.get('/prediction', status_code=200)
async def prediction():
    return 'prediction classifier is all ready to go!'
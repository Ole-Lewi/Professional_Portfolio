from groq import Groq

client = Groq(api_key="your_groq_api_key_here")

models = client.models.list()
for model in models.data:
    print(model.id)
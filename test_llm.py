from app.llm.router import get_cheap_llm
import litellm
llm = get_cheap_llm()
print("Model:", llm.model)
print("API Key:", "yes" if llm.api_key else "no")
if hasattr(llm, "base_url"):
    print("Base URL:", llm.base_url)

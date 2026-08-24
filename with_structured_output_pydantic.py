from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation"
)

chat_model = ChatHuggingFace(llm=llm)


class Review(BaseModel):
    Summary: str = Field(
        description="A brief summary of the product review"
    )

    specs: list[str] = Field(
        description="Specifications related to the product"
    )

    cost: int = Field(
        default=22000,
        description="The cost of the product in India, if available"
    )


structured_model = chat_model.with_structured_output(Review)

response = structured_model.invoke(
    """
    The Motorola Edge 50 Fusion is a well-balanced mid-range smartphone
    that offers a premium design, smooth performance, and an excellent user
    experience. The phone features a stylish curved-edge design with a
    comfortable grip and an IP68 rating for water and dust resistance.
    It has a 6.7-inch 144Hz pOLED display, which provides vibrant colors,
    deep blacks, and a very smooth scrolling experience.
    """
)

print(response)
print(response.Summary)
print(response.specs)
print(response.cost)
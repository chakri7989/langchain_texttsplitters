from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation"
)

chat = ChatHuggingFace(llm=llm)

#schema

class abilities(TypedDict):
    summary:str
    sentiment:Annotated[Literal["positive", "negative", "neutral"], "The sentiment of the product review"]
    specifications:Annotated[list[str], "A list of specifications of the product"]
    cost:Annotated[Optional[str], "The cost of the product in USD, if available"]

structure=chat.with_structured_output(abilities)
response = structure.invoke("""The Motorola Edge 50 Fusion is a well-balanced mid-range smartphone that offers a premium design, smooth performance, and an excellent user experience. The phone features a stylish curved-edge design with a comfortable grip and an IP68 rating for water and dust resistance, making it feel more like a flagship device. One of its biggest highlights is the 6.7-inch 144Hz pOLED display, which provides vibrant colors, deep blacks, and a very smooth scrolling experience, making it great for watching videos, browsing, and everyday usage. The clean Android interface with minimal bloatware is another major advantage, as the software feels fast and user-friendly. The Snapdragon 7s Gen 2 processor handles daily tasks like social media, multitasking, and streaming smoothly, although it is not the best choice for heavy gaming. The 50MP main camera with OIS captures good-quality photos in daylight with natural colors, but low-light photography and camera processing could be improved. The 5000mAh battery provides a full day of usage, and the 68W fast charging support helps quickly recharge the device. However, the phone has some drawbacks, including average gaming performance, no expandable storage option, and a curved display that may be more difficult and expensive to repair. Overall, the Motorola Edge 50 Fusion is a great choice for users who prioritize a premium design, excellent display, clean software, and reliable battery life, but gamers and users looking for the highest performance may find better alternatives in the same price range.""")

print(response['specifications'])
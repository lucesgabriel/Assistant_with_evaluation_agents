from pydantic import BaseModel
from openai import OpenAI
from enum import Enum
from typing import Optional
from datetime import datetime
import os
from pathlib import Path

class ContentType(Enum):
    TWEET = "tweet"
    EMAIL = "email"
    TEXT_MESSAGE = "text_message"
    LINKEDIN_POST = "linkedin_post"
    INSTAGRAM_CAPTION = "instagram_caption"

class WritingContent(BaseModel):
    content: str
    tone: str
    word_count: Optional[int]

class WritingAssistant:
    def __init__(self, output_dir: str = "generated_content"):
        self.client = OpenAI()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each content type
        for content_type in ContentType:
            (self.output_dir / content_type.value).mkdir(exist_ok=True)

    def generate_content(
        self,
        topic: str,
        content_type: ContentType,
        tone: str = "professional",
        additional_context: str = ""
    ) -> WritingContent:
        system_prompts = {
            ContentType.TWEET: "Write a compelling tweet (max 280 characters)",
            ContentType.EMAIL: "Write a clear and concise email",
            ContentType.TEXT_MESSAGE: "Write a brief text message",
            ContentType.LINKEDIN_POST: "Write an engaging LinkedIn post",
            ContentType.INSTAGRAM_CAPTION: "Write an engaging Instagram caption with hashtags"
        }

        prompt = f"Topic: {topic}\nTone: {tone}\n"
        if additional_context:
            prompt += f"Additional context: {additional_context}\n"

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompts[content_type]},
                {"role": "user", "content": prompt},
            ],
            response_format=WritingContent,
        )

        result = completion.choices[0].message.parsed
        self._save_to_file(result, topic, content_type)
        return result

    def _save_to_file(self, content: WritingContent, topic: str, content_type: ContentType):
        # Create a filename with timestamp and sanitized topic
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_topic = "".join(c if c.isalnum() else "_" for c in topic)
        filename = f"{timestamp}_{sanitized_topic[:30]}"
        
        # Add appropriate file extension
        file_extensions = {
            ContentType.EMAIL: ".eml",
            ContentType.TWEET: ".txt",
            ContentType.TEXT_MESSAGE: ".txt",
            ContentType.LINKEDIN_POST: ".md",
            ContentType.INSTAGRAM_CAPTION: ".txt"
        }
        
        filepath = self.output_dir / content_type.value / f"{filename}{file_extensions[content_type]}"
        
        # Save the content with metadata
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Topic: {topic}\n")
            f.write(f"Tone: {content.tone}\n")
            if content.word_count:
                f.write(f"Word count: {content.word_count}\n")
            f.write("\n---\n\n")
            f.write(content.content)

    def interactive_generate(self):
        """Interactive content generation through command line input"""
        print("\n=== Writing Assistant ===")
        
        # Display available content types
        print("\nAvailable content types:")
        for i, content_type in enumerate(ContentType, 1):
            print(f"{i}. {content_type.value.replace('_', ' ').title()}")
        
        # Get content type
        while True:
            try:
                choice = int(input("\nSelect content type (enter number): "))
                if 1 <= choice <= len(ContentType):
                    content_type = list(ContentType)[choice - 1]
                    break
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get topic
        topic = input("\nEnter the topic: ").strip()
        
        # Get tone
        tone = input("\nEnter the desired tone (e.g., professional, casual, exciting): ").strip()
        
        # Get additional context (optional)
        additional_context = input("\nEnter any additional context (press Enter to skip): ").strip()
        
        print("\nGenerating content...")
        result = self.generate_content(
            topic=topic,
            content_type=content_type,
            tone=tone,
            additional_context=additional_context
        )
        
        print("\n=== Generated Content ===")
        print(f"\nContent Type: {content_type.value}")
        print(f"Topic: {topic}")
        print(f"Tone: {result.tone}")
        if result.word_count:
            print(f"Word Count: {result.word_count}")
        print("\nContent:")
        print("-------------------")
        print(result.content)
        print("-------------------")
        print(f"\nContent has been saved to the {content_type.value} directory.")
        
        return result

# Example usage
if __name__ == "__main__":
    assistant = WritingAssistant()
    
    while True:
        assistant.interactive_generate()
        
        # Ask if user wants to generate more content
        if input("\nWould you like to generate more content? (y/n): ").lower() != 'y':
            print("\nThank you for using Writing Assistant!")
            break

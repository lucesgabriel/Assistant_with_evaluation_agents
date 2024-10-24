from openai import AsyncOpenAI
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import asyncio

from models_final import (
    ContentType, 
    WritingContent, 
    ContentEvaluation, 
    ContentRewrite
)
from content_evaluator_final import ContentEvaluator

class BaseWritingAgent:
    def __init__(self, content_type: ContentType):
        self.client = AsyncOpenAI()
        self.content_type = content_type
        
    @property
    def system_prompt(self) -> str:
        raise NotImplementedError
        
    def _build_prompt(self, topic: str, tone: str, additional_context: str = "") -> str:
        prompt = f"Create content about: {topic}\nDesired tone: {tone}"
        if additional_context:
            prompt += f"\nAdditional context: {additional_context}"
        prompt += "\n\nProvide the content in a clear, well-structured format appropriate for the platform."
        return prompt

    async def generate_content(self, topic: str, tone: str, additional_context: str = "") -> WritingContent:
        result = await self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._build_prompt(topic, tone, additional_context)},
            ],
            response_format=WritingContent,
        )
        return result.choices[0].message.parsed

class TweetAgent(BaseWritingAgent):
    @property
    def system_prompt(self) -> str:
        return "You are a social media expert crafting engaging tweets that maximize engagement while staying within character limits."

class EmailAgent(BaseWritingAgent):
    @property
    def system_prompt(self) -> str:
        return "You are a professional email writer crafting clear, effective, and well-structured emails."

class TextMessageAgent(BaseWritingAgent):
    @property
    def system_prompt(self) -> str:
        return "You are crafting clear, concise, and appropriate text messages that effectively communicate the message."

class LinkedInAgent(BaseWritingAgent):
    @property
    def system_prompt(self) -> str:
        return "You are a LinkedIn content expert creating engaging professional posts that resonate with business audiences."

class InstagramAgent(BaseWritingAgent):
    @property
    def system_prompt(self) -> str:
        return "You are crafting engaging and relevant Instagram captions that drive engagement and complement visual content."

class WritingAssistant:
    def __init__(self, output_dir: str = "generated_content"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize specialized agents
        self.agents = {
            ContentType.TWEET: TweetAgent(ContentType.TWEET),
            ContentType.EMAIL: EmailAgent(ContentType.EMAIL),
            ContentType.TEXT_MESSAGE: TextMessageAgent(ContentType.TEXT_MESSAGE),
            ContentType.LINKEDIN_POST: LinkedInAgent(ContentType.LINKEDIN_POST),
            ContentType.INSTAGRAM_CAPTION: InstagramAgent(ContentType.INSTAGRAM_CAPTION)
        }
        
        self.file_extensions = {
            ContentType.EMAIL: ".eml",
            ContentType.TWEET: ".txt",
            ContentType.TEXT_MESSAGE: ".txt",
            ContentType.LINKEDIN_POST: ".md",
            ContentType.INSTAGRAM_CAPTION: ".txt"
        }
        
        # Create content type directories
        for content_type in ContentType:
            (self.output_dir / content_type.value).mkdir(exist_ok=True)
        
        self.evaluator = ContentEvaluator(self.output_dir)

    def _save_to_file(self, content: WritingContent, topic: str, content_type: ContentType, content_id: str) -> str:
        """Save content to file using the provided content_id"""
        filepath = self.output_dir / content_type.value / f"{content_id}{self.file_extensions[content_type]}"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Topic: {topic}\n")
            f.write(f"Tone: {content.tone}\n")
            if content.word_count:
                f.write(f"Word count: {content.word_count}\n")
            f.write("\n---\n\n")
            f.write(content.content)
        
        print(f"✓ Content saved to: {filepath}")
        return content_id

    def _save_rewrite_to_file(self, rewrite: ContentRewrite, original_id: str, content_type: ContentType):
        rewrite_dir = self.output_dir / content_type.value / "rewrites"
        rewrite_dir.mkdir(exist_ok=True)
        
        filepath = rewrite_dir / f"{original_id}_rewrite{self.file_extensions[content_type]}"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== Original Content ===\n\n")
            f.write(rewrite.original_content)
            f.write("\n\n=== Improved Content ===\n\n")
            f.write(rewrite.improved_content)
            f.write("\n\n=== Changes Made ===\n")
            for change in rewrite.changes_made:
                f.write(f"• {change}\n")
            f.write("\n=== Improvement Focus ===\n")
            for focus in rewrite.improvement_focus:
                f.write(f"• {focus}\n")
        
        print(f"✓ Rewrite saved to: {filepath}")
        return filepath

    async def generate_content_async(
        self,
        topic: str,
        content_type: ContentType,
        tone: str = "professional",
        additional_context: str = "",
        auto_rewrite: bool = True
    ) -> tuple[WritingContent, ContentEvaluation, Optional[ContentRewrite]]:
        print(f"\n🎯 Generating {content_type.value.replace('_', ' ')}...")
        print(f"• Topic: {topic}")
        print(f"• Tone: {tone}")
        if additional_context:
            print(f"• Additional context provided")
        
        agent = self.agents[content_type]
        result = await agent.generate_content(topic, tone, additional_context)
        print("✓ Content generated successfully")
        
        # Generate content ID first so we can use it for both content and evaluation
        content_id = self._generate_content_id(topic)
        
        # Save original content
        self._save_to_file(result, topic, content_type, content_id)
        
        # Evaluate content
        evaluation = await self.evaluator.evaluate_content(
            content=result.content,
            intended_tone=tone,
            content_type=content_type
        )
        
        # Save evaluation
        self.evaluator.save_evaluation(evaluation, content_id, content_type)
        
        rewrite = None
        if auto_rewrite:
            needs_rewrite = any(
                getattr(evaluation, aspect).score < 8.0 
                for aspect in ["clarity", "engagement", "tone_consistency"]
            )
            
            if needs_rewrite:
                print("\n🔄 Content scored below threshold, generating rewrite...")
                rewrite = await self.evaluator.rewrite_content(
                    result, evaluation, content_type
                )
                self._save_rewrite_to_file(rewrite, content_id, content_type)
        
        return result, evaluation, rewrite

    def _generate_content_id(self, topic: str) -> str:
        """Generate a unique content ID based on timestamp and topic"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_topic = "".join(c if c.isalnum() else "_" for c in topic)
        return f"{timestamp}_{sanitized_topic[:30]}"

    def generate_content(self, *args, **kwargs) -> tuple[WritingContent, ContentEvaluation, Optional[ContentRewrite]]:
        """Synchronous wrapper for generate_content_async"""
        return asyncio.run(self.generate_content_async(*args, **kwargs))

    async def interactive_generate_async(self):
        print("\n🤖 === Writing Assistant ===")
        
        print("\n📝 Available content types:")
        for i, content_type in enumerate(ContentType, 1):
            print(f"{i}. {content_type.value.replace('_', ' ').title()}")
        
        while True:
            try:
                choice = int(input("\n🔍 Select content type (enter number): "))
                if 1 <= choice <= len(ContentType):
                    content_type = list(ContentType)[choice - 1]
                    break
                print("❌ Invalid choice. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")
        
        topic = input("\n📋 Enter the topic: ").strip()
        tone = input("\n🎭 Enter the desired tone (e.g., professional, casual, exciting): ").strip()
        additional_context = input("\n📌 Enter any additional context (press Enter to skip): ").strip()
        
        print("\n🚀 Starting content generation process...")
        result, evaluation, rewrite = await self.generate_content_async(
            topic=topic,
            content_type=content_type,
            tone=tone,
            additional_context=additional_context,
            auto_rewrite=True
        )
        
        self._display_results(result, evaluation, rewrite, content_type, topic)
        return result, evaluation, rewrite

    def _display_results(self, result: WritingContent, evaluation: ContentEvaluation, 
                        rewrite: Optional[ContentRewrite], content_type: ContentType, topic: str):
        print("\n📄 === Generated Content ===")
        print(f"\nContent Type: {content_type.value}")
        print(f"Topic: {topic}")
        print(f"Tone: {result.tone}")
        if result.word_count:
            print(f"Word Count: {result.word_count}")
        print("\nContent:")
        print("-------------------")
        print(result.content)
        print("-------------------")
        
        print("\n📊 === Content Evaluation ===")
        for aspect, score in evaluation.model_dump().items():
            if aspect != "timestamp":
                print(f"\n🎯 {aspect.replace('_', ' ').title()}:")
                print(f"Reasoning: {score['reasoning']}")
                print(f"Score: {score['score']}/10")
                print("Suggestions:")
                for suggestion in score['suggestions']:
                    print(f"• {suggestion}")
        
        if rewrite:
            print("\n📝 === Content Rewrite ===")
            print("Original Content:")
            print("-------------------")
            print(rewrite.original_content)
            print("-------------------")
            print("Improved Content:")
            print("-------------------")
            print(rewrite.improved_content)
            print("-------------------")
            print("Changes Made:")
            for change in rewrite.changes_made:
                print(f"• {change}")
            print("\nImprovement Focus:")
            for focus in rewrite.improvement_focus:
                print(f"• {focus}")

    def interactive_generate(self):
        """Synchronous wrapper for interactive_generate_async"""
        return asyncio.run(self.interactive_generate_async())

if __name__ == "__main__":
    assistant = WritingAssistant()
    
    while True:
        assistant.interactive_generate()
        
        if input("\nWould you like to generate more content? (y/n): ").lower() != 'y':
            print("\nThank you for using Writing Assistant!")
            break

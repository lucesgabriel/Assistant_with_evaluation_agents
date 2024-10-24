from pydantic import BaseModel
from openai import AsyncOpenAI
from enum import Enum
from typing import Optional, List, Dict
from datetime import datetime
import os
from pathlib import Path
import json
import asyncio

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

class EvaluationScore(BaseModel):
    reasoning: str
    score: float  # 0-10 scale
    suggestions: List[str]

class ContentEvaluation(BaseModel):
    clarity: EvaluationScore
    engagement: EvaluationScore
    tone_consistency: EvaluationScore
    timestamp: str

class ContentRewrite(BaseModel):
    original_content: str
    improved_content: str
    changes_made: List[str]
    improvement_focus: List[str]

class ContentEvaluator:
    def __init__(self, output_dir: Path):
        self.client = AsyncOpenAI()
        self.eval_dir = output_dir / "evaluations"
        self.eval_dir.mkdir(exist_ok=True)

    async def _evaluate_aspect(self, aspect: str, content: str, intended_tone: str, content_type: ContentType) -> EvaluationScore:
        print(f"🔍 Evaluating {aspect.replace('_', ' ')}...")
        
        system_prompts = {
            "clarity": "Evaluate the clarity and readability of the content. Consider structure, coherence, and accessibility.",
            "engagement": "Evaluate how engaging and compelling the content is for its intended platform.",
            "tone_consistency": "Evaluate how well the content maintains the intended tone throughout."
        }
        
        messages = [
            {"role": "system", "content": system_prompts[aspect]},
            {"role": "user", "content": f"Content: {content}\nContent Type: {content_type.value}"}
        ]
        
        if aspect == "tone_consistency":
            messages[1]["content"] += f"\nIntended Tone: {intended_tone}"

        completion = await self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            response_format=EvaluationScore
        )
        
        print(f"✓ Completed {aspect.replace('_', ' ')} evaluation")
        return completion.choices[0].message.parsed

    async def evaluate_content(self, content: str, intended_tone: str, content_type: ContentType) -> ContentEvaluation:
        print("\n📊 Starting parallel content evaluation...")
        
        # Create tasks for parallel evaluation
        tasks = {
            aspect: asyncio.create_task(self._evaluate_aspect(aspect, content, intended_tone, content_type))
            for aspect in ["clarity", "engagement", "tone_consistency"]
        }
        
        # Run evaluations in parallel
        results = await asyncio.gather(*tasks.values())
        evaluations = dict(zip(tasks.keys(), results))
        
        print("✨ All evaluations completed successfully")
        return ContentEvaluation(
            **evaluations,
            timestamp=datetime.now().isoformat()
        )

    def save_evaluation(self, evaluation: ContentEvaluation, content_id: str, content_type: ContentType):
        print(f"\n💾 Saving evaluation results...")
        eval_path = self.eval_dir / content_type.value
        eval_path.mkdir(exist_ok=True)
        
        filepath = eval_path / f"{content_id}_evaluation.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(evaluation.model_dump(), f, indent=2)
        print(f"✓ Evaluation saved to: {filepath}")

class WritingAssistant:
    def __init__(self, output_dir: str = "generated_content"):
        self.client = AsyncOpenAI()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Add system prompts for each content type
        self.system_prompts = {
            ContentType.TWEET: "You are a social media expert crafting engaging tweets.",
            ContentType.EMAIL: "You are a professional email writer crafting clear and effective emails.",
            ContentType.TEXT_MESSAGE: "You are crafting clear and appropriate text messages.",
            ContentType.LINKEDIN_POST: "You are a LinkedIn content expert creating engaging professional posts.",
            ContentType.INSTAGRAM_CAPTION: "You are crafting engaging and relevant Instagram captions."
        }
        
        # Add file extensions for different content types
        self.file_extensions = {
            ContentType.EMAIL: ".txt",
            ContentType.TWEET: ".txt",
            ContentType.TEXT_MESSAGE: ".txt",
            ContentType.LINKEDIN_POST: ".md",
            ContentType.INSTAGRAM_CAPTION: ".txt"
        }
        
        # Create content type directories
        for content_type in ContentType:
            (self.output_dir / content_type.value).mkdir(exist_ok=True)
        
        self.evaluator = ContentEvaluator(self.output_dir)

    def _build_prompt(self, topic: str, tone: str, additional_context: str = "") -> str:
        """Build the prompt for content generation."""
        prompt = f"Create content about: {topic}\nDesired tone: {tone}"
        
        if additional_context:
            prompt += f"\nAdditional context: {additional_context}"
            
        prompt += "\n\nProvide the content in a clear, well-structured format appropriate for the platform."
        
        return prompt

    async def generate_content_async(
        self,
        topic: str,
        content_type: ContentType,
        tone: str = "professional",
        additional_context: str = "",
        auto_rewrite: bool = True  # New parameter
    ) -> tuple[WritingContent, ContentEvaluation, Optional[ContentRewrite]]:
        print(f"\n🎯 Generating {content_type.value.replace('_', ' ')}...")
        print(f"• Topic: {topic}")
        print(f"• Tone: {tone}")
        if additional_context:
            print(f"• Additional context provided")
        
        # Generate content
        print("\n✍️ Crafting content...")
        result = await self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompts[content_type]},
                {"role": "user", "content": self._build_prompt(topic, tone, additional_context)},
            ],
            response_format=WritingContent,
        )
        result = result.choices[0].message.parsed
        print("✓ Content generated successfully")
        
        # Evaluate content
        evaluation = await self.evaluator.evaluate_content(
            content=result.content,
            intended_tone=tone,
            content_type=content_type
        )
        
        rewrite = None
        if auto_rewrite:
            # Save original content first to get content_id
            content_id = self._save_to_file(result, topic, content_type)
            
            needs_rewrite = any(
                getattr(evaluation, aspect).score < 8.0 
                for aspect in ["clarity", "engagement", "tone_consistency"]
            )
            
            if needs_rewrite:
                print("\n🔄 Content scored below threshold, generating rewrite...")
                rewrite = await self.rewrite_content_async(result, evaluation, content_type)
                self._save_rewrite_to_file(rewrite, content_id, content_type)
        
        return result, evaluation, rewrite

    def generate_content(self, *args, **kwargs) -> tuple[WritingContent, ContentEvaluation]:
        """Synchronous wrapper for generate_content_async"""
        return asyncio.run(self.generate_content_async(*args, **kwargs))

    def _save_to_file(self, content: WritingContent, topic: str, content_type: ContentType) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_topic = "".join(c if c.isalnum() else "_" for c in topic)
        content_id = f"{timestamp}_{sanitized_topic[:30]}"
        
        # Add appropriate file extension
        file_extensions = {
            ContentType.EMAIL: ".eml",
            ContentType.TWEET: ".txt",
            ContentType.TEXT_MESSAGE: ".txt",
            ContentType.LINKEDIN_POST: ".md",
            ContentType.INSTAGRAM_CAPTION: ".txt"
        }
        
        filepath = self.output_dir / content_type.value / f"{content_id}{file_extensions[content_type]}"
        
        # Save the content with metadata
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Topic: {topic}\n")
            f.write(f"Tone: {content.tone}\n")
            if content.word_count:
                f.write(f"Word count: {content.word_count}\n")
            f.write("\n---\n\n")
            f.write(content.content)
        
        print(f"✓ Content saved to: {filepath}")
        return content_id

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
            print(f"Original Content:")
            print("-------------------")
            print(rewrite.original_content)
            print("-------------------")
            print(f"Improved Content:")
            print("-------------------")
            print(rewrite.improved_content)
            print("-------------------")
            print(f"Changes Made:")
            print("-------------------")
            for change in rewrite.changes_made:
                print(f"• {change}")
            print("-------------------")
            print(f"Improvement Focus:")
            print("-------------------")
            for focus in rewrite.improvement_focus:
                print(f"• {focus}")
            print("-------------------")
        
        return result, evaluation, rewrite

    def interactive_generate(self):
        """Synchronous wrapper for interactive_generate_async"""
        return asyncio.run(self.interactive_generate_async())

    async def rewrite_content_async(
        self,
        original_content: WritingContent,
        evaluation: ContentEvaluation,
        content_type: ContentType
    ) -> ContentRewrite:
        print("\n✏️ Generating content rewrite based on evaluation...")
        
        # Prepare the evaluation summary
        eval_summary = {
            "clarity": {
                "score": evaluation.clarity.score,
                "suggestions": evaluation.clarity.suggestions
            },
            "engagement": {
                "score": evaluation.engagement.score,
                "suggestions": evaluation.engagement.suggestions
            },
            "tone_consistency": {
                "score": evaluation.tone_consistency.score,
                "suggestions": evaluation.tone_consistency.suggestions
            }
        }
        
        messages = [
            {"role": "system", "content": f"{self.system_prompts[content_type]}\nYou are now improving content based on specific evaluation feedback."},
            {"role": "user", "content": f"""
Original Content: {original_content.content}
Tone: {original_content.tone}
Content Type: {content_type.value}

Evaluation Feedback:
{json.dumps(eval_summary, indent=2)}

Please rewrite the content addressing the evaluation feedback while maintaining the original intent and tone.
"""}
        ]
        
        result = await self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            response_format=ContentRewrite
        )
        
        print("✓ Content rewrite completed")
        return result.choices[0].message.parsed

    def _save_rewrite_to_file(self, rewrite: ContentRewrite, original_id: str, content_type: ContentType):
        """Save the rewritten content alongside the original"""
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

# Example usage
if __name__ == "__main__":
    assistant = WritingAssistant()
    
    while True:
        assistant.interactive_generate()
        
        # Ask if user wants to generate more content
        if input("\nWould you like to generate more content? (y/n): ").lower() != 'y':
            print("\nThank you for using Writing Assistant!")
            break

from openai import AsyncOpenAI
from datetime import datetime
from pathlib import Path
import json
import asyncio
from models_final import ContentType, WritingContent, EvaluationScore, ContentEvaluation, ContentRewrite

class BaseEvaluationAgent:
    def __init__(self):
        self.client = AsyncOpenAI()
        
    @property
    def clarity_prompt(self) -> str:
        raise NotImplementedError
        
    @property
    def engagement_prompt(self) -> str:
        raise NotImplementedError
        
    @property
    def tone_consistency_prompt(self) -> str:
        raise NotImplementedError

    @property
    def originality_prompt(self) -> str:
        raise NotImplementedError

    @property
    def platform_fit_prompt(self) -> str:
        raise NotImplementedError

    async def evaluate_aspect(self, aspect: str, content: str, intended_tone: str) -> EvaluationScore:
        prompts = {
            "clarity": self.clarity_prompt,
            "engagement": self.engagement_prompt,
            "tone_consistency": self.tone_consistency_prompt,
            "originality": self.originality_prompt,
            "platform_fit": self.platform_fit_prompt
        }
        
        messages = [
            {"role": "system", "content": prompts[aspect]},
            {"role": "user", "content": f"Content: {content}"}
        ]
        
        if aspect == "tone_consistency":
            messages[1]["content"] += f"\nIntended Tone: {intended_tone}"

        completion = await self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            response_format=EvaluationScore
        )
        
        return completion.choices[0].message.parsed

class TweetEvaluationAgent(BaseEvaluationAgent):
    @property
    def clarity_prompt(self) -> str:
        return """Evaluate the clarity of this tweet. Consider:
- Is the message immediately clear?
- Does it use appropriate hashtags?
- Is it within character limits?
- Are abbreviations clear and appropriate?"""

    @property
    def engagement_prompt(self) -> str:
        return """Evaluate the tweet's engagement potential. Consider:
- Does it encourage interaction (likes, retweets, replies)?
- Is it timely and relevant?
- Does it use trending topics effectively?
- Does it have viral potential?"""

    @property
    def tone_consistency_prompt(self) -> str:
        return """Evaluate the tweet's tone consistency. Consider:
- Does it maintain the intended voice throughout?
- Is the tone appropriate for Twitter?
- Does it align with brand voice (if applicable)?
- Is it authentic and relatable?"""

    @property
    def originality_prompt(self) -> str:
        return """Evaluate the tweet's originality. Consider:
- Does it offer a unique perspective?
- Is it different from common Twitter content?
- Does it bring fresh insights?
- Is it creative and innovative?"""

    @property
    def platform_fit_prompt(self) -> str:
        return """Evaluate how well the tweet fits Twitter as a platform. Consider:
- Does it use Twitter-specific features effectively?
- Is it optimized for the Twitter audience?
- Does it follow platform best practices?
- Would it perform well in the Twitter environment?"""

class EmailEvaluationAgent(BaseEvaluationAgent):
    @property
    def clarity_prompt(self) -> str:
        return """Evaluate the email's clarity. Consider:
- Is the subject clear and compelling?
- Is the structure logical and easy to follow?
- Are paragraphs concise and well-organized?
- Is the call-to-action clear?"""

    @property
    def engagement_prompt(self) -> str:
        return """Evaluate the email's engagement potential. Consider:
- Does it grab attention from the start?
- Is the content relevant to the recipient?
- Does it maintain interest throughout?
- Is the call-to-action compelling?"""

    @property
    def tone_consistency_prompt(self) -> str:
        return """Evaluate the email's tone consistency. Consider:
- Is the tone appropriate for the audience?
- Does it maintain professionalism?
- Is it consistent from greeting to signature?
- Does it reflect the relationship with the recipient?"""

    @property
    def originality_prompt(self) -> str:
        return """Evaluate the email's originality. Consider:
- Does it stand out in a crowded inbox?
- Does it offer unique value?
- Is the approach fresh and innovative?
- Does it avoid common email clichés?"""

    @property
    def platform_fit_prompt(self) -> str:
        return """Evaluate how well the email fits email as a medium. Consider:
- Is it properly formatted for email clients?
- Does it respect email conventions?
- Is it scannable and well-structured?
- Does it use email-appropriate language?"""

class TextMessageEvaluationAgent(BaseEvaluationAgent):
    @property
    def clarity_prompt(self) -> str:
        return """Evaluate the text message's clarity. Consider:
- Is the message direct and clear?
- Are abbreviations appropriate and understandable?
- Is it concise yet complete?
- Does it avoid ambiguity?"""

    @property
    def engagement_prompt(self) -> str:
        return """Evaluate the text message's effectiveness. Consider:
- Does it prompt a response if needed?
- Is it appropriate for the medium?
- Is it engaging without being overwhelming?
- Does it respect the recipient's time?"""

    @property
    def tone_consistency_prompt(self) -> str:
        return """Evaluate the text message's tone. Consider:
- Is it appropriate for texting?
- Does it match the relationship context?
- Is it consistent throughout?
- Does it avoid potential misinterpretation?"""

    @property
    def originality_prompt(self) -> str:
        return """Evaluate the text message's originality. Consider:
- Does it offer a unique perspective?
- Is it different from common text messages?
- Does it bring fresh insights?
- Is it creative and innovative?"""

    @property
    def platform_fit_prompt(self) -> str:
        return """Evaluate how well the text message fits texting as a medium. Consider:
- Is it properly formatted for texting clients?
- Does it respect texting conventions?
- Is it scannable and well-structured?
- Does it use texting-appropriate language?"""

class LinkedInEvaluationAgent(BaseEvaluationAgent):
    @property
    def clarity_prompt(self) -> str:
        return """Evaluate the LinkedIn post's clarity. Consider:
- Is the professional message clear?
- Is it well-structured for the platform?
- Are industry terms used appropriately?
- Does it maintain professional standards?"""

    @property
    def engagement_prompt(self) -> str:
        return """Evaluate the LinkedIn post's engagement potential. Consider:
- Does it encourage professional discussion?
- Is it relevant to the target audience?
- Does it provide value to professionals?
- Will it generate meaningful interactions?"""

    @property
    def tone_consistency_prompt(self) -> str:
        return """Evaluate the LinkedIn post's tone consistency. Consider:
- Does it maintain professional tone?
- Is it appropriate for business networking?
- Does it reflect industry standards?
- Is it consistently authoritative yet approachable?"""

    @property
    def originality_prompt(self) -> str:
        return """Evaluate the LinkedIn post's originality. Consider:
- Does it offer a unique perspective?
- Is it different from common LinkedIn content?
- Does it bring fresh insights?
- Is it creative and innovative?"""

    @property
    def platform_fit_prompt(self) -> str:
        return """Evaluate how well the LinkedIn post fits LinkedIn as a platform. Consider:
- Does it use LinkedIn-specific features effectively?
- Is it optimized for the LinkedIn audience?
- Does it follow platform best practices?
- Would it perform well in the LinkedIn environment?"""

class InstagramEvaluationAgent(BaseEvaluationAgent):
    @property
    def clarity_prompt(self) -> str:
        return """Evaluate the Instagram caption's clarity. Consider:
- Does it complement visual content?
- Are hashtags used effectively?
- Is the message clear and concise?
- Is it formatted for easy reading?"""

    @property
    def engagement_prompt(self) -> str:
        return """Evaluate the Instagram caption's engagement potential. Consider:
- Does it encourage likes and comments?
- Is it visually descriptive?
- Does it use trending elements effectively?
- Will it resonate with the Instagram audience?"""

    @property
    def tone_consistency_prompt(self) -> str:
        return """Evaluate the Instagram caption's tone consistency. Consider:
- Does it match the platform's style?
- Is it consistently engaging and authentic?
- Does it maintain brand voice?
- Is it appropriate for visual social media?"""

    @property
    def originality_prompt(self) -> str:
        return """Evaluate the Instagram caption's originality. Consider:
- Does it offer a unique perspective?
- Is it different from common Instagram captions?
- Does it bring fresh insights?
- Is it creative and innovative?"""

    @property
    def platform_fit_prompt(self) -> str:
        return """Evaluate how well the Instagram caption fits Instagram as a platform. Consider:
- Does it use Instagram-specific features effectively?
- Is it optimized for the Instagram audience?
- Does it follow platform best practices?
- Would it perform well in the Instagram environment?"""

class ContentEvaluator:
    def __init__(self, output_dir: Path):
        self.eval_dir = output_dir / "evaluations"
        self.eval_dir.mkdir(exist_ok=True)
        
        # Add client initialization
        self.client = AsyncOpenAI()
        
        # Initialize specialized evaluation agents
        self.evaluation_agents = {
            ContentType.TWEET: TweetEvaluationAgent(),
            ContentType.EMAIL: EmailEvaluationAgent(),
            ContentType.TEXT_MESSAGE: TextMessageEvaluationAgent(),
            ContentType.LINKEDIN_POST: LinkedInEvaluationAgent(),
            ContentType.INSTAGRAM_CAPTION: InstagramEvaluationAgent()
        }

    async def _evaluate_aspect(self, aspect: str, content: str, intended_tone: str, content_type: ContentType) -> EvaluationScore:
        print(f"🔍 Evaluating {aspect.replace('_', ' ')}...")
        
        # Get the appropriate evaluation agent
        agent = self.evaluation_agents[content_type]
        return await agent.evaluate_aspect(aspect, content, intended_tone)

    async def evaluate_content(self, content: str, intended_tone: str, content_type: ContentType) -> ContentEvaluation:
        print("\n📊 Starting parallel content evaluation...")
        
        tasks = {
            aspect: asyncio.create_task(self._evaluate_aspect(aspect, content, intended_tone, content_type))
            for aspect in ["clarity", "engagement", "tone_consistency", "originality", "platform_fit"]
        }
        
        results = await asyncio.gather(*tasks.values())
        evaluations = dict(zip(tasks.keys(), results))
        
        return ContentEvaluation(
            **evaluations,
            timestamp=datetime.now().isoformat()
        )

    async def rewrite_content(
        self,
        original_content: WritingContent,
        evaluation: ContentEvaluation,
        content_type: ContentType
    ) -> ContentRewrite:
        print("\n✏️ Generating content rewrite based on evaluation...")
        
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
            },
            "originality": {
                "score": evaluation.originality.score,
                "suggestions": evaluation.originality.suggestions
            },
            "platform_fit": {
                "score": evaluation.platform_fit.score,
                "suggestions": evaluation.platform_fit.suggestions
            }
        }
        
        messages = [
            {"role": "system", "content": "You are improving content based on specific evaluation feedback."},
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
        
        return result.choices[0].message.parsed

    def save_evaluation(self, evaluation: ContentEvaluation, content_id: str, content_type: ContentType):
        """Save evaluation results to a JSON file"""
        print(f"\n💾 Saving evaluation results...")
        eval_path = self.eval_dir / content_type.value
        eval_path.mkdir(exist_ok=True)
        
        filepath = eval_path / f"{content_id}_evaluation.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(evaluation.model_dump(), f, indent=2)
        print(f"✓ Evaluation saved to: {filepath}")

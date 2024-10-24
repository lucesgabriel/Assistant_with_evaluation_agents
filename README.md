# AI Writing & Evaluation System

A sophisticated content generation and evaluation system that specializes in creating and analyzing content across multiple platforms.

## Overview

This system combines specialized writing agents with comprehensive evaluation capabilities to generate, assess, and improve content for different platforms:

- Tweets
- Emails
- Text Messages
- LinkedIn Posts
- Instagram Captions

## Key Features

### 1. Platform-Specific Content Generation
- Dedicated writing agents for each platform
- Context-aware content creation
- Tone-sensitive writing
- Platform-appropriate formatting

### 2. Multi-Dimensional Content Evaluation
Each piece of content is evaluated across five key dimensions:

- **Clarity**: Assesses readability and message comprehension
- **Engagement**: Measures potential for audience interaction
- **Tone Consistency**: Evaluates voice and style alignment
- **Originality**: Analyzes uniqueness and creativity
- **Platform Fit**: Assesses appropriateness for the medium

### 3. Automated Content Improvement
- Automatic detection of content needing improvement
- AI-powered content rewriting
- Detailed improvement suggestions
- Before/after comparisons

### 4. Content Management
- Organized file storage by platform
- Evaluation results in JSON format
- Rewrite history tracking
- Unique content identifiers

## Getting Started

1. Install the required packages:
   ```
   pip install openai pydantic python-dotenv
   ```

2. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Run the main script:
   ```
   python writing_agents_final.py
   ```

## Usage

The system provides an interactive interface for content generation:

1. Choose the content type (Tweet, Email, Text Message, LinkedIn Post, or Instagram Caption)
2. Enter the topic
3. Specify the desired tone
4. Provide any additional context (optional)

The system will then:
- Generate the content
- Evaluate it across multiple dimensions
- Automatically rewrite if necessary
- Save all outputs to appropriate directories

## Project Structure

- `writing_agents_final.py`: Main script with writing agents and assistant logic
- `models_final.py`: Pydantic models for data structures
- `content_evaluator_final.py`: Content evaluation and rewriting logic


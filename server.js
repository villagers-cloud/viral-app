/**
 * Viral Content & Script Generator - Backend
 * Node.js / Express server that securely proxies requests to the
 * Groq API (OpenAI-compatible chat completions endpoint).
 *
 * The GROQ_API_KEY is read from environment variables ONLY.
 * It is never sent to, or exposed on, the frontend.
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';
const MODEL = 'llama-3.3-70b-versatile';

/* ------------------------------------------------------------------ */
/* Prompt builders                                                     */
/* ------------------------------------------------------------------ */

function buildSystemPrompt(language) {
  return (
    `You are an expert viral social media content strategist, copywriter, and scriptwriter. ` +
    `You always respond with valid, minified JSON only — no markdown code fences, no commentary, ` +
    `no text before or after the JSON object. All generated content (values) must be written fluently ` +
    `and naturally in ${language}, using native script/spelling conventions for that language. ` +
    `The JSON keys themselves must always remain exactly as instructed, in English.`
  );
}

function buildFullPrompt(topic, language, tone) {
  return `Create complete viral short-form video/social content for the following:

Topic / Product: "${topic}"
Tone/Style: ${tone}
Output Language: ${language}

Return ONLY a JSON object with exactly these keys and nothing else:
{
  "idea": "A short, punchy viral content idea/concept (2-4 sentences) explaining the angle and why it will perform well",
  "hook": "A single, powerful attention-grabbing opening line/hook for the first 3 seconds of the video",
  "script": "A full video script with a clear structure: Hook, Body, Call-To-Action. Use \\n for line breaks and label each section",
  "title": "One catchy, highly clickable title for the video or post",
  "description": "An SEO-optimized description for the video/post (2-3 sentences, naturally keyword rich)",
  "hashtags": "10 to 15 relevant, trending, viral hashtags separated by single spaces, each starting with #"
}

Rules:
- Every value must be written entirely in ${language}.
- Match the requested tone: ${tone}.
- Do not wrap the JSON in markdown/code fences.
- Do not include any explanation outside the JSON object.`;
}

const SECTION_INSTRUCTIONS = {
  idea: 'Regenerate ONLY a brand new "idea" — a short, punchy viral content idea/concept (2-4 sentences), different from before.',
  hook_script: 'Regenerate ONLY a fresh "hook" (a powerful attention-grabbing opening line) and a fresh "script" (full video script with Hook, Body, and Call-To-Action sections, using \\n for line breaks).',
  title: 'Regenerate ONLY a fresh "title" — one new catchy, clickable title, different from before.',
  description: 'Regenerate ONLY a fresh "description" — a new SEO-optimized description (2-3 sentences).',
  hashtags: 'Regenerate ONLY a fresh set of "hashtags" — 10 to 15 relevant, trending hashtags separated by single spaces, each starting with #.'
};

const SECTION_SHAPES = {
  idea: '{"idea": "..."}',
  hook_script: '{"hook": "...", "script": "..."}',
  title: '{"title": "..."}',
  description: '{"description": "..."}',
  hashtags: '{"hashtags": "..."}'
};

function buildSectionPrompt(section, topic, language, tone, context) {
  const instruction = SECTION_INSTRUCTIONS[section];
  const shape = SECTION_SHAPES[section];

  if (!instruction || !shape) {
    throw new Error(`Unknown section: ${section}`);
  }

  return `Topic / Product: "${topic}"
Tone/Style: ${tone}
Output Language: ${language}

Existing content already generated for this topic (for context only — do NOT repeat it verbatim, produce something new but consistent in theme):
${JSON.stringify(context || {})}

${instruction}

Return ONLY a JSON object in exactly this shape: ${shape}

Rules:
- Every value must be written entirely in ${language}.
- Match the requested tone: ${tone}.
- Do not wrap the JSON in markdown/code fences.
- Do not include any explanation outside the JSON object.`;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

function extractJson(text) {
  // Try a direct parse first.
  try {
    return JSON.parse(text);
  } catch (e) {
    // Fall back to pulling the first {...} block out of the text,
    // in case the model added stray whitespace/fences.
    const match = text.match(/\{[\s\S]*\}/);
    if (match) {
      return JSON.parse(match[0]);
    }
    throw new Error('Could not parse AI response as JSON.');
  }
}

const VALID_SECTIONS = new Set(['all', 'idea', 'hook_script', 'title', 'description', 'hashtags']);

/* ------------------------------------------------------------------ */
/* Routes                                                               */
/* ------------------------------------------------------------------ */

app.post('/api/generate', async (req, res) => {
  try {
    const { topic, language, tone, section, context } = req.body || {};

    const cleanTopic = typeof topic === 'string' ? topic.trim() : '';
    const cleanLanguage = typeof language === 'string' && language.trim() ? language.trim() : 'English';
    const cleanTone = typeof tone === 'string' && tone.trim() ? tone.trim() : 'Viral';
    const cleanSection = VALID_SECTIONS.has(section) ? section : 'all';

    if (!cleanTopic) {
      return res.status(400).json({ error: 'Please provide a topic or product name.' });
    }

    if (!process.env.GROQ_API_KEY) {
      return res.status(500).json({
        error: 'Server misconfiguration: GROQ_API_KEY environment variable is not set.'
      });
    }

    const userPrompt =
      cleanSection === 'all'
        ? buildFullPrompt(cleanTopic, cleanLanguage, cleanTone)
        : buildSectionPrompt(cleanSection, cleanTopic, cleanLanguage, cleanTone, context);

    const payload = {
      model: MODEL,
      messages: [
        { role: 'system', content: buildSystemPrompt(cleanLanguage) },
        { role: 'user', content: userPrompt }
      ],
      temperature: 0.95,
      top_p: 0.9,
      max_tokens: 1600
    };

    const response = await axios.post(GROQ_API_URL, payload, {
      headers: {
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
        'Content-Type': 'application/json',
        Accept: 'application/json'
      },
      timeout: 60000
    });

    const rawContent = response.data?.choices?.[0]?.message?.content;

    if (!rawContent) {
      return res.status(502).json({ error: 'Empty response received from the AI provider.' });
    }

    const parsed = extractJson(rawContent);
    return res.json(parsed);
  } catch (err) {
    const providerMessage = err.response?.data?.error?.message || err.response?.data?.message;
    console.error('Generation error:', providerMessage || err.message);

    return res.status(err.response?.status || 500).json({
      error: 'Failed to generate content. Please try again.',
      details: providerMessage || err.message
    });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', keyConfigured: Boolean(process.env.GROQ_API_KEY) });
});

// Fallback: serve the frontend for any other GET route.
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 Viral Content Generator running at http://localhost:${PORT}`);
  if (!process.env.GROQ_API_KEY) {
    console.warn('⚠️  Warning: GROQ_API_KEY is not set. Set it in a .env file or your environment.');
  }
});

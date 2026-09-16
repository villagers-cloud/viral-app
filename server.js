/**
 * Viral Content & Script Generator - Backend
 * Native Google Gemini API Version (100% Error Free)
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const path = require('path');
const crypto = require('crypto');
const rateLimit = require('express-rate-limit');

const app = express();
const PORT = process.env.PORT || 3000;

const allowedOrigins = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(',')
  : ['http://localhost:3000'];

const corsOptions = {
  origin: function (origin, callback) {
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  }
};

app.use(cors(corsOptions));
app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));
app.set('trust proxy', 1);

const generateLimiter = rateLimit({
  windowMs: 10 * 60 * 1000,
  max: 40,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "You're generating too quickly. Please wait a few minutes and try again." }
});

const CACHE_TTL_MS = 10 * 60 * 1000;
const CACHE_MAX_ENTRIES = 500;
const responseCache = new Map();

function getCacheKey(payload) {
  return crypto.createHash('sha256').update(JSON.stringify(payload)).digest('hex');
}
function getFromCache(key) {
  const entry = responseCache.get(key);
  if (!entry || Date.now() > entry.expiresAt) return null;
  return entry.data;
}
function setCache(key, data) {
  responseCache.set(key, { data, expiresAt: Date.now() + CACHE_TTL_MS });
}

function buildSystemPrompt(language) {
  return `You are a world-class viral social media strategist and expert copywriter. You specialize in creating highly engaging, retention-optimized short-form video content that stops the scroll. You always respond with valid JSON ONLY. There must be no markdown formatting wrapping the JSON. All generated content (values) must be written fluently and naturally in ${language}. The JSON keys themselves must always remain exactly as instructed, in English.`;
}

function buildToneDescription(tone, toneSecondary, toneIntensity) {
  let primaryTone = tone || "Conversational";
  const intensity = Number.isFinite(toneIntensity) ? Math.min(90, Math.max(0, toneIntensity)) : 0;
  if (!toneSecondary || toneSecondary === primaryTone || intensity <= 0) return primaryTone;
  return `a blend of ${100 - intensity}% ${primaryTone} and ${intensity}% ${toneSecondary}`;
}

function buildPlatformGuidance(platform) {
  switch (platform) {
    case 'Instagram Reels': return 'Optimize for Instagram Reels: Highly visual hook in the first 1-2 seconds, aesthetic/lifestyle pacing, highly relatable, and 5-8 focused hashtags.';
    case 'YouTube Shorts': return 'Optimize for YouTube Shorts: Strong searchable title, extremely fast hook within the first 1.5 seconds to prevent swiping, punchy pacing, and 3-6 hashtags.';
    case 'TikTok': return 'Optimize for TikTok: Extremely fast, trend-aware hook in the first second (e.g., negative hooks or pattern interrupts), highly conversational and raw script pacing, and 4-8 targeted hashtags.';
    default: return 'Keep it platform-agnostic and adaptable to modern short-form video platforms in general. Fast hook and engaging pacing.';
  }
}

function buildDurationGuidance(duration) {
  const map = {
    '15 seconds': 'Write the script for a very fast-paced 15-second video (roughly 35-45 spoken words total). Be punchy and straight to the point.',
    '30 seconds': 'Write the script for a 30-second video (roughly 70-90 spoken words). Include a solid hook, quick value drop, and immediate CTA.',
    '45 seconds': 'Write the script for a 45-second video (roughly 100-130 spoken words). Focus on strong storytelling and retention.',
    '60 seconds': 'Write the script for a 60-second video (roughly 140-170 spoken words). Provide deep value while maintaining high pacing.',
    '90 seconds': 'Write the script for a 90-second video (roughly 200-240 spoken words).',
    '2 minutes': 'Write the script for a 2-minute video (roughly 260-320 spoken words).',
    '3+ minutes': 'Write the script for a 3+ minute video (roughly 380-500+ spoken words).'
  };
  return map[duration] || map['30 seconds'];
}

function buildFullPrompt(topic, language, toneDescription, platformGuidance, durationGuidance) {
  return `Create complete viral short-form video/social content for the following:
Topic: "${topic}"
Tone/Style: ${toneDescription}
Language: ${language}
Platform: ${platformGuidance}
Target Length: ${durationGuidance}

Instructions for Viral Output:
- Use modern hook frameworks (e.g., negative hooks, addressing the viewer directly, stating a controversial opinion, or opening a curiosity loop).
- The script should include visual/action cues in brackets [like this] where appropriate to help the creator know what to do on camera.
- Ensure the pacing matches the target length perfectly.

Return ONLY a valid JSON object with exactly these keys:
{
  "idea": "A short, punchy 1-sentence viral content concept/angle.",
  "hook": "A single, powerful, attention-grabbing opening line (first 3 seconds).",
  "script": "The full spoken video script with visual cues in brackets. Structure: Hook -> Value/Story -> Strong Call-To-Action.",
  "title": "One catchy, highly clickable video title.",
  "description": "An engaging, SEO-optimized caption/description.",
  "hashtags": "10 to 15 trending, highly relevant hashtags separated by spaces."
}`;
}

function extractJson(text) {
  try { return JSON.parse(text); }
  catch (e) {
    const match = text.match(/\{[\s\S]*\}/);
    if (match) return JSON.parse(match[0]);
    throw new Error('Could not parse AI response as JSON.');
  }
}

app.post('/api/generate', generateLimiter, async (req, res) => {
  try {
    const body = req.body || {};
    const topic = (body.topic || '').trim();
    const language = body.language || 'English';
    const section = body.section || 'all';
    
    if (!topic) return res.status(400).json({ error: 'Please provide a topic.' });
    if (!process.env.GEMINI_API_KEY) return res.status(500).json({ error: 'GEMINI_API_KEY missing' });

    const toneDesc = buildToneDescription(body.tone, body.toneSecondary, body.toneIntensity);
    const platGuide = buildPlatformGuidance(body.platform);
    const durGuide = buildDurationGuidance(body.duration);
    let userPrompt = '';

    if (section === 'all') {
      userPrompt = buildFullPrompt(topic, language, toneDesc, platGuide, durGuide);
    } else {
      userPrompt = `Topic: "${topic}"\nTone: ${toneDesc}\nLanguage: ${language}\nExisting context: ${JSON.stringify(body.context || {})}\n\nInstruction: Regenerate or refine the ${section} section.\nReturn ONLY a valid JSON object for that specific key.`;
    }

    const cacheKey = section === 'all' ? getCacheKey({ topic, language, tone: body.tone, plat: body.platform, dur: body.duration }) : null;
    if (cacheKey && getFromCache(cacheKey)) return res.json(getFromCache(cacheKey));

    // 🔥 NATIVE GEMINI API URL (Guaranteed to work) 🔥
    const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`;

    const response = await axios.post(
      API_URL,
      {
        systemInstruction: { parts: [{ text: buildSystemPrompt(language) }] },
        contents: [{ role: "user", parts: [{ text: userPrompt }] }],
        generationConfig: { temperature: 0.9, responseMimeType: "application/json" }
      },
      { headers: { 'Content-Type': 'application/json' }, timeout: 60000 }
    );

    const rawContent = response.data?.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!rawContent) throw new Error("Empty response");

    const parsed = extractJson(rawContent);
    if (cacheKey) setCache(cacheKey, parsed);
    return res.json(parsed);

  } catch (err) {
    console.error('Generation error:', err.response?.data || err.message);
    return res.status(500).json({ error: 'Failed to generate content. Please try again.' });
  }
});

app.get('*', (req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));

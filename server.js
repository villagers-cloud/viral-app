/**
 * Viral Content & Script Generator - Backend
 * Node.js / Express server that securely proxies requests to the
 * Groq API (OpenAI-compatible chat completions endpoint).
 *
 * The GROQ_API_KEY is read from environment variables ONLY.
 * It is never sent to, or exposed on, the frontend.
 *
 * Features in this file:
 *  - Retry with exponential backoff on Groq 429 rate-limit errors
 *  - Model fallback chain (tries next model if one fails/exhausts retries)
 *  - In-memory response caching for identical full-generation requests
 *  - Rate limiting on /api/generate (per IP)
 *  - Input validation & length limits
 *  - Platform-specific guidance (Instagram Reels / YouTube Shorts / TikTok)
 *  - Video duration guidance (controls script length/pacing)
 *  - Tone blending (primary + optional secondary tone with intensity)
 *  - "Refine with instructions" — freeform edit instruction for one field
 *  - "Variations" — 3 alternative options for Idea or Hook
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

app.use(cors());
app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';

// Model fallback chain: primary model first, then fallbacks used automatically
// if the primary fails outright or exhausts its retries on rate limits.
const MODEL = 'openai/gpt-oss-20b';

/* ------------------------------------------------------------------ */
/* Validation constants                                                */
/* ------------------------------------------------------------------ */

const MAX_TOPIC_LENGTH = 200;
const MAX_INSTRUCTION_LENGTH = 300;

const ALLOWED_LANGUAGES = new Set([
  'English', 'Hindi', 'Gujarati', 'Marathi', 'Tamil', 'Telugu',
  'Bengali', 'Kannada', 'Malayalam', 'Punjabi', 'Odia', 'Urdu'
]);

const ALLOWED_TONES = new Set(['Viral', 'Funny', 'Professional', 'Educational']);

const ALLOWED_PLATFORMS = new Set(['General', 'Instagram Reels', 'YouTube Shorts', 'TikTok']);

const ALLOWED_DURATIONS = new Set([
  '15 seconds', '30 seconds', '45 seconds', '60 seconds',
  '90 seconds', '2 minutes', '3+ minutes'
]);

const VALID_SECTIONS = new Set([
  'all', 'idea', 'hook_script', 'title', 'description', 'hashtags',
  'idea_variations', 'hook_variations', 'refine'
]);

const REFINE_TARGETS = new Set(['idea', 'hook', 'script', 'title', 'description', 'hashtags']);

/* ------------------------------------------------------------------ */
/* Rate limiting                                                       */
/* ------------------------------------------------------------------ */

// If deployed behind a reverse proxy (Render, Heroku, Vercel, etc.),
// uncomment the line below so express-rate-limit reads the real client IP.
// app.set('trust proxy', 1);

const generateLimiter = rateLimit({
  windowMs: 10 * 60 * 1000, // 10 minutes
  max: 40, // 40 requests per IP per window (covers regenerate/refine/variations too)
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "You're generating too quickly. Please wait a few minutes and try again." }
});

/* ------------------------------------------------------------------ */
/* In-memory response cache (full-generation requests only)            */
/* ------------------------------------------------------------------ */

const CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes
const CACHE_MAX_ENTRIES = 500;
const responseCache = new Map(); // key -> { data, expiresAt }

function getCacheKey(payload) {
  return crypto.createHash('sha256').update(JSON.stringify(payload)).digest('hex');
}

function getFromCache(key) {
  const entry = responseCache.get(key);
  if (!entry) return null;
  if (Date.now() > entry.expiresAt) {
    responseCache.delete(key);
    return null;
  }
  return entry.data;
}

function setCache(key, data) {
  responseCache.set(key, { data, expiresAt: Date.now() + CACHE_TTL_MS });
  if (responseCache.size > CACHE_MAX_ENTRIES) {
    const oldestKey = responseCache.keys().next().value;
    responseCache.delete(oldestKey);
  }
}

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

function buildToneDescription(tone, toneSecondary, toneIntensity) {
  const intensity = Number.isFinite(toneIntensity) ? Math.min(90, Math.max(0, toneIntensity)) : 0;
  if (!toneSecondary || toneSecondary === tone || intensity <= 0) {
    return tone;
  }
  const primaryPct = 100 - intensity;
  return `a blend of ${primaryPct}% ${tone} and ${intensity}% ${toneSecondary}`;
}

function buildPlatformGuidance(platform) {
  switch (platform) {
    case 'Instagram Reels':
      return 'Optimize for Instagram Reels: fast hook in the first 1-2 seconds, vertical short-form pacing, 5-8 focused hashtags mixing broad and niche tags, casual on-brand caption style.';
    case 'YouTube Shorts':
      return 'Optimize for YouTube Shorts: strong searchable title (SEO matters more here), hook within the first 2 seconds, description written for YouTube search discovery, 3-6 hashtags including #Shorts.';
    case 'TikTok':
      return 'Optimize for TikTok: extremely fast, trend-aware hook in the first second, conversational script pacing, 4-8 hashtags mixing trending and niche tags, caption that feels native to TikTok culture.';
    default:
      return 'Keep it platform-agnostic and adaptable to short-form video platforms in general.';
  }
}

function buildDurationGuidance(duration) {
  const map = {
    '15 seconds': 'Write the script for a 15-second video — roughly 35-45 spoken words total. One quick beat before the CTA. Extremely tight.',
    '30 seconds': 'Write the script for a 30-second video — roughly 70-90 spoken words. Quick hook, 1-2 key points, then a CTA.',
    '45 seconds': 'Write the script for a 45-second video — roughly 100-130 spoken words. Hook, 2-3 points, then a CTA.',
    '60 seconds': 'Write the script for a 60-second video — roughly 140-170 spoken words. Hook, up to 3 points with brief detail, then a CTA.',
    '90 seconds': 'Write the script for a 90-second video — roughly 200-240 spoken words. Hook, 3-4 points with more detail/storytelling, then a CTA.',
    '2 minutes': 'Write the script for a 2-minute video — roughly 260-320 spoken words. Hook, a fuller narrative with multiple points, then a CTA.',
    '3+ minutes': 'Write the script for a 3+ minute video — roughly 380-500+ spoken words. Hook, in-depth narrative across multiple sections, then a strong CTA.'
  };
  return map[duration] || map['30 seconds'];
}

function computeMaxTokens(section, duration, refineTarget) {
  const isScriptHeavy =
    section === 'all' ||
    section === 'hook_script' ||
    (section === 'refine' && refineTarget === 'script');

  if (!isScriptHeavy) return 700;

  const map = {
    '15 seconds': 1400,
    '30 seconds': 1600,
    '45 seconds': 1700,
    '60 seconds': 1900,
    '90 seconds': 2200,
    '2 minutes': 2500,
    '3+ minutes': 3000
  };
  return map[duration] || 1600;
}

function buildFullPrompt(topic, language, toneDescription, platformGuidance, durationGuidance) {
  return `Create complete viral short-form video/social content for the following:

Topic / Product: "${topic}"
Tone/Style: ${toneDescription}
Output Language: ${language}
Platform Guidance: ${platformGuidance}
Script Length Guidance: ${durationGuidance}

Return ONLY a JSON object with exactly these keys and nothing else:
{
  "idea": "A short, punchy viral content idea/concept (2-4 sentences) explaining the angle and why it will perform well",
  "hook": "A single, powerful attention-grabbing opening line/hook for the first few seconds of the video",
  "script": "A full video script with a clear structure: Hook, Body, Call-To-Action. Use \\n for line breaks and label each section. Follow the Script Length Guidance above closely",
  "title": "One catchy, highly clickable title for the video or post",
  "description": "An SEO-optimized description for the video/post (2-3 sentences, naturally keyword rich)",
  "hashtags": "10 to 15 relevant, trending, viral hashtags separated by single spaces, each starting with #"
}

Rules:
- Every value must be written entirely in ${language}.
- Match the requested tone/style: ${toneDescription}.
- Follow the platform guidance and script length guidance above.
- Do not wrap the JSON in markdown/code fences.
- Do not include any explanation outside the JSON object.`;
}

const SECTION_INSTRUCTIONS = {
  idea: 'Regenerate ONLY a brand new "idea" — a short, punchy viral content idea/concept (2-4 sentences), different from before.',
  hook_script: 'Regenerate ONLY a fresh "hook" (a powerful attention-grabbing opening line) and a fresh "script" (full video script with Hook, Body, and Call-To-Action sections, using \\n for line breaks, following the Script Length Guidance closely).',
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

const FIELD_SHAPES = {
  idea: '{"idea": "..."}',
  hook: '{"hook": "..."}',
  script: '{"script": "..."}',
  title: '{"title": "..."}',
  description: '{"description": "..."}',
  hashtags: '{"hashtags": "..."}'
};

function buildSectionPrompt(section, topic, language, toneDescription, platformGuidance, durationGuidance, context) {
  const instruction = SECTION_INSTRUCTIONS[section];
  const shape = SECTION_SHAPES[section];

  if (!instruction || !shape) {
    throw new Error(`Unknown section: ${section}`);
  }

  return `Topic / Product: "${topic}"
Tone/Style: ${toneDescription}
Output Language: ${language}
Platform Guidance: ${platformGuidance}
Script Length Guidance: ${durationGuidance}

Existing content already generated for this topic (for context only — do NOT repeat it verbatim, produce something new but consistent in theme):
${JSON.stringify(context || {})}

${instruction}

Return ONLY a JSON object in exactly this shape: ${shape}

Rules:
- Every value must be written entirely in ${language}.
- Match the requested tone/style: ${toneDescription}.
- Do not wrap the JSON in markdown/code fences.
- Do not include any explanation outside the JSON object.`;
}

function buildVariationsPrompt(kind, topic, language, toneDescription, platformGuidance, durationGuidance, context) {
  const label = kind === 'idea' ? 'viral content ideas' : 'video hooks (opening lines)';

  return `Topic / Product: "${topic}"
Tone/Style: ${toneDescription}
Output Language: ${language}
Platform Guidance: ${platformGuidance}
${kind === 'hook' ? `Script Length Guidance: ${durationGuidance}` : ''}

Existing content for context (do not repeat verbatim, stay thematically consistent):
${JSON.stringify(context || {})}

Generate exactly 3 distinct, fresh ${label} for this topic. Each option must take a noticeably different angle/approach so the user has real choices between them.

Return ONLY a JSON object in exactly this shape: {"options": ["...", "...", "..."]}

Rules:
- Every value must be written entirely in ${language}.
- Match the requested tone/style: ${toneDescription}.
- Do not wrap the JSON in markdown/code fences.
- Do not include any explanation outside the JSON object.`;
}

function buildRefinePrompt(target, topic, language, toneDescription, platformGuidance, durationGuidance, context, instruction) {
  const shape = FIELD_SHAPES[target];

  return `Topic / Product: "${topic}"
Tone/Style: ${toneDescription}
Output Language: ${language}
Platform Guidance: ${platformGuidance}
Script Length Guidance: ${durationGuidance}

Current content for context:
${JSON.stringify(context || {})}

The user gave this specific instruction for revising the "${target}" field:
"${instruction}"

Rewrite ONLY the "${target}" field, following the user's instruction closely while staying true to the topic, tone, and (if the field is "script") the Script Length Guidance above.

Return ONLY a JSON object in exactly this shape: ${shape}

Rules:
- Every value must be written entirely in ${language}.
- Do not wrap the JSON in markdown/code fences.
- Do not include any explanation outside the JSON object.`;
}

/* ------------------------------------------------------------------ */
/* Groq call with retry (429 backoff) + model fallback                 */
/* ------------------------------------------------------------------ */

async function callGroqWithRetryAndFallback(payloadBase) {
  const MAX_RETRIES_PER_MODEL = 2; // extra attempts after the first, on 429 only
  let lastError = null;

  for (const model of MODEL_CHAIN) {
    let attempt = 0;
    while (attempt <= MAX_RETRIES_PER_MODEL) {
      try {
        const response = await axios.post(
          GROQ_API_URL,
          { ...payloadBase, model },
          {
            headers: {
              Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
              'Content-Type': 'application/json',
              Accept: 'application/json'
            },
            timeout: 60000
          }
        );
        return { response, modelUsed: model };
      } catch (err) {
        lastError = err;
        const status = err.response?.status;

        if (status === 429 && attempt < MAX_RETRIES_PER_MODEL) {
          const retryAfterHeader = err.response?.headers?.['retry-after'];
          const backoffMs = retryAfterHeader
            ? parseFloat(retryAfterHeader) * 1000
            : 800 * Math.pow(2, attempt); // 800ms, 1600ms, ...
          await new Promise((resolve) => setTimeout(resolve, backoffMs));
          attempt += 1;
          continue;
        }

        // Non-429 error, or retries exhausted on this model -> try next model.
        break;
      }
    }
  }

  throw lastError;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

function extractJson(text) {
  try {
    return JSON.parse(text);
  } catch (e) {
    const match = text.match(/\{[\s\S]*\}/);
    if (match) {
      return JSON.parse(match[0]);
    }
    throw new Error('Could not parse AI response as JSON.');
  }
}

function validateAndCleanInputs(body) {
  const errors = [];

  const topic = typeof body.topic === 'string' ? body.topic.trim() : '';
  if (!topic) errors.push('Please provide a topic or product name.');
  if (topic.length > MAX_TOPIC_LENGTH) {
    errors.push(`Topic must be ${MAX_TOPIC_LENGTH} characters or fewer.`);
  }

  const language = ALLOWED_LANGUAGES.has(body.language) ? body.language : 'English';
  const tone = ALLOWED_TONES.has(body.tone) ? body.tone : 'Viral';
  const toneSecondary = ALLOWED_TONES.has(body.toneSecondary) ? body.toneSecondary : null;
  const toneIntensity = Number.isFinite(Number(body.toneIntensity)) ? Number(body.toneIntensity) : 0;
  const platform = ALLOWED_PLATFORMS.has(body.platform) ? body.platform : 'General';
  const duration = ALLOWED_DURATIONS.has(body.duration) ? body.duration : '30 seconds';
  const section = VALID_SECTIONS.has(body.section) ? body.section : 'all';

  return {
    errors,
    topic: topic.slice(0, MAX_TOPIC_LENGTH),
    language,
    tone,
    toneSecondary,
    toneIntensity,
    platform,
    duration,
    section
  };
}

/* ------------------------------------------------------------------ */
/* Routes                                                               */
/* ------------------------------------------------------------------ */

app.post('/api/generate', generateLimiter, async (req, res) => {
  try {
    const body = req.body || {};
    const {
      errors, topic, language, tone, toneSecondary, toneIntensity,
      platform, duration, section
    } = validateAndCleanInputs(body);

    if (errors.length) {
      return res.status(400).json({ error: errors[0] });
    }

    if (!process.env.GROQ_API_KEY) {
      return res.status(500).json({
        error: 'Server misconfiguration: GROQ_API_KEY environment variable is not set.'
      });
    }

    const toneDescription = buildToneDescription(tone, toneSecondary, toneIntensity);
    const platformGuidance = buildPlatformGuidance(platform);
    const durationGuidance = buildDurationGuidance(duration);
    const context = body.context;

    let userPrompt;
    let cacheKey = null;
    let refineTarget = null;

    if (section === 'all') {
      userPrompt = buildFullPrompt(topic, language, toneDescription, platformGuidance, durationGuidance);
      cacheKey = getCacheKey({ topic, language, tone, toneSecondary, toneIntensity, platform, duration, section });
    } else if (section === 'idea_variations' || section === 'hook_variations') {
      const kind = section === 'idea_variations' ? 'idea' : 'hook';
      userPrompt = buildVariationsPrompt(kind, topic, language, toneDescription, platformGuidance, durationGuidance, context);
    } else if (section === 'refine') {
      refineTarget = REFINE_TARGETS.has(body.refineTarget) ? body.refineTarget : null;
      const instruction = typeof body.refineInstruction === 'string'
        ? body.refineInstruction.trim().slice(0, MAX_INSTRUCTION_LENGTH)
        : '';

      if (!refineTarget || !instruction) {
        return res.status(400).json({ error: 'Please provide an instruction and a valid field to refine.' });
      }

      userPrompt = buildRefinePrompt(refineTarget, topic, language, toneDescription, platformGuidance, durationGuidance, context, instruction);
    } else {
      userPrompt = buildSectionPrompt(section, topic, language, toneDescription, platformGuidance, durationGuidance, context);
    }

    if (cacheKey) {
      const cached = getFromCache(cacheKey);
      if (cached) {
        return res.json(cached);
      }
    }

    const payloadBase = {
      messages: [
        { role: 'system', content: buildSystemPrompt(language) },
        { role: 'user', content: userPrompt }
      ],
      temperature: 0.95,
      top_p: 0.9,
      max_tokens: computeMaxTokens(section, duration, refineTarget)
    };

    const { response } = await callGroqWithRetryAndFallback(payloadBase);
    const rawContent = response.data?.choices?.[0]?.message?.content;

    if (!rawContent) {
      return res.status(502).json({ error: 'Empty response received from the AI provider.' });
    }

    const parsed = extractJson(rawContent);

    if (cacheKey) {
      setCache(cacheKey, parsed);
    }

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

---
name: documentary-storyboard
description: Turns a short topic description into a ~10 minute documentary narrative split into storyboard scenes, each with a script, an image-generation visual prompt, and motion direction. Use when the user asks for a documentary narrative, storyboard, scene breakdown, or video script about a person, event, invention, or era.
---

# Documentary Storyboard

Take a short explanation of what the narrative should be about and produce a ~10 minute narrated documentary as a numbered list of storyboard scenes.

## Input

The user gives a sample explanation of the topic (a person, event, invention, era, etc.).
If key facts are missing, use well-known, verifiable facts. Never invent dates, quotes, or events.

## Length

- Target about 1,500 words of total script (about 10 minutes of narration at ~150 words per minute).
- Use roughly 25 to 35 scenes.
- Each scene's script should be about 30 to 60 words (roughly 12 to 24 seconds of narration).

## Narrative rules

- Only include necessary, important, or exciting details. Cut anything mundane, repetitive, or boring.
- Open with a hook that puts the viewer in a moment or asks them to imagine something. Example: "Imagine standing on solid ground... and being told that a giant metal ship could float above your head."
- Build a clear arc: hook, setup, struggle, turning point, payoff, lesson.
- Write for the ear: short sentences, plain words, natural spoken rhythm, and pauses shown with "..."
- Add exclamation marks on the important or dramatic beats so the AI voice gives them emphasis. Example: "And his name was Ferdinand von Zeppelin!"
- Use exclamations on key reveals only (names, breakthroughs, turning points). Do not put one on every sentence, or the emphasis is lost.
- Keep the tone vivid and human, not like an encyclopedia.

## Scene format

Every scene uses exactly this format:

```
Scene #N
* Script: the narration spoken over this scene
* Visual prompt: the image-generation prompt for this scene
* Direction: director notes for how the motion should work
```

### Script

Narration only. No stage directions, no scene labels, no text meant to appear on screen.

### Visual prompt

Write it as a standalone image-generation prompt. The image model sees nothing else, so be explicit and realistic about what is depicted.

- Start with the visual style (for example, "vintage sepia photograph style, 1890s Germany").
- Describe the era accurately: clothing, architecture, vehicles, tools, technology, and materials that belong to that exact time and place.
- Describe the foreground, midground, and background, plus the lighting, mood, and camera framing.
- Describe the subject precisely (shape, size, materials, colors, key features). If it could be confused with something modern or similar, say what it is not. Example: "an early rigid Zeppelin, not a modern blimp."
- Make it a faithful depiction of what the script says, with concrete details.

Every visual prompt must also include these constraints, stated clearly and in detail:

- **Faces:** Do not fill the image with faces. Keep the number of visible faces very low. Prefer wide shots, silhouettes, backs of heads, figures seen from behind or far away, objects, and landscapes. If a face must appear, name how many, and give clear detail: natural proportions, correct eyes, nose, and mouth, and a believable expression.
- **Hands:** If hands appear, spell out the detail: exactly five fingers on each hand, natural finger length and joints, correct thumb placement, and realistic grip on any object. If hands are not needed, compose the shot so they stay out of view or hidden.
- **No text:** No text anywhere in the image. No words, letters, numbers, signs, captions, logos, watermarks, newspaper print, or writing on any surface. Blank or unreadable surfaces only.

### Direction

Director notes for animating the still image into motion:

- Camera move: push in, pull out, pan, tilt, or slow drift.
- Pacing: slow, tense, sweeping, quick.
- Transitions: fade in from black, cut, dissolve.
- Say where the motion starts, where it ends, and what it reveals.
- Match the mood to the script (awe, tension, triumph).

## Final scene: the lesson

The last scene must be a lesson that closes the story. It takes the form "So what can we learn from [subject]?" followed by a short, punchy answer built on contrast, ending on the main quality. Example:

> So what can we learn from Ferdinand von Zeppelin? He was not the youngest inventor. Not the richest. Not the most popular. He was simply the most stubborn.

Give the lesson scene the same Script, Visual prompt, and Direction fields, with a fitting closing image and a slow, reflective ending (for example, a slow pull-back and a fade to black).

## Process

1. Read the user's topic description.
2. Outline the arc: hook, key events, turning point, payoff, lesson.
3. Keep only the exciting, important beats.
4. Write all scenes in the format above, numbered in order.
5. Check before output:
   - Total script is about 1,500 words.
   - Every scene has Script, Visual prompt, and Direction.
   - Every visual prompt states the faces, hands, and no-text constraints.
   - Eras, clothing, and technology in the visual prompts match the timeline.
   - Exclamations appear only on key beats.
   - The last scene is the lesson.

## Output

Output only the numbered scenes. No preamble, no closing summary, no commentary.

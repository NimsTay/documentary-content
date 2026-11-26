---
name: real-image-suggester
description: Takes a documentary narrative made of numbered scenes (Script, Visual prompt, Direction) and produces one table per scene suggesting real Wikimedia Commons images, split into files actually found in Commons searches and search ideas to try. Use when the user pastes scenes and wants real photos or archival images from Wikimedia Commons.
---

# Commons Image Suggester

## Input

A narrative made of scenes in this format (repeated, e.g. "Scene #31"):

```
Scene #N
* Script: <narration text>
* Visual prompt: <AI image prompt>
* Direction: <motion / pacing notes>
```

## Goal

For every scene, suggest real images from Wikimedia Commons (commons.wikimedia.org) that could replace or supplement the AI visual prompt.

## Steps (do these for each scene)

1. Read the Script and Visual prompt. Pick the real-world subject (for example "Soyuz spacecraft", "Yuri Gagarin", "Vostok 1 capsule").
   - Ignore AI-prompt style words ("crisp photorealistic", "no text", "no faces", mood words).
   - Use the Script for the real facts (names, years, places) so the image matches the narration.
2. Search Commons with the file-namespace API using WebFetch:
   `https://commons.wikimedia.org/w/api.php?action=query&list=search&srnamespace=6&srlimit=10&format=json&srsearch=<url-encoded query>`
   - Try 1-2 queries per scene (a specific one and a broader one).
3. Keep only results that clearly fit the scene. Copy file names EXACTLY as returned (e.g. `File:Soyuz TMA-7.jpg`). Never invent or guess a file name.
4. If fewer than 3 good files were found, fill the remaining slots with search suggestions: Commons search terms, categories (e.g. `Category:Soyuz spacecraft`), or photo types that were not found or not searched.
5. for every real image, indicate its license if available and its attribution anchor tag
## Hard limits

- **Max 3 items per scene in total** (found files + suggestions combined).
- Only mark something as found if it really came back from a Commons search in this run.
- Prefer images with a clear free license; mention the license only if it was visible in the results.

## Output format

One table per scene, in scene order. Use exactly these two columns:

```
SCENE N
| Found on Commons | Search term | License | Attribution |
|---|---|---|---|
| File:example.jpg | "vostok" | Apache License | [Attribution](ANCHOR_LINK) |


Rules for the table:
- "Found on Commons" holds only file names that came up in searches.
- "Suggestions to look for" holds search terms or categories that were not found or not looked up.
- Together the two columns have at most 3 entries per scene. Leave a cell empty rather than padding.
- Add a one-line note under the table only if the scene has no good real photo (e.g. purely abstract visual).


save output md into inputs/<docName>/real_images_suggestions.md (docName is the project folder name in inputs/)
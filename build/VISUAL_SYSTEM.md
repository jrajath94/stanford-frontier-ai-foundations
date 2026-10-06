# Supplementary Prompt: Visual System

Attach this file with the project prompt.
This file controls every figure.
The project prompt controls the words.
If the two files conflict on a picture, this file wins.
Use this file on every project.
Do not copy project names into this file.

## Decision

Draw a figure only when the figure changes the reader state.
A state change is a count, a merge, a score, a mask, a move, or a new symbol.
Do not draw a figure to decorate a paragraph.
The two paragraph rule is a ceiling, not a quota.
After two short paragraphs, stop and test the next claim.
If the claim changes state, draw it.
If the claim does not change state, write the next sentence and do not draw.

## Reader

The reader is one person who must check the claim without a second search.
Name the project in the caption.
Do not name a course unless the project prompt names it.

## Source order for a figure

1. Official source figure.
2. Official note or spec figure.
3. Board, demo, or screenshot, with a timestamp or a file path.
4. Figure from a cited paper or doc.
5. Original figure only after those four fail.

Label the source on the figure.
Use source, paper, or original.
Do not invent a quote.
Do not invent a number.
If a value is not in the source, write Not in source.

## Four tests before you draw

1. What does the object look like?
2. Why does the rule force that shape?
3. What one number can the reader change?
4. What symbol does the next page reuse?

Fail any test and do not draw.
Do not copy another teacher's colors, creatures, or layouts.

## Russian doll

Open one shell per figure.
Do not open the full system in one figure.

Shell 0. Name the question in one line.
Shell 1. Show a toy with at most 8 items or 4 numbers.
Shell 2. Count or score the toy.
Shell 3. Apply one rule.
Shell 4. Show the new symbol.
Shell 5. Name the next page that reuses the symbol.

One figure owns one shell.
The next figure owns the next shell.
Reuse the same chip, arrow, and color for the same object.

## Visual system

Background: #F7F4EE.
Ink: #1B2838.
Muted ink: #5C6B7A.
Line: #D9D3C7.
Panel: #FFFDF8.
Count box: #E7F1F8.
New object: #E7F4EF.
Active step: #F4E6D4.
Chip gray: #E6E2DA.
Teal mark: #1F7A72.
Orange mark: #C46B2C.
Focus mark: #1E4D8C.
Pink pool: #F3D4D8.
Yellow control: #F6E7A8.
Green device: #D9E8D3.

Type stack, in order: Anthropic Sans, Inter, Source Sans 3, IBM Plex Sans.
Serif only for a formula caption: Source Serif 4, then Newsreader.
Mono for an id or a shape: IBM Plex Mono, then ui-monospace.
Do not use Comic Sans, Papyrus, cursive, or a display script.
Title: 28 to 36 px. Weight 600.
Body: 16 to 18 px. Weight 450.
Label: 13 to 15 px. Weight 500.
Line height: 1.35.
One meaning per label.
Maximum 8 words per label.
If Anthropic Sans is not on the machine, use the next face in the stack.
Do not substitute a handwritten face.

Layout: title, one claim line, left before, center rule, right after, one footer claim.
Arrow means one operation.
A chip means one symbol.
A circle means one count mark.
Do not add a second meaning to a mark.

Ban in every figure: gradient, glow, drop shadow, fake texture, decorative frame, logo, watermark, author line, 3D bevel, talking head, stock robot, brain icon, circuit wallpaper.

## Shape rules

Use an 8 px grid.
Every edge, gap, and padding value is a multiple of 8.
Stroke is 1.5 px on screen and 2 px in an export.
One corner radius per object class.
Do not mix radii on one plate.

| Object | Shape | Radius | Padding |
| --- | --- | --- | --- |
| Item or token | Pill | 999 px | 8 px by 12 px |
| Service or pool | Rectangle | 12 px | 16 px |
| Block or record | Square or short rectangle | 8 px | 8 px |
| Store or device | Square | 12 px | 16 px |
| Count or result | Rectangle | 8 px | 12 px |
| Controller | Rectangle | 12 px | 12 px by 16 px |

Align boxes to one left edge inside a panel.
Equal gaps. Gap is 8 px or 16 px.
A row of peers has equal height.
A column of peers has equal width.
Center the label on both axes.
Do not let a label touch a stroke.
Hatch is 45 degrees and 4 px apart.
Hatch means absent, released, or not yet created.
Solid fill means live.
An arrow is one straight segment or one elbow.
The arrow label sits on the line and names the operation.

## Tool install

Install the skill that matches the unit.
Do not install a second skill for the same unit.

```
npx skills add tt-a1i/archify -g
npx skills add Agents365-ai/365-skills -g
npx skills add magnus919/agent-skills --skill mermaid-diagrams
npx skills add heygen-com/hyperframes
npx skills add remotion-dev/skills
npx skills add Yusuke710/manim-skill
npx skills add PaulLemaistre/explainer-video
npx skills add videozero/skills
```

Page libraries, once per site:

```
npm install three gsap
```

Render tools, once per machine:

```
brew install ffmpeg cairo pkg-config
```

Manim also needs LaTeX. Kokoro voice ships with the Manim skill.
Piper, ElevenLabs, and Gemini TTS have no install line in the source search.
Do not invent a command for them.

| Unit | Skill or library | Output |
| --- | --- | --- |
| Architecture map | Archify | One checked HTML file |
| Ink lesson plate | Excalidraw skill | SVG or PNG |
| Formal boxes | drawio skill in the 365 pack | SVG |
| Order only | Mermaid skill | SVG |
| Seekable clip | Hyperframes | MP4 |
| React clip | Remotion skills | MP4 |
| Timed proof | Manim skill | MP4 |
| Word-locked narration | explainer-video | MP4 |
| Scene graph clip | VideoZero skills | MP4 |
| Space or device | three.js | Page scene |
| Seek on the page | GSAP | Same state as the plate |
| Mux and probe | FFmpeg | Final file |

Prompt line for a still plate:

```
Font: Anthropic Sans, then Inter. No Comic Sans.
Grid: 8 px. Stroke: 1.5 px. One radius per object class.
Item is a pill. Block is a square. Pool is a 12 px rectangle.
Flat fill. No gradient. No glow. No shadow.
```

## Medium ladder

Use the first medium that passes the four tests.

| Medium | Use when | Do not use when |
| --- | --- | --- |
| Table | The claim is a comparison of values | The claim is a move |
| Equation block | The claim is a definition | The reader cannot see the parts |
| ASCII | The claim is a trace of at most 12 lines | The trace needs position or color |
| Mermaid | The claim is a flow or a dependency | The claim is a count or a geometry |
| SVG | The claim is position, count, or a small change | The reader must drag a value |
| Canvas 2D | The reader must change one number and see the result | The scene is a static list |
| three.js | The claim is space, depth, or device layout | A flat map already shows the claim |
| Manim | The claim is a timed proof or a state change | The frame is static |
| Hyperframes | The page must become a seekable clip | A still or a canvas toy is enough |

Do not render a Manim clip and a Hyperframes clip for the same shell.

## ASCII rule

One trace.
One character width for one item.
Show the before line and the after line.
Maximum 12 lines.
No box art that does not encode a step.

## Mermaid rule

One graph.
Maximum 8 nodes.
Node text maximum 4 words.
Use the graph for order, not for beauty.

## SVG and still image rule

Left panel is the before state.
Right panel is the after state.
Center arrow names the one rule.
Footer states the key idea in one sentence.
Compute the numbers.
Do not hand wave a score.

## Canvas rule

The page must compute the figure.
Do not paint a fake result.
Expose one control only.
Update the right panel from the control.
Keep the same chip colors as the still plate.

## three.js rule

Use one scene per claim.
Orthographic camera unless depth is the claim.
Flat materials.
No bloom. No fog. No particle field.
Objects are blocks with labels.
A block is a record, a device, or a store page.
Motion means a data move.
Click a block to read its name.
Library: three from a pinned CDN or a local file.
Do not add a second 3D library in the same scene.

## Manim rule

One scene per shell.
Scene length at most 20 seconds.
Use the same ground and ink as the page.
Show the toy, then the count, then the rule, then the new symbol.
Do not add a character.
Code must compute the score.
Render 1080p only for the final clip.
Link the clip beside the still plate.

## Hyperframes rule

Use Hyperframes only for a seekable clip.
Source is HTML and CSS.
Seek must land on the same state as the canvas toy.
Duration at most 20 seconds.
One claim per clip.
Do not autoplay.

## Atomic unit

An atomic unit is the smallest claim the reader must remember.
Examples: a definition, a shape, a named block, an edge, a formula part, a before state, an after state, a failure.
One unit gets one lesson plate.
Do not merge two units into one plate.
Do not leave a unit as text only.

## Cadence

The trigger is the first of these two events.

1. Two short paragraphs have passed.
2. A new atomic unit has appeared.

Then draw the unit before the next paragraph.
The figure must show the before state and the after state.
The arrow between them names the one rule.
If the unit is an architecture, name every block and every edge.
If the unit is a change, place before on the left and after on the right.

Steps:
1. Write the claim.
2. Draw the before state.
3. Draw the one rule.
4. Draw the after state.
5. Write at most two short paragraphs.
6. Stop at the next unit.
7. Link the figure to the source.
8. Reuse the symbol on the next page.

## Object path

Keep the same object visible from input to output.
The chip keeps its color after a change.
Do not replace the chip with a generic box.
The reader must point at the chip and name the step.
Stop the path at the current unit.
Do not draw later steps on a lesson plate.

## Architecture and change

These units always get a figure.
No exception.

| Unit | Required figure |
| --- | --- |
| System block | Named boxes and named edges |
| Shape or schema change | Before shape and after shape |
| Score or mix | Inputs, score, output |
| Store write | Old blocks and one new block |
| Memory or file move | Source, call name, destination |
| Split | Which axis is cut, and which part holds it |
| Step loop | Input, action, check, next state |
| Pair update | Chosen, rejected, and the update |

Hatch means absent or released.
Solid fill means live.
The call name on an arrow must be the real operation.

## Page audit

Audit every HTML page before ship.
A page fails if any unit has no figure.

1. List every heading, equation, code block, and architecture noun.
2. Give each item a unit id.
3. Map each unit id to one figure id.
4. Open the page and read it from top to bottom.
5. Fail the page if two paragraphs contain a new unit and no figure.
6. Fail the page if a before state has no after state.
7. Fail the page if an architecture block has no edge label.
8. Fail the page if a chip disappears without a named rule.
9. Fail the page if a 3D scene uses a name that the 2D plate does not use.
10. Repair the page. Run the audit again.

Audit table for each page:

| Unit id | Claim | Before | After | Figure id | Medium | Source |
| --- | --- | --- | --- | --- | --- | --- |
| u01 | one claim | old state | new state | f01 | SVG | source or original |

A blank figure cell fails the page.
The chapter plate does not replace the unit figures.
The chapter plate is extra, at the end of the concept.

## Symbol table

Fill this table in the project prompt.
Do not hard code another project's symbols here.

| Symbol | First page | Reuse |
| --- | --- | --- |
| name | page | later pages |

Do not redraw the full symbol.
Draw only the new difference.

## Figure caption

One sentence.
Name the source.
Name the shell.
Example: Shell 3. Apply the one rule. Source: original toy.

## Two plate types

Use two plate types only.
Do not mix them in one figure.

### Lesson plate

Use this plate after a state change.
Match a clean ink diagram or a structured panel plate.
Do not match a dense comic plate here.

Rules:
- White or warm paper ground.
- Black or dark navy ink line.
- Flat fill only.
- Pink for one pool. Blue for a store. Green for a device. Yellow for a controller.
- Hatch means unmapped, released, or not yet created.
- Solid fill means live.
- One label inside each block.
- Arrow text names the real operation.
- Font is Anthropic Sans, then Inter.
- Do not use Comic Sans.
- Do not use a site credit, a logo, or a watermark.
- One claim per plate.

### Chapter plate

Place one chapter plate at the end of each concept.
This plate connects the lesson plates.
It may be dense.
It must stay clean.

Rules:
- One title.
- Left region: cost without the rule.
- Center region: the stored object.
- Right region: cost with the rule.
- Bottom region: the tradeoff in one line.
- Reuse the same chip colors from the lesson plates.
- Font is Anthropic Sans, then Inter.
- No stars, no speedometer, no rocket, no sticky-note clip art, no talking cloud.
- No gradient border. No sketch spray. No watermark.
- Every number on the plate must come from the page or be marked Not in source.
- The footer states the one connection in one sentence.

## Reject list

Reject the figure if it has a robot, a brain, a glowing network, a stock photo, Comic Sans, a watermark, clip art, more than one rule on a lesson plate, or a number that the code did not compute.

## Check before ship

1. The figure changes state.
2. The number matches the source or the toy code.
3. The next page can reuse the symbol.
4. The medium is the first medium that passed.
5. No gradient, glow, shadow, logo, or watermark.
6. Caption names the source.
7. A lesson plate has one claim. A chapter plate is the only dense plate.
8. The page audit table has no blank figure cell.
9. Every architecture unit and every change unit has a before state and an after state.
10. Font is Anthropic Sans or the next face in the stack.
11. Every box sits on the 8 px grid.

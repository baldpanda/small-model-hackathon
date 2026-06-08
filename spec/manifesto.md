# Project Manifesto

## Mission

Build a playful, useful AI speech practice app for me and my best friends as we prepare to give best man speeches.

The app should help us rehearse, improve, and gain confidence without turning the speech into something generic. It should support the real job: helping people who know each other well give a speech that feels warm, funny, structured, and genuinely personal.

This project is for the [Hugging Face Build Small Hackathon](https://huggingface.co/build-small-hackathon), especially the Backyard AI challenge: solve a real problem for someone we actually know, make it specific, and build something they can actually use.

## Hackathon Constraints

- Dates: June 5 to June 15
- Model limit: total model size up to 32B parameters
- App format: Gradio
- Hosting: Hugging Face Space
- Goal: a working end-to-end experience, not a perfect product

## Spec-Driven Approach

Specs are here to help us think clearly, learn faster, and avoid building a pile of clever parts that do not become a usable app.

For this repo, spec-driven development means:

- Start with the user and the real speech practice workflow.
- Write down the intended behavior before implementing major features.
- Keep specs short, specific, and easy to change.
- Prefer acceptance criteria over vague intentions.
- Treat each spec as a learning artifact: what we thought, what we built, what changed.
- Let the specs guide the build, without slowing down hackathon momentum.

The process should be lightweight. If a spec is blocking progress, simplify it. If code teaches us something new, update the spec.

## Principles

### Make It Real

This is for a real group of friends preparing for real speeches. The product should reflect that: personal, practical, and grounded in how people actually rehearse.

### Get End-to-End Early

The first priority is a minimal working app that runs in Gradio and can be deployed to a Hugging Face Space. A thin but complete version beats a deep feature that only works locally.

### Automate the Boring Path

Commits should move the project toward an automated deploy flow. Ideally, pushing to the right branch updates the Hugging Face Space so the app is easy to share and test.

### Fit the Small-Model Constraint Honestly

The app should be designed around the total 32B model limit instead of pretending bigger models are available. Good product design, focused prompts, clear workflows, and smaller models should do the heavy lifting together.

### Keep the Human Voice

The app should help people improve their speech, not flatten it. Suggestions should preserve personal stories, natural humour, and the speaker's own voice.

### Learn in Public, Iteratively

This is mainly a learning project. Specs, commits, and notes should make the learning visible: what worked, what did not, and what we would try next.

### Polish the Core Loop

The hackathon judges care about whether the person actually used it and the polish of the Gradio app. Prioritize the main practice loop over extra features.

## Roadmap

### Phase 1: Define the Core Loop

- Identify the first speech practice workflow.
- Decide what the user inputs: draft speech, bullet points, stories, target tone, time limit, or practice recording notes.
- Define the first useful output: feedback, rewrite suggestions, structure critique, joke calibration, rehearsal prompts, or a practice scorecard.
- Write a short feature spec with acceptance criteria.

### Phase 2: Build a Minimal Gradio App

- Create a simple Gradio interface.
- Wire one small-model-backed workflow end to end.
- Make the app usable locally.
- Capture obvious failure cases and rough edges.

### Phase 3: Deploy to Hugging Face Spaces

- Add the files needed for a Hugging Face Space.
- Configure the runtime and dependencies.
- Deploy the Gradio app.
- Move toward commit-based deployment so sharing the app is low-friction.

### Phase 4: Improve with Real Feedback

- Try the app with the intended users.
- Note what helped and what felt awkward.
- Improve prompts, interface copy, and output structure.
- Keep changes small and tied to observed use.

### Phase 5: Hackathon Polish

- Tighten the Gradio UI.
- Add a concise README with usage and deployment details.
- Document the small-model fit.
- Prepare a short demo path that shows the problem, the app, and the result.

## Tech Stack

- Python for the app and model integration.
- Gradio for the user interface.
- Hugging Face Spaces for hosting.
- Hugging Face models or inference APIs, with total model size no larger than 32B parameters.
- GitHub for source control.
- GitHub Actions or Hugging Face Space Git integration for deployment automation.

## Definition of Done

The first version is done when:

- A user can open the Hugging Face Space.
- They can enter speech material or practice context.
- The app returns useful, specific feedback for best man speech practice.
- The workflow uses model resources within the 32B total limit.
- The repo explains how to run, deploy, and improve the app.

## Working Agreement

We will aim for momentum, clarity, and good taste.

Specs should make the project easier to build. Code should prove or challenge the specs. The final app should feel like something a friend would actually use before standing up to give a speech.

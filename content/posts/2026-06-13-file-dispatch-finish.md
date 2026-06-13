---
title: "File, Dispatch, Finish"
date: 2026-06-13T00:00:00Z
slug: "file-dispatch-finish"
disqus_identifier: 2026-06-13
author: martin
summary: ""
comments: true
---

My approach to coding after graduating university has not changed much up until early 2024, when I have seriously started integrating AI in my daily workflows, but it seems that the most significant changes are still yet to come.
The way how we build software is being reshaped and I think we have not settled on any concrete approach yet, and who knows we might just end up with many more different approaches than we have known until now.
At first, when experimenting with Github Copilot, I have used it only for basic autocomplete of code in the closest scope.
Then in 2025, I moved to Cursor which offered much better autocomplete capabilities, it worked throughout the whole file and multiple files as well.
I could move faster and with very surgical precision.
I have tried to use the chat interface as well, but its capabilities were quite limited.
It was useful mainly for debugging of known problematic parts of code, or for writing small-scope functions, and the experience was basically on-par with just using free version of ChatGPT.
All this time, the approaches of software engineers were more or less same, because the tools were designed with a very specific use in mind.

It all changed when Claude Opus has started showing promising results in late 2025.
Suddenly, the previously awkward chat interface became the way how we express our needs and requirements when building systems.
It might seem that chat interfaces has already been well known at that point, because everybody had experience with it in any of freely available AI chats.
However, unlike small-scope tab completion, or chat regarding specific small parts of code, the chat interface connected with very capable AI has changed the game.
Suddenly, anybody with access to AI was able to build cool looking demos, pretend to work while off-loading everything mindlessly on AI without any understanding, and accepting all the work without review scrutiny.
Developers became divided on whether the AI is good enough to take their jobs, while the AI models kept improving and breaking new records.
Now, the AI is not just chat interface, it is markdown documents describing how to behave, skills that can be readily loaded, memories of past conversation and many others.
The softward development work has become more of how to control the AI in order to get what you want, yet I still see people not taking advantage of it, and producing subpar work.
It takes a lot of energy to stay up to date with new developments and approaches, but most importantly, there are now nearly infinite number of ways how to do the same thing.

When I thought how take advantage of this rapidly changing environment, I knew that my new approach has to be very simple.
It shouldn't require me to build complex pipelines, install new software, or integrate something because of some feature that is already supported by required stack.
The only requirements are Github (issues, PR, CI/CD), because I already have all the code there, and Claude (coding, orchestration, memory), because that is my current AI of choice, but any other AI would work the same way.
The approach is defined as an abstract pipeline with three distinct consecutive steps: 1) File, 2) Dispatch and 3) Finish.
The pipeline has to be executed in order, but there can be unlimited number of pipelines running at the same time.

## File

The first step in the pipeline is to record what we want to do or what we should do.
When there is feature, we want to develop, or when there is a bug we want to fix, we can file an issue through Claude code.
Claude code comes with `/new-issue` skill which allows us to generate and post issue to GitHub.
Claude code has access to the context of the repository on which we run the skill, therefore it can provide many important details even before we start working on the issue.
The issue can be filed by any other system that has write rights to the GitHub repository.
We can file as many issues as we want, or just one and move on to the next stage, but important thing is that it depends on situation and we are not limited.


## Dispatch

The dispatch step is when the AI actually starts working on the problem.
Similarly to the previous step, there are multiple ways how we can trigger it: manually or automatically by system that has access to GitHub to read issue, write code, and open PR.
When working on new features or bug fixes, I trigger this step manually from within a new `claude agents` session using custom skill `/dispatch-issue` which requires me to specify the issue ID (e.g. `/dispatch-issue 593`).
Sometimes, I might want to add a little more context, but most of the time, I just specify the ID.
Agent then starts working on the issue following the dispatch issue skill (paraphrased below):

1. Attach WIP prefix to the issue that was dispatch
2. Your goal is to solve the problem explained in the issue
3. Before coding anything, ask for confirmation if there are multiple way of achieving the goal
4. Split the work in to the steps that are separated by commits
5. Open PR and link it to the issue
6. (repeat up to 5 times)
  * 6a. Launch review agent on your PR
  * 6b. Fix his findings directly in PR if they are related
  * 6c. Files new issues based on unresolved findings
7. Notify about completion

The review agent finds at least 1 issue in most of the PRs that the original agent created, unless the PR is very small.
The limit of five review rounds was set just as guess, it has never went over 4 rounds of reviews.


## Finish

The last step exist mainly to make sure that the generated code solves the problem we wanted to tackle, and to help AI to learn from this completed issue.
It is less complex, and composed of three substeps:

1. Merge the PR if all checks passed
2. Remove the WIP prefix from the related issue
3. Update your memory about the project based on this work

---
title: "Sandboxing and bypassing permissions"
date: 2026-05-26T00:00:00Z
slug: "sandboxing-and-bypassing-permissions"
disqus_identifier: 2026-05-26
author: martin
summary: "AI-written code quality has reached the level of production codebases, and AI agents often follow engineering practices comparable to experienced human engineers. There are still, however, many engineers who do not take full advantage of these advances in the field. They might be limited by the processes set up in their companies, belief in their own superiority or just pure fear, but more often than not, engineers have transformed into approval machines: they set a task, then mindlessly keep approving the agent's requests (running scripts, accessing the web, editing files). How do I know that? I was one of them. It worked well, I had a good understanding of what was happening at every moment, my productivity increased because agents are faster than humans in most tasks, but unfortunately it led to an artificial speed bump for the whole process."
comments: true
---

AI-written code quality has reached the level of production codebases, and AI agents often follow engineering practices comparable to experienced human engineers. There are still, however, many engineers who do not take full advantage of these advances in the field. They might be limited by the processes set up in their companies, belief in their own superiority or just pure fear, but more often than not, engineers have transformed into approval machines: they set a task, then mindlessly keep approving the agent's requests (running scripts, accessing the web, editing files). How do I know that? I was one of them. It worked well, I had a good understanding of what was happening at every moment, my productivity increased because agents are faster than humans in most tasks, but unfortunately it led to an artificial speed bump for the whole process.

## `--dangerously-skip-permissions`

More than a year ago, April 2025, when we organized vibe coding day at Bisonai for engineers, we did not get much out of the available AI tools when applied to our production codebase. At that time, we saw the potential but our engineers were still more productive by writing code directly or with the help of Cursor's Tab autocomplete. This experience built a distrust with AI tooling, and even when models started rapidly improving at the end of 2025, I did not think it was safe to let agents work without supervision.

Fast forward a couple weeks, I stopped coding manually and stopped using git and GitHub directly. I started operating fully through agents, and even if some part of the process (e.g. making a commit or pushing changes) could have been done faster if done manually by myself, I would have been attached to the agent's workflow and slowed down the agent's progress. I wanted to be fully separated from the internal processes that agents go through and was only interested in the final outcome; trying to avoid agent micromanagement. This suddenly allowed me to work on multiple things at the same time, and after recovering from my distrust of agents, I started using `--dangerously-skip-permissions`.

`--dangerously-skip-permissions` allows Claude to execute tasks, run shell commands, and modify files without constantly prompting you for approval. Simply put, everything is allowed, and by everything I mean everything. It solves approval fatigue and lets agents scale, but with a significant drawback: what if it does something you would never do like `rm -rf /`? I am going to discuss the protection mechanism in the Sandbox section.

You can launch Claude Code with `--dangerously-skip-permissions`, or you can specify the config in `settings.json`.

```json
{
  "skipDangerousModePermissionPrompt": true
}
```

## Sandbox

To make `--dangerously-skip-permissions` actually safer, the agent needs guardrails of a different kind. Claude Code supports six different ways of sandboxing the agent, and therefore limiting the agent's capabilities. For local small-scale development, there are two options: Sandboxed Bash tool and Sandbox runtime. The Sandbox runtime is more powerful because it constrains every tool, hook, and MCP server in the session, not only Bash. In this guide I am just showing how to use the Sandboxed Bash tool, because the Sandbox runtime is a beta research preview, but I am planning to migrate to it soon.

Sandboxed Bash tool is integrated into Claude Code, and can be set up through properties in `settings.json`. Personally, I store them in my global settings.

```json
{
  "sandbox": {
    "network": {
      "allowedDomains": [
        "*"
      ]
    },
    "filesystem": {
      "allowWrite": [
        ".",
        "/tmp",
        "~/.claude/jobs"
      ],
      "denyRead": [
        "~/"
      ],
      "allowRead": [
        ".",
        "~/.config/gh/config.yml",
        "~/.config/gh/hosts.yml",
        "~/.gitconfig"
      ]
    }
  }
}
```

The settings above grant unrestricted internet access while strictly limiting filesystem access. It blocks access to your home directory, but makes specific exceptions: the tool can read and write to your current workspace, `/tmp`, and `~/.claude/jobs`, and read Git credentials to operate on git autonomously.

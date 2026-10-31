# WithdrawChan Website README

[English](README.md) | [简体中文](../doc/web.README.zh.md)

This directory contains everything needed to build the WithdrawChan site. It is a static website, and recommendations are computed locally in the browser.

- For the project's aims, see the [repository README](../README.md).
- For the recommendation rule, see [`src/README.md`](../src/README.md).

## Build

To build it yourself, install Node.js 22.18 or newer and run from the repository root:

```bash
npm run setup
npm run build
```

You can also have an agent do it for you, using this [`SKILL.md`](SKILL.md).

The whole site is then generated in `web/public/`. Serve it with any static file server, or preview it locally with `npm run preview`.

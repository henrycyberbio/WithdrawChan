# WithdrawChan 网站 README

[English](../web/README.md) | [简体中文](web.README.zh.md)

这里包含构建 WithdrawChan 站点所需要的一切。这是一个静态网站，推荐金额在浏览器本地计算。

- 项目目标请见[仓库 README](README.zh.md)
- 推荐规则请见 [`src/README.md`](src.README.zh.md)。

## 构建
自己动手的话，安装 Node.js 22.18 或更新版本，在仓库根目录运行：

```bash
npm run setup
npm run build
```

你也可以让 Agent 替你做，这里有一个 [`SKILL.md`](../web/SKILL.md)。

整个网站会生成在 `web/public/`，用任意静态文件服务器托管即可，或者可以用 `npm run preview` 在本地预览。

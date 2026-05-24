---
name: CI 问题反馈
about: 反馈 GitHub Actions CI 失败或异常，便于排查
title: "[CI] "
labels: ci
assignees: ''
---

## 失败的 Workflow

<!-- 在 Actions 页打开失败的那次运行，复制浏览器地址栏链接 -->
链接：

## 分支与提交

- 分支：
- Commit SHA（可选）：

## 失败的 Job / Step

<!-- 例如：Shell Script Syntax Check / Dataset Download URL Check / Dataset Extract Logic Test -->
- Job 名称：
- Step 名称：

## 是否与数据集下载相关？

- [ ] 是（请写明数据集：TID2013 / KoNViD-1k / T2VQA-DB）
- [ ] 否

下载脚本：`scripts/manage_data.sh`

## 错误现象

<!-- 粘贴关键报错日志，或截图说明 -->
```
在此粘贴日志
```

## 复现方式

<!-- 本地如何复现？例如：bash -n scripts/manage_data.sh -->
1.
2.

## 环境信息（可选）

- 操作系统：
- 是否在 AutoDL / 云 GPU 上：
- 代理 / 网络情况：

## 补充说明

<!-- 近期是否改过 scripts/ 下的 .sh、.github/workflows/ 等 -->

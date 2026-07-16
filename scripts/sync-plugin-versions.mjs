#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf-8"));
}
function writeJson(path, data) {
  writeFileSync(path, `${JSON.stringify(data, null, 2)}\n`, "utf-8");
}

const repoRoot = process.cwd();
const packageJson = readJson(resolve(repoRoot, "package.json"));
const leaf = packageJson.name.split("/").pop();
if (!packageJson.version) throw new Error("package.json version is missing or invalid");
const targetVersion = String(packageJson.version).trim();

const claudePluginRoot = resolve(repoRoot, "plugins", `claude-${leaf}`);
const cursorPluginRoot = resolve(repoRoot, "plugins", `cursor-${leaf}`);
const claudePluginPath = resolve(claudePluginRoot, ".claude-plugin", "plugin.json");
const cursorPluginPath = resolve(cursorPluginRoot, ".cursor-plugin", "plugin.json");

for (const pluginPath of [claudePluginPath, cursorPluginPath]) {
  const pluginJson = readJson(pluginPath);
  pluginJson.version = targetVersion;
  writeJson(pluginPath, pluginJson);
  process.stdout.write(`synced ${pluginPath} to version ${targetVersion}\n`);
}

const skillSource = resolve(repoRoot, "SKILL.md");
for (const pluginRoot of [claudePluginRoot, cursorPluginRoot]) {
  const content = readFileSync(skillSource, "utf-8");
  const targetDir = resolve(pluginRoot, "skills", leaf);
  const target = resolve(targetDir, "SKILL.md");
  mkdirSync(targetDir, { recursive: true });
  writeFileSync(target, content, "utf-8");
  process.stdout.write(`synced ${target} from repo SKILL.md\n`);
}

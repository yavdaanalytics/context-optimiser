#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import os from 'os';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const homeDir = os.homedir();
const isLocal = process.argv.includes('--local');

// 1. Determine paths
const geminiSettingsPath = path.join(homeDir, '.gemini', 'settings.json');
const cursorMcpPath = path.join(homeDir, '.cursor', 'mcp.json');
const claudeConfigDir = path.join(homeDir, 'AppData', 'Roaming', 'Claude');
const claudeSettingsPath = path.join(homeDir, '.claude', 'settings.json');
const claudeConfigPath = path.join(claudeConfigDir, 'claude_desktop_config.json');

// Global Skills Directories
const geminiSkillDir = path.join(homeDir, '.gemini', 'config', 'skills', 'context-optimiser');
const cursorSkillDir = path.join(homeDir, '.cursor', 'skills', 'context-optimiser');
const claudeSkillDir = path.join(homeDir, '.claude', 'skills', 'context-optimiser');

// Resolve the command for MCP registration
// If installed globally, run command 'context-optimiser'. Otherwise run using python from current location.
const serverCmd = 'context-optimiser';
const serverArgs = [];
const localServerCmd = 'python';
const localServerArgs = [path.normalize(path.join(__dirname, '../mcp_server.py'))];

const mcpConfig = isLocal ? {
  command: localServerCmd,
  args: localServerArgs
} : {
  command: serverCmd,
  args: serverArgs
};

const cursorMcpConfig = isLocal ? {
  type: 'stdio',
  command: localServerCmd,
  args: localServerArgs
} : {
  type: 'stdio',
  command: serverCmd,
  args: serverArgs
};

console.log(`Setting up context-optimiser MCP server config (mode: ${isLocal ? 'local' : 'global'})...`);

// Helper to write to JSON configuration
function updateMcpConfig(filePath, configValue, isCursor = false) {
  try {
    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    
    let data = {};
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf-8').trim();
      if (content) {
        data = JSON.parse(content);
      }
    }
    
    if (!data.mcpServers) {
      data.mcpServers = {};
    }
    
    data.mcpServers['context-optimiser'] = configValue;
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
    console.log(`  Successfully updated: ${filePath}`);
  } catch (err) {
    console.error(`  Failed to update config at ${filePath}:`, err.message);
  }
}

// Update settings
updateMcpConfig(geminiSettingsPath, mcpConfig);
updateMcpConfig(cursorMcpPath, cursorMcpConfig, true);
updateMcpConfig(claudeConfigPath, mcpConfig);
updateMcpConfig(claudeSettingsPath, mcpConfig);

// 2. Copy SKILL.md to global skill folders
const skillSourcePath = path.normalize(path.join(__dirname, '../SKILL.md'));

function copySkillFile(targetDir) {
  try {
    if (!fs.existsSync(skillSourcePath)) {
      console.warn(`  Warning: Source SKILL.md not found at ${skillSourcePath}`);
      return;
    }
    
    if (!fs.existsSync(targetDir)) {
      fs.mkdirSync(targetDir, { recursive: true });
    }
    
    const targetFile = path.join(targetDir, 'SKILL.md');
    fs.copyFileSync(skillSourcePath, targetFile);
    console.log(`  Copied SKILL.md to: ${targetFile}`);
  } catch (err) {
    console.error(`  Failed to copy SKILL.md to ${targetDir}:`, err.message);
  }
}

console.log('Copying loader skills to global directories...');
copySkillFile(geminiSkillDir);
copySkillFile(cursorSkillDir);
copySkillFile(claudeSkillDir);

console.log('Setup completed successfully.');

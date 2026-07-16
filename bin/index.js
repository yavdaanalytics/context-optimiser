#!/usr/bin/env node
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Path to mcp_server.py relative to the bin folder
const serverPath = join(__dirname, '../mcp_server.py');

// Spawn Python process, pipe all stdio (which MCP fastmcp needs) and forward arguments
const pythonProcess = spawn('python', [serverPath, ...process.argv.slice(2)], {
  stdio: 'inherit'
});

pythonProcess.on('error', (err) => {
  console.error('Error starting context-optimiser Python MCP server:', err);
  console.error('Please make sure python is installed and available in your system PATH.');
  process.exit(1);
});

pythonProcess.on('exit', (code) => {
  process.exit(code || 0);
});

#!/usr/bin/env bun
/**
 * Development server for Multi-Platform AI Orchestration
 */

import { spawn } from 'child_process';

console.log('🚀 Starting Multi-Platform AI Orchestration Development Server...\n');

// Start the Python FastAPI server
console.log('📡 Starting FastAPI server...');

const pythonServer = spawn('python', ['-m', 'uvicorn', 'ai_orchestration.api:app', '--reload', '--host', '0.0.0.0', '--port', '8000'], {
  stdio: 'inherit'
});

// Handle cleanup
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down development server...');
  pythonServer.kill();
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Shutting down development server...');
  pythonServer.kill();
  process.exit(0);
});

pythonServer.on('close', (code) => {
  console.log(`\n📡 FastAPI server exited with code ${code}`);
  process.exit(code || 0);
});

export {};
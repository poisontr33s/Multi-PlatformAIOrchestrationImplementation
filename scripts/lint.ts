#!/usr/bin/env bun
/**
 * Linting script for Multi-Platform AI Orchestration
 */

import { spawn } from 'child_process';

console.log('🔍 Linting Multi-Platform AI Orchestration...\n');

let exitCode = 0;

// Python linting
console.log('🐍 Linting Python code...');

const pythonLinters = [
  { name: 'Black', cmd: 'black', args: ['--check', 'src/', 'tests/'] },
  { name: 'isort', cmd: 'isort', args: ['--check-only', 'src/', 'tests/'] },
  { name: 'flake8', cmd: 'flake8', args: ['src/', 'tests/'] },
];

for (const linter of pythonLinters) {
  try {
    console.log(`  Running ${linter.name}...`);
    const proc = spawn(linter.cmd, linter.args, { stdio: 'inherit' });
    
    await new Promise((resolve) => {
      proc.on('close', (code) => {
        if (code !== 0) {
          console.log(`  ❌ ${linter.name} failed`);
          exitCode = code || 1;
        } else {
          console.log(`  ✅ ${linter.name} passed`);
        }
        resolve(code);
      });
    });
    
  } catch (error) {
    console.log(`  ⚠️  ${linter.name} not available, skipping...`);
  }
}

// TypeScript linting  
console.log('\n📦 Linting TypeScript code...');

try {
  console.log('  Running TypeScript check...');
  const tsCheck = spawn('bun', ['--check', 'scripts/*.ts'], { stdio: 'inherit' });
  
  await new Promise((resolve) => {
    tsCheck.on('close', (code) => {
      if (code !== 0) {
        console.log('  ❌ TypeScript check failed');
        exitCode = code || 1;
      } else {
        console.log('  ✅ TypeScript check passed');
      }
      resolve(code);
    });
  });
  
} catch (error) {
  console.log('  ⚠️  TypeScript linting not available');
}

console.log('\n' + '='.repeat(50));

if (exitCode === 0) {
  console.log('🎉 All linting checks passed!');
} else {
  console.log('❌ Some linting issues found. Please fix them.');
  console.log('\n💡 To auto-fix Python formatting:');
  console.log('  black src/ tests/');
  console.log('  isort src/ tests/');
}

process.exit(exitCode);

export {};
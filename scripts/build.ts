#!/usr/bin/env bun
/**
 * Build script for Multi-Platform AI Orchestration
 */

import { spawn } from 'child_process';

console.log('🔨 Building Multi-Platform AI Orchestration...\n');

let exitCode = 0;

// Build Python package
console.log('🐍 Building Python package...');
try {
  const pythonBuild = spawn('python', ['-m', 'build'], { stdio: 'inherit' });
  
  pythonBuild.on('close', (code) => {
    if (code !== 0) {
      console.log(`❌ Python build failed with code ${code}`);
      exitCode = code || 1;
    } else {
      console.log('✅ Python package built successfully');
    }
  });
  
  await new Promise((resolve) => pythonBuild.on('close', resolve));
  
} catch (error) {
  console.log('❌ Python build failed:', error);
  exitCode = 1;
}

// Compile TypeScript
console.log('\n📦 Compiling TypeScript...');
try {
  const tsBuild = spawn('bun', ['build', 'scripts/*.ts', '--outdir', 'dist'], { stdio: 'inherit' });
  
  tsBuild.on('close', (code) => {
    if (code !== 0) {
      console.log(`❌ TypeScript build failed with code ${code}`);
      exitCode = code || 1;
    } else {
      console.log('✅ TypeScript compiled successfully');
    }
  });
  
  await new Promise((resolve) => tsBuild.on('close', resolve));
  
} catch (error) {
  console.log('❌ TypeScript build failed:', error);
  exitCode = 1;
}

console.log('\n' + '='.repeat(50));

if (exitCode === 0) {
  console.log('🎉 Build completed successfully!');
  console.log('\n📦 Artifacts:');
  console.log('  dist/          # Python wheel and sdist');
  console.log('  dist/          # Compiled TypeScript');
} else {
  console.log('❌ Build failed!');
}

process.exit(exitCode);

export {};
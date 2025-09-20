#!/usr/bin/env bun
/**
 * Setup checks for the dual-runtime environment
 */

console.log('🔍 Checking Multi-Platform AI Orchestration Setup...\n');

let allGood = true;

// Check Bun
if (typeof Bun !== 'undefined') {
  console.log(`✅ Bun runtime: ${Bun.version}`);
} else {
  console.log('❌ Bun runtime not detected');
  allGood = false;
}

// Check Node.js compatibility
try {
  const nodeVersion = process.version;
  console.log(`✅ Node.js compatibility: ${nodeVersion}`);
} catch (error) {
  console.log('❌ Node.js compatibility issue');
  allGood = false;
}

// Check package.json
try {
  const packageJson = await Bun.file('package.json').json();
  console.log(`✅ Package: ${packageJson.name}@${packageJson.version}`);
} catch (error) {
  console.log('❌ package.json not found or invalid');
  allGood = false;
}

// Check Python environment
try {
  const proc = Bun.spawn(['python', '--version'], { stdout: 'pipe' });
  const output = await new Response(proc.stdout).text();
  console.log(`✅ Python: ${output.trim()}`);
} catch (error) {
  console.log('❌ Python not found');
  allGood = false;
}

// Check AI Orchestration package
try {
  const proc = Bun.spawn(['python', '-c', 'import ai_orchestration; print("✅ AI Orchestration package imported")'], { 
    stdout: 'pipe',
    stderr: 'pipe'
  });
  const output = await new Response(proc.stdout).text();
  console.log(output.trim());
} catch (error) {
  console.log('❌ AI Orchestration package not available');
  allGood = false;
}

// Check CLI
try {
  const proc = Bun.spawn(['python', '-m', 'ai_orchestration.cli', '--help'], { 
    stdout: 'pipe',
    stderr: 'pipe'
  });
  await proc.exited;
  if (proc.exitCode === 0) {
    console.log('✅ AI Orchestration CLI working');
  } else {
    console.log('❌ AI Orchestration CLI not working');
    allGood = false;
  }
} catch (error) {
  console.log('❌ AI Orchestration CLI not available');
  allGood = false;
}

// Check TypeScript
try {
  const proc = Bun.spawn(['bun', '--version'], { stdout: 'pipe' });
  await proc.exited;
  console.log('✅ TypeScript support (via Bun)');
} catch (error) {
  console.log('❌ TypeScript support not available');
  allGood = false;
}

console.log('\n' + '='.repeat(50));

if (allGood) {
  console.log('🎉 All systems operational!');
  console.log('\n🚀 Ready for multi-platform AI orchestration');
  console.log('\nAvailable commands:');
  console.log('  bun run gemini --help     # Google Gemini CLI');
  console.log('  bun run claude --help     # Anthropic Claude CLI');
  console.log('  bun run openai --help     # OpenAI GPT CLI');
  console.log('  ai-orchestrator --help    # Python orchestration CLI');
} else {
  console.log('⚠️  Some issues detected. Please review the setup.');
  console.log('\n💡 Try running the setup script:');
  console.log('  ./setup-jules.sh    # For Jules/container environments');
  console.log('  ./setup-local.sh    # For local development');
  process.exit(1);
}

export {};
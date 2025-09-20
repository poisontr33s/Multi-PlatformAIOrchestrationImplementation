#!/usr/bin/env bun
/**
 * Claude CLI Bridge  
 * Thin Node.js bridge to run Claude CLI via bunx/npx without global installs
 */

import { spawn } from 'child_process';
import { parseArgs } from 'util';

interface ClaudeOptions {
  model?: string;
  prompt?: string;
  file?: string;
  system?: string;
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
  verbose?: boolean;
  help?: boolean;
}

function showHelp() {
  console.log(`
🤖 Claude CLI Bridge - Multi-Platform AI Orchestration

Usage: bun run claude [options]

Options:
  --model <model>         Claude model to use (default: claude-3-5-sonnet-20241022)
  --prompt <text>         Prompt text to send to Claude
  --file <path>           Read prompt from file
  --system <text>         System prompt
  --temperature <num>     Temperature (0.0-1.0)
  --max-tokens <num>      Maximum tokens to generate
  --stream                Enable streaming response
  --verbose               Enable verbose output
  --help, -h              Show this help

Examples:
  bun run claude --prompt "Hello, how are you?"
  bun run claude --file prompt.txt --model claude-3-5-haiku-20241022
  bun run claude --prompt "Explain AI" --system "You are a helpful teacher"
  
Environment Variables:
  ANTHROPIC_API_KEY      Anthropic API key (required)
  CLAUDE_API_KEY         Alternative name for Anthropic API key
  
Note: This is a bridge script. Install claude CLI for full functionality:
  bunx @anthropic-ai/claude-cli [args]
`);
}

async function runClaudeCLI(options: ClaudeOptions) {
  // Check for API key
  const apiKey = process.env.ANTHROPIC_API_KEY || process.env.CLAUDE_API_KEY;
  if (!apiKey) {
    console.error('❌ Anthropic API key required. Set ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable.');
    process.exit(1);
  }

  // Build command arguments
  const args: string[] = [];
  
  if (options.model) {
    args.push('--model', options.model);
  }
  
  if (options.prompt) {
    args.push('--prompt', options.prompt);
  }
  
  if (options.file) {
    args.push('--file', options.file);
  }
  
  if (options.system) {
    args.push('--system', options.system);
  }
  
  if (options.temperature !== undefined) {
    args.push('--temperature', options.temperature.toString());
  }
  
  if (options.maxTokens) {
    args.push('--max-tokens', options.maxTokens.toString());
  }
  
  if (options.stream) {
    args.push('--stream');
  }
  
  if (options.verbose) {
    args.push('--verbose');
  }

  try {
    // Try to run via bunx first, then fallback to npx
    const runtime = process.env.BUN_INSTALL ? 'bunx' : 'npx';
    
    if (options.verbose) {
      console.log(`🔧 Running: ${runtime} @anthropic-ai/claude-cli ${args.join(' ')}`);
    }
    
    const child = spawn(runtime, ['@anthropic-ai/claude-cli', ...args], {
      stdio: 'inherit',
      env: {
        ...process.env,
        ANTHROPIC_API_KEY: apiKey
      }
    });
    
    child.on('close', (code) => {
      if (code !== 0 && runtime === 'bunx') {
        // Fallback to npx if bunx fails
        console.log('🔄 Falling back to npx...');
        const fallbackChild = spawn('npx', ['@anthropic-ai/claude-cli', ...args], {
          stdio: 'inherit',
          env: {
            ...process.env,
            ANTHROPIC_API_KEY: apiKey
          }
        });
        
        fallbackChild.on('close', (fallbackCode) => {
          process.exit(fallbackCode || 0);
        });
      } else {
        process.exit(code || 0);
      }
    });
    
  } catch (error) {
    console.error('❌ Failed to run Claude CLI:', error);
    console.log('\n💡 Try installing the CLI manually:');
    console.log('   npm install -g @anthropic-ai/claude-cli');
    console.log('   or use bunx/npx directly:');
    console.log('   bunx @anthropic-ai/claude-cli --help');
    process.exit(1);
  }
}

// Parse command line arguments
const args = process.argv.slice(2);

try {
  const { values } = parseArgs({
    args,
    options: {
      model: { type: 'string' },
      prompt: { type: 'string' },
      file: { type: 'string' },
      system: { type: 'string' },
      temperature: { type: 'string' },
      'max-tokens': { type: 'string' },
      stream: { type: 'boolean' },
      verbose: { type: 'boolean' },
      help: { type: 'boolean' },
      h: { type: 'boolean' }
    },
    allowPositionals: true
  });

  const options: ClaudeOptions = {
    model: values.model,
    prompt: values.prompt,
    file: values.file,
    system: values.system,
    temperature: values.temperature ? parseFloat(values.temperature) : undefined,
    maxTokens: values['max-tokens'] ? parseInt(values['max-tokens']) : undefined,
    stream: values.stream,
    verbose: values.verbose,
    help: values.help || values.h
  };

  if (options.help) {
    showHelp();
    process.exit(0);
  }

  // If no prompt or file provided, show help
  if (!options.prompt && !options.file) {
    console.log('❌ Please provide either --prompt or --file');
    showHelp();
    process.exit(1);
  }

  await runClaudeCLI(options);

} catch (error) {
  console.error('❌ Error parsing arguments:', error);
  showHelp();
  process.exit(1);
}
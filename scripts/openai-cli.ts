#!/usr/bin/env bun
/**
 * OpenAI CLI Bridge
 * Thin Node.js bridge to run OpenAI CLI via bunx/npx without global installs
 */

import { spawn } from 'child_process';
import { parseArgs } from 'util';

interface OpenAIOptions {
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
🤖 OpenAI CLI Bridge - Multi-Platform AI Orchestration

Usage: bun run openai [options]

Options:
  --model <model>         OpenAI model to use (default: gpt-4-turbo-preview)
  --prompt <text>         Prompt text to send to OpenAI
  --file <path>           Read prompt from file
  --system <text>         System prompt
  --temperature <num>     Temperature (0.0-2.0)
  --max-tokens <num>      Maximum tokens to generate
  --stream                Enable streaming response
  --verbose               Enable verbose output
  --help, -h              Show this help

Examples:
  bun run openai --prompt "Hello, how are you?"
  bun run openai --file prompt.txt --model gpt-4
  bun run openai --prompt "Explain AI" --system "You are a helpful teacher"
  
Environment Variables:
  OPENAI_API_KEY         OpenAI API key (required)
  
Note: This is a bridge script. Install openai CLI for full functionality:
  bunx openai-cli [args]
`);
}

async function runOpenAICLI(options: OpenAIOptions) {
  // Check for API key
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.error('❌ OpenAI API key required. Set OPENAI_API_KEY environment variable.');
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
      console.log(`🔧 Running: ${runtime} openai-cli ${args.join(' ')}`);
    }
    
    const child = spawn(runtime, ['openai-cli', ...args], {
      stdio: 'inherit',
      env: {
        ...process.env,
        OPENAI_API_KEY: apiKey
      }
    });
    
    child.on('close', (code) => {
      if (code !== 0 && runtime === 'bunx') {
        // Fallback to npx if bunx fails
        console.log('🔄 Falling back to npx...');
        const fallbackChild = spawn('npx', ['openai-cli', ...args], {
          stdio: 'inherit',
          env: {
            ...process.env,
            OPENAI_API_KEY: apiKey
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
    console.error('❌ Failed to run OpenAI CLI:', error);
    console.log('\n💡 Try installing the CLI manually:');
    console.log('   npm install -g openai-cli');
    console.log('   or use bunx/npx directly:');
    console.log('   bunx openai-cli --help');
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

  const options: OpenAIOptions = {
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

  await runOpenAICLI(options);

} catch (error) {
  console.error('❌ Error parsing arguments:', error);
  showHelp();
  process.exit(1);
}
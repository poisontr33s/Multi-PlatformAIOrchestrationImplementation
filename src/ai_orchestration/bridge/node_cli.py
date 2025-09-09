"""
Node CLI Bridge for Multi-Platform AI Orchestration
Executes bunx commands with npx fallback for Node.js CLI integration.
"""

import asyncio
import shutil
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class NodeCLIBridge:
    """
    Bridge for executing Node.js CLI commands with bunx-first, npx-fallback strategy.

    This bridge enables the Python orchestration system to interface with
    Node.js-based CLI tools and commands.
    """

    def __init__(self, prefer_bun: bool = True):
        """
        Initialize the Node CLI bridge.

        Args:
            prefer_bun: Whether to prefer bunx over npx when available
        """
        self.prefer_bun = prefer_bun
        self.bun_available = shutil.which("bun") is not None
        self.npx_available = shutil.which("npx") is not None

        logger.info(
            "NodeCLIBridge initialized",
            bun_available=self.bun_available,
            npx_available=self.npx_available,
            prefer_bun=prefer_bun,
        )

    def get_executor(self) -> str:
        """
        Get the preferred command executor (bunx or npx).

        Returns:
            Command to use for execution

        Raises:
            RuntimeError: If neither bun nor npx is available
        """
        if self.prefer_bun and self.bun_available:
            return "bunx"
        if self.npx_available:
            return "npx"
        if self.bun_available:
            return "bunx"
        raise RuntimeError(
            "Neither bun nor npx is available. Install Node.js/npm or Bun to use CLI bridge.",
        )

    async def execute_command(
        self,
        command: str,
        args: List[str] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute a Node.js CLI command.

        Args:
            command: Command to execute (e.g., "create-next-app", "typescript")
            args: Additional arguments for the command
            cwd: Working directory for command execution
            env: Environment variables
            timeout: Command timeout in seconds

        Returns:
            Dictionary with execution results
        """
        executor = self.get_executor()
        full_command = [executor, command] + (args or [])

        logger.info(
            "Executing Node CLI command",
            executor=executor,
            command=command,
            args=args,
            cwd=cwd,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *full_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            result = {
                "success": process.returncode == 0,
                "returncode": process.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "command": " ".join(full_command),
                "executor": executor,
            }

            if result["success"]:
                logger.info(
                    "Command executed successfully",
                    command=" ".join(full_command),
                    returncode=process.returncode,
                )
            else:
                logger.error(
                    "Command execution failed",
                    command=" ".join(full_command),
                    returncode=process.returncode,
                    stderr=result["stderr"][:500],  # First 500 chars of error
                )

            return result

        except asyncio.TimeoutError:
            logger.error(
                "Command timed out", command=" ".join(full_command), timeout=timeout
            )
            return {
                "success": False,
                "error": "Command timed out",
                "command": " ".join(full_command),
                "executor": executor,
                "timeout": timeout,
            }

        except Exception as e:
            logger.error(
                "Command execution error", command=" ".join(full_command), error=str(e)
            )
            return {
                "success": False,
                "error": str(e),
                "command": " ".join(full_command),
                "executor": executor,
            }

    async def check_package_available(self, package: str) -> bool:
        """
        Check if a Node.js package is available for execution.

        Args:
            package: Package name to check

        Returns:
            True if package is available
        """
        try:
            result = await self.execute_command(package, ["--version"], timeout=10.0)
            return result["success"]
        except Exception:
            return False

    async def gemini_cli_placeholder(self, args: List[str] = None) -> Dict[str, Any]:
        """
        Placeholder for future gemini CLI command.

        TODO: Implement actual gemini CLI interface when available.
        Expected interface:
        - gemini generate --prompt "text" --model "model-name"
        - gemini models --list
        - gemini config --set-key "api-key"

        Args:
            args: Arguments for the gemini CLI

        Returns:
            Placeholder response indicating CLI is not yet implemented
        """
        logger.warning(
            "Gemini CLI placeholder called - not yet implemented",
            args=args,
        )

        return {
            "success": False,
            "error": "Gemini CLI not yet implemented",
            "todo": [
                "Implement gemini CLI package or script",
                "Add generate command: gemini generate --prompt 'text' --model 'model-name'",
                "Add models command: gemini models --list",
                "Add config command: gemini config --set-key 'api-key'",
                "Integrate with google-generativeai Python SDK backend",
            ],
            "expected_interface": {
                "generate": "gemini generate --prompt <text> [--model <model>] [--max-tokens <n>]",
                "models": "gemini models --list",
                "config": "gemini config --set-key <api-key>",
                "help": "gemini --help",
            },
            "placeholder": True,
        }


# Convenience function for direct usage
async def execute_node_command(
    command: str,
    args: List[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to execute a Node.js command.

    Args:
        command: Command to execute
        args: Command arguments
        **kwargs: Additional arguments for execute_command

    Returns:
        Command execution results
    """
    bridge = NodeCLIBridge()
    return await bridge.execute_command(command, args, **kwargs)

"""
Command Line Interface for Multi-Platform AI Orchestration

Provides CLI commands for managing and interacting with the orchestration system.
"""

import asyncio
import json
import sys
from typing import Optional, Dict, Any
from datetime import datetime

import typer
import structlog
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.json import JSON

from ai_orchestration import (
    Orchestrator, 
    OrchestrationConfig, 
    OrchestrationTask,
    TaskClassification,
    PriorityLevel,
    ComplexityRating,
    SubscriptionManager,
    SubscriptionTier,
    AgentCoordinator,
    AgentType,
    AgentCapability,
    EmergentIntelligence,
    EmergentConfig
)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# CLI application
app = typer.Typer(
    name="ai-orchestrator",
    help="Multi-Platform AI Orchestration Implementation CLI",
    rich_markup_mode="rich"
)

# Global console for rich output
console = Console()

# Global system components
orchestrator: Optional[Orchestrator] = None
subscription_manager: Optional[SubscriptionManager] = None
agent_coordinator: Optional[AgentCoordinator] = None
emergent_intelligence: Optional[EmergentIntelligence] = None


@app.command()
def init(
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    subscription_mode: bool = typer.Option(True, "--subscription/--no-subscription", help="Enable subscription management"),
    emergent_mode: bool = typer.Option(True, "--emergent/--no-emergent", help="Enable emergent intelligence"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Initialize the AI Orchestration system."""
    
    if verbose:
        console.print("🚀 [bold green]Initializing Multi-Platform AI Orchestration System[/bold green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task = progress.add_task("Initializing components...", total=None)
        
        try:
            # Initialize orchestrator
            progress.update(task, description="Setting up orchestrator...")
            orchestration_config = OrchestrationConfig()
            global orchestrator
            orchestrator = Orchestrator(orchestration_config)
            asyncio.run(orchestrator.initialize())
            
            # Initialize subscription manager
            if subscription_mode:
                progress.update(task, description="Setting up subscription management...")
                global subscription_manager
                subscription_manager = SubscriptionManager()
                asyncio.run(subscription_manager.initialize())
            
            # Initialize agent coordinator
            progress.update(task, description="Setting up agent coordination...")
            global agent_coordinator
            agent_coordinator = AgentCoordinator()
            asyncio.run(agent_coordinator.initialize())
            
            # Initialize emergent intelligence
            if emergent_mode:
                progress.update(task, description="Setting up emergent intelligence...")
                emergent_config = EmergentConfig()
                global emergent_intelligence
                emergent_intelligence = EmergentIntelligence(emergent_config)
                asyncio.run(emergent_intelligence.initialize())
            
            progress.update(task, description="✅ System initialized successfully!")
            
        except Exception as e:
            console.print(f"❌ [bold red]Initialization failed:[/bold red] {str(e)}")
            raise typer.Exit(1)
    
    console.print("\n🎉 [bold green]AI Orchestration System is ready![/bold green]")
    
    if verbose:
        status_table = Table(title="System Status")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        
        status_table.add_row("Orchestrator", "✅ Active")
        status_table.add_row("Subscription Manager", "✅ Active" if subscription_mode else "❌ Disabled")
        status_table.add_row("Agent Coordinator", "✅ Active")
        status_table.add_row("Emergent Intelligence", "✅ Active" if emergent_mode else "❌ Disabled")
        
        console.print(status_table)


@app.command()
def status():
    """Show system status and metrics."""
    
    if not orchestrator:
        console.print("❌ [bold red]System not initialized. Run 'ai-orchestrator init' first.[/bold red]")
        raise typer.Exit(1)
    
    console.print("📊 [bold blue]System Status Report[/bold blue]\n")
    
    # Orchestrator metrics
    orch_metrics = orchestrator.get_metrics()
    orch_table = Table(title="🎯 Orchestrator Metrics")
    orch_table.add_column("Metric", style="cyan")
    orch_table.add_column("Value", style="yellow")
    
    for key, value in orch_metrics.items():
        orch_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(orch_table)
    
    # Agent coordinator metrics
    if agent_coordinator:
        agent_metrics = agent_coordinator.get_coordination_metrics()
        agent_table = Table(title="🤖 Agent Coordination Metrics")
        agent_table.add_column("Metric", style="cyan")
        agent_table.add_column("Value", style="yellow")
        
        for key, value in agent_metrics.items():
            if key != "agent_type_distribution" and key != "pool_sizes":
                agent_table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(agent_table)
    
    # Subscription metrics
    if subscription_manager:
        sub_metrics = subscription_manager.get_subscription_metrics()
        sub_table = Table(title="💎 Subscription Metrics")
        sub_table.add_column("Metric", style="cyan")
        sub_table.add_column("Value", style="yellow")
        
        for key, value in sub_metrics.items():
            if key != "tier_distribution":
                sub_table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(sub_table)
    
    # Emergent intelligence metrics
    if emergent_intelligence:
        intel_metrics = emergent_intelligence.get_intelligence_metrics()
        intel_table = Table(title="🧠 Intelligence Metrics")
        intel_table.add_column("Metric", style="cyan")
        intel_table.add_column("Value", style="yellow")
        
        intel_table.add_row("Patterns Discovered", str(intel_metrics.patterns_discovered))
        intel_table.add_row("Adaptations Made", str(intel_metrics.adaptations_made))
        intel_table.add_row("Prediction Accuracy", f"{intel_metrics.prediction_accuracy:.2%}")
        intel_table.add_row("Emergent Behaviors", str(intel_metrics.emergent_behaviors))
        
        console.print(intel_table)


@app.command()
def submit_task(
    task_id: str = typer.Argument(..., help="Unique task identifier"),
    classification: str = typer.Option("operational_execution", "--type", "-t", help="Task classification"),
    priority: str = typer.Option("standard_operation", "--priority", "-p", help="Task priority level"),
    complexity: str = typer.Option("component_implementation", "--complexity", "-c", help="Task complexity"),
    payload: str = typer.Option("{}", "--payload", help="Task payload as JSON string"),
    user_id: Optional[str] = typer.Option(None, "--user", "-u", help="User ID for subscription check")
):
    """Submit a task for orchestration."""
    
    if not orchestrator:
        console.print("❌ [bold red]System not initialized. Run 'ai-orchestrator init' first.[/bold red]")
        raise typer.Exit(1)
    
    try:
        # Parse task parameters
        task_classification = TaskClassification(classification)
        priority_level = PriorityLevel(priority)
        complexity_rating = ComplexityRating(complexity)
        task_payload = json.loads(payload)
        
        # Create orchestration task
        task = OrchestrationTask(
            id=task_id,
            classification=task_classification,
            priority=priority_level,
            complexity=complexity_rating,
            payload=task_payload
        )
        
        # Check subscription if user provided
        if user_id and subscription_manager:
            subscription = subscription_manager.get_user_subscription(user_id)
            if not subscription:
                console.print(f"❌ [bold red]User {user_id} not found. Register user first.[/bold red]")
                raise typer.Exit(1)
        
        # Submit task
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            submit_task = progress.add_task("Submitting task...", total=None)
            
            task_id_result = asyncio.run(orchestrator.submit_task(task))
            
            progress.update(submit_task, description="✅ Task submitted successfully!")
        
        console.print(f"🎯 [bold green]Task submitted with ID:[/bold green] {task_id_result}")
        
    except Exception as e:
        console.print(f"❌ [bold red]Task submission failed:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def list_tasks():
    """List all orchestration tasks."""
    
    if not orchestrator:
        console.print("❌ [bold red]System not initialized. Run 'ai-orchestrator init' first.[/bold red]")
        raise typer.Exit(1)
    
    tasks_table = Table(title="📋 Orchestration Tasks")
    tasks_table.add_column("Task ID", style="cyan")
    tasks_table.add_column("Classification", style="yellow")
    tasks_table.add_column("Priority", style="magenta")
    tasks_table.add_column("Status", style="green")
    tasks_table.add_column("Created", style="blue")
    
    for task_id, task in orchestrator.tasks.items():
        tasks_table.add_row(
            task_id,
            task.classification.value,
            task.priority.value,
            task.status,
            task.created_at.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    console.print(tasks_table)


@app.command()
def register_user(
    user_id: str = typer.Argument(..., help="User identifier"),
    tier: str = typer.Option("free", "--tier", "-t", help="Subscription tier"),
    duration: Optional[int] = typer.Option(None, "--duration", "-d", help="Subscription duration in days")
):
    """Register a new user with subscription."""
    
    if not subscription_manager:
        console.print("❌ [bold red]Subscription manager not initialized.[/bold red]")
        raise typer.Exit(1)
    
    try:
        subscription_tier = SubscriptionTier(tier)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            reg_task = progress.add_task("Registering user...", total=None)
            
            subscription = asyncio.run(
                subscription_manager.register_user(user_id, subscription_tier, duration)
            )
            
            progress.update(reg_task, description="✅ User registered successfully!")
        
        console.print(f"👤 [bold green]User registered:[/bold green] {user_id}")
        console.print(f"💎 [bold blue]Subscription tier:[/bold blue] {tier}")
        
        if duration:
            console.print(f"⏰ [bold blue]Duration:[/bold blue] {duration} days")
        
    except Exception as e:
        console.print(f"❌ [bold red]User registration failed:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def deploy_agent(
    agent_type: str = typer.Option("general_purpose", "--type", "-t", help="Agent type"),
    capabilities: Optional[str] = typer.Option(None, "--capabilities", "-c", help="Comma-separated capabilities"),
    tier: str = typer.Option("free", "--tier", help="Required subscription tier")
):
    """Deploy a new agent."""
    
    if not agent_coordinator:
        console.print("❌ [bold red]Agent coordinator not initialized.[/bold red]")
        raise typer.Exit(1)
    
    try:
        agent_type_enum = AgentType(agent_type)
        
        agent_capabilities = None
        if capabilities:
            capability_names = [cap.strip() for cap in capabilities.split(",")]
            agent_capabilities = [AgentCapability(cap) for cap in capability_names]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            deploy_task = progress.add_task("Deploying agent...", total=None)
            
            agent_id = asyncio.run(
                agent_coordinator.deploy_agent(agent_type_enum, agent_capabilities, tier)
            )
            
            progress.update(deploy_task, description="✅ Agent deployed successfully!")
        
        console.print(f"🤖 [bold green]Agent deployed with ID:[/bold green] {agent_id}")
        
    except Exception as e:
        console.print(f"❌ [bold red]Agent deployment failed:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def list_agents():
    """List all deployed agents."""
    
    if not agent_coordinator:
        console.print("❌ [bold red]Agent coordinator not initialized.[/bold red]")
        raise typer.Exit(1)
    
    agents_table = Table(title="🤖 Deployed Agents")
    agents_table.add_column("Agent ID", style="cyan")
    agents_table.add_column("Type", style="yellow")
    agents_table.add_column("Status", style="green")
    agents_table.add_column("Tasks", style="magenta")
    agents_table.add_column("Success Rate", style="blue")
    
    for agent_id, agent in agent_coordinator.agents.items():
        agents_table.add_row(
            agent_id[:8] + "...",
            agent.type.value,
            agent.status.value,
            str(len(agent.assigned_tasks)),
            f"{agent.metrics.success_rate:.2%}"
        )
    
    console.print(agents_table)


@app.command()
def patterns():
    """Show discovered patterns from emergent intelligence."""
    
    if not emergent_intelligence:
        console.print("❌ [bold red]Emergent intelligence not initialized.[/bold red]")
        raise typer.Exit(1)
    
    learned_patterns = emergent_intelligence.get_learned_patterns()
    
    if not learned_patterns:
        console.print("ℹ️  [bold yellow]No patterns discovered yet.[/bold yellow]")
        return
    
    patterns_table = Table(title="🧠 Discovered Patterns")
    patterns_table.add_column("Pattern ID", style="cyan")
    patterns_table.add_column("Type", style="yellow")
    patterns_table.add_column("Confidence", style="green")
    patterns_table.add_column("Impact", style="magenta")
    patterns_table.add_column("Description", style="blue")
    
    for pattern in learned_patterns:
        patterns_table.add_row(
            pattern.id[:12] + "...",
            pattern.pattern_type.value,
            f"{pattern.confidence:.2%}",
            f"{pattern.impact_score:.2f}",
            pattern.description[:50] + "..." if len(pattern.description) > 50 else pattern.description
        )
    
    console.print(patterns_table)


@app.command()
def feed_data(
    data_file: Optional[str] = typer.Option(None, "--file", "-f", help="JSON file with data"),
    data_json: Optional[str] = typer.Option(None, "--json", "-j", help="JSON data string")
):
    """Feed data to emergent intelligence system."""
    
    if not emergent_intelligence:
        console.print("❌ [bold red]Emergent intelligence not initialized.[/bold red]")
        raise typer.Exit(1)
    
    try:
        data = None
        
        if data_file:
            with open(data_file, 'r') as f:
                data = json.load(f)
        elif data_json:
            data = json.loads(data_json)
        else:
            console.print("❌ [bold red]Either --file or --json must be provided.[/bold red]")
            raise typer.Exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            feed_task = progress.add_task("Feeding data to intelligence system...", total=None)
            
            if isinstance(data, list):
                for item in data:
                    asyncio.run(emergent_intelligence.feed_data(item))
            else:
                asyncio.run(emergent_intelligence.feed_data(data))
            
            progress.update(feed_task, description="✅ Data fed successfully!")
        
        console.print("🧠 [bold green]Data fed to emergent intelligence system[/bold green]")
        
    except Exception as e:
        console.print(f"❌ [bold red]Data feeding failed:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def shutdown():
    """Shutdown the AI Orchestration system."""
    
    console.print("🛑 [bold yellow]Shutting down AI Orchestration System...[/bold yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task = progress.add_task("Shutting down components...", total=None)
        
        try:
            if emergent_intelligence:
                progress.update(task, description="Shutting down emergent intelligence...")
                asyncio.run(emergent_intelligence.shutdown())
            
            if agent_coordinator:
                progress.update(task, description="Shutting down agent coordinator...")
                asyncio.run(agent_coordinator.shutdown())
            
            if subscription_manager:
                progress.update(task, description="Shutting down subscription manager...")
                asyncio.run(subscription_manager.shutdown())
            
            if orchestrator:
                progress.update(task, description="Shutting down orchestrator...")
                asyncio.run(orchestrator.shutdown())
            
            progress.update(task, description="✅ System shutdown completed!")
            
        except Exception as e:
            console.print(f"❌ [bold red]Shutdown failed:[/bold red] {str(e)}")
            raise typer.Exit(1)
    
    console.print("👋 [bold green]AI Orchestration System shutdown complete[/bold green]")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
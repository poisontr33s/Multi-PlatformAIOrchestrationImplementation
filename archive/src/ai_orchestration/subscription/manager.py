"""
Subscription Management System

Manages premium subscription matrix and user access to advanced orchestration features.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class SubscriptionTier(Enum):
    """Available subscription tiers."""
    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class SubscriptionFeature(Enum):
    """Features available based on subscription."""
    BASIC_ORCHESTRATION = "basic_orchestration"
    STRATEGIC_INTELLIGENCE = "strategic_intelligence"
    EMERGENT_CAPABILITIES = "emergent_capabilities"
    DISTRIBUTED_AGENTS = "distributed_agents"
    PRIORITY_EXECUTION = "priority_execution"
    ADVANCED_ANALYTICS = "advanced_analytics"
    CUSTOM_INTEGRATIONS = "custom_integrations"
    DEDICATED_RESOURCES = "dedicated_resources"


@dataclass
class SubscriptionPlan:
    """Defines a subscription plan with features and limits."""
    tier: SubscriptionTier
    features: List[SubscriptionFeature]
    max_concurrent_tasks: int
    max_agents: int
    api_rate_limit: int  # requests per minute
    storage_limit_gb: int
    compute_credits: int
    priority_weight: float


@dataclass
class UserSubscription:
    """Represents a user's subscription."""
    user_id: str
    plan: SubscriptionPlan
    start_date: datetime
    end_date: Optional[datetime]
    usage_stats: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


class SubscriptionMatrix(BaseModel):
    """Premium subscription matrix configuration."""
    tier_plans: Dict[str, SubscriptionPlan]
    feature_weights: Dict[str, float]
    usage_tracking_enabled: bool = True
    billing_cycle_days: int = 30


class SubscriptionManager:
    """
    Manages the premium subscription matrix that enables emergent intelligence
    potential and advanced orchestration capabilities.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.subscriptions: Dict[str, UserSubscription] = {}
        self.matrix = self._initialize_subscription_matrix()
        self._running = False
        
    def _initialize_subscription_matrix(self) -> SubscriptionMatrix:
        """Initialize the subscription matrix with predefined plans."""
        
        # Define subscription plans
        plans = {
            "free": SubscriptionPlan(
                tier=SubscriptionTier.FREE,
                features=[SubscriptionFeature.BASIC_ORCHESTRATION],
                max_concurrent_tasks=2,
                max_agents=1,
                api_rate_limit=10,
                storage_limit_gb=1,
                compute_credits=100,
                priority_weight=0.1
            ),
            "standard": SubscriptionPlan(
                tier=SubscriptionTier.STANDARD,
                features=[
                    SubscriptionFeature.BASIC_ORCHESTRATION,
                    SubscriptionFeature.DISTRIBUTED_AGENTS,
                    SubscriptionFeature.ADVANCED_ANALYTICS
                ],
                max_concurrent_tasks=10,
                max_agents=5,
                api_rate_limit=100,
                storage_limit_gb=10,
                compute_credits=1000,
                priority_weight=0.5
            ),
            "premium": SubscriptionPlan(
                tier=SubscriptionTier.PREMIUM,
                features=[
                    SubscriptionFeature.BASIC_ORCHESTRATION,
                    SubscriptionFeature.STRATEGIC_INTELLIGENCE,
                    SubscriptionFeature.EMERGENT_CAPABILITIES,
                    SubscriptionFeature.DISTRIBUTED_AGENTS,
                    SubscriptionFeature.PRIORITY_EXECUTION,
                    SubscriptionFeature.ADVANCED_ANALYTICS
                ],
                max_concurrent_tasks=50,
                max_agents=20,
                api_rate_limit=1000,
                storage_limit_gb=100,
                compute_credits=10000,
                priority_weight=0.8
            ),
            "enterprise": SubscriptionPlan(
                tier=SubscriptionTier.ENTERPRISE,
                features=list(SubscriptionFeature),  # All features
                max_concurrent_tasks=200,
                max_agents=100,
                api_rate_limit=10000,
                storage_limit_gb=1000,
                compute_credits=100000,
                priority_weight=1.0
            )
        }
        
        feature_weights = {
            "basic_orchestration": 1.0,
            "strategic_intelligence": 2.0,
            "emergent_capabilities": 3.0,
            "distributed_agents": 2.5,
            "priority_execution": 2.0,
            "advanced_analytics": 1.5,
            "custom_integrations": 3.0,
            "dedicated_resources": 4.0
        }
        
        return SubscriptionMatrix(
            tier_plans=plans,
            feature_weights=feature_weights,
            usage_tracking_enabled=True,
            billing_cycle_days=30
        )
        
    async def initialize(self) -> None:
        """Initialize the subscription manager."""
        self.logger.info("Initializing SubscriptionManager")
        self._running = True
        
        # Start background tasks
        asyncio.create_task(self._usage_tracking_loop())
        asyncio.create_task(self._subscription_monitoring_loop())
        
    async def shutdown(self) -> None:
        """Shutdown the subscription manager."""
        self.logger.info("Shutting down SubscriptionManager")
        self._running = False
        
    async def register_user(
        self, 
        user_id: str, 
        tier: SubscriptionTier = SubscriptionTier.FREE,
        duration_days: Optional[int] = None
    ) -> UserSubscription:
        """
        Register a new user with a subscription plan.
        
        Args:
            user_id: Unique identifier for the user
            tier: Subscription tier to assign
            duration_days: Duration of subscription in days (None for indefinite)
            
        Returns:
            UserSubscription object
        """
        plan = self.matrix.tier_plans[tier.value]
        start_date = datetime.utcnow()
        end_date = start_date + timedelta(days=duration_days) if duration_days else None
        
        subscription = UserSubscription(
            user_id=user_id,
            plan=plan,
            start_date=start_date,
            end_date=end_date,
            usage_stats={
                "tasks_executed": 0,
                "agents_deployed": 0,
                "api_calls": 0,
                "compute_credits_used": 0,
                "storage_used_gb": 0.0
            }
        )
        
        self.subscriptions[user_id] = subscription
        self.logger.info("User registered", user_id=user_id, tier=tier.value)
        
        return subscription
        
    def get_user_subscription(self, user_id: str) -> Optional[UserSubscription]:
        """Get subscription information for a user."""
        return self.subscriptions.get(user_id)
        
    def check_feature_access(self, user_id: str, feature: SubscriptionFeature) -> bool:
        """Check if a user has access to a specific feature."""
        subscription = self.subscriptions.get(user_id)
        if not subscription or not subscription.is_active:
            return False
            
        # Check if subscription is expired
        if subscription.end_date and datetime.utcnow() > subscription.end_date:
            subscription.is_active = False
            return False
            
        return feature in subscription.plan.features
        
    def check_usage_limits(self, user_id: str, resource_type: str, amount: int = 1) -> bool:
        """Check if user is within usage limits for a resource."""
        subscription = self.subscriptions.get(user_id)
        if not subscription or not subscription.is_active:
            return False
            
        plan = subscription.plan
        usage = subscription.usage_stats
        
        if resource_type == "concurrent_tasks":
            return usage.get("active_tasks", 0) + amount <= plan.max_concurrent_tasks
        elif resource_type == "agents":
            return usage.get("active_agents", 0) + amount <= plan.max_agents
        elif resource_type == "api_calls":
            # Check rate limit (per minute)
            recent_calls = usage.get("recent_api_calls", [])
            now = datetime.utcnow()
            recent_calls = [call_time for call_time in recent_calls if (now - call_time).seconds < 60]
            return len(recent_calls) + amount <= plan.api_rate_limit
        elif resource_type == "compute_credits":
            return usage.get("compute_credits_used", 0) + amount <= plan.compute_credits
        elif resource_type == "storage":
            return usage.get("storage_used_gb", 0) + amount <= plan.storage_limit_gb
            
        return True
        
    async def record_usage(self, user_id: str, resource_type: str, amount: int = 1) -> None:
        """Record resource usage for a user."""
        subscription = self.subscriptions.get(user_id)
        if not subscription:
            return
            
        usage = subscription.usage_stats
        
        if resource_type == "task_execution":
            usage["tasks_executed"] = usage.get("tasks_executed", 0) + amount
        elif resource_type == "agent_deployment":
            usage["agents_deployed"] = usage.get("agents_deployed", 0) + amount
        elif resource_type == "api_call":
            usage["api_calls"] = usage.get("api_calls", 0) + amount
            # Track recent API calls for rate limiting
            recent_calls = usage.get("recent_api_calls", [])
            recent_calls.append(datetime.utcnow())
            # Keep only last hour of calls
            now = datetime.utcnow()
            recent_calls = [call_time for call_time in recent_calls if (now - call_time).seconds < 3600]
            usage["recent_api_calls"] = recent_calls
        elif resource_type == "compute_credits":
            usage["compute_credits_used"] = usage.get("compute_credits_used", 0) + amount
        elif resource_type == "storage":
            usage["storage_used_gb"] = usage.get("storage_used_gb", 0) + amount
            
        self.logger.debug("Usage recorded", user_id=user_id, resource_type=resource_type, amount=amount)
        
    def get_subscription_metrics(self) -> Dict[str, Any]:
        """Get metrics about subscription usage and distribution."""
        total_users = len(self.subscriptions)
        active_users = len([s for s in self.subscriptions.values() if s.is_active])
        
        tier_distribution = {}
        for tier in SubscriptionTier:
            count = len([s for s in self.subscriptions.values() if s.plan.tier == tier and s.is_active])
            tier_distribution[tier.value] = count
            
        total_revenue_potential = sum(
            self._get_tier_price(s.plan.tier) for s in self.subscriptions.values() if s.is_active
        )
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "tier_distribution": tier_distribution,
            "total_revenue_potential": total_revenue_potential,
            "premium_adoption_rate": (tier_distribution.get("premium", 0) + tier_distribution.get("enterprise", 0)) / active_users if active_users > 0 else 0
        }
        
    def _get_tier_price(self, tier: SubscriptionTier) -> float:
        """Get the price for a subscription tier (for metrics calculation)."""
        prices = {
            SubscriptionTier.FREE: 0.0,
            SubscriptionTier.STANDARD: 29.99,
            SubscriptionTier.PREMIUM: 99.99,
            SubscriptionTier.ENTERPRISE: 499.99
        }
        return prices.get(tier, 0.0)
        
    async def upgrade_subscription(self, user_id: str, new_tier: SubscriptionTier) -> bool:
        """Upgrade a user's subscription to a higher tier."""
        subscription = self.subscriptions.get(user_id)
        if not subscription:
            return False
            
        new_plan = self.matrix.tier_plans[new_tier.value]
        old_tier = subscription.plan.tier
        
        subscription.plan = new_plan
        subscription.start_date = datetime.utcnow()  # Reset start date for new tier
        
        self.logger.info("Subscription upgraded", user_id=user_id, old_tier=old_tier.value, new_tier=new_tier.value)
        return True
        
    async def _usage_tracking_loop(self) -> None:
        """Background loop for tracking usage and enforcing limits."""
        while self._running:
            try:
                # Reset daily usage counters and enforce limits
                for user_id, subscription in self.subscriptions.items():
                    if subscription.is_active:
                        # Check if subscription expired
                        if subscription.end_date and datetime.utcnow() > subscription.end_date:
                            subscription.is_active = False
                            self.logger.info("Subscription expired", user_id=user_id)
                            
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                self.logger.error("Error in usage tracking loop", error=str(e))
                await asyncio.sleep(60)
                
    async def _subscription_monitoring_loop(self) -> None:
        """Background loop for subscription monitoring and optimization."""
        while self._running:
            try:
                # Monitor subscription health and usage patterns
                metrics = self.get_subscription_metrics()
                self.logger.info("Subscription metrics", **metrics)
                
                await asyncio.sleep(900)  # Run every 15 minutes
            except Exception as e:
                self.logger.error("Error in subscription monitoring loop", error=str(e))
                await asyncio.sleep(60)
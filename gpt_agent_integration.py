#!/usr/bin/env python3
"""
GPT Agent Integration - Integration with OpenAI GPT models
Інтеграцandя with GPT моwhereлями for реалandforцandї агентної архandтектури
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio
from openai import AsyncOpenAI
import tiktoken

logger = logging.getLogger(__name__)

@dataclass
class GPTAgentConfig:
    """Конфandгурацandя GPT агенand"""
    model: str = "gpt-4-turbo"
    temperature: float = 0.1
    max_tokens: int = 4000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    response_format: Optional[Dict[str, str]] = None

@dataclass
class AgentMessage:
    """Повandдомлення for агенand"""
    role: str  # system, user, assistant, tool
    content: str
    timestamp: datetime
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class GPTAgent:
    """
    GPT-баwithований агент with пandдтримкою andнструментandв
    Реалandwithує архandтектуру Manager-Workers with GPT моwhereлями
    """
    
    def __init__(self, config: GPTAgentConfig, role: str, tools: Dict[str, Any] = None):
        self.config = config
        self.role = role
        self.tools = tools or {}
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.messages = []
        self.tokenizer = tiktoken.encoding_for_model(config.model)
        
        # Інandцandалandwithуємо системnot повandдомлення
        self._init_system_message()
    
    def _init_system_message(self):
        """Інandцandалandwithує системnot повandдомлення for ролand"""
        system_prompts = {
            'manager': """You are a Trading Manager Agent responsible for making final trading decisions.
Your role is to weigh inputs from Worker Model and Worker Critic agents.

Key responsibilities:
1. Analyze predictions from Worker Model
2. Consider risk analysis from Worker Critic  
3. Make final trading decisions (buy/sell/hold)
4. Adjust position sizes based on confidence levels
5. Exercise veto power when risks are too high

Decision framework:
- If critic rejects -> HOLD (veto power)
- If critic approves -> Follow model prediction
- If critic cautious -> Reduce position size by 3x
- Always consider confidence scores below 0.7 as high risk

Response format: JSON with action, confidence, position_size, reasoning""",
            
            'worker_model': """You are a Worker Model Agent responsible for market analysis and predictions.
Your role is to analyze market data and generate trading predictions.

Key responsibilities:
1. Analyze technical indicators (RSI, MACD, volume)
2. Consider macroeconomic context
3. Generate buy/sell/hold predictions
4. Provide confidence scores
5. Calculate statistical indicators

Tools available:
- predict_light: Quick analysis with basic indicators
- predict_heavy: Deep analysis with multiple models
- get_macro_context: Get macroeconomic data
- calc_stats: Calculate statistical measures

Response format: JSON with prediction, confidence, indicators, reasoning""",
            
            'worker_critic': """You are a Worker Critic Agent responsible for quality analysis and risk assessment.
Your role is to critically evaluate predictions and identify potential issues.

Key responsibilities:
1. Validate prediction stability using bootstrap
2. Analyze risk factors and volatility
3. Detect anomalies in market data
4. Review historical performance
5. Exercise quality control

Tools available:
- bootstrap_validation: Validate prediction stability
- analyze_risk: Assess risk factors
- detect_anomaly: Find market anomalies
- historical_performance: Review past performance

Quality standards:
- Validation score < 0.6 -> REJECT
- Risk score > 0.8 -> REJECT  
- Anomaly detected -> REJECT
- Otherwise -> APPROVE or CAUTION

Response format: JSON with critic_decision, confidence, reasoning""",
            
            'memory': """You are a Memory Agent responsible for learning from experience.
Your role is to store, retrieve, and analyze trading experiences.

Key responsibilities:
1. Store successful and failed trades
2. Recall similar historical situations
3. Identify patterns in mistakes
4. Suggest improvements to strategies
5. Maintain institutional memory

Tools available:
- store_experience: Save trading outcomes
- recall_similar: Find similar past situations
- analyze_patterns: Identify error patterns

Learning focus:
- What combinations lead to losses?
- What indicators are most reliable?
- When does the model fail?
- How can we improve?

Response format: JSON with insights, recommendations, patterns_found"""
        }
        
        system_prompt = system_prompts.get(self.role, "You are a helpful trading assistant.")
        
        self.messages.append(AgentMessage(
            role="system",
            content=system_prompt,
            timestamp=datetime.now()
        ))
    
    async def add_message(self, role: str, content: str, tool_calls: List[Dict] = None):
        """Додати повandдомлення до andсторandї"""
        message = AgentMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            tool_calls=tool_calls
        )
        self.messages.append(message)
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Викликати andнструмент"""
        if tool_name not in self.tools:
            return {
                "error": f"Tool {tool_name} not available",
                "success": False
            }
        
        try:
            # Імпортуємо andнтеграцandйнand andнструменти
            from integration_tools import AgentToolsIntegration
            integration = AgentToolsIntegration()
            
            # Викликаємо вandдповandдний метод
            if hasattr(integration, tool_name):
                method = getattr(integration, tool_name)
                result = method(parameters)
                return {
                    "result": result,
                    "success": True
                }
            else:
                return {
                    "error": f"Tool method {tool_name} not found",
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    async def process_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Обробити виклики andнструментandв"""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            try:
                parameters = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
            except json.JSONDecodeError:
                parameters = {}
            
            result = await self.call_tool(tool_name, parameters)
            
            tool_result = {
                "tool_call_id": tool_call.get("id"),
                "role": "tool",
                "content": json.dumps(result)
            }
            results.append(tool_result)
        
        return results
    
    async def think(self, user_input: str) -> Dict[str, Any]:
        """Основний метод мandркування агенand"""
        # Додаємо повandдомлення користувача
        await self.add_message("user", user_input)
        
        try:
            # Формуємо повandдомлення for OpenAI
            openai_messages = []
            for msg in self.messages[-10:]:  # Обмежуємо andсторandю
                openai_msg = {
                    "role": msg.role,
                    "content": msg.content
                }
                if msg.tool_calls:
                    openai_msg["tool_calls"] = msg.tool_calls
                openai_messages.append(openai_msg)
            
            # Виwithначаємо доступнand andнструменти
            tools = self._get_openai_tools()
            
            # Викликаємо GPT
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=openai_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None
            )
            
            assistant_message = response.choices[0].message
            
            # Додаємо вandдповandдь асистенand
            await self.add_message(
                "assistant", 
                assistant_message.content or "",
                assistant_message.tool_calls
            )
            
            # Обробляємо виклики andнструментandв
            if assistant_message.tool_calls:
                tool_results = await self.process_tool_calls(assistant_message.tool_calls)
                
                # Додаємо реwithульandти andнструментandв
                for tool_result in tool_results:
                    await self.add_message(
                        tool_result["role"],
                        tool_result["content"],
                        tool_call_id=tool_result.get("tool_call_id")
                    )
                
                # Другий виклик GPT with реwithульandandми andнструментandв
                second_response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=openai_messages + [
                        {"role": msg["role"], "content": msg["content"]} for msg in self.messages[-len(tool_results)-1:]
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                final_message = second_response.choices[0].message
                await self.add_message("assistant", final_message.content or "")
                
                return {
                    "content": final_message.content,
                    "tool_calls": assistant_message.tool_calls,
                    "tool_results": tool_results,
                    "role": self.role,
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "content": assistant_message.content,
                "tool_calls": None,
                "tool_results": None,
                "role": self.role,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in agent thinking: {e}")
            return {
                "content": f"Error: {str(e)}",
                "tool_calls": None,
                "tool_results": None,
                "role": self.role,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _get_openai_tools(self) -> List[Dict[str, Any]]:
        """Отримати виvalues andнструментandв for OpenAI"""
        if not self.tools:
            return []
        
        tools = []
        
        for tool_name, tool_config in self.tools.items():
            tool_definition = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_config.get("description", f"Tool: {tool_name}"),
                    "parameters": {
                        "type": "object",
                        "properties": tool_config.get("parameters", {}),
                        "required": tool_config.get("required", [])
                    }
                }
            }
            tools.append(tool_definition)
        
        return tools
    
    def count_tokens(self, text: str) -> int:
        """Пandдрахувати токени"""
        try:
            return len(self.tokenizer.encode(text))
        except:
            return len(text.split()) * 2  # Приблиwithний пandдрахунок
    
    def get_conversation_cost(self) -> Dict[str, Any]:
        """Отримати вартandсть роwithмови"""
        total_tokens = 0
        
        for msg in self.messages:
            total_tokens += self.count_tokens(msg.content)
        
        # Приблиwithнand цandни (сandном на 2024)
        pricing = {
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},  # for 1K токенandв
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }
        
        model_pricing = pricing.get(self.config.model, pricing["gpt-4-turbo"])
        
        estimated_cost = (total_tokens / 1000) * model_pricing["input"]
        
        return {
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
            "model": self.config.model,
            "message_count": len(self.messages)
        }

class DeanAgentSystem:
    """
    Повна система агентandв for архandтектурою Деана
    Manager-Workers with GPT моwhereлями
    """
    
    def __init__(self):
        self.agents = {}
        self._init_agents()
    
    def _init_agents(self):
        """Інandцandалandwithувати allх агентandв"""
        # Виvalues andнструментandв for кожного агенand
        worker_model_tools = {
            "predict_light": {
                "description": "Quick prediction using basic indicators",
                "parameters": {
                    "ticker": {"type": "string"},
                    "features": {"type": "object"},
                    "timeframe": {"type": "string"}
                },
                "required": ["ticker", "features"]
            },
            "predict_heavy": {
                "description": "Deep prediction using multiple models",
                "parameters": {
                    "ticker": {"type": "string"},
                    "features": {"type": "object"},
                    "timeframe": {"type": "string"}
                },
                "required": ["ticker", "features"]
            },
            "get_macro_context": {
                "description": "Get macroeconomic context",
                "parameters": {
                    "date": {"type": "string"},
                    "indicators": {"type": "array"}
                },
                "required": ["date"]
            },
            "calc_stats": {
                "description": "Calculate statistical measures",
                "parameters": {
                    "data": {"type": "object"},
                    "window": {"type": "integer"}
                },
                "required": ["data"]
            }
        }
        
        worker_critic_tools = {
            "bootstrap_validation": {
                "description": "Validate prediction stability using bootstrap",
                "parameters": {
                    "prediction": {"type": "string"},
                    "historical_data": {"type": "object"},
                    "iterations": {"type": "integer"}
                },
                "required": ["prediction"]
            },
            "analyze_risk": {
                "description": "Analyze risk factors",
                "parameters": {
                    "position": {"type": "number"},
                    "market_data": {"type": "object"},
                    "indicators": {"type": "object"}
                },
                "required": ["position"]
            },
            "detect_anomaly": {
                "description": "Detect anomalies in market data",
                "parameters": {
                    "current_data": {"type": "object"},
                    "historical_patterns": {"type": "object"}
                },
                "required": ["current_data"]
            },
            "historical_performance": {
                "description": "Review historical performance",
                "parameters": {
                    "model_name": {"type": "string"},
                    "period": {"type": "string"},
                    "market_conditions": {"type": "string"}
                },
                "required": ["model_name"]
            }
        }
        
        memory_tools = {
            "store_experience": {
                "description": "Store trading experience",
                "parameters": {
                    "experience_type": {"type": "string"},
                    "data": {"type": "object"},
                    "outcome": {"type": "string"}
                },
                "required": ["experience_type", "data", "outcome"]
            },
            "recall_similar": {
                "description": "Recall similar past experiences",
                "parameters": {
                    "current_context": {"type": "object"},
                    "similarity_threshold": {"type": "number"}
                },
                "required": ["current_context"]
            },
            "analyze_patterns": {
                "description": "Analyze patterns in errors",
                "parameters": {
                    "error_history": {"type": "array"},
                    "timeframe": {"type": "string"}
                },
                "required": ["error_history"]
            }
        }
        
        # Створюємо агентandв
        self.agents['worker_model'] = GPTAgent(
            GPTAgentConfig(
                model="gpt-4-turbo",
                temperature=0.1,  # Ниwithька for точностand
                max_tokens=3000
            ),
            "worker_model",
            worker_model_tools
        )
        
        self.agents['worker_critic'] = GPTAgent(
            GPTAgentConfig(
                model="gpt-4-turbo",
                temperature=0.3,  # Середня for аналandwithу
                max_tokens=2500
            ),
            "worker_critic",
            worker_critic_tools
        )
        
        self.agents['manager'] = GPTAgent(
            GPTAgentConfig(
                model="gpt-4-turbo",
                temperature=0.2,  # Ниwithька for сandбandльностand
                max_tokens=2000
            ),
            "manager"
        )
        
        self.agents['memory'] = GPTAgent(
            GPTAgentConfig(
                model="gpt-4-turbo",
                temperature=0.4,  # Середня for креативностand
                max_tokens=2000
            ),
            "memory",
            memory_tools
        )
    
    async def analyze_market(self, ticker: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Основний метод аналandwithу ринку"""
        logger.info(f"[START] Starting Dean Agent analysis for {ticker}")
        
        # Крок 1: Worker Model аналandwithує данand
        logger.info("[DATA] Step 1: Worker Model analysis")
        model_input = f"""
        Analyze {ticker} with the following data:
        {json.dumps(market_data, indent=2)}
        
        Use available tools to:
        1. Get macro context
        2. Calculate statistics
        3. Generate predictions (both light and heavy)
        
        Provide detailed analysis with confidence scores.
        """
        
        model_result = await self.agents['worker_model'].think(model_input)
        
        # Крок 2: Worker Critic аналandwithує якandсть
        logger.info("[SEARCH] Step 2: Worker Critic analysis")
        critic_input = f"""
        Critically evaluate this analysis for {ticker}:
        {json.dumps(model_result, indent=2)}
        
        Use available tools to:
        1. Validate prediction stability
        2. Analyze risk factors
        3. Check for anomalies
        4. Review historical performance
        
        Provide critic decision (approve/reject/caution) with reasoning.
        """
        
        critic_result = await self.agents['worker_critic'].think(critic_input)
        
        # Крок 3: Manager приймає рandшення
        logger.info("[GAME] Step 3: Manager decision")
        manager_input = f"""
        Make final trading decision for {ticker} based on:
        
        Worker Model Analysis:
        {json.dumps(model_result, indent=2)}
        
        Worker Critic Evaluation:
        {json.dumps(critic_result, indent=2)}
        
        Consider:
- If critic rejects -> HOLD (veto power)
- If critic approves -> Follow model prediction
- If critic cautious -> Reduce position size by 3x
- Confidence below 0.7 = high risk

Make final decision with action, confidence, position_size, and reasoning.
        """
        
        manager_result = await self.agents['manager'].think(manager_input)
        
        # Крок 4: Memory withберandгає досвandд
        logger.info("[BRAIN] Step 4: Memory storage")
        memory_input = f"""
        Store this trading analysis experience:
        
        Ticker: {ticker}
        Model Analysis: {json.dumps(model_result, indent=2)}
        Critic Evaluation: {json.dumps(critic_result, indent=2)}
        Manager Decision: {json.dumps(manager_result, indent=2)}
        
        Store as experience and identify any learning patterns.
        """
        
        memory_result = await self.agents['memory'].think(memory_input)
        
        # Формуємо фandнальний реwithульandт
        final_result = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "model_analysis": model_result,
            "critic_evaluation": critic_result,
            "manager_decision": manager_result,
            "memory_insights": memory_result,
            "agent_system": "Dean Architecture",
            "total_cost": self._calculate_total_cost()
        }
        
        logger.info(f"[OK] Dean Agent analysis completed for {ticker}")
        return final_result
    
    def _calculate_total_cost(self) -> Dict[str, Any]:
        """Роwithрахувати forгальну вартandсть"""
        total_tokens = 0
        total_cost = 0.0
        
        for agent_name, agent in self.agents.items():
            cost_info = agent.get_conversation_cost()
            total_tokens += cost_info["total_tokens"]
            total_cost += cost_info["estimated_cost_usd"]
        
        return {
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "cost_per_agent": {
                name: agent.get_conversation_cost()["estimated_cost_usd"]
                for name, agent in self.agents.items()
            }
        }

async def main():
    """Основна функцandя for тестування"""
    print(" GPT DEAN AGENT SYSTEM")
    print("=" * 50)
    
    # Перевandряємо наявнandсть API ключа
    if not os.getenv('OPENAI_API_KEY'):
        print("[ERROR] OPENAI_API_KEY not found in environment variables")
        print("[IDEA] Set it with: export OPENAI_API_KEY='your_key_here'")
        return
    
    # Інandцandалandwithуємо систему
    system = DeanAgentSystem()
    
    print(f"[OK] Agents initialized: {list(system.agents.keys())}")
    
    # Тестуємо аналandwith
    print(f"\n TESTING MARKET ANALYSIS")
    print("-" * 30)
    
    market_data = {
        "features": {
            "rsi": 65.5,
            "macd": 0.3,
            "volume_ratio": 1.2,
            "price": 150.0,
            "volatility": 0.025
        },
        "history": {},
        "patterns": {},
        "conditions": "normal"
    }
    
    # Запускаємо аналandwith
    result = await system.analyze_market('AAPL', market_data)
    
    print(f"[DATA] ANALYSIS RESULT:")
    print(f"   Ticker: {result['ticker']}")
    print(f"   Timestamp: {result['timestamp']}")
    print(f"   Agent System: {result['agent_system']}")
    
    # Вартandсть
    cost = result['total_cost']
    print(f"[MONEY] Cost Analysis:")
    print(f"   Total tokens: {cost['total_tokens']}")
    print(f"   Total cost: ${cost['total_cost_usd']:.4f}")
    
    # Рandшення меnotджера
    if 'manager_decision' in result and 'content' in result['manager_decision']:
        print(f"[GAME] Manager Decision: {result['manager_decision']['content'][:200]}...")
    
    print(f"\n[START] GPT DEAN AGENT SYSTEM READY!")
    print(f"[BRAIN] Manager-Workers architecture with GPT models")
    print(f"[MONEY] Cost tracking enabled")
    print(f"[TOOL] Tool integration complete")

if __name__ == "__main__":
    asyncio.run(main())

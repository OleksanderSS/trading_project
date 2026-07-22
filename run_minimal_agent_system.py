import asyncio
from datetime import datetime, timezone
from argparse import Namespace
from run_agent_orchestrator import _run

async def main():
    print("=" * 60)
    print("🚀 Запуск Мінімально Готової Агентної Системи (DEAN-OS) 🚀")
    print("=" * 60)
    print("Ініціалізація усіх гілок (Orchestrator, Pipeline Branch, Analytical Branch)...")
    print("Завантаження останніх даних (Ціни, Макро, Новини, Фундаментал)...\n")
    
    # Створюємо параметри для запуску "як є" (out-of-the-box)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    args = Namespace(
        as_of=now_iso,
        ticker=["NVDA"],
        timeframe="15m",
        soft_mode=True,          # Дозволяємо працювати без жорстких блокувань для демо
        enable_logging=False,
        pipeline_mode=None,      # Тільки review, без важкого конвеєра
        batch_name="main_database",
        preload_prices="latest",
        preload_tickers=["NVDA"],
        preload_macro="latest",
        preload_news="latest",
        preload_prediction="latest",
        preload_prediction_timeframe=None,
        preload_fundamentals="latest",
        preload_regime=True,
        preload_risk=True,
        preload_tabular_dir=None,
        preload_duckdb=None
    )
    
    # Викликаємо ядро системи
    decision, orchestrator = await _run(args)
    
    print("=" * 60)
    print("🏁 Рішення Оркестратора DEAN-OS 🏁")
    print("=" * 60)
    print(f"Рішення:         {decision.decision}")
    print(f"Впевненість:     {decision.confidence}")
    print(f"Фінальний бал:   {decision.final_score}")
    print(f"Потребує людини: {decision.requires_human_approval}")
    print()
    
    if decision.blocking_agents:
        print(f"Блокуючі агенти: {', '.join(decision.blocking_agents)}")
    if decision.supporting_agents:
        print(f"Підтримуючі агенти: {', '.join(decision.supporting_agents)}")
    if decision.opposing_agents:
        print(f"Протидіючі агенти: {', '.join(decision.opposing_agents)}")
    print()
    
    print("Причини:")
    for reason in decision.reasons:
        print(f"  - {reason}")
    print("\nІнтеграція та побудова пройшли успішно!")

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Main Entry Point з паузою після етапу 3 для перенесення в Colab
"""

# [TARGET] ПРИКЛАД ВИКОРИСТАННЯ З ПАУЗОЮ
if __name__ == "__main__":
    from pipeline.main_entry_point import run_trading_pipeline
    
    print("[START] Running pipeline with Stage 3 pause for Colab transfer...")
    
    # [TARGET] Запуск з паузою після етапу 3
    result = run_trading_pipeline(
        tickers='balanced_growth',
        timeframes='default',
        config_overrides={
            'interactive_pauses': {
                'stage_3_pause': True  # Вмикаємо паузу після етапу 3
            }
        },
        progress_callback=lambda msg, prog: print(f"[DATA] {prog:.1f}% - {msg}")
    )
    
    print("[SUCCESS] Pipeline completed!")

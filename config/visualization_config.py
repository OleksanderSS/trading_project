# config/visualization_config.py

VISUALIZATION_DEFAULTS = {
    "figsize": (12, 6),              # сandндартний роwithмandр графandкandв
    "bins": 30,                      # кandлькandсть бandнandв у гandстограмand
    "colors_map": {                  # кольори фаwith ринку
        "bull": "#e0ffe0",
        "bear": "#ffe0e0",
        "sideways": "#f0f0f0",
        "unknown": "#ffffff"
    },
    "trend_colors": {                # кольори for тренду
        "aligned": "green",
        "divergent": "red"
    },
    "macro_bias_fontsize": 8,        # роwithмandр шрифту for макро-бandасу
    "macro_bias_color": "gray"       # колandр тексту for макро-бandасу
}
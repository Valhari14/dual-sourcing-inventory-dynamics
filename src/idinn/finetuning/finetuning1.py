"""
finetuning1.py -- Fine-tuning entry point (legacy alias).

Directs calls to the unified hp_grid_search runner for CyclicDualNeuralController.
"""

from src.idinn.finetuning.hp_grid_search import main

if __name__ == "__main__":
    main()

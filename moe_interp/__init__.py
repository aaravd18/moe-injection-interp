"""moe_interp: routing and activation tracing for MoE language models (OLMoE) under prompt injection.

Main entry points:
    moe_interp.dataset.load_dataset         JSONL -> list[Sample]
    moe_interp.model.load_model             config -> LoadedModel (model, tokenizer, MoE layout)
    moe_interp.tracing.run_traced_forward   one instrumented prefill pass -> routing + activations
    moe_interp.runner.run_experiment        dataset -> results directory
    moe_interp.analysis.run_analysis        results directory -> metrics/*.csv (no model)
    moe_interp.plotting.make_all_plots      metrics -> plots/*.png
"""

__version__ = "0.1.0"

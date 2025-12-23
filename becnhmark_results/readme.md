# DSBench Experiment Results

This folder contains the benchmark results for various agent configurations, discussed in Section **V.B.1** in the EpidemIQs' paper. To view the results, you need to configure and run the provided Python script.

## 🚀 How to View Results

1.  Open the file `show_result.py`.
2.  Uncomment the line corresponding to the model configuration you wish to analyze.
3.  Run the script.

### Configuration Options
The following configurations are available in `show_result.py`:

```python
# --- Uncomment one model below to run ---

model = "full_scientist"       # Scientist with 5-step reflection and plan step
# model = "react"                # Scientist with ReAct framework (no reflection or plan)
# model = "plan"                 # Scientist with plan step (no reflection)
# model = "reflect1"             # Scientist with 1-step reflection (no plan)
# model = "reflect"              # Scientist with 5-step reflection (no plan)
#model = "full_scientist1"        # Scientist with 1-step reflection and plan step
# model = "gpt-4.1-mini-2025-04-14" # LLM-only baseline

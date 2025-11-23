# PDMBench Agentic Implementation

Multi-agent framework for Predictive Maintenance Benchmark (PDMBench) tasks.

## Structure

### Core Files
- **pdmbench_agent.py**: Main agent implementation with pre-defined sub-agents
- **pdmbench_benchmark.py**: Benchmarking system
- **multi_llm_benchmark.py**: Multi-LLM (WatsonX/OpenAI) benchmarking
- **datasets_scenarios.py**: Dataset management and scenario generation
- **run_benchmark_final.py**: Main execution script

### Sub-Agents
- **Fault Classification Agent**: Identifies and classifies equipment faults
- **RUL Prediction Agent**: Predicts remaining useful life with ground truth verification
- **Cost-Benefit Analysis Agent**: Analyzes maintenance costs vs benefits
- **Safety/Policies Agent**: Evaluates safety risks and policy compliance

### Tools
- **tools/**: Code execution and dynamic tool creation
- **dataset_tools.py**: Dataset loading and management
- **ground_truth_verification.py**: RUL prediction verification
- **table_formatter.py**: Result formatting

### Supporting Systems
- **memory_system.py**: Learning from mistakes
- **reflector_agent.py**: Reasoning chain auditing
- **learning_agent.py**: Mistake analysis and heuristics

## Running

```bash
python -m multi_agent_implementation_demo.run_benchmark_final
```

Output will be saved to `final_agent_output.txt`.

## Research Paper

LaTeX paper available in `research_paper.tex` (Overleaf-ready).


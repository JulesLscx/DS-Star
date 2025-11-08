import os
import json
import subprocess
import re
import uuid
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import sys
import atexit
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime
import google.generativeai as genai

# =============================================================================
# CONFIGURATION & PROMPT TEMPLATES
# =============================================================================

with open("prompt.yaml", "r") as f:
    PROMPT_TEMPLATES = yaml.safe_load(f)

@dataclass
class DSConfig:
    """Centralized configuration for the entire pipeline."""
    run_id: str = None
    max_refinement_rounds: int = 5
    api_key: Optional[str] = None
    model_name: str = 'gemini-2.5-flash'
    interactive: bool = False
    auto_debug: bool = True
    execution_timeout: int = 60
    preserve_artifacts: bool = True
    runs_dir: str = "runs"
    data_dir: str = "data"
    code_library_dir: str = "code_library"
    
    def __post_init__(self):
        if self.run_id is None:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
        if self.api_key is None:
            self.api_key = os.environ.get("GEMINI_API_KEY")




# =============================================================================
# ARTIFACT STORAGE SYSTEM
# =============================================================================

class ArtifactStorage:
    """Persistently stores every step of the pipeline for reproducibility."""
    
    def __init__(self, config: DSConfig):
        self.config = config
        self.run_dir = Path(config.runs_dir) / config.run_id
        self._setup_directories()
        
    def _setup_directories(self):
        """Create directory structure for this run."""
        dirs = [
            self.run_dir,
            self.run_dir / "steps",
            self.run_dir / "data_cache",
            self.run_dir / "logs",
            self.run_dir / "final_output"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            
    def save_step(self, step_id: str, step_type: str, prompt: str, 
                  code: Optional[str], result: str, metadata: Dict[str, Any]):
        """Save all artifacts for a single pipeline step."""
        step_dir = self.run_dir / "steps" / f"{step_id}_{step_type}"
        step_dir.mkdir(exist_ok=True)
        
        # Save prompt
        (step_dir / "prompt.md").write_text(prompt, encoding='utf-8')
        
        # Save generated code
        if code:
            (step_dir / "code.py").write_text(code, encoding='utf-8')
            
        # Save execution result
        (step_dir / "result.txt").write_text(result, encoding='utf-8')
        
        # Save metadata
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "step_type": step_type,
            "step_id": step_id
        })
        with open(step_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
    def get_step(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a previous step's artifacts."""
        step_dirs = list(self.run_dir.glob(f"steps/{step_id}_*"))
        if not step_dirs:
            return None
        
        step_dir = step_dirs[0]
        return {
            "prompt": (step_dir / "prompt.md").read_text(encoding='utf-8'),
            "code": (step_dir / "code.py").read_text(encoding='utf-8') 
                   if (step_dir / "code.py").exists() else None,
            "result": (step_dir / "result.txt").read_text(encoding='utf-8'),
            "metadata": json.loads((step_dir / "metadata.json").read_text())
        }
    
    def list_steps(self) -> List[Dict[str, Any]]:
        """List all steps in chronological order."""
        steps = []
        for step_path in sorted(self.run_dir.glob("steps/*")):
            with open(step_path / "metadata.json") as f:
                metadata = json.load(f)
                steps.append(metadata)
        return steps
    
    def get_current_state(self) -> Dict[str, Any]:
        """Load the pipeline state."""
        state_file = self.run_dir / "pipeline_state.json"
        if state_file.exists():
            return json.loads(state_file.read_text())
        return {"current_step": 0, "completed_steps": [], "plan": [], "data_descriptions": {}}
    
    def save_state(self, state: Dict[str, Any]):
        """Save the pipeline state."""
        state_file = self.run_dir / "pipeline_state.json"
        state_file.write_text(json.dumps(state, indent=2))

# =============================================================================
# STATE MANAGEMENT & EXECUTION CONTROL
# =============================================================================

class PipelineController:
    """Manages pipeline execution with resume and editing capabilities."""
    
    def __init__(self, config: DSConfig, storage: ArtifactStorage, agent: 'DS_STAR_Agent'):
        self.config = config
        self.storage = storage
        self.agent = agent
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup structured logging."""
        import logging
        log_file = self.storage.run_dir / "logs" / "pipeline.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def should_execute_step(self, step_index: int) -> bool:
        """Check if step should be executed (for resuming)."""
        state = self.storage.get_current_state()
        return step_index >= state["current_step"]
    
    def execute_step(self, step_name: str, step_func, **kwargs) -> Any:
        """Execute a single step with full artifact preservation."""
        step_id = f"{self._get_next_step_index():03d}_{step_name}"
        
        self.logger.info(f"{'='*50}")
        self.logger.info(f"STEP {step_id}")
        self.logger.info(f"{'='*50}")
        
        # Execute the step
        result = step_func(**kwargs)
        
        # Save artifacts
        if self.config.preserve_artifacts:
            metadata = kwargs.copy()
            code = result.get("code") if isinstance(result, dict) else None
            self.storage.save_step(
                step_id=step_id,
                step_type=step_name,
                prompt=kwargs.get("prompt", ""),
                code=code,
                result=str(result.get("result") if isinstance(result, dict) else result),
                metadata=metadata
            )
        
        # Update state
        state = self.storage.get_current_state()
        state["current_step"] = self._get_next_step_index()
        state["completed_steps"].append(step_id)
        self.storage.save_state(state)
        
        # Interactive mode
        if self.config.interactive:
            input(f"Step {step_id} complete. Press Enter to continue...")
            
        return result
    
    def _get_next_step_index(self) -> int:
        """Get the next step index."""
        steps = self.storage.list_steps()
        return len(steps)
    
    def edit_last_step_code(self):
        """Allow manual editing of the last generated code."""
        steps = self.storage.list_steps()
        if not steps:
            return
        
        last_step = steps[-1]
        step_dir = self.storage.run_dir / "steps" / f"{last_step['step_id']}_{last_step['step_type']}"
        code_file = step_dir / "code.py"
        
        if code_file.exists():
            self.logger.info(f"Opening {code_file} for editing...")
            # Use system editor
            editor = os.environ.get('EDITOR', 'nano')
            subprocess.run([editor, str(code_file)])
            self.logger.info("Code updated. Re-executing step...")
            # Re-execute the modified code
            code = code_file.read_text()
            result, error = self.agent._execute_code(code)
            (step_dir / "result.txt").write_text(result)
            (step_dir / "metadata.json").write_text(json.dumps({**last_step, "edited": True}, indent=2))

# =============================================================================
# CORE AGENT (Refactored)
# =============================================================================

class DS_STAR_Agent:
    """DS-STAR agent with persistent artifact storage."""
    
    def __init__(self, config: DSConfig):
        self.config = config
        self.storage = ArtifactStorage(config)
        self.controller = PipelineController(config, self.storage, self)
        if config.api_key:
            genai.configure(api_key=config.api_key)
        else:
            raise ValueError("GEMINI_API_KEY must be set in environment or config.")
        self.model = genai.GenerativeModel(config.model_name)
        
        # Setup execution environment
        self.exec_dir = Path(config.runs_dir) / config.run_id / "exec_env"
        self.exec_dir.mkdir(exist_ok=True)
        
        self._setup_tee_logging()
        
    def _setup_tee_logging(self):
        """Tee stdout/stderr to both console and log file."""
        log_path = self.storage.run_dir / "logs" / "execution.log"
        self.log_file = open(log_path, 'a', encoding='utf-8')
        
        class _Tee:
            def __init__(self, *writers):
                self.writers = writers
            def write(self, data):
                for w in self.writers:
                    try:
                        w.write(data)
                        w.flush()
                    except Exception:
                        pass
            def flush(self):
                for w in self.writers:
                    try:
                        w.flush()
                    except Exception:
                        pass
        
        sys.stdout = _Tee(sys.stdout, self.log_file)
        sys.stderr = _Tee(sys.stderr, self.log_file)
        
        atexit.register(lambda: self.log_file.close())
    
    def _call_gemini(self, agent_name: str, prompt: str) -> str:
        """Call Gemini API with logging."""
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            self.controller.logger.info(f"[{agent_name}] Response received ({len(response_text)} chars)")
            return response_text
        except Exception as e:
            error_msg = f"Error calling Gemini API: {str(e)}"
            self.controller.logger.error(error_msg)
            raise
    
    def _extract_code_block(self, response: str) -> str:
        """Extract Python code from markdown blocks."""
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', response, re.DOTALL)
        return code_blocks[0] if code_blocks else response.strip()
    
    def _execute_code(self, code_script: str, data_files: Optional[List[str]] = None) -> Tuple[str, Optional[str]]:
        """Execute code in isolated environment."""
        self.controller.logger.info("Executing code...")
        
        # Validate data files
        if data_files:
            missing = []
            for f in data_files:
                p = Path(f)
                if p.is_absolute():
                    if not p.exists():
                        missing.append(f)
                else:
                    if not (Path(self.config.data_dir) / f).exists():
                        missing.append(f)
            if missing:
                return "", f"Missing data files: {missing}"
        
        # Write to persistent location
        exec_id = uuid.uuid4().hex[:8]
        exec_path = self.exec_dir / f"exec_{exec_id}.py"
        exec_path.write_text(code_script, encoding='utf-8')
        
        try:
            result = subprocess.run(
                [sys.executable, str(exec_path)],
                capture_output=True,
                text=True,
                timeout=self.config.execution_timeout,
                cwd=Path.cwd()
            )
            
            if result.returncode == 0:
                self.controller.logger.info("Execution successful")
                return result.stdout, None
            else:
                error_msg = result.stderr or "Unknown execution error"
                self.controller.logger.error(f"Execution failed: {error_msg}")
                return "", error_msg
                
        except subprocess.TimeoutExpired:
            return "", f"Timeout after {self.config.execution_timeout}s"
        except Exception as e:
            return "", f"Execution error: {str(e)}"


    def analyze_data(self, filename: str) -> Dict[str, str]:
        prompt = PROMPT_TEMPLATES["analyzer"].format(filename=filename)
        
        result = self.controller.execute_step(
            "analyzer",
            step_func=lambda prompt=prompt, **kwargs: self._call_gemini("ANALYZER", prompt),  # FIXED
            prompt=prompt,
            filename=filename
        )
        
        code = self._extract_code_block(result)
        exec_result, error = self._execute_code(code, [filename])
        
        if error:
            self.controller.logger.warning(f"Analysis failed: {error}")
            self._create_fallback_file(filename)
            exec_result = f"Created fallback for {filename}"
        
        return {"code": code, "result": exec_result, "filename": filename}

    def plan_next_step(self, query: str, data_desc: str, current_plan: List[str], last_result: Optional[str]) -> str:
        if not current_plan:
            prompt = PROMPT_TEMPLATES["planner_init"].format(question=query, summaries=data_desc)
            step_type = "planner_init"
        else:
            plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(current_plan))
            prompt = PROMPT_TEMPLATES["planner_next"].format(
                question=query, summaries=data_desc,
                plan=plan_str, result=last_result, current_step=current_plan[-1]
            )
            step_type = "planner_next"
        
        return self.controller.execute_step(
            step_type,
            step_func=lambda prompt=prompt, **kwargs: self._call_gemini("PLANNER", prompt),  # FIXED
            prompt=prompt,
            query=query,
            plan_length=len(current_plan)
        )

    def generate_code(self, plan: List[str], data_desc: str, base_code: Optional[str] = None) -> str:
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        
        if not base_code:
            prompt = PROMPT_TEMPLATES["coder_init"].format(
                summaries=data_desc, plan=plan_str
            )
        else:
            prompt = PROMPT_TEMPLATES["coder_next"].format(
                summaries=data_desc, base_code=base_code,
                plan=plan_str, current_plan=plan[-1]
            )
        
        result = self.controller.execute_step(
            "coder",
            step_func=lambda prompt=prompt, **kwargs: self._call_gemini("CODER", prompt),  # FIXED
            prompt=prompt,
            plan_length=len(plan),
            has_base_code=base_code is not None
        )
        
        return self._extract_code_block(result)

    def verify_plan(self, plan: List[str], code: str, result: str, query: str, data_desc: str) -> str:
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        prompt = PROMPT_TEMPLATES["verifier"].format(
            plan=plan_str, code=code, result=result, question=query, summaries=data_desc, current_step=plan[-1]
        )
        
        return self.controller.execute_step(
            "verifier",
            step_func=lambda prompt=prompt, **kwargs: self._call_gemini("VERIFIER", prompt),  # FIXED
            prompt=prompt,
            plan_length=len(plan)
        ).strip()

    def route_plan(self, plan: List[str], query: str, result: str, data_desc: str) -> str:
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        prompt = PROMPT_TEMPLATES["router"].format(
            question=query, summaries=data_desc,
            plan=plan_str, result=result, current_step=plan[-1]
        )
        
        return self.controller.execute_step(
            "router",
            step_func=lambda prompt=prompt, **kwargs: self._call_gemini("ROUTER", prompt),  # FIXED
            prompt=prompt,
            plan_length=len(plan)
        ).strip()

    def debug_code(self, code: str, error: str, data_desc: str, filenames: List[str]) -> str:
        prompt = PROMPT_TEMPLATES["debugger"].format(
            summaries=data_desc, code=code,
            bug=error, filenames=", ".join(filenames)
        )
        
        result = self.controller.execute_step(
            "debugger",
            step_func=lambda prompt=prompt, **kwargs: self._call_gemini("DEBUGGER", prompt),  # FIXED
            prompt=prompt,
            error_type=error.split(":")[0]
        )
        
        return self._extract_code_block(result)

    def finalize_solution(self, code: str, result: str, query: str, 
                        guidelines: str, data_desc: str) -> str:
        prompt = PROMPT_TEMPLATES["finalyzer"].format(
            summaries=data_desc, code=code,
            result=result, question=query, guidelines=guidelines
        )
        
        result = self.controller.execute_step(
            "finalyzer",
            step_func=lambda prompt=prompt, **kwargs: self._call_gemini("FINALYZER", prompt),  # FIXED
            prompt=prompt
        )
        
        return self._extract_code_block(result)
    def run_pipeline(self, query: str, data_files: List[str]) -> Dict[str, Any]:
        """Main pipeline with full persistence and resume capability."""
        self.controller.logger.info(f"Starting pipeline: {self.config.run_id}")
        self.controller.logger.info(f"Query: {query}")
        self.controller.logger.info(f"Data files: {data_files}")
        
        # Check for resume state
        state = self.storage.get_current_state()
        if state["completed_steps"]:
            self.controller.logger.info(f"Resuming from step {state['current_step']}")
        
        # Ensure data directory exists
        Path(self.config.data_dir).mkdir(exist_ok=True)
        
        # Initialize variables that might be loaded from previous runs
        code = None
        exec_result = None
        
        # PHASE 1: Data Analysis
        if self.controller.should_execute_step(0):
            self.controller.logger.info("=== PHASE 1: ANALYZING DATA FILES ===")
            data_descriptions = {}
            absolute_data_files = []
            for i, f in enumerate(data_files):
                self.controller.logger.info(f"Analyzing {f}...")
                abs_path = str(Path(self.config.data_dir).joinpath(f).resolve())
                absolute_data_files.append(abs_path)
                analysis = self.analyze_data(abs_path)
                data_descriptions[abs_path] = analysis["result"]
            
            state = self.storage.get_current_state()
            state["data_descriptions"] = data_descriptions
            self.storage.save_state(state)
        else:
            # Load from previous run
            data_descriptions = state["data_descriptions"]
            absolute_data_files = list(data_descriptions.keys())
        
        data_desc_str = "\n".join([f"File: {k}\n{v}" for k, v in data_descriptions.items()])
        
        # PHASE 2: Iterative Planning & Execution
        if self.controller.should_execute_step(len(absolute_data_files)):
            self.controller.logger.info("=== PHASE 2: ITERATIVE PLANNING & VERIFICATION ===")
            plan = []
            plan.append(self.plan_next_step(query, data_desc_str, plan, ""))
            
            code = self.generate_code(plan, data_desc_str)
            exec_result, error = self._execute_code(code, absolute_data_files)
            
            # Debug loop
            while error and self.config.auto_debug:
                self.controller.logger.warning("Debugging...")
                code = self.debug_code(code, error, data_desc_str, absolute_data_files)
                exec_result, error = self._execute_code(code, absolute_data_files)
            
            # Refinement rounds
            for round_idx in range(self.config.max_refinement_rounds):
                self.controller.logger.info(f"--- Refinement Round {round_idx+1} ---")
                
                verdict = self.verify_plan(plan, code, exec_result, query, data_desc_str)
                
                if verdict == "Sufficient":
                    self.controller.logger.info("Plan verified as sufficient!")
                    break
                
                routing = self.route_plan(plan, query, exec_result, data_desc_str)
                
                if "is wrong!" in routing:
                    # Truncate plan and retry
                    try:
                        step_to_remove = int(routing.split()[1]) - 1
                        plan = plan[:step_to_remove]
                        self.controller.logger.info(f"Truncated plan to step {step_to_remove}")
                    except:
                        plan = []
                else:
                    self.controller.logger.info("Adding new step...")
                
                # Generate next step
                next_plan = self.plan_next_step(query, data_desc_str, plan, exec_result)
                plan.append(next_plan)
                
                # Generate and execute new code
                code = self.generate_code(plan, data_desc_str, base_code=code)
                exec_result, error = self._execute_code(code, absolute_data_files)
                
                while error and self.config.auto_debug:
                    code = self.debug_code(code, error, data_desc_str, absolute_data_files)
                    exec_result, error = self._execute_code(code, absolute_data_files)
            else:
                self.controller.logger.warning("Max refinement rounds reached")
        
        # Load code and exec_result from previous run if not defined (resuming case)
        if code is None or exec_result is None:
            steps = self.storage.list_steps()
            
            # Find the last step with code
            for step in reversed(steps):
                step_data = self.storage.get_step(step['step_id'])
                if step_data and step_data.get('code'):
                    code = step_data['code']
                    exec_result = step_data.get('result', '')
                    self.controller.logger.info(f"Loaded code and results from step {step['step_id']}")
                    break
            
            # If still not found, this is an error
            if code is None:
                raise ValueError("Could not load code from previous steps. Please ensure the pipeline has been run before.")
            if exec_result is None:
                exec_result = ""  # Default to empty string if not found
        
        # PHASE 3: Finalization
        self.controller.logger.info("=== PHASE 3: FINALIZING ===")
        final_code = self.finalize_solution(
            code, exec_result, query,
            "Format as JSON with key 'final_answer'",
            data_desc_str
        )
        
        final_result, _ = self._execute_code(final_code, absolute_data_files)
        
        # Save final output
        output_file = self.storage.run_dir / "final_output" / "result.json"
        output_file.write_text(final_result)
        
        self.controller.logger.info("Pipeline completed successfully!")
        
        return {
            "run_id": self.config.run_id,
            "final_result": final_result,
            "output_file": str(output_file),
            "total_steps": len(self.storage.list_steps())
        }
# =============================================================================
# CLI & USAGE
# =============================================================================

def main():
    """CLI interface with resume and edit capabilities."""
    import argparse
    
    # Load config from file to set defaults
    try:
        with open("config.yaml", 'r') as f:
            config_defaults = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config_defaults = {}

    parser = argparse.ArgumentParser(description="DS-STAR Data Science Agent")
    parser.add_argument("--resume", type=str, help="Resume from run ID")
    parser.add_argument("--interactive", action="store_true", help="Pause between steps")
    parser.add_argument("--edit-last", action="store_true", help="Edit last generated code")
    parser.add_argument("--data-files", nargs="+", help="Data files to analyze")
    parser.add_argument("--query", type=str, help="Analysis query")
    parser.add_argument("--max-rounds", type=int, help="Max refinement rounds")
    
    args = parser.parse_args()
    
    # Combine config sources (CLI args take precedence)
    config_params = {
        'run_id': args.resume or config_defaults.get('run_id'),
        'interactive': args.interactive or config_defaults.get('interactive', False),
        'max_refinement_rounds': args.max_rounds or config_defaults.get('max_refinement_rounds', 5),
        'model_name': config_defaults.get('model_name', 'gemini-1.5-flash'),
        'preserve_artifacts': config_defaults.get('preserve_artifacts', True)
    }
    
    # Filter out None values so dataclass defaults are used
    config_params = {k: v for k, v in config_params.items() if v is not None}
    
    config = DSConfig(**config_params)
    
    agent = DS_STAR_Agent(config)
    
    # Edit mode
    if args.edit_last and config.run_id:
        agent.controller.edit_last_step_code()
        return

    # Check for required arguments for a new run
    if not (args.data_files and args.query):
        parser.error("--data-files and --query are required for a new run.")

    # Run pipeline
    result = agent.run_pipeline(args.query, args.data_files)
    print(f"\n{'='*60}")
    print(f"RUN COMPLETED: {result['run_id']}")
    print(f"OUTPUT: {result['output_file']}")
    print(f"FINAL RESULT:\n{result['final_result']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
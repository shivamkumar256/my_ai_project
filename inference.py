import os
from ed_triage import EDTriageEnv, TASK_EASY, Action

# Environment variables required by evaluation system
API_BASE_URL = os.getenv("API_BASE_URL", "https://example.com")
MODEL_NAME = os.getenv("MODEL_NAME", "ed-triage-model")
HF_TOKEN = os.getenv("HF_TOKEN")


def run_inference():
    """
    Entry point for evaluation.
    Initializes environment, selects an action based on patient severity,
    and executes one step in the environment.
    """

    print("[START]")

    # Initialize triage environment
    env = EDTriageEnv(**TASK_EASY.env_kwargs)
    state = env.reset()

    # Fetch current patient from environment state
    patient = state.current_patient

    if patient is None:
        print("[STEP] No patient available")
        print("[END]")
        return

    # Rule-based triage decision logic
    severity = patient.severity_score

    if severity <= 2:
        action = Action.ASSIGN_ICU
    elif severity <= 3:
        action = Action.ASSIGN_GENERAL
    else:
        action = Action.ASSIGN_OBSERVATION

    # Execute action in environment
    next_state, reward, done, info = env.step(action)

    # Output logs for evaluation visibility
    print(f"[STEP] Patient: {patient.name}")
    print(f"[STEP] Severity: {severity}")
    print(f"[STEP] Action Taken: {action}")
    print(f"[STEP] Reward: {reward}")

    print("[END]")


if __name__ == "__main__":
    run_inference()

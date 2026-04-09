import os
from openai import OpenAI
from ed_triage import EDTriageEnv, TASK_EASY, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def format_bool(value):
    return "true" if value else "false"

def main():
    print(f"[START] task=ed-triage env=openenv model={MODEL_NAME}")

    env = None
    rewards = []
    steps = 0
    success = False

    try:
        env = EDTriageEnv(**TASK_EASY.env_kwargs)
        state = env.reset()

        patient = getattr(state, "current_patient", None)

        if patient is None:
            steps = 1
            rewards.append(0.0)
            print("[STEP] step=1 action=none reward=0.00 done=false error=null")
        else:
            severity = getattr(patient, "severity_score", None)

            if severity is None:
                steps = 1
                rewards.append(0.0)
                print("[STEP] step=1 action=none reward=0.00 done=false error=missing_severity")
            else:
                if severity <= 2:
                    action = Action.ASSIGN_ICU
                elif severity <= 3:
                    action = Action.ASSIGN_GENERAL
                else:
                    action = Action.ASSIGN_OBSERVATION

                _, reward, done, _ = env.step(action)

                steps = 1
                rewards.append(float(reward))
                success = bool(done)

                print(
                    f"[STEP] step=1 action={action} reward={float(reward):.2f} "
                    f"done={format_bool(done)} error=null"
                )

    except Exception as e:
        steps = 1
        rewards = [0.0]
        print(f"[STEP] step=1 action=none reward=0.00 done=false error={str(e)}")
        success = False

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={format_bool(success)} steps={steps} rewards={rewards_str}")

if __name__ == "__main__":
    main()

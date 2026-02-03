import os, json
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi
from graph_definition import create_options_graph

DATASET_REPO = "manikandan18ramalingam/agentic-ai-options-results"  # create this dataset
OUT = Path("latest_results.json")

def main():
    graph = create_options_graph()
    result = graph.invoke({})  # your "mega-cap default" behavior

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(
        path_or_fileobj=str(OUT),
        path_in_repo=OUT.name,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="Update latest results",
    )

if __name__ == "__main__":
    main()


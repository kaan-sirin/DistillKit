from datetime import datetime
import os
import pandas as pd
import json
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import HfApi, create_repo, login

load_dotenv()

# Load the CSV file
df = pd.read_csv("medqa-swe.csv")



# Function to build the prompt for each row
def build_prompt(sample):
    return f"""
            {sample['question']}\n\n
            {sample['options']}\n\n
            Rätt svar: {sample['answer']}
        """.strip()


def create_batch_requests(df, file_path):
    with open(file_path, "w", encoding="utf-8") as outfile:
        for idx, row in df.iterrows():
            prompt = build_prompt(row)
            request = {
                "custom_id": f"{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Du är en medicinsk expert. Din uppgift är att ge en steg-för-steg-förklaring på varför ett visst svar är korrekt för en medicinsk flervalsfråga. Svara kortfattat med fokus på resonemanget bakom varför svaret är korrekt, och börja alltid med 'Rätt svar är ' följt av din förklaring.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0, 
                    "max_tokens": 512,
                }
            }
            # Write one JSON object per line
            outfile.write(json.dumps(request) + "\n")

    print(f"Batch requests written to {file_path}")
    return file_path


def upload_batch_requests(file_path):
    client = OpenAI()
    
    batch_input_file = client.files.create(
        file=open(file_path, "rb"), purpose="batch"
    )

    print(batch_input_file)

    batch_input_file_id = batch_input_file.id

    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )


def process_batch_output(batch_output_file, csv_file, output_file):
    """
    Process batch output responses and combine with original CSV data.
    Only includes rows that have responses in the output file.
    
    Args:
        batch_output_file: Path to the batch output JSONL file
        csv_file: Path to the original CSV file
        output_file: Path to save the combined CSV output
    """
    # Read the original CSV data
    df = pd.read_csv(csv_file)
    
    # Read the batch output responses
    responses = {}
    with open(batch_output_file, 'r', encoding='utf-8') as file:
        for line in file:
            response_data = json.loads(line)
            custom_id = response_data["custom_id"]
            
            # Extract the actual response content from the nested structure
            if response_data["response"]["status_code"] == 200:
                content = response_data["response"]["body"]["choices"][0]["message"]["content"]
                responses[int(custom_id)] = content
    
    # Filter the dataframe to only include rows that have responses
    response_indices = list(responses.keys())
    filtered_df = df.iloc[response_indices].copy()
    
    # Add the responses to the filtered dataframe
    filtered_df['model_response'] = filtered_df.index.map(lambda idx: responses[idx])
    
    # Save the combined data to a new CSV file
    filtered_df.to_csv(output_file, index=False)
    
    print(f"Combined data saved to {output_file} with {len(filtered_df)} rows")


def upload_to_huggingface(csv_file, repo_id):
    # Login to Hugging Face
    token = os.getenv("HF_WRITE_TOKEN")
    if not token:
        raise ValueError("HF_WRITE_TOKEN environment variable is required")
    
    # Force the use of this specific token by setting it in the environment
    os.environ["HF_TOKEN"] = token
    login(token=token)
    
    print(f"Logged in to Hugging Face Hub")
    
    # Initialize the Hugging Face API
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repository {repo_id} is ready")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload the CSV file
    api.upload_file(
        path_or_fileobj=csv_file,
        path_in_repo=csv_file.split("/")[-1],  # Use the filename as the path in repo
        repo_id=repo_id,
        repo_type="dataset",
    )
    
    print(f"Successfully uploaded {csv_file} to {repo_id}")
    print(f"View it at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    # file_path = create_batch_requests(df, "batch_requests.jsonl")
    # upload_batch_requests(file_path)
    

    # process_batch_output("batch_output.jsonl", "medqa-swe.csv", "medqa-swe-with-responses.csv")
    # df = pd.read_csv("medqa-swe-with-responses.csv")
    # Print size
    # print(df.shape)
    # for i in range(3170,3180):
    #     df = pd.read_csv("medqa-swe-with-responses.csv")
    #     # Print "model_response" of first row
    #     print(df.iloc[i]["answer"], df.iloc[i]["model_response"])
    #     print("--------------------------------")
        
    # upload_to_huggingface(
    #     csv_file="medqa-swe-with-responses.csv",
    #     repo_id="kaans/medqa-swe-with-responses"
    # )
    
    upload_to_huggingface(
        csv_file="rule_qa_answers.tsv",
        repo_id="kaans/rule_qa"
    )
    
    
    